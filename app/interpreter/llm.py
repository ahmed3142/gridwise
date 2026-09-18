"""OpenAI (or any OpenAI-compatible endpoint) client for structured note interpretation.

* Uses Structured Outputs: `chat.completions.parse(response_format=SemanticInterpretation)`. The
  model's tokens are constrained to our JSON schema, so the directive type is always one of the six
  supported enum values - the model cannot invent a new directive type.
* Adapts per model to unsupported parameters (temperature on reasoning models, reasoning_effort on
  non-reasoning models, seed, max_completion_tokens) by dropping them once and remembering it.
* Falls back to JSON mode + Pydantic validation for OpenAI-compatible servers without json_schema.
* Maps every provider failure to `LLMCallError(kind, retryable)`; the SDK's own retries are
  disabled so the service controls the total time budget.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field

import openai
from openai import AsyncOpenAI
from pydantic import ValidationError

from ..config import Settings
from .semantic import SemanticInterpretation

log = logging.getLogger("gridwise.llm")


class LLMCallError(Exception):
    def __init__(self, kind: str, message: str, retryable: bool, retry_after: float | None = None):
        super().__init__(f"{kind}: {message}")
        self.kind = kind
        self.retryable = retryable
        self.retry_after = retry_after


@dataclass
class _ModelState:
    dropped: set[str] = field(default_factory=set)
    effort: str | None = None
    json_mode: bool = False
    use_max_tokens: bool = False


_SECRET_RE = re.compile(r"(sk-[A-Za-z0-9_\-*]{4,}|Bearer\s+[A-Za-z0-9._\-]+)")


def _short(exc: Exception) -> str:
    """One-line, secret-free error text (provider messages can echo a masked key)."""
    text = _SECRET_RE.sub("[redacted]", str(exc).replace("\n", " "))
    return text[:300]


class OpenAIInterpreterClient:
    provider = "openai"

    def __init__(self, settings: Settings):
        kwargs: dict = {"api_key": settings.openai_api_key, "max_retries": 0, "timeout": settings.llm_timeout_s}
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        self._client = AsyncOpenAI(**kwargs)
        self._settings = settings
        self._state: dict[str, _ModelState] = {}
        self._dead: set[str] = set()
        self._resolve_lock = asyncio.Lock()
        self._resolved = False
        self.primary: str | None = None
        self.fallback: str | None = None
        self.available: set[str] | None = None

    # ------------------------------------------------------------------ model selection

    async def resolve_models(self, timeout: float = 6.0) -> None:
        if self._resolved:
            return
        async with self._resolve_lock:
            if self._resolved:
                return
            s = self._settings
            preference = [m for m in s.model_preference]
            if s.model and s.model.lower() != "auto":
                preference = [s.model] + [m for m in preference if m != s.model]
            try:
                ids: set[str] = set()

                async def _list() -> None:
                    async for model in self._client.models.list():
                        ids.add(model.id)

                await asyncio.wait_for(_list(), timeout=timeout)
                self.available = ids
            except Exception as exc:  # no key / network / non-OpenAI server without /models
                log.warning("model listing unavailable (%s); using configured preference order", type(exc).__name__)
                self.available = None

            if s.model and s.model.lower() != "auto":
                self.primary = s.model
            elif self.available:
                self.primary = next((m for m in preference if m in self.available), preference[0])
            else:
                self.primary = preference[0]

            if s.fallback_model:
                self.fallback = s.fallback_model if s.fallback_model != self.primary else None
            else:
                pool = [m for m in preference if m != self.primary]
                if self.available:
                    pool = [m for m in pool if m in self.available]
                self.fallback = pool[0] if pool else None
            self._resolved = True
            log.info("LLM models resolved: primary=%s fallback=%s", self.primary, self.fallback)

    async def model_chain(self) -> list[str]:
        await self.resolve_models()
        chain = [m for m in (self.primary, self.fallback) if m and m not in self._dead]
        if not chain:  # everything marked dead: try the preference list once more
            chain = [m for m in self._settings.model_preference if m not in self._dead][:2]
        return chain

    def mark_dead(self, model: str) -> None:
        self._dead.add(model)

    # ------------------------------------------------------------------ calls

    def _model_state(self, model: str) -> _ModelState:
        state = self._state.get(model)
        if state is None:
            state = _ModelState(effort=self._settings.reasoning_effort)
            name = model.lower()
            # Known capabilities, so the first calls do not waste a round trip on a 400:
            # reasoning families reject temperature; GPT-4.x/3.5 reject reasoning_effort.
            if name.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4")):
                state.dropped.add("temperature")
            if name.startswith(("gpt-4", "gpt-3.5")):
                state.dropped.add("reasoning_effort")
            self._state[model] = state
        return state

    def _kwargs(self, model: str, messages: list[dict], timeout: float) -> dict:
        st = self._model_state(model)
        kw: dict = {"model": model, "messages": messages, "timeout": timeout}
        if st.use_max_tokens:
            kw["max_tokens"] = self._settings.max_output_tokens
        else:
            kw["max_completion_tokens"] = self._settings.max_output_tokens
        if "temperature" not in st.dropped:
            kw["temperature"] = 0
        if "seed" not in st.dropped:
            kw["seed"] = 20260918
        if st.effort and "reasoning_effort" not in st.dropped:
            kw["reasoning_effort"] = st.effort
        return kw

    def _adapt(self, model: str, exc: openai.BadRequestError, sent: dict) -> bool:
        """Adjust the rejected parameter. Returns True when a retry with the new settings makes sense.

        Concurrency-safe: several calls to a new model can be rejected at the same moment. The first
        one updates the shared state; the others see that the state no longer matches what they
        sent and simply retry with the updated settings (instead of failing).
        """
        st = self._model_state(model)
        param = (getattr(exc, "param", None) or "").lower()
        msg = str(exc).lower()

        def mentions(name: str) -> bool:
            return name in param or name in msg

        if mentions("reasoning_effort") or mentions("reasoning.effort") or mentions("reasoning effort"):
            if "reasoning_effort" not in sent:
                return False
            if st.effort == sent["reasoning_effort"] and "reasoning_effort" not in st.dropped:
                if "unsupported value" in msg and st.effort not in ("low", "medium"):
                    st.effort = "low"  # e.g. 'minimal' not offered by this model
                else:
                    st.dropped.add("reasoning_effort")
            return True
        if mentions("temperature"):
            if "temperature" not in sent:
                return False
            st.dropped.add("temperature")
            return True
        if mentions("seed"):
            if "seed" not in sent:
                return False
            st.dropped.add("seed")
            return True
        if mentions("max_completion_tokens"):
            if "max_completion_tokens" not in sent:
                return False
            st.use_max_tokens = True
            return True
        if mentions("response_format") or mentions("json_schema"):
            if sent.get("_json_mode"):
                return False
            st.json_mode = True
            return True
        return False

    async def interpret(self, model: str, messages: list[dict], timeout: float) -> SemanticInterpretation:
        started = time.perf_counter()
        for _ in range(6):
            st = self._model_state(model)
            kwargs = self._kwargs(model, messages, timeout)
            sent = dict(kwargs, _json_mode=st.json_mode)
            try:
                if st.json_mode:
                    result = await self._json_mode_call(kwargs)
                else:
                    completion = await self._client.chat.completions.parse(
                        response_format=SemanticInterpretation, **kwargs
                    )
                    result = self._extract(completion)
                log.debug("llm %s ok in %.0f ms", model, (time.perf_counter() - started) * 1000)
                return result
            except openai.BadRequestError as exc:
                if self._adapt(model, exc, sent):
                    log.info("model %s rejected a parameter; adapting (%s)", model, _short(exc)[:120])
                    continue
                raise LLMCallError("bad_request", _short(exc), retryable=False) from None
            except openai.LengthFinishReasonError:
                raise LLMCallError("length", "output truncated", retryable=True) from None
            except openai.ContentFilterFinishReasonError:
                raise LLMCallError("refusal", "content filter", retryable=False) from None
            except openai.APITimeoutError:
                raise LLMCallError("timeout", f"no answer within {timeout:.1f}s", retryable=True) from None
            except openai.RateLimitError as exc:
                retry_after = None
                try:
                    retry_after = float(exc.response.headers.get("retry-after", "")) if exc.response else None
                except (TypeError, ValueError):
                    retry_after = None
                kind = "quota" if "insufficient_quota" in str(exc) else "rate_limit"
                raise LLMCallError(kind, _short(exc), retryable=kind == "rate_limit", retry_after=retry_after) from None
            except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
                raise LLMCallError("auth", _short(exc), retryable=False) from None
            except openai.NotFoundError as exc:
                raise LLMCallError("not_found", _short(exc), retryable=False) from None
            except openai.APIConnectionError as exc:
                raise LLMCallError("connection", _short(exc), retryable=True) from None
            except openai.APIStatusError as exc:
                raise LLMCallError("server", _short(exc), retryable=exc.status_code >= 500) from None
            except (ValidationError, ValueError) as exc:
                raise LLMCallError("invalid_output", _short(exc), retryable=True) from None
        raise LLMCallError("bad_request", "parameter adaptation exhausted", retryable=False)

    @staticmethod
    def _extract(completion) -> SemanticInterpretation:
        if not completion.choices:
            raise LLMCallError("invalid_output", "empty choices", retryable=True)
        message = completion.choices[0].message
        if getattr(message, "refusal", None):
            raise LLMCallError("refusal", str(message.refusal)[:200], retryable=False)
        parsed = getattr(message, "parsed", None)
        if parsed is None:
            raise LLMCallError("invalid_output", "no parsed output", retryable=True)
        return parsed

    async def _json_mode_call(self, kwargs: dict) -> SemanticInterpretation:
        schema_hint = SemanticInterpretation.model_json_schema()
        messages = list(kwargs.pop("messages"))
        messages.append(
            {
                "role": "system",
                "content": "Respond with ONE JSON object only, matching this JSON schema exactly: "
                + str(schema_hint),
            }
        )
        completion = await self._client.chat.completions.create(
            messages=messages, response_format={"type": "json_object"}, **kwargs
        )
        if not completion.choices:
            raise LLMCallError("invalid_output", "empty choices", retryable=True)
        choice = completion.choices[0]
        if choice.finish_reason == "length":
            raise LLMCallError("length", "output truncated", retryable=True)
        content = choice.message.content or ""
        return SemanticInterpretation.model_validate_json(content)

    async def aclose(self) -> None:
        try:
            await self._client.close()
        except Exception:  # pragma: no cover
            pass
