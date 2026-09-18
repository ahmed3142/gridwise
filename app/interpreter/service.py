"""Orchestrates LLM interpretation of all notes in a request.

Per request:  one parallel LLM call per note (notes are independent, so N notes cost one round trip)
Per note:     primary model -> [one re-ask if guardrails/cross-check flag the answer]
              -> fallback model on provider failure -> controlled no_op if everything fails
Budget:       every attempt is capped by the time left before the request deadline
Cache:        LRU of the model's semantic answers keyed on (prompt version, all notes, note index);
              concurrent identical requests share one in-flight LLM call (single-flight)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from ..config import Settings
from ..directives import Directive, no_op
from ..schemas import Battery
from .crosscheck import crosscheck
from .llm import LLMCallError
from .normalize import normalize
from .prompt import PROMPT_VERSION, SYSTEM_PROMPT, feedback_message, user_message
from .semantic import SemanticInterpretation

log = logging.getLogger("gridwise.interpreter")

SOLVE_RESERVE_S = 1.5  # time kept back for optimization + validation after interpretation
MIN_ATTEMPT_S = 1.5  # never start an LLM attempt with less time than this


@dataclass
class NoteResult:
    directive: Directive
    semantic: SemanticInterpretation | None = None
    model: str | None = None
    attempts: int = 0
    cached: bool = False
    degraded: bool = False
    issues: list[str] = field(default_factory=list)
    error: str | None = None


class _LRU:
    def __init__(self, size: int):
        self.size = size
        self._data: OrderedDict[str, tuple[SemanticInterpretation, str | None]] = OrderedDict()

    def get(self, key: str):
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        return None

    def put(self, key: str, value) -> None:
        if self.size <= 0:
            return
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.size:
            self._data.popitem(last=False)


def _cache_key(notes: list[str], index: int) -> str:
    payload = json.dumps([PROMPT_VERSION, [" ".join(n.split()) for n in notes], index])
    return hashlib.sha256(payload.encode()).hexdigest()


class InterpretationService:
    def __init__(self, client, settings: Settings):
        self.client = client
        self.settings = settings
        self._cache = _LRU(settings.cache_size)
        self._inflight: dict[str, asyncio.Future] = {}
        self._semaphore: asyncio.Semaphore | None = None
        self._stats: dict = {"llm_calls_ok": 0, "llm_calls_failed": 0, "last_error": None,
                             "last_error_unix": None, "last_success_unix": None}

    @property
    def configured(self) -> bool:
        return self.client is not None

    def _record(self, ok: bool, error: str | None) -> None:
        if ok:
            self._stats["llm_calls_ok"] += 1
            self._stats["last_success_unix"] = round(time.time())
        else:
            self._stats["llm_calls_failed"] += 1
            self._stats["last_error"] = error
            self._stats["last_error_unix"] = round(time.time())

    def status(self) -> dict:
        """Operational visibility: makes a dead LLM path obvious instead of silent no_op answers."""
        return dict(self._stats, cache_entries=len(self._cache._data))

    def _sem(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.settings.llm_max_concurrency)
        return self._semaphore

    # ------------------------------------------------------------------ public API

    async def interpret(self, notes: list[str], battery: Battery, deadline: float) -> list[NoteResult]:
        return list(
            await asyncio.gather(*(self._interpret_note(i, notes, battery, deadline) for i in range(len(notes))))
        )

    async def reinterpret(
        self, index: int, notes: list[str], battery: Battery, deadline: float, feedback: list[str]
    ) -> NoteResult:
        """Fresh interpretation with extra feedback (used when directives make the day infeasible)."""
        if not self.configured:
            return NoteResult(directive=no_op(index, "LLM not configured.", "fallback"), degraded=True)
        sem, meta = await self._ask(index, notes, battery, deadline, extra_feedback=feedback)
        return self._finish(index, notes, battery, sem, meta, cached=False, cache_key=_cache_key(notes, index))

    # ------------------------------------------------------------------ internals

    async def _interpret_note(self, index: int, notes: list[str], battery: Battery, deadline: float) -> NoteResult:
        if not self.configured:
            return NoteResult(
                directive=no_op(
                    index,
                    "LLM interpretation unavailable (OPENAI_API_KEY not configured); treated as no_op.",
                    "fallback",
                ),
                degraded=True,
                error="llm_not_configured",
            )
        key = _cache_key(notes, index)
        hit = self._cache.get(key)
        if hit is not None:
            sem, model = hit
            return self._finish(index, notes, battery, sem, {"model": model, "attempts": 0}, cached=True, cache_key=None)

        if key in self._inflight:
            try:
                sem, meta = await asyncio.shield(self._inflight[key])
                return self._finish(index, notes, battery, sem, dict(meta, attempts=0), cached=True, cache_key=None)
            except Exception:
                pass  # the leader failed; try on our own below

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._inflight[key] = future
        try:
            sem, meta = await self._ask(index, notes, battery, deadline)
            if not future.done():
                future.set_result((sem, meta))
        except BaseException as exc:  # pragma: no cover - _ask never raises by design
            if not future.done():
                future.set_exception(exc)
                future.exception()  # mark retrieved
            raise
        finally:
            self._inflight.pop(key, None)
        return self._finish(index, notes, battery, sem, meta, cached=False, cache_key=key)

    def _finish(self, index, notes, battery, sem, meta, cached: bool, cache_key: str | None) -> NoteResult:
        if sem is None:
            reason = meta.get("error") or "unknown error"
            log.warning("note %d: LLM interpretation failed (%s); using no_op", index, reason)
            return NoteResult(
                directive=no_op(index, "LLM interpretation unavailable for this note; treated as no_op.", "fallback"),
                model=meta.get("model"),
                attempts=meta.get("attempts", 0),
                degraded=True,
                error=reason,
            )
        directive, problems = normalize(sem, battery, index, notes[index], strict=False)
        if cache_key is not None:
            self._cache.put(cache_key, (sem, meta.get("model")))
        return NoteResult(
            directive=directive,
            semantic=sem,
            model=meta.get("model"),
            attempts=meta.get("attempts", 0),
            cached=cached,
            degraded=directive.source == "fallback",
            issues=problems + meta.get("issues", []),
        )

    def _review(self, sem: SemanticInterpretation, battery: Battery, index: int, note: str) -> list[str]:
        _, problems = normalize(sem, battery, index, note, strict=True)
        return problems + crosscheck(sem, note)

    async def _ask(
        self,
        index: int,
        notes: list[str],
        battery: Battery,
        deadline: float,
        extra_feedback: list[str] | None = None,
    ) -> tuple[SemanticInterpretation | None, dict]:
        loop = asyncio.get_running_loop()
        note = notes[index]
        base = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message(notes, index)},
        ]
        if extra_feedback:
            base.append({"role": "user", "content": feedback_message(extra_feedback)})

        meta: dict = {"model": None, "attempts": 0, "issues": [], "error": None}
        best: SemanticInterpretation | None = None
        reasked = False
        try:
            chain = await asyncio.wait_for(self.client.model_chain(), timeout=max(0.5, deadline - loop.time() - 5))
        except Exception:
            chain = list(self.settings.model_preference[:2])

        for model in chain:
            messages = list(base)
            tries = 0
            while tries < 2:
                remaining = deadline - loop.time() - SOLVE_RESERVE_S
                if remaining < MIN_ATTEMPT_S:
                    meta["error"] = meta["error"] or "deadline"
                    return best, meta
                tries += 1
                meta["attempts"] += 1
                meta["model"] = model
                timeout = min(self.settings.llm_timeout_s, remaining)
                started = time.perf_counter()
                try:
                    async with self._sem():
                        sem = await self.client.interpret(model, messages, timeout)
                except LLMCallError as exc:
                    meta["error"] = exc.kind
                    self._record(False, exc.kind)
                    log.warning(
                        "note %d: %s attempt %d failed after %.0f ms: %s",
                        index, model, tries, (time.perf_counter() - started) * 1000, exc,
                    )
                    if exc.kind == "not_found":
                        self.client.mark_dead(model)
                        break
                    if exc.kind in ("auth", "bad_request", "refusal", "quota"):
                        break
                    if exc.kind == "rate_limit" and exc.retry_after:
                        pause = min(exc.retry_after, 2.0, max(0.0, remaining - MIN_ATTEMPT_S))
                        if pause > 0:
                            await asyncio.sleep(pause)
                    continue
                except Exception as exc:  # defensive: never let one note crash the request
                    meta["error"] = type(exc).__name__
                    self._record(False, type(exc).__name__)
                    log.error("note %d: unexpected LLM client error %s", index, type(exc).__name__)
                    continue

                self._record(True, None)
                issues = self._review(sem, battery, index, note)
                best = sem
                meta["error"] = None
                if not issues:
                    meta["issues"] = []
                    return sem, meta
                meta["issues"] = issues
                if reasked:
                    log.info("note %d: accepting re-asked answer despite: %s", index, issues)
                    return sem, meta
                reasked = True
                log.info("note %d: re-asking %s because: %s", index, model, issues)
                messages = messages + [
                    {"role": "assistant", "content": sem.model_dump_json()},
                    {"role": "user", "content": feedback_message(issues)},
                ]
            if best is not None:
                return best, meta
        return best, meta
