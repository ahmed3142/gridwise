"""Runtime configuration, read once from environment variables (and an optional .env file).

No secret is ever logged or returned. See README "Environment variables" for the full list.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:  # python-dotenv is optional at runtime; real environment variables always win.
    from dotenv import load_dotenv

    load_dotenv(override=False)
except Exception:  # pragma: no cover - dotenv missing or unreadable .env
    pass


# Tried in order when OPENAI_MODEL=auto. The first model the API key can access becomes the
# primary interpreter; the next accessible one becomes the fallback.
DEFAULT_MODEL_PREFERENCE: tuple[str, ...] = (
    "gpt-4.1-mini",
    "gpt-5.4-mini",
    "gpt-5-mini",
    "gpt-4o-mini",
    "gpt-4.1",
)


def _env_str(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _env_float(name: str, default: float, low: float, high: float) -> float:
    raw = _env_str(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return min(max(value, low), high)


def _env_int(name: str, default: int, low: int, high: int) -> int:
    raw = _env_str(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return min(max(value, low), high)


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = _env_str(name)
    if raw is None:
        return default
    items = tuple(part.strip() for part in raw.split(",") if part.strip())
    return items or default


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    openai_base_url: str | None
    model: str
    fallback_model: str | None
    model_preference: tuple[str, ...]
    reasoning_effort: str | None
    llm_timeout_s: float
    request_deadline_s: float
    llm_max_concurrency: int
    max_output_tokens: int
    cache_size: int
    max_body_bytes: int
    log_level: str

    @property
    def llm_configured(self) -> bool:
        return bool(self.openai_api_key)


def load_settings() -> Settings:
    effort = _env_str("OPENAI_REASONING_EFFORT", "low")
    if effort is not None and effort.lower() in {"off", "omit", "none-param"}:
        effort = None
    return Settings(
        openai_api_key=_env_str("OPENAI_API_KEY"),
        openai_base_url=_env_str("OPENAI_BASE_URL"),
        model=_env_str("OPENAI_MODEL", "auto") or "auto",
        fallback_model=_env_str("OPENAI_FALLBACK_MODEL"),
        model_preference=_env_list("OPENAI_MODEL_PREFERENCE", DEFAULT_MODEL_PREFERENCE),
        reasoning_effort=effort,
        llm_timeout_s=_env_float("LLM_TIMEOUT_SECONDS", 9.0, 1.0, 25.0),
        request_deadline_s=_env_float("REQUEST_DEADLINE_SECONDS", 24.0, 5.0, 28.0),
        llm_max_concurrency=_env_int("LLM_MAX_CONCURRENCY", 12, 1, 64),
        max_output_tokens=_env_int("LLM_MAX_OUTPUT_TOKENS", 2500, 256, 16000),
        cache_size=_env_int("INTERPRETATION_CACHE_SIZE", 2048, 0, 100000),
        max_body_bytes=_env_int("MAX_BODY_BYTES", 1_000_000, 10_000, 20_000_000),
        log_level=(_env_str("LOG_LEVEL", "INFO") or "INFO").upper(),
    )
