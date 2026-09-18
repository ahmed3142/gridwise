"""Single-line structured logging. Never logs API keys, request bodies, or raw prompts."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if getattr(root, "_gridwise_configured", False):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.handlers[:] = [handler]
    root.setLevel(getattr(logging, level, logging.INFO))
    for noisy in ("httpx", "httpx2", "httpcore", "openai", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    root._gridwise_configured = True  # type: ignore[attr-defined]
