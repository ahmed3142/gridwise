"""`python -m app` - start the API on 0.0.0.0:$PORT (default 8080). Works the same on Windows, Linux, Docker."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    try:  # make PORT / HOST / LOG_LEVEL / WEB_CONCURRENCY from .env effective here too
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except Exception:  # pragma: no cover
        pass
    level = os.environ.get("LOG_LEVEL", "info").strip().lower()
    level = {"warn": "warning", "fatal": "critical"}.get(level, level)
    if level not in {"critical", "error", "warning", "info", "debug", "trace"}:
        level = "info"
    uvicorn.run(
        "app.main:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8080")),
        workers=int(os.environ.get("WEB_CONCURRENCY", "1")),
        proxy_headers=True,
        forwarded_allow_ips="*",
        timeout_keep_alive=30,
        access_log=False,
        log_level=level,
    )


if __name__ == "__main__":
    main()
