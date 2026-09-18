"""`python -m app` - start the API on 0.0.0.0:$PORT (default 8080). Works the same on Windows, Linux, Docker."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "app.main:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8080")),
        workers=int(os.environ.get("WEB_CONCURRENCY", "1")),
        proxy_headers=True,
        forwarded_allow_ips="*",
        timeout_keep_alive=30,
        access_log=False,
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()
