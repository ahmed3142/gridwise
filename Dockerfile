# GridWise LLM - production image (linux/amd64 recommended: docker buildx build --platform linux/amd64 ...)
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080 \
    HOST=0.0.0.0

WORKDIR /srv

COPY requirements.txt requirements.lock ./
RUN pip install -r requirements.lock

COPY app ./app

# Unprivileged runtime user; no secrets are copied into the image (.env is excluded by .dockerignore).
RUN useradd --create-home --uid 10001 gridwise
USER gridwise

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','8080'), timeout=4)" || exit 1

# `python -m app` binds 0.0.0.0:$PORT (defaults to 8080) - works on Fly.io, Railway, Render, Cloud Run.
CMD ["python", "-m", "app"]
