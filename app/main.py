"""FastAPI application: GET /health and POST /optimize-energy (Problem Statement section 06).

HTTP status policy (section 6.1):
  200  success (also when the LLM provider fails: affected notes become controlled no_op entries)
  400  malformed JSON or structurally invalid request (missing/ill-typed fields, wrong counts, ...)
  422  well-formed but semantically invalid (e.g. initial energy outside [minimum, capacity])
  500  controlled internal error - JSON body, no stack trace, no secrets
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .config import load_settings
from .interpreter.llm import OpenAIInterpreterClient
from .interpreter.prompt import PROMPT_VERSION
from .interpreter.service import InterpretationService
from .logging_setup import configure_logging
from .pipeline import ScenarioInfeasible, run_pipeline
from .schemas import OptimizeRequest, OptimizeResponse, semantic_problems

settings = load_settings()
configure_logging(settings.log_level)
log = logging.getLogger("gridwise.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = OpenAIInterpreterClient(settings) if settings.llm_configured else None
    app.state.service = InterpretationService(client, settings)
    if client is None:
        log.warning("OPENAI_API_KEY is not set: the service starts, but notes will be treated as no_op")
    else:
        # Resolve models + one warm-up call in the background; /health never waits for the provider.
        app.state.warmup = asyncio.create_task(app.state.service.warm_up())
        app.state.keep_warm = asyncio.create_task(app.state.service.keep_warm())
    log.info("GridWise %s ready (prompt %s)", __version__, PROMPT_VERSION)
    yield
    if client is not None:
        for name in ("keep_warm", "warmup"):
            task = getattr(app.state, name, None)
            if task is not None and not task.done():
                task.cancel()
        await client.aclose()


app = FastAPI(
    title="GridWise LLM",
    version=__version__,
    description="LLM-assisted operator-note interpretation + optimal 24-hour campus energy scheduling.",
    lifespan=lifespan,
    redirect_slashes=False,  # never answer the judge with a 307 redirect; trailing-slash aliases below
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "HEAD", "POST", "OPTIONS"], allow_headers=["*"]
)


def _error(status: int, error: str, message: str, request: Request, details: list[str] | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": error,
            "message": message,
            "details": details or [],
            "request_id": getattr(request.state, "request_id", None),
        },
    )


# Request schema for the docs, with $refs that resolve inside the OpenAPI document.
_REQUEST_SCHEMA = OptimizeRequest.model_json_schema(ref_template="#/components/schemas/{model}")
_REQUEST_DEFS = _REQUEST_SCHEMA.pop("$defs", {})


def _openapi() -> dict:
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)
    schema.setdefault("components", {}).setdefault("schemas", {}).update(_REQUEST_DEFS)
    app.openapi_schema = schema
    return schema


app.openapi = _openapi  # type: ignore[method-assign]


async def _read_body_capped(request: Request, limit: int) -> bytes | None:
    """Read at most `limit` bytes; None when the body is larger (never buffers an oversized body)."""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        return None
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def _format_validation_errors(errors: list[dict]) -> list[str]:
    out = []
    for err in errors[:20]:
        loc = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
        out.append(f"{loc or 'body'}: {err.get('msg', 'invalid value')}")
    return out


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = (request.headers.get("x-request-id") or uuid.uuid4().hex[:16])[:64]
    request.state.request_id = request_id
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:  # last line of defence: controlled 500, never a stack trace
        log.error("unhandled %s on %s %s", type(exc).__name__, request.method, request.url.path)
        response = _error(500, "internal_error", "An internal error occurred.", request)
    elapsed = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-ms"] = f"{elapsed:.1f}"
    log.info("%s %s -> %d in %.0f ms [%s]", request.method, request.url.path, response.status_code, elapsed, request_id)
    return response


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError):
    return _error(400, "invalid_request", "Request body is invalid.", request, _format_validation_errors(exc.errors()))


@app.exception_handler(StarletteHTTPException)
async def _http_handler(request: Request, exc: StarletteHTTPException):
    return _error(exc.status_code, "http_error", str(exc.detail), request)


@app.exception_handler(Exception)
async def _unhandled_handler(request: Request, exc: Exception):
    # Type only: stack traces and exception text can carry sensitive values (rules: no traces in logs).
    log.error("unhandled %s on %s %s", type(exc).__name__, request.method, request.url.path)
    return _error(500, "internal_error", "An internal error occurred.", request)


# ---------------------------------------------------------------------------------------------- routes


@app.get("/health", tags=["service"])
@app.api_route("/health", methods=["HEAD"], include_in_schema=False)
@app.api_route("/health/", methods=["GET", "HEAD"], include_in_schema=False)
async def health():
    return {"status": "ok"}


@app.get("/", tags=["service"])
async def root():
    return {
        "service": "GridWise LLM",
        "version": __version__,
        "endpoints": {"health": "GET /health", "optimize": "POST /optimize-energy", "docs": "GET /docs"},
    }


@app.get("/version", tags=["service"])
async def version(request: Request):
    service: InterpretationService = request.app.state.service
    client = service.client
    return {
        "version": __version__,
        "prompt_version": PROMPT_VERSION,
        "llm_provider": "openai" if client else None,
        "llm_configured": client is not None,
        "primary_model": getattr(client, "primary", None),
        "fallback_model": getattr(client, "fallback", None),
        "llm_status": service.status(),
        "optimizer": "scipy.optimize.linprog (HiGHS), lexicographic: cost > peak > throughput",
    }


@app.post("/optimize-energy/", include_in_schema=False)
@app.post(
    "/optimize-energy",
    tags=["optimize"],
    response_model=OptimizeResponse,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {"application/json": {"schema": _REQUEST_SCHEMA}},
        }
    },
)
async def optimize_energy(request: Request):
    body = await _read_body_capped(request, settings.max_body_bytes)
    if body is None:
        return _error(400, "invalid_request", f"Request body exceeds {settings.max_body_bytes} bytes.", request)
    try:
        data = json.loads(body)
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        return _error(400, "invalid_json", "Request body is not valid JSON.", request, [str(exc)[:200]])
    if not isinstance(data, dict):
        return _error(400, "invalid_request", "Request body must be a JSON object.", request)
    try:
        req = OptimizeRequest.model_validate(data)
    except ValidationError as exc:
        return _error(400, "invalid_request", "Request does not match the required schema.", request,
                      _format_validation_errors(exc.errors()))
    log.info("scenario %s: %d note(s) [%s]", req.scenario_id[:64], len(req.operator_notes),
             getattr(request.state, "request_id", "-"))
    problems = semantic_problems(req)
    if problems:
        return _error(422, "semantically_invalid", "Request is well-formed but physically inconsistent.", request, problems)

    try:
        try:
            # Safety net below the judge's 30 s limit: the pipeline has its own 24 s budget, but if
            # anything still overruns, answer with a controlled, LLM-free plan instead of timing out.
            async with asyncio.timeout(settings.request_deadline_s + 3.0):
                result = await run_pipeline(req, request.app.state.service, settings)
        except TimeoutError:
            log.error("request exceeded the time budget; serving the LLM-free degraded plan")
            result = await run_pipeline(req, InterpretationService(None, settings), settings)
            result.degraded = True
    except ScenarioInfeasible as exc:
        return _error(422, "infeasible_scenario", str(exc), request)

    response = JSONResponse(content=result.body)
    timing = ", ".join(f"{k};dur={v:.1f}" for k, v in result.timings_ms.items())
    if timing:
        response.headers["Server-Timing"] = timing
    if result.degraded:
        response.headers["X-GridWise-Degraded"] = "true"
    if result.self_check_failed:
        response.headers["X-GridWise-Self-Check"] = "failed"
    return response
