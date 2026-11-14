"""FastAPI application factory.

Mount layout: the public surface lives under ``/api/v1``; the original flat
routes stay mounted (marked deprecated in docs) for backward compatibility.

Author: venkateshwara doijode
"""

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routers import admin, answer, audit, health
from app.core.config import get_settings
from app.core.errors import AppError, ForbiddenError, NotFoundError, UnauthorizedError
from app.core.logging import configure_logging, get_logger
from app.observability.middleware import RequestContextMiddleware

log = get_logger(__name__)

_STATUS = {
    UnauthorizedError: 401,
    ForbiddenError: 403,
    NotFoundError: 404,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: configure logging and ensure the audit table exists (best-effort)."""
    configure_logging()
    get_settings().assert_safe_for_production()
    try:
        from app.core.dependencies import get_audit

        get_audit().ensure_table()
    except Exception as exc:
        log.warning("audit.table_init_failed", error=str(exc)[:200])
    yield


app = FastAPI(title="Agentic RAG Client-Service Copilot", lifespan=lifespan)
app.add_middleware(RequestContextMiddleware)


def _envelope(request, code: str, message: str) -> dict:
    """Build the consistent ``{"error": {...}}`` body with the request id."""
    return {
        "error": {
            "code": code,
            "message": message,
            "request_id": getattr(request.state, "request_id", uuid.uuid4().hex[:16]),
        }
    }


@app.exception_handler(AppError)
async def app_error_handler(request, exc: AppError):
    """Map application errors to status + the stable error envelope."""
    status = _STATUS.get(type(exc), 500 if exc.code == "internal_error" else 400)
    if exc.code == "idempotency_conflict":
        status = 409
    return JSONResponse(status_code=status, content=_envelope(request, exc.code, str(exc)))


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request, exc: RequestValidationError):
    """Map request-validation failures to a 422 with the error envelope."""
    return JSONResponse(
        status_code=422,
        content=_envelope(request, "invalid_input", "request validation failed"),
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request, exc: Exception):
    """Catch-all: a sanitized 500 — internals never reach the client."""
    log.exception("unhandled_error", error=type(exc).__name__)
    return JSONResponse(
        status_code=500,
        content=_envelope(request, "internal_error", "unexpected server error"),
    )


# versioned public API
app.include_router(answer.router, prefix="/api/v1", tags=["copilot"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(admin.router, prefix="/api/v1", tags=["admin"])
app.include_router(audit.router, prefix="/api/v1", tags=["audit"])

# legacy flat routes — kept for existing clients; deprecated in the docs
app.include_router(answer.router, tags=["copilot-legacy"], deprecated=True)
app.include_router(health.router, tags=["health-legacy"], deprecated=True)
app.include_router(admin.router, tags=["admin-legacy"], deprecated=True)
app.include_router(audit.router, tags=["audit-legacy"], deprecated=True)
