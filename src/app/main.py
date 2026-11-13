"""FastAPI application factory."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api.routers import admin, answer, audit, health
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
    configure_logging()
    try:
        from app.core.dependencies import get_audit

        get_audit().ensure_table()
    except Exception as exc:
        log.warning("audit.table_init_failed", error=str(exc))
    yield


app = FastAPI(title="Agentic RAG Client-Service Copilot", lifespan=lifespan)
app.add_middleware(RequestContextMiddleware)


@app.exception_handler(AppError)
async def app_error_handler(request, exc: AppError):
    status = _STATUS.get(type(exc), 500)
    return JSONResponse(status_code=status, content={"detail": str(exc)})


app.include_router(answer.router, tags=["copilot"])
app.include_router(health.router, tags=["health"])
app.include_router(admin.router, tags=["admin"])
app.include_router(audit.router, tags=["audit"])
