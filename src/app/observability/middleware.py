"""Request middleware — request_id generation + structlog context.

Author: venkateshwara doijode
"""

import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Binds a request id into structlog context and echoes it back as a header."""

    async def dispatch(self, request: Request, call_next):
        """Assign/propagate the request id, bind log contextvars, and set X-Request-Id."""
        request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex[:16]
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path)
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response
