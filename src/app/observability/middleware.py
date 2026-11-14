"""Request middleware — request_id generation + structlog context.

Author: venkateshwara doijode
"""

import re
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Binds a request id into structlog context and echoes it back as a header."""

    async def dispatch(self, request: Request, call_next):
        """Assign/propagate the request id, bind log contextvars, and set X-Request-Id."""
        supplied = request.headers.get("X-Request-Id") or ""
        # caller-controlled values land in logs/audit — accept only a safe
        # charset/length, else mint our own
        request_id = supplied if _REQUEST_ID_RE.match(supplied) else uuid.uuid4().hex[:16]
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path)
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response
