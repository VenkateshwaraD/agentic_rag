"""Shared API schemas — the consistent error envelope.

Every error response (auth failure, guardrail rejection, validation error,
unexpected exception) uses ``{"error": {"code", "message", "request_id"}}``
so clients can branch on stable codes rather than parsing exception text.

Author: venkateshwara doijode
"""

from pydantic import BaseModel


class ErrorBody(BaseModel):
    """Machine-readable error payload."""

    code: str
    message: str
    request_id: str = ""


class ErrorEnvelope(BaseModel):
    """Top-level error response wrapper."""

    error: ErrorBody
