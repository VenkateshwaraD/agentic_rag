"""Application error types mapped to HTTP responses at the API layer.

Every error carries a stable machine-readable ``code`` so the API can emit
a consistent ``{"error": {"code", "message"}}`` envelope without leaking
internal exception text.

Author: venkateshwara doijode
"""


class AppError(Exception):
    """Base application error with a stable public error code."""

    code = "internal_error"

    def __init__(self, message: str = "", *, code: str | None = None):
        """Store the sanitized message and optional explicit code."""
        super().__init__(message or self.code)
        self.message = message or self.code
        if code:
            self.code = code


class UnauthorizedError(AppError):
    """Missing/invalid API key or unknown user."""

    code = "unauthorized"


class ForbiddenError(AppError):
    """Authenticated but not permitted (admin-only resource, etc.)."""

    code = "forbidden"


class GuardrailRejection(AppError):
    """Input or output blocked by a guardrail."""

    code = "guardrail_rejected"


class NotFoundError(AppError):
    """Requested resource does not exist."""

    code = "not_found"


class BudgetExceededError(AppError):
    """A hard agent budget limit was hit mid-request."""

    code = "budget_exceeded"


class IdempotencyConflictError(AppError):
    """An idempotency key was replayed with a different request body."""

    code = "idempotency_conflict"
