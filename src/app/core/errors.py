"""Application error types mapped to HTTP responses at the API layer."""


class AppError(Exception):
    """Base application error."""


class UnauthorizedError(AppError):
    """Missing/invalid API key or unknown user."""


class ForbiddenError(AppError):
    """Authenticated but not permitted (admin-only resource, etc.)."""


class GuardrailRejection(AppError):
    """Input or output blocked by a guardrail."""


class NotFoundError(AppError):
    """Requested resource does not exist."""
