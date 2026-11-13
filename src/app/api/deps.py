"""FastAPI dependencies — principal auth and admin check."""

from fastapi import Header

from app.core.config import get_settings
from app.core.errors import ForbiddenError
from app.security.authn import Principal, authenticate


def get_principal(
    x_api_key: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
) -> Principal:
    return authenticate(x_api_key, x_user_id)


def get_admin(
    x_admin_token: str | None = Header(default=None),
) -> None:
    if x_admin_token != get_settings().admin_token:
        raise ForbiddenError("admin token required")
