"""Prototype authentication: X-API-Key + X-User-Id → Principal.

Users and entitlements are loaded from a JSON fixture (db/seed/users.json).
Production path: replace with OIDC/JWT introspection — the Principal
interface and authz filters stay identical.

Author: venkateshwara doijode
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings
from app.core.errors import UnauthorizedError


@dataclass(frozen=True)
class Principal:
    """Authenticated caller identity: user id, role, and entitled client ids."""

    user_id: str
    role: str
    client_ids: tuple[str, ...]

    @property
    def is_admin(self) -> bool:
        """Return True when the principal has the ``admin`` role."""
        return self.role == "admin"


@lru_cache
def _load_users(path: str) -> dict[str, dict]:
    """Load and cache the users fixture file, keyed by ``user_id``."""
    return {u["user_id"]: u for u in json.loads(Path(path).read_text())}


def authenticate(api_key: str | None, user_id: str | None) -> Principal:
    """Validate API key + user id headers and return the caller's ``Principal``.

    Raises ``UnauthorizedError`` for a missing/invalid key, missing user id,
    unknown user, or a key that doesn't match the user record.
    """
    settings = get_settings()
    if not api_key or api_key not in settings.valid_api_keys:
        raise UnauthorizedError("missing or invalid API key")
    if not user_id:
        raise UnauthorizedError("missing X-User-Id header")
    try:
        user = _load_users(str(settings.users_file))[user_id]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise UnauthorizedError(f"unknown user: {user_id}") from exc
    if user.get("api_key") != api_key:
        raise UnauthorizedError("API key does not match user")
    return Principal(user_id=user_id, role=user["role"], client_ids=tuple(user["client_ids"]))
