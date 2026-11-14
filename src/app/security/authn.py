"""Prototype authentication: X-API-Key + X-User-Id → Principal.

Users and entitlements are loaded from a JSON fixture (db/seed/users.json).
Production path: replace with OIDC/JWT introspection — the Principal
interface and authz filters stay identical.

Author: venkateshwara doijode
"""

import hmac
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
def _load_users_cached(path: str, _mtime: float) -> dict[str, dict]:
    """Parse the users fixture; ``_mtime`` keys the cache so edits reload."""
    return {u["user_id"]: u for u in json.loads(Path(path).read_text())}


def load_users(path: str | Path | None = None) -> dict[str, dict]:
    """Load the users fixture keyed by ``user_id``, rejecting shared keys.

    One key may map to exactly one user — a shared key lets its holder
    present any ``X-User-Id`` whose record carries it (role escalation +
    cross-tenant reads), so a fixture that violates this fails closed.
    Cached but reloaded when the file's mtime changes.
    """
    users_path = Path(path or get_settings().users_file)
    users = _load_users_cached(str(users_path), users_path.stat().st_mtime)
    seen: dict[str, str] = {}
    for uid, u in users.items():
        key = u.get("api_key")
        if key:
            if key in seen:
                raise UnauthorizedError(
                    f"users fixture shares one API key across users ({seen[key]}, {uid})"
                )
            seen[key] = uid
    return users


def authenticate(api_key: str | None, user_id: str | None) -> Principal:
    """Validate API key + user id headers and return the caller's ``Principal``.

    Raises ``UnauthorizedError`` for a missing/invalid key, missing user id,
    unknown user, or a key that doesn't match the user record. All secret
    comparisons are constant-time.
    """
    settings = get_settings()
    if not api_key or not any(hmac.compare_digest(api_key, k) for k in settings.valid_api_keys):
        raise UnauthorizedError("missing or invalid API key")
    if not user_id:
        raise UnauthorizedError("missing X-User-Id header")
    try:
        user = load_users(settings.users_file)[user_id]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise UnauthorizedError(f"unknown user: {user_id}") from exc
    if not hmac.compare_digest(user.get("api_key") or "", api_key):
        raise UnauthorizedError("API key does not match user")
    return Principal(user_id=user_id, role=user["role"], client_ids=tuple(user["client_ids"]))
