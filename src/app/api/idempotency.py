"""In-process idempotency cache for retryable answer requests.

Keyed by ``(user_id, Idempotency-Key)``: a replay with the same request body
returns the stored response marked ``idempotent_replay``; a replay with a
*different* body is rejected as ``idempotency_conflict``. In-process TTL —
sufficient for the single-replica prototype; swap the store for Redis/DB
behind the same interface when scaling out.

Author: venkateshwara doijode
"""

import hashlib
import time
from typing import Any

_TTL_S = 15 * 60
_MAX_KEYS = 10_000

_store: dict[tuple[str, str], tuple[str, float, dict]] = {}


def _body_hash(question: str, client_id: str | None) -> str:
    """Stable hash of the request inputs a cached response is valid for."""
    return hashlib.sha256(f"{question}\x00{client_id or ''}".encode()).hexdigest()


def _sweep(now: float) -> None:
    """Drop expired entries; hard-cap the store when it grows unboundedly."""
    expired = [k for k, (_, ts, _) in _store.items() if now - ts > _TTL_S]
    for k in expired:
        _store.pop(k, None)
    if len(_store) > _MAX_KEYS:
        for k in sorted(_store, key=lambda k: _store[k][1])[: len(_store) - _MAX_KEYS]:
            _store.pop(k, None)


def lookup(user_id: str, key: str, question: str, client_id: str | None) -> dict[str, Any] | None:
    """Return the cached response for a key replay, else None.

    Raises ``IdempotencyConflictError`` when the key was seen with a
    different request body (client retried with changed inputs).
    """
    from app.core.errors import IdempotencyConflictError

    _sweep(time.monotonic())
    entry = _store.get((user_id, key))
    if entry is None:
        return None
    body_hash, _, response = entry
    if body_hash != _body_hash(question, client_id):
        raise IdempotencyConflictError(
            "Idempotency-Key was already used with a different request body"
        )
    return response


def store(user_id: str, key: str, question: str, client_id: str | None, response: dict) -> None:
    """Cache a response under ``(user_id, key)`` for future replays."""
    _sweep(time.monotonic())
    _store[(user_id, key)] = (_body_hash(question, client_id), time.monotonic(), response)


def clear() -> None:
    """Flush the cache — used by tests."""
    _store.clear()
