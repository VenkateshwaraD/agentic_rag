"""Idempotency store for retryable answer requests.

Keyed by ``(user_id, Idempotency-Key)``: a replay with the same request body
returns the stored response marked ``idempotent_replay``; a replay with a
*different* body is rejected as ``idempotency_conflict``.

Storage is Postgres-backed (``idempotency_keys`` in the rag database) so the
guarantee holds across replicas — the k8s deployment scales to several pods
and a process-local dict would let duplicates execute on different pods.
``claim`` performs an atomic insert-if-absent *before* the request runs, so
concurrent duplicates cannot both execute; ``complete`` fills in the
response afterwards. When the database is unreachable the store degrades to
a process-local dict (best-effort, per-replica only) — requests proceed but
cross-replica idempotency is lost; that failure is logged, not hidden.

TTL comes from ``settings.idempotency_ttl_s``.

Author: venkateshwara doijode
"""

import hashlib
import time
from typing import Any, Literal

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger("idempotency")

_MAX_KEYS = 10_000

# process-local fallback, same shape as the PG row:
# (user_id, key) -> (body_hash, epoch_seconds, response|None)
_local: dict[tuple[str, str], tuple[str, float, dict | None]] = {}

_DDL = """
CREATE TABLE IF NOT EXISTS idempotency_keys (
    user_id     TEXT NOT NULL,
    key         TEXT NOT NULL,
    body_hash   TEXT NOT NULL,
    response    JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, key)
)
"""

ClaimStatus = Literal["claimed", "replay", "conflict", "in_flight", "unavailable"]


def _ttl() -> float:
    """Configured idempotency TTL in seconds."""
    return float(get_settings().idempotency_ttl_s)


def _body_hash(question: str, client_id: str | None) -> str:
    """Stable hash of the request inputs a cached response is valid for."""
    return hashlib.sha256(f"{question}\x00{client_id or ''}".encode()).hexdigest()


# --- postgres store ---------------------------------------------------------


def _pg_claim(user_id: str, key: str, body_hash: str) -> tuple[ClaimStatus, dict | None]:
    """Atomically insert-if-absent in Postgres; classify an existing row."""
    from sqlalchemy import text

    from app.db.session import rag_engine

    with rag_engine().begin() as conn:
        conn.execute(text(_DDL))
        claimed = conn.execute(
            text(
                "INSERT INTO idempotency_keys (user_id, key, body_hash) "
                "VALUES (:u, :k, :h) ON CONFLICT DO NOTHING"
            ),
            {"u": user_id, "k": key, "h": body_hash},
        ).rowcount
        if claimed:
            return "claimed", None
        row = conn.execute(
            text(
                "SELECT body_hash, response, "
                "extract(epoch from now() - created_at) AS age "
                "FROM idempotency_keys WHERE user_id = :u AND key = :k"
            ),
            {"u": user_id, "k": key},
        ).fetchone()
        if row is None:
            return "claimed", None  # raced delete — proceed
        if row[2] > _ttl():
            # expired — reclaim the key
            conn.execute(
                text("DELETE FROM idempotency_keys WHERE user_id = :u AND key = :k"),
                {"u": user_id, "k": key},
            )
            conn.execute(
                text(
                    "INSERT INTO idempotency_keys (user_id, key, body_hash) "
                    "VALUES (:u, :k, :h) ON CONFLICT DO NOTHING"
                ),
                {"u": user_id, "k": key, "h": body_hash},
            )
            return "claimed", None
        if row[0] != body_hash:
            return "conflict", None
        if row[1] is None:
            return "in_flight", None
        return "replay", row[1]


def _pg_complete(user_id: str, key: str, response: dict) -> None:
    """Fill in the stored response for a claimed key."""
    import json

    from sqlalchemy import text

    from app.db.session import rag_engine

    with rag_engine().begin() as conn:
        conn.execute(
            text(
                "UPDATE idempotency_keys SET response = CAST(:r AS jsonb) "
                "WHERE user_id = :u AND key = :k"
            ),
            {"u": user_id, "k": key, "r": json.dumps(response)},
        )


# --- in-process fallback ----------------------------------------------------


def _local_sweep() -> None:
    """Drop expired local entries; hard-cap the fallback store."""
    now = time.time()
    expired = [k for k, (_, ts, _) in _local.items() if now - ts > _ttl()]
    for k in expired:
        _local.pop(k, None)
    if len(_local) > _MAX_KEYS:
        for k in sorted(_local, key=lambda k: _local[k][1])[: len(_local) - _MAX_KEYS]:
            _local.pop(k, None)


def _local_claim(user_id: str, key: str, body_hash: str) -> tuple[ClaimStatus, dict | None]:
    """Process-local claim — atomic under the GIL but per-replica only."""
    _local_sweep()
    entry = _local.get((user_id, key))
    if entry is None:
        _local[(user_id, key)] = (body_hash, time.time(), None)
        return "claimed", None
    if entry[0] != body_hash:
        return "conflict", None
    return ("in_flight", None) if entry[2] is None else ("replay", entry[2])


# --- public API ---------------------------------------------------------------


def claim(
    user_id: str, key: str, question: str, client_id: str | None
) -> tuple[ClaimStatus, dict | None]:
    """Atomically claim ``(user_id, key)`` before the request executes.

    Returns ``(status, response)``:
    - ``claimed`` — caller owns the key; run the request then ``complete``
    - ``replay`` — cached response returned
    - ``conflict`` — same key, different body
    - ``in_flight`` — a duplicate is still executing (safe to 409/425)
    - ``unavailable`` — store down; proceeds without idempotency
    """
    body_hash = _body_hash(question, client_id)
    try:
        return _pg_claim(user_id, key, body_hash)
    except Exception as exc:
        log.warning("idempotency.store_degraded", error=str(exc)[:200])
        return _local_claim(user_id, key, body_hash)


def complete(user_id: str, key: str, response: dict) -> None:
    """Store the response for a previously claimed key."""
    try:
        _pg_complete(user_id, key, response)
        return
    except Exception as exc:
        log.warning("idempotency.complete_degraded", error=str(exc)[:200])
    entry = _local.get((user_id, key))
    if entry is not None:
        _local[(user_id, key)] = (entry[0], entry[1], response)


def lookup(user_id: str, key: str, question: str, client_id: str | None) -> dict[str, Any] | None:
    """Return the cached response for a key replay, else None.

    Raises ``IdempotencyConflictError`` when the key was seen with a
    different request body (client retried with changed inputs).
    """
    from app.core.errors import IdempotencyConflictError

    status, response = claim(user_id, key, question, client_id)
    if status == "conflict":
        raise IdempotencyConflictError(
            "Idempotency-Key was already used with a different request body"
        )
    return response if status == "replay" else None


def store(user_id: str, key: str, question: str, client_id: str | None, response: dict) -> None:
    """Claim + fill in one shot — used by tests and non-streaming writes."""
    status, _ = claim(user_id, key, question, client_id)
    if status == "claimed":
        complete(user_id, key, response)


def clear() -> None:
    """Flush the cache — used by tests."""
    _local.clear()
    try:
        from sqlalchemy import text

        from app.db.session import rag_engine

        with rag_engine().begin() as conn:
            conn.execute(text("DELETE FROM idempotency_keys"))
    except Exception:
        pass  # table may not exist / db down — tests only
