"""Human-review queue — flags high-impact or ambiguous responses for
asynchronous reviewer approval.

Triggers: partial or conflicting answers, low citation correctness,
validator exhaustion, ops-tool errors, or high-impact ops categories.
Queue rows are persisted in ragdb (self-DDL like ``audit_events``);
failures degrade to a log warning so a DB outage never blocks a reply.

Author: venkateshwara doijode
"""

from typing import Any

from sqlalchemy import text

from app.core.config import get_settings
from app.core.constants import Answerability
from app.core.logging import get_logger
from app.db.session import rag_engine

log = get_logger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS review_queue (
    id            SERIAL PRIMARY KEY,
    request_id    TEXT NOT NULL,
    user_id       TEXT NOT NULL,
    queued_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    reason        TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending',
    reviewer_note TEXT DEFAULT '',
    resolved_at   TIMESTAMPTZ,
    reply_snapshot TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_review_status ON review_queue(status);
CREATE INDEX IF NOT EXISTS idx_review_request ON review_queue(request_id);
"""

# Ops ticket categories considered high-impact for client-service replies.
HIGH_IMPACT_CATEGORIES = {"fee_dispute", "failed_wire"}


def _reasons(state) -> list[str]:
    """Return the list of review triggers that fired for this request."""
    reasons: list[str] = []
    ans = state.get("answerability")
    if ans in (Answerability.PARTIAL, Answerability.CONFLICTING):
        reasons.append(f"answerability:{ans}")
    validation = state.get("validation", {})
    claims = validation.get("claims", [])
    if claims:
        supported = sum(1 for c in claims if c.get("supported"))
        if supported / len(claims) < get_settings().review_citation_threshold:
            reasons.append("low_claim_support")
    if validation.get("verdict") == "fail":
        reasons.append("validation_failed")
    if state.get("budget_breach"):
        reasons.append(f"budget_breach:{state['budget_breach'].get('limit')}")
    for tr in state.get("tool_results", []):
        if tr.get("error"):
            reasons.append(f"ops_error:{tr.get('tool')}")
        for row in tr.get("rows", []):
            if isinstance(row, dict) and row.get("category") in HIGH_IMPACT_CATEGORIES:
                reasons.append(f"high_impact:{row['category']}")
                break
    return sorted(set(reasons))


def needs_human_review(state) -> bool:
    """Return True when any review trigger fired and review is enabled."""
    return get_settings().human_review_enabled and bool(_reasons(state))


def ensure_table() -> None:
    """Create the review_queue table and indexes if they don't exist."""
    with rag_engine().begin() as conn:
        for stmt in _DDL.split(";"):
            if stmt.strip():
                conn.execute(text(stmt))


def enqueue(state, reply: str) -> list[str] | None:
    """Insert a pending review row; return trigger reasons or None on failure."""
    reasons = _reasons(state)
    if not reasons:
        return None
    try:
        ensure_table()
        with rag_engine().begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO review_queue (request_id, user_id, reason, reply_snapshot) "
                    "VALUES (:rid, :uid, :reason, :snap)"
                ),
                {
                    "rid": state.get("request_id", ""),
                    "uid": state["principal"].user_id,
                    "reason": ", ".join(reasons),
                    "snap": reply[:4000],
                },
            )
        log.info("review.enqueued", request_id=state.get("request_id"), reasons=reasons)
        return reasons
    except Exception as exc:
        log.warning("review.enqueue_failed", error=str(exc)[:200])
        return None  # no row exists — pending_review must not claim one does


def list_pending(limit: int = 50) -> list[dict]:
    """Return pending review rows newest-first for the admin endpoint."""
    with rag_engine().begin() as conn:
        rows = conn.execute(
            text(
                "SELECT id, request_id, user_id, queued_at, reason, status "
                "FROM review_queue WHERE status = 'pending' "
                "ORDER BY queued_at DESC LIMIT :lim"
            ),
            {"lim": limit},
        ).mappings()
    return [dict(r) for r in rows]


def resolve(review_id: int, status: str, note: str = "") -> dict[str, Any] | None:
    """Mark a review row approved/rejected/edited; return the row or None."""
    with rag_engine().begin() as conn:
        row = (
            conn.execute(
                text(
                    "UPDATE review_queue SET status = :status, reviewer_note = :note, "
                    "resolved_at = now() WHERE id = :id RETURNING id, status"
                ),
                {"id": review_id, "status": status, "note": note},
            )
            .mappings()
            .first()
        )
    return dict(row) if row else None


def count_pending() -> int:
    """Return the number of pending review items (health/status surfaces)."""
    try:
        ensure_table()
        with rag_engine().begin() as conn:
            return int(
                conn.execute(
                    text("SELECT count(*) FROM review_queue WHERE status = 'pending'")
                ).scalar_one()
            )
    except Exception:
        return -1


_REVIEW_STATUS = {"pending", "approved", "rejected", "edited"}


def valid_status(status: str) -> bool:
    """Return True when ``status`` is an allowed review resolution value."""
    return status in _REVIEW_STATUS - {"pending"}
