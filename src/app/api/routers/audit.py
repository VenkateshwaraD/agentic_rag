"""GET /audit/{request_id} — audit trail lookup (admin only).

Author: venkateshwara doijode
"""

from fastapi import APIRouter, Depends

from app.api.deps import get_admin
from app.core.dependencies import get_audit
from app.core.errors import NotFoundError

router = APIRouter(dependencies=[Depends(get_admin)])


@router.get("/audit/{request_id}")
def get_audit_events(request_id: str):
    """Return all audit events for a request id, or 404 when none exist."""
    events = get_audit().events_for_request(request_id)
    if not events:
        raise NotFoundError(f"no audit events for request {request_id}")
    return {"request_id": request_id, "events": events}
