from pydantic import BaseModel


class ReindexResponse(BaseModel):
    collections: dict[str, int]


class AuditEventOut(BaseModel):
    request_id: str
    trace_id: str
    user_id: str
    ts: str
    event_type: str
    payload: dict
    latency_ms: float
