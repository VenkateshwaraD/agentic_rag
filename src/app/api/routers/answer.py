"""Answer endpoints — ``POST /answer`` (legacy) + ``/api/v1/answer`` and SSE.

Author: venkateshwara doijode
"""

import json
import uuid

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import StreamingResponse

from app.agent.graph import answer, stream_answer
from app.api import idempotency
from app.api.deps import get_principal
from app.api.schemas.answer import AnswerRequest, AnswerResponse
from app.security.authn import Principal

router = APIRouter()

_MAX_TIMEOUT_MS = 120_000


def _run_answer(
    body: AnswerRequest,
    request: Request,
    principal: Principal,
    idempotency_key: str | None,
    timeout_ms: int | None,
) -> dict:
    """Shared handler for the versioned and legacy answer endpoints.

    Applies the idempotency cache (replay → cached response, conflicting
    body → 409), propagates the middleware request id into agent state, and
    forwards the optional per-request timeout as a budget override.
    """
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:16])
    if idempotency_key:
        cached = idempotency.lookup(
            principal.user_id, idempotency_key, body.question, body.client_id
        )
        if cached is not None:
            return {**cached, "idempotent_replay": True}
    result = answer(
        question=body.question,
        principal=principal,
        request_id=request_id,
        max_elapsed_ms=timeout_ms,
    )
    if idempotency_key:
        idempotency.store(principal.user_id, idempotency_key, body.question, body.client_id, result)
    return result


@router.post("/answer", response_model=AnswerResponse)
def post_answer(
    body: AnswerRequest,
    request: Request,
    principal: Principal = Depends(get_principal),  # noqa: B008 — FastAPI idiom
    idempotency_key: str | None = Header(default=None, max_length=128),
    x_timeout_ms: int | None = Header(default=None, ge=1, le=_MAX_TIMEOUT_MS),
):
    """Run the agentic RAG graph for an authenticated user's question.

    ``Idempotency-Key`` (optional) makes retries safe: a replayed key with
    the same body returns the first response unchanged. ``X-Timeout-Ms``
    tightens the per-request elapsed-time budget.
    """
    return _run_answer(body, request, principal, idempotency_key, x_timeout_ms)


@router.post("/answer/stream")
def post_answer_stream(
    body: AnswerRequest,
    request: Request,
    principal: Principal = Depends(get_principal),  # noqa: B008 — FastAPI idiom
    x_timeout_ms: int | None = Header(default=None, ge=1, le=_MAX_TIMEOUT_MS),
):
    """Server-sent-events stream of graph node completions + final result.

    Emits ``event: node_update`` per node and one ``event: final`` carrying
    the same normalized payload as ``POST /answer``. Client disconnects are
    checked between events (cooperative cancellation).
    """
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:16])

    def events():
        """Yield SSE-formatted node/final events until the graph finishes."""
        for ev in stream_answer(body.question, principal, request_id, max_elapsed_ms=x_timeout_ms):
            name = "final" if ev["type"] == "final" else "node_update"
            yield f"event: {name}\ndata: {json.dumps(ev, default=str)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Request-Id": request_id},
    )
