"""POST /answer — the copilot endpoint.

Author: venkateshwara doijode
"""

import uuid

from fastapi import APIRouter, Depends, Request

from app.agent.graph import answer
from app.api.deps import get_principal
from app.api.schemas.answer import AnswerRequest, AnswerResponse
from app.security.authn import Principal

router = APIRouter()


@router.post("/answer", response_model=AnswerResponse)
def post_answer(
    body: AnswerRequest,
    request: Request,
    principal: Principal = Depends(get_principal),  # noqa: B008 — FastAPI idiom
):
    """Run the agentic RAG graph for an authenticated user's question.

    Reuses the middleware-assigned request id (or generates one) and returns
    the graph's reply, citations, retrieved chunks, tool calls, and refusal
    flag as an ``AnswerResponse``.
    """
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:16])
    return answer(
        question=body.question,
        principal=principal,
        request_id=request_id,
    )
