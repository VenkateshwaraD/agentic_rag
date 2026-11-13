"""Pydantic schemas for the ``POST /answer`` request and response bodies.

Author: venkateshwara doijode
"""

from typing import Literal

from pydantic import BaseModel, Field


class AnswerRequest(BaseModel):
    """Inbound copilot question plus an optional (still ACL-checked) client hint."""

    question: str = Field(min_length=1, max_length=2000)
    client_id: str | None = None  # optional hint; still ACL-checked


class Citation(BaseModel):
    """A source citation — resolves to a versioned, dated document passage."""

    doc_id: str
    section: str
    chunk_id: str = ""
    doc_version: str = ""
    effective_date: str = ""


class ToolCallInfo(BaseModel):
    """Summary of an ops tool invocation (tool, scope, rows, freshness)."""

    tool: str
    client_id: str | None = None
    row_count: int
    has_more: bool = False
    fresh_at: str | None = None
    source_updated_at: str | None = None
    snapshot_ts: str | None = None
    error: dict | None = None


class RetrievedChunk(BaseModel):
    """A retrieved evidence chunk with provenance and rerank score."""

    doc_id: str
    section: str
    text: str
    client_id: str = ""
    source: str = ""
    chunk_id: str = ""
    doc_version: str = ""
    effective_date: str = ""
    rerank_score: float | None = None


class BudgetInfo(BaseModel):
    """Execution-budget consumption reported back to the client."""

    iterations: int = 0
    tool_calls: int = 0
    llm_calls: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    breach: dict | None = None


AnswerabilityState = Literal[
    "fully_answerable",
    "partially_answerable",
    "insufficient_evidence",
    "conflicting_evidence",
    "unauthorized",
    "",
]


class AnswerResponse(BaseModel):
    """Full copilot response: reply, citations, evidence, tool calls, and ids.

    ``answerability`` distinguishes answer/refusal/abstain states;
    ``pending_review`` marks responses queued for asynchronous human review;
    ``idempotent_replay`` marks a response served from the idempotency cache.
    """

    reply: str
    citations: list[Citation]
    retrieved: list[RetrievedChunk]
    tool_calls: list[ToolCallInfo]
    refusal: bool
    abstained: bool = False
    answerability: AnswerabilityState = ""
    pending_review: bool = False
    unsupported_facts: list[str] = Field(default_factory=list)
    validation: dict = Field(default_factory=dict)
    budget: BudgetInfo = Field(default_factory=BudgetInfo)
    errors: list[str] = Field(default_factory=list)
    diagnostics: dict = Field(default_factory=dict)
    idempotent_replay: bool = False
    trace_id: str
    request_id: str
