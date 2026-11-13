"""Pydantic schemas for the ``POST /answer`` request and response bodies.

Author: venkateshwara doijode
"""

from pydantic import BaseModel, Field


class AnswerRequest(BaseModel):
    """Inbound copilot question plus an optional (still ACL-checked) client hint."""

    question: str = Field(min_length=1, max_length=2000)
    client_id: str | None = None  # optional hint; still ACL-checked


class Citation(BaseModel):
    """A source citation: document id and the section within it."""

    doc_id: str
    section: str


class ToolCallInfo(BaseModel):
    """Summary of an ops tool invocation (tool name, client scope, rows returned)."""

    tool: str
    client_id: str | None = None
    row_count: int


class RetrievedChunk(BaseModel):
    """A retrieved evidence chunk with provenance and rerank score."""

    doc_id: str
    section: str
    text: str
    client_id: str = ""
    source: str = ""
    rerank_score: float | None = None


class AnswerResponse(BaseModel):
    """Full copilot response: reply, citations, evidence, tool calls, and ids."""

    reply: str
    citations: list[Citation]
    retrieved: list[RetrievedChunk]
    tool_calls: list[ToolCallInfo]
    refusal: bool
    diagnostics: dict = Field(default_factory=dict)
    trace_id: str
    request_id: str
