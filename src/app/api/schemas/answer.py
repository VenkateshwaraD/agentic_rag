from pydantic import BaseModel, Field


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    client_id: str | None = None  # optional hint; still ACL-checked


class Citation(BaseModel):
    doc_id: str
    section: str


class ToolCallInfo(BaseModel):
    tool: str
    client_id: str | None = None
    row_count: int


class RetrievedChunk(BaseModel):
    doc_id: str
    section: str
    text: str
    client_id: str = ""
    source: str = ""
    rerank_score: float | None = None


class AnswerResponse(BaseModel):
    reply: str
    citations: list[Citation]
    retrieved: list[RetrievedChunk]
    tool_calls: list[ToolCallInfo]
    refusal: bool
    trace_id: str
    request_id: str
