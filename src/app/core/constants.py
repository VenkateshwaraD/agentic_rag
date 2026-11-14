"""Shared constants: collection names, event types, roles, agent enums.

Author: venkateshwara doijode
"""

from enum import StrEnum

# Document collections (mirrored in pgvector + OpenSearch)
COLLECTION_AGREEMENTS = "agreements"
COLLECTION_SLAS = "slas"
COLLECTION_SERVICE_TEAMS = "service_teams"
COLLECTIONS = (COLLECTION_AGREEMENTS, COLLECTION_SLAS, COLLECTION_SERVICE_TEAMS)

# Roles
ROLE_AGENT = "agent"
ROLE_SENIOR_AGENT = "senior_agent"
ROLE_ADMIN = "admin"
ALL_ROLES = (ROLE_AGENT, ROLE_SENIOR_AGENT, ROLE_ADMIN)

# Audit event types
EV_QUERY = "query"
EV_RETRIEVAL = "retrieval"
EV_TOOL_CALL = "tool_call"
EV_GUARDRAIL = "guardrail"
EV_AGENT_STEP = "agent_step"
EV_ANSWER = "answer"
EV_REFUSAL = "refusal"
EV_ABSTAIN = "abstain"

# Citation marker format used in prompts and parsed by the eval harness
CITATION_PATTERN = r"\[([A-Z]{2,4}-\d{3})\s*§([^\]]+)\]"

# words common to most document titles — they carry no identity, so a
# doc is "named" in a question only when its *distinctive* title words
# all appear (e.g. {payments, operations} for "Payments Operations SLA")
TITLE_BOILER = frozenset(
    {
        "the",
        "service",
        "level",
        "agreement",
        "agreements",
        "sla",
        "slas",
        "client",
        "clients",
        "services",
        "bank",
        "team",
        "teams",
        "policy",
        "faq",
    }
)

# User-facing refusal message (single source for synthesize + guardrail_out)
REFUSAL_MESSAGE = (
    "I don't have enough entitled information to answer that question. "
    "Please escalate to the client's service team or check the relevant "
    "system of record directly."
)

# User-facing abstention message — distinct from REFUSAL_MESSAGE: abstention
# means evidence was insufficient rather than the request being blocked.
ABSTENTION_MESSAGE = (
    "I don't have enough evidence in the client's entitled documents and "
    "systems to answer that reliably, so I'm not going to guess. "
    "Please escalate to the client's service team or check the system of "
    "record directly."
)


class Answerability(StrEnum):
    """Answerability classification produced by the grade node."""

    FULL = "fully_answerable"
    PARTIAL = "partially_answerable"
    INSUFFICIENT = "insufficient_evidence"
    CONFLICTING = "conflicting_evidence"
    UNAUTHORIZED = "unauthorized"


class FailureType(StrEnum):
    """Why retrieval evidence was judged inadequate — drives rewrite strategy."""

    MISSING_EVIDENCE = "missing_evidence"
    LOW_RELEVANCE = "low_relevance"
    CONFLICTING_DOCS = "conflicting_documents"
    INCOMPLETE_FACTS = "incomplete_facts"


# Estimated USD cost per 1M tokens (input, output) by model id; "default"
# is used for unlisted models so cost budgeting still works.
LLM_COST_RATES: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-opus-4-20250514": (15.0, 75.0),
    "claude-haiku-3-5-20241022": (0.8, 4.0),
    "default": (3.0, 15.0),
}
