"""Shared constants: collection names, event types, roles."""

# Document collections (mirrored in pgvector + OpenSearch)
COLLECTION_AGREEMENTS = "agreements"
COLLECTION_SLAS = "slas"
COLLECTION_SERVICE_TEAMS = "service_teams"
COLLECTIONS = (COLLECTION_AGREEMENTS, COLLECTION_SLAS, COLLECTION_SERVICE_TEAMS)

# Roles
ROLE_AGENT = "agent"
ROLE_SENIOR_AGENT = "senior_agent"
ROLE_ADMIN = "admin"

# Audit event types
EV_QUERY = "query"
EV_RETRIEVAL = "retrieval"
EV_TOOL_CALL = "tool_call"
EV_GUARDRAIL = "guardrail"
EV_AGENT_STEP = "agent_step"
EV_ANSWER = "answer"
EV_REFUSAL = "refusal"

# Citation marker format used in prompts and parsed by the eval harness
CITATION_PATTERN = r"\[([A-Z]{2,4}-\d{3})\s*§([^\]]+)\]"
