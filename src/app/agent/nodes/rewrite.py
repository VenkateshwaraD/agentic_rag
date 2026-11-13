"""Rewrite node — transforms the question for a bounded retrieval retry.

Each ``FailureType`` maps to a different strategy: broaden for missing
evidence, rephrase for low relevance, expand across versions for
conflicting documents, and focus on the client/record ids for incomplete
facts. An LLM rewrite is used when a key is configured; otherwise a
deterministic heuristic keeps the pipeline runnable offline.

Author: venkateshwara doijode
"""

from app.agent import budget
from app.agent.llm import get_llm
from app.agent.nodes.common import Timer, add_error, audit
from app.core.config import get_settings
from app.core.constants import EV_AGENT_STEP, FailureType

# failure_type → heuristic suffix appended to the question when no LLM is
# available (and as context for the LLM rewrite when it is)
_HEURISTIC = {
    FailureType.MISSING_EVIDENCE: (
        " related policies, procedures, fees, cutoffs and general terms"
    ),
    FailureType.LOW_RELEVANCE: " key terms and synonyms of the original question",
    FailureType.CONFLICTING_DOCS: (" across all document versions and effective dates"),
    FailureType.INCOMPLETE_FACTS: " client-specific details and record identifiers",
}


def _failure_type(raw) -> FailureType:
    """Coerce a state's failure_type value (str or enum) into ``FailureType``."""
    try:
        return FailureType(raw)
    except ValueError:
        return FailureType.MISSING_EVIDENCE


def _heuristic_rewrite(question: str, failure_type: FailureType, client_id: str | None) -> str:
    """Append strategy context + client id to the question (offline path)."""
    return question + (f" client {client_id}" if client_id else "") + _HEURISTIC[failure_type]


def rewrite(state):
    """Produce a rewritten question for the next retrieve pass.

    Increments ``retries`` (bounded by ``agent_max_retries`` in grade),
    applies the failure-type strategy — LLM-assisted when configured,
    heuristic otherwise — audits the transformation, and returns the new
    question plus bookkeeping updates.
    """
    failure_type = _failure_type(state.get("failure_type"))
    client_id = state.get("extra", {}).get("client_id")
    original = state["question"]
    updates: dict = {"retries": state.get("retries", 0) + 1, "rewritten": True}

    with Timer() as t:
        if get_settings().anthropic_api_key:
            try:
                prompt = (
                    "Rewrite this client-service question for a document "
                    "search. The previous retrieval failed with "
                    f"'{failure_type}'. Apply this strategy: "
                    f"{_HEURISTIC.get(failure_type, 'improve recall')}. "
                    f"Client: {client_id or 'unspecified'}.\n"
                    f"Original: {original}\nRewritten (one line):"
                )
                response = get_llm().invoke(prompt)
                updates.update(budget.record_llm_usage(state, response))
                candidate = response.content.strip().split("\n")[0].strip()
                new_question = candidate or _heuristic_rewrite(original, failure_type, client_id)
            except Exception as exc:
                updates["errors"] = add_error(state, f"rewrite.llm_failed: {exc}")
                new_question = _heuristic_rewrite(original, failure_type, client_id)
        else:
            new_question = _heuristic_rewrite(original, failure_type, client_id)
    audit(
        state,
        EV_AGENT_STEP,
        {
            "node": "rewrite",
            "failure_type": str(failure_type),
            "from": original,
            "to": new_question,
            "retry": updates["retries"],
        },
        t.ms,
    )
    return {"question": new_question, **updates}
