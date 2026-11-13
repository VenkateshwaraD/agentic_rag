"""Grade node — classifies answerability and retrieval failure type.

Produces the five-way ``Answerability`` decision (full / partial /
insufficient / conflicting / unauthorized) plus a ``FailureType`` when
evidence is inadequate — the rewrite node maps each failure type to a
different query-rewrite strategy.

Conflict detection is heuristic: two chunks sharing ``(doc_id, section)``
with different ``doc_version`` values, or the same section title across
different doc ids whose texts disagree on shared numeric values, count as
contradicting documents.

Author: venkateshwara doijode
"""

import re
from collections import defaultdict

from app.agent.nodes.common import Timer, audit
from app.core.config import get_settings
from app.core.constants import EV_AGENT_STEP, Answerability, FailureType

_MIN_RERANK_SCORE = -2.0  # BGE logits; negative ~0 is a reasonable relevance bar
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")


def _detect_conflicts(retrieved: list[dict]) -> list[dict]:
    """Return contradiction pairs found among retrieved chunks.

    Two heuristics: (1) same doc+section with different versions indexed
    (``doc_version_policy=keep_all``); (2) same normalized section title in
    different docs asserting different numeric values for shared keys.
    """
    conflicts: list[dict] = []
    by_doc_section: dict[tuple, list[dict]] = defaultdict(list)
    for r in retrieved:
        by_doc_section[(r.get("doc_id"), r.get("section"))].append(r)
    for (doc_id, section), rows in by_doc_section.items():
        versions = {r.get("doc_version") for r in rows if r.get("doc_version")}
        if len(versions) > 1:
            a, b = rows[0], rows[-1]
            conflicts.append(
                {
                    "kind": "version_divergence",
                    "doc_id": doc_id,
                    "section": section,
                    "versions": sorted(versions),
                    "text_a": a.get("text", "")[:200],
                    "text_b": b.get("text", "")[:200],
                }
            )

    # same section title in different docs with divergent numerics on
    # shared "key" phrasing — cheap proxy for contradictory facts
    by_section: dict[str, list[dict]] = defaultdict(list)
    for r in retrieved:
        key = str(r.get("section", "")).strip().lower()
        if key:
            by_section[key].append(r)
    for section, rows in by_section.items():
        doc_ids = {r.get("doc_id") for r in rows}
        if len(doc_ids) < 2:
            continue
        nums: dict[str, set[str]] = {}
        for r in rows:
            nums.setdefault(r["doc_id"], set()).update(_NUM_RE.findall(r.get("text", "")))
        distinct = set().union(*nums.values()) if nums else set()
        if len(distinct) > 1 and len(doc_ids) == len(nums):
            a, b = rows[0], rows[-1]
            if not (nums[a["doc_id"]] & nums[b["doc_id"]]):
                conflicts.append(
                    {
                        "kind": "numeric_divergence",
                        "section": section,
                        "doc_a": a["doc_id"],
                        "doc_b": b["doc_id"],
                        "values_a": sorted(nums[a["doc_id"]])[:10],
                        "values_b": sorted(nums[b["doc_id"]])[:10],
                    }
                )
    return conflicts


def grade(state):
    """Classify whether the retrieved evidence can support an answer.

    Sets ``answerability`` (five-way) and ``failure_type`` (when evidence
    is inadequate), then picks the graph action: ``answer`` → synthesize,
    ``rewrite`` → retry with a failure-specific rewrite, or ``refuse`` →
    abstain/refuse when retries are exhausted or the request is
    unauthorized. Audits the classification and returns state updates.
    """
    retrieved = state.get("retrieved", [])
    tool_results = state.get("tool_results", [])
    settings = get_settings()
    retries = state.get("retries", 0)

    has_rows = any(tr.get("rows") for tr in tool_results)
    tool_errors = [tr for tr in tool_results if tr.get("error")]
    store_errors = [c for c, d in state.get("diagnostics", {}).items() if d.get("error")]
    good_docs = [r for r in retrieved if r.get("rerank_score", -99) > _MIN_RERANK_SCORE]
    expected_tools = len(state.get("ops_calls", []))
    missing_tools = sum(1 for tr in tool_results if not tr.get("rows") and not tr.get("error"))

    with Timer() as t:
        conflicts = _detect_conflicts(retrieved)
        updates: dict = {"conflicts": conflicts}

        if state.get("refusal"):
            action = "refuse"
            answerability = Answerability.UNAUTHORIZED
        elif conflicts:
            action = "answer"  # synthesize with conflict caveat
            answerability = Answerability.CONFLICTING
            updates["failure_type"] = FailureType.CONFLICTING_DOCS
        elif not retrieved and not has_rows:
            # nothing at all — infrastructure error or genuinely no evidence;
            # both retry as MISSING_EVIDENCE (outage may be transient)
            answerability = Answerability.INSUFFICIENT
            updates["failure_type"] = FailureType.MISSING_EVIDENCE
            action = "rewrite" if retries < settings.agent_max_retries else "refuse"
        elif not good_docs and retrieved:
            answerability = Answerability.INSUFFICIENT
            updates["failure_type"] = FailureType.LOW_RELEVANCE
            action = "rewrite" if retries < settings.agent_max_retries else "refuse"
        elif (tool_errors or store_errors) or (expected_tools and missing_tools):
            # evidence exists but coverage is incomplete → partial answer
            answerability = Answerability.PARTIAL
            updates["failure_type"] = FailureType.INCOMPLETE_FACTS
            updates["unsupported_facts"] = [
                f"{tr['tool']} lookup returned no data" for tr in tool_results if not tr.get("rows")
            ] + [f"{c} collection unavailable" for c in store_errors]
            action = "answer"
        else:
            answerability = Answerability.FULL
            action = "answer"

        if action == "refuse" and answerability == Answerability.INSUFFICIENT:
            updates["abstained"] = True
            updates["abstention_reason"] = "insufficient evidence in entitled sources"
        elif action == "refuse":
            updates["refusal"] = True
            updates["refusal_reason"] = state.get(
                "refusal_reason", "insufficient evidence in entitled sources"
            )
        updates["grade_action"] = action
        updates["answerability"] = answerability
    audit(
        state,
        EV_AGENT_STEP,
        {
            "node": "grade",
            "answerability": str(updates["answerability"]),
            "failure_type": updates.get("failure_type"),
            "good_docs": len(good_docs),
            "tool_rows": sum(len(tr.get("rows", [])) for tr in tool_results),
            "conflicts": len(conflicts),
            "retries": retries,
            "action": action,
        },
        t.ms,
    )
    return updates
