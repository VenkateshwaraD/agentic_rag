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
_NUM_RE = re.compile(r"\d+(?:[.,:]\d+)*")


def _relevance_ok(r: dict) -> bool:
    """Relevance gate: rerank_score applies only when the reranker ran.

    Dense-only, BM25-only and hybrid (no rerank) results carry ``score`` /
    ``rrf_score`` but no ``rerank_score`` — they already survived top-k
    ranking, so absence of a rerank score must not mark them irrelevant.
    """
    if "rerank_score" in r:
        return r["rerank_score"] > _MIN_RERANK_SCORE
    return "score" in r or "rrf_score" in r


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

    # same section title in different docs asserting different numerics on
    # a shared "key" line — cheap proxy for contradictory facts. A claim
    # slot is the line text with numbers stripped, so "| Domestic wires |
    # 15:00 ET |" and "| Domestic wires | 16:00 ET |" map to one slot; when
    # the number sets in that slot differ across docs the sources conflict.
    # Version/effective-date boilerplate lines are skipped — a different
    # amendment date is not a disagreement about the governed fact.
    _SKIP_KEYS = ("version", "effective", "amend", "entered into")
    # unit/table syntax tokens stripped from claim keys — "| Domestic wires
    # | 17:00 ET |" and the same line at 16:00 must share the key
    # "domestic wires", not "domestic wires et" vs "domestic wires et"
    _UNIT_WORDS = frozenset(
        {
            "et",
            "ist",
            "utc",
            "gmt",
            "est",
            "edt",
            "pst",
            "pdt",
            "cst",
            "cdt",
            "bst",
            "min",
            "mins",
            "hr",
            "hrs",
            "hour",
            "hours",
            "per",
            "usd",
            "eur",
            "gbp",
            "inr",
        }
    )

    def _claim_key(line: str) -> str:
        """Claim-slot key: line text minus numbers, pipes, and unit words."""
        text = _NUM_RE.sub(" ", line).replace("|", " ").replace("§", " ").lower()
        return " ".join(w for w in text.split() if w not in _UNIT_WORDS)[:80]

    by_section: dict[str, list[dict]] = defaultdict(list)
    for r in retrieved:
        key = str(r.get("section", "")).strip().lower()
        if key:
            by_section[key].append(r)
    for section, rows in by_section.items():
        doc_ids = {r.get("doc_id") for r in rows}
        if len(doc_ids) < 2:
            continue
        per_doc: dict[str, dict[str, set[str]]] = {}
        for r in rows:
            claims = per_doc.setdefault(r["doc_id"], defaultdict(set))
            for line in r.get("text", "").splitlines():
                nums = _NUM_RE.findall(line)
                if not nums:
                    continue
                key = _claim_key(line)
                if not key or any(s in key for s in _SKIP_KEYS):
                    continue
                claims[key].update(nums)
        # claim slots populated by 2+ docs whose number sets disagree
        slots: dict[str, set[str]] = defaultdict(set)
        for doc, claims in per_doc.items():
            for key in claims:
                slots[key].add(doc)
        for key, docs in slots.items():
            if len(docs) < 2:
                continue
            sets = {d: per_doc[d][key] for d in docs}
            if len({frozenset(v) for v in sets.values()}) <= 1:
                continue
            ordered = sorted(docs)
            first = ordered[0]
            other = next(x for x in ordered[1:] if sets[x] != sets[first])
            conflicts.append(
                {
                    "kind": "numeric_divergence",
                    "section": section,
                    "doc_a": first,
                    "doc_b": other,
                    "field": key,
                    "values_a": sorted(sets[first])[:10],
                    "values_b": sorted(sets[other])[:10],
                }
            )
            break  # one divergent slot is enough to flag the section pair
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
    good_docs = [r for r in retrieved if _relevance_ok(r)]
    expected_tools = len(state.get("ops_calls", []))
    missing_tools = sum(1 for tr in tool_results if not tr.get("rows") and not tr.get("error"))

    with Timer() as t:
        # Conflicts matter among the evidence the question actually bears
        # on — the tail of a broad retrieval often contains divergent
        # boilerplate unrelated to the answer (e.g. a ticket question
        # pulling SLA escalation tables). Score the pool relative to the
        # best hit: chunks within half the top rerank score participate.
        pool = retrieved
        scores = [r["rerank_score"] for r in retrieved if r.get("rerank_score") is not None]
        if scores:
            bar = max(scores) * 0.5
            pool = [
                r for r in retrieved if r.get("rerank_score") is None or r["rerank_score"] >= bar
            ]
        conflicts = _detect_conflicts(pool)
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
