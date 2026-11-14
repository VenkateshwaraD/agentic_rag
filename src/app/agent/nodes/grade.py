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

from app.agent.nodes.common import QUESTION_STOPWORDS, Timer, audit
from app.core.config import get_settings
from app.core.constants import EV_AGENT_STEP, TITLE_BOILER, Answerability, FailureType

_NUM_RE = re.compile(r"\d+(?:[.,:]\d+)*")


def _relevance_ok(r: dict, min_score: float | None = None) -> bool:
    """Relevance gate: rerank_score applies only when the reranker ran.

    Dense-only, BM25-only and hybrid (no rerank) results carry ``score`` /
    ``rrf_score`` but no ``rerank_score`` — they already survived top-k
    ranking, so absence of a rerank score must not mark them irrelevant.
    The bar is a settings knob (``grade_min_rerank_score``): reranker
    scores are model-dependent, not a universal constant.
    """
    if "rerank_score" in r:
        if min_score is None:
            min_score = get_settings().grade_min_rerank_score
        return r["rerank_score"] > min_score
    return "score" in r or "rrf_score" in r


def _question_anchors(question: str) -> set[str]:
    """Distinctive content terms the question is *about* — what evidence
    must mention for an answer to actually be on-topic.

    Frame words are excluded: "do X and Y agree on the cutoff" asks *about*
    cutoffs, not about "agree"; "the SLA" names a genre, not a topic. Only
    4+-letter words count — "fee"/"tax" are too generic to anchor on.
    """
    return {w for w in re.findall(r"[a-z]{4,}", question.lower()) if w not in QUESTION_STOPWORDS}


def _uncovered_anchors(question: str, retrieved: list[dict], tool_surface: str = "") -> set[str]:
    """Anchors absent from every retrieved chunk's surface (text + doc
    id/title/client id) *and* the ops-tool surface (tool names + returned
    rows). 'Crypto custody fee for bluepeak' leaves 'crypto' uncovered
    when no entitled source mentions crypto — the honest verdict is
    missing evidence, not a guessed fee. Tool-backed aspects ('tickets',
    a status value inside the rows) count as covered by the tool."""
    anchors = _question_anchors(question)
    if not anchors:
        return set()
    hay = " ".join(
        " ".join(str(r.get(k) or "") for k in ("text", "doc_title", "doc_id", "client_id"))
        for r in retrieved
    ).lower()
    hay += " " + tool_surface.lower()

    def _covered(a: str) -> bool:
        return bool(re.search(rf"\b{re.escape(a)}s?\b", hay))

    uncovered = {a for a in anchors if not _covered(a)}
    # an "A or B" anchor pair is disjunctive — 'failed or returned trades'
    # is satisfied by evidence for either status; the alternative only
    # counts as missing when *every* disjunct is absent
    for m in re.finditer(r"\b([a-z]{4,})\s+or\s+([a-z]{4,})\b", question.lower()):
        a, b = m.group(1), m.group(2)
        if a in anchors and b in anchors and (_covered(a) or _covered(b)):
            uncovered -= {a, b}
    return uncovered


def _title_named_docs(question: str, rows: list[dict]) -> set[str]:
    """Doc ids the question names via the document's *title* words.

    Distinct from client-id naming: 'Under the Statements & Reporting SLA'
    scopes the ask to one document (other docs' disagreements are beside
    the point), while 'the fee for bluepeak' names a *scope* — several
    client docs can still contradict each other within it.
    """
    question_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
    named: set[str] = set()
    for r in rows:
        doc = str(r.get("doc_id") or "")
        if not doc:
            continue
        toks = set(re.findall(r"[a-z]+", str(r.get("doc_title", "")).lower())) - TITLE_BOILER
        if toks and toks <= question_tokens:
            named.add(doc)
    return named


def _named_docs(question: str, rows: list[dict]) -> set[str]:
    """Doc ids the question names — by client id or by the distinctive
    words of the document's title.

    'Do the Payments Operations SLA and the Wire Transfer SLA agree…'
    names SLA-001 + SLA-006 via their titles; 'wire fee for bluepeak'
    names AGR-002 via its client id. A divergent claim slot should lead
    with the docs the user asked about, not whichever pair sorts first.
    """
    question_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
    named = _title_named_docs(question, rows)
    for r in rows:
        doc = str(r.get("doc_id") or "")
        if doc and r.get("client_id") and r["client_id"] in question_tokens:
            named.add(doc)
    return named


def _stem(word: str) -> str:
    """Naive singular for slot/anchor matching — 'cutoffs' ≈ 'cutoff',
    'wires' ≈ 'wire'. Consistent on both sides, so 'status' → 'statu'
    still matches 'statuses' → 'statu'."""
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def _detect_conflicts(
    retrieved: list[dict], question: str = "", anchors: set[str] | None = None
) -> list[dict]:
    """Return contradiction groups found among retrieved chunks.

    Heuristics: (1) same doc+section with different versions indexed
    (``doc_version_policy=keep_all``); (2) same normalized section title in
    different docs asserting different numeric values for shared keys —
    but only within one client scope, and only when the divergent slot is
    *on-topic*: a trades question that incidentally pulled SLA cutoff
    chunks must not "detect" a conflict it never asked about. Four
    clients' agreements quoting four different fees is ambiguity ("which
    client?"), not contradictory evidence; bank-wide docs (``client_id``
    empty) share a scope, so SLAs disagreeing on a cutoff IS a real
    conflict.
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

    named = _named_docs(question, retrieved) if question else set()
    by_section: dict[str, list[dict]] = defaultdict(list)
    for r in retrieved:
        key = str(r.get("section", "")).strip().lower()
        if key:
            by_section[key].append(r)
    for section, rows in by_section.items():
        # the conflict record carries the canonical section label, not the
        # normalized grouping key — citations emitted downstream must
        # match the indexed section's real casing
        section_label = str(rows[0].get("section") or section)
        doc_ids = {r.get("doc_id") for r in rows}
        if len(doc_ids) < 2:
            continue
        per_doc: dict[str, dict[str, set[str]]] = {}
        client_of: dict[str, str] = {}
        for r in rows:
            doc = str(r.get("doc_id") or "")
            client_of.setdefault(doc, r.get("client_id") or "")
            claims = per_doc.setdefault(doc, defaultdict(set))
            title_line = str(r.get("doc_title") or "")
            for line in r.get("text", "").splitlines():
                if title_line and line.strip() == title_line:
                    continue  # the doc-name line is metadata, never a claim

                nums = _NUM_RE.findall(line)
                if not nums:
                    continue
                key = _claim_key(line)
                if not key or any(s in key for s in _SKIP_KEYS):
                    continue
                claims[key].update(nums)
        # claim slots populated by 2+ docs in the same client scope whose
        # number sets disagree
        slots: dict[str, set[str]] = defaultdict(set)
        for doc, claims in per_doc.items():
            for key in claims:
                slots[key].add(doc)
        for key, docs in slots.items():
            if len(docs) < 2:
                continue
            groups: dict[str, list[str]] = defaultdict(list)
            for d in sorted(docs):
                groups[client_of.get(d, "")].append(d)
            divergent = [
                g
                for g in groups.values()
                if len(g) >= 2 and len({frozenset(per_doc[d][key]) for d in g}) > 1
            ]
            if not divergent:
                continue
            group = divergent[0]
            # topicality gate: when the question has distinctive anchors
            # and none of them appear in the divergent slot's key or its
            # section title, the disagreement is retrieval noise, not an
            # answer to the ask — unless a named doc is in the group, in
            # which case the divergence is on-topic by construction.
            if anchors and not (set(group) & named):
                slot_terms = {
                    _stem(w) for w in re.findall(r"[a-z]+", (key + " " + section).lower())
                }
                if not slot_terms & {_stem(a) for a in anchors}:
                    continue
            ordered = sorted(group, key=lambda d: (d not in named, d))
            doc_values = [{"doc_id": d, "values": sorted(per_doc[d][key])[:10]} for d in ordered]
            n_named = sum(1 for d in ordered if d in named)
            conflicts.append(
                {
                    "kind": "numeric_divergence",
                    "section": section_label,
                    "doc_a": ordered[0],
                    "doc_b": ordered[1],
                    "field": key,
                    "values_a": doc_values[0]["values"],
                    "values_b": doc_values[1]["values"],
                    "doc_values": doc_values,
                    "lead_docs": ordered[: max(2, min(n_named, 4))],
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
    # a tool call that returns zero rows without error *is* the answer
    # ("no open tickets") — only errored lookups count as missing data
    missing_tools = sum(1 for tr in tool_results if tr.get("error"))
    # tool names + row payload — an aspect the tools answered counts as
    # covered evidence for the anchor gate and can name a doc the
    # question itself left generic ("that team's SLA")
    import json as _json

    tool_surface = " ".join(
        str(tr.get("tool", "")) + " " + _json.dumps(tr.get("rows") or [], default=str)
        for tr in tool_results
    )

    with Timer() as t:
        # Conflicts matter among the evidence the question actually bears
        # on — the tail of a broad retrieval often contains divergent
        # boilerplate unrelated to the answer (e.g. a ticket question
        # pulling SLA escalation tables). The pool is scored relative to
        # the best hit — half its distance from zero, so a *negative* top
        # logit widens the margin instead of emptying the pool (the old
        # max*0.5 bar exceeded every score). When nothing clears the
        # relevance bar at all, every chunk is equally weak evidence and
        # divergence across all of it is the ambiguity/conflict signal.
        pool = retrieved
        scores = [r["rerank_score"] for r in retrieved if r.get("rerank_score") is not None]
        if scores and good_docs:
            bar = max(scores) - abs(max(scores)) * 0.5
            pool = [
                r for r in retrieved if r.get("rerank_score") is None or r["rerank_score"] >= bar
            ]
        # a question scoped to a single named document ("Under the
        # Statements & Reporting SLA…") can't be contradicted by *other*
        # docs — only intra-doc version divergence still counts. Naming a
        # client instead of a title is a scope, not a doc, so cross-doc
        # conflicts within that scope still apply. A doc named by the
        # *tool results* ("that team's SLA" resolved via accounts) scopes
        # the same way.
        question = state.get("question", "")
        title_named = _title_named_docs(question, pool)
        if not title_named and tool_surface:
            title_named = _title_named_docs(tool_surface, pool)
        if len(title_named) == 1:
            pool = [r for r in pool if str(r.get("doc_id") or "") in title_named]
        # the question's subject terms vs. the evidence surface — a topic
        # absent from every chunk can't be answered no matter how politely
        # the reranker scored the near-misses. A strongly relevant top hit
        # overrides: confident retrieval with vocabulary mismatch is a
        # synonym gap, not missing evidence — unless the same anchor was
        # already missing on an earlier pass (the rewrite didn't fix it,
        # so a high score is a confident near-miss)
        anchors = _question_anchors(question)
        uncovered = _uncovered_anchors(question, retrieved, tool_surface)
        confident = bool(scores) and max(scores) >= settings.grade_rerank_confident
        conflicts = _detect_conflicts(pool, question, anchors)
        updates: dict = {"conflicts": conflicts}

        if state.get("refusal"):
            action = "refuse"
            # a hard budget breach is an infra failure, not an auth verdict
            answerability = (
                Answerability.INSUFFICIENT
                if state.get("budget_breach")
                else Answerability.UNAUTHORIZED
            )
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
        elif not anchors and retrieved and not has_rows:
            # no distinctive topic term at all ("What is the fee?") — a
            # rewrite can't sharpen a question that was never specific, so
            # abstain immediately and let the out-guardrail ask for the
            # missing client/topic rather than draft a guess
            answerability = Answerability.INSUFFICIENT
            updates["failure_type"] = FailureType.MISSING_EVIDENCE
            updates["abstention_reason"] = "question lacks a specific topic or client"
            action = "refuse"
        elif uncovered and (not confident or retries > 0):
            # the asked-about topic appears in no retrieved chunk (e.g.
            # 'crypto' when no source covers crypto) — a rewrite may fix
            # vocabulary mismatch, but a confident score on a *retry* is
            # a confirmed near-miss, so the override applies on the first
            # pass only; after retries this must abstain, never draft a
            # guess from tangentially related docs. When tools answered
            # another aspect, the honest verdict is a partial answer
            # noting what's missing, not an abstention.
            updates["missing_anchors"] = sorted(uncovered)
            if has_rows:
                answerability = Answerability.PARTIAL
                updates["failure_type"] = FailureType.INCOMPLETE_FACTS
                updates["unsupported_facts"] = [
                    f"no entitled evidence for: {a}" for a in sorted(uncovered)
                ]
                action = "answer"
            else:
                answerability = Answerability.INSUFFICIENT
                updates["failure_type"] = FailureType.MISSING_EVIDENCE
                action = "rewrite" if retries < settings.agent_max_retries else "refuse"
        elif not good_docs and retrieved and not has_rows:
            # weak doc scores can't sink a tool-backed answer — the ops
            # rows are the evidence, same exemption as the anchor gate
            answerability = Answerability.INSUFFICIENT
            updates["failure_type"] = FailureType.LOW_RELEVANCE
            action = "rewrite" if retries < settings.agent_max_retries else "refuse"
        elif (tool_errors or store_errors) or (expected_tools and missing_tools):
            # evidence exists but coverage is incomplete → partial answer
            answerability = Answerability.PARTIAL
            updates["failure_type"] = FailureType.INCOMPLETE_FACTS
            updates["unsupported_facts"] = [
                f"{tr['tool']} lookup failed: {tr['error']}"
                for tr in tool_results
                if tr.get("error")
            ] + [f"{c} collection unavailable" for c in store_errors]
            action = "answer"
        else:
            answerability = Answerability.FULL
            action = "answer"

        if action == "refuse" and answerability == Answerability.INSUFFICIENT:
            updates["abstained"] = True
            updates.setdefault("abstention_reason", "insufficient evidence in entitled sources")
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
            "uncovered_anchors": sorted(uncovered),
            "tool_rows": sum(len(tr.get("rows", [])) for tr in tool_results),
            "conflicts": len(conflicts),
            "retries": retries,
            "action": action,
        },
        t.ms,
    )
    return updates
