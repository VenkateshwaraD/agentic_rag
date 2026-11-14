"""Route node — LLM classifies which sources/tools the inquiry needs,
with a keyword heuristic fallback when no LLM key is configured.

Author: venkateshwara doijode
"""

import json
import re

from app.agent import budget
from app.agent.llm import get_llm
from app.agent.nodes.common import Timer, add_error, audit, load_prompt
from app.core.config import get_settings
from app.core.constants import (
    COLLECTION_AGREEMENTS,
    COLLECTION_SERVICE_TEAMS,
    COLLECTION_SLAS,
    EV_AGENT_STEP,
    Answerability,
)

ALL_SOURCES = [COLLECTION_AGREEMENTS, COLLECTION_SLAS, COLLECTION_SERVICE_TEAMS]
ALL_OPS = ["accounts", "trades", "tickets"]

_KEYWORDS = {
    COLLECTION_AGREEMENTS: (
        "fee",
        "fees",
        "charge",
        "statement",
        "billing",
        "agreement",
        "pricing",
    ),
    COLLECTION_SLAS: (
        "cutoff",
        "cut-off",
        "deadline",
        "sla",
        "sev",
        "incident",
        "escalat",
        "same-day",
        "processing time",
    ),
    COLLECTION_SERVICE_TEAMS: (
        "team",
        "contact",
        "who owns",
        "escalation desk",
        "coverage",
        "hours",
        "owner",
    ),
}
_OPS_KEYWORDS = {
    "trades": (
        "trades",
        "trade status",
        "wire status",
        "payment status",
        "transfer status",
        "settled",
        "failed wire",
        "transaction",
        "transactions",
    ),
    "tickets": (
        "ticket",
        "tickets",
        "case history",
        "previous issue",
        "complaint",
        "complaints",
        "tkt-",
    ),
    "accounts": ("account", "accounts", "tier", "region", "agreement on file", "account details"),
}
# keyword → schema-value maps for heuristic ops filters — the typed
# TradesQuery/TicketsQuery fields, extracted from the question so
# "show failed trades" actually filters status=failed
_OPS_FILTER_KEYWORDS = {
    "trades": {
        "status": {
            "failed": "failed",
            "returned": "returned",
            "settled": "settled",
            "pending": "pending",
        },
        "trade_type": {
            "wire": "wire",
            "ach": "ach",
            "check": "check",
            "book transfer": "book_transfer",
        },
    },
    "tickets": {
        "status": {
            "open": "open",
            "in progress": "in_progress",
            "resolved": "resolved",
            "closed": "closed",
        },
        "category": {
            "fee dispute": "fee_dispute",
            "missing statement": "missing_statement",
            "failed wire": "failed_wire",
            "cutoff": "cutoff_inquiry",
            "access request": "access_request",
        },
    },
}


def _ops_filters(question: str, ops: list[str]) -> dict:
    """Extract ops-query filters from question keywords (heuristic path)."""
    q = question.lower()
    out: dict = {}
    for tool in ops:
        fields = {}
        for field, mapping in _OPS_FILTER_KEYWORDS.get(tool, {}).items():
            for kw, val in mapping.items():
                if kw in q:
                    fields[field] = val
                    break
        if fields:
            out[tool] = fields
    return out


def _known_clients() -> frozenset[str]:
    """Every client id the deployment knows — the union of entitlements in
    the users fixture. Tenant names are data, not code: a hardcoded list
    silently stops recognizing clients the fixture later gains.
    """
    from app.security.authn import load_users

    try:
        users = load_users(get_settings().users_file)
    except Exception:
        return frozenset()
    return frozenset(cid for u in users.values() for cid in u.get("client_ids", ()))


def _extract_client_id(question: str) -> str | None:
    """Return the first *known* client id mentioned in the question.

    Words are matched positionally so the leftmost mention wins; unknown
    names return None rather than a partial guess.
    """
    known = _known_clients()
    for word in re.findall(r"[a-z][a-z0-9]+", question.lower()):
        if word in known:
            return word
    return None


def _heuristic_route(question: str) -> dict:
    """Classify sources/ops tools by keyword matching (offline fallback).

    Also extracts a client id when a known client name appears in the
    question. Defaults to all collections when no keyword matches.
    """
    q = question.lower()
    sources = [c for c, kws in _KEYWORDS.items() if any(k in q for k in kws)]
    # word-boundary match for ops keywords — "Trade Settlement SLA" must
    # not fire the trades tool, and "showcase" must not fire tickets
    ops = [
        o
        for o, kws in _OPS_KEYWORDS.items()
        if any(re.search(rf"\b{re.escape(k)}\b", q) for k in kws)
    ]
    cid = _extract_client_id(q)
    return {"sources": sources or ALL_SOURCES, "ops": ops, "client_id": cid}


def route(state):
    """Decide which collections and ops tools the question needs.

    Uses the LLM router when an API key is configured, else the keyword
    heuristic. Enforces client entitlements (refusing on a denied client),
    extracts TKT-/TRD- record ids for point lookups, audits the decision,
    and returns sources, ops calls, and routing extras — or a refusal.
    """
    principal = state["principal"]
    updates: dict = {}
    with Timer() as t:
        decision = None
        if get_settings().anthropic_api_key:
            try:
                prompt = load_prompt("route.md").format(
                    question=state["question"], clients=", ".join(principal.client_ids)
                )
                response = get_llm().invoke(prompt)
                updates.update(budget.record_llm_usage(state, response))
                raw = response.content
                decision = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
            except Exception as exc:
                # LLM timeout/parse failure → heuristic fallback, not a crash
                updates["errors"] = add_error(state, f"route.llm_failed: {exc}")
                decision = None
        if decision is None:
            decision = _heuristic_route(state["question"])
        # constrain to entitled clients — asking about a client the user
        # can't see refuses rather than answering with other clients' data.
        # an explicit request hint (API client_id) takes precedence over
        # what the router extracted from the question text.
        hint = (state.get("extra") or {}).get("client_id")
        cid = (hint or decision.get("client_id") or "").lower() or None
        denied = bool(cid) and cid not in principal.client_ids
        if denied:
            cid = None
        sources = [s for s in decision.get("sources", []) if s in ALL_SOURCES] or ALL_SOURCES
        ops = [o for o in decision.get("ops", []) if o in ALL_OPS]
        # record ids (TKT-xxxxx / TRD-xxxxx) mentioned in the question so
        # retrieve can do point lookups instead of broad scans
        extra_ids = {
            "ticket_id": (
                m.group(0) if (m := re.search(r"\bTKT-\d+\b", state["question"], re.I)) else None
            ),
            "trade_id": (
                m.group(0).upper()
                if (m := re.search(r"\btrd-\d+\b", state["question"], re.I))
                else None
            ),
        }
    audit(
        state,
        EV_AGENT_STEP,
        {
            "node": "route",
            "decision": decision,
            "effective_client_id": cid,
            "entitlement_denied": denied,
            "record_ids": extra_ids,
        },
        t.ms,
    )
    extra = dict(state.get("extra", {}))
    extra.update({"client_id": cid, **extra_ids})
    if filters := _ops_filters(state["question"], ops):
        extra["ops_filters"] = filters
    if denied:
        return {
            "refusal": True,
            "refusal_reason": f"client '{decision.get('client_id')}' not in user entitlements",
            "answerability": Answerability.UNAUTHORIZED,
            "extra": extra,
            **updates,
        }
    return {
        "sources": sources,
        "ops_calls": [{"tool": o} for o in ops],
        "extra": extra,
        **updates,
    }
