"""Route node — LLM classifies which sources/tools the inquiry needs,
with a keyword heuristic fallback when no LLM key is configured."""

import json
import re

from app.agent.llm import get_llm
from app.agent.nodes.common import Timer, audit, load_prompt
from app.core.config import get_settings
from app.core.constants import (
    COLLECTION_AGREEMENTS,
    COLLECTION_SERVICE_TEAMS,
    COLLECTION_SLAS,
    EV_AGENT_STEP,
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
        "trade",
        "wire status",
        "payment status",
        "transfer status",
        "settled",
        "failed wire",
        "transaction",
    ),
    "tickets": ("ticket", "case", "history", "previous issue", "complaint", "tkt-"),
    "accounts": ("account", "tier", "region", "agreement on file", "account details"),
}


def _heuristic_route(question: str) -> dict:
    q = question.lower()
    sources = [c for c, kws in _KEYWORDS.items() if any(k in q for k in kws)]
    ops = [o for o, kws in _OPS_KEYWORDS.items() if any(k in q for k in kws)]
    cid = None
    m = re.search(
        r"\b(acme|bluepeak|crestline|deltaworks|everfield|foxglove|"
        r"grainmark|harborline|ironvale|junipersky|kestrelbay|"
        r"latticepoint)\b",
        q,
    )
    if m:
        cid = m.group(1)
    return {"sources": sources or ALL_SOURCES, "ops": ops, "client_id": cid}


def route(state):
    principal = state["principal"]
    with Timer() as t:
        decision = None
        if get_settings().anthropic_api_key:
            prompt = load_prompt("route.md").format(
                question=state["question"], clients=", ".join(principal.client_ids)
            )
            raw = get_llm().invoke(prompt).content
            try:
                decision = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
            except (AttributeError, json.JSONDecodeError):
                decision = None
        if decision is None:
            decision = _heuristic_route(state["question"])
        # constrain to entitled clients — asking about a client the user
        # can't see refuses rather than answering with other clients' data
        cid = decision.get("client_id")
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
    if denied:
        return {
            "refusal": True,
            "refusal_reason": f"client '{decision.get('client_id')}' not in user entitlements",
            "extra": extra,
        }
    return {"sources": sources, "ops_calls": [{"tool": o} for o in ops], "extra": extra}
