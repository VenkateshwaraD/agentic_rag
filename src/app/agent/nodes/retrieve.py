"""Retrieve node — runs the hybrid pipeline per routed collection and the
routed ops tools (read-only, entitlement-scoped). Every call is audited.

Author: venkateshwara doijode
"""

from app.agent.nodes.common import Timer, audit
from app.core.constants import EV_RETRIEVAL, EV_TOOL_CALL
from app.core.dependencies import get_hybrid
from app.security.authz import acl_for
from app.tools.ops import accounts, tickets, trades


def _run_ops_tool(tool: str, principal, client_id: str | None, extra: dict) -> dict:
    """Dispatch a routed ops call with entitlement-scoped params.
    Point-lookup ids from the question take precedence over broad scans."""
    if tool == "accounts":
        rows = (
            accounts.get_account(client_id, principal)
            if client_id
            else accounts.list_accounts(principal)
        )
    elif tool == "trades":
        rows = (
            trades.get_trade(extra["trade_id"], principal)
            if extra.get("trade_id")
            else trades.search_trades(principal, client_id=client_id)
        )
    elif tool == "tickets":
        rows = (
            tickets.get_ticket(extra["ticket_id"], principal)
            if extra.get("ticket_id")
            else tickets.search_tickets(principal, client_id=client_id)
        )
    else:
        rows = []
    return {"tool": tool, "client_id": client_id, "rows": rows}


def retrieve(state):
    """Gather evidence for the question.

    Resolves the caller's ACL, adds implicit tickets/trades tool calls when
    record ids appear in the question, runs hybrid search over each routed
    collection, executes each routed ops tool, audits both stages, and
    returns retrieved chunks, per-collection diagnostics, and tool rows.
    """
    principal = state["principal"]
    acl = acl_for(principal)
    extra = state.get("extra", {})
    client_id = extra.get("client_id")
    query = state["question"]

    # a TKT-/TRD- id in the question implies the system tool even if the
    # router didn't select it
    ops_calls = [c["tool"] for c in state.get("ops_calls", [])]
    if extra.get("ticket_id") and "tickets" not in ops_calls:
        ops_calls.append("tickets")
    if extra.get("trade_id") and "trades" not in ops_calls:
        ops_calls.append("trades")

    retrieved: list[dict] = []
    diagnostics: dict = {}
    with Timer() as t:
        for coll in state.get("sources", []):
            out = get_hybrid().search(query, coll, acl, focus_client=client_id)
            for r in out["results"]:
                r["collection"] = coll
            retrieved.extend(out["results"])
            diagnostics[coll] = out["diagnostics"]
    audit(
        state,
        EV_RETRIEVAL,
        {
            "collections": state.get("sources", []),
            "doc_ids": [r["doc_id"] for r in retrieved],
            "counts": {c: d["counts"] for c, d in diagnostics.items()},
        },
        t.ms,
    )

    tool_results = []
    for tool in ops_calls:
        with Timer() as tt:
            res = _run_ops_tool(tool, principal, client_id, extra)
        audit(
            state,
            EV_TOOL_CALL,
            {"tool": res["tool"], "client_id": res["client_id"], "row_count": len(res["rows"])},
            tt.ms,
        )
        tool_results.append(res)

    return {"retrieved": retrieved, "diagnostics": diagnostics, "tool_results": tool_results}
