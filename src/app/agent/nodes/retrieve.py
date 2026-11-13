"""Retrieve node — runs the hybrid pipeline per routed collection and the
routed ops tools (read-only, entitlement-scoped). Every call is audited.

Author: venkateshwara doijode
"""

from contextlib import nullcontext

from app.agent.nodes.common import Timer, add_error, audit
from app.core.constants import EV_RETRIEVAL, EV_TOOL_CALL
from app.core.dependencies import (
    get_accounts_store,
    get_hybrid,
    get_tickets_store,
    get_trades_store,
)
from app.security.authz import acl_for
from app.tools.ops.client import ops_snapshot
from app.tools.ops.schemas import AccountsQuery, TicketsQuery, TradesQuery


def _run_ops_tool(tool: str, principal, client_id: str | None, extra: dict, conn=None) -> dict:
    """Dispatch a routed ops call through the typed store interface.

    Point-lookup ids from the question take precedence over broad scans.
    The OpsResult envelope is flattened into the tool-result dict the rest
    of the graph consumes — typed rows become dicts, structured errors and
    freshness/pass-through fields are carried alongside.
    """
    if tool == "accounts":
        res = get_accounts_store().get(AccountsQuery(client_id=client_id), principal, conn=conn)
    elif tool == "trades":
        res = get_trades_store().get(
            TradesQuery(trade_id=extra.get("trade_id"), client_id=client_id),
            principal,
            conn=conn,
        )
    elif tool == "tickets":
        res = get_tickets_store().get(
            TicketsQuery(ticket_id=extra.get("ticket_id"), client_id=client_id),
            principal,
            conn=conn,
        )
    else:
        return {"tool": tool, "client_id": client_id, "rows": []}
    return {
        "tool": tool,
        "client_id": client_id,
        "rows": [r.model_dump() for r in res.rows],
        "has_more": res.has_more,
        "fresh_at": res.fresh_at.isoformat() if res.fresh_at else None,
        "source_updated_at": res.source_updated_at,
        "snapshot_ts": res.snapshot_ts.isoformat() if res.snapshot_ts else None,
        "latency_ms": res.latency_ms,
        "error": res.error.model_dump() if res.error else None,
    }


def retrieve(state):
    """Gather evidence for the question.

    Resolves the caller's ACL, adds implicit tickets/trades tool calls when
    record ids appear in the question, runs hybrid search over each routed
    collection, executes each routed ops tool, audits both stages, and
    returns retrieved chunks, per-collection diagnostics, and tool rows.

    Store and tool failures are captured per-call in ``errors`` (and the
    collection's diagnostics) instead of crashing the graph — the grade
    node sees partial evidence and may still answer or refuse.
    """
    principal = state["principal"]
    acl = acl_for(principal)
    extra = state.get("extra", {})
    client_id = extra.get("client_id")
    query = state["question"]
    updates: dict = {"iterations": state.get("iterations", 0) + 1}

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
            try:
                out = get_hybrid().search(query, coll, acl, focus_client=client_id)
            except Exception as exc:
                # vector store / OpenSearch unavailable → record + continue
                updates["errors"] = add_error(
                    {**state, "errors": updates.get("errors", [])},
                    f"retrieve.{coll}_failed: {exc}",
                )
                diagnostics[coll] = {"error": "store_unavailable", "detail": str(exc)[:200]}
                continue
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
            "counts": {c: d.get("counts", {}) for c, d in diagnostics.items()},
            "errors": updates.get("errors", []),
        },
        t.ms,
    )

    tool_results = []
    tool_call_count = state.get("tool_call_count", 0)
    # shared REPEATABLE-READ snapshot when >1 ops system is read — rows from
    # different stores then reflect one consistent point in time
    snapshot_ctx = ops_snapshot() if len(ops_calls) > 1 else nullcontext()
    with snapshot_ctx as snap:
        conn = snap[0] if snap else None
        snapshot_ts = snap[1].isoformat() if snap else None
        for tool in ops_calls:
            with Timer() as tt:
                try:
                    res = _run_ops_tool(tool, principal, client_id, extra, conn=conn)
                    res.setdefault("snapshot_ts", snapshot_ts)
                    tool_call_count += 1
                except Exception as exc:
                    # ops store down / malformed response → empty result + error
                    updates["errors"] = add_error(
                        {**state, "errors": updates.get("errors", [])},
                        f"retrieve.tool_{tool}_failed: {exc}",
                    )
                    res = {
                        "tool": tool,
                        "client_id": client_id,
                        "rows": [],
                        "error": {"code": "store_unavailable", "detail": str(exc)[:200]},
                    }
            audit(
                state,
                EV_TOOL_CALL,
                {
                    "tool": res["tool"],
                    "client_id": res["client_id"],
                    "row_count": len(res["rows"]),
                    "error": res.get("error"),
                },
                tt.ms,
            )
            tool_results.append(res)

    return {
        "retrieved": retrieved,
        "diagnostics": diagnostics,
        "tool_results": tool_results,
        "tool_call_count": tool_call_count,
        **updates,
    }
