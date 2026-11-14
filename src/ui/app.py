"""Streamlit prototype UI — chat with citations + eval dashboard.

Run: streamlit run src/ui/app.py

Author: venkateshwara doijode
"""

import json
import os
from pathlib import Path

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
USERS_FILE = Path(__file__).resolve().parents[2] / "db" / "seed" / "users.json"
REPORTS_DIR = Path(__file__).resolve().parents[2] / "eval" / "reports"


@st.cache_data
def _users():
    """Load and cache the users fixture, keyed by ``user_id``."""
    return {u["user_id"]: u for u in json.loads(USERS_FILE.read_text())}


def call_answer(question: str, user_id: str) -> dict:
    """POST the question to ``/answer`` as the selected user; returns the response JSON."""
    user = _users()[user_id]
    resp = httpx.post(
        f"{API_URL}/answer",
        json={"question": question},
        headers={"X-API-Key": user["api_key"], "X-User-Id": user_id},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


_ANSWER_FIELDS = (
    "reply",
    "citations",
    "tool_calls",
    "refusal",
    "abstained",
    "abstention_reason",
    "answerability",
    "pending_review",
    "validation",
    "budget",
    "errors",
    "diagnostics",
    "trace_id",
)


def _verdict_banner(turn: dict) -> None:
    """One-line colored verdict so refusals/abstains read as decisions, not errors."""
    if turn.get("refusal"):
        st.error("**Refused** — outside this user's entitlements or blocked by a guardrail")
    elif turn.get("abstained"):
        st.warning("**Abstained** — not enough entitled evidence to answer reliably")
    else:
        verdict = {
            "fully_answerable": ("Fully answerable", st.success),
            "partially_answerable": ("Partially answerable", st.info),
            "conflicting_evidence": (
                "Conflicting evidence — sources disagree",
                st.warning,
            ),
            "insufficient_evidence": ("Insufficient evidence", st.warning),
            "unauthorized": ("Unauthorized", st.error),
        }.get(turn.get("answerability"))
        if verdict:
            verdict[1](f"**{verdict[0]}**")
    if turn.get("pending_review"):
        st.caption("flagged for human review")


def _render_citations(turn: dict) -> None:
    """Visible citation list — document refs with version/date + system-of-record rows."""
    cits = turn.get("citations") or []
    tools = [t for t in (turn.get("tool_calls") or []) if not t.get("error")]
    if not (cits or tools):
        return
    st.markdown("**Sources**")
    for c in cits:
        parts = [f"`{c['doc_id']}`", c.get("section", "")]
        if c.get("doc_version"):
            parts.append(f"v{c['doc_version']}")
        if c.get("effective_date"):
            parts.append(f"effective {c['effective_date']}")
        st.markdown(f"- {' · '.join(p for p in parts if p)}")
    for t in tools:
        st.markdown(
            f"- `{t['tool']} system` — {t.get('row_count', 0)} row(s)"
            + (f" for `{t['client_id']}`" if t.get("client_id") else "")
        )


def _agent_steps(turn: dict) -> list[tuple[str, str]]:
    """Rebuild the pipeline timeline from the response diagnostics."""
    steps: list[tuple[str, str]] = []
    diag = turn.get("diagnostics") or {}
    tools = turn.get("tool_calls") or []
    cols = ", ".join(diag.keys()) or "—"
    ops = ", ".join(t["tool"] for t in tools) or "none"
    steps.append(("route", f"collections → {cols} · ops tools → {ops}"))
    for coll, d in diag.items():
        c = d.get("counts", {})
        ids = d.get("reranked_ids") or d.get("fused_ids") or d.get("dense_ids") or []
        top = ", ".join(dict.fromkeys(ids)) if ids else "—"
        line = (
            f"**{coll}** — strategy `{d.get('strategy')}` · dense {c.get('dense', 0)}"
            f" / bm25 {c.get('bm25', 0)} / fused {c.get('fused', 0)}"
        )
        if c.get("stale"):
            line += f" · stale {c['stale']}"
        if c.get("parent_expanded"):
            line += f" · parents +{c['parent_expanded']}"
        line += f" → {top}"
        if d.get("error"):
            line += f" · ⚠ {d['error']}"
        steps.append(("retrieve", line))
    for t in tools:
        line = f"`{t['tool']}` → {t.get('row_count', 0)} row(s)"
        if t.get("client_id"):
            line += f" for `{t['client_id']}`"
        if t.get("error"):
            err = t["error"]
            line += f" ⚠ {err.get('code') if isinstance(err, dict) else err}"
        steps.append(("tool", line))
    if turn.get("answerability"):
        steps.append(("grade", f"answerability → `{turn['answerability']}`"))
    v = turn.get("validation") or {}
    if v.get("verdict") not in (None, "", "skipped"):
        claims = v.get("claims") or []
        supported = sum(1 for c in claims if c.get("supported"))
        cited = sum(1 for c in claims if c.get("cited"))
        steps.append(
            (
                "validate",
                f"verdict `{v['verdict']}` — {supported}/{len(claims)} claims supported"
                f" · {cited} cited · {len(v.get('invalid_citations') or [])} invalid citations",
            )
        )
    if turn.get("refusal"):
        steps.append(("guardrail", "reply blocked — refusal issued"))
    if turn.get("pending_review"):
        steps.append(("review", "enqueued for a human reviewer"))
    b = turn.get("budget") or {}
    steps.append(
        (
            "budget",
            f"iterations {b.get('iterations')} · tool calls {b.get('tool_calls')}"
            f" · llm calls {b.get('llm_calls')} · tokens {b.get('tokens_used')}"
            f" · ${b.get('cost_usd') or 0:.3f}",
        )
    )
    return steps


def _render_answer(turn: dict) -> None:
    """Verdict banner → reply → visible sources → agent-steps timeline."""
    _verdict_banner(turn)
    st.markdown(turn["reply"])
    _render_citations(turn)
    with st.expander("Agent steps — route → retrieve → tools → grade → validate", expanded=True):
        for i, (node, detail) in enumerate(_agent_steps(turn), 1):
            st.markdown(f"{i}. **`{node}`** — {detail}")
        if turn.get("errors"):
            st.markdown(f"- errors: `{turn['errors']}`")
        st.caption(f"trace `{turn.get('trace_id')}`")


def chat_page():
    """Render the chat page: user picker, history, and the question input.

    Each turn renders a verdict banner, the reply, a visible sources list,
    and an expandable per-node agent-steps timeline.
    """
    st.title("Client-Service Copilot")
    user_id = st.sidebar.selectbox("Acting as", list(_users()), index=0)
    user = _users()[user_id]
    st.sidebar.caption(f"role: `{user['role']}` — entitled clients: {len(user['client_ids'])}")
    st.sidebar.markdown(
        "**Try:**\n"
        "- What is the outgoing domestic wire fee for bluepeak?\n"
        "- Show open tickets for acme\n"
        "- What is the wire cutoff for domestic wires?\n"
        "- Show accounts for grainmark *(not entitled)*\n"
        "- What is the crypto custody fee for bluepeak? *(unanswerable)*"
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["q"])
        with st.chat_message("assistant"):
            _render_answer(turn)

    if q := st.chat_input("Ask about fees, cutoffs, tickets, accounts…"):
        with st.spinner("Searching agreements, SLAs, and systems of record…"):
            try:
                out = call_answer(q, user_id)
            except httpx.HTTPError as e:
                st.error(f"API error: {e}")
                return
        st.session_state.history.append({"q": q, **{k: out[k] for k in _ANSWER_FIELDS if k in out}})
        st.rerun()


def eval_page():
    """Render the eval dashboard: pick a report JSON and view summary/by-category/failures."""
    st.title("Eval Reports")
    reports = sorted(REPORTS_DIR.glob("*.json"), reverse=True)
    if not reports:
        st.info("No reports yet — run `copilot eval run`.")
        return
    names = [r.name for r in reports]
    qp_report = st.query_params.get("report", "")
    eval_names = [n for n in names if n.startswith("eval_")]
    default = qp_report if qp_report in names else (eval_names[0] if eval_names else names[0])
    chosen = st.selectbox("Report", names, index=names.index(default))
    data = json.loads((REPORTS_DIR / chosen).read_text())
    st.subheader("Summary")
    st.json(data.get("summary", {}))
    if data.get("by_category"):
        st.subheader("By category")
        st.dataframe(data["by_category"])
    if data.get("failures"):
        st.subheader("Failures")
        st.dataframe(data["failures"])


_PAGES = {"chat": "Chat", "eval": "Eval dashboard"}
_default_page = _PAGES.get(st.query_params.get("page", ""), "Chat")
page = st.sidebar.radio(
    "Page", list(_PAGES.values()), index=list(_PAGES.values()).index(_default_page)
)
chat_page() if page == "Chat" else eval_page()
