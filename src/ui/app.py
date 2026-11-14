"""Streamlit prototype UI — chat with citations + eval dashboard.

Run: streamlit run src/ui/app.py

Author: venkateshwara doijode
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
REPO = Path(__file__).resolve().parents[2]
USERS_FILE = REPO / "db" / "seed" / "users.json"
REPORTS_DIR = REPO / "eval" / "reports"
SYNTHETIC_DIR = REPO / "data" / "synthetic"


@st.cache_data
def _users():
    """Load and cache the users fixture, keyed by ``user_id``."""
    return {u["user_id"]: u for u in json.loads(USERS_FILE.read_text())}


@st.cache_data
def _doc_titles() -> dict[str, str]:
    """doc_id → document title, scanned from corpus file headers."""
    out = {}
    for p in SYNTHETIC_DIR.rglob("*.md"):
        try:
            head = p.read_text()[:4000]
        except OSError:
            continue
        doc_match = re.search(r"doc_id:\s*([A-Z0-9-]+)", head)
        title_match = re.search(r"^#\s+(.+)$", head, re.MULTILINE)
        if doc_match:
            out[doc_match.group(1)] = title_match.group(1).strip() if title_match else p.stem
    return out


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
    """Visible citation list — doc id + title, version/date, system-of-record rows."""
    titles = _doc_titles()
    cits = turn.get("citations") or []
    tools = [t for t in (turn.get("tool_calls") or []) if not t.get("error")]
    if not (cits or tools):
        return
    st.markdown("**Sources**")
    for c in cits:
        title = titles.get(c["doc_id"], "")
        head = f"`{c['doc_id']}`" + (f" — {title}" if title else "")
        tail = [f"§{c['section']}" if c.get("section") else ""]
        if c.get("doc_version"):
            tail.append(f"v{c['doc_version']}")
        if c.get("effective_date"):
            tail.append(f"effective {c['effective_date']}")
        st.markdown(f"- {head} · {' · '.join(tail)}")
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
        counts = d.get("counts", {})
        ids = d.get("reranked_ids") or d.get("fused_ids") or d.get("dense_ids") or []
        top = ", ".join(dict.fromkeys(ids)) if ids else "—"
        line = (
            f"**{coll}** — strategy `{d.get('strategy')}` · dense {counts.get('dense', 0)}"
            f" / bm25 {counts.get('bm25', 0)} / fused {counts.get('fused', 0)}"
        )
        if counts.get("stale"):
            line += f" · stale {counts['stale']}"
        if counts.get("parent_expanded"):
            line += f" · parents +{counts['parent_expanded']}"
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
    validation = turn.get("validation") or {}
    if validation.get("verdict") not in (None, "", "skipped"):
        claims = validation.get("claims") or []
        supported = sum(1 for c in claims if c.get("supported"))
        cited = sum(1 for c in claims if c.get("cited"))
        via_tool = sum(1 for c in claims if c.get("via") == "tool")
        steps.append(
            (
                "validate",
                f"verdict `{validation['verdict']}` — {supported}/{len(claims)} claims supported"
                f" ({via_tool} via tools) · {cited} cited"
                f" · {len(validation.get('invalid_citations') or [])} invalid citations",
            )
        )
    if turn.get("refusal"):
        steps.append(("guardrail", "reply blocked — refusal issued"))
    if turn.get("pending_review"):
        steps.append(("review", "enqueued for a human reviewer"))
    budget = turn.get("budget") or {}
    budget_line = (
        f"iterations {budget.get('iterations')} · tool calls {budget.get('tool_calls')}"
        f" · llm calls {budget.get('llm_calls')} · tokens {budget.get('tokens_used')}"
        f" · ${budget.get('cost_usd') or 0:.3f}"
    )
    if budget.get("breach"):
        budget_line += f" · **BREACH: {budget['breach']}**"
    steps.append(("budget", budget_line))
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


_EXAMPLES = {
    "u_agent1": [
        "What is the outgoing domestic wire fee for bluepeak?",
        "Show open tickets for acme",
        "Do the Trade Settlement SLA and the Digital Channels SLA agree on the domestic wire same-day cutoff?",
        "Show accounts for grainmark",
    ],
    "u_agent2": [
        "Show open tickets for grainmark",
        "What fees apply for harborline?",
        "What is the wire cutoff for domestic wires?",
        "Show accounts for acme",
    ],
}
_EXAMPLES_DEFAULT = [
    "What is the outgoing domestic wire fee for bluepeak?",
    "Show open tickets for acme",
    "Do the Trade Settlement SLA and the Digital Channels SLA agree on the domestic wire same-day cutoff?",
    "What is the crypto custody fee for bluepeak?",
]


def _ask_and_store(q: str, user_id: str) -> None:
    """Run one question through the API and append the turn to history."""
    with st.spinner("Searching agreements, SLAs, and systems of record…"):
        out = call_answer(q, user_id)
    st.session_state.history.append({"q": q, **{k: out[k] for k in _ANSWER_FIELDS if k in out}})


def chat_page():
    """Render the chat page: user picker, clickable examples, history, input.

    Each turn renders a verdict banner, the reply, a visible sources list,
    and an expandable per-node agent-steps timeline.
    """
    st.title("Client-Service Copilot")
    user_id = st.sidebar.selectbox("Demo — acting as", list(_users()), index=0)
    user = _users()[user_id]
    st.sidebar.caption(f"role: `{user['role']}` — entitled: {', '.join(user['client_ids'])}")
    st.sidebar.markdown("**Try:**")
    for ex in _EXAMPLES.get(user_id, _EXAMPLES_DEFAULT):
        if st.sidebar.button(ex, key=f"ex-{ex[:48]}"):
            try:
                _ask_and_store(ex, user_id)
            except httpx.HTTPError as e:
                st.error(f"API error: {e}")
            st.rerun()

    if "history" not in st.session_state:
        st.session_state.history = []

    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["q"])
        with st.chat_message("assistant"):
            _render_answer(turn)

    if q := st.chat_input("Ask about fees, cutoffs, tickets, accounts…"):
        try:
            _ask_and_store(q, user_id)
        except httpx.HTTPError as e:
            st.error(f"API error: {e}")
            return
        st.rerun()


# --- eval dashboard -----------------------------------------------------------


def _fmtv(v, nd: int = 3) -> str:
    """Display value — ``None`` renders ``n/a``, never ``None``."""
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _report_label(p: Path) -> str:
    """'agentic · pr · 14 Nov 10:42' from eval[_system]_<ts>.json filenames."""
    eval_match = re.match(r"eval(?:_([a-z-]+))?_(\d{8})_(\d{6})", p.stem)
    if not eval_match:
        return p.stem
    system = eval_match.group(1) or "agentic"
    try:
        ts = datetime.strptime(eval_match.group(2) + eval_match.group(3), "%Y%eval_match%d%H%M%S")
        stamp = ts.strftime("%d %b %H:%M")
    except ValueError:
        stamp = eval_match.group(2)
    suite = ""
    try:
        suite = Path(
            json.loads(p.read_text()).get("version_snapshot", {}).get("dataset", {}).get("path", "")
        ).stem
    except Exception:  # unreadable report → label still works without a suite name
        suite = ""
    return f"{system} · {suite or '?'} · {stamp}"


def _metric_rows(summary: dict, n_q: int) -> list[dict]:
    """metric | value | n — hides ``*_n`` coverage keys and formats ``n/a``."""
    rows = []
    for k, v in summary.items():
        if k.endswith("_n"):
            continue
        rows.append({"metric": k, "value": _fmtv(v), "n": summary.get(f"{k}_n", n_q)})
    return rows


def _category_table(by_category: dict) -> list[dict]:
    """One row per metric, one column per category (+ its sample size)."""
    cats = list(by_category)
    metrics = sorted({k for c in by_category.values() for k in c if k != "n"})
    header = {c: f"{c} (n={by_category[c].get('n', '?')})" for c in cats}
    return [
        {"metric": m, **{header[c]: _fmtv(by_category[c].get(m)) for c in cats}} for m in metrics
    ]


def eval_page():
    """Eval dashboard — labeled picker, metadata, metric tables, failure reasons."""
    st.title("Eval Reports")
    reports = sorted(REPORTS_DIR.glob("eval*.json"), reverse=True)
    reports = [r for r in reports if not r.stem.startswith("compare_")]
    if not reports:
        st.info("No reports yet — run `copilot eval run`.")
        return
    qp_report = st.query_params.get("report", "")
    names = [r.name for r in reports]
    default = qp_report if qp_report in names else names[0]
    chosen = st.selectbox(
        "Report",
        reports,
        index=names.index(default),
        format_func=_report_label,
    )
    data = json.loads(chosen.read_text())

    snap = data.get("version_snapshot", {})
    with st.expander("Run metadata", expanded=False):
        cols = st.columns(4)
        cols[0].metric("system", data.get("system", "agentic"))
        cols[1].metric("model", snap.get("model", "n/a"))
        cols[2].metric("embedding", snap.get("embedding", {}).get("provider", "n/a"))
        cols[3].metric("git", snap.get("git_sha", "n/a"))
        st.caption(
            f"suite `{Path(snap.get('dataset', {}).get('path', 'n/a')).stem}`"
            f" · rows {snap.get('dataset', {}).get('rows', 'n/a')}"
            f" · dataset sha `{snap.get('dataset', {}).get('sha256', 'n/a')}`"
            f" · judge `{snap.get('judge_model', 'n/a')}`"
            f" · repeat {snap.get('repeat', data.get('runs', 1))}"
        )
    if snap.get("llm_configured") is False:
        st.warning(
            "This run used the deterministic **offline draft** — scores are "
            "comparable to other offline runs, not to LLM runs."
        )

    st.subheader("Summary")
    st.dataframe(
        _metric_rows(data.get("summary", {}), data.get("n_questions")),
        use_container_width=True,
        hide_index=True,
    )
    if data.get("variance"):
        st.caption("±95% CI across repeats")
        st.dataframe(data["variance"], use_container_width=True)

    if data.get("by_category"):
        st.subheader("By category")
        st.dataframe(_category_table(data["by_category"]), use_container_width=True)

    rows = data.get("rows") or []
    failures = data.get("failures") or []
    if rows:
        st.subheader("Rows")
        only_fail = st.toggle("Only failures", value=True)
        shown = [r for r in rows if r.get("failed_metrics")] if only_fail else rows
        st.dataframe(
            [
                {
                    "id": r["id"],
                    "category": r["category"],
                    "answerability": _fmtv(r.get("answerability")),
                    "failed": ", ".join(r.get("failed_metrics") or []) or "—",
                    "cost $": _fmtv(r.get("cost_usd"), 4),
                    "llm": _fmtv(r.get("llm_calls")),
                    "tokens": _fmtv(r.get("tokens_used")),
                }
                for r in shown
            ],
            use_container_width=True,
            hide_index=True,
        )

    if failures:
        st.subheader(f"Failures ({len(failures)} of {data.get('n_questions', '?')})")
        failed_metric_names = sorted({m for f in failures for m in f["metrics"]})
        picked = st.multiselect("Filter by failed metric", failed_metric_names)
        for f in failures:
            if picked and not (set(f["metrics"]) & set(picked)):
                continue
            title = f"{f['id']} · {f['category']} — " + ", ".join(
                f"{k} {_fmtv(v)}" for k, v in f["metrics"].items()
            )
            with st.expander(title):
                st.markdown(f"**Q:** {f['question']}")
                cols = st.columns(2)
                cols[0].markdown("**Expected**")
                cols[0].json(f.get("expected") or {"—": "no expectation recorded"})
                cols[1].markdown("**Actual**")
                cols[1].json({k: v for k, v in (f.get("actual") or {}).items() if v is not None})
                st.markdown("**Reply**")
                st.markdown(f["reply"])


def _ts_label(stem: str) -> str:
    """'compare · 14 Nov 10:42' from compare_<ts> stems."""
    date_match = re.search(r"(\d{8})_(\d{6})", stem)
    if not date_match:
        return stem
    try:
        return datetime.strptime(
            date_match.group(1) + date_match.group(2), "%Y%date_match%d%H%M%S"
        ).strftime("%d %b %H:%M")
    except ValueError:
        return stem


def compare_page():
    """Baseline comparison — all systems side by side + deltas vs a chosen reference."""
    st.title("Baseline comparison")
    files = sorted(REPORTS_DIR.glob("compare_*.json"), reverse=True)
    if not files:
        st.info("No comparison reports — run `runner.py --system all`.")
        return
    chosen = st.selectbox("Report", files, format_func=lambda p: _ts_label(p.stem))
    data = json.loads(chosen.read_text())
    systems = data.get("systems", [])

    leaks = data.get("acl_leaks") or {}
    if leaks:
        st.markdown("**ACL leaks:** " + " · ".join(f"`{s}` {n}" for s, n in leaks.items()))

    refs = systems or ["agentic"]
    ref = st.selectbox(
        "Δ reference system",
        refs,
        index=refs.index("plain-hybrid-acl") if "plain-hybrid-acl" in refs else 0,
    )

    st.subheader("Summary")
    st.dataframe(
        [
            {"metric": k, **{s: _fmtv(data["summary"][k].get(s)) for s in systems}}
            for k in data.get("summary", {})
            if not k.endswith("_n")
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(f"Δ vs {ref}")
    delta_rows = []
    for k, per_sys in data.get("summary", {}).items():
        if k.endswith("_n"):
            continue
        rv = per_sys.get(ref)
        delta_rows.append(
            {
                "metric": k,
                **{
                    s: (
                        _fmtv(round(v - rv, 4), 4)
                        if isinstance(v, (int, float)) and isinstance(rv, (int, float))
                        else "n/a"
                    )
                    for s, v in per_sys.items()
                },
            }
        )
    st.dataframe(delta_rows, use_container_width=True, hide_index=True)

    by_cat = data.get("by_category") or {}
    if by_cat:
        st.subheader("Per-category gap")
        metric = st.selectbox("Metric", list(by_cat))
        cats = by_cat[metric]
        st.dataframe(
            [{"category": c, **{s: _fmtv(per.get(s)) for s in systems}} for c, per in cats.items()],
            use_container_width=True,
            hide_index=True,
        )


_PAGES = {"chat": "Chat", "eval": "Eval dashboard", "compare": "Baseline compare"}
_default_page = _PAGES.get(st.query_params.get("page", ""), "Chat")
page = st.sidebar.radio(
    "Page", list(_PAGES.values()), index=list(_PAGES.values()).index(_default_page)
)
_result = (
    chat_page() if page == "Chat" else (eval_page() if page == "Eval dashboard" else compare_page())
)
