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


def chat_page():
    """Render the chat page: user picker, history, and the question input.

    Each turn renders the reply with a citations/diagnostics expander; new
    questions call the API and are appended to session history.
    """
    st.title("Client-Service Copilot")
    user_id = st.sidebar.selectbox("Acting as", list(_users()), index=0)
    user = _users()[user_id]
    st.sidebar.caption(f"role: `{user['role']}` — entitled clients: {len(user['client_ids'])}")

    if "history" not in st.session_state:
        st.session_state.history = []

    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["q"])
        with st.chat_message("assistant"):
            st.write(turn["reply"])
            with st.expander("Citations & diagnostics"):
                st.json(
                    {
                        "citations": turn["citations"],
                        "tool_calls": turn["tool_calls"],
                        "refusal": turn["refusal"],
                        "trace_id": turn["trace_id"],
                        "diagnostics": turn.get("diagnostics"),
                    }
                )

    if q := st.chat_input("Ask about fees, cutoffs, tickets, accounts…"):
        with st.spinner("Searching agreements, SLAs, and systems of record…"):
            try:
                out = call_answer(q, user_id)
            except httpx.HTTPError as e:
                st.error(f"API error: {e}")
                return
        st.session_state.history.append(
            {
                "q": q,
                **{
                    k: out[k]
                    for k in (
                        "reply",
                        "citations",
                        "tool_calls",
                        "refusal",
                        "trace_id",
                        "diagnostics",
                    )
                },
            }
        )
        st.rerun()


def eval_page():
    """Render the eval dashboard: pick a report JSON and view summary/by-category/failures."""
    st.title("Eval Reports")
    reports = sorted(REPORTS_DIR.glob("*.json"), reverse=True)
    if not reports:
        st.info("No reports yet — run `copilot eval run`.")
        return
    chosen = st.selectbox("Report", [r.name for r in reports])
    data = json.loads((REPORTS_DIR / chosen).read_text())
    st.subheader("Summary")
    st.json(data.get("summary", {}))
    if data.get("by_category"):
        st.subheader("By category")
        st.dataframe(data["by_category"])
    if data.get("failures"):
        st.subheader("Failures")
        st.dataframe(data["failures"])


page = st.sidebar.radio("Page", ["Chat", "Eval dashboard"])
chat_page() if page == "Chat" else eval_page()
