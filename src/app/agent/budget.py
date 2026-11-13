"""Per-request execution budget — iterations, tool calls, tokens, elapsed, cost.

Hard limits (iterations, tool calls, elapsed time) produce a refusal when
breached; soft limits (token usage, estimated cost) are best-effort — the
run records the breach but is allowed to reach synthesis with what it has.

Author: venkateshwara doijode
"""

import time
from typing import Any

from app.core.config import get_settings
from app.core.constants import LLM_COST_RATES

# Limit names enforced as hard vs soft (per approved budget policy).
HARD_LIMITS = ("iterations", "tool_calls", "elapsed_ms")
SOFT_LIMITS = ("tokens", "cost_usd")


def begin() -> dict[str, Any]:
    """Return the initial budget counters seeded onto the agent state."""
    return {
        "started_at": time.monotonic(),
        "iterations": 0,
        "tool_call_count": 0,
        "llm_calls": 0,
        "tokens_used": 0,
        "cost_usd": 0.0,
        "errors": [],
    }


def _elapsed_ms(state) -> float:
    """Return wall-clock milliseconds since ``started_at`` (0 when unset)."""
    started = state.get("started_at")
    return (time.monotonic() - started) * 1000 if started else 0.0


def breach(state) -> dict | None:
    """Return ``{limit, used, cap, hard}`` for the first breached limit, else None."""
    s = get_settings()
    checks = [
        ("iterations", state.get("iterations", 0), s.agent_max_iterations, True),
        ("tool_calls", state.get("tool_call_count", 0), s.agent_max_tool_calls, True),
        # per-request timeout header can tighten (never widen) the cap
        (
            "elapsed_ms",
            _elapsed_ms(state),
            min(
                state.get("max_elapsed_ms_override") or s.agent_max_elapsed_ms,
                s.agent_max_elapsed_ms,
            ),
            True,
        ),
        ("tokens", state.get("tokens_used", 0), s.agent_max_tokens, False),
        ("cost_usd", state.get("cost_usd", 0.0), s.agent_max_cost_usd, False),
    ]
    for limit, used, cap, hard in checks:
        if used >= cap:
            return {"limit": limit, "used": used, "cap": cap, "hard": hard}
    return None


def refusal_update(state) -> dict | None:
    """Return a refusal state update when a *hard* budget limit is breached.

    Soft breaches (tokens/cost) only record ``budget_breach`` — the request
    continues to synthesis with the evidence gathered so far.
    """
    b = breach(state)
    if b is None:
        return None
    if not b["hard"]:
        return {"budget_breach": b}
    return {
        "budget_breach": b,
        "refusal": True,
        "refusal_reason": (f"execution budget exceeded: {b['limit']} ({b['used']} ≥ {b['cap']})"),
    }


def record_llm_usage(state, response) -> dict[str, Any]:
    """Accumulate LLM call count, token usage, and estimated cost into state updates.

    Reads ``response.usage_metadata`` (LangChain ``AIMessage``); unknown
    models fall back to the ``default`` rate so cost is never silently zero.
    """
    usage = getattr(response, "usage_metadata", None) or {}
    in_tok = int(usage.get("input_tokens", 0) or 0)
    out_tok = int(usage.get("output_tokens", 0) or 0)
    s = get_settings()
    in_rate, out_rate = LLM_COST_RATES.get(s.anthropic_model, LLM_COST_RATES["default"])
    cost = (in_tok * in_rate + out_tok * out_rate) / 1_000_000
    return {
        "llm_calls": state.get("llm_calls", 0) + 1,
        "tokens_used": state.get("tokens_used", 0) + in_tok + out_tok,
        "cost_usd": state.get("cost_usd", 0.0) + cost,
    }
