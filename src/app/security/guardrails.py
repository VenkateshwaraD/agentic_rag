"""Guardrail checks used by the agent's input/output guardrail nodes.

Author: venkateshwara doijode
"""

import re
from dataclasses import dataclass

from app.core.dependencies import get_pii_redactor

INJECTION_PATTERNS = [
    re.compile(r"ignore (all |any )?previous instructions", re.I),
    re.compile(r"you are now in (maintenance|debug|admin) mode", re.I),
    re.compile(r"(reveal|print|show) (all |your )?(system )?prompt", re.I),
    re.compile(r"disregard (your|all|the) (instructions|rules|guidelines)", re.I),
]

MAX_QUESTION_LEN = 2000


@dataclass(frozen=True)
class GuardrailResult:
    """Outcome of a guardrail check: whether content passed and why not."""

    allowed: bool
    reason: str = ""


def check_input(question: str) -> GuardrailResult:
    """Input screen: length cap + prompt-injection pattern match."""
    if not question or not question.strip():
        return GuardrailResult(False, "empty question")
    if len(question) > MAX_QUESTION_LEN:
        return GuardrailResult(False, "question exceeds length limit")
    for pat in INJECTION_PATTERNS:
        if pat.search(question):
            return GuardrailResult(False, "prompt-injection pattern detected")
    return GuardrailResult(True)


def check_output(draft: str) -> GuardrailResult:
    """Output screen: PII rescan of the drafted reply (belt-and-suspenders
    on top of ingest-time redaction) + echoed-injection check."""
    if get_pii_redactor().contains_pii(draft):
        return GuardrailResult(False, "draft contains unredacted PII")
    for pat in INJECTION_PATTERNS:
        if pat.search(draft):
            return GuardrailResult(False, "draft echoes injection-style content")
    return GuardrailResult(True)
