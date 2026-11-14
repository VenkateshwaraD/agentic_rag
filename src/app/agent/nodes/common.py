"""Shared helpers for graph nodes: audit logging + prompt loading.

Author: venkateshwara doijode
"""

import time
from pathlib import Path

from app.core.dependencies import get_audit
from app.core.logging import get_logger

log = get_logger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"

# Words that describe the *shape* of a question, not its subject — function
# words, question-frame verbs (agree/compare/differ), and corpus-generic
# nouns (document/SLA/service) that appear everywhere and so can't identify
# a topic. Used to strip a question down to its content words: for the
# clarification ask ("which fee, which client?") and for grade's anchor
# coverage check (does any evidence mention the asked-about topic at all).
QUESTION_STOPWORDS = frozenset(
    {
        # function words / auxiliaries / pronouns
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "let",
        "may",
        "me",
        "might",
        "must",
        "my",
        "no",
        "not",
        "of",
        "on",
        "or",
        "our",
        "ours",
        "s",
        "shall",
        "she",
        "should",
        "so",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "us",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "whose",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        # question-frame verbs/adverbs — they describe what the asker wants
        # done with the subject, never the subject itself
        "agree",
        "agrees",
        "compare",
        "compares",
        "comparison",
        "conflict",
        "conflicts",
        "consistent",
        "contradict",
        "differ",
        "difference",
        "differences",
        "different",
        "explain",
        "give",
        "list",
        "match",
        "matches",
        "mean",
        "please",
        "say",
        "says",
        "show",
        "summarize",
        "summary",
        "tell",
        "versus",
        "vs",
        # corpus-generic nouns — present in (or applicable to) nearly every
        # indexed doc, so they can't anchor a topic either
        "agreement",
        "agreements",
        "bank",
        "client",
        "clients",
        "customer",
        "doc",
        "docs",
        "document",
        "documents",
        "faq",
        "policy",
        "policies",
        "service",
        "services",
        "sla",
        "slas",
        "team",
        "teams",
        # frame prepositions/adverbs — they structure the ask, never name a
        # topic ("under the X SLA", "between A and B", "per the agreement")
        "about",
        "above",
        "across",
        "after",
        "against",
        "along",
        "among",
        "around",
        "before",
        "behind",
        "below",
        "between",
        "beyond",
        "despite",
        "during",
        "except",
        "inside",
        "near",
        "onto",
        "outside",
        "over",
        "per",
        "regarding",
        "since",
        "through",
        "throughout",
        "till",
        "toward",
        "towards",
        "under",
        "until",
        "upon",
        "via",
        "within",
        "without",
        # qualifier adjectives/adverbs — question fillers that carry no
        # topic ("the annual fee", "recent trades", "the current cutoff").
        # Leaving them out makes unanswerable topics ('crypto custody')
        # distinguishable from filler words absent from the corpus
        "annual",
        "annually",
        "any",
        "average",
        "certain",
        "current",
        "each",
        "exact",
        "first",
        "general",
        "latest",
        "main",
        "monthly",
        "new",
        "normal",
        "old",
        "particular",
        "primary",
        "recent",
        "same",
        "second",
        "some",
        "specific",
        "standard",
        "total",
        "typical",
        "usual",
        "various",
        # status/count/relation frame words — "open tickets", "how many",
        # "who owns the account" describe the shape of the ask, not a
        # corpus topic
        "count",
        "counts",
        "hold",
        "holds",
        "many",
        "much",
        "number",
        "numbers",
        "open",
        "opens",
        "own",
        "owned",
        "owner",
        "owners",
        "owns",
    }
)


def load_prompt(name: str) -> str:
    """Read and return the prompt template file ``name`` from the prompts dir."""
    return (PROMPTS_DIR / name).read_text()


def audit(state, event_type: str, payload: dict, latency_ms: float = 0.0) -> None:
    """Write an audit event for the current request using ids and user from state."""
    get_audit().log_event(
        request_id=state.get("request_id", ""),
        trace_id=state.get("trace_id", ""),
        user_id=state["principal"].user_id,
        event_type=event_type,
        payload=payload,
        latency_ms=latency_ms,
    )


def add_error(state, message: str) -> list[str]:
    """Return ``state['errors']`` with a PII-sanitized ``message`` appended.

    Errors are surfaced in API diagnostics, so they must never carry raw
    exception payloads containing user or client data — the redactor is
    applied defensively even though exception strings *should* be safe.
    """
    from app.core.dependencies import get_pii_redactor

    errors = list(state.get("errors", []))
    errors.append(get_pii_redactor().redact(str(message))[:300])
    return errors


class Timer:
    """Context manager that measures elapsed wall-clock time in ``self.ms``."""

    def __enter__(self):
        """Record the start time and return the timer instance."""
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        """Compute elapsed milliseconds into ``self.ms``; never suppresses errors."""
        self.ms = (time.perf_counter() - self.start) * 1000
        return False
