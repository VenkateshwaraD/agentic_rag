"""Deterministic query classification for retrieval strategy selection.

Decides whether a question is best served by dense (semantic), keyword
(BM25), or both retrieval legs when ``retrieval_strategy == "auto"``:

- record ids (TKT-/TRD-/DOC-style), quoted phrases, and id/number-heavy
  questions lean KEYWORD — exact-match lookup territory;
- purely conceptual natural-language questions lean DENSE;
- questions mixing identifiers with descriptive phrasing need BOTH legs.

Author: venkateshwara doijode
"""

import re
from enum import StrEnum

_RECORD_ID_RE = re.compile(r"\b[A-Z]{2,5}-\d{3,}\b")
_QUOTED_RE = re.compile(r"['\"][^'\"]+['\"]")
_EXACT_NUM_RE = re.compile(r"\$\s?\d|\b\d+[.,]?\d*\s?(%|days?|usd|dollars?)\b", re.I)
_WORD_RE = re.compile(r"[a-zA-Z]+")


class QueryMode(StrEnum):
    """Retrieval leg(s) a query should use when strategy resolution is 'auto'."""

    DENSE = "dense"
    KEYWORD = "keyword"
    BOTH = "both"


def classify(query: str) -> QueryMode:
    """Classify a question into ``DENSE``, ``KEYWORD``, or ``BOTH``."""
    has_id = bool(_RECORD_ID_RE.search(query) or _QUOTED_RE.search(query))
    has_exact_num = bool(_EXACT_NUM_RE.search(query))
    descriptive_words = len(_WORD_RE.findall(query))

    if has_id and descriptive_words > 8:
        # identifier plus a real question around it — need both legs
        return QueryMode.BOTH
    if has_id:
        return QueryMode.KEYWORD
    if has_exact_num and descriptive_words <= 8:
        return QueryMode.KEYWORD
    return QueryMode.DENSE


# QueryMode → concrete strategy the hybrid pipeline executes.
MODE_TO_STRATEGY = {
    QueryMode.DENSE: "dense",
    QueryMode.KEYWORD: "bm25",
    QueryMode.BOTH: "hybrid_rerank",
}
