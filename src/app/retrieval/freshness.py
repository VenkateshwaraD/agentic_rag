"""Document freshness / expiry filtering, applied symmetrically in both stores.

``doc_expiry_mode`` controls behavior at search time:
- ``exclude`` — expired docs are filtered out in the same query that applies
  ACLs (pre-filter, identical semantics in pgvector and OpenSearch);
- ``down_rank`` — expired docs stay retrievable but are flagged ``stale``
  and penalized during fusion.

Author: venkateshwara doijode
"""

from datetime import date
from typing import Any


def _today() -> str:
    """Return today's ISO date string (comparable to stored expiry_date)."""
    return date.today().isoformat()


def is_expired(meta: dict[str, Any], today: str | None = None) -> bool:
    """Return True when ``meta['expiry_date']`` is set and in the past."""
    expiry = (meta.get("expiry_date") or "").strip()
    return bool(expiry) and expiry < (today or _today())


def pg_freshness_where(mode: str) -> tuple[str, dict[str, Any]]:
    """Return a (clause, params) pair filtering expired docs, or empty when not excluding."""
    if mode != "exclude":
        return "", {}
    # NULL / empty expiry means the document never expires
    return "AND (expiry_date IS NULL OR expiry_date = '' OR expiry_date >= :fresh_today)", {
        "fresh_today": _today()
    }


def os_freshness_filter(mode: str) -> dict | None:
    """Return an OpenSearch filter clause excluding expired docs, or None."""
    if mode != "exclude":
        return None
    return {
        "bool": {
            "should": [
                {"bool": {"must_not": {"exists": {"field": "expiry_date"}}}},
                {"term": {"expiry_date": ""}},
                {"range": {"expiry_date": {"gte": _today()}}},
            ],
            "minimum_should_match": 1,
        }
    }
