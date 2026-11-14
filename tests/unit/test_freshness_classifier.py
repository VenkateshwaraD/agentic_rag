"""Unit tests for freshness filtering, query classification, text normalization.

Author: venkateshwara doijode
"""

from app.data.ingest.normalize import normalize_text
from app.retrieval.freshness import is_expired, os_freshness_filter, pg_freshness_where
from app.retrieval.query_classifier import MODE_TO_STRATEGY, QueryMode, classify


class TestIsExpired:
    """expiry_date semantics: past = expired; absent/empty/future = fresh."""

    def test_past_date_is_expired(self):
        assert is_expired({"expiry_date": "2020-01-01"}, today="2025-01-01")

    def test_future_date_is_fresh(self):
        assert is_expired({"expiry_date": "2099-01-01"}, today="2025-01-01") is False

    def test_today_boundary_is_fresh(self):
        # expiry_date == today is still valid (>= today kept by the stores)
        assert is_expired({"expiry_date": "2025-01-01"}, today="2025-01-01") is False

    def test_missing_and_blank_are_fresh(self):
        assert is_expired({}, today="2025-01-01") is False
        assert is_expired({"expiry_date": ""}, today="2025-01-01") is False
        assert is_expired({"expiry_date": "  "}, today="2025-01-01") is False

    def test_default_today(self):
        assert is_expired({"expiry_date": "2000-01-01"})


class TestFreshnessFilters:
    """exclude mode emits filters; any other mode emits nothing."""

    def test_pg_clause_only_in_exclude(self):
        clause, params = pg_freshness_where("exclude")
        assert "expiry_date" in clause and "fresh_today" in params
        assert pg_freshness_where("down_rank") == ("", {})

    def test_os_clause_only_in_exclude(self):
        f = os_freshness_filter("exclude")
        assert f["bool"]["minimum_should_match"] == 1
        assert os_freshness_filter("down_rank") is None


class TestClassify:
    """auto-strategy query classification."""

    def test_record_id_only_is_keyword(self):
        assert classify("TKT-00001") == QueryMode.KEYWORD

    def test_id_plus_prose_is_both(self):
        q = "What is the current status of ticket TKT-00001 and when was it last updated?"
        assert classify(q) == QueryMode.BOTH

    def test_quoted_phrase_is_keyword(self):
        assert classify('"same-day cutoff"') == QueryMode.KEYWORD

    def test_exact_amount_short_is_keyword(self):
        assert classify("fee $12.50") == QueryMode.KEYWORD

    def test_conceptual_question_is_dense(self):
        assert classify("how are wire transfer disputes usually resolved?") == QueryMode.DENSE

    def test_strategy_mapping_covers_all_modes(self):
        assert set(MODE_TO_STRATEGY) == set(QueryMode)
        assert MODE_TO_STRATEGY[QueryMode.BOTH] == "hybrid_rerank"


class TestNormalizeText:
    """BOM/CRLF/trailing-whitespace cleanup, idempotent."""

    def test_bom_and_crlf(self):
        assert normalize_text("\ufeffa\r\nb\r\n") == "a\nb\n"

    def test_trailing_ws_trimmed_per_line(self):
        assert normalize_text("a   \nb\t\n") == "a\nb\n"

    def test_idempotent(self):
        t = normalize_text("\ufeffx \r\ny\t\r\n")
        assert normalize_text(t) == t

    def test_empty(self):
        assert normalize_text("") == ""
