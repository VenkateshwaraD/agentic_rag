"""Unit tests for the regex-based PII redactor (offline fallback).

Author: venkateshwara doijode
"""

from dataclasses import dataclass

from app.security.pii import RegexRedactor


def test_regex_redactor_email():
    """Verify email addresses are replaced with the ``<EMAIL_ADDRESS>`` token."""
    r = RegexRedactor()
    out = r.redact("contact maria.delgado@acme.example.com now")
    assert "@" not in out and "<EMAIL_ADDRESS>" in out


def test_regex_redactor_phone_and_ssn():
    """Verify phone numbers and SSNs are replaced with their placeholder tokens."""
    r = RegexRedactor()
    out = r.redact("call +1-212-555-0148, ssn 123-45-6789")
    assert "<PHONE_NUMBER>" in out and "<US_SSN>" in out


def test_contains_pii():
    """Verify ``contains_pii`` flags text with PII and passes clean text."""
    r = RegexRedactor()
    assert r.contains_pii("email x@y.com")
    assert not r.contains_pii("no pii here")


@dataclass
class _Hit:
    entity_type: str
    start: int
    end: int
    score: float


class _StubAnalyzer:
    """Minimal stand-in for the Presidio analyzer (no spaCy model needed)."""

    def __init__(self, hits):
        self._hits = hits

    def analyze(self, text, entities, language):
        return self._hits


def _presidio_with(hits):
    from app.security.pii import PresidioRedactor

    r = PresidioRedactor.__new__(PresidioRedactor)
    r._analyzer = _StubAnalyzer(hits)
    return r


def test_contains_pii_ignores_below_floor_detections():
    """A weak-shape hit ('382914' inside 'amount=382914.49' → PHONE_NUMBER
    @0.4) must not block a legitimate draft — the floor only drops
    detections too weak to refuse an answer on."""
    text = "trade_id=TRD-00073, amount=382914.49, currency=USD"
    weak = _Hit("PHONE_NUMBER", text.index("382914"), text.index("382914") + 6, 0.4)
    assert not _presidio_with([weak]).contains_pii(text)


def test_contains_pii_blocks_confident_detection():
    """A detection at/above the floor still blocks the draft."""
    text = "reach maria.delgado@acme.example.com"
    hit = _Hit("EMAIL_ADDRESS", text.index("maria"), len(text), 0.95)
    assert _presidio_with([hit]).contains_pii(text)


def test_contains_pii_ignores_spans_a_preserved_id_claims():
    """'TRD-00073' read as PERSON loses to the TRADE_ID recognizer's span
    — citation ids are business keys, not PII, even at high confidence."""
    text = "trade_id=TRD-00073, status=failed"
    s, e = text.index("TRD-00073"), text.index("TRD-00073") + len("TRD-00073")
    hits = [
        _Hit("TRADE_ID", s, e, 0.9),
        _Hit("PERSON", s, e, 0.85),
    ]
    assert not _presidio_with(hits).contains_pii(text)
