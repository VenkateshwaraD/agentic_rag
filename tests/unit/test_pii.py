"""Unit tests for the regex-based PII redactor (offline fallback).

Author: venkateshwara doijode
"""

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
