"""PII redaction — Presidio primary, regex fallback.

build_redactor() tries Presidio (needs a spaCy model); if unavailable it
falls back to regex recognizers so the pipeline still runs offline.
Redaction is irreversible: typed placeholders, no mapping persisted.

Author: venkateshwara doijode
"""

import re
from abc import ABC, abstractmethod

from app.core.logging import get_logger

log = get_logger(__name__)

ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "IBAN_CODE",
]


class PIIRedactor(ABC):
    """Interface for PII detection/redaction backends."""

    @abstractmethod
    def redact(self, text: str) -> str:
        """Return ``text`` with PII entities replaced by typed placeholders."""
        ...

    @abstractmethod
    def contains_pii(self, text: str) -> bool:
        """Return True when ``text`` appears to contain PII."""
        ...


class PresidioRedactor(PIIRedactor):
    """Presidio-based redactor; replaces detected entities with ``<TYPE>`` tokens."""

    def __init__(self):
        """Initialize the Presidio analyzer/anonymizer and per-entity replace operators."""
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        from presidio_anonymizer.entities import OperatorConfig

        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()
        self._operators = {e: OperatorConfig("replace", {"new_value": f"<{e}>"}) for e in ENTITIES}

    def redact(self, text: str) -> str:
        """Analyze the text for the configured entities and replace each match."""
        results = self._analyzer.analyze(text=text, entities=ENTITIES, language="en")
        if not results:
            return text
        return self._anonymizer.anonymize(
            text=text, analyzer_results=results, operators=self._operators
        ).text

    def contains_pii(self, text: str) -> bool:
        """Return True when Presidio detects any configured entity in the text."""
        return bool(self._analyzer.analyze(text=text, entities=ENTITIES, language="en"))


class RegexRedactor(PIIRedactor):
    """Offline fallback redactor using regexes for email/phone/SSN/card."""

    PATTERNS = [
        (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "<EMAIL_ADDRESS>"),
        (re.compile(r"\b\+?1?[-. (]*\d{3}[-. )]*\d{3}[-. ]*\d{4}\b"), "<PHONE_NUMBER>"),
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "<US_SSN>"),
        (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "<CREDIT_CARD>"),
    ]

    def redact(self, text: str) -> str:
        """Replace every pattern match in the text with its placeholder token."""
        out = text
        for pat, repl in self.PATTERNS:
            out = pat.sub(repl, out)
        return out

    def contains_pii(self, text: str) -> bool:
        """Return True when any PII regex matches the text."""
        return any(p.search(text) for p, _ in self.PATTERNS)


def build_redactor() -> PIIRedactor:
    """Construct the best available redactor: Presidio if usable, else regex fallback."""
    try:
        r = PresidioRedactor()
        r.contains_pii("warmup test@example.com")  # smoke: spaCy model load check
        log.info("pii.redactor", engine="presidio")
        return r
    except Exception as exc:  # spaCy model missing, etc.
        log.warning("pii.redactor.fallback", engine="regex", reason=str(exc))
        return RegexRedactor()
