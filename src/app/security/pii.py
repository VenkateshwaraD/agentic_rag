"""PII redaction — Presidio primary, regex fallback.

build_redactor() tries Presidio (needs a spaCy model); if unavailable it
falls back to regex recognizers so the pipeline still runs offline.
Redaction is irreversible: typed placeholders, no mapping persisted.
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
    @abstractmethod
    def redact(self, text: str) -> str: ...

    @abstractmethod
    def contains_pii(self, text: str) -> bool: ...


class PresidioRedactor(PIIRedactor):
    def __init__(self):
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        from presidio_anonymizer.entities import OperatorConfig

        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()
        self._operators = {e: OperatorConfig("replace", {"new_value": f"<{e}>"}) for e in ENTITIES}

    def redact(self, text: str) -> str:
        results = self._analyzer.analyze(text=text, entities=ENTITIES, language="en")
        if not results:
            return text
        return self._anonymizer.anonymize(
            text=text, analyzer_results=results, operators=self._operators
        ).text

    def contains_pii(self, text: str) -> bool:
        return bool(self._analyzer.analyze(text=text, entities=ENTITIES, language="en"))


class RegexRedactor(PIIRedactor):
    PATTERNS = [
        (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "<EMAIL_ADDRESS>"),
        (re.compile(r"\b\+?1?[-. (]*\d{3}[-. )]*\d{3}[-. ]*\d{4}\b"), "<PHONE_NUMBER>"),
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "<US_SSN>"),
        (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "<CREDIT_CARD>"),
    ]

    def redact(self, text: str) -> str:
        out = text
        for pat, repl in self.PATTERNS:
            out = pat.sub(repl, out)
        return out

    def contains_pii(self, text: str) -> bool:
        return any(p.search(text) for p, _ in self.PATTERNS)


def build_redactor() -> PIIRedactor:
    try:
        r = PresidioRedactor()
        r.contains_pii("warmup test@example.com")  # smoke: spaCy model load check
        log.info("pii.redactor", engine="presidio")
        return r
    except Exception as exc:  # spaCy model missing, etc.
        log.warning("pii.redactor.fallback", engine="regex", reason=str(exc))
        return RegexRedactor()
