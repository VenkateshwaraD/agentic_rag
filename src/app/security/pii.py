"""PII redaction — Presidio primary, regex fallback, explicit entity policy.

build_redactor() tries Presidio (needs a spaCy model); if unavailable it
falls back to regex recognizers so the pipeline still runs offline.
Redaction is irreversible: typed placeholders, no mapping persisted.

Entity policy — two disjoint sets:
- ``REDACT_ENTITIES``: personal PII and sensitive business identifiers
  (account/employee ids) — masked everywhere;
- ``PRESERVE_ENTITIES``: lookup/citation identifiers (TKT-/TRD-/DOC-style
  ids and client ids) — detected for measurement but never masked, because
  redacting them would destroy retrieval keys and citation resolution.

Author: venkateshwara doijode
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache

from app.core.logging import get_logger

log = get_logger(__name__)

# Entities that are always masked.
REDACT_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "IBAN_CODE",
    "ACCOUNT_NUMBER",
    "EMPLOYEE_ID",
]

# Entities detected for measurement/audit but never masked — they are the
# retrieval and citation keys the application depends on.
PRESERVE_ENTITIES = ["TICKET_ID", "TRADE_ID", "DOC_ID", "CLIENT_ID"]

ENTITIES = REDACT_ENTITIES  # backwards-compat alias used by tests/eval


@dataclass(frozen=True)
class CustomPattern:
    """A business-identifier pattern: regex, entity name, and mask policy."""

    entity: str
    regex: str
    redact: bool


CUSTOM_PATTERNS: list[CustomPattern] = [
    CustomPattern("TICKET_ID", r"\bTKT-\d{3,}\b", redact=False),
    CustomPattern("TRADE_ID", r"\bTRD-\d{3,}\b", redact=False),
    CustomPattern("DOC_ID", r"\b(?:AGR|SLA|SVC|FAQ|DOC)-\d{3,}\b", redact=False),
    CustomPattern("CLIENT_ID", r"\b[a-z]{4,}(?=\s+client\b)", redact=False),
    CustomPattern("ACCOUNT_NUMBER", r"\bACC-\d{6,}\b", redact=True),
    CustomPattern("EMPLOYEE_ID", r"\bEMP-\d{4,}\b", redact=True),
]


def build_custom_recognizers() -> list:
    """Return Presidio ``PatternRecognizer``s for the business-identifier table.

    Each recognizer is registered under its entity name; the redact/preserve
    policy is applied separately — preserved entities are detected (they
    show up in ``analyze()``) but excluded from anonymization.
    """
    from presidio_analyzer import Pattern, PatternRecognizer

    recognizers = []
    for cp in CUSTOM_PATTERNS:
        recognizers.append(
            PatternRecognizer(
                supported_entity=cp.entity,
                patterns=[Pattern(cp.entity, cp.regex, 0.9)],
                supported_language="en",
            )
        )
    return recognizers


class PIIRedactor(ABC):
    """Interface for PII detection/redaction backends."""

    @abstractmethod
    def redact(self, text: str) -> str:
        """Return ``text`` with redactable entities replaced by typed placeholders."""
        ...

    @abstractmethod
    def analyze(self, text: str) -> list[dict]:
        """Return ``[{entity, span, start, end}]`` for every detected entity —
        redactable *and* preserved — used by the redaction eval harness."""
        ...

    @abstractmethod
    def contains_pii(self, text: str) -> bool:
        """Return True when ``text`` contains an entity that would be redacted."""
        ...


class PresidioRedactor(PIIRedactor):
    """Presidio-based redactor; masks redactable entities with ``<TYPE>`` tokens."""

    def __init__(self):
        """Initialize Presidio with built-in + custom business-id recognizers."""
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        from presidio_anonymizer.entities import OperatorConfig

        self._analyzer = AnalyzerEngine()
        for recognizer in build_custom_recognizers():
            self._analyzer.registry.add_recognizer(recognizer)
        self._anonymizer = AnonymizerEngine()
        self._operators = {
            e: OperatorConfig("replace", {"new_value": f"<{e}>"}) for e in REDACT_ENTITIES
        }

    def analyze(self, text: str) -> list[dict]:
        """Detect every entity (redactable + preserved) and return span dicts."""
        results = self._analyzer.analyze(
            text=text, entities=REDACT_ENTITIES + PRESERVE_ENTITIES, language="en"
        )
        return [
            {
                "entity": r.entity_type,
                "span": text[r.start : r.end],
                "start": r.start,
                "end": r.end,
            }
            for r in results
        ]

    def redact(self, text: str) -> str:
        """Analyze for redactable entities only and replace each match."""
        results = self._analyzer.analyze(text=text, entities=REDACT_ENTITIES, language="en")
        if not results:
            return text
        return self._anonymizer.anonymize(
            text=text, analyzer_results=results, operators=self._operators
        ).text

    def contains_pii(self, text: str) -> bool:
        """Return True when Presidio detects any redactable entity in the text."""
        return bool(self._analyzer.analyze(text=text, entities=REDACT_ENTITIES, language="en"))


class RegexRedactor(PIIRedactor):
    """Offline fallback redactor — same entity policy via regex tables.

    ``REDACT_PATTERNS`` are masked by ``redact()``; ``PRESERVE_PATTERNS``
    are only used by ``analyze()`` for measurement and never masked.
    """

    REDACT_PATTERNS = [
        (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "EMAIL_ADDRESS"),
        (re.compile(r"\b\+?1?[-. (]*\d{3}[-. )]*\d{3}[-. ]*\d{4}\b"), "PHONE_NUMBER"),
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "US_SSN"),
        (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "CREDIT_CARD"),
        (re.compile(r"\bACC-\d{6,}\b"), "ACCOUNT_NUMBER"),
        (re.compile(r"\bEMP-\d{4,}\b"), "EMPLOYEE_ID"),
    ]
    PRESERVE_PATTERNS = [
        (re.compile(cp.regex), cp.entity) for cp in CUSTOM_PATTERNS if not cp.redact
    ]

    def redact(self, text: str) -> str:
        """Replace every redactable pattern match with ``<ENTITY>`` tokens."""
        out = text
        for pat, entity in self.REDACT_PATTERNS:
            out = pat.sub(f"<{entity}>", out)
        return out

    def analyze(self, text: str) -> list[dict]:
        """Return span dicts for every detected entity (redact + preserve)."""
        findings = []
        for pat, entity in self.REDACT_PATTERNS + self.PRESERVE_PATTERNS:
            for m in pat.finditer(text):
                findings.append(
                    {"entity": entity, "span": m.group(0), "start": m.start(), "end": m.end()}
                )
        return findings

    def contains_pii(self, text: str) -> bool:
        """Return True when any redactable regex matches the text."""
        return any(p.search(text) for p, _ in self.REDACT_PATTERNS)


@lru_cache(maxsize=1)
def build_redactor() -> PIIRedactor:
    """Construct the best available redactor: Presidio if usable, else regex fallback.

    Cached — engine construction loads ~30 recognizers and takes seconds;
    callers (audit payloads, prompt redaction, SSE deltas) share one engine.
    """
    try:
        r = PresidioRedactor()
        r.contains_pii("warmup test@example.com")  # smoke: spaCy model load check
        log.info("pii.redactor", engine="presidio")
        return r
    except Exception as exc:  # spaCy model missing, etc.
        log.warning("pii.redactor.fallback", engine="regex", reason=str(exc))
        return RegexRedactor()
