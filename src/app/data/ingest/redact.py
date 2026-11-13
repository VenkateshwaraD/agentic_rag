"""Ingest-time PII redaction hook — delegates to the security-layer redactor.

Redaction is irreversible by design: entities are replaced with typed
placeholders and no mapping is persisted, so the index can never leak raw PII.

Author: venkateshwara doijode
"""

from app.core.dependencies import get_pii_redactor


def redact_chunk_text(text: str) -> str:
    """Return ``text`` with PII entities replaced by typed placeholders."""
    return get_pii_redactor().redact(text)
