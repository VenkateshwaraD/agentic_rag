"""File validation stage — detect corrupted, empty, and unsupported inputs.

Runs before parsing (extension/size pre-checks) and after parsing
(whitespace-only content). Results feed the ingestion pipeline's
skip/dead-letter decisions.

Author: venkateshwara doijode
"""

from dataclasses import dataclass
from pathlib import Path

from app.data.ingest.loaders import SUPPORTED, RawDocument


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of file/document validation: ``ok`` or a skip/fail reason."""

    status: str  # ok | unsupported | empty | corrupted
    detail: str = ""


def validate_path(path: Path) -> ValidationResult:
    """Pre-parse validation: extension allowlist + zero-byte detection."""
    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        return ValidationResult("unsupported", f"extension '{ext}' not supported")
    if not path.is_file() or path.stat().st_size == 0:
        return ValidationResult("empty", "missing or zero-byte file")
    return ValidationResult("ok")


def validate_document(raw: RawDocument) -> ValidationResult:
    """Post-parse validation: whitespace-only content counts as empty."""
    if not raw.text.strip():
        return ValidationResult("empty", "document produced no text")
    return ValidationResult("ok")
