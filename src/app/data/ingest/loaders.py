"""Document loaders for synthetic and real corpora.

Supported: .md, .txt, .csv, .pdf, .docx. ``parse_document`` wraps the raw
loaders and returns a ``RawDocument`` — parse failures raise
``CorruptedDocumentError`` so the ingestion pipeline can dead-letter
corrupted files distinctly from unsupported or empty ones.

Author: venkateshwara doijode
"""

import csv
from dataclasses import dataclass
from pathlib import Path

from app.core.errors import AppError
from app.core.logging import get_logger

log = get_logger(__name__)

SUPPORTED = {".md", ".txt", ".csv", ".pdf", ".docx"}


@dataclass
class RawDocument:
    """A parsed source document: filesystem path plus extracted text."""

    path: Path
    text: str


class CorruptedDocumentError(AppError):
    """Raised when a supported file type fails to parse (truncated/binary/etc.)."""


def parse_document(path: Path) -> RawDocument:
    """Load a supported document and return it as a ``RawDocument``.

    ``AppError`` (unsupported type) propagates unchanged; any other loader
    failure is wrapped in ``CorruptedDocumentError`` for dead-lettering.
    """
    try:
        return RawDocument(path=path, text=load_document(path))
    except AppError:
        raise
    except Exception as exc:
        raise CorruptedDocumentError(f"{path.name}: {exc}") from exc


def load_document(path: Path) -> str:
    """Dispatch to the per-extension loader and return the document's text.

    Raises ``AppError`` for unsupported file types.
    """
    ext = path.suffix.lower()
    if ext in (".md", ".txt"):
        return path.read_text(encoding="utf-8")
    if ext == ".csv":
        return _load_csv(path)
    if ext == ".pdf":
        return _load_pdf(path)
    if ext == ".docx":
        return _load_docx(path)
    raise AppError(f"Unsupported document type: {ext} ({path})")


def _load_csv(path: Path) -> str:
    """Flatten a CSV into one ``key: value;`` line per row for indexing."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append("; ".join(f"{k}: {v}" for k, v in row.items()))
    return "\n".join(rows)


def _load_pdf(path: Path) -> str:
    """Extract text per page, emitting ``<!-- page: N -->`` markers so chunk
    metadata can preserve page numbers."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts = []
    for i, page in enumerate(reader.pages, start=1):
        parts.append(f"<!-- page: {i} -->\n{page.extract_text() or ''}")
    return "\n".join(parts)


def _load_docx(path: Path) -> str:
    """Join the paragraph texts of a .docx file via python-docx."""
    import docx

    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def iter_documents(root: Path):
    """Yield supported files under a directory, sorted for determinism."""
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED:
            yield p
