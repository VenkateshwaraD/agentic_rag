"""Document loaders for synthetic and real corpora.

Supported: .md, .txt, .csv, .pdf, .docx. Each loader returns (text, doc_id_hint).
"""

import csv
from pathlib import Path

from app.core.errors import AppError
from app.core.logging import get_logger

log = get_logger(__name__)

SUPPORTED = {".md", ".txt", ".csv", ".pdf", ".docx"}


def load_document(path: Path) -> str:
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
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append("; ".join(f"{k}: {v}" for k, v in row.items()))
    return "\n".join(rows)


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _load_docx(path: Path) -> str:
    import docx

    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def iter_documents(root: Path):
    """Yield supported files under a directory, sorted for determinism."""
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED:
            yield p
