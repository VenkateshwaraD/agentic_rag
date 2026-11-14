"""Unit tests for document metadata parsing and section chunking.

Author: venkateshwara doijode
"""

from pathlib import Path

from app.data.ingest.chunking import (
    chunk_document,
    parse_metadata_comment,
)

DOC = """# Client Services Agreement — Acme

<!-- doc_id: AGR-001 | client_id: acme | version: 2.0 | effective_date: 2024-01-01 -->

## §1 Parties
Body one.

## §2 Fees
Wire fee is $20.00.
"""


def test_parse_metadata_comment():
    """Verify the HTML metadata comment is parsed into key/value fields."""
    meta = parse_metadata_comment(DOC)
    assert meta["doc_id"] == "AGR-001"
    assert meta["client_id"] == "acme"
    assert meta["effective_date"] == "2024-01-01"


def test_chunk_document_metadata():
    """Verify chunking produces per-section chunks with correct metadata/entitlements."""
    chunks = chunk_document(DOC, Path("AGR-001_acme.md"), "agreements")
    assert len(chunks) == 2
    c = chunks[0]
    assert c.doc_id == "AGR-001"
    assert c.source == "agreements"
    assert c.client_id == "acme"
    assert c.date == "2024-01-01"
    assert c.doc_version == "2.0"
    assert c.section.startswith("§")
    # client-scoped entitlements
    assert c.entitlements["clients"] == ["acme"]
    assert "agent" in c.entitlements["roles"]


def test_chunk_document_doc_title():
    """The H1 title rides on every chunk — metadata AND a searchable first
    text line, so questions naming a document can retrieve it."""
    chunks = chunk_document(DOC, Path("AGR-001_acme.md"), "agreements")
    for c in chunks:
        assert c.doc_title == "Client Services Agreement — Acme"
        assert c.text.startswith("Client Services Agreement — Acme\n")


def test_chunk_document_doc_title_fallback():
    """Non-markdown loads (PDF/DOCX/TXT) have no H1 — the first
    title-shaped content line becomes doc_title; CSV headers don't."""
    txt = "<!-- doc_id: POL-101 -->\nClient Service Desk — Holiday Coverage\n\nRow one.\n"
    chunks = chunk_document(txt, Path("POL-101_x.txt"), "service_teams")
    assert chunks[0].doc_title == "Client Service Desk — Holiday Coverage"
    csv = "<!-- doc_id: SVC-101 -->\nteam,coverage_window,region\nOps,18:00-08:00,us\n"
    chunks = chunk_document(csv, Path("SVC-101_x.csv"), "service_teams")
    assert chunks[0].doc_title == ""


def test_chunk_document_bank_wide():
    """Verify docs without a client_id get wildcard (bank-wide) entitlements."""
    doc = "<!-- doc_id: SLA-001 -->\n\n## §1 Cutoffs\nWire cutoff is 16:00 ET."
    chunks = chunk_document(doc, Path("SLA-001_x.md"), "slas")
    assert chunks[0].entitlements["clients"] == "*"
