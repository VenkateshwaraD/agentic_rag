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
    meta = parse_metadata_comment(DOC)
    assert meta["doc_id"] == "AGR-001"
    assert meta["client_id"] == "acme"
    assert meta["effective_date"] == "2024-01-01"


def test_chunk_document_metadata():
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


def test_chunk_document_bank_wide():
    doc = "<!-- doc_id: SLA-001 -->\n\n## §1 Cutoffs\nWire cutoff is 16:00 ET."
    chunks = chunk_document(doc, Path("SLA-001_x.md"), "slas")
    assert chunks[0].entitlements["clients"] == "*"
