"""Unit tests for citation-marker extraction from drafted replies.

Author: venkateshwara doijode
"""

from app.agent.nodes.synthesize import _extract_citations


def test_extract_citations():
    """Verify ``[DOC-ID §section]`` markers are parsed into citation dicts."""
    draft = "Wire fee is $20.00 [AGR-001 §2]. Cutoff is 16:00 [SLA-003 §1]."
    cits = _extract_citations(draft, [])
    assert {c["doc_id"] for c in cits} == {"AGR-001", "SLA-003"}
    assert cits[0]["section"] == "2"


def test_citations_enriched_with_versioned_metadata():
    """Verify citations pick up version/effective-date from the retrieved chunk."""
    draft = "Wire fee is $20.00 [AGR-001 §2]."
    retrieved = [
        {
            "doc_id": "AGR-001",
            "section": "2",
            "chunk_id": "AGR-001:v2:fees:0",
            "doc_version": "2.0",
            "effective_date": "2024-01-01",
        }
    ]
    cits = _extract_citations(draft, retrieved)
    assert cits[0]["doc_version"] == "2.0"
    assert cits[0]["effective_date"] == "2024-01-01"
    assert cits[0]["chunk_id"] == "AGR-001:v2:fees:0"


def test_no_citations():
    """Verify text without markers yields an empty citation list."""
    assert _extract_citations("no citations here", []) == []
