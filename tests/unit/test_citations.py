from app.agent.nodes.synthesize import _extract_citations


def test_extract_citations():
    draft = "Wire fee is $20.00 [AGR-001 §2]. Cutoff is 16:00 [SLA-003 §1]."
    cits = _extract_citations(draft)
    assert {c["doc_id"] for c in cits} == {"AGR-001", "SLA-003"}
    assert cits[0]["section"] == "2"


def test_no_citations():
    assert _extract_citations("no citations here") == []
