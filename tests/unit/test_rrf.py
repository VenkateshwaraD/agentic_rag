from app.retrieval.hybrid import rrf_fuse


def _item(doc_id, section="§1"):
    return {"doc_id": doc_id, "section": section, "text": "x"}


def test_rrf_dedup_and_order():
    dense = [_item("A"), _item("B"), _item("C")]
    bm25 = [_item("B"), _item("D"), _item("A")]
    fused = rrf_fuse([dense, bm25], k=60)
    ids = [f["doc_id"] for f in fused]
    # B ranks 2nd in both lists → highest fused score
    assert ids[0] == "B"
    assert sorted(ids) == ["A", "B", "C", "D"]
    assert all("rrf_score" in f for f in fused)


def test_rrf_single_list():
    fused = rrf_fuse([[_item("A"), _item("B")]], k=60)
    assert [f["doc_id"] for f in fused] == ["A", "B"]
