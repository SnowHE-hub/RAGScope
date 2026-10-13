import json
import numpy as np
import pytest
from ragscope.indexing.array_index import ArrayIndex, ArrayCache


def chunks():
    return [{"chunk_id": "c1", "doc_id": "d1", "text": "cats feline pets"},
            {"chunk_id": "c2", "doc_id": "d2", "text": "rare quasar physics"},
            {"chunk_id": "c3", "doc_id": "d1", "text": "trees green forest"}]


def test_index_roundtrip_scope_and_model_identity(tmp_path):
    identity = {"embedding": {"model": "A", "revision": "one"}, "corpus": "one", "scope": "all"}
    vectors = np.array([[1, 0], [0, 1], [.8, .6]], dtype=np.float32)
    index = ArrayIndex(chunks(), identity, vectors)
    index.save(tmp_path)
    loaded = ArrayIndex.load(tmp_path, identity)
    assert loaded.search("quasar", scope="d1")[0]["doc_id"] == "d1"
    hits = loaded.search("quasar", method="hybrid", query_vector=[0, 1], embedding_identity=identity["embedding"])
    assert hits[0]["chunk_id"] == "c2"
    with pytest.raises(ValueError, match="space"):
        loaded.search("quasar", method="dense", query_vector=[0, 1], embedding_identity={"model": "B"})
    assert loaded.search("anything", scope="absent") == []
    with pytest.raises(ValueError, match="identity"):
        ArrayIndex.load(tmp_path, {**identity, "corpus": "different"})
    (tmp_path / "chunks.json").write_text("[]")
    with pytest.raises(ValueError, match="validation"):
        ArrayIndex.load(tmp_path, identity)


def test_array_cache_namespace_and_corruption(tmp_path):
    first = ArrayCache(tmp_path, {"model": "a"})
    second = ArrayCache(tmp_path, {"model": "b"})
    first.put("text", [1, 0])
    assert np.array_equal(first.get("text"), [1, 0])
    assert second.get("text") is None
    with pytest.raises(ValueError):
        first.put("nan", [float("nan")])
    path = first.path / (first.key("text") + ".npy")
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="checksum"):
        first.get("text")


def test_rrf_uses_50_candidates_not_2k():
    data = [{"chunk_id": f"c{i:02}", "doc_id": "d", "text": f"token{i}"} for i in range(60)]
    identity = {"embedding": {"id": "unit"}}
    vectors = np.stack([np.array([i], dtype=np.float32) for i in range(60)])
    index = ArrayIndex(data, identity, vectors)
    hits = index.search("token5", method="hybrid", query_vector=[1], embedding_identity=identity["embedding"], k=50)
    assert len(hits) == 50
    assert "c05" in [x["chunk_id"] for x in hits]
