"""Numeric fixtures test exact scoped semantics; no real-model claims."""
import json

import numpy as np
import pytest

from ragscope.indexing.array_index import ArrayIndex
from ragscope.indexing.identity import artifact, atomic_json, signature
from ragscope.indexing.scoped_index import ENGINE, ScopedArrayIndex


@pytest.fixture
def scoped_fixture(tmp_path):
    texts = ["alpha alpha gamma", "beta gamma", "alpha beta", "gamma alpha alpha", "gamma gamma beta"]
    full = [{"chunk_id": f"c{i}", "doc_id": "paper" if i < 3 else "other", "text": text}
            for i, text in enumerate(texts)]
    path = tmp_path / "reference.jsonl"
    path.write_text("".join(json.dumps(c) + "\n" for c in full), encoding="utf-8")
    manifest = tmp_path / "chunk_manifest.json"
    source_identity = {"selected_doc_ids": ["other", "paper"], "fixture": "not_real_model"}
    atomic_json(manifest, {"identity": source_identity, "signature": signature(source_identity),
        "count": len(full), "artifacts": [artifact(path)]})
    source = {"chunks": artifact(path), "chunk_manifest": artifact(manifest),
        "chunk_signature": signature(source_identity), "count": len(full),
        "chunks_content_sha256": signature(full), "doc_ids": ["other", "paper"]}
    embedding = {"model": "offline_numeric_fixture", "dimensions": 3}
    identity = {"engine": ENGINE, "scope": "explicit_paper_only", "selected_doc_ids": ["paper"],
        "selected_doc_ids_sha256": signature(["paper"]), "sparse_reference": source,
        "embedding": embedding}
    vectors = np.asarray([[1, 0, 0], [0, 1, 0], [.6, .8, 0], [0, 0, 1], [.8, 0, .6]], dtype=np.float32)
    return full, identity, vectors


def test_scoped_bm25_dense_and_hybrid_keep_full_reference_rankings(scoped_fixture):
    full, identity, vectors = scoped_fixture
    reference = ArrayIndex(full, {"embedding": identity["embedding"]}, vectors)
    scoped = ScopedArrayIndex(full[:3], identity, vectors[:3])
    for method in ("bm25", "dense", "hybrid"):
        for question in ("alpha", "beta gamma", "missing term"):
            args = dict(scope="paper", method=method, k=3, candidate_depth=3,
                query_vector=np.asarray([.6, .8, 0], dtype=np.float32), embedding_identity=identity["embedding"])
            assert scoped.search(question, **args) == reference.search(question, **args)
    # This fixture also detects the invalid shortcut of recomputing subset IDF.
    subset = ArrayIndex(full[:3], {"embedding": identity["embedding"]}, vectors[:3])
    assert subset.bm25.get_scores(["alpha"]).tolist() != scoped.bm25.get_scores(["alpha"]).tolist()


def test_scoped_roundtrip_rejects_global_unknown_paper_and_changed_reference(tmp_path, scoped_fixture):
    full, identity, vectors = scoped_fixture
    index = ScopedArrayIndex(full[:3], identity, vectors[:3])
    target = tmp_path / "index"
    index.save(target)
    loaded = ScopedArrayIndex.load(target, identity)
    assert loaded.search("alpha", scope="paper") == index.search("alpha", scope="paper")
    for scope in (None, "other", "missing"):
        with pytest.raises(ValueError, match="global search is forbidden"):
            loaded.search("alpha", scope=scope)
    source = identity["sparse_reference"]["chunks"]["path"]
    from pathlib import Path
    Path(source).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="reference source changed"):
        ScopedArrayIndex.load(target, identity)


def test_scoped_membership_requires_all_same_paper_chunks(scoped_fixture):
    full, identity, _ = scoped_fixture
    with pytest.raises(ValueError, match="all unchanged reference chunks"):
        ScopedArrayIndex(full[:2], identity)
    changed = [{**full[0], "text": "changed"}, *full[1:3]]
    with pytest.raises(ValueError, match="all unchanged reference chunks"):
        ScopedArrayIndex(changed, identity)
    with pytest.raises(ValueError, match="membership/hash"):
        ScopedArrayIndex(full[:3], {**identity, "selected_doc_ids_sha256": "wrong"})


def test_scoped_dense_campaign_resume_uses_specialized_loader(tmp_path, scoped_fixture):
    from ragscope.experiments.retrieval_campaign import verified_dense_index
    full, identity, vectors = scoped_fixture

    class OfflineFixture:
        calls = 0
        last_metrics = []

        def embed(self, texts):
            self.calls += 1
            self.last_metrics = [{"input_tokens": len(t), "cache_hit": False, "seconds": 0} for t in texts]
            return vectors[:len(texts)]

    embedding = OfflineFixture()
    embedding.identity = identity["embedding"]
    sparse_identity = {**identity, "embedding": None}
    first, destination = verified_dense_index(full[:3], sparse_identity, tmp_path / "dense",
        {"embedding_input": "fixture_body"}, embedding, index_type=ScopedArrayIndex)
    resumed, _ = verified_dense_index(full[:3], sparse_identity, tmp_path / "dense",
        {"embedding_input": "fixture_body"}, embedding, index_type=ScopedArrayIndex)
    assert embedding.calls == 1
    assert first.search("alpha", scope="paper") == resumed.search("alpha", scope="paper")
    assert json.loads((destination / "manifest.json").read_text())["scoped_only"] is True
