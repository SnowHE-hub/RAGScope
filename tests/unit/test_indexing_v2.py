"""Small offline doubles validate accounting and identity, never model quality."""
import json
from pathlib import Path

import numpy as np
import pytest

from ragscope.indexing.dense_index import DenseIndex
from ragscope.indexing.embedding import EmbeddingService
from ragscope.retrieval.hybrid_retriever import HybridRetriever
from ragscope.retrieval.reranker import Reranker


class TinyTokenizer:
    name_or_path = "fixture/tokenizer"
    model_max_length = 512

    def encode(self, text, add_special_tokens=False, truncation=False):
        assert truncation is False
        return [ord(char) for char in text] + ([1, 2] if add_special_tokens else [])

    def __call__(self, query, text, add_special_tokens=True, truncation=False):
        assert add_special_tokens and not truncation
        return {"input_ids": [0] * (len(query) + len(text) + 3)}


class TinyEmbedding:
    tokenizer = TinyTokenizer()
    max_seq_length = 512

    def __init__(self):
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):
        assert kwargs["prompt"] == ""
        assert kwargs["device"] == "cpu"
        self.calls.append(list(texts))
        return np.asarray([[len(text), 1, 2] for text in texts], dtype=np.float32)


def service(tmp_path, revision="fixture-r1", **kwargs):
    return EmbeddingService("fixture/model", revision=revision, model=TinyEmbedding(),
                            cache_dir=str(tmp_path), **kwargs)


def test_embedding_npy_cache_repeat_dedup_and_identity_isolation(tmp_path):
    first = service(tmp_path)
    vectors = first.embed(["first", "second", "first"])
    assert first.model.calls == [["first", "second"]]
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1)
    assert np.array_equal(first.embed(["first", "second", "first"]), vectors)
    assert first.last_cache_hits == 3
    assert len(first.model.calls) == 1
    second = service(tmp_path, revision="fixture-r2")
    second.embed(["first"])
    assert second.model.calls == [["first"]]
    assert first.cache.path != second.cache.path
    assert list(tmp_path.rglob("*.npy"))
    assert not list(tmp_path.rglob("*.pkl"))
    assert first._cache_key("first") != second._cache_key("first")
    assert first.embed([]).shape == (0, 3)


def test_embedding_budget_includes_query_instruction_and_special_tokens(tmp_path):
    embedding = service(tmp_path, max_input_tokens=12, query_prefix="PREFIX:")
    embedding.embed(["a" * 10])
    assert embedding.last_input_token_counts == [12]
    with pytest.raises(ValueError, match="including special"):
        embedding.embed(["a" * 11])
    with pytest.raises(ValueError, match="including special"):
        embedding.embed_query("1234")
    assert len(embedding.model.calls) == 1
    embedding.embed_query("123")
    assert embedding.model.calls[-1] == ["PREFIX:123"]


def test_embedding_refuses_unidentified_model_and_invalid_outputs(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="revision"):
        EmbeddingService("fixture", model=TinyEmbedding(), cache_dir=str(tmp_path))
    with pytest.raises(ValueError, match="explicit"):
        service(tmp_path, device="auto")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="CPU-only"):
        service(tmp_path, device="cuda")
    embedding = service(tmp_path)
    with pytest.raises(ValueError, match="batch_size"):
        embedding.embed(["x"], batch_size=0)
    embedding.model.encode = lambda *args, **kwargs: np.array([[np.nan, 0, 0]])
    with pytest.raises(ValueError, match="non-finite"):
        embedding.embed(["x"])
    assert not list(tmp_path.rglob("*.npy"))
    embedding.model.encode = lambda *args, **kwargs: np.zeros((1, 3))
    with pytest.raises(ValueError, match="zero"):
        embedding.embed(["x"])


def test_embedding_cached_dimension_and_configuration_identity(tmp_path):
    embedding = service(tmp_path)
    embedding.cache.put("bad", np.ones(4))
    with pytest.raises(ValueError, match="dimension"):
        embedding.embed(["bad"])
    changed = service(tmp_path, max_input_tokens=128, query_prefix="instruction:")
    assert changed.cache.path != embedding.cache.path
    with pytest.raises(ValueError, match="positive"):
        service(tmp_path, max_input_tokens=0)
    with pytest.raises(ValueError, match="tokenizer"):
        EmbeddingService("fixture", revision="r1", model=object(), cache_dir=str(tmp_path))


def test_embedding_uses_resolved_revision_and_model_dimension_method(tmp_path):
    from types import SimpleNamespace
    model = TinyEmbedding()
    model._first_module = lambda: SimpleNamespace(auto_model=SimpleNamespace(config=SimpleNamespace(_commit_hash="resolved-revision")))
    model.get_embedding_dimension = lambda: 3
    embedding = EmbeddingService("BAAI/bge-fixture", revision="main", model=model, cache_dir=str(tmp_path))
    assert embedding.identity["revision"] == "resolved-revision"
    assert embedding.query_prefix.startswith("Represent")


class TinyCollection:
    def __init__(self):
        self.rows = {}

    def count(self):
        return len(self.rows)

    def add(self, ids, documents, metadatas, embeddings):
        for cid, text, metadata, vector in zip(ids, documents, metadatas, embeddings):
            assert cid not in self.rows
            self.rows[cid] = (text, metadata, vector)

    def get(self, include):
        return {"ids": list(self.rows), "documents": [row[0] for row in self.rows.values()],
                "metadatas": [row[1] for row in self.rows.values()]}

    def query(self, query_embeddings, n_results, include):
        ids = list(self.rows)[:n_results]
        return {"ids": [ids], "documents": [[self.rows[cid][0] for cid in ids]],
                "metadatas": [[self.rows[cid][1] for cid in ids]], "distances": [[0.1] * len(ids)]}


class TinyClient:
    def __init__(self):
        self.collection = TinyCollection()

    def get_or_create_collection(self, **kwargs):
        return self.collection


ROWS = [{"chunk_id": "c1", "doc_id": "d", "text": "first", "unit_spans": [{"unit_id": "u", "start": 0, "end": 5}]},
        {"chunk_id": "c2", "doc_id": "d", "text": "second", "parent_id": "p"}]


def test_dense_identity_reuse_content_mismatch_and_query_space(tmp_path):
    client = TinyClient()
    index = DenseIndex("test", str(tmp_path / "index"), client=client)
    embedding = service(tmp_path / "cache")
    assert not index.exists()
    index.build(ROWS, embedding, batch_size=1)
    assert index.exists()
    index.build(ROWS, embedding)
    assert len(embedding.model.calls) == 2
    result = index.search(np.array([1, 0, 0]), embedding_identity=embedding.identity)
    assert result[0]["unit_spans"] == ROWS[0]["unit_spans"]
    assert result[1]["parent_id"] == "p"
    assert result[0]["score"] == pytest.approx(.9)
    other = service(tmp_path / "cache", revision="fixture-r2")
    with pytest.raises(ValueError, match="identity mismatch"):
        index.build(ROWS, other)
    with pytest.raises(ValueError, match="identity mismatch"):
        index.build([dict(ROWS[0], text="changed"), ROWS[1]], embedding)
    with pytest.raises(ValueError, match="vector space identity"):
        index.search(np.array([1, 0, 0]), embedding_identity=other.identity)
    with pytest.raises(ValueError, match="vector space identity"):
        index.search(np.array([1, 0, 0]))
    assert client.collection.count() == 2


def test_dense_resumes_interrupted_owned_batches_without_duplicate_rows(tmp_path):
    index = DenseIndex("test", str(tmp_path / "index"), client=TinyClient())
    embedding = service(tmp_path / "cache")
    real_embed = embedding.embed
    def interrupted(texts, **kwargs):
        if texts == ["second"]:
            raise RuntimeError("controlled interruption")
        return real_embed(texts, **kwargs)
    embedding.embed = interrupted
    with pytest.raises(RuntimeError, match="controlled"):
        index.build(ROWS, embedding, batch_size=1)
    assert index.collection.count() == 1
    with pytest.raises(ValueError, match="incomplete"):
        index.search(np.array([1, 0, 0]), embedding_identity=embedding.identity)
    embedding.embed = real_embed
    index.build(ROWS, embedding, batch_size=1)
    assert index.collection.count() == 2 and index.exists()
    assert embedding.model.calls == [["first"], ["second"]]


def test_dense_preserves_unmanifested_legacy_and_rejects_count_tamper(tmp_path):
    index = DenseIndex("test", str(tmp_path / "index"), client=TinyClient())
    embedding = service(tmp_path / "cache")
    index.collection.rows["legacy"] = ("old", {}, [])
    with pytest.raises(ValueError, match="legacy"):
        index.build(ROWS, embedding)
    assert index.collection.rows == {"legacy": ("old", {}, [])}
    fresh = DenseIndex("fresh", str(tmp_path / "fresh"), client=TinyClient())
    fresh.build(ROWS, embedding)
    fresh.collection.rows.pop("c2")
    with pytest.raises(ValueError, match="count"):
        fresh.exists()


def test_dense_empty_and_validation(tmp_path):
    index = DenseIndex("test", str(tmp_path / "index"), client=TinyClient())
    embedding = service(tmp_path / "cache")
    with pytest.raises(ValueError, match="unique"):
        index.build([ROWS[0], ROWS[0]], embedding)
    with pytest.raises(ValueError, match="batch_size"):
        index.build(ROWS, embedding, batch_size=0)
    with pytest.raises(ValueError, match="manifest"):
        index.search(np.zeros(3), embedding_identity=embedding.identity)
    index.build([], embedding)
    assert index.search(np.zeros(3), embedding_identity=embedding.identity) == []
    with pytest.raises(ValueError, match="Invalid query"):
        index.search(np.zeros(4), embedding_identity=embedding.identity)
    with pytest.raises(ValueError, match="positive"):
        index.search(np.zeros(3), k=0, embedding_identity=embedding.identity)
    manifest = json.loads(index.manifest_path.read_text())
    manifest["identity"]["dimension"] = 77
    index.manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="signature"):
        index.exists()


def test_dense_retrieved_payload_tamper_is_not_returned(tmp_path):
    index = DenseIndex("test", str(tmp_path / "index"), client=TinyClient())
    embedding = service(tmp_path / "cache")
    index.build(ROWS, embedding)
    assert not index.exists({"wrong": "identity"})
    text, metadata, vector = index.collection.rows["c1"]
    index.collection.rows["c1"] = (text + "tampered", metadata, vector)
    with pytest.raises(ValueError, match="integrity"):
        index.search(np.ones(3), embedding_identity=embedding.identity)


class TinyCrossEncoder:
    tokenizer = TinyTokenizer()
    max_length = 512

    def __init__(self):
        self.pairs = []

    def predict(self, pairs, **kwargs):
        self.pairs.extend(pairs)
        return np.asarray([100 if "Z" in text else len(text) * .01 for _, text in pairs])


def reranker(**kwargs):
    return Reranker("fixture/reranker", revision="fixture-r1", model=TinyCrossEncoder(), **kwargs)


def test_reranker_covers_tail_with_query_in_budget_and_window_max():
    rerank = reranker(max_length=32, window_overlap=4)
    text = "中🙂a" * 25 + "Z"
    candidates = [{"chunk_id": "long", "text": text}, {"chunk_id": "short", "text": "plain"}]
    result = rerank.rerank("QUESTION", candidates)
    assert result[0]["chunk_id"] == "long" and result[0]["rerank_score"] == 100
    assert result[0]["rerank_aggregation"] == "windowed_max"
    records = result[0]["rerank_windows"]
    assert len(records) > 1 and records[0]["start"] == 0 and records[-1]["end"] == len(text)
    assert all(record["pair_tokens"] <= 32 for record in records)
    assert all(records[i]["start"] <= records[i - 1]["end"] for i in range(1, len(records)))
    assert all(len(query) + len(window) + 3 <= 32 for query, window in rerank.model.pairs)
    assert "rerank_score" not in candidates[0]


class TinyRetriever:
    def __init__(self, rows):
        self.rows, self.depths = rows, []
    def retrieve(self, query, k=10):
        self.depths.append(k)
        return self.rows[:k]


def test_hybrid_50_candidates_and_rerank_50_to_10_changes_membership():
    rows = [{"chunk_id": f"c{i:02}", "text": "Z" if i == 49 else "text"} for i in range(50)]
    dense, sparse = TinyRetriever(rows), TinyRetriever(rows)
    hybrid = HybridRetriever(dense, sparse)
    baseline = hybrid.retrieve("question", k=10)
    assert dense.depths == [50] and sparse.depths == [50]
    assert baseline[0]["score"] == pytest.approx(1 / 61)
    result = reranker().retrieve("question", hybrid)
    assert len(result) == 10 and result[0]["chunk_id"] == "c49"
    assert "c49" not in {row["chunk_id"] for row in baseline}
    assert dense.depths == [50, 50] and sparse.depths == [50, 50]


def test_hybrid_deduplicates_and_validates():
    rows = [{"chunk_id": "a", "text": "a"}, {"chunk_id": "b", "text": "b"}, {"chunk_id": "a", "text": "a"}]
    ret = TinyRetriever(rows)
    results = HybridRetriever(ret, ret).retrieve("q")
    assert len(results) == 2 and results[0]["score"] == pytest.approx(1 / 61)
    with pytest.raises(ValueError):
        HybridRetriever(ret, ret, alpha=2)
    with pytest.raises(ValueError):
        HybridRetriever(ret, ret, candidate_depth=0)
    with pytest.raises(ValueError):
        HybridRetriever(ret, ret).retrieve("q", 0)
    with pytest.raises(ValueError):
        HybridRetriever(ret, ret, rrf_constant=float("nan"))
    assert HybridRetriever(ret, TinyRetriever([]), alpha=0).retrieve("q") == []
    assert HybridRetriever(TinyRetriever([]), ret, alpha=1).retrieve("q") == []


def test_reranker_empty_overlong_query_and_invalid_scores():
    rerank = reranker(max_length=16, window_overlap=4)
    assert rerank.rerank("q", []) == []
    assert rerank.rerank("q", [{"chunk_id": "empty", "text": ""}])[0]["rerank_score"] == 0
    with pytest.raises(ValueError, match="Question"):
        rerank.rerank("q" * 14, [{"chunk_id": "c", "text": "abc"}])
    with pytest.raises(ValueError):
        rerank.rerank("q", [], top_k=0)
    with pytest.raises(ValueError):
        rerank.retrieve("q", TinyRetriever([]), candidate_depth=5, top_k=10)
    rerank.model.predict = lambda *args, **kwargs: [np.nan]
    with pytest.raises(ValueError, match="scores"):
        rerank.rerank("q", [{"chunk_id": "c", "text": "abc"}])
    with pytest.raises(ValueError, match="unique"):
        rerank.rerank("q", [{"chunk_id": "c", "text": "a"}, {"chunk_id": "c", "text": "a"}])


def test_reranker_tight_budget_reduces_overlap_and_checks_tokenizer_limit():
    model = TinyCrossEncoder()
    model.tokenizer = TinyTokenizer()
    model.tokenizer.model_max_length = 12
    rerank = Reranker("fixture", revision="r1", model=model, max_length=16, window_overlap=10)
    result = rerank.rerank("1234567", [{"chunk_id": "c", "text": "abcdefgh"}])
    assert rerank.max_length == 12
    records = result[0]["rerank_windows"]
    assert records[-1]["end"] == 8
    assert all(record["pair_tokens"] <= 12 for record in records)
    assert all(records[i]["end"] > records[i - 1]["end"] for i in range(1, len(records)))


@pytest.mark.parametrize("kwargs", [{"max_length": 600}, {"max_length": 0},
    {"max_length": 16, "window_overlap": 16}, {"batch_size": 0}, {"device": "auto"}])
def test_reranker_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        reranker(**kwargs)
