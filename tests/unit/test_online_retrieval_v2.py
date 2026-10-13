"""Offline numeric fixtures plus actual BM25; no model-quality claims."""
import json

import numpy as np
import pytest

from ragscope.generation.context import pack_context
from ragscope.indexing.array_index import ArrayIndex
from ragscope.indexing.identity import artifact, file_hash, signature
from ragscope.routing import online_retrieval as module


class OfflineEmbedding:
    identity = {"backend": "offline_fixture", "dimensions": 2, "device": "cpu", "revision": "fixture"}
    calls = 0
    last_metrics = []

    def embed_query(self, text):
        self.calls += 1
        self.last_metrics = [{"cache_hit": self.calls > 1, "body_tokens": len(text), "seconds": .001}]
        return np.array([1.0, 0.0], dtype=np.float32)


class OfflineReranker:
    device = "cpu"
    identity = {"model": "offline_fixture", "revision": "fixture", "device": "cpu",
                "aggregation": "windowed_max", "max_length": 512}
    calls = []
    last_window_records = []

    def rerank(self, question, candidates, top_k):
        self.calls.append((question, len(candidates), top_k))
        self.last_window_records = [{"chunk_id": c["chunk_id"], "start": 0, "end": len(c["text"]),
                                     "pair_tokens": 12, "score": float(i)} for i, c in enumerate(candidates)]
        return [{**c, "rerank_score": 1.0} for c in reversed(candidates[-top_k:])]


@pytest.fixture
def chunks():
    result = []
    for i in range(70):
        doc_id = "d1" if i < 60 else "d2"
        text = ("alpha evidence " if i < 60 else "othermarker evidence ") + str(i)
        result.append({"chunk_id": f"c{i:02d}", "doc_id": doc_id, "text": text,
            "canonical_start": i*100, "canonical_end": i*100+len(text), "parent_id": None,
            "unit_spans": [{"unit_id": f"u{i}", "doc_id": doc_id, "start": i*100,
                            "end": i*100+len(text), "chunk_start": 0, "chunk_end": len(text)}]})
    return result


@pytest.fixture(autouse=True)
def no_tokenizer_download(monkeypatch):
    monkeypatch.setattr(module, "pack_context", lambda *args, **kwargs: pack_context(*args, **kwargs, token_counter=len))


def live(chunks, routes=module.ROUTES):
    embedding, reranker = OfflineEmbedding(), OfflineReranker()
    reranker.calls = []
    index = ArrayIndex(chunks, {"embedding": embedding.identity}, np.array([[1., 0.]]*len(chunks), dtype=np.float32))
    return module.LiveRetriever(index, [], expansion="none", identity={"data_kind": "offline_fixture"},
        embedding=embedding, reranker=reranker, routes=routes), embedding, reranker


def test_live_each_route_calls_actual_search_and_50_to_10_reranker(chunks):
    retriever, embedding, reranker = live(chunks)
    scope = {"kind": "document", "doc_ids": ["d1"]}
    bm25 = retriever.retrieve("alpha", scope, "bm25")
    assert bm25["error"] is None and embedding.calls == 0 and reranker.calls == []
    hybrid = retriever.retrieve("alpha", scope, "hybrid")
    assert hybrid["error"] is None and embedding.calls == 1 and hybrid["query_cache_state"] == "miss"
    ranked = retriever.retrieve("alpha", scope, "hybrid_rerank")
    assert ranked["error"] is None and embedding.calls == 2
    assert reranker.calls == [("alpha", 50, 10)]
    assert len(ranked["raw_retrieval"]) == 50 and len(ranked["selected_retrieval"]) == 10
    assert len(ranked["rerank_windows"]) == 50
    assert ranked["query_cache_state"] == "hit"
    assert all(c["doc_id"] == "d1" for c in ranked["raw_retrieval"])
    assert ranked["context_bundle"]["budget"]["max_reference_tokens"] == 2048
    assert ranked["timings_ms"]["total_ms"] >= ranked["timings_ms"]["rerank_ms"] > 0
    # A new question actually runs search again and can retrieve the other doc.
    global_row = retriever.retrieve("othermarker", {"kind": "global"}, "bm25")
    assert {c["doc_id"] for c in global_row["raw_retrieval"]} == {"d1", "d2"}


def test_live_scope_is_explicit_and_never_accepts_gold(chunks):
    retriever, embedding, _ = live(chunks)
    with pytest.raises(ValueError, match="question string"):
        retriever.retrieve({"question": "alpha", "answer": "gold"}, {}, "hybrid")
    with pytest.raises(ValueError, match="gold"):
        retriever.retrieve("alpha", {"kind": "global", "gold_doc_id": "d1"}, "hybrid")
    with pytest.raises(ValueError, match="absent"):
        retriever.retrieve("alpha", {"doc_ids": ["unknown"]}, "hybrid")
    assert embedding.calls == 0
    small = retriever.retrieve("othermarker", {"doc_ids": ["d2"]}, "hybrid_rerank")
    assert small["actual_candidate_count"] == 10  # Never pretend 50 exist in a small scope.


def test_injected_reranker_actual_gpu_device_is_rejected(chunks):
    from types import SimpleNamespace
    retriever, embedding, reranker = live(chunks)
    reranker.model = SimpleNamespace(device="cuda:0")
    with pytest.raises(ValueError, match="actually resides outside CPU"):
        module.LiveRetriever(retriever.index, [], expansion="none", identity={},
                             embedding=embedding, reranker=reranker)


def test_live_backend_failure_retains_error_without_silent_fallback(chunks, monkeypatch):
    retriever, embedding, _ = live(chunks)
    def fail(_):
        embedding.last_metrics = []
        raise ConnectionError("offline backend failure fixture")
    monkeypatch.setattr(embedding, "embed_query", fail)
    row = retriever.retrieve("alpha", {"doc_ids": ["d1"]}, "hybrid")
    assert row["error"]["type"] == "ConnectionError"
    assert row["raw_retrieval"] == [] and row["context_bundle"] == {}
    assert row["timings_ms"]["query_encoding_ms"] > 0
    assert row["timings_ms"]["search_ms"] == 0


def test_live_common_model_budget_checks_match_exact_context_packer(chunks):
    retriever, _, _ = live(chunks)
    scope = {"doc_ids": ["d1"]}
    ordinary = retriever.retrieve("alpha", scope, "bm25")
    checks = (lambda rendered: len(rendered) <= 60,)
    checked = retriever.retrieve("alpha", scope, "bm25", model_budget_checks=checks)
    expected = pack_context(checked["retrieval_context_bundle"]["pieces"], max_tokens=2048,
        scope=checked["scope"], model_budget_checks=checks,
        token_counter=len).to_dict()
    assert checked["context_bundle"] == expected
    assert checked["context_bundle"]["budget"]["model_budget_checks"] == 1
    assert len(checked["context_bundle"]["pieces"]) < len(ordinary["context_bundle"]["pieces"])
    assert any(t["reason"] == "model_budget" for t in checked["context_bundle"]["truncations"])


def test_common_budget_cannot_resurrect_reference_excluded_candidate(chunks):
    selected = []
    for i, (chunk, text) in enumerate(zip(chunks[:2], ("alpha " + "x"*1400, "other " + "y"*1000))):
        selected.append({**chunk, "text": text, "canonical_start": i*2000, "canonical_end": i*2000+len(text),
                         "unit_spans": []})
    retriever, _, _ = live(selected, routes=("bm25",))
    row = retriever.retrieve("alpha", {"doc_ids": ["d1"]}, "bm25",
                             model_budget_checks=(lambda rendered: len(rendered) <= 1100,))
    assert row["error"] is None
    assert [p["chunk_id"] for p in row["retrieval_context_bundle"]["pieces"]] == ["c00"]
    assert row["context_bundle"]["pieces"] == []
    assert any(t["chunk_id"] == "c01" and t["reason"] == "reference_budget"
               for t in row["retrieval_context_bundle"]["truncations"])


def test_factory_missing_final_data_never_loads_index(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Index loading occurred before final selection exists")
    monkeypatch.setattr(module.ArrayIndex, "load", forbidden)
    with pytest.raises(module.RetrievalNotReady, match="CANDIDATE_SELECTION"):
        module.create_live_retriever(tmp_path)


def test_factory_loads_only_verified_existing_selection_and_rejects_changed_hash(tmp_path, chunks):
    corpus, run = tmp_path / "corpora/fixture", tmp_path / "runs/fixture_retrieval"
    corpus.mkdir(parents=True); run.mkdir(parents=True)
    data = corpus / "documents.jsonl"
    data.write_text('{}\n', encoding="utf-8")  # Provenance fixture only, never a formal corpus.
    split = corpus / "SPLIT_MANIFEST.json"
    split.write_text(json.dumps({"files": {"documents.jsonl": {"sha256": file_hash(data)}}}), encoding="utf-8")
    selection = run / "CANDIDATE_SELECTION.json"
    selection.write_text(json.dumps({"selected": {"simple": "F512_O0", "structured": "R512"},
        "method": "hybrid", "split": "dev", "data_manifest_sha256": file_hash(split)}), encoding="utf-8")
    evidence = run / "dev_expansion_fixture.json"
    evidence.write_text('{"data_kind":"offline_fixture"}', encoding="utf-8")
    contexts = run / "CONTEXT_SELECTION.json"
    context_value = {"selected": {"R512": "none"}, "split": "dev", "method": "hybrid", "test_used_for_selection": False,
                     "candidate_selection_sha256": file_hash(selection), "evidence": {"R512": [artifact(evidence)]}}
    contexts.write_text(json.dumps(context_value), encoding="utf-8")
    directory = tmp_path / "indexes/fixture/all_chunks/frozen_fixture"
    directory.mkdir(parents=True)
    for name, text in (("chunks.jsonl", '\n'.join(json.dumps(c) for c in chunks)), ("parents.jsonl", "")):
        (directory / name).write_text(text, encoding="utf-8")
    identity = {"group": {"id": "R512"}, "data_manifest_sha256": file_hash(split), "corpus_sha256": file_hash(data),
                "split": "frozen_all_splits_corpus", "scope": "global_or_explicit_paper"}
    (directory / "chunk_manifest.json").write_text(json.dumps({"identity": identity, "signature": signature(identity),
        "artifacts": [artifact(directory / name) for name in ("chunks.jsonl", "parents.jsonl")]}), encoding="utf-8")
    bm25 = {"implementation": "rank_bm25.BM25Okapi", "tokenizer": "alnum-cjk-v1", "k1": 1.5,
            "b": .75, "epsilon": .25, "idf_scope": "frozen_509_paper_corpus"}
    index = ArrayIndex(chunks, {**identity, "engine": "ArrayIndex-v2", "bm25": bm25, "embedding": None})
    index.save(directory / "sparse")
    retriever = module.create_live_retriever(tmp_path, "fixture", routes=("bm25",))
    row = retriever.retrieve("alpha", {"doc_ids": ["d1"]}, "bm25")
    assert row["error"] is None and retriever.load_timings_ms["index_load_ms"] > 0
    assert retriever.identity["group"] == "R512" and "index_load_ms" not in retriever.identity
    context_value["candidate_selection_sha256"] = "tampered"
    contexts.write_text(json.dumps(context_value), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        module.create_live_retriever(tmp_path, "fixture", routes=("bm25",))
