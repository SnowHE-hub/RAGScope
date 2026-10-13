"""Offline retrieval experiment invariants; no models/network/accelerators."""
import json
from pathlib import Path

import pytest

from ragscope.experiments import retrieval_campaign as campaign_module
from ragscope.experiments.ledger import Campaign
from ragscope.experiments.retrieval_campaign import (build_group, chunker_for, read_jsonl,
    retrieval_scope, run_queries, score_retrieval, select_candidates, summarize_run, validate_chunks,
    verified_dense_index, run_final_contexts, select_context_strategy, write_jsonl)
from ragscope.generation.context import pack_context
from ragscope.indexing.array_index import ArrayIndex
from ragscope.ingestion.chunkers import TokenizerAdapter
from ragscope.ingestion.qasper import make_document


@pytest.fixture
def corpus():
    return [make_document("test", {"title": "Example", "abstract": "Short abstract.",
            "full_text": [{"section_name": "Evidence", "paragraphs": ["Alpha model is good. "*20,
                "Beta model uses a large corpus. "*15]}]}, "dev")]


@pytest.mark.parametrize("group", [
    {"id": "f", "method": "fixed", "chunk_size": 64, "overlap": 8},
    {"id": "s", "method": "sentence", "max_tokens": 64, "overlap_sentences": 1},
    {"id": "r", "method": "structure_recursive", "max_tokens": 64},
    {"id": "p", "method": "parent_child", "child_tokens": 64, "child_overlap": 8, "parent_tokens": 128},
])
def test_group_indexes_validate_source_coverage_and_resume(tmp_path, corpus, group):
    tokenizer = TokenizerAdapter()
    first = build_group(corpus, group, tokenizer, tmp_path, {"split": "dev", "source_hash": "test"})
    again = build_group(corpus, group, tokenizer, tmp_path, {"split": "dev", "source_hash": "test"})
    assert first == again
    chunks, parents, manifest, directory = first
    assert manifest["count"] == len(chunks)
    assert manifest["preflight"]["documents"] == 1
    assert all(c["tokenizer_id"] == "unicode_codepoint" for c in chunks)
    with (directory / "chunks.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("corrupt")
    with pytest.raises(ValueError, match="identity/hash"):
        build_group(corpus, group, tokenizer, tmp_path, {"split": "dev", "source_hash": "test"})


def test_failure_denominators_and_unanswerable_na(corpus):
    doc = corpus[0]
    query = {"answerability": "answerable", "doc_id": doc["doc_id"],
             "evidence_reference_sets": [[doc["evidence_units"][-1]["unit_id"]]]}
    units = {u["unit_id"]: u for u in doc["evidence_units"]}
    scores = score_retrieval(query, [], {}, units, failed=True)
    assert scores["continuous_coverage"] == 0
    query["answerability"] = "unanswerable"
    unanswered = score_retrieval(query, [], {}, units, failed=True)
    assert unanswered["continuous_coverage"] is None
    summary = summarize_run([{"error": {"message": "broken"}, "scores": scores,
                              "timings": {"total_ms": 100}, "context_bundle": {}},
                             {"error": None, "scores": unanswered, "timings": {"total_ms": 1}, "context_bundle": {}}])
    assert summary["n_total"] == 2 and summary["n_failed"] == 1
    assert summary["continuous_coverage"]["overall"] == 0
    assert summary["continuous_coverage"]["n_applicable"] == 1


def test_dev_candidate_selection_respects_tolerance_and_structure():
    def row(name, method, full, continuous, size):
        return {"group": {"id": name, "method": method},
                "full_reference_coverage": {"overall": full}, "continuous_coverage": {"overall": continuous},
                "index_bytes": size, "latency_ms": {"p50": 1}}
    rows = [row("simple-big", "fixed", .8, .85, 1000), row("simple-small", "fixed", .795, .845, 100),
            row("structured", "parent_child", .7, .8, 2000), row("bad-small", "sentence", .6, .7, 1)]
    assert select_candidates(rows) == {"simple": "simple-small", "structured": "structured"}


def test_real_bm25_query_records_scope_and_idempotence(tmp_path, corpus, monkeypatch):
    # Explicit codepoint counter keeps this offline and avoids tokenizer download.
    monkeypatch.setattr(campaign_module, "pack_context", lambda *a, **kw: pack_context(*a, **kw, token_counter=len))
    group = {"id": "f", "method": "fixed", "chunk_size": 256, "overlap": 32}
    chunks, parents, _, _ = build_group(corpus, group, TokenizerAdapter(), tmp_path / "indexes", {})
    index = ArrayIndex(chunks, {"embedding": None})
    doc = corpus[0]
    queries = [{"query_id": "query1", "question": "What does the Beta model use?", "doc_id": doc["doc_id"],
                "scope": {"doc_ids": [doc["doc_id"]]}, "split": "dev", "answerability": "answerable",
                "evidence_reference_sets": [[doc["evidence_units"][-1]["unit_id"]]]}]
    units = {u["unit_id"]: u for u in doc["evidence_units"]}
    config = {"seed": 42, "warmup_queries": 3, "candidate_depth": 50, "final_k": 10, "context": {"max_tokens": 2048}}
    with Campaign(tmp_path, "offline") as campaign:
        first = run_queries(index, chunks, parents, queries, units, config, group, "bm25", "none", campaign)
        again = run_queries(index, chunks, parents, queries, units, config, group, "bm25", "none", campaign)
        assert first == again and first["n_total"] == 1 and first["n_failed"] == 0
        rows = list(read_jsonl(first["records"]))
        assert len(rows) == 1 and rows[0]["raw_retrieval"]
        assert rows[0]["selected_retrieval"] == rows[0]["raw_retrieval"][:10]
        assert rows[0]["retrieval_record_schema_version"] == 2
        assert all(c["doc_id"] == doc["doc_id"] for c in rows[0]["raw_retrieval"])
        assert rows[0]["context_bundle"]["budget"]["reference_tokens"] <= 2048


def test_immutable_chunk_writer(tmp_path):
    path = tmp_path / "chunks.jsonl"
    write_jsonl(path, [{"value": "中文"}])
    write_jsonl(path, [{"value": "中文"}])
    with pytest.raises(ValueError, match="Immutable"):
        write_jsonl(path, [{"value": "changed"}])


def test_global_scope_never_uses_gold_document():
    query = {"scope": {"kind": "global"}, "doc_id": "gold-is-not-scope",
             "metadata": {"gold_doc_id_evaluation_only": "also-not-scope"}}
    assert retrieval_scope(query) is None
    query["scope"] = {"kind": "document", "doc_ids": ["user-selected"]}
    assert retrieval_scope(query) == "user-selected"
    query["scope"] = {"kind": "document", "doc_ids": []}
    with pytest.raises(ValueError, match="exactly one"):
        retrieval_scope(query)


def test_global_run_searches_distractors_and_keeps_gold_for_scoring_only(tmp_path, corpus, monkeypatch):
    monkeypatch.setattr(campaign_module, "pack_context", lambda *a, **kw: pack_context(*a, **kw, token_counter=len))
    extra = make_document("distractor", {"title": "Another paper", "abstract": "Beta model corpus",
            "full_text": [{"section_name": "Test", "paragraphs": ["Beta model uses a large corpus."]}]}, "test")
    documents = corpus + [extra]
    group = {"id": "global", "method": "fixed", "chunk_size": 256, "overlap": 0}
    chunks, parents, _, _ = build_group(documents, group, TokenizerAdapter(), tmp_path / "indexes", {})
    index = ArrayIndex(chunks, {"embedding": None})
    doc = corpus[0]
    queries = [{"query_id": "global-query", "question": "Beta model corpus", "split": "test",
                "scope": {"kind": "global", "corpus": "fixed-test-corpus"}, "answerability": "answerable",
                "metadata": {"gold_doc_id_evaluation_only": doc["doc_id"]},
                "evidence_reference_sets": [[doc["evidence_units"][-1]["unit_id"]]]}]
    units = {u["unit_id"]: u for d in documents for u in d["evidence_units"]}
    config = {"seed": 42, "warmup_queries": 3, "candidate_depth": 50, "final_k": 10, "context": {"max_tokens": 2048}}
    with Campaign(tmp_path, "offline") as campaign:
        summary = run_queries(index, chunks, parents, queries, units, config, group, "bm25", "none", campaign, experiment_id="E8")
    row = next(read_jsonl(summary["records"]))
    assert not summary["selection_eligible"]
    assert row["scope"]["kind"] == "global" and "doc_ids" not in row["scope"]
    assert {c["doc_id"] for c in row["raw_retrieval"]} == {d["doc_id"] for d in documents}
    assert row["doc_id"] == doc["doc_id"]


def test_dense_index_resume_reuses_verified_arrays(tmp_path, corpus):
    import numpy as np

    class OfflineEmbeddingFixture:
        # Unit-test numeric fixture only; no model-quality results are produced.
        identity = {"model": "offline-test-only", "revision": "1", "dimensions": 3}
        calls = 0
        last_metrics = []

        def embed(self, texts):
            self.calls += 1
            self.last_metrics = [{"input_tokens": len(t), "cache_hit": False, "seconds": 0} for t in texts]
            return np.array([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)

    group = {"id": "cache", "method": "fixed", "chunk_size": 256, "overlap": 0}
    chunks, _, _, directory = build_group(corpus, group, TokenizerAdapter(), tmp_path / "indexes", {})
    embedding = OfflineEmbeddingFixture()
    config = {"embedding_input": "body"}
    first, destination = verified_dense_index(chunks, {"embedding": None}, directory, config, embedding)
    calls = embedding.calls
    second, resumed_destination = verified_dense_index(chunks, {"embedding": None}, directory, config, embedding)
    assert embedding.calls == calls and calls > 0
    assert first.signature == second.signature and destination == resumed_destination
    with pytest.raises(ValueError):
        verified_dense_index(chunks, {"embedding": None}, directory, {"embedding_input": "changed"}, embedding)


def test_partial_encoding_metrics_survive_resume_without_cold_cost_relabel(tmp_path, corpus):
    import numpy as np

    class InterruptedOfflineFixture:
        identity = {"model": "offline-interruption-only", "revision": "1", "dimensions": 3}
        fail_once = True
        last_metrics = []

        def embed(self, texts):
            self.last_metrics = []
            if self.fail_once:
                self.last_metrics.append({"input_tokens": len(texts[0]), "cache_hit": False, "seconds": 2.0})
                self.fail_once = False
                raise ConnectionError("offline fixture interrupted after first vector")
            self.last_metrics = [{"input_tokens": len(t), "cache_hit": True, "seconds": .001} for t in texts]
            return np.array([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)

    group = {"id": "partial", "method": "fixed", "chunk_size": 256, "overlap": 0}
    chunks, _, _, directory = build_group(corpus, group, TokenizerAdapter(), tmp_path / "indexes", {})
    embedding = InterruptedOfflineFixture()
    with pytest.raises(ConnectionError):
        verified_dense_index(chunks, {"embedding": None}, directory, {"embedding_input": "body"}, embedding)
    _, destination = verified_dense_index(chunks, {"embedding": None}, directory, {"embedding_input": "body"}, embedding)
    rows = list(read_jsonl(destination / "embedding_first_observations.jsonl"))
    assert len(rows) == len(chunks)
    assert rows[0]["cache_hit"] is False and rows[0]["seconds"] == 2.0
    assert rows[0]["real_model"] is False  # Never call this numerical fixture a real model.
    assert all(r["cache_hit"] for r in rows[1:])


def test_final_retrieval_rejects_unknown_dataset_or_empty_methods(tmp_path):
    with pytest.raises(ValueError, match="Unsupported"):
        run_final_contexts(tmp_path, dataset="unregistered")
    with pytest.raises(ValueError, match="At least one"):
        run_final_contexts(tmp_path, methods=[])
    with pytest.raises(ValueError, match="family"):
        run_final_contexts(tmp_path, family="unfrozen")
    with pytest.raises(ValueError, match="family"):
        run_final_contexts(tmp_path, dataset="squad", family="simple")


def test_context_strategy_uses_dev_and_prefers_simpler_approximate_tie():
    def summary(expansion, full, continuous, split="dev"):
        return {"expansion": expansion, "split": split, "full_reference_coverage": {"overall": full},
                "continuous_coverage": {"overall": continuous}, "latency_ms": {"p50": 1}}
    assert select_context_strategy([summary("none", .795, .845), summary("neighbor", .8, .85)]) == "none"
    assert select_context_strategy([summary("none", .70, .84), summary("parent", .8, .80)]) == "parent"
    with pytest.raises(ValueError, match="dev"):
        select_context_strategy([summary("none", .9, .9, "test")])
