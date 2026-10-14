"""Offline identity/locking/resource fixtures, never counted as real results."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import json

import numpy as np
import pytest
from filelock import FileLock
from ragscope.experiments import aux_preencode as aux
from ragscope.experiments.ledger import Campaign
from ragscope.indexing.identity import atomic_json, file_hash, signature


def test_guard_retains_only_completed_measurements_before_low_ram(monkeypatch):
    embedding = object.__new__(aux.RequestBoundaryGuard)
    embedding.identity = {"dimensions": 2}
    embedding.last_metrics = [{"stale": True}]
    memory = iter([2 * 1024**3, 512 * 1024**2])
    monkeypatch.setattr(aux.psutil, "virtual_memory", lambda: SimpleNamespace(available=next(memory)))
    def base(self, texts):
        self.last_metrics = [{"text": texts[0]}]
        return np.array([[1., 0.]])
    monkeypatch.setattr(aux.LocalEmbeddingService, "embed", base)
    with pytest.raises(MemoryError):
        embedding.embed(["completed", "blocked"])
    assert embedding.last_metrics == [{"text": "completed"}]


def test_expired_deadline_never_calls_backend(monkeypatch):
    embedding = object.__new__(aux.RequestBoundaryGuard)
    embedding.deadline = datetime.now(timezone.utc) - timedelta(seconds=1)
    embedding.last_metrics = [{"stale": True}]
    monkeypatch.setattr(aux.LocalEmbeddingService, "embed", lambda *a: pytest.fail("Backend must not be called"))
    with pytest.raises(TimeoutError):
        embedding.embed(["blocked"])
    assert embedding.last_metrics == []


def test_index_only_uses_disjoint_lock_and_exact_consumer_identity(tmp_path, monkeypatch):
    main = tmp_path / "runs/campaign_retrieval"
    main.mkdir(parents=True)
    state = main / "state.json"
    atomic_json(state, {"owner": "R3", "status": "RUNNING"})
    original = file_hash(state)
    group, base = {"id": "R512", "method": "fixture"}, {"corpus_sha256": "fixture"}
    expected = {**base, "group": group, "chunking_signature": "fixture_signature"}
    job = dict(dataset="squad", area="squad_chunks", group=group, base_identity=base,
               documents=[], index_type=aux.ArrayIndex, config={"bm25": {"idf_scope": "full"}, "embedding_input": "body"},
               chunking={"max_cumulative_chunks": 180000}, sources=[], reference_count=0)
    chunks = [{"chunk_id": "c", "doc_id": "paper", "text": "body"}]
    monkeypatch.setattr(aux, "chunker_for", lambda *a: SimpleNamespace(chunking_signature="fixture_signature"))
    def build(documents, group, tokenizer, root, identity):
        directory = root / signature(expected)
        directory.mkdir()
        manifest = {"identity": expected, "count": 1, "parent_count": 0}
        atomic_json(directory / "chunk_manifest.json", manifest)
        return chunks, [], manifest, directory
    monkeypatch.setattr(aux, "build_group", build)
    def dense(chunks, sparse, directory, config, embedding, **kwargs):
        assert sparse == {**expected, "engine": "ArrayIndex-v2", "bm25": job["config"]["bm25"], "embedding": None}
        path = directory / ("dense_" + signature(embedding.identity)[:16])
        atomic_json(path / "manifest.json", {"fixture": True})
        return SimpleNamespace(identity={**sparse, "embedding": embedding.identity, "embedding_input": "body"}), path
    monkeypatch.setattr(aux, "verified_dense_index", dense)
    embedding = SimpleNamespace(identity={"device": "cpu", "model": "offline_fixture"},
        cache=SimpleNamespace(get=lambda _: None), tokenizer=SimpleNamespace(encode=lambda text: list(text)))
    with FileLock(str(main / "writer.lock"), timeout=0):
        with Campaign(tmp_path, "campaign_aux_preencode") as campaign:
            result = aux.encode_job(tmp_path, "campaign", job, None, embedding, campaign)
    assert result["main_campaign_opened"] is False
    assert result["quality_experiment_rows"] == 0
    assert file_hash(state) == original and not (main / "records").exists()
    assert result["directory"].endswith(signature(expected))
    assert json.loads((tmp_path / "runs/campaign_aux_preencode/state.json").read_text())["tasks"]["squad_R512"]["status"] == "PASS"
