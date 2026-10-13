"""Offline routing engineering tests. Synthetic labels never enter campaign results."""
import json
import hashlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ragscope.routing.core import (ROUTES, FixedRouter, RuleRouter, RouteDecision, RoutingRequest,
                                  build_proxy_labels, freeze_query_ids, validate_training_splits)
from ragscope.routing.evaluate import calibration, deployment_gate, online, replay, summarize
from ragscope.routing.laya_adapter import LayaRouter
from ragscope.routing.learned import TfidfRouter
from ragscope.routing.lstm import LSTMRouter, parameter_budget, words


def fixture_query(qid="q", split="train", question="Why compare these methods?"):
    return {"query_id": qid, "question": question, "split": split, "scope": {"doc_id": split + "_" + qid}}


def fixture_records(query):
    # Contract doubles with true real_model flags only to exercise formal input validation.
    # These are never persisted as campaign model outcomes.
    return [{"query_id": query["query_id"], "split": query["split"], "path": route,
             "answer_f1": [.5, .51, .50][i], "evidence_coverage": .7, "total_latency_s": i + 1,
             "error": None, "real_model": True, "comparison_config_hash": "fixture-config"}
            for i, route in enumerate(ROUTES)]


def test_visible_features_exclude_gold_and_question_length_shortcut():
    query = fixture_query()
    query.update(answer_refs=["hybrid"], gold_doc_id="secret", evidence_reference_sets=[["gold"]])
    query["metadata"] = {"document_length_tokens": 1000, "document_type": "paper", "gold_answer": "bm25"}
    request = RoutingRequest.from_query(query)
    assert "gold" not in request.features_text() and "secret" not in request.features_text()
    assert request.metadata == {"document_length_tokens": 1000, "document_type": "paper"}
    rules = RuleRouter()
    assert rules.choose(request).route == "hybrid_rerank"
    assert rules.choose(RoutingRequest("Why?")).route == "hybrid_rerank"
    assert rules.choose(RoutingRequest('Find "batch_size".')).route == "bm25"
    assert rules.choose(RoutingRequest("What approach is used?")).route == "hybrid"
    assert rules.choose(RoutingRequest("Compare `batch_size` and `max_tokens`.")).route == "hybrid_rerank"
    for route in ROUTES:
        assert FixedRouter(route).choose(request).route == route
    with pytest.raises(ValueError):
        FixedRouter("oracle")
    with pytest.raises(ValueError):
        RouteDecision("oracle", "invalid")


def test_frozen_ids_are_deterministic_and_insufficient_n_is_explicit():
    queries = [fixture_query(str(i), split) for split in ("train", "dev", "test") for i in range(4)]
    for row in queries:
        row["query_id"] = row["split"] + row["query_id"]
    first = freeze_query_ids(queries)
    assert first == freeze_query_ids(list(reversed(queries)))
    assert first["actual_counts"] == {"train": 4, "dev": 4, "test": 4}
    assert first["requested_counts"]["test"] == 100
    with pytest.raises(ValueError, match="Duplicate"):
        freeze_query_ids(queries + [queries[0]])
    with pytest.raises(ValueError):
        freeze_query_ids(queries, {"test": 0})


def test_proxy_f1_tolerance_cost_then_evidence_and_failures_remain():
    query = fixture_query()
    records = fixture_records(query)
    labels = build_proxy_labels([query], records, split="train", query_ids=["q"], expected_config_hash="fixture-config")
    assert labels[0]["label"] == "bm25"  # .50 within .01 of .51; lower cost wins.
    records[0]["answer_f1"] = .499
    assert build_proxy_labels([query], records, split="train", query_ids=["q"], expected_config_hash="fixture-config")[0]["label"] == "hybrid"
    for row in records:
        row.update(answer_f1=0., error={"type": "timeout"}, total_latency_s=2, evidence_coverage=None)
    labels = build_proxy_labels([query], records, split="train", query_ids=["q"], expected_config_hash="fixture-config")
    assert len(labels) == 1 and labels[0]["all_paths_failed"]
    assert len(labels[0]["route_scores"]) == 3
    for row in records:
        row.update(answer_f1=.5, error=None, total_latency_s=1, evidence_coverage=.5)
    records[2]["evidence_coverage"] = .8
    assert build_proxy_labels([query], records, split="train", query_ids=["q"], expected_config_hash="fixture-config")[0]["label"] == "hybrid_rerank"


@pytest.mark.parametrize("change", ["mock", "wrong_config", "missing", "duplicate", "wrong_split", "negative_cost", "nan", "error_f1"])
def test_invalid_label_evidence_is_rejected(change):
    query, records = fixture_query(), fixture_records(fixture_query())
    if change == "mock": records[0]["real_model"] = False
    elif change == "wrong_config": records[0]["comparison_config_hash"] = "another-model"
    elif change == "missing": records.pop()
    elif change == "duplicate": records.append(records[0])
    elif change == "wrong_split": records[0]["split"] = "test"
    elif change == "negative_cost": records[0]["total_latency_s"] = -1
    elif change == "nan": records[0]["answer_f1"] = float("nan")
    elif change == "error_f1": records[0]["error"] = {"type": "oom"}
    with pytest.raises(ValueError):
        build_proxy_labels([query], records, split="train", query_ids=["q"], expected_config_hash="fixture-config")


def labelled_fixtures():
    phrases = ["find quoted constant foo_bar identifier", "describe semantic method approach", "compare why different results relationship"]
    train = [dict(fixture_query(f"train{i}", "train", phrases[i % 3]), label=ROUTES[i % 3]) for i in range(12)]
    dev = [dict(fixture_query(f"dev{i}", "dev", phrases[i % 3]), label=ROUTES[i % 3]) for i in range(6)]
    return train, dev


def test_train_dev_and_document_leakage_guards():
    train, dev = labelled_fixtures()
    assert validate_training_splits(train, dev)["train_n"] == 12
    dev[0]["scope"] = train[0]["scope"]
    with pytest.raises(ValueError, match="Document leakage"):
        validate_training_splits(train, dev)
    dev[0]["split"] = "test"
    with pytest.raises(ValueError, match="test is prohibited"):
        validate_training_splits(train, dev)


@pytest.mark.parametrize("kind", ["linear", "mlp"])
def test_small_sklearn_functional_training_and_safe_roundtrip(tmp_path, kind):
    train, dev = labelled_fixtures()
    router = TfidfRouter(kind, max_features=64, reduction_components=4, hidden_units=4, max_epochs=3, patience=2)
    with pytest.raises(ValueError, match="verified"):
        router.fit(train, dev)
    router.fit(train, dev, data_kind="offline_fixture")
    before = router.choose(RoutingRequest.from_query(dev[0]))
    assert set(before.probabilities) == set(ROUTES)
    assert sum(before.probabilities.values()) == pytest.approx(1)
    assert router.report["data_kind"] == "offline_fixture"
    assert len(router.report["history"]) <= 3
    router.save(tmp_path)
    restored = TfidfRouter.load(tmp_path)
    after = restored.choose(RoutingRequest.from_query(dev[0]))
    assert before.route == after.route
    assert before.probabilities == pytest.approx(after.probabilities)
    assert not list(tmp_path.glob("*.pkl"))
    with pytest.raises(FileExistsError):
        router.save(tmp_path)
    weights = tmp_path / "weights.npz"
    weights.write_bytes(weights.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        TfidfRouter.load(tmp_path)


def test_single_class_collapse_is_declared_for_every_learner(tmp_path):
    train, dev = labelled_fixtures()
    for row in train:
        row["label"] = "hybrid"
    for name, router in [("linear", TfidfRouter("linear")), ("mlp", TfidfRouter("mlp")), ("lstm", LSTMRouter())]:
        router.fit(train, dev, data_kind="offline_fixture")
        assert router.report["status"] == "PASS_FALLBACK_SINGLE_CLASS"
        assert router.choose(RoutingRequest("Why?")).route == "hybrid"
        router.save(tmp_path / name)
        assert type(router).load(tmp_path / name).choose(RoutingRequest("query")).route == "hybrid"


def test_lstm_parameter_and_token_bounds_without_loading_torch():
    assert parameter_budget(8000) < 2_000_000
    with pytest.raises(ValueError, match="2M"):
        LSTMRouter(max_vocabulary=100_000)
    with pytest.raises(ValueError):
        LSTMRouter(max_epochs=21)
    router = LSTMRouter(max_tokens=2)
    router.vocabulary = {"hello": 2, "world": 3}
    assert router._encode("hello world unknown") == ([2, 3], 1)
    assert router._encode("") == ([1], 0)
    assert words("中Text!") == ["中", "text"]
    with pytest.raises(ValueError, match="not been trained"):
        router.choose(RoutingRequest("q"))
    # This test does not exercise torch training; a real environment check is still required.


def test_small_cpu_lstm_real_training_and_safe_roundtrip(tmp_path):
    torch = pytest.importorskip("torch", reason="CPU Torch is isolated in envs/models")
    train, dev = labelled_fixtures()
    router = LSTMRouter(max_vocabulary=64, embedding_size=4, hidden_size=4,
                        max_tokens=8, max_epochs=3, patience=2)
    with pytest.raises(ValueError, match="verified"):
        router.fit(train, dev)
    with pytest.raises(ValueError, match="untrained"):
        router.save(tmp_path / "empty")
    router.fit(train, dev, data_kind="offline_fixture")
    assert router.report["status"] == "TRAINED"
    assert 0 < router.report["parameter_count"] < 2_000_000
    assert 1 <= len(router.report["history"]) <= 3
    assert next(router.model.parameters()).device.type == "cpu"
    assert all(np.isfinite(row["dev_log_loss"]) for row in router.report["history"])
    request = RoutingRequest("unseen " * 12)
    before = router.choose(request)
    assert before.metadata["truncated_router_tokens"] > 0
    assert sum(before.probabilities.values()) == pytest.approx(1)
    path = tmp_path / "lstm"
    router.save(path)
    restored = LSTMRouter.load(path)
    assert restored.choose(request).probabilities == pytest.approx(before.probabilities)
    with pytest.raises(FileExistsError):
        router.save(path)
    with (path / "weights.npz").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        LSTMRouter.load(path)


def test_formal_training_provenance_rejects_mixed_config_and_split_duplicates():
    from ragscope.routing.core import validate_proxy_provenance
    train, dev = labelled_fixtures()
    with pytest.raises(ValueError, match="Duplicate"):
        validate_training_splits(train + [train[0]], dev)
    with pytest.raises(ValueError, match="data_kind"):
        validate_proxy_provenance(train, dev, "invented")
    for row in train + dev:
        row.update(proxy_label=True, comparison_config_hash="fixture-shared")
    validate_proxy_provenance(train, dev, "real_generation_proxy")
    dev[0]["comparison_config_hash"] = "different-generator"
    with pytest.raises(ValueError, match="same-configuration"):
        validate_proxy_provenance(train, dev, "real_generation_proxy")


class FakeLaya:
    def __init__(self, mode="valid"):
        self.mode = mode
    def predict(self, text, questions, max_len):
        assert max_len == 512 and "gold" not in text
        routes = list(questions["retrieval_path"]["criteria"])
        probabilities = {route: (.8 if i == 0 else .1) for i, route in enumerate(routes)}
        if self.mode == "low": probabilities = {route: 1 / 3 for route in routes}
        if self.mode == "invalid": probabilities[routes[0]] = float("nan")
        return {"answers": {"retrieval_path": {"choice": routes[0], "probabilities": probabilities}},
                "usage": {"truncated": self.mode == "truncated"}}


def test_laya_order_study_low_confidence_invalid_and_truncation():
    request = RoutingRequest("What method is used?")
    router = LayaRouter(FakeLaya())
    study = router.permutation_study(request)
    assert study["choice_changed"] and len(study["decisions"]) == 3
    assert study["real_model"] is False
    for mode in ("low", "invalid", "truncated"):
        result = LayaRouter(FakeLaya(mode)).choose(request)
        assert result.route == "hybrid" and result.fallback
    with pytest.raises(ValueError):
        router.choose(request, order=("bm25", "hybrid"))
    with pytest.raises(ValueError):
        LayaRouter(FakeLaya(), min_confidence=2)
    with pytest.raises(ValueError, match="audited"):
        LayaRouter.load_local("missing", expected_sha256={})


def test_laya_local_loader_boundary_is_hash_checked_and_network_disabled(tmp_path, monkeypatch):
    import os
    import ragscope.routing.laya_adapter as module
    hashes = {}
    required = ["model.safetensors", "rl_agent_config.json", "encoder/config.json",
                "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]
    for name in required:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"offline-loader-fixture")
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    # Only this unit test replaces the production official weight digest.
    monkeypatch.setattr(module, "LAYA_WEIGHT_SHA256", hashes["model.safetensors"])
    calls = []
    def fake_load(path, **kwargs):
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert kwargs["device"] == "cpu" and kwargs["backend"] == "eager"
        calls.append(path)
        return FakeLaya()
    monkeypatch.setitem(sys.modules, "laya", SimpleNamespace(__version__="0.4.6", load=fake_load))
    monkeypatch.setenv("HF_HUB_OFFLINE", "prior-value")
    router = LayaRouter.load_local(tmp_path, expected_sha256=hashes)
    assert calls == [str(tmp_path.resolve())]
    assert router.identity["checkpoint_hashes_before"] == router.identity["checkpoint_hashes_after"]
    assert os.environ["HF_HUB_OFFLINE"] == "prior-value"
    monkeypatch.delitem(os.environ, "HF_HUB_OFFLINE")
    monkeypatch.setitem(sys.modules, "laya", SimpleNamespace(__version__="invalid", load=fake_load))
    with pytest.raises(ValueError, match="SDK"):
        LayaRouter.load_local(tmp_path, expected_sha256=hashes)
    assert "HF_HUB_OFFLINE" not in os.environ
    with pytest.raises(ValueError, match="every local"):
        LayaRouter.load_local(tmp_path, expected_sha256={"model.safetensors": hashes["model.safetensors"]})
    (tmp_path / "encoder/config.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash/path"):
        LayaRouter.load_local(tmp_path, expected_sha256=hashes)


def test_binary_logistic_roundtrip_and_untrained_guard(tmp_path):
    train, dev = labelled_fixtures()
    train, dev = [row for row in train if row["label"] != "hybrid_rerank"], [row for row in dev if row["label"] != "hybrid_rerank"]
    router = TfidfRouter(max_features=64, max_epochs=3)
    with pytest.raises(ValueError, match="not been trained"):
        router.choose(RoutingRequest("question"))
    with pytest.raises(ValueError, match="untrained"):
        router.save(tmp_path)
    router.fit(train, dev, data_kind="offline_fixture")
    probabilities = router.choose(RoutingRequest("semantic method approach")).probabilities
    assert probabilities["hybrid_rerank"] == 0
    assert sum(probabilities.values()) == pytest.approx(1)


def test_calibration_is_against_proxy_labels_not_claimed_truth():
    perfect = calibration([[1, 0, 0], [0, 1, 0], [0, 0, 1]], ROUTES)
    assert perfect["ece"] == 0 and perfect["brier"] == 0
    wrong = calibration([[1, 0, 0]], ["hybrid"])
    assert wrong["ece"] == 1 and wrong["brier"] == 2
    assert "proxy" in wrong["target"]
    with pytest.raises(ValueError):
        calibration([[.3, .3, .3]], ["bm25"])


def test_replay_includes_failure_and_distinguishes_latency_kind():
    query = fixture_query("q", "test")
    records = fixture_records(query)
    records[0].update(answer_f1=0, error={"type": "timeout"})
    rows, summary = replay(FixedRouter("bm25"), [query], records, query_ids=["q"], comparison_config_hash="fixture-config")
    assert summary["n"] == 1 and summary["answer_f1"] == 0 and summary["failure_rate"] == 1
    assert summary["latency_kind"] == "component_replay_estimate" and rows[0]["latency_s"] >= 1
    assert not deployment_gate(summary, summary, online_n=100)["deploy"]
    with pytest.raises(ValueError, match="all frozen"):
        replay(FixedRouter(), [query], records[:-1], query_ids=["q"], comparison_config_hash="fixture-config")


def test_online_ledger_is_actual_timing_not_zero_cost_replay(tmp_path):
    queries = [fixture_query(str(i), "dev") for i in range(30)]
    def execute(request, route):
        return {"real_model": True, "answer_f1": .5, "evidence_coverage": None, "error": None}
    path = tmp_path / "offline_fixture_online_ledger.jsonl"
    rows, summary = online(FixedRouter(), queries, execute, path)
    assert len(path.read_text().splitlines()) == 30
    assert summary["latency_kind"] == "actual_online" and all(row["latency_s"] > 0 for row in rows)
    assert summary["evidence_n"] == 0 and summary["evidence_coverage"] is None
    with pytest.raises(FileExistsError):
        online(FixedRouter(), queries, execute, path)
    with pytest.raises(ValueError):
        online(FixedRouter(), queries[:2], execute, tmp_path / "too_few.jsonl")
    best = dict(summary, answer_f1=.5, p50_latency_s=10)
    candidate = dict(summary, answer_f1=.49, p50_latency_s=9)
    assert deployment_gate(candidate, best, online_n=30)["deploy"]
    assert not deployment_gate(dict(candidate, answer_f1=.48), best, online_n=30)["deploy"]
