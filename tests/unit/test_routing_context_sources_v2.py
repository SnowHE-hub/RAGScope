"""Frozen sampler identity and split integrity, using offline source fixtures."""
import json

import pytest

from ragscope.experiments.retrieval_campaign import verified_routing_queries
from ragscope.indexing.identity import artifact, file_hash, signature
from scripts.freeze_generation_v2 import seeded_subset


def setup_source(tmp_path):
    corpus = tmp_path / "corpora/fixture"
    run = tmp_path / "runs/fixture"
    corpus.mkdir(parents=True); run.mkdir(parents=True)
    split_manifest, ids = {"files": {}}, {}
    for split, count in (("train", 200), ("dev", 60), ("test", 100)):
        population = [{"query_id": f"{split}-{i:03d}", "split": split,
                       "scope": {"doc_ids": ["doc-" + split]}} for i in range(count+1)]
        path = corpus / f"queries_{split}.jsonl"
        path.write_text('\n'.join(json.dumps(q) for q in population), encoding="utf-8")
        split_manifest["files"][path.name] = {"sha256": file_hash(path)}
        ids[split] = [q["query_id"] for q in seeded_subset(population, count)]
    split_path = corpus / "SPLIT_MANIFEST.json"
    split_path.write_text(json.dumps(split_manifest), encoding="utf-8")
    manifest = {"query_ids": ids, "seed": 42, "source": artifact(split_path), "version": "offline_fixture"}
    manifest["manifest_hash"] = signature(manifest)
    manifest_path = run / "ROUTING_QUERY_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, manifest


def test_exact_freezer_order_is_verified_not_resampled(tmp_path):
    path, manifest = setup_source(tmp_path)
    rows, evidence = verified_routing_queries(tmp_path, "fixture", manifest["manifest_hash"])
    assert {key: len(value) for key, value in rows.items()} == {"train": 200, "dev": 60, "test": 100}
    assert [q["query_id"] for q in rows["train"]] == manifest["query_ids"]["train"]
    assert evidence["new_sampling_performed"] is False
    # Even a recomputed manifest hash cannot substitute a differently ordered
    # population for the exact independently frozen sampler output.
    manifest["query_ids"]["dev"].reverse()
    manifest["manifest_hash"] = signature({k: v for k, v in manifest.items() if k != "manifest_hash"})
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="freezer subset"):
        verified_routing_queries(tmp_path, "fixture", manifest["manifest_hash"])


def test_routing_source_and_external_expected_hash_must_match(tmp_path):
    _, manifest = setup_source(tmp_path)
    with pytest.raises(ValueError, match="manifest hash"):
        verified_routing_queries(tmp_path, "fixture", "wrong-pinned-hash")
    path = tmp_path / "corpora/fixture/queries_train.jsonl"
    path.write_text(path.read_text() + '\n{}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="source query file changed"):
        verified_routing_queries(tmp_path, "fixture", manifest["manifest_hash"])
