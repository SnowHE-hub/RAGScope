"""Offline fixtures verify statistical denominators, never model results."""
import json

import pytest

from ragscope.indexing.identity import signature
from scripts.summarize_retrieval_statistics_v2 import verified_rows


def fixture_run(tmp_path, queries, scores, errors=None):
    run_id = "fixture_only"
    (tmp_path / "records").mkdir()
    (tmp_path / "summaries").mkdir()
    config = {"query_ids": [q["query_id"] for q in queries]}
    rows = []
    for q, value, error in zip(queries, scores, errors or [None]*len(queries)):
        row = {"query_id": q["query_id"], "config_hash": signature(config),
               "scores": {"coverage": value}, "error": error}
        row["record_hash"] = signature(row)
        rows.append(row)
    (tmp_path / "records" / (run_id + ".jsonl")).write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (tmp_path / "records" / (run_id + ".config.json")).write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "summaries" / (run_id + ".json")).write_text(json.dumps({"n_total": len(rows)}), encoding="utf-8")
    return run_id


def test_original_article_cluster_and_failure_are_preserved(tmp_path):
    queries = [{"query_id": "a", "answerability": "answerable", "metadata": {
        "gold_doc_id_evaluation_only": "paragraph1", "article_cluster_id": "article"}},
        {"query_id": "b", "answerability": "unanswerable", "doc_id": "other"}]
    run_id = fixture_run(tmp_path, queries, [0, None], ["real failure fixture", None])
    rows, evidence = verified_rows(tmp_path, run_id, queries, ["coverage"])
    assert rows[0]["doc_id"] == "article"
    assert rows[0]["scores"]["coverage"] == 0
    assert rows[1]["scores"]["coverage"] is None
    assert evidence["n_total"] == 2 and evidence["n_failed"] == 1


def test_query_denominator_and_tamper_are_rejected(tmp_path):
    queries = [{"query_id": "a", "answerability": "answerable", "doc_id": "paper"}]
    run_id = fixture_run(tmp_path, queries, [.5])
    with pytest.raises(ValueError, match="denominator"):
        verified_rows(tmp_path, run_id, queries + [{**queries[0], "query_id": "b"}], ["coverage"])
    path = tmp_path / "records" / (run_id + ".jsonl")
    row = json.loads(path.read_text())
    row["scores"]["coverage"] = 1
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="Record hash"):
        verified_rows(tmp_path, run_id, queries, ["coverage"])


def test_unavailable_run_is_not_scored_as_zero(tmp_path):
    rows, evidence = verified_rows(tmp_path, "missing", [], ["coverage"])
    assert rows is None and evidence["status"] == "UNAVAILABLE"
