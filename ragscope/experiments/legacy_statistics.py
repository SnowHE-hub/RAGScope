"""Descriptive E0 intervals from unchanged rows and verified original articles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ragscope.evaluation.v2 import cluster_bootstrap
from ragscope.experiments.ledger import read_records
from ragscope.indexing.identity import artifact, atomic_json, signature, verify_artifacts

METRICS = ("recall5", "hit5", "recall10", "mrr", "ndcg10")


def article_mapping(queries, documents, raw):
    """Match the archived loader's str.strip context/title/question/answer exactly."""
    wanted = {query["gold_doc_id"] for query in queries}
    docs = {doc["doc_id"]: doc for doc in documents if doc["doc_id"] in wanted}
    if set(docs) != wanted:
        raise ValueError("Original SQuAD paragraph mapping is incomplete")
    candidates = {}
    for position, article in enumerate(raw["data"]):
        cluster = "squad_article:" + signature({"title": article["title"], "position": position})
        for paragraph in article["paragraphs"]:
            # Archived original ingestion/loaders.py:67 explicitly used strip().
            # Do not normalize interior whitespace or perform fuzzy text matching.
            key = (article["title"], paragraph["context"].strip())
            candidates.setdefault(key, []).append((cluster, paragraph["qas"]))
    output = []
    for query in queries:
        doc = docs[query["gold_doc_id"]]
        exact = candidates.get((doc["title"], doc["text"]), [])
        matches = [(cluster, qa["id"]) for cluster, qas in exact for qa in qas
                   if qa["question"] == query["question"] and not qa.get("is_impossible", False)
                   and query["answer"] in {answer["text"] for answer in qa["answers"]}]
        if not matches or len({cluster for cluster, _ in matches}) != 1:
            raise ValueError(f"Missing/ambiguous exact article provenance: {query['query_id']}")
        output.append({"query_id": query["query_id"], "paragraph_doc_id": doc["doc_id"],
                       "article_title": doc["title"], "cluster_id": matches[0][0],
                       "original_qa_ids": sorted({qid for _, qid in matches}),
                       "canonical_paragraph_sha256": signature(doc["text"]),
                       "question_sha256": signature(query["question"]),
                       "match": "exact_title_historical_str_strip_context_question_and_annotated_answer"})
    return output


def summarize(home, campaign):
    home = Path(home)
    corpus = home / "corpora" / campaign
    result_dir = home / "runs" / (campaign + "_legacy_reconstructed")
    source_dir = home / "runs" / (campaign + "_legacy") / "reconstructed"
    validation = json.loads((result_dir / "INDEPENDENT_VALIDATION.json").read_text(encoding="utf-8"))
    if validation.get("status") != "PASS" or not validation.get("original_indexes_unchanged"):
        raise ValueError("E0 needs its completed independent integrity/metric audit")
    legacy = json.loads((corpus / "legacy_manifest.json").read_text(encoding="utf-8"))
    required_names = {"squad_corpus.jsonl", "squad_eval_queries.jsonl", "train-v2.0.json"}
    sources = {Path(item["snapshot"]).name: {"path": item["snapshot"], "bytes": item["bytes"], "sha256": item["sha256"]}
               for item in legacy["files"] if Path(item["snapshot"]).name in required_names}
    if set(sources) != required_names or not verify_artifacts(list(sources.values())):
        raise ValueError("Original public-data snapshot files changed or are missing")
    provenance = json.loads((source_dir / "PROVENANCE.json").read_text(encoding="utf-8"))
    if not verify_artifacts([provenance["bm25_queries"]]):
        raise ValueError("Frozen legacy 200-query population changed")
    queries = [row["query"] for row in json.loads((source_dir / "queries_bm25_top20.json").read_text(encoding="utf-8"))]
    query_index = {query["query_id"]: query for query in queries}
    if len(queries) != 200 or len(query_index) != 200:
        raise ValueError("Legacy population must retain all 200 unique original queries")
    with Path(sources["squad_eval_queries.jsonl"]["path"]).open(encoding="utf-8") as handle:
        originals = {row["query_id"]: row for line in handle if (row := json.loads(line))["query_id"] in query_index}
    if originals != query_index:
        raise ValueError("Frozen reconstruction queries differ from original public-data snapshot")
    with Path(sources["squad_corpus.jsonl"]["path"]).open(encoding="utf-8") as handle:
        documents = [json.loads(line) for line in handle]
    raw = json.loads(Path(sources["train-v2.0.json"]["path"]).read_text(encoding="utf-8"))
    mapping = article_mapping(queries, documents, raw)
    del documents, raw
    clusters = {row["query_id"]: row["cluster_id"] for row in mapping}
    groups, record_sources = {}, []
    for method in ("dense", "hybrid"):
        evidence = validation["actual_results"][method]
        if not verify_artifacts([evidence["records"], evidence["config"], evidence["summary"]]):
            raise ValueError("Completed E0 records/config/summary changed after independent validation")
        path = Path(evidence["records"]["path"])
        config = json.loads(Path(evidence["config"]["path"]).read_text(encoding="utf-8"))
        rows = read_records(path)
        if len(rows) != 200 or {row["query_id"] for row in rows} != set(query_index):
            raise ValueError("E0 formal rows omit or duplicate frozen questions")
        transformed = []
        for row in rows:
            query = query_index[row["query_id"]]
            if (not row.get("record_hash") or row["config_hash"] != signature(config)
                    or row["doc_id"] != query["gold_doc_id"] or row["question"] != query["question"]
                    or row.get("real_model") is not True):
                raise ValueError("E0 record lacks exact frozen identity or real-model provenance")
            transformed.append({"query_id": row["query_id"], "doc_id": clusters[row["query_id"]],
                                "scores": row["metrics"]})
        transformed.sort(key=lambda row: row["query_id"])
        intervals = {name: cluster_bootstrap(transformed, name, n_resamples=2000, seed=42) for name in METRICS}
        for name, value in intervals.items():
            if abs(value["mean"] - evidence["metrics"][name]) > 1e-12:
                raise ValueError("Descriptive interval mean differs from independently recomputed metric")
            value["method"] = "SQuAD_original_article_cluster_percentile_row_weighted_mean"
        groups[method] = {"n_total": 200, "n_failed": sum(bool(row.get("error")) for row in rows),
                          "descriptive_ci95": intervals}
        record_sources.extend([evidence["records"], evidence["config"], evidence["summary"]])
    result = {"status": "COMPLETE", "analysis_type": "descriptive_only_no_significance_claim_no_new_experiment",
              "reconstruction_label": "legacy_reconstructed_fixed_revision", "original_weight_revision": "UNKNOWN",
              "query_count": 200, "n_article_clusters": len(set(clusters.values())), "seed": 42,
              "n_resamples": 2000, "confidence": .95, "article_mapping": mapping, "groups": groups,
              "sources": [artifact(corpus / "legacy_manifest.json"), *sources.values(), provenance["bm25_queries"],
                          artifact(source_dir / "source_original/ragscope/ingestion/loaders.py"),
                          artifact(result_dir / "INDEPENDENT_VALIDATION.json"), *record_sources],
              "limitations": ["Original legacy relevance is paragraph-document hit, not v2 evidence coverage.",
                              "Historical model weights are unknown; pinned reconstruction uses exact cosine instead of HNSW.",
                              "Intervals describe two already observed runs, are not adjusted simultaneous intervals or causal improvements.",
                              "Article mapping uses the preserved original train-v2.0.json snapshot, not a newly downloaded dataset.",
                              "A cluster repeated in bootstrap keeps all its sampled questions together; statistic remains question-weighted."]}
    result["analysis_hash"] = signature(result)
    output = result_dir / "E0_ARTICLE_CLUSTER_CI.json"
    if output.exists() and json.loads(output.read_text(encoding="utf-8")) != result:
        raise ValueError("Existing immutable E0 statistical analysis differs")
    atomic_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", default="D:/RAGScope_Runtime")
    parser.add_argument("--campaign", default="20261014_v2")
    args = parser.parse_args()
    result = summarize(args.home, args.campaign)
    print(json.dumps({key: result[key] for key in ("status", "query_count", "n_article_clusters", "groups")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
