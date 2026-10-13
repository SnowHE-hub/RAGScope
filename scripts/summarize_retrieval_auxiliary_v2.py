"""Source-verified auxiliary chunk ranking; never modifies real experiment rows."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragscope.experiments.ledger import RecordStore
from ragscope.indexing.identity import artifact, atomic_json, file_hash, signature, verify_artifacts


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def rank_metrics(ranked_ids, relevant_ids, *, answerability, failed=False, k=10):
    if answerability == "unanswerable":
        return {"mrr_at_10": None, "ndcg_at_10": None}
    if len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("Ranking contains repeated chunk identities")
    if not relevant_ids:
        raise ValueError("Answerable source has no relevant indexed chunks")
    if failed:
        return {"mrr_at_10": 0.0, "ndcg_at_10": 0.0}
    gains = [float(cid in relevant_ids) for cid in ranked_ids[:k]]
    reciprocal = next((1 / rank for rank, gain in enumerate(gains, 1) if gain), 0.0)
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant_ids)) + 1))
    return {"mrr_at_10": reciprocal, "ndcg_at_10": dcg / ideal}


def indexed_units(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["signature"] != signature(manifest["identity"]) or not verify_artifacts(manifest["artifacts"]):
        raise ValueError("Index source identity/hash changed")
    chunks = json.loads(manifest_path.with_name("chunks.json").read_text(encoding="utf-8"))
    if len(chunks) != manifest["count"]:
        raise ValueError("Index source count differs")
    unit_chunks, by_doc, ids = defaultdict(set), defaultdict(set), set()
    for chunk in chunks:
        cid = chunk["chunk_id"]
        if cid in ids:
            raise ValueError("Index has repeated chunk identities")
        ids.add(cid)
        by_doc[chunk["doc_id"]].add(cid)
        for span in chunk["unit_spans"]:
            if span["end"] > span["start"]:
                unit_chunks[span["unit_id"]].add(cid)
    return unit_chunks, by_doc, ids


def report(home, campaign):
    run = home / "runs" / (campaign + "_retrieval")
    protocol_path = Path(__file__).resolve().parents[1] / "configs/v2/retrieval_auxiliary.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    corpus = home / "corpora" / campaign
    queries = {}
    source_artifacts = []
    known_data_manifests = set()
    for path in [*(corpus / f"queries_{s}.jsonl" for s in ("train", "dev", "test")),
                 corpus / "squad_e4/queries_test.jsonl"]:
        split_path = path.parent / "SPLIT_MANIFEST.json"
        split = json.loads(split_path.read_text(encoding="utf-8"))
        if file_hash(path) != split["files"][path.name]["sha256"]:
            raise ValueError("Frozen query source changed")
        known_data_manifests.add(file_hash(split_path))
        source_artifacts.extend([artifact(path), artifact(split_path)])
        for query in read_jsonl(path):
            if query["query_id"] in queries:
                raise ValueError("Query ID repeated across datasets/splits")
            queries[query["query_id"]] = query
    manifests = {}
    for path in (home / "indexes" / campaign).rglob("manifest.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("identity", {}).get("engine") in {"ArrayIndex-v2", "ScopedArrayIndex-v1-full-reference-bm25"}:
            if manifest["identity"].get("data_manifest_sha256") not in known_data_manifests:
                raise ValueError("Index points to a different data split manifest")
            if manifest["signature"] in manifests:
                raise ValueError("Index signature resolves to multiple paths")
            manifests[manifest["signature"]] = path
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in (run / "summaries").glob("*.json")]
    grouped = defaultdict(list)
    for summary in summaries:
        grouped[summary["index_signature"]].append(summary)
    outputs = []
    for index_signature, summaries in grouped.items():
        manifest_path = manifests[index_signature]
        unit_chunks, by_doc, all_ids = indexed_units(manifest_path)
        for summary in summaries:
            run_id = summary["run_id"]
            path = run / "records" / (run_id + ".jsonl")
            config_path = path.with_suffix(".config.json")
            original_config = json.loads(config_path.read_text(encoding="utf-8"))
            rows = read_jsonl(path)
            expected = set(original_config["query_ids"])
            if len(rows) != summary["n_total"] or len(rows) != len(expected) or {r["query_id"] for r in rows} != expected:
                raise ValueError("Completed source denominator differs: " + run_id)
            side_config = {"protocol": protocol, "protocol_sha256": file_hash(protocol_path),
                           "source": artifact(path), "source_config": artifact(config_path),
                           "index_manifest": artifact(manifest_path)}
            destination = run / "auxiliary" / (run_id + ".jsonl")
            store = RecordStore(destination, side_config, max_records=len(rows))
            for row in rows:
                if row["config_hash"] != signature(original_config) or row["record_hash"] != signature({k: v for k, v in row.items() if k != "record_hash"}):
                    raise ValueError("Original row/config hash mismatch")
                if row["query_id"] in store.ids:
                    continue
                query = queries[row["query_id"]]
                relevant = set().union(*(unit_chunks[uid] for ref in query["evidence_reference_sets"] for uid in ref))
                scope = row["scope"]
                eligible = all_ids if scope.get("kind") == "global" else by_doc[scope["doc_ids"][0]]
                relevant &= eligible
                reranked = summary["method"] == "hybrid_rerank"
                if reranked and "selected_retrieval" not in row:
                    raise ValueError("Rerank order unavailable; cannot substitute upstream Hybrid order")
                ranked = row["selected_retrieval"] if reranked else row["raw_retrieval"][:10]
                ranked_ids = [item["chunk_id"] for item in ranked]
                if set(ranked_ids) - eligible:
                    raise ValueError("Ranked item is outside actual index/scope")
                values = rank_metrics(ranked_ids, relevant, answerability=query["answerability"], failed=bool(row.get("error")))
                store.add(row["query_id"], {"query_id": row["query_id"], "source_record_hash": row["record_hash"],
                    "run_id": run_id, "experiment_type": "offline_auxiliary_analysis", "scores": values,
                    "answerability": query["answerability"], "failed": bool(row.get("error")),
                    "ranked_chunk_ids": ranked_ids, "relevant_indexed_chunks": len(relevant),
                    "rank_source": "selected_retrieval" if reranked else "raw_retrieval_first10"})
            metrics = {}
            for metric in ("mrr_at_10", "ndcg_at_10"):
                values = [r["scores"][metric] for r in store.records if r["scores"][metric] is not None]
                metrics[metric] = {"mean": sum(values) / len(values) if values else None, "n_applicable": len(values)}
            outputs.append({"run_id": run_id, "n_total": len(store.records), "n_failed": sum(r["failed"] for r in store.records),
                            "metrics": metrics, "records": artifact(destination)})
    result = {"status": "PASS", "protocol": protocol, "protocol_sha256": file_hash(protocol_path),
              "completed_runs": len(outputs), "actual_rows": sum(r["n_total"] for r in outputs), "runs": outputs,
              "query_source_artifacts": source_artifacts,
              "unfinished_runs_included": False, "test_used_for_selection": False}
    atomic_json(run / "RETRIEVAL_AUXILIARY_METRICS.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path("D:/RAGScope_Runtime"))
    parser.add_argument("--campaign", default="20261014_v2")
    args = parser.parse_args()
    result = report(args.home, args.campaign)
    print(json.dumps({k: result[k] for k in ("status", "completed_runs", "actual_rows")}))
