"""Read-only BM25 legacy replay; old raw rows/reports never overwritten."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import random
import time
from ragscope.experiments.runtime import configure, runtime_home
from ragscope.experiments.ledger import Campaign, RecordStore
from ragscope.indexing.identity import file_hash, atomic_json
from ragscope.indexing.sparse_index import BM25Index
from ragscope.evaluation.retrieval_metrics import recall_at_k, hit_rate_at_k, mrr, ndcg_at_k

def main():
    configure(cpu_only=True)
    project = Path(__file__).resolve().parents[1]
    data = project / "data/processed/eval_queries.jsonl"
    index_path = project / "data/indexes/bm25_fixed.pkl"
    queries = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines()]
    selected = random.Random(42).sample(queries, min(200, len(queries)))
    config = {"seed": 42, "query_ids": [q["query_id"] for q in selected], "data_hash": file_hash(data),
              "index_hash": file_hash(index_path), "method": "legacy_bm25", "trusted_source": "existing_user_project_only"}
    with Campaign(runtime_home(), "20261014_v2_legacy") as campaign:
        if not campaign.begin("E0.BM25", config, [sys.executable, "scripts/run_legacy_v2.py"]):
            return
        index = BM25Index.load(str(index_path), trusted_legacy=True)
        store = RecordStore(campaign.path / "bm25.jsonl", config)
        for query in selected:
            if query["query_id"] in store.ids:
                continue
            start = time.perf_counter()
            try:
                hits = index.search(query["question"], k=10)
                retrieved = [hit["doc_id"] for hit in hits]
                gold = {query["gold_doc_id"]}
                metrics = {"recall5": recall_at_k(retrieved, gold, 5), "hit5": hit_rate_at_k(retrieved, gold, 5),
                           "recall10": recall_at_k(retrieved, gold, 10), "mrr": mrr(retrieved, gold), "ndcg10": ndcg_at_k(retrieved, gold, 10)}
                error = None
            except Exception as exc:
                hits, metrics, error = [], {key: 0 for key in ("recall5", "hit5", "recall10", "mrr", "ndcg10")}, str(exc)
            store.add(query["query_id"], {"query_id": query["query_id"], "doc_id": query["gold_doc_id"], "question": query["question"],
                       "experiment_type": "real_retrieval_no_generator", "raw_retrieval": hits, "metrics": metrics,
                       "error": error, "seconds": time.perf_counter() - start})
        summary = {"run_id": "20261014_v2_legacy", "n": len(store.records), "failures": sum(bool(r["error"]) for r in store.records),
                   "metrics": {key: sum(r["metrics"][key] for r in store.records) / len(store.records) for key in store.records[0]["metrics"]},
                   "note": "Single relevant document makes Recall@5 equal HitRate@5. Old CSV retained; its mismatched column is not repaired."}
        path = campaign.path / "summary.json"
        atomic_json(path, summary)
        campaign.finish("E0.BM25", "PASS", [store.path, path], summary)
        print(json.dumps(summary, ensure_ascii=False))

if __name__ == "__main__":
    main()
