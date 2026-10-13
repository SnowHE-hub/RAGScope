"""Freeze 100 actual short SQuAD2 questions and retain the full dev paragraph corpus."""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from ragscope.ingestion.qasper import digest
from ragscope.ingestion.squad_v2 import SOURCE, convert_squad, select_squad_short
from scripts.prepare_v2_data import download, immutable_json, immutable_jsonl, sha_file


def main(home, campaign):
    home = Path(home)
    base = home / "corpora" / campaign / "squad_e4"
    base.mkdir(parents=True, exist_ok=True)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(home / "cache/tiktoken")
    source = download(SOURCE, base / "raw" / "dev-v2.0.json", base / "download_attempts.jsonl")
    config = json.loads((Path(__file__).resolve().parents[1] / "configs/v2/squad_e4.json").read_text(encoding="utf-8"))
    immutable_json(base / "FROZEN_CONFIG.json", {"config": config, "config_hash": digest(config)})
    import tiktoken
    encoder = tiktoken.get_encoding("cl100k_base")
    docs, all_queries, quarantined = convert_squad(json.loads((base / "raw/dev-v2.0.json").read_text(encoding="utf-8")),
                                                  lambda text: len(encoder.encode(text, disallowed_special=())))
    selected = select_squad_short(all_queries, config["target_n"], config["seed"])
    immutable_jsonl(base / "documents.jsonl", docs)
    immutable_jsonl(base / "queries_test.jsonl", selected)
    immutable_jsonl(base / "all_queries_audit.jsonl", all_queries)
    immutable_jsonl(base / "quarantine.jsonl", quarantined)
    manifest = {"status": "PASS", "source": source, "config_hash": digest(config), "corpus_documents": len(docs),
                "source_questions": len(all_queries), "quarantined": len(quarantined), "selected_questions": len(selected),
                "selected_query_ids": sorted(q["query_id"] for q in selected), "selected_answerable": len(selected),
                "selected_document_ids": sorted({q["metadata"]["gold_doc_id_evaluation_only"] for q in selected}),
                "selected_article_clusters": len({q["metadata"]["article_cluster_id"] for q in selected}),
                "max_selected_reference_tokens": max(q["metadata"]["reference_n_tokens"] for q in selected),
                "official_split": "dev", "evaluation_partition": "E4_frozen_100", "is_official_hidden_test": False,
                "normalization": "identity_unicode_v1", "seed": config["seed"], "retrieval_corpus": "all_official_dev_paragraphs",
                "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha_file(p)} for p in base.glob("*.jsonl") if p.name != "download_attempts.jsonl"}}
    immutable_json(base / "SPLIT_MANIFEST.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("status", "corpus_documents", "source_questions", "quarantined", "selected_questions", "selected_article_clusters", "max_selected_reference_tokens")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", default=os.environ.get("RAGSCOPE_HOME", "D:/RAGScope_Runtime"))
    parser.add_argument("--campaign", default="20261014_v2")
    args = parser.parse_args()
    main(args.home, args.campaign)
