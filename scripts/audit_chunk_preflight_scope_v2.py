"""Expose historical preflight labels without rewriting immutable indexes."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragscope.indexing.identity import artifact, atomic_json, verify_artifacts


def audit(home, campaign):
    sources = {}
    corpus = home / "corpora" / campaign
    for path in (corpus / "documents.jsonl", corpus / "squad_e4/documents.jsonl"):
        if path.exists():
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    doc = json.loads(line)
                    sources[doc["doc_id"]] = doc["split"]
    rows = []
    for path in sorted((home / "indexes" / campaign).rglob("chunk_manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not verify_artifacts(manifest["artifacts"]):
            raise ValueError("Chunk artifacts changed")
        preflight = manifest["preflight"]
        sampled = [d["doc_id"] for d in manifest["document_checks"][:preflight["documents"]]]
        identity_scope = manifest["identity"]["split"]
        rows.append({"group": manifest["identity"]["group"]["id"], "manifest": artifact(path),
                     "identity_scope": identity_scope, "reported_preflight_split": preflight["split"],
                     "metadata_label_matches_identity": identity_scope == preflight["split"],
                     "actual_preflight_document_ids": sampled,
                     "actual_source_document_splits": {doc_id: sources[doc_id] for doc_id in sampled},
                     "index_identity_or_results_modified": False})
    mismatches = sum(not row["metadata_label_matches_identity"] for row in rows)
    result = {"status": "PASS_WITH_METADATA_LIMITATION" if mismatches else "PASS", "indexes_checked": len(rows),
              "historical_label_mismatches": mismatches, "rows": rows,
              "interpretation": "Old all-corpus manifests hardcoded preflight.split=dev. Their actual immutable identity and sampled source IDs above define scope; index/data/quality results are unchanged. Future manifests use actual identity scope and explicit sample IDs. No timing preflight result selected a retrieval configuration."}
    atomic_json(home / "runs" / (campaign + "_retrieval") / "CHUNK_PREFLIGHT_SCOPE_AUDIT.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path("D:/RAGScope_Runtime"))
    parser.add_argument("--campaign", default="20261014_v2")
    args = parser.parse_args()
    result = audit(args.home, args.campaign)
    print(json.dumps({k: result[k] for k in ("status", "indexes_checked", "historical_label_mismatches")}))
