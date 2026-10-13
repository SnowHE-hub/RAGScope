"""Export fixed source-coordinate visualization data, independently of scores."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragscope.indexing.identity import artifact, atomic_json, file_hash, verify_artifacts


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def export(home, campaign):
    corpus = home / "corpora" / campaign
    split_path = corpus / "SPLIT_MANIFEST.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    for name in ("documents.jsonl", "queries_dev.jsonl"):
        if file_hash(corpus / name) != split["files"][name]["sha256"]:
            raise ValueError("Frozen source changed")
    questions = list(rows(corpus / "queries_dev.jsonl"))
    ids = sorted({q["doc_id"] for q in questions})
    random.Random(42).shuffle(ids)
    wanted = ids[:3]
    documents = {d["doc_id"]: d for d in rows(corpus / "documents.jsonl") if d["doc_id"] in wanted}
    output = []
    source_artifacts = [artifact(split_path), artifact(corpus / "documents.jsonl"), artifact(corpus / "queries_dev.jsonl")]
    for doc_id in wanted:
        doc = documents[doc_id]
        units = []
        for unit in doc["evidence_units"]:
            start, end = unit["canonical_start"], unit["canonical_end"]
            if doc["text"][start:end] != unit["text"]:
                raise ValueError("Evidence source coordinates differ")
            units.append({k: unit[k] for k in ("unit_id", "section_id", "paragraph_index", "canonical_start", "canonical_end")})
        output.append({"doc_id": doc_id, "title": doc["title"], "canonical_text": doc["text"],
                       "canonical_chars": len(doc["text"]), "source_revision": doc["source_revision"],
                       "evidence_units": units, "groups": {},
                       "evaluation_only_annotations": [{k: q[k] for k in ("query_id", "question", "answerability", "evidence_reference_sets")}
                           for q in sorted((q for q in questions if q["doc_id"] == doc_id), key=lambda q: q["query_id"])[:2]]})
    by_id = {d["doc_id"]: d for d in output}
    groups = []
    for path in sorted((home / "indexes" / campaign / "dev_chunks").glob("*/chunk_manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["identity"]["data_manifest_sha256"] != file_hash(split_path) or not verify_artifacts(manifest["artifacts"]):
            raise ValueError("Chunk source manifest changed")
        group = manifest["identity"]["group"]
        groups.append(group["id"])
        source_artifacts.append(artifact(path))
        for doc in output:
            doc["groups"][group["id"]] = {"parameters": group, "children": [], "parents": []}
        for name, target in (("chunks.jsonl", "children"), ("parents.jsonl", "parents")):
            for chunk in rows(path.with_name(name)):
                if chunk["doc_id"] not in by_id:
                    continue
                doc = by_id[chunk["doc_id"]]
                start, end = chunk["canonical_start"], chunk["canonical_end"]
                if doc["canonical_text"][start:end] != chunk["text"]:
                    raise ValueError("Chunk source coordinates differ")
                doc["groups"][group["id"]][target].append({key: chunk[key] for key in (
                    "chunk_id", "parent_id", "canonical_start", "canonical_end", "n_tokens", "unit_spans")})
    if len(groups) != 11 or len(set(groups)) != 11:
        raise ValueError("Source map requires the exact eleven frozen dev chunk groups")
    result = {"schema": "canonical_coordinate_map_v1", "source_data_manifest_sha256": file_hash(split_path),
              "selection": "Sort unique dev paper IDs, Random(42).shuffle, first3; independent of outcomes and scores.",
              "coordinate_system": "Unicode code-point, global document, zero-based half-open intervals [start,end).",
              "annotation_use": "Offline visualization/evaluation only; never input to live retrieval.",
              "seed": 42, "groups": sorted(groups), "documents": output, "source_artifacts": source_artifacts}
    destination = home / "runs" / (campaign + "_retrieval") / "SOURCE_MAP.json"
    if destination.exists() and json.loads(destination.read_text(encoding="utf-8")) != result:
        raise ValueError("Frozen source visualization data differs; preserve existing output")
    atomic_json(destination, result)
    receipt = {"status": "PASS", "documents": len(output), "groups": len(groups),
               "children": sum(len(g["children"]) for d in output for g in d["groups"].values()),
               "parents": sum(len(g["parents"]) for d in output for g in d["groups"].values()),
               "evidence_units": sum(len(d["evidence_units"]) for d in output), "artifact": artifact(destination)}
    atomic_json(destination.with_name("SOURCE_MAP_MANIFEST.json"), receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path("D:/RAGScope_Runtime"))
    parser.add_argument("--campaign", default="20261014_v2")
    args = parser.parse_args()
    print(json.dumps(export(args.home, args.campaign)))
