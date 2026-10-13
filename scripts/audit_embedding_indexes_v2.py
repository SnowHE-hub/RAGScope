"""Read-only checks of completed real embedding index arrays and token budgets."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def audit(home, campaign):
    root = home / "indexes" / campaign
    model = json.loads((home / "runs" / campaign / "embedding_identity.json").read_text(encoding="utf-8"))
    tokenizer_path = home / "models/qwen3-embedding-0.6b/tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    cache = home / "cache/ragscope/v2_embeddings" / signature(model)
    observations, unique_usage, missing_usage = [], {}, set()
    for manifest_path in sorted(root.glob("*/*/dense_*/manifest.json")):
        directory = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity = manifest["identity"]
        assert identity["embedding"] == model
        assert identity["tokenizer"]["tokenizer_file_sha256"] == sha(tokenizer_path)
        assert manifest["signature"] == signature(identity)
        for item in manifest["artifacts"]:
            path = Path(item["path"])
            assert path.resolve().parent == directory.resolve()
            assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"]
        chunks = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        if identity.get("engine") == "ScopedArrayIndex-v1-full-reference-bm25":
            assert identity["scope"] == "explicit_paper_only" and manifest["scoped_only"] is True
            selected = identity["selected_doc_ids"]
            assert selected == sorted(set(selected)) and signature(selected) == identity["selected_doc_ids_sha256"]
            reference = identity["sparse_reference"]
            for source in (reference["chunks"], reference["chunk_manifest"]):
                path = Path(source["path"])
                assert path.stat().st_size == source["bytes"] and sha(path) == source["sha256"]
            full = [json.loads(line) for line in Path(reference["chunks"]["path"]).read_text(encoding="utf-8").splitlines() if line.strip()]
            assert len(full) == reference["count"] and signature(full) == reference["chunks_content_sha256"]
            assert chunks == [row for row in full if row["doc_id"] in set(selected)]
            assert manifest["neural_vector_count"] == len(chunks)
            del full
        vectors = np.load(directory / "vectors.npy", allow_pickle=False, mmap_mode="r")
        assert vectors.shape == (manifest["count"], model["dimensions"])
        assert len(chunks) == len(vectors) and vectors.dtype == np.float32
        assert np.isfinite(vectors).all()
        norms = np.linalg.norm(vectors, axis=1)
        max_norm_error = float(np.max(np.abs(norms - 1)))
        assert max_norm_error <= 1e-5
        group = identity["group"]
        limit = group.get("chunk_size", group.get("max_tokens", group.get("child_tokens")))
        keys, body_counts, observed = [], [], []
        for chunk in chunks:
            count = len(tokenizer.encode(chunk["text"], add_special_tokens=False).ids)
            assert count == chunk["n_tokens"] and count <= limit
            body_counts.append(count)
            key = signature({"text": chunk["text"], "model": model})
            keys.append(key)
            usage_path = cache / (key + ".usage.json")
            if usage_path.exists():
                usage = json.loads(usage_path.read_text(encoding="utf-8"))
                assert usage["key"] == key and usage["body_tokens"] == count
                backend = usage.get("usage")
                if backend and backend.get("prompt_tokens") is not None:
                    assert 0 < backend["prompt_tokens"] <= 1100
                    unique_usage[key] = {"body_tokens": count, "backend_prompt_tokens": backend["prompt_tokens"]}
                    observed.append(backend["prompt_tokens"])
                else:
                    missing_usage.add(key)
            else:
                missing_usage.add(key)
        samples = random.Random(42).sample(range(len(chunks)), min(20, len(chunks)))
        for index in samples:
            path = cache / (keys[index] + ".npy")
            receipt = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            assert receipt["model"] == model and receipt["key"] == keys[index]
            assert Path(receipt["array"]["path"]).resolve() == path.resolve()
            assert sha(path) == receipt["array"]["sha256"]
            assert np.array_equal(np.load(path, allow_pickle=False), vectors[index])
        observations.append({"manifest": str(manifest_path), "manifest_sha256": sha(manifest_path),
            "group": group["id"], "split": identity["split"], "vectors": len(chunks),
            "shape": list(vectors.shape), "dtype": str(vectors.dtype), "max_l2_norm_error": max_norm_error,
            "body_tokens_max": max(body_counts), "body_token_limit": limit,
            "backend_usage_available_rows": len(observed), "backend_usage_missing_rows": len(chunks)-len(observed),
            "observed_backend_prompt_tokens_max": max(observed) if observed else None,
            "alignment_cache_samples_seed42": [chunks[i]["chunk_id"] for i in samples]})
        del vectors, chunks
    result = {"status": "PASS", "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "completed_indexes": observations, "index_vectors_checked": sum(r["vectors"] for r in observations),
        "sampled_vector_cache_alignment_checks": sum(len(r["alignment_cache_samples_seed42"]) for r in observations),
        "unique_texts_with_observed_backend_usage": len(unique_usage),
        "unique_texts_without_backend_usage": len(missing_usage - unique_usage.keys()),
        "observed_backend_minus_body_tokens": sorted({r["backend_prompt_tokens"]-r["body_tokens"] for r in unique_usage.values()}),
        "model_identity": model, "unfinished_indexes_not_promoted": True,
        "note": "All completed arrays checked for shape, finite values, normalization, identity and hashes. Every body token count recomputed. Historical absent backend usage stays unobserved; cache alignment uses bounded seed42 samples, not all vectors."}
    output = home / "runs" / (campaign + "_retrieval") / "EMBEDDING_INDEX_AUDIT.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "completed_indexes": len(observations),
        "vectors": result["index_vectors_checked"], "unique_usage_available": len(unique_usage),
        "unique_usage_unavailable": result["unique_texts_without_backend_usage"]}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path("D:/RAGScope_Runtime"))
    parser.add_argument("--campaign", default="20261014_v2")
    args = parser.parse_args()
    audit(args.home, args.campaign)
