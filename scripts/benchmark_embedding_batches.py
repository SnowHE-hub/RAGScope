"""Bounded CPU request-array benchmark against the existing project service."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import psutil
import requests
from filelock import FileLock
from ragscope.experiments.ledger import append_jsonl, now
from ragscope.experiments.retrieval_campaign import read_jsonl, verify_corpus_files
from ragscope.indexing.identity import artifact, atomic_json, signature, verify_artifacts
from ragscope.indexing.local_embedding import HFJsonTokenizer, QUERY_INSTRUCTION


def normalized_vectors(payload, count, dimensions=1024):
    data = payload["data"]
    if len(data) != count or sorted(row["index"] for row in data) != list(range(count)):
        raise ValueError("Backend array length or input-index mapping differs")
    values = np.asarray([row["embedding"] for row in sorted(data, key=lambda row: row["index"])], dtype=np.float32)
    if values.shape != (count, dimensions) or not np.isfinite(values).all():
        raise ValueError("Backend embedding shape/finiteness failed")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero backend vector")
    return values / norms


def main(home, campaign):
    root = home / "runs" / (campaign + "_retrieval")
    output = root / "cpu_batch_benchmark"
    if (output / "RESULT.json").exists() or (output / "CONFIG.json").exists():
        raise ValueError("Bounded benchmark already attempted; inspect its evidence instead of repeating")
    service_path = home / "runs" / campaign / "embedding_service.json"
    service = json.loads(service_path.read_text(encoding="utf-8"))
    worker = json.loads((root / "worker.json").read_text(encoding="utf-8"))
    if psutil.pid_exists(worker["pid"]) and abs(psutil.Process(worker["pid"]).create_time() - worker["creation_time"]) < .01:
        raise RuntimeError("Original matrix worker is still running")
    process = psutil.Process(service["pid"])
    if abs(process.create_time() - service["creation_time"]) > .01 or process.cmdline() != service["command"]:
        raise RuntimeError("Project CPU service process identity differs")
    command = process.cmdline()
    if service["device"] != "cpu" or command[command.index("-ngl") + 1] != "0" or "--no-op-offload" not in command:
        raise RuntimeError("CPU-only backend identity required")
    for flag, expected in (("-t", "12"), ("-tb", "12"), ("-np", "1")):
        if command[command.index(flag) + 1] != expected:
            raise RuntimeError("Existing CPU scheduling identity differs")
    corpus = home / "corpora" / campaign
    verify_corpus_files(corpus, ["queries_dev.jsonl"])
    identity_path = home / "runs" / campaign / "embedding_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    tokenizer_path = home / "models/qwen3-embedding-0.6b/tokenizer.json"
    tokenizer = HFJsonTokenizer(tokenizer_path, identity["model_id"], identity["tokenizer_revision"])
    inputs, sources = [], []
    for group, length in (("F512_O0", 512), ("F1024_O0", 1024)):
        found = []
        for path in sorted((home / "indexes" / campaign / "dev_chunks").glob("*/chunk_manifest.json")):
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest["identity"]["group"]["id"] == group:
                found.append((path, manifest))
        if len(found) != 1 or not verify_artifacts(found[0][1]["artifacts"]):
            raise ValueError("Missing or changed fixed dev chunk source")
        path, manifest = found[0]
        rows = sorted((row for row in read_jsonl(path.parent / "chunks.jsonl") if row["n_tokens"] == length), key=lambda row: row["chunk_id"])
        random.Random(42).shuffle(rows)
        if len(rows) < 4:
            raise ValueError("Insufficient exact-length fixed dev chunks")
        for row in rows[:4]:
            if len(tokenizer.encode(row["text"])) != length:
                raise ValueError("Actual tokenizer count differs")
            inputs.append({"chunk_id": row["chunk_id"], "doc_id": row["doc_id"], "text": row["text"], "body_tokens": length, "group": group})
        sources.append(artifact(path))
    query = sorted(read_jsonl(corpus / "queries_dev.jsonl"), key=lambda row: row["query_id"])[0]
    config = {"schema_version": 1, "device": "cpu", "service": service,
              "model_identity": identity, "identity_artifact": artifact(identity_path),
              "tokenizer_artifact": artifact(tokenizer_path), "sources": sources,
              "query_source": artifact(corpus / "queries_dev.jsonl"),
              "input_selection": "sort chunk_id; independent seed42 shuffle per exact512/1024 group; first4 each",
              "inputs": inputs, "ranking_query": {"query_id": query["query_id"], "text": QUERY_INSTRUCTION + query["question"]},
              "batch_order": [1, 2, 4, 4, 2, 1], "repetitions_per_batch": 2,
              "reserve_bytes": int(1.25 * 1024**3), "max_abs_tolerance": 1e-6,
              "cosine_tolerance": 1e-6, "speedup_threshold": 1.15, "http_timeout_seconds": 180,
              "maximum_http_requests": 29, "cache_policy": "No shared embedding cache reads or writes",
              "usage_policy": "Backend usage retained at request level; no invented per-input allocation",
              "reference": "https://github.com/ggml-org/llama.cpp/blob/b11429/tools/server/README.md"}
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "CONFIG.json", config)
    config_hash = signature(config)
    request_number = 0
    timings, checks, vectors = [], [], {}

    def request(texts, label, batch_size):
        nonlocal request_number
        available = psutil.virtual_memory().available
        if available < config["reserve_bytes"]:
            raise MemoryError(f"RAM below declared reserve before request: {available}")
        request_number += 1
        if request_number > config["maximum_http_requests"]:
            raise RuntimeError("Bounded request budget exhausted")
        started = time.perf_counter()
        response = requests.post("http://127.0.0.1:11436/v1/embeddings",
                                 json={"input": texts[0] if batch_size == 1 else texts}, timeout=180)
        response.raise_for_status()
        payload = response.json()
        result = normalized_vectors(payload, len(texts))
        elapsed = time.perf_counter() - started
        row = {"request": request_number, "label": label, "batch_size": batch_size,
               "input_count": len(texts), "seconds": elapsed, "backend_usage": payload.get("usage"),
               "body_tokens": sum(len(tokenizer.encode(text)) for text in texts),
               "available_before": available, "available_after": psutil.virtual_memory().available,
               "input_hashes": [signature({"text": text}) for text in texts], "at": now(), "config_hash": config_hash}
        append_jsonl(output / "requests.jsonl", row)
        return result, elapsed

    status, error = "PASS", None
    try:
        with FileLock(str(root / "writer.lock"), timeout=0):
            requests.get("http://127.0.0.1:11436/health", timeout=10).raise_for_status()
            query_vector, _ = request([config["ranking_query"]["text"]], "fixed_query_warmup", 1)
            baseline = None
            for ordinal, batch in enumerate(config["batch_order"]):
                parts, elapsed = [], 0.0
                for start in range(0, len(inputs), batch):
                    part, duration = request([row["text"] for row in inputs[start:start+batch]], f"round{ordinal}", batch)
                    parts.append(part)
                    elapsed += duration
                matrix = np.vstack(parts)
                if baseline is None:
                    baseline = matrix
                diff = float(np.max(np.abs(matrix - baseline)))
                min_cosine = float(np.min(np.sum(matrix * baseline, axis=1)))
                ranking = np.argsort(-(matrix @ query_vector[0]), kind="stable").tolist()
                baseline_ranking = np.argsort(-(baseline @ query_vector[0]), kind="stable").tolist()
                passed = diff <= config["max_abs_tolerance"] and min_cosine >= 1-config["cosine_tolerance"] and ranking == baseline_ranking
                checks.append({"round": ordinal, "batch_size": batch, "max_abs_diff": diff,
                               "min_cosine": min_cosine, "ranking": ranking, "baseline_ranking": baseline_ranking, "passed": passed})
                timings.append({"round": ordinal, "batch_size": batch, "seconds": elapsed,
                                "body_tokens_per_second": sum(row["body_tokens"] for row in inputs)/elapsed})
                vectors[f"round{ordinal}_batch{batch}"] = matrix
                print(json.dumps({"round": ordinal, "batch_size": batch, "seconds": elapsed, "equivalence": passed}), flush=True)
            vectors["query"] = query_vector
    except Exception as exc:
        status, error = "RETAIN_SERIAL_BENCHMARK_FAILED", {"type": type(exc).__name__, "message": str(exc)}
    np.savez(output / "observed_vectors.npz", **vectors)
    totals = {batch: sum(row["seconds"] for row in timings if row["batch_size"] == batch) for batch in (1, 2, 4)}
    speedups = {batch: totals[1] / value for batch, value in totals.items() if value and len([r for r in timings if r["batch_size"] == batch]) == 2}
    eligible = [batch for batch in (2, 4) if status == "PASS" and speedups.get(batch, 0) >= 1.15
                and all(row["passed"] for row in checks if row["batch_size"] == batch)]
    chosen = max(eligible, key=lambda batch: speedups[batch]) if eligible else 1
    result = {"status": status, "error": error, "config_hash": config_hash, "completed_at": now(),
              "timings": timings, "equivalence_checks": checks, "speedups": speedups,
              "recommended_request_batch": chosen, "requests_observed": request_number,
              "scope_limit": "Only these eight fixed dev chunks and one ranking query; not a universal numerical guarantee",
              "implementation_action": "Retain current serial adapter" if chosen == 1 else "Eligible bounded scheduling optimization; adapter integration/provenance still required before use",
              "available_ram_after": psutil.virtual_memory().available,
              "service_still_running": process.is_running(), "service_pid": process.pid,
              "artifacts": [artifact(output / name) for name in ("CONFIG.json", "requests.jsonl", "observed_vectors.npz") if (output/name).exists()]}
    atomic_json(output / "RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path("D:/RAGScope_Runtime"))
    parser.add_argument("--campaign", default="20261014_v2")
    args = parser.parse_args()
    main(args.home, args.campaign)
