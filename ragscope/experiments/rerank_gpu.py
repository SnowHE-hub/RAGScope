"""Explicit FP32 GPU migration: preserve CPU300+partial29; run fixed GPU100+200+60."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from ragscope.experiments.ledger import Campaign, read_records
from ragscope.experiments.rerank_campaign import (
    read, jsonl, frozen_jobs, verified_source, find_index_material, run_job, load_reranker_with_receipt)
from ragscope.indexing.identity import artifact, atomic_json, signature, verify_artifacts

GPU_PREFIXES = ("E8", "ROUTE_CONTEXT_TRAIN", "ROUTE_CONTEXT_DEV")
IDLE_CODEX_PID = 11276
IDLE_CODEX_CREATED = 1791922058.5791745
IDLE_CODEX_EXE = r"C:\Program Files\WindowsApps\OpenAI.Codex_26.1007.2314.0_x64__2p2nqsd0c76g0\app\ChatGPT.exe"


def gpu_jobs(home, campaign):
    jobs = []
    for original in frozen_jobs(home, campaign):
        if original["prefix"] not in GPU_PREFIXES:
            continue
        job = dict(original)
        job["destination"] = original["destination"].with_stem(original["destination"].stem + "_gpu")
        jobs.append(job)
    if [j["prefix"] for j in jobs] != list(GPU_PREFIXES) or [len(j["queries"]) for j in jobs] != [100, 200, 60]:
        raise ValueError("GPU migration must retain exactly global100 and ROUTE200/60")
    return jobs


def migration_path(home, campaign):
    return Path(home) / "runs" / (campaign + "_retrieval") / "R3_GPU_MIGRATION.json"


def prepare_migration(home, campaign):
    """No torch import: immutable source and historical CPU receipts only."""
    from ragscope.retrieval.verified_reranker import verified_checkpoint, REVISION
    home = Path(home)
    run = home / "runs" / (campaign + "_retrieval")
    path = migration_path(home, campaign)
    if path.exists():
        return verify_migration(home, campaign)
    _, checkpoint = verified_checkpoint(home)
    protected, cpu_groups = [], []
    for job in frozen_jobs(home, campaign)[:3]:
        dest = job["destination"]
        rows = read_records(dest)
        required = 29 if job["prefix"] == "E8" else len(job["queries"])
        config = read(dest.with_suffix(".config.json"))
        if (len(rows) != required or any(r.get("error") or r.get("device") != "cpu"
                or not r.get("record_hash") or r.get("config_hash") != signature(config) for r in rows)):
            raise ValueError("Historical CPU300+partial29 differs; do not infer migration")
        paths = [dest, dest.with_suffix(".config.json")]
        if job["prefix"] != "E8":
            receipt = run / f'R3_GROUP_{job["prefix"]}_VALIDATION.json'
            checked = read(receipt)
            if checked.get("status") != "PASS" or checked.get("n_total") != required or checked.get("n_failed") != 0:
                raise ValueError("CPU sealed group must have an actual complete acceptance")
            paths += [receipt, run / "summaries" / (dest.stem + ".json")]
        protected += [artifact(p) for p in paths]
        cpu_groups.append({"run_id": dest.stem, "n_saved": len(rows), "n_failed": 0})
    for name in ("LOAD_STATE.json", "RESOURCE_PAUSE_PID30108.json"):
        protected.append(artifact(run / "R3_RUNTIME" / name))
    inherited = read(run / "R3_RUNTIME/LOAD_STATE.json")
    if inherited.get("attempts") != 2 or inherited.get("backend_failures") != 0:
        raise ValueError("Historical CPU load counters differ; preserve and diagnose")
    groups = []
    for job in gpu_jobs(home, campaign):
        _, rows = verified_source(job)
        groups.append({"prefix": job["prefix"], "run_id": job["destination"].stem,
                       "destination": str(job["destination"].resolve()),
                       "n_expected": len(rows), "query_ids": [q["query_id"] for q in job["queries"]],
                       "source_artifacts": [artifact(job["source"]), artifact(job["source"].with_suffix(".config.json"))],
                       "selection_provenance": job["provenance"]})
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")}
    if "+cu128" not in versions["torch"] or not versions["transformers"].startswith("5."):
        raise ValueError("Prepare GPU migration with the existing isolated train environment")
    result = {"status": "FROZEN_NOT_EXECUTED", "campaign": campaign,
              "reason": "Explicit user GPU authorization; CPU runtime RAM guard after329; deadline-bound device migration",
              "cpu_groups_preserved": cpu_groups, "protected_cpu_artifacts": protected,
              "historical_extra_partial_calls": 29, "historical_partial_in_final_gpu_denominator": False,
              "inherited_load_state": inherited, "checkpoint_manifest": checkpoint, "revision": REVISION,
              "device": "cuda", "dtype": "torch.float32", "tf32": False, "autocast": False,
              "backend": "transformers_sequence_classifier", "activation": "sigmoid", "attention_implementation": "eager",
              "cpu_threads": 4, "max_pair_tokens": 512, "window_overlap": 32, "batch_size": 2,
              "aggregation": "windowed_max", "backend_versions": versions,
              "groups": groups, "expected_gpu_rows": 360,
              "comparison_limit": "CPU TEST200 remains E2 input; full GPU global100 replaces partial CPU29 for global statistics. Routing CPU/GPU backend differences disclosed, never claim device equivalence.",
              "source_code_artifacts": [artifact(p) for p in (Path(__file__),
                  Path(__file__).with_name("rerank_campaign.py"),
                  Path(__file__).parents[1] / "retrieval/verified_reranker.py")]}
    result["migration_hash"] = signature(result)
    atomic_json(path, result)
    return result


def verify_migration(home, campaign):
    value = read(migration_path(home, campaign))
    if (value.get("migration_hash") != signature({k: v for k, v in value.items() if k != "migration_hash"})
            or value.get("expected_gpu_rows") != 360 or value.get("historical_extra_partial_calls") != 29
            or not verify_artifacts(value["protected_cpu_artifacts"] + value["source_code_artifacts"] + [value["checkpoint_manifest"]])):
        raise ValueError("GPU migration or protected CPU evidence changed")
    jobs = gpu_jobs(home, campaign)
    for job, group in zip(jobs, value["groups"], strict=True):
        if (group["run_id"] != job["destination"].stem or group["destination"] != str(job["destination"].resolve())
                or group["query_ids"] != [q["query_id"] for q in job["queries"]]
                or group["selection_provenance"] != job["provenance"]
                or group["source_artifacts"] != [artifact(job["source"]), artifact(job["source"].with_suffix(".config.json"))]):
            raise ValueError("GPU migration no longer matches the frozen full source groups")
    return value


def require_service_exit(home, campaign):
    """Inspect only our exact owned service; never stop another process."""
    import psutil
    service = read(Path(home) / "runs" / campaign / "generation_service.json")
    try:
        process = psutil.Process(service["pid"])
        if abs(process.create_time() - service["creation_time"]) < .01:
            raise RuntimeError("Owned generation service must actually exit before GPU R3")
    except psutil.NoSuchProcess:
        pass
    return {"generation_service": artifact(Path(home) / "runs" / campaign / "generation_service.json"),
            "exact_service_exited": True}


def require_empty_gpu():
    """No model resident; one observed idle Codex utility is a narrow exception."""
    import psutil
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True, timeout=15)
    memory = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,memory.total",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=15)
    fields = list(csv.reader(memory.stdout.strip().splitlines()))
    if len(fields) != 1 or len(fields[0]) != 4 or int(fields[0][0]) != 0:
        raise RuntimeError("Expected one explicitly observed GPU0")
    _, used, free, total = [int(v.strip()) for v in fields[0]]
    if used > 64 or free < 3072:
        raise MemoryError("GPU must have <=64 MiB total used and >=3 GiB free before R3 load")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if any(not line.isdigit() for line in lines):
        raise RuntimeError("Unrecognized GPU process observation; do not infer an empty device")
    pids = [int(line) for line in lines]
    allowed = []
    for pid in pids:
        if pid != IDLE_CODEX_PID:
            raise RuntimeError("GPU compute process still resident; no concurrent model permitted")
        process = psutil.Process(pid)
        argv = process.cmdline()
        if (abs(process.create_time() - IDLE_CODEX_CREATED) > .01 or process.exe().casefold() != IDLE_CODEX_EXE.casefold()
                or '--type=utility' not in argv
                or '--utility-sub-type=on_device_model.mojom.OnDeviceModelService' not in argv
                or '--service-sandbox-type=on_device_model_execution' not in argv):
            raise RuntimeError("Observed idle Codex utility identity changed")
        allowed.append({"pid": pid, "creation_time": process.create_time(), "exe": process.exe(), "argv": argv,
                        "interpretation": "idle Codex on-device utility, not a display-only process or original user workload"})
    return {"compute_pids": pids, "allowed_idle_utility": allowed, "memory_mib": {"used": used, "free": free, "total": total},
            "nvidia_smi_output": result.stdout.strip(), "nvidia_smi_memory_output": memory.stdout.strip(),
            "observed_at": time.time(), "policy": "Exact existing idle utility only; no unknown/model PID; total used<=64MiB"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", default="D:/RAGScope_Runtime")
    parser.add_argument("--campaign", default="20261014_v2")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    home = Path(args.home)
    migration = prepare_migration(home, args.campaign)
    if not args.run:
        print(json.dumps({"status": "READY_NOT_EXECUTED", "migration": artifact(migration_path(home, args.campaign)),
                          "expected_gpu_rows": 360, "model_loaded": False, "torch_imported": "torch" in sys.modules}))
        return 0
    import psutil
    from filelock import FileLock
    from ragscope.experiments.runtime import validate_gpu_lease
    lease_path = home / "runs" / args.campaign / "gpu_lease.json"
    lease = validate_gpu_lease(lease_path)
    if str(lease["gpu_device"]) != "0" or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Exactly leased GPU0 visibility required")
    if psutil.virtual_memory().available < 3 * 1024**3:
        raise MemoryError("GPU R3 preserves 3 GiB host preload reserve")
    runtime = home / "runs" / (args.campaign + "_retrieval") / "R3_GPU_RUNTIME"
    runtime.mkdir(exist_ok=True)
    with FileLock(str(home / "runs/gpu.lock"), timeout=0), Campaign(home, args.campaign + "_retrieval") as campaign:
        guards = {**require_service_exit(home, args.campaign), **require_empty_gpu(), "lease": artifact(lease_path)}
        atomic_json(runtime / "PRELOAD_GUARD.json", guards)
        path = runtime / "LOAD_STATE.json"
        if not path.exists():
            atomic_json(path, dict(migration["inherited_load_state"], status="INHERITED_CPU_COUNTERS"))
        process = psutil.Process(os.getpid())
        worker = {"pid": process.pid, "creation_time": process.create_time(), "command": process.cmdline(),
                  "cwd": process.cwd(), "device": "cuda", "status": "RUNNING", "migration_hash": migration["migration_hash"]}
        atomic_json(runtime / "worker.json", worker)
        from ragscope.experiments.legacy_reconstruction import _sample_resources
        stop = threading.Event()
        monitor = threading.Thread(target=_sample_resources, args=(runtime / "resource_samples.jsonl", stop), daemon=True)
        monitor.start()
        reranker = None
        try:
            reranker = load_reranker_with_receipt(home, runtime, device="cuda")
            atomic_json(runtime / "MODEL_LOADED.json", {"identity": reranker.identity, "gpu_memory": reranker.model.resource_snapshot()})
            documents = jsonl(home / "corpora" / args.campaign / "documents.jsonl")
            units = {u["unit_id"]: u for doc in documents for u in doc["evidence_units"]}
            del documents
            failures = 0
            for job in gpu_jobs(home, args.campaign):
                job["provenance"] = job["provenance"] + [artifact(migration_path(home, args.campaign))]
                config, _ = verified_source(job)
                chunks, parents, artifacts = find_index_material(home, args.campaign, config["index_identity"])
                failures += run_job(job, chunks, parents, units, reranker, campaign, artifacts)["n_failed"]
                del chunks, parents
            verify_migration(home, args.campaign)
        finally:
            try:
                if reranker is not None:
                    import torch
                    atomic_json(runtime / "FINAL_GPU_MEMORY.json", reranker.model.resource_snapshot())
                    del reranker
                    torch.cuda.empty_cache()
            finally:
                stop.set(); monitor.join(timeout=2)
                worker.update(status="EXITING", ended_at=time.time())
                atomic_json(runtime / "worker.json", worker)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
