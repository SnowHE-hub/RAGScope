"""One CPU4 checkpoint re-entry after the independently observed host restart."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import psutil
import requests
from filelock import FileLock
from ragscope.experiments.ledger import now, read_records
from ragscope.experiments.runtime import configure
from ragscope.indexing.array_index import ArrayCache
from ragscope.indexing.identity import artifact, atomic_json, file_hash, signature


def main():
    configure(cpu_only=True)
    home = Path("D:/RAGScope_Runtime")
    run, main_run = home / "runs/20261014_v2_retrieval", home / "runs/20261014_v2"
    receipt_path = run / "HOST_RESTART_REENTRY.json"
    if receipt_path.exists():
        raise RuntimeError("Host restart re-entry already attempted; inspect the receipt")
    boot_path = main_run / "HOST_RESTART_OBSERVATION_20261014_1239.json"
    boot = json.loads(boot_path.read_text(encoding="utf-8"))
    old_worker = json.loads((run / "worker.json").read_text(encoding="utf-8"))
    service_path = main_run / "embedding_service.json"
    old_service = json.loads(service_path.read_text(encoding="utf-8"))
    assert boot["restart_cause"] == "UNKNOWN" and not boot["assistant_issued_restart"]
    assert psutil.boot_time() > old_worker["creation_time"]
    for previous in (old_worker, old_service):
        if psutil.pid_exists(previous["pid"]):
            assert psutil.Process(previous["pid"]).create_time() != previous["creation_time"]
    with FileLock(str(run / "writer.lock"), timeout=0):
        pass
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 11436))
    failures = read_records(run / "embedding_runtime_failures.jsonl")
    assert len(failures) == 3 and failures[-1]["retry_number"] == 3
    base = home / "indexes/20261014_v2/all_chunks/ba23a7f21f5bc3ed9fccb5778163c07a6b2fda530c769ff235e15d44dd5238f6"
    observations = base / "dense_397d90b3f4fbe03e/embedding_first_observations.jsonl"
    rows = read_records(observations)
    assert len(rows) == 5976 and observations.read_bytes().endswith(b"\n")
    identity_path = main_run / "embedding_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    cache = ArrayCache(home / "cache/ragscope/v2_embeddings", identity)
    tail_ids = {row["chunk_id"] for row in rows[-16:]}
    verified_tail = []
    for line in (base / "chunks.jsonl").open(encoding="utf-8"):
        chunk = json.loads(line)
        if chunk["chunk_id"] in tail_ids:
            vector = cache.get(chunk["text"])
            assert vector is not None and len(vector) == 1024
            verified_tail.append(chunk["chunk_id"])
    assert len(verified_tail) == 16
    command = list(old_service["command"])
    command[command.index("-t") + 1] = "4"
    command[command.index("-tb") + 1] = "4"
    assert command[command.index("-ngl") + 1] == "0" and "--no-op-offload" in command
    assert file_hash(command[0]) == old_service["binary_sha256"]
    assert file_hash(command[command.index("-m") + 1]) == old_service["model_sha256"]
    available = psutil.virtual_memory().available
    assert available >= 3 * 1024**3, f"CPU re-entry requires 3 GiB available, got {available}"
    performance = {"backend": "llama.cpp-b11429-cpu", "threads": 4, "batch_threads": 4}
    receipt = {"status": "RESERVED_BEFORE_CPU_LOAD", "started_at": now(),
        "reason": "External host restart observed; cause UNKNOWN; checkpoint re-entry, not backend retry",
        "boot_observation": artifact(boot_path), "boot_time_psutil": psutil.boot_time(),
        "historical_backend_recoveries": 3, "maximum_backend_recoveries": 3,
        "failure_journal": artifact(run / "embedding_runtime_failures.jsonl"),
        "terminal_22849": "Unknown process id after restart; not an observed exit code",
        "durable_observations": len(rows), "observations": artifact(observations),
        "verified_tail_cache_ids": verified_tail, "available_before_bytes": available,
        "old_worker": old_worker, "old_service": old_service, "command": command,
        "performance_identity": performance, "performance_config_sha256": signature(performance),
        "semantic_identity": artifact(identity_path),
        "cache_compatibility": "Same pinned weights/tokenizer/pooling/normalization/input; prior bounded CPU4/8/12 benchmark vectors identical. Thread scheduling is separate performance provenance; no universal numerical equivalence claim.",
        "source_artifacts": [artifact(Path(x)) for x in (__file__, "scripts/run_retrieval_v2.py",
            "ragscope/experiments/retrieval_campaign.py", "ragscope/indexing/local_embedding.py")],
        "worker_started": False}
    with receipt_path.open("x", encoding="utf-8") as out:
        json.dump(receipt, out, ensure_ascii=False, indent=2)
        out.flush()
        os.fsync(out.fileno())
    atomic_json(run / "worker_before_host_restart_reentry.json", old_worker)
    atomic_json(run / "embedding_service_before_host_restart_reentry.json", old_service)
    log = (run / "embedding_cpu4_after_host_restart.log").open("w", encoding="utf-8")
    server = None
    process = None
    created = None
    try:
        server = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        process = psutil.Process(server.pid)
        created = process.create_time()
        receipt.update(server_pid=server.pid, server_creation_time=created)
        atomic_json(receipt_path, receipt)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError("Own CPU service exited during loading")
            if psutil.virtual_memory().available < 1024**3:
                raise RuntimeError("Host RAM below 1 GiB during CPU loading")
            try:
                if requests.get("http://127.0.0.1:11436/health", timeout=1).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(.25)
        else:
            raise TimeoutError("CPU health did not become ready within 60 seconds")
        result = requests.post("http://127.0.0.1:11436/v1/embeddings",
            json={"input": "Host restart CPU four thread checkpoint verification"}, timeout=30)
        result.raise_for_status()
        payload = result.json()
        assert len(payload["data"][0]["embedding"]) == 1024
        assert psutil.virtual_memory().available >= 1024**3
        service = dict(old_service, status="RUNNING", pid=server.pid, creation_time=created,
            parent_pid=os.getpid(), parent_alive=False, supervisor_pid=None, command=command,
            performance_identity=performance, performance_config_sha256=signature(performance),
            time=now(), observed_at=now(), resume_receipt=str(receipt_path),
            supervisor_note="One-time CPU4 restart controller exits after verified handover",
            available_ram_at_observation=psutil.virtual_memory().available)
        for key in ("ended_at", "stop_reason", "last_resume_error", "worker_resumed"):
            service.pop(key, None)
        atomic_json(service_path, service)
        receipt.update(status="CPU4_READY_FOR_CHECKPOINT_WORKER", verification_dimensions=1024,
            verification_usage=payload.get("usage"), available_after_bytes=psutil.virtual_memory().available,
            completed_at=now())
    except Exception as exc:
        receipt.update(status="FAILED_BEFORE_WORKER_START", error={"type": type(exc).__name__, "message": str(exc)})
        if (server is not None and server.poll() is None and process is not None
                and process.create_time() == created and process.cmdline() == command):
            process.terminate()
            process.wait(timeout=15)
        receipt["owned_service_exited"] = server is None or server.poll() is not None
        raise
    finally:
        atomic_json(receipt_path, receipt)
        log.close()
        print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
