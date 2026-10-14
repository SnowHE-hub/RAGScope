"""One supervised CPU4 service lifetime for independent auxiliary indexes."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import psutil
import requests
from filelock import FileLock, Timeout
from ragscope.experiments.aux_preencode import run_indexes
from ragscope.experiments.ledger import now, read_records
from ragscope.experiments.runtime import configure
from ragscope.indexing.identity import artifact, atomic_json, file_hash, signature


def main():
    configure(cpu_only=True)
    home, campaign_id = Path("D:/RAGScope_Runtime"), "20261014_v2"
    main_run, retrieval = home / "runs" / campaign_id, home / "runs" / (campaign_id + "_retrieval")
    auxiliary = home / "runs" / (campaign_id + "_aux_preencode")
    auxiliary.mkdir(parents=True, exist_ok=True)
    receipt_path = auxiliary / "SERVICE_LIFETIME.json"
    if receipt_path.exists():
        raise RuntimeError("Auxiliary service already attempted; no automatic backend recovery")
    deadline_path = main_run / "EXECUTION_DEADLINE.json"
    deadline = datetime.fromisoformat(json.loads(deadline_path.read_text(encoding="utf-8"))["deadline_utc"])
    if datetime.now(timezone.utc) >= deadline:
        raise TimeoutError("Execution deadline reached")
    service_path = main_run / "embedding_service.json"
    previous = json.loads(service_path.read_text(encoding="utf-8"))
    if psutil.pid_exists(previous["pid"]) and psutil.Process(previous["pid"]).create_time() == previous["creation_time"]:
        raise RuntimeError("Previous service identity still exists")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 11436))
    journal = retrieval / "embedding_runtime_failures.jsonl"
    if len(read_records(journal)) != 3:
        raise ValueError("Historical recovery journal changed; inspect before any new load")
    # A lock availability probe never reads or writes the main campaign state.
    main_lock_held = False
    try:
        with FileLock(str(retrieval / "writer.lock"), timeout=0):
            pass
    except Timeout:
        main_lock_held = True
    command = list(previous["command"])
    assert command[command.index("-t") + 1] == command[command.index("-tb") + 1] == "4"
    assert command[command.index("-ngl") + 1] == "0" and "--no-op-offload" in command
    assert file_hash(command[0]) == previous["binary_sha256"]
    assert file_hash(command[command.index("-m") + 1]) == previous["model_sha256"]
    available = psutil.virtual_memory().available
    if available < 3 * 1024**3:
        atomic_json(auxiliary / "START_RESOURCE_BLOCK.json", {"time": now(), "available_bytes": available,
                    "required_bytes": 3 * 1024**3, "service_started": False})
        raise MemoryError("Auxiliary CPU load requires 3 GiB available; no threshold fallback")
    performance = {"backend": "llama.cpp-b11429-cpu", "threads": 4, "batch_threads": 4}
    worker = psutil.Process()
    receipt = {"status": "RESERVED_BEFORE_CPU_LOAD", "started_at": now(),
        "worker_pid": os.getpid(), "worker_creation_time": worker.create_time(), "worker_command": worker.cmdline(),
        "main_retrieval_lock_observed_held": main_lock_held, "main_retrieval_campaign_opened": False,
        "command": command, "previous_service": previous, "available_before_bytes": available,
        "historical_backend_recoveries": 3, "maximum_backend_recoveries": 3,
        "automatic_retry": False, "failure_journal_before": artifact(journal),
        "deadline": artifact(deadline_path), "performance_identity": performance,
        "performance_config_sha256": signature(performance),
        "semantic_identity": artifact(main_run / "embedding_identity.json"),
        "source_artifacts": [artifact(Path(p)) for p in (__file__, "ragscope/experiments/aux_preencode.py",
            "ragscope/experiments/retrieval_campaign.py", "ragscope/indexing/local_embedding.py")],
        "runtime_guard": "Per-request 1 GiB guard before HTTP calls; durable completed cache and metrics retained; no backend retry"}
    with receipt_path.open("x", encoding="utf-8") as out:
        json.dump(receipt, out, ensure_ascii=False, indent=2)
        out.flush()
        os.fsync(out.fileno())
    atomic_json(auxiliary / "embedding_service_before_aux.json", previous)
    server, process, created, service = None, None, None, None
    with (auxiliary / "embedding_cpu4.log").open("w", encoding="utf-8") as log:
        try:
            server = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
            process = psutil.Process(server.pid)
            created = process.create_time()
            receipt.update(server_pid=server.pid, server_creation_time=created)
            atomic_json(receipt_path, receipt)
            health_deadline = time.monotonic() + 60
            while time.monotonic() < health_deadline:
                if server.poll() is not None:
                    raise RuntimeError("Owned CPU service exited during loading")
                if psutil.virtual_memory().available < 1024**3:
                    raise MemoryError("Available RAM below 1 GiB during service loading")
                try:
                    if requests.get("http://127.0.0.1:11436/health", timeout=1).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(.25)
            else:
                raise TimeoutError("CPU service health timeout after 60 seconds")
            result = requests.post("http://127.0.0.1:11436/v1/embeddings",
                json={"input": "Auxiliary frozen index CPU four thread verification"}, timeout=30)
            result.raise_for_status()
            payload = result.json()
            assert len(payload["data"][0]["embedding"]) == 1024
            assert psutil.virtual_memory().available >= 1024**3
            service = dict(previous, status="RUNNING_AUX_PREENCODE", pid=server.pid, creation_time=created,
                parent_pid=os.getpid(), parent_alive=True, supervisor_pid=os.getpid(), command=command,
                performance_identity=performance, performance_config_sha256=signature(performance),
                time=now(), observed_at=now(), available_ram_at_observation=psutil.virtual_memory().available,
                supervisor_note="Independent aux controller owns lifetime and exact identity cleanup",
                aux_receipt=str(receipt_path))
            for key in ("ended_at", "exit_code", "stop_reason", "release_receipt", "resume_receipt", "pause_receipt"):
                service.pop(key, None)
            atomic_json(service_path, service)
            receipt.update(status="RUNNING_INDEX_ONLY", verification_dimensions=1024,
                           verification_usage=payload.get("usage"), available_after_bytes=psutil.virtual_memory().available)
            atomic_json(receipt_path, receipt)
            print(json.dumps({"status": receipt["status"], "worker_pid": os.getpid(), "server_pid": server.pid,
                              "main_lock_held": main_lock_held, "available": psutil.virtual_memory().available}), flush=True)
            receipt["indexes"] = run_indexes(home, campaign_id, deadline)
            receipt["status"] = "PASS_INDEXES_READY"
        except BaseException as exc:
            receipt.update(status="BLOCKED" if isinstance(exc, (MemoryError, TimeoutError)) else "FAILED",
                           error={"type": type(exc).__name__, "message": str(exc)})
            raise
        finally:
            # run_indexes has returned/raised: no index request remains in this controller.
            if (server is not None and server.poll() is None and process is not None
                    and process.create_time() == created and process.cmdline() == command):
                process.terminate()
                process.wait(timeout=15)
            receipt.update(ended_at=now(), service_exited=server is None or server.poll() is not None,
                           service_exit_code=None if server is None else server.poll(),
                           available_at_exit_bytes=psutil.virtual_memory().available,
                           failure_journal_unchanged=file_hash(journal) == receipt["failure_journal_before"]["sha256"])
            if service is not None:
                service.update(status="EXITED_AUX_PREENCODE" if receipt["service_exited"] else "CLEANUP_INCOMPLETE",
                    ended_at=now(), exit_code=receipt["service_exit_code"], parent_alive=False,
                    stop_reason="Auxiliary index-only controller reached success or retained failure boundary")
                atomic_json(service_path, service)
            atomic_json(receipt_path, receipt)
            print(json.dumps({"status": receipt["status"], "service_exited": receipt["service_exited"],
                              "receipt": str(receipt_path)}), flush=True)


if __name__ == "__main__":
    main()
