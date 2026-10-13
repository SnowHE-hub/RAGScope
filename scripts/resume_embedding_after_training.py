"""Resume this campaign's precisely identified CPU service and suspended worker."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from datetime import datetime, timezone

import psutil
import requests


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    home = Path("D:/RAGScope_Runtime/runs")
    main_run = home / "20261014_v2"
    retrieval_run = home / "20261014_v2_retrieval"
    receipt_path = retrieval_run / "TRAINING_RAM_RESUME.json"
    if receipt_path.exists():
        raise RuntimeError("Resume receipt exists; inspect it rather than launching twice")
    release = json.loads((home / "20261014_v2_lora/train_2b_1024_seed42/GPU_RELEASE_RECEIPT.json").read_text(encoding="utf-8"))
    assert release["status"] == "GPU_RELEASED_AFTER_TRAINING"
    assert release["gpu_lock_reacquired_and_released"] and not release["training_processes"]
    pause = json.loads((retrieval_run / "PLANNED_TRAINING_RAM_PAUSE.json").read_text(encoding="utf-8"))
    service_path = main_run / "embedding_service.json"
    service = json.loads(service_path.read_text(encoding="utf-8"))
    assert service["status"] == "EXITED_PLANNED_RAM_RELEASE"
    worker = psutil.Process(pause["worker_pid"])
    assert worker.create_time() == pause["worker_create_time"]
    assert worker.cmdline() == pause["worker_command"] and worker.status() == psutil.STATUS_STOPPED
    command = pause["server_command"]
    assert command[command.index("-ngl") + 1] == "0" and "--no-op-offload" in command
    assert command[command.index("-t") + 1] == "12" and command[command.index("-tb") + 1] == "12"
    assert command[command.index("--host") + 1] == "127.0.0.1" and command[command.index("--port") + 1] == "11436"
    assert sha256(command[0]) == service["binary_sha256"]
    assert sha256(command[command.index("-m") + 1]) == service["model_sha256"]
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 11436)) == 0:
            raise RuntimeError("Expected released service port is occupied")
    available = psutil.virtual_memory().available
    if available < 3 * 1024**3:
        raise RuntimeError(f"CPU service restart needs 3 GiB available; observed {available}")
    archive = retrieval_run / "embedding_service_before_training_resume.json"
    if archive.exists():
        raise RuntimeError("Resume archive already exists; inspect partial attempt")
    archive.write_bytes(service_path.read_bytes())
    receipt = {"status": "STARTING", "started_at": datetime.now(timezone.utc).isoformat(),
               "available_ram_before": available, "worker_pid": worker.pid,
               "worker_creation_time": worker.create_time(), "command": command,
               "binary_sha256": service["binary_sha256"], "model_sha256": service["model_sha256"],
               "training_release_receipt": str(home / "20261014_v2_lora/train_2b_1024_seed42/GPU_RELEASE_RECEIPT.json")}
    atomic(receipt_path, receipt)
    output = (retrieval_run / "embedding_cpu12_after_training.log").open("w", encoding="utf-8")
    server = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                              creationflags=subprocess.CREATE_NO_WINDOW,
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"CPU service exited {server.returncode}")
            try:
                if requests.get("http://127.0.0.1:11436/health", timeout=1).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(.25)
        else:
            raise TimeoutError("CPU service did not become healthy")
        check = requests.post("http://127.0.0.1:11436/v1/embeddings",
                              json={"input": "CPU training handover verification"}, timeout=30)
        check.raise_for_status()
        assert len(check.json()["data"][0]["embedding"]) == 1024
        process = psutil.Process(server.pid)
        now = datetime.now(timezone.utc).isoformat()
        service.update(status="RUNNING", pid=server.pid, creation_time=process.create_time(),
                       parent_pid=os.getpid(), parent_alive=False, supervisor_pid=None,
                       supervisor_note="Controlled resume launcher exits after identity-verified handover.",
                       command=command, time=now, observed_at=now,
                       available_ram_at_observation=psutil.virtual_memory().available,
                       replaced_receipt_archive=str(archive), resume_receipt=str(receipt_path))
        for key in ("ended_at", "stop_reason"):
            service.pop(key, None)
        atomic(service_path, service)
        worker.resume()
        worker_path = retrieval_run / "worker.json"
        worker_receipt = json.loads(worker_path.read_text(encoding="utf-8"))
        worker_receipt.update(status="RUNNING", observed_at=now, embedding_service_pid=server.pid,
                              resume_receipt=str(receipt_path))
        atomic(worker_path, worker_receipt)
        receipt.update(status="PASS", server_pid=server.pid, server_creation_time=process.create_time(),
                       dimensions=1024, verification_usage=check.json().get("usage"), worker_resumed=True,
                       available_ram_after=psutil.virtual_memory().available, ended_at=now)
    except Exception as exc:
        receipt.update(status="FAILED", server_pid=server.pid, worker_resumed=False,
                       error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        atomic(receipt_path, receipt)
        print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
