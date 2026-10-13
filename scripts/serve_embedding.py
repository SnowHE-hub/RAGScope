"""Project-owned CPU embedding service. A failed RAM guard never loads a model."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import os
import signal
import socket
import subprocess
import time
import psutil
from ragscope.experiments.runtime import configure, runtime_home, require_ram
from ragscope.indexing.identity import atomic_json, file_hash
from ragscope.experiments.ledger import append_jsonl, now

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=11436)
    args = parser.parse_args()
    configure(cpu_only=True)
    root = runtime_home()
    record = {"time": now(), "device": "cpu", "cuda_mask": os.environ["CUDA_VISIBLE_DEVICES"],
              "available_ram": psutil.virtual_memory().available, "requested_ram": 1024**3, "reserve_ram": 512 * 1024**2}
    try:
        require_ram(record["requested_ram"], record["reserve_ram"])
    except RuntimeError as exc:
        record.update(status="BLOCKED_RESOURCE", reason=str(exc), model_loaded=False)
        append_jsonl(root / "runs/20261014_v2/embedding_preflight.jsonl", record)
        print(json.dumps(record))
        return 2
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    binary = root / "tools/llama-b11429-cpu/llama-server.exe"
    model = root / "models/qwen3-embedding-0.6b/Qwen3-Embedding-0.6B-Q8_0.gguf"
    command = [str(binary), "-m", str(model), "--embedding", "--pooling", "last", "--embd-normalize", "2",
               "--host", "127.0.0.1", "--port", str(args.port), "-ngl", "0", "--no-op-offload", "-t", "4", "-tb", "4",
               "-c", "2048", "-b", "1100", "-ub", "1100", "-np", "1", "--threads-http", "2"]
    record.update(status="STARTING", command=command, binary_sha256=file_hash(binary), model_sha256=file_hash(model))
    append_jsonl(root / "runs/20261014_v2/embedding_preflight.jsonl", record)
    with (root / "logs/embedding_server.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        record.update(pid=process.pid, creation_time=psutil.Process(process.pid).create_time(), parent_pid=os.getpid(), status="RUNNING")
        path = root / "runs/20261014_v2/embedding_service.json"
        atomic_json(path, record)
        def cleanup(*_):
            if process.poll() is None:
                process.terminate()  # own stateless inference child only, never another workload
        signal.signal(signal.SIGTERM, cleanup)
        signal.signal(signal.SIGINT, cleanup)
        try:
            return process.wait()
        finally:
            cleanup()
            record.update(status="EXITED", exit_code=process.poll(), ended_at=now())
            atomic_json(path, record)

if __name__ == "__main__":
    raise SystemExit(main())
