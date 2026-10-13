"""CPU-only WSL alternative inside existing WSL. No services or user jobs altered."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

root = Path("/mnt/d/RAGScope_Runtime")
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                  TMPDIR=str(root / "tmp"), XDG_CACHE_HOME=str(root / "cache"),
                  HF_HOME=str(root / "cache/hf"), TORCH_HOME=str(root / "cache/torch"), PYTHONDONTWRITEBYTECODE="1")
memory = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines() if len(line.split()) >= 2}
if memory["MemAvailable"] < 2 * 1024**3:
    raise RuntimeError("Existing WSL available RAM below 2GiB CPU preflight guard")
binary = root / "tools/llama-b11429-linux-cpu/llama-b11429/llama-server"
if not binary.exists():
    matches = list((root / "tools/llama-b11429-linux-cpu").rglob("llama-server"))
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one private CPU llama-server")
    binary = matches[0]
env = dict(os.environ, LD_LIBRARY_PATH=str(binary.parent))
stop = root / "runs/20261014_v2/embedding_wsl.stop"
if stop.exists():
    raise RuntimeError("Previous stop marker exists; inspect previous run before retry")
command = [str(binary), "-m", str(root / "models/qwen3-embedding-0.6b/Qwen3-Embedding-0.6B-Q8_0.gguf"),
           "--embedding", "--pooling", "last", "--embd-normalize", "2", "--host", "127.0.0.1", "--port", "11436",
           "-ngl", "0", "--no-op-offload", "-t", "4", "-tb", "4", "-c", "2048", "-b", "1100", "-ub", "1100", "-np", "1", "--threads-http", "2"]
with (root / "logs/embedding_wsl_server.log").open("a") as log:
    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    record = {"pid": child.pid, "parent_pid": os.getpid(), "start_ticks": Path(f"/proc/{child.pid}/stat").read_text().split()[21],
              "command": command, "device": "cpu", "cuda_mask": "", "guest_available_before": memory["MemAvailable"], "status": "RUNNING"}
    path = root / "runs/20261014_v2/embedding_wsl_service.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    def cleanup(*_):
        if child.poll() is None:
            child.send_signal(signal.SIGINT)
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)
    try:
        while child.poll() is None:
            if stop.exists():
                cleanup()
                break
            time.sleep(1)
        child.wait(timeout=30)
    finally:
        cleanup()
        record.update(status="EXITED", exit_code=child.poll())
        path.write_text(json.dumps(record), encoding="utf-8")
