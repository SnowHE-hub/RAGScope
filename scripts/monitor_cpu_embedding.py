"""Own-process host RAM guard around the existing WSL CPU alternative."""
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import psutil
from ragscope.experiments.runtime import configure, runtime_home
from ragscope.indexing.identity import atomic_json
from ragscope.experiments.ledger import append_jsonl, now

configure(cpu_only=True)
root = runtime_home()
script = Path(__file__).with_name("serve_embedding_wsl.py").resolve()
linux_script = "/mnt/" + script.drive[0].lower() + script.as_posix()[2:]
record = {"time": now(), "host_available_before": psutil.virtual_memory().available,
          "alternative": "same_Q8_CPU_backend_in_existing_WSL", "minimum_host_available": 512*1024**2}
if record["host_available_before"] < 1024**3:
    record.update(status="BLOCKED_RESOURCE", reason="host_available_before_below_1GiB")
    append_jsonl(root / "runs/20261014_v2/embedding_preflight.jsonl", record)
    raise SystemExit(2)
stop = root / "runs/20261014_v2/embedding_wsl.stop"
command = ["wsl", "-d", "Ubuntu", "--exec", "python3", "-u", linux_script]
record.update(command=command, status="STARTING")
append_jsonl(root / "runs/20261014_v2/embedding_preflight.jsonl", record)
with (root / "logs/embedding_wsl_monitor.log").open("a") as log:
    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    low_count, minimum = 0, record["host_available_before"]
    try:
        while child.poll() is None:
            available = psutil.virtual_memory().available
            minimum = min(minimum, available)
            low_count = low_count + 1 if available < 512*1024**2 else 0
            if low_count >= 3:
                stop.write_text("host RAM guard requested graceful exit of project CPU child only", encoding="utf-8")
                record.update(guard_triggered=True, status="BLOCKED_RESOURCE")
                child.wait(timeout=45)
                break
            time.sleep(1)
    finally:
        if child.poll() is None:
            stop.write_text("monitor cleanup", encoding="utf-8")
            child.wait(timeout=45)
        record.update(exit_code=child.poll(), minimum_host_available=minimum, ended_at=now())
        atomic_json(root / "runs/20261014_v2/embedding_wsl_resource.json", record)
