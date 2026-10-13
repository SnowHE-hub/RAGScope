"""Read-only environment audit, with only project relevant fields."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import psutil
from ragscope.experiments.runtime import configure, runtime_home
from ragscope.indexing.identity import atomic_json, artifact

def command(argv):
    result = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    return {"argv": argv, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}

def main():
    paths = configure(cpu_only=True)
    report = {"platform": platform.platform(), "python": sys.version, "executable": sys.executable,
              "logical_cpus": os.cpu_count(), "memory": dict(psutil.virtual_memory()._asdict()), "runtime": paths,
              "cuda_mask": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu_requests_made": False,
              "gpu_workload_modified": False, "disks": {}, "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}
    for drive in ("C:/", "D:/"):
        if Path(drive).exists():
            report["disks"][drive] = dict(shutil.disk_usage(drive)._asdict())
    report["nvidia_smi"] = command(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,driver_version", "--format=csv"])
    report["gpu_processes"] = command(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"])
    project = Path(__file__).resolve().parents[1]
    report["git_head"] = command(["git", "rev-parse", "HEAD"])
    report["git_branch"] = command(["git", "branch", "--show-current"])
    report["artifacts"] = [artifact(project / f) for f in ["requirements-v2.lock", "docs/execution/MASTER_PLAN.md", "reports/retrieval_results.csv"]]
    path = runtime_home() / "runs/20261014_v2/environment.json"
    atomic_json(path, report)
    print(json.dumps({"report": str(path), "memory_available": report["memory"]["available"], "cpu_only": True}))

if __name__ == "__main__":
    main()
