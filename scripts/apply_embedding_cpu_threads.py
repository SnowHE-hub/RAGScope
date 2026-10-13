"""Apply an audited CPU-only thread count to this project's exact server."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import psutil
import requests


def main(benchmark):
    outcome = json.loads(benchmark.read_text(encoding="utf-8"))
    assert outcome["status"] == "PASS" and outcome["selected_speedup"] >= 1.15
    threads = outcome["selected_threads"]
    assert threads in (8, 12)
    chosen = next(c for c in outcome["configurations"] if c["threads"] == threads)
    assert chosen["numerical_equivalence"] and chosen["ranking_matches"]
    destination = benchmark.parent / "THREAD_TRANSITION.json"
    if destination.exists():
        raise RuntimeError("Thread transition already recorded; inspect exact running process instead of repeating")
    worker = psutil.Process(outcome["worker_pid"])
    server = psutil.Process(outcome["original_server_pid"])
    original = outcome["original_command"]
    assert server.cmdline() == original
    assert any(arg.endswith("run_retrieval_v2.py") for arg in worker.cmdline())
    assert original[original.index("--port")+1] == "11436"
    assert original[original.index("-ngl")+1] == "0" and "--no-op-offload" in original
    updated = list(original)
    for flag in ("-t", "-tb"):
        updated[updated.index(flag)+1] = str(threads)
    snapshot = {"benchmark_path": str(benchmark), "benchmark_sha256": hashlib.sha256(benchmark.read_bytes()).hexdigest(),
                "old_server_pid": server.pid, "old_server_create_time": server.create_time(), "worker_pid": worker.pid,
                "old_threads": 4, "new_threads": threads, "old_command": original, "new_command": updated,
                "semantic_model_identity": outcome["model_identity"], "semantic_identity_unchanged": True,
                "compatibility_basis": "All three length cases plus query were numerically equivalent; all sampled max_abs differences were zero and rankings identical. This is a bounded empirical check, not a universal numerical proof.",
                "in_flight_old_thread_cache_outputs_upper_bound": 1, "cpu_only": True,
                "performance_identity": {"threads": threads, "batch_threads": threads, "backend": "llama.cpp-b11429-cpu"}}
    snapshot["performance_config_sha256"] = hashlib.sha256(json.dumps(snapshot["performance_identity"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    worker.suspend()
    replacement = None
    try:
        # With one inference slot, a completed barrier request lets the previous
        # in-flight model request finish before the read-only server is replaced.
        barrier = requests.post("http://127.0.0.1:11436/v1/embeddings", json={"input": "CPU thread transition barrier"}, timeout=30)
        barrier.raise_for_status()
        cache = Path("D:/RAGScope_Runtime/cache/ragscope/v2_embeddings")
        previous_keys = [str(p.relative_to(cache)) for p in cache.rglob("*.json") if len(p.stem) == 64]
        (benchmark.parent / "CPU4_CACHE_KEYS_AT_TRANSITION.json").write_text(json.dumps(sorted(previous_keys), indent=2), encoding="utf-8")
        snapshot["preexisting_cpu4_cache_manifest_count"] = len(previous_keys)
        snapshot["transition_started_unix"] = time.time()
        server.terminate()  # Exact project-owned stateless inference process only.
        server.wait(timeout=15)
        output = (benchmark.parent / f"active_server_t{threads}_11436.log").open("w", encoding="utf-8")
        info = subprocess.STARTUPINFO()
        info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        info.wShowWindow = 0
        replacement = subprocess.Popen(updated, stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            startupinfo=info, creationflags=subprocess.CREATE_NO_WINDOW, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        deadline = time.monotonic()+60
        while time.monotonic() < deadline:
            if replacement.poll() is not None:
                raise RuntimeError("Replacement CPU server exited")
            try:
                if requests.get("http://127.0.0.1:11436/health", timeout=1).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(.25)
        else:
            raise TimeoutError("Replacement CPU server did not become healthy")
        check = requests.post("http://127.0.0.1:11436/v1/embeddings", json={"input": "CPU thread transition verification"}, timeout=30)
        check.raise_for_status()
        assert len(check.json()["data"][0]["embedding"]) == outcome["model_identity"]["dimensions"]
        snapshot.update(status="PASS", new_server_pid=replacement.pid, new_server_create_time=psutil.Process(replacement.pid).create_time(),
                        verification_usage=check.json().get("usage"), transition_completed_unix=time.time())
    except Exception as exc:
        snapshot.update(status="FAILED", error={"type": type(exc).__name__, "message": str(exc)},
                        new_server_pid=replacement.pid if replacement else None)
        raise
    finally:
        worker.resume()
        snapshot["worker_resumed_unix"] = time.time()
        destination.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"transition": str(destination), "status": snapshot.get("status"),
                          "new_server_pid": snapshot.get("new_server_pid"), "threads": threads, "worker_resumed": True}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    args = parser.parse_args()
    main(args.benchmark)
