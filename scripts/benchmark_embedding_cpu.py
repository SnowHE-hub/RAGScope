"""Bounded CPU thread benchmark; suspend/resume only the exact project worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np
import psutil
import requests


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main(home, campaign, worker_pid, server_pid):
    directory = home / "runs" / campaign / ("cpu_embedding_benchmark_" + time.strftime("%Y%m%d_%H%M%S"))
    directory.mkdir(parents=True, exist_ok=False)
    worker, original = psutil.Process(worker_pid), psutil.Process(server_pid)
    worker_args, command = worker.cmdline(), original.cmdline()
    assert any(a.endswith("run_retrieval_v2.py") for a in worker_args)
    assert "20261014_v2" in worker_args and "dense" in worker_args
    assert Path(worker.exe()).is_relative_to(home)
    assert command[0].lower().endswith("llama-server.exe") and Path(command[0]).is_relative_to(home)
    assert command[command.index("--port")+1] == "11436" and command[command.index("-ngl")+1] == "0"
    assert "--no-op-offload" in command and command[command.index("--pooling")+1] == "last"
    guard = {"available_bytes": psutil.virtual_memory().available, "minimum_bytes": 2*1024**3}
    save(directory / "resource_guard.json", guard)
    if guard["available_bytes"] < guard["minimum_bytes"]:
        raise RuntimeError("Insufficient RAM for a second CPU-only benchmark server")
    with socket.socket() as sock:
        assert sock.connect_ex(("127.0.0.1", 11437)) != 0, "Benchmark port already occupied"
    cases = []
    for group in ("F256_O0", "F512_O0", "F1024_O0"):
        for manifest in (home / "indexes" / campaign / "dev_chunks").glob("*/chunk_manifest.json"):
            value = json.loads(manifest.read_text(encoding="utf-8"))
            if value["identity"]["group"]["id"] != group:
                continue
            chunks = [json.loads(line) for line in (manifest.parent / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
            limit = int(group.split("_")[0][1:])
            chunk = sorted((c for c in chunks if c["n_tokens"] == limit), key=lambda c: c["chunk_id"])[0]
            cases.append({"id": group, "text": chunk["text"], "body_tokens": limit, "chunk_id": chunk["chunk_id"],
                          "source_manifest": str(manifest), "text_sha256": hashlib.sha256(chunk["text"].encode()).hexdigest()})
    assert len(cases) == 3
    question = json.loads(next((home / "corpora" / campaign / "queries_dev.jsonl").open(encoding="utf-8")))["question"]
    query_text = "Instruct: Given a question about a scientific paper, retrieve relevant passages that answer the query\nQuery:" + question
    save(directory / "cases.json", cases)
    result = {"cpu_only": True, "original_server_pid": server_pid, "worker_pid": worker_pid,
              "original_command": command, "model_identity": json.loads((home / "runs" / campaign / "embedding_identity.json").read_text()),
              "material_speedup_threshold": 1.15, "cosine_minimum": .999999, "max_abs_tolerance": 1e-5,
              "repetitions": 2, "cases": [c["id"] for c in cases], "configurations": [], "errors": []}
    active = None
    suspended = False

    def request(url, text):
        started = time.perf_counter()
        response = requests.post(url + "/v1/embeddings", json={"input": text}, timeout=120)
        response.raise_for_status()
        payload = response.json()
        vector = np.asarray(payload["data"][0]["embedding"], dtype=np.float32)
        vector /= np.linalg.norm(vector)
        return vector, {"seconds": time.perf_counter()-started, "backend_usage": payload.get("usage")}

    def measure(url, threads):
        request(url, "CPU embedding benchmark warmup")
        records, vectors = [], {}
        for repetition in range(2):
            for case in cases:
                vector, measurement = request(url, case["text"])
                vectors[case["id"]] = vector
                records.append({"case": case["id"], "repetition": repetition, "body_tokens": case["body_tokens"], **measurement})
        query, query_timing = request(url, query_text)
        ranking = sorted(vectors, key=lambda key: (-float(vectors[key] @ query), key))
        body_seconds = sum(r["seconds"] for r in records)
        row = {"threads": threads, "batch_threads": threads, "records": records, "query_timing": query_timing,
               "body_seconds": body_seconds, "body_tokens_per_second": sum(r["body_tokens"] for r in records)/body_seconds,
               "ranking": ranking, "query_passage_scores": {key: float(value @ query) for key, value in vectors.items()}}
        return row, {**vectors, "query": query}

    try:
        worker.suspend()
        suspended = True
        result["worker_suspended_at"] = time.time()
        save(directory / "STATE.json", result)
        print("Exact project worker suspended; original CPU server preserved", flush=True)
        baseline, reference = measure("http://127.0.0.1:11436", 4)
        baseline.update(speedup=1.0, numerical_equivalence=True, ranking_matches=True)
        result["configurations"].append(baseline)
        np.savez(directory / "baseline_vectors.npz", **reference)
        save(directory / "STATE.json", result)
        for threads in (8, 12):
            assert psutil.virtual_memory().available >= 2*1024**3, "RAM changed below benchmark guard"
            args = list(command)
            for flag, value in (("--port", "11437"), ("-t", str(threads)), ("-tb", str(threads))):
                args[args.index(flag)+1] = value
            output = (directory / f"server_t{threads}.log").open("w", encoding="utf-8")
            info = subprocess.STARTUPINFO()
            info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            info.wShowWindow = 0
            active = subprocess.Popen(args, stdout=output, stderr=subprocess.STDOUT, startupinfo=info,
                                      creationflags=subprocess.CREATE_NO_WINDOW, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
            try:
                deadline = time.monotonic()+60
                while time.monotonic() < deadline:
                    if active.poll() is not None:
                        raise RuntimeError("Benchmark server exited during startup")
                    try:
                        if requests.get("http://127.0.0.1:11437/health", timeout=1).status_code == 200:
                            break
                    except requests.RequestException:
                        pass
                    time.sleep(.3)
                else:
                    raise TimeoutError("Benchmark server health deadline")
                row, vectors = measure("http://127.0.0.1:11437", threads)
                comparisons = {key: {"cosine": float(np.dot(value.astype(np.float64), reference[key].astype(np.float64)) /
                    (np.linalg.norm(value.astype(np.float64))*np.linalg.norm(reference[key].astype(np.float64)))),
                    "max_abs": float(np.max(np.abs(value-reference[key])))} for key, value in vectors.items()}
                row.update(command=args, pid=active.pid, speedup=baseline["body_seconds"]/row["body_seconds"],
                           comparisons=comparisons, ranking_matches=row["ranking"] == baseline["ranking"],
                           numerical_equivalence=all(c["cosine"] >= .999999 and c["max_abs"] <= 1e-5 for c in comparisons.values()))
                result["configurations"].append(row)
                np.savez(directory / f"vectors_t{threads}.npz", **vectors)
                save(directory / "STATE.json", result)
                print(json.dumps({"threads": threads, "speedup": row["speedup"], "equivalent": row["numerical_equivalence"],
                                  "ranking_matches": row["ranking_matches"]}), flush=True)
            finally:
                # Only this Popen child: inference server has no user computation
                # checkpoint; weights are read-only and benchmark rows are saved.
                active.terminate()
                active.wait(timeout=10)
                active = None
                output.close()
        eligible = [r for r in result["configurations"] if r["numerical_equivalence"] and r["ranking_matches"] and r["speedup"] >= 1.15]
        selected = max(eligible, key=lambda r: r["speedup"]) if eligible else baseline
        result.update(status="PASS", selected_threads=selected["threads"], selected_speedup=selected["speedup"])
    except Exception as exc:
        result["status"] = "FAILED"
        result["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        if active is not None and active.poll() is None:
            active.terminate()
            active.wait(timeout=10)
        if suspended:
            worker.resume()
            result["worker_resumed_at"] = time.time()
        result["original_server_still_running"] = original.is_running()
        save(directory / "RESULT.json", result)
        print(json.dumps({"result": str(directory / "RESULT.json"), "status": result.get("status"),
                          "selected_threads": result.get("selected_threads"), "worker_resumed": suspended}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path("D:/RAGScope_Runtime"))
    parser.add_argument("--campaign", default="20261014_v2")
    parser.add_argument("--worker-pid", type=int, required=True)
    parser.add_argument("--server-pid", type=int, required=True)
    args = parser.parse_args()
    main(args.home, args.campaign, args.worker_pid, args.server_pid)
