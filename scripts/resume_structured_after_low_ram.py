"""One controlled CPU-service restart after the recorded structured RAM pause.

This resumes only the exact suspended worker. A later socket reset remains an
observable worker failure; the caller must inspect it before a bounded replay.
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
from datetime import datetime, timezone

import psutil
import requests

from resume_embedding_after_training import atomic, sha256


class SupervisedLoadMonitor:
    """May stop only the exact new service, and must be joined before handoff."""
    def __init__(self, process, created, command, path, *, read_available=None, interval=.25):
        self.process, self.created, self.command, self.path = process, created, command, Path(path)
        self.read_available = read_available or (lambda: psutil.virtual_memory().available)
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.samples, self.failure = [], None

    def start(self):
        self.thread.start()

    def _run(self):
        try:
            while not self.stop_event.is_set():
                available = self.read_available()
                memory = self.process.memory_info()
                row = {"at_utc": datetime.now(timezone.utc).isoformat(), "available_bytes": available,
                    "service_rss_bytes": memory.rss, "service_peak_wset_bytes": getattr(memory, "peak_wset", None)}
                self.samples.append(row)
                with self.path.open("a", encoding="utf-8") as out:
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                if available < 1024**3:
                    self.failure = {"kind": "host_ram_below_1gib_during_load", **row}
                    if self.process.create_time() == self.created and self.process.cmdline() == self.command:
                        self.process.terminate()
                        self.process.wait(timeout=15)
                    else:
                        self.failure["cleanup_identity_mismatch"] = True
                    return
                self.stop_event.wait(self.interval)
        except psutil.NoSuchProcess:
            self.failure = {"kind": "owned_service_exited_during_monitoring"}
        except Exception as exc:
            self.failure = {"kind": "monitor_error", "type": type(exc).__name__, "message": str(exc)}

    def close(self):
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=16)
        if self.thread.is_alive():
            raise RuntimeError("Load monitor did not join; worker must remain paused")
        return {"joined_before_worker_resume": True, "failure": self.failure,
            "samples": len(self.samples),
            "minimum_available_bytes": min((r["available_bytes"] for r in self.samples), default=None),
            "max_sampled_service_rss_bytes": max((r["service_rss_bytes"] for r in self.samples), default=None),
            "max_observed_process_peak_wset_bytes": max((r["service_peak_wset_bytes"] for r in self.samples
                if r["service_peak_wset_bytes"] is not None), default=None)}


def choose_guard_path(available, supervised_fallback, observations):
    if available >= 3 * 1024**3:
        return "original_3gib"
    if not supervised_fallback or available < 2.5 * 1024**3:
        raise RuntimeError(f"Original3GiB/authorized once-only2.5GiB load guard unavailable: {available}")
    recent = sorted((r for r in observations if r.get("action") == "guard_observation_only_no_model_launch"),
        key=lambda r: r["at_utc"])[-2:]
    now = datetime.now(timezone.utc)
    if (len(recent) != 2 or any(not 2.5 * 1024**3 <= r["available_bytes"] < 3 * 1024**3 for r in recent)
            or (datetime.fromisoformat(recent[-1]["at_utc"]) - datetime.fromisoformat(recent[0]["at_utc"])).total_seconds() < 30
            or (now - datetime.fromisoformat(recent[-1]["at_utc"])).total_seconds() > 90):
        raise RuntimeError("Supervised fallback requires two recent stable2.5–3GiB observations at least30s apart")
    return "once_only_supervised_2_5_to_3gib"


def handle_resume_failure(receipt, service_receipt, service_path, server, process,
                          created, worker_resumed, exc):
    """Clean up only before handoff; after resume the worker may own requests."""
    receipt.update(status="FAILED_AFTER_WORKER_RESUMED" if worker_resumed else "FAILED_BEFORE_WORKER_RESUME",
        worker_resumed=worker_resumed, error={"type": type(exc).__name__, "message": str(exc)})
    if not worker_resumed and server.poll() is None:
        try:
            exact = process.create_time() == created and process.cmdline() == service_receipt["command"]
            receipt["cleanup_identity_verified"] = exact
            if exact:
                process.terminate()
                process.wait(timeout=15)
        except psutil.NoSuchProcess:
            receipt["cleanup_service_already_exited"] = True
    exited = server.poll() is not None
    receipt.update(failed_owned_service_exited=exited,
        owned_service_retained_after_worker_resume=worker_resumed and not exited)
    service_receipt.update(pid=server.pid, creation_time=created,
        status=("EXITED_FAILED_LOW_RAM_RESUME" if exited else
                "RUNNING" if worker_resumed else "FAILED_RESUME_OWNED_SERVICE_STILL_RUNNING"),
        last_resume_error=receipt["error"], worker_resumed=worker_resumed)
    if exited:
        service_receipt["ended_at"] = datetime.now(timezone.utc).isoformat()
    atomic(service_path, service_receipt)


def main(supervised_fallback=False):
    home = Path("D:/RAGScope_Runtime")
    run = home / "runs/20261014_v2_retrieval"
    main_run = home / "runs/20261014_v2"
    path = run / "STRUCTURED_LOW_RAM_RESUME.json"
    if path.exists():
        raise RuntimeError("Resume receipt exists; inspect it rather than repeat a model launch")
    pause = json.loads((run / "STRUCTURED_LOW_RAM_PAUSE.json").read_text(encoding="utf-8"))
    service_path = main_run / "embedding_service.json"
    service_receipt = json.loads(service_path.read_text(encoding="utf-8"))
    if pause["status"] != "SUSPENDED_SERVICE_STOPPED" or service_receipt["status"] != "EXITED_LOW_RAM_RELEASE":
        raise RuntimeError("Expected completed project-owned RAM release")
    original_worker = pause["worker"]
    worker = psutil.Process(original_worker["pid"])
    if (worker.create_time() != original_worker["creation_time"] or worker.cmdline() != original_worker["command"]
            or worker.status() != psutil.STATUS_STOPPED):
        raise RuntimeError("Exact suspended worker identity differs")
    command = pause["service"]["command"]
    for flag, value in (("-ngl", "0"), ("-t", "12"), ("-tb", "12"), ("--host", "127.0.0.1"), ("--port", "11436")):
        if command[command.index(flag) + 1] != value:
            raise RuntimeError("Frozen CPU service command differs")
    if "--no-op-offload" not in command:
        raise RuntimeError("CPU-only offload guard is missing")
    if psutil.pid_exists(pause["service"]["pid"]):
        old = psutil.Process(pause["service"]["pid"])
        if old.create_time() == pause["service"]["creation_time"]:
            raise RuntimeError("Old owned service remains alive")
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 11436)) == 0:
            raise RuntimeError("Released loopback port is occupied")
    if sha256(command[0]) != service_receipt["binary_sha256"] or sha256(command[command.index("-m") + 1]) != service_receipt["model_sha256"]:
        raise RuntimeError("Pinned CPU binary/model changed")
    available = psutil.virtual_memory().available
    observations_path = run / "structured_resource_observations.jsonl"
    observations = [json.loads(line) for line in observations_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    guard_mode = choose_guard_path(available, supervised_fallback, observations)
    fallback = guard_mode == "once_only_supervised_2_5_to_3gib"
    fallback_path = run / "STRUCTURED_LOW_RAM_SUPERVISED_ATTEMPT.json"
    if fallback:
        # Exclusive creation makes a second fallback impossible, even after failure.
        with fallback_path.open("x", encoding="utf-8") as out:
            json.dump({"status": "RESERVED_BEFORE_MODEL_LAUNCH", "source_sha256": sha256(__file__),
                "at_utc": datetime.now(timezone.utc).isoformat(), "available_bytes": available,
                "guard_mode": guard_mode, "startup_minimum_bytes": int(2.5 * 1024**3),
                "during_load_and_before_resume_minimum_bytes": 1024**3,
                "basis": "RESUME_RAM_GUARD_PROVENANCE.json; observed CPU steady RSS1.28–1.31GiB; coordinator-authorized once-only supervised alternative; model/CPU12 configuration unchanged",
                "recent_guard_observations": [r for r in observations if r.get("action") == "guard_observation_only_no_model_launch"][-2:]}, out, indent=2)
            out.flush()
            os.fsync(out.fileno())
    receipt = {"status": "STARTING", "started_at": datetime.now(timezone.utc).isoformat(),
        "available_before_bytes": available, "original_restart_guard_bytes": 3 * 1024**3,
        "guard_mode": guard_mode, "supervised_fallback_consumed": fallback,
        "worker": original_worker, "command": command,
        "source_sha256_at_launch": sha256(__file__),
        "source_helper_sha256_at_launch": sha256(Path(__file__).with_name("resume_embedding_after_training.py")),
        "prior_unplanned_recovery_count": pause["prior_unplanned_recovery_count"],
        "unplanned_recovery_limit": pause["unplanned_recovery_limit"],
        "worker_source_note": "Resume existing pre-patch in-memory module; no hot reload",
        "pause_receipt_sha256": sha256(run / "STRUCTURED_LOW_RAM_PAUSE.json")}
    atomic(path, receipt)
    output = (run / "embedding_cpu12_after_structured_low_ram.log").open("w", encoding="utf-8")
    server = process = monitor = None
    created = None
    worker_resumed = False
    try:
        server = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        process = psutil.Process(server.pid)
        created = process.create_time()
        receipt.update(server_pid=server.pid, server_creation_time=created)
        atomic(path, receipt)
        monitor = SupervisedLoadMonitor(process, created, command, run / "SUPERVISED_LOAD_RAM_SAMPLES.jsonl") if fallback else None
        if monitor:
            monitor.start()
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"Owned CPU service exited {server.returncode}")
            if monitor and monitor.failure:
                raise RuntimeError("CPU load monitor failed; worker remains paused")
            try:
                if requests.get("http://127.0.0.1:11436/health", timeout=1).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(.25)
        else:
            raise TimeoutError("Owned CPU service health did not become ready")
        response = requests.post("http://127.0.0.1:11436/v1/embeddings",
            json={"input": "Controlled structured CPU RAM resume verification"}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if len(payload["data"][0]["embedding"]) != 1024:
            raise RuntimeError("Real CPU embedding dimension differs")
        if monitor:
            receipt["load_monitor"] = monitor.close()
            if receipt["load_monitor"]["failure"]:
                raise RuntimeError("Supervised CPU loading failed; worker remains paused")
        minimum_after = 1024**3 if fallback else service_receipt["reserve_ram"]
        if psutil.virtual_memory().available < minimum_after:
            raise RuntimeError("Existing CPU reserve RAM is unavailable after service load")
        now = datetime.now(timezone.utc).isoformat()
        service_receipt.update(status="RUNNING", pid=server.pid, creation_time=created,
            parent_pid=os.getpid(), parent_alive=False, supervisor_pid=None,
            supervisor_note="Controlled resume launcher exits after verified handover",
            command=command, time=now, observed_at=now,
            available_ram_at_observation=psutil.virtual_memory().available, resume_receipt=str(path))
        for key in ("ended_at", "stop_reason"):
            service_receipt.pop(key, None)
        atomic(service_path, service_receipt)
        if (worker.create_time() != original_worker["creation_time"] or worker.cmdline() != original_worker["command"]
                or worker.status() != psutil.STATUS_STOPPED):
            raise RuntimeError("Suspended worker identity changed immediately before resume")
        if psutil.virtual_memory().available < minimum_after:
            raise RuntimeError("Host RAM changed below handoff minimum immediately before resume")
        worker.resume()
        worker_resumed = True
        worker_path = run / "worker.json"
        worker_receipt = json.loads(worker_path.read_text(encoding="utf-8"))
        worker_receipt.update(status="RUNNING", observed_at=now, embedding_service_pid=server.pid,
            resume_receipt=str(path))
        atomic(worker_path, worker_receipt)
        receipt.update(status="PASS_WORKER_RESUMED_NOT_YET_COMPLETE", worker_resumed=True,
            dimensions=1024, verification_usage=payload.get("usage"),
            available_after_bytes=psutil.virtual_memory().available, ended_at=now)
    except Exception as exc:
        if monitor:
            try:
                receipt["load_monitor"] = monitor.close()
            except Exception as monitor_exc:
                receipt["monitor_join_error"] = {"type": type(monitor_exc).__name__, "message": str(monitor_exc)}
        if server is not None and process is not None and created is not None:
            handle_resume_failure(receipt, service_receipt, service_path, server, process,
                created, worker_resumed, exc)
        else:
            receipt.update(status="FAILED_BEFORE_WORKER_RESUME", worker_resumed=False,
                error={"type": type(exc).__name__, "message": str(exc)},
                cleanup_note="No identity-verified service available; worker remains paused")
        raise
    finally:
        atomic(path, receipt)
        if fallback:
            attempt = json.loads(fallback_path.read_text(encoding="utf-8"))
            attempt.update(status=receipt["status"], resume_receipt=str(path),
                worker_resumed=receipt.get("worker_resumed", False), load_monitor=receipt.get("load_monitor"))
            atomic(fallback_path, attempt)
        output.close()
        print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervised-fallback", action="store_true")
    main(parser.parse_args().supervised_fallback)
