"""One writer, durable JSONL records, immutable configs, atomic task transitions."""
from __future__ import annotations
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from filelock import FileLock
from ragscope.indexing.identity import artifact, atomic_json, signature, verify_artifacts

STATES = {"PENDING", "RUNNING", "PASS", "PASS_FALLBACK", "BLOCKED", "FAILED", "NOT_APPLICABLE"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_records(path: Path, repair_tail: bool = False) -> list[dict]:
    """Only newline-terminated, hash-verified records are successes; retain partial bytes."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    rows, offset = [], 0
    for number, line in enumerate(lines):
        if not line.endswith(b"\n"):
            if not repair_tail:
                raise ValueError(f"Incomplete record at {path}:{number + 1}")
            backup = path.with_name(path.name + f".partial.{time.time_ns()}")
            backup.write_bytes(line)
            with path.open("r+b") as handle:
                handle.truncate(offset)
                handle.flush()
                os.fsync(handle.fileno())
            break
        row = json.loads(line)
        expected = row.pop("record_hash", None)
        if expected is not None and signature(row) != expected:
            raise ValueError(f"Corrupt record at {path}:{number + 1}")
        if expected is not None:
            row["record_hash"] = expected
        rows.append(row)
        offset += len(line)
    return rows


class RecordStore:
    """The owning campaign lock must be held across reads/writes."""
    def __init__(self, path: Path, config: dict, *, real_only: bool = False, max_records: int = 10000):
        self.path, self.config_hash = Path(path), signature(config)
        self.real_only, self.max_records = real_only, max_records
        self.path.parent.mkdir(parents=True, exist_ok=True)
        manifest = self.path.with_suffix(".config.json")
        if manifest.exists():
            if json.loads(manifest.read_text(encoding="utf-8")) != config:
                raise ValueError("Immutable record config differs; create a new run")
        else:
            atomic_json(manifest, config)
        self.records = read_records(self.path, repair_tail=True)
        self.ids = set()
        for row in self.records:
            if row.get("config_hash") != self.config_hash or row["record_id"] in self.ids:
                raise ValueError("Record identity/config mismatch")
            if self.real_only and row.get("real_model") is not True:
                raise ValueError("Mock/missing model identity in formal run")
            self.ids.add(row["record_id"])

    def add(self, record_id: str, row: dict) -> bool:
        if record_id in self.ids:
            return False
        if len(self.ids) >= self.max_records:
            raise RuntimeError("Frozen call budget exhausted")
        if self.real_only and (row.get("real_model") is not True or not row.get("model_identity")):
            raise ValueError("Formal generation requires real model and weight/backend identity")
        row = dict(row, record_id=record_id, config_hash=self.config_hash, recorded_at=now())
        row["record_hash"] = signature(row)
        append_jsonl(self.path, row)
        self.records.append(row)
        self.ids.add(record_id)
        return True


class Campaign:
    def __init__(self, root: Path, campaign_id: str):
        if not campaign_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in campaign_id):
            raise ValueError("Unsafe campaign ID")
        self.path = Path(root) / "runs" / campaign_id
        self.path.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(self.path / "writer.lock"), timeout=0)
        self.campaign_id = campaign_id

    def __enter__(self):
        self.lock.acquire()
        state_path = self.path / "state.json"
        self.state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {
            "campaign_id": self.campaign_id, "created_at": now(), "status": "RUNNING", "tasks": {}, "generation_calls": 0}
        for name in ("tasks", "events", "commands"):
            (self.path / f"{name}.jsonl").touch(exist_ok=True)
            read_records(self.path / f"{name}.jsonl", repair_tail=True)
        for task_id, task in self.state["tasks"].items():
            if task["status"] == "RUNNING":
                task["status"] = "PENDING"
                self.event("interrupted_task_requeued", task_id=task_id)
        self.save()
        return self

    def __exit__(self, *args):
        self.save()
        self.lock.release()

    def save(self):
        self.state["updated_at"] = now()
        atomic_json(self.path / "state.json", self.state)

    def event(self, kind: str, **fields):
        append_jsonl(self.path / "events.jsonl", {"time": now(), "event": kind, **fields})

    def begin(self, task_id: str, config: dict, command: list[str], dependencies: list[str] = ()) -> bool:
        old = self.state["tasks"].get(task_id, {})
        fingerprint = signature(config)
        if old.get("status") in {"PASS", "PASS_FALLBACK"}:
            if old.get("input_fingerprint") != fingerprint or not verify_artifacts(old.get("outputs", [])):
                raise ValueError(f"Completed task changed inputs/outputs: {task_id}; use a new run")
            self.event("resume_skip", task_id=task_id)
            return False
        if old.get("attempts", 0) >= 3:
            raise RuntimeError(f"Task retry limit reached: {task_id}")
        for dependency in dependencies:
            if self.state["tasks"].get(dependency, {}).get("status") not in {"PASS", "PASS_FALLBACK"}:
                raise ValueError(f"Unmet dependency: {dependency}")
        self.state["tasks"][task_id] = {
            "task_id": task_id, "dependencies": list(dependencies), "status": "RUNNING",
            "attempts": old.get("attempts", 0) + 1, "input_fingerprint": fingerprint, "config": config,
            "command": command, "started_at": now(), "ended_at": None, "outputs": [], "validation": {}}
        append_jsonl(self.path / "commands.jsonl", {"time": now(), "task_id": task_id, "argv": command})
        self.save()
        return True

    def finish(self, task_id: str, status: str, outputs: list[Path], validation: dict):
        if status not in STATES - {"RUNNING", "PENDING"}:
            raise ValueError(status)
        task = self.state["tasks"][task_id]
        task.update(status=status, ended_at=now(), outputs=[artifact(p) for p in outputs], validation=validation)
        append_jsonl(self.path / "tasks.jsonl", task)
        self.save()
        manifest = {key: item["outputs"] for key, item in self.state["tasks"].items()}
        atomic_json(self.path / "artifact_manifest.json", manifest)
