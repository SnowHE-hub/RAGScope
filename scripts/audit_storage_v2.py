"""Read-only accounting of project runtime files, aliases and effective caches.

Never traverses directory links, scans user caches, deletes files, or attributes
whole-drive free-space changes exclusively to this project.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragscope.experiments.runtime import configure, runtime_home
from ragscope.indexing.identity import artifact, atomic_json


def inventory(root):
    root = Path(root).resolve()
    buckets = defaultdict(lambda: {"files": 0, "logical_bytes": 0,
                                   "unique_file_logical_bytes": 0})
    seen, links, errors, pending = set(), [], [], [root]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(directory.iterdir())
        except OSError as exc:
            errors.append({"path": str(directory), "error": str(exc)})
            continue
        for entry in entries:
            relative = entry.relative_to(root)
            try:
                if entry.is_symlink() or entry.is_junction():
                    target = entry.resolve()
                    links.append({"path": str(relative), "target": str(target),
                                  "within_runtime": target.is_relative_to(root),
                                  "traversed": False})
                    continue
                if entry.is_dir():
                    pending.append(entry)
                    continue
                stat = entry.stat()
                bucket = buckets[relative.parts[0] if len(relative.parts) > 1 else "<root>"]
                bucket["files"] += 1
                bucket["logical_bytes"] += stat.st_size
                key = (stat.st_dev, stat.st_ino) if stat.st_ino else str(entry)
                if key not in seen:
                    bucket["unique_file_logical_bytes"] += stat.st_size
                    seen.add(key)
            except OSError as exc:
                errors.append({"path": str(entry), "error": str(exc)})
    return {"root": str(root), "buckets": dict(sorted(buckets.items())),
            "totals": {key: sum(value[key] for value in buckets.values())
                       for key in ("files", "logical_bytes", "unique_file_logical_bytes")},
            "links": links, "errors": errors,
            "measurement": "File lengths; unique lengths deduplicate hard links by file identity. "
                           "Not physical allocated bytes (compression, sparse extents and allocation slack differ). "
                           "Files can change during a live snapshot; no links are traversed."}


def build_report(home, campaign, snapshot_name):
    paths = configure(cpu_only=True)
    if Path(paths["home"]).resolve() != home.resolve():
        raise ValueError("Requested runtime differs from effective process configuration")
    run = home / "runs" / campaign
    baseline = run / "disk_before.json"
    before = json.loads(baseline.read_text(encoding="utf-8-sig"))
    disks = []
    for item in before:
        drive = Path(item["DriveLetter"] + ":/")
        if drive.exists():
            usage = shutil.disk_usage(drive)
            disks.append({"drive": str(drive), "free_before_bytes": item["SizeRemaining"],
                          "free_now_bytes": usage.free, "total_bytes": usage.total,
                          "whole_drive_used_delta_bytes": item["SizeRemaining"] - usage.free,
                          "exclusive_project_attribution": False})
    data = inventory(home)
    cache_paths = [{"variable": key, "path": value, "resolved_path": str(Path(value).resolve()),
                    "within_runtime": Path(value).resolve().is_relative_to(home.resolve())}
                   for key, value in paths["paths"].items()]
    report = {"version": "runtime-storage-audit-v1", "snapshot": snapshot_name,
              "time_utc": datetime.now(timezone.utc).isoformat(), "runtime": data,
              "disks": disks, "cache_paths": cache_paths, "baseline": artifact(baseline),
              "python": {"executable": sys.executable, "base_prefix": sys.base_prefix},
              "known_c_drive_exceptions": [
                  {"path": "C:/Windows/Fonts/msyh.ttc", "use": "Existing Chinese font read for local figures; not copied or installed"},
                  {"path": "C:/Users/lenovo/.codex", "use": "Pre-existing desktop agent/plugin state outside project control; not scanned or redirected"},
                  {"path": "C:/Users/lenovo/.local/bin/uv.exe", "use": "Existing dependency manager executable; its download/cache/Python destinations are process-locally set to D"},
                  {"path": "C:/Users/lenovo/AppData/Local/Programs/Ollama", "use": "Pre-existing executable only; project service explicitly uses D:/RAGScope_Runtime/models/ollama"}],
              "limitations": ["Cache variables are for this audit process; service and model receipts separately prove their inherited runtime paths.",
                              "Whole-drive C/D deltas include other applications and OS activity; they cannot prove zero project writes to C.",
                              "WSL was used for read-only resource/original-job checks and scoped signals; alternative CPU embedding was blocked before model startup. No project Linux model environment was installed."],
              "mutations": ["Own JSON audit snapshot and process-local runtime environment only"]}
    report["status"] = "PASS_PATH_ACCOUNTING" if not data["errors"] and all(
        row["within_runtime"] for row in cache_paths + data["links"]) else "REVIEW_REQUIRED"
    target = run / f"storage_{snapshot_name}.json"
    if target.exists():
        raise FileExistsError("Storage snapshots are immutable; select a fresh snapshot name")
    atomic_json(target, report)
    return report, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="20261014_v2")
    parser.add_argument("--snapshot", required=True)
    args = parser.parse_args()
    if not args.snapshot.replace("_", "").replace("-", "").isalnum():
        parser.error("Snapshot must be a simple alphanumeric name")
    report, target = build_report(runtime_home(), args.campaign, args.snapshot)
    print(json.dumps({"path": str(target), "status": report["status"],
                      "totals": report["runtime"]["totals"], "disks": report["disks"]}))
    return 0 if report["status"] == "PASS_PATH_ACCOUNTING" else 2


if __name__ == "__main__":
    raise SystemExit(main())
