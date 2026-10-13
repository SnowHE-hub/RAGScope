"""Additive plan progress backed by immutable executions and verified child evidence.

This module never starts/stops a model service or the user's original workload.
Historical Campaign.tasks entries are not rewritten by reconciliation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fnmatch
import json
import os
from pathlib import Path
import re
import sys

from filelock import FileLock, Timeout
from ragscope.experiments.ledger import append_jsonl, now
from ragscope.indexing.identity import artifact, atomic_json, signature, verify_artifacts

PROJECT = Path(__file__).resolve().parents[2]
COMPLETE = {"PASS", "PASS_FALLBACK"}
PLAN_IDS = tuple(f"S{stage}.{number}" for stage, size in ((1, 6), (2, 6), (3, 7), (4, 6))
                 for number in range(1, size + 1))


def load_catalog(path=None):
    catalog = json.loads(Path(path or PROJECT / "configs/v2/tasks.json").read_text(encoding="utf-8-sig"))
    tasks = catalog["tasks"]
    if tuple(task["id"] for task in tasks) != PLAN_IDS:
        raise ValueError("The master plan has 25 mandatory tasks including S3.7")
    seen = set()
    for task in tasks:
        if any(dependency not in seen for dependency in task.get("dependencies", [])):
            raise ValueError("Task dependency must precede its dependent")
        seen.add(task["id"])
    return catalog


def expand(value, home, campaign, project=PROJECT):
    suffix = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    # Linux receives its own venv; never point Linux at Scripts/python.exe.
    values = {"{python}": sys.executable, "{runtime}": str(home), "{campaign}": campaign,
              "{project}": str(project), "{core_python}": str(home / "envs" / ("core" if os.name == "nt" else "core-linux") / suffix),
              "{models_python}": str(home / "envs" / ("models" if os.name == "nt" else "models-linux") / suffix),
              "{train_python}": str(home / "envs" / ("train" if os.name == "nt" else "train-linux") / suffix)}
    for key, replacement in values.items():
        value = value.replace(key, replacement)
    return value


def locked(path):
    lock = FileLock(str(path), timeout=0)
    try:
        lock.acquire()
    except Timeout:
        return True
    else:
        lock.release()
        return False


def verify_execution(row, home):
    outputs = row.get("outputs", [])
    if not outputs or not row.get("input_fingerprint"):
        return False
    if row.get("config") is not None and signature(row["config"]) != row["input_fingerprint"]:
        return False
    if any(not Path(item["path"]).resolve().is_relative_to(home.resolve()) for item in outputs):
        return False
    return verify_artifacts(outputs)


def live_worker(directory):
    """A durable PID receipt supplements short or absent campaign lock ownership."""
    path = directory / "worker.json"
    if not path.is_file():
        return None
    import psutil
    try:
        worker = json.loads(path.read_text(encoding="utf-8"))
        if worker.get("status") not in {"RUNNING", "SUSPENDED"} or worker.get("campaign_id") != directory.name:
            return None
        process = psutil.Process(worker["pid"])
        if (not process.is_running() or process.status() == psutil.STATUS_ZOMBIE
                or abs(process.create_time() - worker["creation_time"]) > .001
                or process.cmdline() != worker["command"]
                or Path(process.cwd()).resolve() != PROJECT.resolve()):
            return None
        return {"receipt": artifact(path), "pid": worker["pid"], "creation_time": worker["creation_time"],
                "command": worker["command"], "stage": worker.get("stage"), "device": worker.get("device"),
                "status": worker["status"]}
    except (psutil.Error, ValueError, OSError, KeyError):
        return None


def child_evidence(task, home, campaign):
    """Read only final child outputs; a live child is RUNNING, never BLOCKED by absence."""
    proofs, errors, active, workers = [], [], False, []
    for source in task.get("child_sources", []):
        directory = home / "runs" / (campaign + source["suffix"])
        state_path = directory / "state.json"
        if not state_path.is_file():
            continue
        child = json.loads(state_path.read_text(encoding="utf-8"))
        worker = live_worker(directory)
        if worker:
            workers.append(worker)
        child_active = worker["status"] == "RUNNING" if worker else locked(directory / "writer.lock")
        active = active or (child_active and source.get("active_when_locked", False))
        for task_id, row in child.get("tasks", {}).items():
            if not any(fnmatch.fnmatchcase(task_id, pattern) for pattern in source["patterns"]):
                continue
            active = active or (child_active and row.get("status") == "RUNNING")
            if row.get("status") not in COMPLETE:
                continue
            if not verify_execution(row, home):
                errors.append({"source": str(state_path), "task_id": task_id, "error": "completed_child_artifact_or_identity_changed"})
                continue
            # Summaries are already hash verified above; retain their observed metrics verbatim.
            summaries = []
            for output in row["outputs"]:
                path = Path(output["path"])
                if path.suffix == ".json" and (path.name == "summary.json" or path.parent.name == "summaries"):
                    summaries.append({"artifact": output, "observed": json.loads(path.read_text(encoding="utf-8"))})
            proofs.append({"source_campaign": child["campaign_id"], "source_task_id": task_id,
                           "source_task_hash": signature(row), "input_fingerprint": row["input_fingerprint"],
                           "status": row["status"], "validation": row.get("validation", {}),
                           "outputs": row["outputs"], "summaries": summaries,
                           "comparison_identity": {"query_ids_hash": signature(row.get("config", {}).get("query_ids", [])),
                             **{key: row.get("config", {}).get("index_identity", {}).get(key)
                                for key in ("embedding", "tokenizer", "data_manifest_sha256", "split")}}})
    return proofs, errors, active, workers


def completion_receipt(task, home, campaign):
    """Explicit stage signoff must cite evidence, not merely a supported code interface."""
    path = home / "runs" / campaign / "completion" / f"{task['id']}.json"
    if not path.exists():
        return None
    receipt = json.loads(path.read_text(encoding="utf-8"))
    required = set(task["completion_requirements"])
    if (receipt.get("task_id") != task["id"] or receipt.get("task_spec_hash") != signature(task)
            or receipt.get("status") not in COMPLETE | {"BLOCKED", "FAILED"}
            or not receipt.get("artifacts") or not verify_artifacts(receipt["artifacts"])
            or any(not any(Path(item["path"]).resolve().is_relative_to(root.resolve())
                           for root in (home, PROJECT)) for item in receipt["artifacts"])):
        raise ValueError("Completion receipt identity or evidence is invalid")
    if receipt["status"] in COMPLETE:
        if not required <= set(receipt.get("requirements_met", [])):
            raise ValueError("Completion receipt does not cover every mandatory requirement")
        if task["kind"] == "experiment" and receipt.get("real_experiment") is not True:
            raise ValueError("Experiment completion requires actual experiments, not interfaces")
    elif not receipt.get("reason"):
        raise ValueError("A hard blocking/failure receipt requires its concrete reason")
    return {"artifact": artifact(path), "receipt": receipt}


def command_views(task, home, campaign, plan_tasks, allow_gpu=False):
    rows = []
    for spec in task.get("commands", []):
        argv = [expand(value, home, campaign) for value in spec["argv"]]
        missing = [expand(value, home, campaign) for value in spec.get("requires", [])
                   if not Path(expand(value, home, campaign)).is_file()]
        reasons = ["missing_file:" + value for value in missing]
        reasons += ["dependency:" + dependency for dependency in task.get("dependencies", [])
                    if plan_tasks.get(dependency, {}).get("status") not in COMPLETE]
        if Path(argv[0]).is_absolute() and not Path(argv[0]).is_file():
            reasons.append("interpreter_unavailable:" + argv[0])
        if spec.get("gpu") and not allow_gpu:
            reasons.append("gpu_execution_not_enabled_in_this_runner_invocation")
        if spec.get("external_coordination"):
            reasons.append("external_resource_handoff_required")
        if spec.get("campaign_scope") and spec["campaign_scope"] != campaign:
            reasons.append("command_is_scoped_to_campaign:" + spec["campaign_scope"])
        for suffix in spec.get("lock_campaigns", []):
            folder = home / "runs" / (campaign + suffix)
            worker = live_worker(folder) if folder.is_dir() else None
            if worker and worker["status"] == "SUSPENDED":
                reasons.append("child_worker_suspended_waiting_resource:" + str(folder))
            elif folder.is_dir() and (worker or locked(folder / "writer.lock")):
                reasons.append("child_campaign_running:" + str(folder))
        rows.append({"argv": argv, "purpose": spec["purpose"], "gpu": spec.get("gpu", False),
                     "runnable": not reasons, "waiting_for": reasons, "spec": spec})
    return rows


def reconcile(campaign, catalog, *, allow_gpu=False, persist=True):
    home = campaign.path.parents[1]
    progress = {}
    for task in catalog["tasks"]:
        task_id = task["id"]
        proofs, errors, active, workers = child_evidence(task, home, campaign.campaign_id)
        completed_execution = None
        executions = [(key, value) for key, value in campaign.state["tasks"].items()
                      if key == task_id or value.get("config", {}).get("plan_task_id") == task_id]
        executions.sort(key=lambda pair: (pair[1].get("started_at") or "", pair[0]))
        for execution_id, row in executions:
            if row["status"] in COMPLETE:
                if verify_execution(row, home):
                    config = row.get("config", {})
                    if execution_id != task_id and (config.get("task_spec_hash") != signature(task)
                            or any(not Path(path).is_file() or artifact(path)["sha256"] != digest
                                   for path, digest in config.get("inputs", {}).items())):
                        continue  # Historical success remains immutable; changed inputs need a new execution.
                    # External commands can end successfully while the experiment remains incomplete.
                    if execution_id == task_id or row.get("config", {}).get("completes_task"):
                        completed_execution = {"execution_id": execution_id, "receipt": row}
                else:
                    errors.append({"execution_id": execution_id, "error": "completed_execution_evidence_changed"})
        receipt = None
        try:
            receipt = completion_receipt(task, home, campaign.campaign_id)
        except (ValueError, OSError, KeyError) as exc:
            errors.append({"error": "invalid_completion_receipt", "detail": str(exc)})
        status = "RUNNING" if active else "PENDING"
        if completed_execution:
            status = completed_execution["receipt"]["status"]
        if receipt:
            status = receipt["receipt"]["status"]
        # A bounded complete child matrix may prove the task directly; partial rows never do.
        expected = task.get("required_child_task_ids", [])
        if expected and set(expected) <= {proof["source_task_id"] for proof in proofs}:
            matrix = [proof for proof in proofs if proof["source_task_id"] in expected and proof["source_task_id"].startswith("E1_")]
            dense = [proof["comparison_identity"]["embedding"] for proof in matrix
                     if "_dense_" in proof["source_task_id"] or "_hybrid_" in proof["source_task_id"]]
            if task.get("verify_fixed_embedding") and (
                    not dense or any(not item or item.get("device") != "cpu" for item in dense)
                    or len({signature(item) for item in dense}) != 1
                    or any(len({signature(proof["comparison_identity"][key]) for proof in matrix}) != 1
                           for key in ("query_ids_hash", "tokenizer", "data_manifest_sha256", "split"))):
                errors.append({"error": "completed_matrix_has_inconsistent_embedding_queries_or_corpus"})
            else:
                status = "PASS"
        if errors:
            status = "FAILED"
        actions = command_views(task, home, campaign.campaign_id, progress, allow_gpu)
        latest = executions[-1][1] if executions else {}
        if (status == "PENDING" and latest.get("status") == "FAILED"
                and not latest.get("validation", {}).get("waiting_for_prerequisites")):
            status = "FAILED"
        item = {"task_id": task_id, "title": task["title"], "kind": task["kind"],
                "dependencies": task.get("dependencies", []), "status": status,
                "attempts": sum(row.get("attempts", 0) for _, row in executions),
                "input_fingerprint": signature(task), "commands": actions,
                "started_at": latest.get("started_at"), "ended_at": latest.get("ended_at"),
                "completion_requirements": task["completion_requirements"],
                "evidence": proofs, "verified_child_tasks": len(proofs),
                "active_workers": workers,
                "resource_wait": any(worker["status"] == "SUSPENDED" for worker in workers),
                "execution": completed_execution, "completion_receipt": receipt,
                "validation_errors": errors, "outputs": latest.get("outputs", []),
                "remaining": [] if status in COMPLETE else task["completion_requirements"]}
        progress[task_id] = item
        if persist:
            previous = campaign.state.get("plan_tasks", {}).get(task_id)
            if previous != item:
                append_jsonl(campaign.path / "plan_events.jsonl", {"time": now(), "task": item})
            # Observed child proof is content-addressed and append-only.
            for proof in proofs:
                target = campaign.path / "reconciled_evidence" / task_id / (signature(proof) + ".json")
                if not target.exists():
                    atomic_json(target, proof)
    result = {"campaign_id": campaign.campaign_id, "catalog_hash": signature(catalog),
              "task_count": len(progress), "state_counts": dict(Counter(row["status"] for row in progress.values())),
              "status": "COMPLETE" if all(row["status"] in COMPLETE for row in progress.values()) else "INCOMPLETE",
              "resource_policy": catalog["resource_policy"], "tasks": progress}
    if persist:
        campaign.state["plan_tasks"] = progress
        campaign.state["plan_status"] = result["status"]
        campaign.state["task_catalog_hash"] = result["catalog_hash"]
        atomic_json(campaign.path / "plan_status.json", result)
        campaign.save()
    return result


def summary(result):
    return {key: result[key] for key in ("campaign_id", "task_count", "status", "state_counts", "resource_policy")} | {
        "tasks": {key: {field: row[field] for field in
                   ("title", "status", "attempts", "verified_child_tasks", "commands", "remaining", "validation_errors")}
                  for key, row in result["tasks"].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-audit", action="store_true")
    parser.add_argument("--check-lease", action="store_true")
    parser.add_argument("--campaign", default="20261014_v2")
    args = parser.parse_args()
    if args.check_audit:
        path = PROJECT / "docs/execution/CODE_AUDIT.md"
        numbers = set(re.findall(r"^\|\s*(\d+)\s*\|", path.read_text(encoding="utf-8"), re.MULTILINE))
        if numbers != {str(number) for number in range(1, 16)}:
            raise ValueError("Local code audit must cover all 15 original findings")
        print(json.dumps({"audit": str(path), "findings": 15, "scope": "audit_completeness_not_experiment_quality"}))
        return 0
    if args.check_lease:
        from ragscope.experiments.runtime import runtime_home
        from ragscope.training.run import verify_lease
        try:
            lease = verify_lease(runtime_home() / "runs" / args.campaign / "gpu_lease.json")
        except (ValueError, OSError) as exc:
            print(json.dumps({"status": "PENDING", "reason": str(exc), "original_restore": "manual_user_only"}))
            return 2
        print(json.dumps({"status": "PASS_LEASE_VALIDATOR_ONLY", "gpu_device": lease["gpu_device"],
                          "original_restore": "manual_user_only", "process_actions": []}))
        return 0
    parser.error("Select --check-audit or --check-lease")


if __name__ == "__main__":
    raise SystemExit(main())
