"""Command-line orchestrator. Every command and record belongs to one locked campaign."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from ragscope.experiments.runtime import configure, runtime_home
from ragscope.experiments.ledger import Campaign, RecordStore
from ragscope.indexing.identity import file_hash, signature
from ragscope.experiments.task_catalog import COMPLETE, load_catalog, reconcile, expand


def fixture(campaign, interrupt_after=0, partial_tail=False):
    spec = {"kind": "offline_recovery_fixture", "count": 5, "version": 1}
    command = [sys.executable, "-m", "ragscope.experiments.run", "--profile", "recovery-fixture", "--resume"]
    if not campaign.begin("RECOVERY_FIXTURE", spec, command):
        return
    records = RecordStore(campaign.path / "fixture.jsonl", spec)
    for i in range(5):
        if f"fixture-{i}" in records.ids:
            continue
        records.add(f"fixture-{i}", {"input": i, "output": i * i, "real_model": False, "purpose": "offline_functional_test"})
        if interrupt_after and len(records.ids) >= interrupt_after:
            if partial_tail:
                with records.path.open("ab") as handle:
                    handle.write(b'{"record_id":"incomplete')
                    handle.flush()
                    os.fsync(handle.fileno())
            os._exit(99)  # only this dedicated artificial fixture process
    campaign.finish("RECOVERY_FIXTURE", "PASS", [records.path, records.path.with_suffix(".config.json")],
                    {"records": len(records.ids), "unique": len(records.ids) == 5, "real_experiment": False})


def execute(campaign, task):
    root = Path(__file__).resolve().parents[2]
    replacements = {"{python}": sys.executable, "{runtime}": str(runtime_home()), "{project}": str(root), "{campaign}": campaign.campaign_id}
    def expand(value):
        for key, replacement in replacements.items():
            value = value.replace(key, replacement)
        return value
    command = [expand(arg) for arg in task["command"]]
    config = {"task": task, "inputs": {path: file_hash(root / path) for path in task.get("input_files", [])}}
    if not campaign.begin(task["id"], config, command, task.get("dependencies", [])):
        print(f"SKIP verified {task['id']}", flush=True)
        return
    attempt = campaign.state["tasks"][task["id"]]["attempts"]
    log = campaign.path / "logs" / f"{task['id']}.attempt{attempt}.log"
    log.parent.mkdir(exist_ok=True)
    print(f"RUN {task['id']} -> {log}", flush=True)
    try:
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.run(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT,
                                     timeout=task.get("timeout_seconds", 7200), env=os.environ.copy())
        outputs = [Path(expand(path)) for path in task.get("outputs", [])]
        missing = [str(path) for path in outputs if not path.is_file()]
        success = process.returncode == 0 and not missing
        campaign.finish(task["id"], "PASS" if success else "FAILED", [log] + [p for p in outputs if p.exists()],
                        {"exit_code": process.returncode, "missing_outputs": missing})
    except subprocess.TimeoutExpired:
        campaign.finish(task["id"], "FAILED", [log], {"exit_code": None, "error": "task_timeout"})
    except BaseException as exc:
        campaign.event("executor_interruption", task_id=task["id"], error=type(exc).__name__)
        raise


def execute_plan_command(campaign, task, action):
    """Version each new execution; never replace historical task identities."""
    root = Path(__file__).resolve().parents[2]
    inputs = {str(path.resolve()): file_hash(path) for path in (root / "ragscope").rglob("*.py")}
    for value in task.get("input_files", []):
        path = root / value
        inputs[str(path.resolve())] = file_hash(path)
    for value in action["argv"]:
        path = root / value
        if value.endswith(".py") and path.is_file():
            inputs[str(path.resolve())] = file_hash(path)
    for value in action["spec"].get("requires", []):
        path = Path(expand(value, campaign.path.parents[1], campaign.campaign_id))
        inputs[str(path.resolve())] = file_hash(path)
    config = {"plan_task_id": task["id"], "task_spec_hash": signature(task), "argv": action["argv"],
              "inputs": inputs, "completes_task": task.get("execution_completes_task", False)}
    execution_id = "PLAN_" + task["id"] + "_" + signature(config)[:16]
    # Failed work has three attempts across input revisions. Prerequisite checks are
    # bounded per unchanged config, but do not consume failures after inputs change.
    attempts = sum(row.get("attempts", 0) for row in campaign.state["tasks"].values()
                   if row.get("config", {}).get("plan_task_id") == task["id"]
                   and row.get("config", {}).get("argv") == action["argv"]
                   and not row.get("validation", {}).get("waiting_for_prerequisites"))
    old = campaign.state["tasks"].get(execution_id, {})
    if old.get("status") not in COMPLETE and attempts >= 3:
        raise RuntimeError("Bounded task command retry limit reached")
    environment = os.environ.copy()
    environment.update(action["spec"].get("environment", {}))
    if action["gpu"]:
        # GPU callers must explicitly enable the invocation and arrange an owned service.
        # Never stop, start, or restore any user process here.
        from ragscope.training.run import verify_lease
        verify_lease(campaign.path / "gpu_lease.json")
    if not campaign.begin(execution_id, config, action["argv"]):
        return
    attempt = campaign.state["tasks"][execution_id]["attempts"]
    log = campaign.path / "logs" / f"{execution_id}.attempt{attempt}.log"
    log.parent.mkdir(exist_ok=True)
    print(f"RUN {task['id']} ({execution_id}) -> {log}", flush=True)
    try:
        with log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(action["argv"], cwd=root, env=environment, stdout=handle,
                                    stderr=subprocess.STDOUT, timeout=task.get("timeout_seconds", 7200))
        outputs = [Path(expand(path, campaign.path.parents[1], campaign.campaign_id)) for path in task.get("outputs", [])]
        missing = [str(path) for path in outputs if not path.is_file()]
        campaign.finish(execution_id, "PASS" if result.returncode == 0 and not missing else "FAILED",
                        [log] + [path for path in outputs if path.is_file()],
                        {"exit_code": result.returncode, "missing_outputs": missing,
                         "purpose": action["purpose"], "experiment_completion_implied": False,
                         "waiting_for_prerequisites": result.returncode in action["spec"].get("pending_exit_codes", [])})
    except subprocess.TimeoutExpired:
        campaign.finish(execution_id, "FAILED", [log], {"error": "task_timeout"})


def main(argv=None):
    parser = argparse.ArgumentParser(description="RAGScope v2 可恢复本地执行器")
    parser.add_argument("--profile", choices=["full", "smoke", "recovery-fixture"], default="full")
    parser.add_argument("--campaign", default="20261014_v2")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--task", action="append")
    parser.add_argument("--reconcile-only", action="store_true", help="Verify evidence and persist all plan states without launching task commands")
    parser.add_argument("--allow-gpu-tasks", action="store_true", help="Allow already coordinated GPU commands; never starts or stops services")
    parser.add_argument("--coordinated-task", action="append", default=[], help="Task ID whose external resource handoff has been performed")
    parser.add_argument("--interrupt-after", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--partial-tail", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.cpu_only and args.allow_gpu_tasks:
        parser.error("--cpu-only and --allow-gpu-tasks conflict")
    configure(cpu_only=not args.allow_gpu_tasks)
    unresolved = []
    with Campaign(runtime_home(), args.campaign) as campaign:
        campaign.event("runner_start", profile=args.profile, cpu_only=not args.allow_gpu_tasks,
                       resume=args.resume, original_restore_policy="manual_user_only")
        if args.profile == "recovery-fixture":
            fixture(campaign, args.interrupt_after, args.partial_tail)
        elif args.profile == "smoke":
            fixture(campaign)
            task = {"id": "CPU_SMOKE", "command": ["{python}", "-m", "pytest", "tests/unit", "-q", "-p", "no:cacheprovider"], "timeout_seconds": 300,
                    "input_files": ["pytest.ini"]}
            execute(campaign, task)
        else:
            catalog = load_catalog()
            tasks = catalog["tasks"]
            if args.task:
                selected = set(args.task)
                unknown = selected - {task["id"] for task in tasks}
                if unknown:
                    parser.error(f"Unknown tasks: {sorted(unknown)}")
                tasks = [task for task in tasks if task["id"] in selected]
            report = reconcile(campaign, catalog, allow_gpu=args.allow_gpu_tasks)
            if not args.reconcile_only:
                for task in tasks:
                    if report["tasks"][task["id"]]["status"] in COMPLETE | {"RUNNING"}:
                        continue
                    for action in report["tasks"][task["id"]]["commands"]:
                        waits = list(action["waiting_for"])
                        if task["id"] in args.coordinated_task:
                            waits = [item for item in waits if item != "external_resource_handoff_required"]
                        if waits:
                            continue
                        try:
                            execute_plan_command(campaign, task, action)
                        except (ValueError, RuntimeError, OSError) as exc:
                            campaign.event("task_cannot_run", task_id=task["id"], error=str(exc))
                            print(f"UNRESOLVED {task['id']}: {exc}", flush=True)
                            break
                    report = reconcile(campaign, catalog, allow_gpu=args.allow_gpu_tasks)
            unresolved = [task["id"] for task in tasks if report["tasks"][task["id"]]["status"] not in COMPLETE]
            print(json.dumps({"task_count": report["task_count"], "status": report["status"],
                              "state_counts": report["state_counts"], "requested_unresolved": unresolved,
                              "status_file": str(campaign.path / "plan_status.json")}, ensure_ascii=False), flush=True)
        failures = [key for key, task in campaign.state["tasks"].items() if task["status"] == "FAILED"]
        campaign.event("runner_end", failures=failures, requested_unresolved=unresolved)
    return 2 if unresolved else (1 if failures and args.profile != "full" else 0)

if __name__ == "__main__":
    raise SystemExit(main())
