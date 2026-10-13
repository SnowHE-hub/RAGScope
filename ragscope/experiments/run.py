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
from ragscope.indexing.identity import file_hash


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


def main():
    parser = argparse.ArgumentParser(description="RAGScope v2 可恢复本地执行器")
    parser.add_argument("--profile", choices=["full", "smoke", "recovery-fixture"], default="full")
    parser.add_argument("--campaign", default="20261014_v2")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--task", action="append")
    parser.add_argument("--interrupt-after", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--partial-tail", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    configure(cpu_only=True)  # GPU tasks must obtain a separate verified child lease.
    with Campaign(runtime_home(), args.campaign) as campaign:
        campaign.event("runner_start", profile=args.profile, cpu_only=True)
        if args.profile == "recovery-fixture":
            fixture(campaign, args.interrupt_after, args.partial_tail)
        elif args.profile == "smoke":
            fixture(campaign)
            task = {"id": "CPU_SMOKE", "command": ["{python}", "-m", "pytest", "tests/unit", "-q", "-p", "no:cacheprovider"], "timeout_seconds": 300,
                    "input_files": ["pytest.ini"]}
            execute(campaign, task)
        else:
            plan = Path(__file__).resolve().parents[2] / "configs/v2/tasks.json"
            tasks = json.loads(plan.read_text(encoding="utf-8"))["tasks"]
            if args.task:
                selected = set(args.task)
                unknown = selected - {task["id"] for task in tasks}
                if unknown:
                    parser.error(f"Unknown tasks: {sorted(unknown)}")
                tasks = [task for task in tasks if task["id"] in selected]
            for task in tasks:
                try:
                    execute(campaign, task)
                except (ValueError, RuntimeError) as exc:
                    campaign.event("task_cannot_run", task_id=task["id"], error=str(exc))
                    print(f"UNRESOLVED {task['id']}: {exc}", flush=True)
        failures = [key for key, task in campaign.state["tasks"].items() if task["status"] == "FAILED"]
        campaign.event("runner_end", failures=failures)
    raise SystemExit(1 if failures else 0)

if __name__ == "__main__":
    main()
