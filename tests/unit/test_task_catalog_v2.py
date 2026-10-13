"""Small offline orchestration fixtures; no models or experiment services run."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ragscope.experiments.ledger import Campaign
from ragscope.experiments.run import execute_plan_command, main as run_main
from ragscope.experiments.task_catalog import (PLAN_IDS, child_evidence, command_views, completion_receipt,
                                             expand, load_catalog, reconcile, verify_dev_matrix_rows)
from ragscope.indexing.identity import artifact, atomic_json, signature


def task(task_id="S2.3", **kwargs):
    return {"id": task_id, "title": "Offline fixture", "kind": "experiment", "dependencies": [],
            "commands": [], "completion_requirements": ["actual_result"], **kwargs}


def catalog(*tasks):
    return {"tasks": list(tasks), "resource_policy": {"original_task_restore": "manual_user_only",
                                                     "automatic_process_control": False}}


def child(tmp_path, name="demo_retrieval", task_id="E1_fixture"):
    with Campaign(tmp_path, name) as campaign:
        campaign.begin(task_id, {"fixture": "offline_only"}, ["offline-test"])
        path = campaign.path / "summaries" / "result.json"
        atomic_json(path, {"n": 2, "failures": 1, "observed_fixture_metric": .5})
        campaign.finish(task_id, "PASS", [path], {"queries": 2, "failures": 1})
    return path


def test_catalog_covers_all_25_master_plan_tasks_and_no_process_control():
    plan = load_catalog()
    assert tuple(row["id"] for row in plan["tasks"]) == PLAN_IDS
    assert len(PLAN_IDS) == 25 and "S3.7" in PLAN_IDS
    assert plan["resource_policy"]["original_task_restore"] == "manual_user_only"
    commands = json.dumps([row["commands"] for row in plan["tasks"]])
    assert "manage_previous_jobs" not in commands and "Stop-Process" not in commands
    assert "core-linux" not in commands  # Resolved per host, never a hard-coded Windows venv for Linux.
    generation = next(row for row in plan["tasks"] if row["id"] == "S3.5")
    validation = next(row for row in generation["commands"] if "scripts/accept_generation_v2.py" in row["argv"])
    assert validation["argv"][-2:] == ["--scope", "main"] and validation["pending_exit_codes"] == [2]
    assert not validation.get("gpu", False)


def test_catalog_rejects_omission_and_bad_dependency(tmp_path):
    plan = load_catalog()
    path = tmp_path / "tasks.json"
    broken = copy.deepcopy(plan)
    broken["tasks"].pop(18)
    atomic_json(path, broken)
    with pytest.raises(ValueError, match="25"):
        load_catalog(path)
    plan["tasks"][0]["dependencies"] = ["S4.6"]
    atomic_json(path, plan)
    with pytest.raises(ValueError, match="precede"):
        load_catalog(path)


def test_full_reconcile_only_returns_nonzero_and_persists_every_pending_task(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGSCOPE_HOME", str(tmp_path))
    monkeypatch.setattr("ragscope.experiments.run.execute_plan_command", lambda *args: pytest.fail("Must not launch"))
    assert run_main(["--profile", "full", "--campaign", "empty", "--resume", "--reconcile-only"]) == 2
    saved = json.loads((tmp_path / "runs/empty/state.json").read_text())
    assert len(saved["plan_tasks"]) == 25
    assert set(row["status"] for row in saved["plan_tasks"].values()) == {"PENDING"}
    assert saved["tasks"] == {}
    assert saved["plan_status"] == "INCOMPLETE"


def test_missing_dependency_is_pending_and_blocks_command_not_permanent_block(tmp_path):
    spec = task(commands=[{"argv": [sys.executable, "-c", "print(1)"], "purpose": "fixture"}], dependencies=["S1.1"])
    with Campaign(tmp_path, "demo") as campaign:
        report = reconcile(campaign, catalog(spec))
        row = report["tasks"][spec["id"]]
        assert row["status"] == "PENDING"
        assert "dependency:S1.1" in row["commands"][0]["waiting_for"]
        assert not row["commands"][0]["runnable"]


def test_verified_child_metrics_are_reconciled_immutably_not_stage_completion(tmp_path):
    child(tmp_path)
    spec = task(child_sources=[{"suffix": "_retrieval", "patterns": ["E1_*"]}])
    with Campaign(tmp_path, "demo") as campaign:
        historical = copy.deepcopy(campaign.state["tasks"])
        report = reconcile(campaign, catalog(spec))
        row = report["tasks"][spec["id"]]
        assert row["status"] == "PENDING" and row["verified_child_tasks"] == 1
        proof = row["evidence"][0]
        assert proof["summaries"][0]["observed"]["failures"] == 1
        assert proof["summaries"][0]["observed"]["observed_fixture_metric"] == .5
        assert campaign.state["tasks"] == historical
        before = (campaign.path / "plan_events.jsonl").read_bytes()
        reconcile(campaign, catalog(spec))
        assert (campaign.path / "plan_events.jsonl").read_bytes() == before
        assert len(list((campaign.path / "reconciled_evidence/S2.3").glob("*.json"))) == 1


def test_child_corrupt_evidence_is_failed_and_never_imported(tmp_path):
    output = child(tmp_path)
    output.write_text('{"n":999}', encoding="utf-8")
    spec = task(child_sources=[{"suffix": "_retrieval", "patterns": ["E1_*"]}])
    with Campaign(tmp_path, "demo") as campaign:
        row = reconcile(campaign, catalog(spec))["tasks"][spec["id"]]
        assert row["status"] == "FAILED" and not row["evidence"]
        assert row["validation_errors"]


def test_live_child_is_running_but_unrelated_future_task_stays_pending(tmp_path):
    spec = task(child_sources=[{"suffix": "_retrieval", "patterns": ["E1_*"], "active_when_locked": True}])
    future = task("S2.6", child_sources=[{"suffix": "_retrieval", "patterns": ["E8_*"]}])
    with Campaign(tmp_path, "demo_retrieval") as running:
        running.state["status"] = "PARTIAL_BLOCKED"  # Stale aggregate label is not a hard blocker.
        running.save()
        with Campaign(tmp_path, "demo") as parent:
            rows = reconcile(parent, catalog(spec, future))["tasks"]
            assert rows["S2.3"]["status"] == "RUNNING"
            assert rows["S2.6"]["status"] == "PENDING"


def test_complete_child_set_required_before_pass(tmp_path):
    child(tmp_path)
    spec = task(child_sources=[{"suffix": "_retrieval", "patterns": ["E1_*"]}],
                required_child_task_ids=["E1_fixture", "E1_missing"])
    with Campaign(tmp_path, "demo") as campaign:
        assert reconcile(campaign, catalog(spec))["tasks"]["S2.3"]["status"] == "PENDING"
        spec["required_child_task_ids"] = ["E1_fixture"]
        assert reconcile(campaign, catalog(spec))["tasks"]["S2.3"]["status"] == "PASS"
        spec["verify_fixed_embedding"] = True
        assert reconcile(campaign, catalog(spec))["tasks"]["S2.3"]["status"] == "FAILED"


def test_old_task_identity_is_unchanged_during_reconciliation(tmp_path):
    spec = task("S1.1", kind="engineering")
    with Campaign(tmp_path, "demo") as campaign:
        campaign.begin("S1.1", {"original": 1}, ["old-command"])
        path = campaign.path / "old.json"
        atomic_json(path, {"actual_historical_result": True})
        campaign.finish("S1.1", "PASS", [path], {"exit_code": 0})
        old = copy.deepcopy(campaign.state["tasks"]["S1.1"])
        assert reconcile(campaign, catalog(spec))["tasks"]["S1.1"]["status"] == "PASS"
        assert campaign.state["tasks"]["S1.1"] == old


@pytest.mark.parametrize("change", ["not_real", "missing_requirement", "wrong_spec", "changed_artifact"])
def test_completion_receipt_requires_real_complete_bound_evidence(tmp_path, change):
    spec = task()
    output = tmp_path / "rows.jsonl"
    output.write_text("offline-test-artifact", encoding="utf-8")
    receipt = {"task_id": spec["id"], "task_spec_hash": signature(spec), "status": "PASS", "real_experiment": True,
               "requirements_met": spec["completion_requirements"], "artifacts": [artifact(output)]}
    # Deliberate contract fixtures, never written under real campaign paths.
    if change == "not_real": receipt["real_experiment"] = False
    elif change == "missing_requirement": receipt["requirements_met"] = []
    elif change == "wrong_spec": receipt["task_spec_hash"] = "wrong"
    elif change == "changed_artifact": output.write_text("changed")
    atomic_json(tmp_path / "runs/demo/completion/S2.3.json", receipt)
    with pytest.raises(ValueError):
        completion_receipt(spec, tmp_path, "demo")


def test_successful_helper_does_not_complete_experiment(tmp_path):
    spec = task(commands=[{"argv": [sys.executable, "-c", "print('offline helper')"], "purpose": "fixture"}])
    with Campaign(tmp_path, "demo") as campaign:
        action = command_views(spec, tmp_path, "demo", {})[0]
        execute_plan_command(campaign, spec, action)
        assert all(row["status"] == "PASS" for row in campaign.state["tasks"].values())
        assert reconcile(campaign, catalog(spec))["tasks"]["S2.3"]["status"] == "PENDING"


def test_engineering_execution_pass_and_current_inputs_bound(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("v = 1")
    spec = task("S1.3", kind="engineering", execution_completes_task=True, input_files=[str(source)],
                commands=[{"argv": [sys.executable, "-c", "print('offline schema test')"], "purpose": "fixture"}])
    with Campaign(tmp_path, "demo") as campaign:
        execute_plan_command(campaign, spec, command_views(spec, tmp_path, "demo", {})[0])
        row = reconcile(campaign, catalog(spec))["tasks"]["S1.3"]
        assert row["status"] == "PASS"
        source.write_text("v = 2")
        assert reconcile(campaign, catalog(spec))["tasks"]["S1.3"]["status"] == "PENDING"
        assert list(campaign.state["tasks"].values())[0]["status"] == "PASS"  # Preserved history.


def test_prerequisite_exit_stays_pending_and_gpu_requires_explicit_handoff(tmp_path):
    spec = task(commands=[{"argv": [sys.executable, "-c", "raise SystemExit(2)"], "purpose": "fixture",
                           "pending_exit_codes": [2]}])
    with Campaign(tmp_path, "demo") as campaign:
        execute_plan_command(campaign, spec, command_views(spec, tmp_path, "demo", {})[0])
        assert reconcile(campaign, catalog(spec))["tasks"]["S2.3"]["status"] == "PENDING"
    spec["commands"][0].update(gpu=True, external_coordination=True)
    guarded = command_views(spec, tmp_path, "demo", {})[0]
    assert len(guarded["waiting_for"]) == 2 and not guarded["runnable"]


def test_status_of_unstarted_campaign_lists_all_tasks_without_creating_state(tmp_path):
    result = subprocess.run([sys.executable, "-m", "ragscope.experiments.status", "--campaign", "not_started"],
                            env={**os.environ, "RAGSCOPE_HOME": str(tmp_path)}, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["task_count"] == 25
    assert not (tmp_path / "runs/not_started/state.json").exists()


def test_native_interpreter_expansion_never_crosses_platform(tmp_path):
    value = expand("{train_python}", tmp_path, "demo")
    assert ("Scripts" in value) == (os.name == "nt")
    assert ("train-linux" in value) == (os.name != "nt")


def test_completion_receipt_allows_project_docs_but_rejects_unrelated_path(tmp_path, monkeypatch):
    runtime, project, unrelated = (tmp_path / name for name in ("runtime", "project", "unrelated"))
    monkeypatch.setattr("ragscope.experiments.task_catalog.PROJECT", project)
    spec = task("S4.6", kind="engineering")
    output = project / "docs/acceptance.json"
    atomic_json(output, {"fixture": True})
    receipt = {"task_id": spec["id"], "task_spec_hash": signature(spec), "status": "PASS",
               "requirements_met": spec["completion_requirements"], "artifacts": [artifact(output)]}
    receipt_path = runtime / "runs/demo/completion/S4.6.json"
    atomic_json(receipt_path, receipt)
    assert completion_receipt(spec, runtime, "demo")["receipt"]["status"] == "PASS"
    output = unrelated / "copied.json"
    atomic_json(output, {"fixture": True})
    receipt["artifacts"] = [artifact(output)]
    atomic_json(receipt_path, receipt)
    with pytest.raises(ValueError):
        completion_receipt(spec, runtime, "demo")


def test_latest_execution_uses_timestamp_not_config_hash_order(tmp_path):
    spec = task()
    with Campaign(tmp_path, "demo") as campaign:
        campaign.state["tasks"] = {
            "PLAN_S2.3_a": {"status": "FAILED", "config": {"plan_task_id": "S2.3"},
                           "started_at": "2026-10-14T02:00:00", "validation": {"waiting_for_prerequisites": True}},
            "PLAN_S2.3_z": {"status": "FAILED", "config": {"plan_task_id": "S2.3"},
                           "started_at": "2026-10-14T01:00:00", "validation": {}}
        }
        assert reconcile(campaign, catalog(spec))["tasks"]["S2.3"]["status"] == "PENDING"


def test_pending_checks_are_bounded_until_required_input_changes(tmp_path):
    prerequisite = tmp_path / "prerequisite.json"
    atomic_json(prerequisite, {"ready": False})
    spec = task(commands=[{"argv": [sys.executable, "-c", "raise SystemExit(2)"], "purpose": "fixture",
                           "requires": [str(prerequisite)], "pending_exit_codes": [2]}])
    with Campaign(tmp_path, "demo") as campaign:
        action = command_views(spec, tmp_path, "demo", {})[0]
        for _ in range(3):
            execute_plan_command(campaign, spec, action)
        with pytest.raises(RuntimeError, match="retry limit"):
            execute_plan_command(campaign, spec, action)
        atomic_json(prerequisite, {"ready": True})
        execute_plan_command(campaign, spec, action)
        assert len(campaign.state["tasks"]) == 2
        assert reconcile(campaign, catalog(spec))["tasks"]["S2.3"]["status"] == "PENDING"


def test_invalid_gpu_lease_never_begins_execution(tmp_path, monkeypatch):
    spec = task(commands=[{"argv": [sys.executable, "-c", "print('not executed')"],
                           "purpose": "fixture", "gpu": True}])
    def reject(_):
        raise ValueError("fixture lease unavailable")
    monkeypatch.setattr("ragscope.training.run.verify_lease", reject)
    with Campaign(tmp_path, "demo") as campaign:
        with pytest.raises(ValueError, match="lease unavailable"):
            execute_plan_command(campaign, spec, command_views(spec, tmp_path, "demo", {}, True)[0])
        assert campaign.state["tasks"] == {}


def test_live_worker_receipt_requires_exact_process_identity(tmp_path, monkeypatch):
    from ragscope.experiments.task_catalog import live_worker, PROJECT
    directory = tmp_path / "runs/demo_retrieval"
    child(tmp_path)
    receipt = {"pid": 999999, "creation_time": 1.5, "command": ["offline-fixture"],
               "status": "RUNNING", "campaign_id": directory.name, "stage": "dense", "device": "cpu"}
    atomic_json(directory / "worker.json", receipt)
    class Process:
        def is_running(self): return True
        def status(self): return "running"
        def create_time(self): return 1.5
        def cmdline(self): return ["offline-fixture"]
        def cwd(self): return str(PROJECT)
    monkeypatch.setattr("psutil.Process", lambda pid: Process())
    assert live_worker(directory)["pid"] == 999999
    spec = task(child_sources=[{"suffix": "_retrieval", "patterns": ["E1_*"], "active_when_locked": True}])
    with Campaign(tmp_path, "demo") as campaign:
        row = reconcile(campaign, catalog(spec))["tasks"]["S2.3"]
        assert row["status"] == "RUNNING" and len(row["active_workers"]) == 1
    receipt["status"] = "SUSPENDED"
    atomic_json(directory / "worker.json", receipt)
    with Campaign(tmp_path, "demo_retrieval"), Campaign(tmp_path, "demo") as campaign:
        row = reconcile(campaign, catalog(spec))["tasks"]["S2.3"]
        assert row["status"] == "PENDING" and row["resource_wait"]
    spec["commands"] = [{"argv": [sys.executable, "-c", "print(1)"], "purpose": "fixture", "lock_campaigns": ["_retrieval"]}]
    assert command_views(spec,tmp_path,"demo",{})[0]["waiting_for"][0].startswith("child_worker_suspended_waiting_resource:")
    receipt["creation_time"] = 1.4  # Reused PID is never trusted.
    atomic_json(directory / "worker.json", receipt)
    assert live_worker(directory) is None


@pytest.mark.parametrize("kind", ["experiment", "delivery"])
def test_historical_exact_task_name_cannot_bypass_formal_completion(tmp_path, kind):
    spec = task(kind=kind, execution_completes_task=True)
    with Campaign(tmp_path, "demo") as campaign:
        campaign.begin(spec["id"], {"interface_only": True}, ["offline-fixture"])
        output = campaign.path / "interface.json"
        atomic_json(output, {"supported": True})
        campaign.finish(spec["id"], "PASS", [output], {"exit_code": 0})
        assert reconcile(campaign, catalog(spec))["tasks"][spec["id"]]["status"] == "PENDING"
        assert campaign.state["tasks"][spec["id"]]["status"] == "PASS"


def dev_matrix_fixture(tmp_path, method="dense", damage=None):
    run_id = f"E1_fixture_{method}_none"
    index = {"embedding": {"device": "cpu", "model_id": "offline_fixture"} if method != "bm25" else None}
    config = {"run_id": run_id, "experiment_type": "retrieval", "method": method,
              "query_ids": [f"q{i}" for i in range(100)], "index_identity": index}
    fingerprint = signature(config)
    rows = [{"query_id": query_id, "record_id": query_id, "config_hash": fingerprint,
             "run_id": run_id, "experiment_type": "retrieval", "split": "dev", "device": "cpu",
             "index_signature": signature(index), "embedding_identity": index["embedding"],
             "real_model": method != "bm25", "error": "retained_fixture_error" if i == 0 else None}
            for i, query_id in enumerate(config["query_ids"])]
    if damage == "missing": rows.pop()
    if damage == "duplicate": rows[-1] = dict(rows[0])
    if damage == "mock_neural": rows[1]["real_model"] = False
    if damage == "wrong_config": rows[1]["config_hash"] = "unrelated"
    if damage == "wrong_index": rows[1]["index_signature"] = "unrelated"
    for row in rows:
        row["record_hash"] = signature(row)
    if damage == "missing_hash": rows[1].pop("record_hash")
    if damage == "bad_hash": rows[1]["record_hash"] = "corrupt"
    path = tmp_path / f"{run_id}.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    atomic_json(path.with_suffix(".config.json"), config)
    return {"source_task_id": run_id, "input_fingerprint": fingerprint, "outputs": [artifact(path)],
            "summaries": [{"observed": {"run_id": run_id, "method": method, "n_total": 100,
                                          "n_failed": 0 if damage == "hidden_failure" else 1}}]}


@pytest.mark.parametrize("method", ["bm25", "dense", "hybrid"])
def test_dev_matrix_accepts_actual_method_and_retains_failures(tmp_path, method):
    audit = verify_dev_matrix_rows([dev_matrix_fixture(tmp_path, method)])
    assert audit[0]["rows"] == 100 and audit[0]["failures"] == 1


@pytest.mark.parametrize("damage", ["missing", "duplicate", "mock_neural", "wrong_config", "wrong_index",
                                     "missing_hash", "bad_hash", "hidden_failure"])
def test_dev_matrix_rejects_incomplete_or_interface_only_rows(tmp_path, damage):
    with pytest.raises(ValueError):
        verify_dev_matrix_rows([dev_matrix_fixture(tmp_path, damage=damage)])
