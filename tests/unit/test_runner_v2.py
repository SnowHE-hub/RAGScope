import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from filelock import Timeout
from ragscope.experiments.ledger import Campaign, RecordStore, read_records
from ragscope.indexing.identity import signature


def test_process_crash_resume_partial_and_idempotence(tmp_path):
    environment = dict(os.environ, RAGSCOPE_HOME=str(tmp_path))
    command = [sys.executable, "-m", "ragscope.experiments.run", "--profile", "recovery-fixture", "--campaign", "recovery"]
    failed = subprocess.run(command + ["--interrupt-after", "2", "--partial-tail"], env=environment, capture_output=True, timeout=30)
    assert failed.returncode == 99, failed.stderr
    path = tmp_path / "runs/recovery/fixture.jsonl"
    with pytest.raises(ValueError, match="Incomplete"):
        read_records(path)
    assert subprocess.run(command + ["--resume"], env=environment, capture_output=True, timeout=30).returncode == 0
    rows = read_records(path)
    assert len(rows) == 5 and len({row["record_id"] for row in rows}) == 5
    before = path.read_bytes()
    assert subprocess.run(command + ["--resume"], env=environment, capture_output=True, timeout=30).returncode == 0
    assert path.read_bytes() == before
    assert list(path.parent.glob("fixture.jsonl.partial.*"))
    assert [row["output"] for row in rows] == [0, 1, 4, 9, 16]


def test_lock_and_completed_artifact_validation(tmp_path):
    with Campaign(tmp_path, "locked") as campaign:
        with pytest.raises(Timeout):
            with Campaign(tmp_path, "locked"):
                pass
        assert campaign.begin("x", {"a": 1}, ["test"])
        output = tmp_path / "output.json"
        output.write_text("{}")
        campaign.finish("x", "PASS", [output], {"verified": True})
        assert not campaign.begin("x", {"a": 1}, ["test"])
        output.write_text("changed")
        with pytest.raises(ValueError, match="changed"):
            campaign.begin("x", {"a": 1}, ["test"])


def test_records_reject_mock_config_changes_and_corruption(tmp_path):
    store = RecordStore(tmp_path / "real.jsonl", {"v": 1}, real_only=True, max_records=1)
    with pytest.raises(ValueError, match="real model"):
        store.add("a", {"real_model": False})
    assert store.add("a", {"real_model": True, "model_identity": {"sha256": "test-weight"}})
    assert not store.add("a", {})
    with pytest.raises(RuntimeError, match="budget"):
        store.add("b", {})
    with pytest.raises(ValueError, match="config"):
        RecordStore(store.path, {"v": 2})
    text = store.path.read_text().replace('"real_model": true', '"real_model": false')
    store.path.write_text(text)
    with pytest.raises(ValueError, match="Corrupt"):
        read_records(store.path)


def test_input_config_signature_and_dependencies(tmp_path):
    assert signature({"x": 1, "y": 2}) == signature({"y": 2, "x": 1})
    with pytest.raises(ValueError):
        signature({"bad": float("nan")})
    with Campaign(tmp_path, "deps") as campaign:
        with pytest.raises(ValueError, match="dependency"):
            campaign.begin("next", {}, [], ["missing"])
