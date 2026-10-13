"""Offline process fixtures verify the handoff failure boundary, without models."""
import importlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone

import pytest


class OwnedServiceFixture:
    pid = 123

    def __init__(self):
        self.exited = False
        self.terminations = 0

    def poll(self):
        return 0 if self.exited else None

    def create_time(self):
        return 42.0

    def cmdline(self):
        return ["fixture_cpu_service"]

    def terminate(self):
        self.terminations += 1
        self.exited = True

    def wait(self, timeout):
        assert self.exited and timeout == 15

    def memory_info(self):
        return SimpleNamespace(rss=100, peak_wset=150)


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("scripts.resume_structured_after_low_ram")


def test_before_resume_failure_cleans_only_exact_owned_service(tmp_path, monkeypatch):
    m = module(monkeypatch)
    process = OwnedServiceFixture()
    receipt = {}
    path = tmp_path / "service.json"
    m.handle_resume_failure(receipt, {"command": process.cmdline()}, path, process,
        process, 42.0, False, RuntimeError("fixture health failed"))
    assert process.terminations == 1
    assert receipt["status"] == "FAILED_BEFORE_WORKER_RESUME"
    assert receipt["worker_resumed"] is False and receipt["failed_owned_service_exited"] is True
    assert json.loads(path.read_text())["status"] == "EXITED_FAILED_LOW_RAM_RESUME"


def test_after_resume_receipt_write_failure_preserves_inflight_service(tmp_path, monkeypatch):
    m = module(monkeypatch)
    process = OwnedServiceFixture()
    receipt = {}
    path = tmp_path / "service.json"
    m.handle_resume_failure(receipt, {"command": process.cmdline()}, path, process,
        process, 42.0, True, OSError("fixture worker receipt write failed"))
    assert process.terminations == 0 and not process.exited
    assert receipt["status"] == "FAILED_AFTER_WORKER_RESUMED"
    assert receipt["worker_resumed"] is True
    assert receipt["owned_service_retained_after_worker_resume"] is True
    saved = json.loads(path.read_text())
    assert saved["status"] == "RUNNING" and saved["worker_resumed"] is True


def test_supervised_load_low_ram_stops_exact_service_before_handoff(tmp_path, monkeypatch):
    m = module(monkeypatch)
    process = OwnedServiceFixture()
    monitor = m.SupervisedLoadMonitor(process, 42.0, process.cmdline(), tmp_path / "ram.jsonl",
        read_available=lambda: 1024**3 - 1)
    monitor.start()
    monitor.thread.join(timeout=2)
    result = monitor.close()
    assert process.terminations == 1 and result["joined_before_worker_resume"]
    assert result["failure"]["kind"] == "host_ram_below_1gib_during_load"
    assert result["max_observed_process_peak_wset_bytes"] == 150


def test_monitor_join_precedes_handoff_and_cannot_stop_service_after(tmp_path, monkeypatch):
    m = module(monkeypatch)
    process = OwnedServiceFixture()
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()

    def read_available():
        entered.set()
        assert release.wait(timeout=2)
        return 2 * 1024**3

    monitor = m.SupervisedLoadMonitor(process, 42.0, process.cmdline(), tmp_path / "ram.jsonl",
        read_available=read_available)
    monitor.start()
    assert entered.wait(timeout=2)
    result = {}

    def close_then_handoff():
        result.update(monitor.close())
        closed.set()  # Represents the resume boundary, only after successful join.

    closer = threading.Thread(target=close_then_handoff)
    closer.start()
    assert not closed.wait(timeout=.05)
    release.set()
    closer.join(timeout=2)
    assert closed.is_set() and result["joined_before_worker_resume"]
    monitor.read_available = lambda: 0
    assert not monitor.thread.is_alive() and process.terminations == 0


def test_fallback_requires_persistent_range_and_original_path_takes_priority(monkeypatch):
    m = module(monkeypatch)
    now = datetime.now(timezone.utc)
    rows = [{"at_utc": (now - timedelta(seconds=ago)).isoformat(),
        "action": "guard_observation_only_no_model_launch", "available_bytes": int(2.7 * 1024**3)}
        for ago in (40, 1)]
    assert m.choose_guard_path(3 * 1024**3, True, []) == "original_3gib"
    assert m.choose_guard_path(int(2.7 * 1024**3), True, rows) == "once_only_supervised_2_5_to_3gib"
    with pytest.raises(RuntimeError):
        m.choose_guard_path(int(2.7 * 1024**3), True, rows[-1:])
