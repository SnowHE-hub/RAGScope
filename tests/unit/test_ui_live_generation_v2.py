"""Offline application integration tests; isolated budget/records, no HTTP."""
import json
import sys
from types import SimpleNamespace

import pytest
from filelock import FileLock, Timeout

from ragscope.experiments.generation_campaign import GenerationBudget
from ragscope.experiments.ledger import read_records
from ui.live_generation import answer_live, save_live_record


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv('RAGSCOPE_HOME', str(tmp_path))
    run = tmp_path/'runs'/'offline-ui-fixture'
    budget = GenerationBudget(run/'generation_budget.json')
    verifier_calls = []
    def verify(*args):
        verifier_calls.append(args)
    # Only these offline fixtures bypass a real service, never production code.
    monkeypatch.setitem(sys.modules, 'run_generation_campaign', SimpleNamespace(
        verify_owned_service=verify, GuardedClient=lambda client, *args: client))
    monkeypatch.setattr('ragscope.generation.resources.MeasuredClient', lambda client, *args, **kwargs: client)
    monkeypatch.setattr('ui.model_identity.verify_model_package', lambda *args: None)
    class Client:
        real_model = True
        identity = {'model': 'offline-integration-fixture', 'provider': 'test'}
        model = 'offline-integration-fixture'
        calls = 0
        last_metadata = {}
        def chat(self, messages, **kwargs):
            self.calls += 1
            return json.dumps({'answer': '', 'citations': [], 'confidence': 0.0, 'abstain': True})
    retriever = SimpleNamespace(retrieve=lambda question, k: [], identity={'offline-fixture': True})
    return Client(), retriever, {'campaign_run': str(run)}, budget, verifier_calls, tmp_path


def test_application_call_counted_and_saved(runtime):
    client, retriever, execution, budget, checks, _ = runtime
    record = answer_live(client, retriever, {'max_tokens': 100}, 'offline test', 1, None, execution, model_entry={})
    assert client.calls == budget.state()['reserved_calls'] == 1
    assert checks and record['call_ticket']['purpose'] == 'UI'
    assert record['experiment_type'] == 'interactive_application_not_benchmark'
    path = record['base_record_path']
    from pathlib import Path
    rows = read_records(Path(path))
    assert len(rows) == 1 and rows[0]['call_ticket'] == record['call_ticket']
    supplemental = save_live_record(record, execution, supplement=True)
    assert supplemental != path
    assert read_records(Path(path)) == rows  # Rendering supplement never replaces raw output.


def test_application_busy_lock_does_not_call_or_reserve(runtime):
    client, retriever, execution, budget, checks, home = runtime
    with FileLock(str(home/'runs/generation-client.lock'), timeout=0):
        with pytest.raises(Timeout):
            answer_live(client, retriever, {}, 'offline test', 1, None, execution, model_entry={})
    assert not checks and client.calls == budget.state()['reserved_calls'] == 0


def test_application_service_failure_and_mock_never_call(runtime, monkeypatch):
    client, retriever, execution, budget, _, _ = runtime
    def fail(*args):
        raise ValueError('owned service exited')
    monkeypatch.setattr(sys.modules['run_generation_campaign'], 'verify_owned_service', fail)
    with pytest.raises(ValueError, match='exited'):
        answer_live(client, retriever, {}, 'offline test', 1, None, execution, model_entry={})
    client.real_model = False
    with pytest.raises(ValueError, match='Mock'):
        answer_live(client, retriever, {}, 'offline test', 1, None, execution, model_entry={})
    assert client.calls == budget.state()['reserved_calls'] == 0
