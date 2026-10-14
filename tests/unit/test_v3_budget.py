import json
import pytest
from filelock import FileLock, Timeout
from ragscope.v3.budget import Budget, STAGES


def setup_budget(tmp_path):
    initial = {'event': 'INITIALIZED', 'request_limit': 6000, 'output_token_limit': 4000000,
               'protected_final_requests': 1800, 'stage_request_limits': STAGES}
    (tmp_path / 'BUDGET.jsonl').write_text(json.dumps(initial) + '\n', encoding='utf-8')
    return Budget(tmp_path)


def reserve(budget, rid, stage='S0_S1_evidence_metrics', tokens=512):
    return budget.reserve(request_id=rid, stage=stage, max_output_tokens=tokens,
                          purpose='functional_test_fixture_only', model_identity={'model': 'fixture'})


def test_failed_and_unknown_requests_are_not_refunded(tmp_path):
    budget = setup_budget(tmp_path)
    reserve(budget, 'warmup')
    budget.finalize('warmup', output_tokens=None, outcome='transport_failure_unknown_output')
    state = budget.snapshot()
    assert state['requests_reserved'] == 1
    assert state['output_tokens_charged_or_reserved'] == 512
    with pytest.raises(RuntimeError, match='Already reserved'):
        reserve(Budget(tmp_path), 'warmup')


def test_token_exhaustion_and_backend_overrun_are_recorded(tmp_path):
    budget = setup_budget(tmp_path)
    reserve(budget, 'first', tokens=4000000)
    with pytest.raises(RuntimeError, match='token budget'):
        reserve(budget, 'second')
    with pytest.raises(RuntimeError, match='exceeded'):
        budget.finalize('first', output_tokens=4000001, outcome='backend_violation')
    assert budget.snapshot()['output_tokens_charged_or_reserved'] == 4000001
    with pytest.raises(RuntimeError, match='violation'):
        reserve(budget, 'after-violation')


def test_stage_cap_does_not_use_final_reserve(tmp_path):
    budget = setup_budget(tmp_path)
    for i in range(800):
        budget._append({'event': 'RESERVED', 'request_id': str(i), 'stage': 'S0_S1_evidence_metrics',
                        'max_output_tokens': 1, 'purpose': 'fixture', 'model_identity': {'model': 'fixture'}})
    with pytest.raises(RuntimeError, match='Request budget'):
        reserve(budget, 'one-too-many')
    reserve(budget, 'final', stage='S6_final_validation')
    assert budget.snapshot()['stage_requests']['S6_final_validation'] == 1


def test_lock_partial_tail_and_hash_tamper_fail_closed(tmp_path):
    budget = setup_budget(tmp_path)
    with FileLock(str(budget.path.with_suffix('.lock')), timeout=0):
        with pytest.raises(Timeout):
            reserve(budget, 'contended')
    with budget.path.open('ab') as handle:
        handle.write(b'{"event":')
    before = budget.path.read_bytes()
    with pytest.raises(ValueError, match='Partial'):
        budget.snapshot()
    assert budget.path.read_bytes() == before
