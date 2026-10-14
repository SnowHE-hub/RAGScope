"""Offline contracts only: fixture clients never enter runtime experiment paths."""
from dataclasses import asdict
from pathlib import Path

import pytest

from ragscope.experiments.generation_campaign import GenerationBudget, freeze_campaign
from ragscope.experiments.ledger import read_records
from ragscope.generation.context import pack_context
from ragscope.indexing.identity import atomic_json, signature, artifact
from ragscope.routing import online_campaign as online
from ragscope.routing.core import FixedRouter, RuleRouter, ROUTES


class FixtureClient:
    real_model = True
    prompt_token_counter = staticmethod(lambda messages: 1)
    last_metadata = {}

    def __init__(self, name):
        self.identity = {'model': name, 'revision': 'offline', 'weight_sha256': 'no-real-weights', 'backend_version': 'fixture'}
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        return '{"answer":"source","citations":["c"],"abstain":false}'


class FixtureRetriever:
    identity = {'kind': 'offline-test-fixture'}

    def __init__(self, bundle, fail_at=None):
        self.bundle, self.fail_at, self.calls = bundle, fail_at, 0

    def retrieve(self, question, scope, route):
        # No labels, gold evidence, original frozen answer or precomputed route result enter retrieval.
        assert isinstance(question, str) and set(scope) == {'kind', 'doc_ids'} and route in ROUTES
        self.calls += 1
        return {'context_bundle': self.bundle, 'error': {'type': 'fixture_failure'} if self.calls == self.fail_at else None}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    campaign, routing, output = tmp_path/'campaign', tmp_path/'routing', tmp_path/'online'
    unit = {'unit_id': 'u', 'doc_id': 'd', 'canonical_start': 0, 'canonical_end': 6, 'text': 'source'}
    chunk = {'chunk_id': 'c', 'doc_id': 'd', 'text': 'source', 'canonical_start': 0, 'canonical_end': 6,
             'unit_spans': [{'unit_id': 'u', 'doc_id': 'd', 'start': 0, 'end': 6}]}
    bundle = asdict(pack_context([chunk]))
    clients = [FixtureClient(name) for name in ('qwen_2b', 'qwen_4b', 'qwen_9b')]
    queries = [{'query_id': f'q{i}', 'question': f'Question {i}?', 'scope': {'kind': 'document', 'doc_ids': ['d']},
                'split': 'dev', 'answerability': 'answerable', 'answer_refs': ['source'],
                'evidence_reference_sets': [['u']]} for i in range(30)]
    inputs = {q['query_id']: {'query': q, 'context_bundle': bundle} for q in queries}
    manifest = freeze_campaign(campaign, {c.identity['model']: c.identity for c in clients},
        [{'experiment_id': 'ROUTE_DEV', 'method': 'hybrid', 'model_ids': ['qwen_2b'], 'input_ids': list(inputs)}], inputs)
    protocol = {'comparison': {'manifest_hash': manifest['manifest_hash'], 'model_id': 'qwen_2b',
        'model_identity': clients[0].identity, 'generation': manifest['generation_config']}, 'online_dev_n': 30,
        'populations': {'dev': {'query_ids': list(inputs), 'visible_queries': queries},
                       'test': {'query_ids': ['testq'], 'visible_queries': []}}, 'seed': 42}
    protocol['protocol_hash'] = signature(protocol)
    atomic_json(routing/'ROUTING_MANIFEST.json', protocol)
    for name in online.REQUIRED_ROUTERS:
        route = name.removeprefix('fixed_') if name.startswith('fixed_') else 'hybrid' if name == 'rules' else 'hybrid_rerank'
        ident = FixedRouter(route).identity if name.startswith('fixed_') else RuleRouter.identity if name == 'rules' else {'offline': name}
        decision = {'name': name, 'protocol_hash': protocol['protocol_hash'], 'router_identity': ident,
            'decisions': {split: [{'query_id': qid, 'decision': {'route': route, 'fallback': False, 'elapsed_s': 0}}
                                 for qid in pop['query_ids']] for split, pop in protocol['populations'].items()}}
        decision['decision_hash'] = signature(decision)
        atomic_json(routing/f'DECISIONS_{name}.json', decision)
    records = [{'query_id': q['query_id'], 'path': route, 'answer_f1': 1, 'answerability': 'answerable',
        'total_latency_s': .1*(i+1), 'latency_kind': 'component_replay_estimate',
        'wrong_abstain': False, 'error': None, 'evidence_coverage': 1} for q in queries for i, route in enumerate(ROUTES)]
    def dev_only(campaign, protocol, splits):
        assert splits == ('dev',), 'Online selection must never request test scores'
        return records, []
    monkeypatch.setattr(online, 'load_route_records', dev_only)
    frozen = online.freeze_online(campaign, routing, output)
    assert frozen['selected'] == {'fixed': 'fixed_bm25', 'nonfixed': 'rules'}
    return {'campaign': campaign, 'routing': routing, 'output': output, 'bundle': bundle, 'clients': clients,
            'units': {'u': unit}, 'frozen': frozen, 'budget': GenerationBudget(tmp_path/'budget.json', limit=200, core_reserved=0)}


def run(s, **kwargs):
    return online.run_online(s['campaign'], s['routing'], s['output'], s['clients'][0], s['budget'],
        kwargs.pop('retriever', FixtureRetriever(s['bundle'])), {'fixed_bm25': FixedRouter('bm25'), 'rules': RuleRouter()},
        common_clients=s['clients'], shared_context=lambda source, clients: source, evidence_units=s['units'], **kwargs)


def test_dev_freeze_stable_and_all_policies_required(setup):
    s = setup
    assert online.freeze_online(s['campaign'], s['routing'], s['output']) == s['frozen']
    (s['routing']/'DECISIONS_lstm.json').unlink()
    with pytest.raises(FileNotFoundError):
        online.freeze_online(s['campaign'], s['routing'], s['output'])


def test_selection_tolerance_prefers_cost_and_rejects_no_answerable():
    rows = {'a': {'answer_f1': .9, 'p50_latency_s': 3}, 'b': {'answer_f1': .895, 'p50_latency_s': 1}}
    assert online.select_dev(rows, ['a', 'b']) == 'b'
    rows['b']['answer_f1'] = .88
    assert online.select_dev(rows, ['a', 'b']) == 'a'
    with pytest.raises(ValueError):
        online.select_dev({'a': {'answer_f1': None}}, ['a'])


def test_live_calls_counted_with_resume_and_separate_warmups(setup):
    s = setup
    partial = run(s, max_tasks=2)
    assert partial['status'] == 'PENDING' and s['budget'].state()['reserved_calls'] == 8
    assert not partial['deployment']['deploy']
    result = run(s)
    assert result['status'] == 'COMPLETE' and result['warmup_records'] == 12
    assert s['budget'].state()['reserved_calls'] == 72
    rows = read_records(s['output']/'records.jsonl')
    assert len(rows) == 60 and all(row['real_model'] and row['generation_attempted'] for row in rows)
    assert all(group['summary']['latency_kind'] == 'actual_online' for group in result['groups'].values())
    assert run(s)['warmup_records'] == 12 and s['clients'][0].calls == 72


def test_failed_retrieval_kept_without_claiming_real_generation(setup):
    s = setup
    result = run(s, retriever=FixtureRetriever(s['bundle'], fail_at=7))
    rows = read_records(s['output']/'records.jsonl')
    failed = [row for row in rows if row['error']]
    assert result['status'] == 'COMPLETE' and len(rows) == 60 and len(failed) == 1
    assert failed[0]['answer_f1'] == 0 and not failed[0]['real_model'] and not failed[0]['generation_attempted']
    assert failed[0]['call_ticket'] is None and s['budget'].state()['reserved_calls'] == 65
    assert not result['deployment']['deploy'] and not result['deployment']['actual_request_minimum_met']
    assert sum(group['n_pre_generation_failures'] for group in result['groups'].values()) == 1


def test_counter_identity_and_mock_guard_before_any_call(setup):
    s = setup
    s['clients'][2].identity = {**s['clients'][2].identity, 'revision': 'changed'}
    with pytest.raises(ValueError, match='three frozen'):
        run(s)
    assert s['clients'][0].calls == 0
    s['clients'][0].real_model = False
    with pytest.raises(ValueError, match='smallest main'):
        run(s)


def test_frozen_decision_tamper_rejected_before_calls(setup):
    s = setup
    path = s['routing']/'DECISIONS_rules.json'
    path.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='provenance'):
        run(s)
    assert s['clients'][0].calls == 0


def test_selected_learned_router_cannot_use_fixture_training_labels(setup):
    s = setup
    path = s['routing']/'DECISIONS_mlp.json'
    saved = online.read_json(path)
    saved.pop('decision_hash')
    for item in saved['decisions']['dev']:
        item['decision']['route'] = 'bm25'  # Makes MLP the fastest eligible nonfixed candidate.
    saved['decision_hash'] = signature(saved)
    atomic_json(path, saved)
    weights = s['routing']/'mlp/weights.npz'
    weights.parent.mkdir()
    weights.write_bytes(b'offline test only')
    state = {'report': {'data_kind': 'offline_fixture'}}
    atomic_json(weights.parent/'manifest.json', {'identity': state, 'signature': signature(state), 'artifacts': [artifact(weights)]})
    with pytest.raises(ValueError, match='real train/dev'):
        online.freeze_online(s['campaign'], s['routing'], s['output'])


def test_budget_failure_circuit_break_retains_pending(setup):
    s = setup
    s['budget'] = GenerationBudget(s['output']/'smallbudget.json', limit=6, core_reserved=0)
    result = run(s)
    assert result['status'] == 'PENDING' and result['stop_reason'] is not None
    assert s['clients'][0].calls == 6
    assert sum(group['n_pending'] for group in result['groups'].values()) == 59


def test_remote_gpu_service_guard_allows_cpu_only_client(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts'))
    import psutil
    from run_generation_campaign import verify_owned_service
    from ragscope.generation.llm_client import GenerationError
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    lease, receipt = tmp_path/'lease.json', tmp_path/'service.json'
    import time
    import sys
    atomic_json(lease, {'status': 'ACQUIRED', 'owner': 'ragscope', 'safety_verified': True, 'gpu_device': 0,
                        'host_platform': sys.platform, 'host_boot_time_unix': psutil.boot_time(), 'validated_at': time.time()})
    identity = {'base_url': 'http://127.0.0.1:11435', 'backend_version': 'fixture',
        'backend_runtime_config': {'prompt_ram_cache_mib': 0, 'prompt_cache_policy': 'backend_history_disabled'}}
    service = {'status': 'READY', 'base_url': identity['base_url'], 'backend_version': 'fixture',
        'pid': 123, 'creation_time': 100, 'binary': str(tmp_path/'ollama.exe'),
        'environment': {'OLLAMA_MAX_LOADED_MODELS': '1', 'OLLAMA_NUM_PARALLEL': '1',
                        'OLLAMA_NO_CLOUD': '1', 'LLAMA_ARG_CACHE_RAM': '0'}}
    atomic_json(receipt, service)
    class Process:
        def create_time(self): return 100
        def exe(self): return service['binary']
    monkeypatch.setattr(psutil, 'Process', lambda pid: Process())
    assert verify_owned_service(receipt, lease, identity)['pid'] == 123
    atomic_json(lease, {'status': 'RELEASED', 'owner': 'ragscope', 'safety_verified': True, 'gpu_device': 0})
    with pytest.raises(GenerationError, match='lease'):
        verify_owned_service(receipt, lease, identity)
