"""UI fixtures test guards only; no fixture is a model-quality result."""
import json

import pytest

from ragscope.indexing.identity import artifact, atomic_json, signature
from ragscope.routing.campaign import freeze_decisions
from ragscope.routing.core import RuleRouter
from ui.routing_options import choose_route, deployable_default, visible_request


@pytest.fixture
def resources(tmp_path):
    docs = tmp_path/'documents.jsonl'
    docs.write_text(json.dumps({'doc_id': 'paper', 'metadata': {
        'reference_n_tokens': 512, 'answer': 'never visible', 'gold_doc_id': 'hidden'}})+'\n', encoding='utf-8')
    populations = {}
    for split in ('train', 'dev', 'test'):
        q = {'query_id': split+'-q', 'split': split, 'question': 'fixture',
             'scope': {'kind': 'document', 'doc_ids': ['paper']}, 'metadata': {}}
        populations[split] = {'query_ids': [q['query_id']], 'visible_queries': [q]}
    protocol = {'populations': populations}
    protocol['protocol_hash'] = signature(protocol)
    protocol_path = tmp_path/'ROUTING_MANIFEST.json'
    atomic_json(protocol_path, protocol)
    freeze_decisions(RuleRouter(), protocol, tmp_path, 'rules')
    policy = {'id': 'experimental-rules', 'kind': 'rules', 'ready': True, 'index_id': 'r512',
        'protocol': artifact(protocol_path), 'decisions': artifact(tmp_path/'DECISIONS_rules.json'),
        'documents': artifact(docs), 'expansion': 'neighbor'}
    index = {'id': 'r512', 'expansion': 'neighbor', 'embedding': {'ready': True}, 'reranker': {'ready': True}}
    identity = {'corpus_sha256': artifact(docs)['sha256'], 'embedding': {'fixture': True}}
    return policy, index, identity, protocol


def test_canonical_visible_metadata_ignores_labels(resources):
    policy, _, identity, _ = resources
    request = visible_request('question', 'paper', policy['documents'], identity['corpus_sha256'])
    assert request.metadata == {'document_length_tokens': 512, 'document_type': 'paper', 'language': 'en'}
    assert request.scope == {'kind': 'document', 'doc_ids': ['paper']}
    assert visible_request('question', None, policy['documents'], identity['corpus_sha256']).metadata == {}
    with pytest.raises(ValueError, match='不在'):
        visible_request('question', 'unrelated', policy['documents'], identity['corpus_sha256'])
    with pytest.raises(ValueError, match='不一致'):
        visible_request('question', 'paper', policy['documents'], 'different-corpus')


def test_route_overrides_only_method_and_measures_load(resources):
    policy, index, identity, _ = resources
    selected, trace = choose_route(policy, 'What is "exact_token"?', 'paper', index, identity)
    assert selected == {**index, 'method': 'bm25'}
    assert 'method' not in index
    assert trace['decision']['reason'] == 'explicit_literal_or_identifier'
    assert trace['visible_request']['question'] == 'What is "exact_token"?'
    assert all(value >= 0 for value in trace['timings'].values())


@pytest.mark.parametrize('damage', ['unready', 'different_index', 'different_context', 'missing_path', 'tampered_decisions'])
def test_invalid_binding_never_runs_a_router(resources, damage):
    policy, index, identity, _ = resources
    if damage == 'unready':
        policy['ready'] = False
    elif damage == 'different_index':
        policy['index_id'] = 'another'
    elif damage == 'different_context':
        policy['expansion'] = 'none'
    elif damage == 'missing_path':
        index['reranker']['ready'] = False
    else:
        with open(policy['decisions']['path'], 'a', encoding='utf-8') as handle:
            handle.write(' ')
    with pytest.raises(ValueError):
        choose_route(policy, 'Compare systems', 'paper', index, identity)


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    """Synthetic RecordStore fixtures exercise guards, never model performance."""
    from copy import deepcopy
    from pathlib import Path
    import random
    from ragscope.experiments.generation_campaign import freeze_campaign, load_campaign
    from ragscope.experiments.ledger import RecordStore, append_jsonl
    from ragscope.generation.context import pack_context
    from ragscope.routing.campaign import REQUIRED_ROUTERS, request_query
    from ragscope.routing.core import ROUTES
    from ui.routing_options import RESIDENT_PROFILE

    monkeypatch.setattr('ragscope.experiments.runtime.runtime_home', lambda: tmp_path)
    campaign, routing, output = tmp_path/'campaign', tmp_path/'routing', tmp_path/'online'
    models = {name: {'model': name, 'revision': 'offline-fixture', 'weight_sha256': 'offline-fixture',
                     'backend_version': 'offline-fixture'} for name in ('qwen_2b', 'qwen_4b', 'qwen_9b')}
    identity = models['qwen_2b']
    bundle = pack_context([]).to_dict()
    queries = [{'query_id': f'dev-{i:02}', 'split': 'dev', 'question': f'Compare fixture {i}',
                'scope': {'kind': 'document', 'doc_ids': ['paper']}, 'answerability': 'answerable',
                'answer_refs': ['source'], 'evidence_reference_sets': []} for i in range(60)]
    inputs = {f'{route}:{q["query_id"]}': {'query': q, 'context_bundle': bundle, 'retrieval_timings': {'total_ms': 0}}
              for route in ROUTES for q in queries}
    groups = [{'experiment_id': 'ROUTE_DEV', 'method': route, 'model_ids': ['qwen_2b'],
               'input_ids': [f'{route}:{q["query_id"]}' for q in queries]} for route in ROUTES]
    manifest = freeze_campaign(campaign, models, groups, inputs, provenance={'spec': {'campaign': 'offline-fixture'}})
    manifest, inputs = load_campaign(campaign)
    comparison = {'manifest_hash': manifest['manifest_hash'], 'model_id': 'qwen_2b',
                  'model_identity': identity, 'generation': manifest['generation_config']}
    protocol = {'comparison': comparison, 'comparison_config_hash': signature(comparison),
                'populations': {'dev': {'query_ids': [q['query_id'] for q in queries],
                                       'visible_queries': [request_query(inputs[f'bm25:{q["query_id"]}']) for q in queries]},
                               'test': {'query_ids': ['test-fixture'], 'visible_queries': [{'query_id': 'test-fixture'}]}},
                'seed': 42, 'online_dev_n': 30, 'source': artifact(campaign/'EXPERIMENT_MANIFEST.json')}
    protocol['protocol_hash'] = signature(protocol)
    atomic_json(routing/'ROUTING_MANIFEST.json', protocol)
    decisions = {}
    for name in REQUIRED_ROUTERS:
        route = name.removeprefix('fixed_') if name.startswith('fixed_') else 'hybrid_rerank'
        saved = {'name': name, 'protocol_hash': protocol['protocol_hash'], 'router_identity': {'fixture_policy': name},
                 'decisions': {split: [{'query_id': qid, 'decision': {'route': route, 'fallback': False, 'elapsed_s': 0}}
                                      for qid in population['query_ids']]
                               for split, population in protocol['populations'].items()}}
        saved['decision_hash'] = signature(saved)
        atomic_json(routing/f'DECISIONS_{name}.json', saved)
        decisions[name] = saved

    ledger = tmp_path/'runs/offline-fixture/generation_budget.calls.jsonl'
    count = 0
    def ticket(purpose, qid):
        nonlocal count
        count += 1
        value = {'call_number': count, 'purpose': purpose, 'query_id': qid,
                 'model_identity': identity, 'reserved_at_unix': 1.0+count}
        append_jsonl(ledger, value)
        return value
    dev_path = campaign/'records/ROUTE_DEV_qwen_2b.jsonl'
    dev_store = RecordStore(dev_path, {'manifest_hash': manifest['manifest_hash'], 'model_id': 'qwen_2b', 'experiment_id': 'ROUTE_DEV'})
    for task in manifest['tasks']:
        source = inputs[task['input_id']]
        result = {**task, 'question': source['query']['question'], 'split': 'dev', 'answer': 'source', 'abstain': False,
                  'real_model': True, 'model_identity': identity, 'error': None, 'context_bundle': bundle,
                  'scores': {'token_f1': 1, 'predicted_abstain': False, 'wrong_abstention': False,
                             'context_evidence': {'continuous_coverage': None}},
                  'timings': {'total_ms': 100*(ROUTES.index(task['method'])+1)},
                  'call_ticket': ticket('ROUTE_DEV', task['query_id'])}
        dev_store.add(task['task_id'], result)

    index_data = tmp_path/'chunks.json'
    atomic_json(index_data, {'fixture': True})
    embedding, reranker = {'device': 'cpu', 'revision': 'fixture-embedding'}, {'device': 'cpu', 'revision': 'fixture-reranker', 'batch_size': 2}
    index_identity = {'corpus_sha256': 'fixture-corpus', 'embedding': embedding}
    index = {'identity': index_identity, 'signature': signature(index_identity), 'artifacts': [artifact(index_data)]}
    index_path = tmp_path/'index_manifest.json'
    atomic_json(index_path, index)
    retrieval = {'index_signature': index['signature'], 'expansion': 'neighbor', 'reference_tokens': 2048,
                 'candidate_depth': 50, 'final_k': 10, 'no_gold_inputs': True,
                 'retrieval_config': {'fixture': True}, 'retrieval_config_hash': signature({'fixture': True}),
                 'artifacts': [artifact(index_path)]}
    ids = sorted(q['query_id'] for q in queries)
    random.Random(42).shuffle(ids)
    frozen = {'selected': {'fixed': 'fixed_bm25', 'nonfixed': 'rules'}, 'protocol_hash': protocol['protocol_hash'],
              'campaign_manifest_hash': manifest['manifest_hash'], 'model_id': 'qwen_2b', 'model_identity': identity,
              'generation_config': manifest['generation_config'], 'query_ids': ids[:30], 'n_per_policy': 30,
              'context_policy': '2048_reference_retrieval_then_common_all_three_qwen_chat_budget',
              'router_identities': {n: decisions[n]['router_identity'] for n in ('fixed_bm25', 'rules')},
              'sources': [artifact(routing/'ROUTING_MANIFEST.json'),
                          *[artifact(routing/f'DECISIONS_{n}.json') for n in REQUIRED_ROUTERS],
                          artifact(dev_path), artifact(dev_path.with_suffix('.config.json'))],
              'implementation': artifact(Path('ragscope/routing/online_campaign.py').resolve()),
              'test_scores_used': False, 'seed': 42}
    frozen['online_hash'] = signature(frozen)
    atomic_json(output/'ONLINE_MANIFEST.json', frozen)
    config = {'online_hash': frozen['online_hash'], 'retrieval_identity': retrieval, 'model_identity': identity,
              'context_models': list(models.values())}
    stores = {False: RecordStore(output/'records.jsonl', config),
              True: RecordStore(output/'warmups.jsonl', {**config, 'purpose': 'warmup'})}
    qmap = {q['query_id']: q for q in queries}
    for warmup, store in stores.items():
        for name in ('fixed_bm25', 'rules'):
            for qid in ids[:3 if warmup else 30]:
                result = {'question': qmap[qid]['question'], 'answer': 'source', 'abstain': False,
                          'real_model': True, 'model_identity': identity, 'error': None}
                route = 'bm25' if name == 'fixed_bm25' else 'hybrid_rerank'
                live = {'retrieval_identity': retrieval, 'index_signature': index['signature'],
                        'route': route, 'question': qmap[qid]['question'],
                        'embedding_identity': embedding if name == 'rules' else None,
                        'reranker_identity': reranker if name == 'rules' else None}
                purpose = 'router_online_warmup' if warmup else 'router_online'
                row = {'policy': name, 'query_id': qid, 'split': 'dev', 'purpose': purpose, 'session': 'session1',
                       'selected_route': route, 'router_decision': {'route': route},
                       'generation_attempted': True, 'call_reserved': True, 'real_model': True,
                       'model_identity': identity, 'call_ticket': ticket(purpose, qid), 'error': None,
                       'latency_kind': 'actual_online', 'latency_s': 1.0 if name == 'fixed_bm25' else .8,
                       'model_residency': 'same_2b_service_resident_after_separate_warmups',
                       'generation_record': result, 'retrieval_record': live, 'evidence_coverage': None}
                store.add(f'session1:{name}:{qid}' if warmup else f'{name}:{qid}', row)
    summary = {'status': 'COMPLETE', 'test_used': False, 'online_hash': frozen['online_hash'],
               'records': artifact(output/'records.jsonl'), 'deployment': {'deploy': True},
               'groups': {'deliberately_untrusted_summary': {'n_generation_attempted': 999}}}
    atomic_json(output/'SUMMARY.json', summary)
    policy = {'id': 'fixture-rules', 'kind': 'rules', 'expansion': 'neighbor',
              'protocol': artifact(routing/'ROUTING_MANIFEST.json'), 'decisions': artifact(routing/'DECISIONS_rules.json'),
              'deployment_evidence': {'summary': artifact(output/'SUMMARY.json'), 'manifest': artifact(output/'ONLINE_MANIFEST.json')}}
    runtime = {'retrieval_identity': retrieval, 'embedding_identity': embedding, 'reranker_identity': reranker,
               'context_models': list(models.values()), 'generation_config': manifest['generation_config'],
               'context_policy': frozen['context_policy'], 'execution_profile': RESIDENT_PROFILE}
    return {'policy': policy, 'kwargs': deepcopy({'model_identity': identity, 'index_manifest': index, 'runtime_config': runtime}),
            'output': output, 'dev_path': dev_path, 'ledger': ledger, 'routing': routing, 'frozen': frozen}


def check(deployment):
    from ui.routing_options import deployment_readiness
    return deployment_readiness(deployment['policy'], **deployment['kwargs'])


def rewrite_rows(path, modify):
    from ragscope.experiments.ledger import read_records
    rows = read_records(path)
    modify(rows)
    for row in rows:
        row.pop('record_hash', None)
        row['record_hash'] = signature(row)
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')


def refresh_online_summary(deployment):
    path = deployment['output']/'SUMMARY.json'
    body = json.loads(path.read_text(encoding='utf-8'))
    body['records'] = artifact(deployment['output']/'records.jsonl')
    atomic_json(path, body)
    deployment['policy']['deployment_evidence']['summary'] = artifact(path)


def refresh_dev_source(deployment):
    """Rehash every container to test semantic rejection, not just SHA checks."""
    path = deployment['output']/'ONLINE_MANIFEST.json'
    frozen = json.loads(path.read_text(encoding='utf-8'))
    frozen['sources'] = [artifact(item['path']) for item in frozen['sources']]
    frozen.pop('online_hash')
    frozen['online_hash'] = signature(frozen)
    atomic_json(path, frozen)
    deployment['policy']['deployment_evidence']['manifest'] = artifact(path)
    for name in ('records', 'warmups'):
        store_path = deployment['output']/f'{name}.jsonl'
        config_path = store_path.with_suffix('.config.json')
        config = json.loads(config_path.read_text(encoding='utf-8'))
        config['online_hash'] = frozen['online_hash']
        atomic_json(config_path, config)
        rewrite_rows(store_path, lambda rows: [row.update(config_hash=signature(config)) for row in rows])
    summary_path = deployment['output']/'SUMMARY.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    summary['online_hash'] = frozen['online_hash']
    atomic_json(summary_path, summary)
    refresh_online_summary(deployment)


def test_default_recomputes_full_dev_and_paired_latency_from_valid_stores(deployment):
    result = check(deployment)
    assert result['deploy'], result
    assert result['dev']['rules']['n'] == 60
    assert result['online']['rules']['n'] == 30
    assert result['online']['rules']['p50_latency_s'] == .8
    assert not deployable_default(deployment['policy'])  # Current identities are mandatory.


@pytest.mark.parametrize('count', [1, 30, 59, 61])
def test_production_dev_gate_cannot_accept_fixture_sized_populations(tmp_path, count):
    from ui.routing_options import _dev_evidence
    with pytest.raises(ValueError, match='full_60_dev_population_required'):
        _dev_evidence(tmp_path, {'tasks': []}, {}, {'populations': {'dev': {'query_ids': list(range(count))}}},
                      {'model_id': 'qwen_2b'}, lambda *args: None)


@pytest.mark.parametrize('damage', ['cold', 'model', 'index', 'embedding', 'reranker', 'context_models', 'context_policy', 'generation'])
def test_default_binds_exact_current_runtime(deployment, damage):
    kwargs = deployment['kwargs']; runtime = kwargs['runtime_config']
    if damage == 'cold': runtime['execution_profile'] = 'cold_application_loads'
    elif damage == 'model': kwargs['model_identity']['revision'] = 'another'
    elif damage == 'index': kwargs['index_manifest']['signature'] = 'changed'
    elif damage in ('embedding', 'reranker'): runtime[f'{damage}_identity']['revision'] = 'another'
    elif damage == 'context_models': runtime['context_models'][2]['revision'] = 'another'
    elif damage == 'context_policy': runtime['context_policy'] = 'single_model_only'
    else: runtime['generation_config']['max_tokens'] = 100
    result = check(deployment)
    assert not result['deploy'], result
    if damage == 'cold': assert result['reason'] == 'execution_profile_mismatch'


@pytest.mark.parametrize('damage', ['plain_text', 'json_scalar', 'missing_hash', 'missing_row', 'duplicate', 'no_attempt', 'fake_ticket', 'duplicate_ticket', 'mock', 'slow', 'wrong_query', 'wrong_route'])
def test_online_summary_cannot_hide_invalid_or_inadequate_rows(deployment, damage):
    path = deployment['output']/'records.jsonl'
    if damage == 'plain_text': path.write_text('offline-fixture-only\n', encoding='utf-8')
    elif damage == 'json_scalar': path.write_text('"offline-fixture-only"\n', encoding='utf-8')
    elif damage == 'missing_hash':
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0].pop('record_hash')
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
    else:
        def change(rows):
            if damage == 'missing_row': rows.pop()
            elif damage == 'duplicate': rows.append(dict(rows[0]))
            elif damage == 'no_attempt': rows[0]['generation_attempted'] = False
            elif damage == 'fake_ticket': rows[0]['call_ticket']['call_number'] = 9999
            elif damage == 'duplicate_ticket': rows[1]['call_ticket'] = rows[0]['call_ticket']
            elif damage == 'mock': rows[0]['generation_record']['real_model'] = False
            elif damage == 'wrong_query': rows[0]['generation_record']['question'] = 'different'
            elif damage == 'wrong_route': rows[0]['selected_route'] = 'hybrid'
            else:
                for row in rows:
                    if row['policy'] == 'rules': row['latency_s'] = .95
        rewrite_rows(path, change)
    refresh_online_summary(deployment)
    assert not check(deployment)['deploy']


def test_full_dev_quality_not_online_subset_or_saved_f1(deployment):
    omitted = next(f'dev-{i:02}' for i in range(60) if f'dev-{i:02}' not in deployment['frozen']['query_ids'])
    def change(rows):
        for row in rows:
            if row['query_id'] == omitted and row['method'] == 'hybrid_rerank':
                row['answer'] = 'wrong'  # Saved F1 and online answers still claim perfect quality.
    rewrite_rows(deployment['dev_path'], change)
    refresh_dev_source(deployment)
    result = check(deployment)
    assert not result['deploy'], result
    assert result['dev']['rules']['n'] == 60
    assert result['dev']['rules']['answer_f1'] == pytest.approx(59/60)
    assert result['online']['rules']['answer_f1'] == 1


def test_failed_dev_sample_stays_in_denominator(deployment):
    def change(rows):
        for row in rows:
            if row['query_id'] == 'dev-00' and row['method'] == 'hybrid_rerank':
                row['error'] = {'kind': 'parse_error'}
    rewrite_rows(deployment['dev_path'], change)
    refresh_dev_source(deployment)
    result = check(deployment)
    assert not result['deploy']
    assert result['dev']['rules']['answer_f1'] == pytest.approx(59/60)
    assert result['dev']['rules']['failure_rate'] == pytest.approx(1/60)


def test_failed_online_answer_retained_and_scored(deployment):
    def change(rows):
        row = next(r for r in rows if r['policy'] == 'rules')
        row['error'] = row['generation_record']['error'] = {'kind': 'parse_error'}
    rewrite_rows(deployment['output']/'records.jsonl', change)
    refresh_online_summary(deployment)
    result = check(deployment)
    assert result['deploy'], result  # Full 60-dev quality still passes, as preregistered.
    assert result['online']['rules']['n'] == 30
    assert result['online']['rules']['answer_f1'] == pytest.approx(29/30)


@pytest.mark.parametrize('damage', ['dev_missing', 'dev_duplicate', 'warmup_missing', 'ledger_missing', 'config_rehashed'])
def test_valid_outer_hashes_do_not_replace_complete_source_evidence(deployment, damage):
    if damage.startswith('dev_'):
        rewrite_rows(deployment['dev_path'], lambda rows: rows.pop() if damage == 'dev_missing' else rows.append(dict(rows[0])))
        refresh_dev_source(deployment)
    elif damage == 'warmup_missing': rewrite_rows(deployment['output']/'warmups.jsonl', lambda rows: rows.pop())
    elif damage == 'ledger_missing':
        lines = deployment['ledger'].read_text().splitlines()
        deployment['ledger'].write_text('\n'.join(lines[1:])+'\n', encoding='utf-8')
    else:
        path = deployment['output']/'records.config.json'
        config = json.loads(path.read_text()); config['online_hash'] = 'changed'
        atomic_json(path, config)
        rewrite_rows(deployment['output']/'records.jsonl', lambda rows: [r.update(config_hash=signature(config)) for r in rows])
        refresh_online_summary(deployment)
    assert not check(deployment)['deploy']
