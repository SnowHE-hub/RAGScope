"""Synthetic corpora/clients are isolated contract fixtures, never real evidence."""
import copy
import json
from dataclasses import asdict

import pytest

from ragscope.experiments.generation_campaign import freeze_campaign, load_campaign, run_model, GenerationBudget
from ragscope.experiments.production_acceptance import accept_campaign, MAIN_MODELS, subset
from ragscope.generation.context import pack_context
from ragscope.generation.llm_client import GenerationError
from ragscope.indexing.identity import artifact, atomic_json, file_hash, signature


def corpus_fixture(path, squad=False):
    path.mkdir(parents=True)
    (path/'documents.jsonl').write_text('[]\n')
    populations = {}
    for split, n in ({'test': 100} if squad else {'train': 1000, 'dev': 100, 'test': 200}).items():
        populations[split] = [dict(query_id=f'{"squad" if squad else "qasper"}:{split}:{i}',
            question=f'Question {i}?', doc_id=f'd{i}', scope=f'd{i}', split=split,
            answerability='unanswerable', answer_refs=[], evidence_reference_sets=[]) for i in range(n)]
        (path/f'queries_{split}.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in populations[split]))
    receipt = {'files': {p.name: {'sha256': file_hash(p)} for p in path.glob('*.jsonl')}}
    if squad:
        receipt['selected_query_ids'] = [q['query_id'] for q in populations['test']]
    else:
        receipt['query_ids'] = {k: [q['query_id'] for q in v] for k, v in populations.items()}
    atomic_json(path/'SPLIT_MANIFEST.json', receipt)
    return populations


def campaign_fixture(tmp_path, scope='e3', block_all=False):
    corpus = tmp_path/'corpus'
    queries = corpus_fixture(corpus)
    squad = corpus_fixture(corpus/'squad_e4', True)
    evidence = tmp_path/'fixture_resource_block.txt'
    evidence.write_text('Synthetic test fixture only')
    identities = {m: {'model': 'offline-fixture-'+m, 'revision': 'fixture',
        'weight_sha256': 'fixture-not-real-weights', 'backend_version': 'fixture'} for m in MAIN_MODELS+('lfm',)}
    groups, inputs = [], {}
    bundle = asdict(pack_context([]))

    def add(exp, methods, models, rows, blocked=False):
        for method in methods:
            group = dict(experiment_id=exp, method=method, model_ids=list(models), expected_n=len(rows))
            if blocked:
                group.update(status='BLOCKED', reason='Synthetic resource fixture', evidence=[artifact(evidence)])
            else:
                group.update(status='READY', input_ids=[])
                for q in rows:
                    iid = f'{exp}:{method}:{q["query_id"]}'
                    inputs[iid] = {'query': q, 'context_bundle': copy.deepcopy(bundle)}
                    group['input_ids'].append(iid)
            groups.append(group)

    if scope in {'main','e2'}:
        add('E2', ('R0','R1','R2','R3'), MAIN_MODELS, queries['test'], block_all)
    if scope in {'main','e3'}:
        add('E3', ('oracle','no_context'), MAIN_MODELS, queries['test'], block_all)
    if scope == 'main':
        add('E4', ('hybrid',), MAIN_MODELS, squad['test'], block_all)
        add('E5', ('bm25','hybrid'), ('lfm',), subset(queries['test'],100), block_all)
        add('E6', ('simple','structured'), MAIN_MODELS[:2], subset(queries['test'],100), block_all)
        for split,n in [('train',200),('dev',60)]:
            add('ROUTE_'+split.upper(), ('bm25','hybrid','hybrid_rerank'), ('qwen_2b',), subset(queries[split],n), block_all)
        add('E7', ('start','middle','end'), MAIN_MODELS[:2], queries['test'][:40], True)
    spec = dict(corpus_dir=str(corpus), main_models=list(MAIN_MODELS), small_models=list(MAIN_MODELS[:2]),
                smallest_model='qwen_2b', heterogeneous_model='lfm')
    run = tmp_path/'run'
    freeze_campaign(run, identities if scope == 'main' else {k: identities[k] for k in MAIN_MODELS}, groups, inputs,
        provenance={'spec': spec, 'corpus_manifest': artifact(corpus/'SPLIT_MANIFEST.json'),
                    'readiness': {'source_artifacts': {'squad_split': artifact(corpus/'squad_e4/SPLIT_MANIFEST.json')}}})
    return run, corpus


def rewrite_manifest(run, change):
    path = run/'EXPERIMENT_MANIFEST.json'
    manifest = json.loads(path.read_text())
    change(manifest)
    manifest.pop('manifest_hash')
    manifest['manifest_hash'] = signature(manifest)
    atomic_json(path, manifest)


def test_actual_scope_and_full_fixed_denominators(tmp_path):
    run, corpus = campaign_fixture(tmp_path)
    result = accept_campaign(run, corpus, scope='e3')
    assert result['status']=='PENDING' and result['integrity_valid'] is True
    assert result['n_expected']==1200 and result['n_missing']==1200
    assert accept_campaign(run, corpus, scope='main')['integrity_valid'] is False


def test_e2_shard_acceptance_requires_all2400_and_is_not_main(tmp_path):
    run,corpus=campaign_fixture(tmp_path,scope='e2')
    result=accept_campaign(run,corpus,scope='e2')
    assert result['integrity_valid'] is True and result['n_expected']==result['n_missing']==2400
    assert result['required_core_n']==2400 and result['status']=='PENDING'
    assert len(result['conditions'])==12 and 'not main' in result['scope_limitation']
    assert accept_campaign(run,corpus,scope='main')['status']=='FAILED_VALIDATION'
    rewrite_manifest(run,lambda m:m['groups'].pop())
    assert accept_campaign(run,corpus,scope='e2')['status']=='FAILED_VALIDATION'


@pytest.mark.parametrize('change', [
    lambda m: m['groups'].pop(),
    lambda m: m['groups'][0].update(expected_n=199,input_ids=m['groups'][0]['input_ids'][:-1]),
    lambda m: m['tasks'].append(copy.deepcopy(m['tasks'][0])),
    lambda m: m['groups'][0]['model_ids'].pop(),
])
def test_rehashed_manifest_cannot_shrink_omit_or_duplicate_production(tmp_path, change):
    run, corpus = campaign_fixture(tmp_path)
    rewrite_manifest(run, change)
    result = accept_campaign(run, corpus, scope='e3')
    assert result['status']=='FAILED_VALIDATION' and result['integrity_valid'] is False


def test_rehashed_query_text_cannot_replace_canonical_question(tmp_path):
    run, corpus = campaign_fixture(tmp_path)
    path = run/'FROZEN_INPUTS.json'
    inputs = json.loads(path.read_text())
    iid = next(iter(inputs))
    row = inputs[iid]
    row['query']['question'] = 'Rewritten test question'
    row.pop('input_hash'); row['input_hash'] = signature(row)
    atomic_json(path, inputs)
    def change(m):
        m['input_hashes'][iid] = row['input_hash']
        for t in m['tasks']:
            if t['input_id']==iid:
                t['input_hash']=row['input_hash']; t.pop('task_id'); t['task_id']=signature(t)
    rewrite_manifest(run, change)
    assert accept_campaign(run, corpus, scope='e3')['integrity_valid'] is False


def test_all_mandatory_extensions_remain_in_denominator(tmp_path):
    run, corpus = campaign_fixture(tmp_path, scope='main')
    report = accept_campaign(run, corpus)
    assert report['integrity_valid'] is True and report['status']=='PENDING'
    assert report['n_expected']==5520 and report['n_blocked']==240 and report['n_missing']==5280
    rewrite_manifest(run, lambda m: m['groups'].__setitem__(slice(None), [g for g in m['groups'] if g['experiment_id']!='E6']))
    assert accept_campaign(run, corpus)['integrity_valid'] is False


def test_complete_runnable_subset_does_not_hide_blocked_matrix(tmp_path):
    run, corpus = campaign_fixture(tmp_path, scope='main', block_all=True)
    result = accept_campaign(run, corpus)
    assert result['status']=='BLOCKED' and result['integrity_valid'] is True
    assert result['n_blocked']==5520 and result['n_missing']==0
    (tmp_path/'fixture_resource_block.txt').write_text('Modified after freeze')
    assert accept_campaign(run, corpus)['integrity_valid'] is False


def test_missing_manifest_is_pending_and_never_complete(tmp_path):
    result = accept_campaign(tmp_path/'absent', tmp_path/'corpus')
    assert result['status']=='PENDING' and result['required_core_n']==3600


def test_failed_real_request_fixture_still_counts_as_observed(tmp_path):
    run, corpus = campaign_fixture(tmp_path)
    manifest,_ = load_campaign(run)
    class OfflineFixtureClient:
        real_model=True
        identity=manifest['models']['qwen_2b']
        last_metadata={}
        prompt_token_counter=staticmethod(lambda messages: 1)
        def __init__(self): self.n=0
        def chat(self,*args,**kwargs):
            self.n+=1
            if self.n>3: raise GenerationError('transport_error','Offline fixture failure')
            return '{"answer":"","citations":[],"abstain":true}'
    run_model(run,'qwen_2b',OfflineFixtureClient(),GenerationBudget(tmp_path/'budget.json'),evidence_units={},max_tasks=1)
    result=accept_campaign(run,corpus,scope='e3')
    assert result['status']=='RUNNING' and result['n_observed']==1 and result['n_failed']==1
    assert result['n_missing']==1199
