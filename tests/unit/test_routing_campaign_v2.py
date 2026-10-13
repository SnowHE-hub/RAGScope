"""Contract fixtures only: no synthetic label is persisted as real evidence."""
import copy
import json

import pytest

from ragscope.routing.core import ROUTES, FixedRouter, build_proxy_labels, proxy_quality
from ragscope.routing.evaluate import summarize
from ragscope.routing.campaign import freeze_routing, normalized_record, freeze_decisions
from ragscope.experiments.generation_campaign import freeze_campaign
from test_generation_campaign_v2 import sample, IDENTITY


def test_unanswerable_quality_is_explicit_and_f1_stays_na():
    query = {'query_id': 'u', 'question': 'unknown', 'split': 'train', 'answerability': 'unanswerable'}
    rows = [dict(query, path=route, answer_f1=None, evidence_coverage=None, real_model=True,
        error=None, correct_abstention=float(i == 1), total_latency_s=i+1,
        comparison_config_hash='fixture') for i, route in enumerate(ROUTES)]
    labels = build_proxy_labels([query], rows, split='train', query_ids=['u'], expected_config_hash='fixture')
    assert labels[0]['label'] == 'hybrid' and labels[0]['quality_proxy'] == 'correct_abstention'
    assert all(r['answer_f1'] is None for r in labels[0]['route_scores'].values())
    summary = summarize([{**r, 'latency_s': 1, 'latency_kind': 'component_replay_estimate'} for r in rows])
    assert summary['answer_f1'] is None and summary['answer_f1_n'] == 0
    assert summary['correct_abstention_rate'] == pytest.approx(1/3)
    bad = dict(rows[1], error={'kind': 'timeout'})
    with pytest.raises(ValueError, match='zero'):
        proxy_quality(bad)
    with pytest.raises(ValueError, match='N/A'):
        proxy_quality(dict(rows[0], answer_f1=1.))


def test_frozen_router_ids_follow_manifest_and_decisions_ignore_gold(tmp_path):
    groups, inputs = [], {}
    for split, experiment in [('train', 'ROUTE_TRAIN'), ('dev', 'ROUTE_DEV'), ('test', 'E2')]:
        for method in (ROUTES if split != 'test' else ('R0', 'R2', 'R3')):
            ids = []
            for i in range(2):
                key = f'{split}_{method}_{i}'
                source = sample(f'{split}{i}')
                source['query'].update(split=split, scope={'kind': 'document', 'doc_ids': ['d']})
                source['query']['metadata'] = {'gold_doc_id_evaluation_only': 'secret', 'gold_answer': 'bm25'}
                source['document_length'] = {'reference_n_tokens': 1000}
                inputs[key] = source; ids.append(key)
            groups.append({'experiment_id': experiment, 'method': method, 'model_ids': ['qwen_2b'], 'input_ids': ids})
    freeze_campaign(tmp_path/'campaign', {'qwen_2b': IDENTITY}, groups, inputs)
    protocol = freeze_routing(tmp_path/'campaign', tmp_path/'route', counts={'train': 2, 'dev': 2, 'test': 1})
    assert protocol['populations']['train']['query_ids'] == ['train0', 'train1']
    assert len(protocol['populations']['test']['query_ids']) == 1
    assert 'gold' not in json.dumps(protocol['populations']) and 'secret' not in json.dumps(protocol['populations'])
    decision = freeze_decisions(FixedRouter(), protocol, tmp_path/'route', 'fixed_hybrid')
    assert freeze_decisions(FixedRouter(), protocol, tmp_path/'route', 'fixed_hybrid') == decision
    path = tmp_path/'route/DECISIONS_fixed_hybrid.json'
    body = json.loads(path.read_text()); body['decisions']['test'][0]['decision']['route'] = 'bm25'
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match='population'):
        freeze_decisions(FixedRouter(), protocol, tmp_path/'route', 'fixed_hybrid')


def test_actual_component_times_are_required_and_never_called_online():
    source = sample(); source['retrieval_timings'] = {'total_ms': 100}
    row = {'query_id': 'q', 'split': 'test', 'method': 'R2', 'scores': {'token_f1': .5,
        'predicted_abstain': False, 'wrong_abstention': 0, 'context_evidence': {'continuous_coverage': .75}},
        'timings': {'total_ms': 900}, 'real_model': True, 'record_hash': 'fixture',
        'task_id': 'fixture', 'call_ticket': {'call_number': 1}, 'context_bundle': source['context_bundle']}
    normalized = normalized_record(row, source, 'fixture')
    assert normalized['total_latency_s'] == 1 and normalized['latency_kind'] == 'component_replay_estimate'
    bad = copy.deepcopy(source); bad.pop('retrieval_timings')
    with pytest.raises(ValueError, match='actually measured'):
        normalized_record(row, bad, 'fixture')
