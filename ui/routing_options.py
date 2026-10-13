"""Lazy, verified experimental routing for the local application."""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter


RESIDENT_PROFILE = 'resident_index_router_reranker_and_generator_after_separate_warmups'


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _read_store(path, expected_config=None):
    """Read a RecordStore without its constructor's tail-repair/write behavior."""
    from ragscope.experiments.ledger import read_records
    from ragscope.indexing.identity import signature
    path = Path(path)
    config = json.loads(path.with_suffix('.config.json').read_text(encoding='utf-8'))
    _require(expected_config is None or config == expected_config, 'record_config_mismatch')
    rows = read_records(path, repair_tail=False)
    ids = set()
    for row in rows:
        _require(row.get('record_hash') and row.get('config_hash') == signature(config)
                 and isinstance(row.get('record_id'), str) and bool(row['record_id'])
                 and row.get('record_id') not in ids, 'record_hash_or_identity_mismatch')
        ids.add(row['record_id'])
    return config, rows


def _answer_quality(query, result):
    """Recompute answer quality; saved scores/summary claims are not evidence."""
    from ragscope.evaluation.generation_metrics import answer_reference_texts, token_f1
    failed = bool(result.get('error'))
    abstain = result.get('abstain') is True and not failed
    unanswerable = query['answerability'] == 'unanswerable'
    refs = answer_reference_texts(query['answer_refs'])
    _require(unanswerable or bool(refs), 'missing_dev_answer_references')
    return {'answer_f1': None if unanswerable else (0.0 if failed or abstain else token_f1(result['answer'], refs)),
            'answerability': query['answerability'], 'error': result.get('error'),
            'correct_abstention': float(abstain) if unanswerable else None,
            'wrong_abstain': abstain and not unanswerable}


def _finite_duration(value):
    import math
    _require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
             'invalid_measured_duration')
    return value


def _ticket_checker(manifest):
    """Compare reservations with the append-only global ledger, not row flags."""
    from ragscope.experiments.ledger import read_records
    from ragscope.experiments.runtime import runtime_home
    ledger = runtime_home()/'runs'/manifest['provenance']['spec']['campaign']/'generation_budget.calls.jsonl'
    calls, seen = {}, set()
    for ticket in read_records(ledger, repair_tail=False):
        number = ticket['call_number']
        _require(type(number) is int and number > 0 and number not in calls, 'invalid_budget_ledger')
        calls[number] = ticket

    def check(ticket, purpose, query_id, identity):
        number = ticket.get('call_number') if isinstance(ticket, dict) else None
        _require(type(number) is int and number not in seen and calls.get(number) == ticket
                 and ticket.get('purpose') == purpose and ticket.get('query_id') == query_id
                 and ticket.get('model_identity') == identity, 'missing_duplicate_or_unbound_call_ticket')
        _finite_duration(ticket['reserved_at_unix'])
        seen.add(number)
    return check


def _dev_evidence(campaign, manifest, inputs, protocol, frozen, check_ticket):
    """Read only the 180 dev rows, never the much larger test journals."""
    from collections import Counter
    from ragscope.experiments.generation_campaign import read_source_records
    from ragscope.indexing.identity import artifact
    from ragscope.routing.campaign import REQUIRED_ROUTERS, normalized_record, request_query, validate_decisions
    from ragscope.routing.core import ROUTES
    from ragscope.routing.evaluate import summarize
    from ragscope.routing.online_campaign import select_dev, verified_body
    model_id = frozen['model_id']
    tasks = [t for t in manifest['tasks'] if t['experiment_id'] == 'ROUTE_DEV' and t['model_id'] == model_id]
    dev_ids = protocol['populations']['dev']['query_ids']
    _require(len(dev_ids) == len(set(dev_ids)) == 60, 'full_60_dev_population_required')
    _require(Counter((t['query_id'], t['method']) for t in tasks)
             == Counter({(qid, route): 1 for qid in dev_ids for route in ROUTES}), 'dev_task_population_mismatch')
    dev_path = campaign/'records'/f'ROUTE_DEV_{model_id}.jsonl'
    _require(all(artifact(p) in frozen['sources'] for p in (dev_path, dev_path.with_suffix('.config.json'))),
             'dev_records_not_frozen_online_sources')
    rows = read_source_records({'directory': campaign, 'manifest': manifest, 'inputs': inputs,
                               'owned_task_ids': {t['task_id'] for t in tasks}})
    _require(len(rows) == 180, 'dev_generation_incomplete')
    visible = {q['query_id']: q for q in protocol['populations']['dev']['visible_queries']}
    records, queries = {}, {}
    for row in rows:
        source = inputs[row['input_id']]
        query = source['query']
        _require(row['question'] == query['question'] and row['split'] == query['split'] == 'dev'
                 and row['query_id'] == query['query_id'] and request_query(source) == visible[query['query_id']],
                 'dev_query_mismatch')
        if query['query_id'] in queries:
            _require(queries[query['query_id']] == query, 'dev_query_differs_across_routes')
        queries[query['query_id']] = query
        check_ticket(row['call_ticket'], 'ROUTE_DEV', query['query_id'], frozen['model_identity'])
        _finite_duration(source['retrieval_timings']['total_ms'])
        _finite_duration(row['timings']['total_ms'])
        quality = _answer_quality(query, row)
        scores = {**row['scores'], 'token_f1': quality['answer_f1'],
                  'predicted_abstain': bool(quality['correct_abstention']) if quality['answerability'] == 'unanswerable' else quality['wrong_abstain'],
                  'wrong_abstention': quality['wrong_abstain']}
        record = normalized_record({**row, 'scores': scores}, source, protocol['comparison_config_hash'])
        records[(row['query_id'], row['method'])] = record
    decisions, summaries = {}, {}
    for name in REQUIRED_ROUTERS:
        matches = [a for a in frozen['sources'] if Path(a['path']).name == f'DECISIONS_{name}.json']
        _require(len(matches) == 1, 'missing_frozen_policy_decisions')
        saved = verified_body(matches[0]['path'], 'decision_hash')
        decisions[name] = validate_decisions(saved, protocol, name)
        selected = []
        for item in saved['decisions']['dev']:
            decision = item['decision']
            record = records[item['query_id'], decision['route']]
            selected.append({**record, 'fallback': decision['fallback'],
                             'latency_s': record['total_latency_s']+decision['elapsed_s']})
        summaries[name] = summarize(selected)
    _require(frozen['selected'] == {'fixed': select_dev(summaries, list(REQUIRED_ROUTERS[:3])),
                                    'nonfixed': select_dev(summaries, list(REQUIRED_ROUTERS[3:]))},
             'dev_selection_not_reproduced')
    return queries, decisions, summaries


def deployment_readiness(policy, *, model_identity=None, index_manifest=None, runtime_config=None):
    """Fail closed unless raw evidence and the *current* runtime support default use.

    runtime_config contains retrieval_identity, embedding_identity, reranker_identity,
    context_models, generation_config, context_policy and execution_profile. Cold UI
    loads cannot use an online experiment that excludes index/router/reranker setup.
    This check never loads a model or reads test-generation journals.
    """
    from collections import Counter
    import random
    from ragscope.experiments.generation_campaign import load_campaign
    from ragscope.indexing.identity import signature, verify_artifacts
    from ragscope.routing.evaluate import deployment_gate, summarize
    from ragscope.routing.online_campaign import verified_body
    try:
        _require(model_identity and index_manifest and runtime_config, 'current_runtime_identity_required')
        _require(runtime_config['execution_profile'] == RESIDENT_PROFILE, 'execution_profile_mismatch')
        evidence = policy['deployment_evidence']
        _require(verify_artifacts([evidence['summary'], evidence['manifest'], policy['protocol'], policy['decisions']]),
                 'deployment_artifact_mismatch')
        frozen = verified_body(evidence['manifest']['path'], 'online_hash')
        _require(verify_artifacts(frozen['sources']+[frozen['implementation']]), 'online_provenance_changed')
        protocol = verified_body(policy['protocol']['path'], 'protocol_hash')
        _require(policy['protocol'] in frozen['sources'] and policy['decisions'] in frozen['sources']
                 and frozen['protocol_hash'] == protocol['protocol_hash'], 'policy_protocol_not_bound')
        _require(frozen['test_scores_used'] is False and frozen['selected']['nonfixed'] == policy['kind'],
                 'selection_not_dev_only')
        _require(model_identity == frozen['model_identity'] == protocol['comparison']['model_identity'], 'model_identity_mismatch')
        _require(runtime_config['generation_config'] == frozen['generation_config'] == protocol['comparison']['generation']
                 and runtime_config['context_policy'] == frozen['context_policy'], 'generation_or_context_policy_mismatch')
        _require(index_manifest['signature'] == signature(index_manifest['identity'])
                 and index_manifest['artifacts'] and verify_artifacts(index_manifest['artifacts']), 'index_manifest_mismatch')
        retrieval = runtime_config['retrieval_identity']
        _require(verify_artifacts(retrieval['artifacts']) and retrieval['index_signature'] == index_manifest['signature']
                 and retrieval['retrieval_config_hash'] == signature(retrieval['retrieval_config'])
                 and retrieval['expansion'] == policy['expansion'] and retrieval['reference_tokens'] == 2048
                 and retrieval['candidate_depth'] == 50 and retrieval['final_k'] == 10
                 and retrieval['no_gold_inputs'] is True
                 and runtime_config['embedding_identity'] == index_manifest['identity']['embedding'], 'retrieval_runtime_mismatch')
        summary = json.loads(Path(evidence['summary']['path']).read_text(encoding='utf-8'))
        _require(summary['status'] == 'COMPLETE' and summary['test_used'] is False
                 and summary['online_hash'] == frozen['online_hash'] and verify_artifacts([summary['records']]),
                 'online_records_not_complete')
        directory = Path(evidence['manifest']['path']).parent
        _require(Path(summary['records']['path']).resolve() == (directory/'records.jsonl').resolve(), 'online_record_path_mismatch')
        expected_config = {'online_hash': frozen['online_hash'], 'retrieval_identity': retrieval,
                           'model_identity': model_identity, 'context_models': runtime_config['context_models']}
        _, rows = _read_store(directory/'records.jsonl', expected_config)
        _, warmups = _read_store(directory/'warmups.jsonl', {**expected_config, 'purpose': 'warmup'})
        _require(verify_artifacts([protocol['source']]), 'source_campaign_changed')
        campaign = Path(protocol['source']['path']).parent
        manifest, inputs = load_campaign(campaign)
        _require(manifest['manifest_hash'] == frozen['campaign_manifest_hash'] == protocol['comparison']['manifest_hash']
                 and manifest['models'][frozen['model_id']] == model_identity
                 and manifest['generation_config'] == runtime_config['generation_config']
                 and protocol['comparison_config_hash'] == signature(protocol['comparison'])
                 and manifest['seed'] == frozen['seed'] == protocol['seed']
                 and runtime_config['context_models'] == [manifest['models'][n] for n in ('qwen_2b', 'qwen_4b', 'qwen_9b')],
                 'campaign_or_common_context_models_mismatch')
        check_ticket = _ticket_checker(manifest)
        queries, decisions, dev = _dev_evidence(campaign, manifest, inputs, protocol, frozen, check_ticket)
        names = list(frozen['selected'].values())
        _require(all(decisions[n]['router_identity'] == frozen['router_identities'][n] for n in names), 'router_identity_mismatch')
        expected_ids = sorted(queries)
        random.Random(protocol['seed']).shuffle(expected_ids)
        _require(frozen['query_ids'] == expected_ids[:30] and frozen['n_per_policy'] == protocol['online_dev_n'] == 30,
                 'paired_30_online_population_required')
        _require(Counter((r['policy'], r['query_id']) for r in rows)
                 == Counter({(n, q): 1 for n in names for q in expected_ids[:30]}), 'online_population_incomplete_or_duplicate')
        groups, warm_counts, observed_rerankers = {n: [] for n in names}, Counter(), []
        selected_routes = {(n, item['query_id']): item['decision']['route']
                           for n in names for item in decisions[n]['decisions']['dev']}
        for row, is_warmup in [(r, False) for r in rows]+[(r, True) for r in warmups]:
            name, qid = row['policy'], row['query_id']
            purpose = 'router_online_warmup' if is_warmup else 'router_online'
            _require(name in names and qid in queries and row['split'] == 'dev' and row['purpose'] == purpose
                     and row['real_model'] is True and row['generation_attempted'] is True and row['call_reserved'] is True
                     and row['model_identity'] == model_identity and row['latency_kind'] == 'actual_online'
                     and row['model_residency'] == 'same_2b_service_resident_after_separate_warmups', 'not_actual_online_generation')
            check_ticket(row['call_ticket'], purpose, qid, model_identity)
            result, live = row['generation_record'], row['retrieval_record']
            _require(result['real_model'] is True and result['model_identity'] == model_identity
                     and result['question'] == queries[qid]['question'] and result.get('error') == row.get('error')
                     and live['retrieval_identity'] == retrieval and live['index_signature'] == index_manifest['signature']
                     and row['selected_route'] == row['router_decision']['route'] == live['route'] == selected_routes[name, qid]
                     and live['question'] == queries[qid]['question'],
                     'online_record_runtime_mismatch')
            _require((result.get('error') or {}).get('kind') not in {'service_not_ready', 'device_policy_error'}, 'online_request_not_started')
            if live.get('embedding_identity') is not None:
                _require(live['embedding_identity'] == runtime_config['embedding_identity'], 'embedding_identity_mismatch')
            if live.get('reranker_identity') is not None:
                _require(live['reranker_identity'] == runtime_config['reranker_identity'], 'reranker_identity_mismatch')
                observed_rerankers.append(live['reranker_identity'])
            _finite_duration(row['latency_s'])
            if is_warmup:
                _require(qid in frozen['query_ids'][:3], 'warmup_query_mismatch')
                warm_counts[(row['session'], name, qid)] += 1
            else:
                _require(row['record_id'] == f'{name}:{qid}', 'online_record_id_mismatch')
                groups[name].append({**row, **_answer_quality(queries[qid], result)})
        _require(observed_rerankers, 'reranker_runtime_not_demonstrated')
        for row in rows:
            _require(all(warm_counts[(row['session'], n, q)] == 1 for n in names for q in frozen['query_ids'][:3]),
                     'session_warmups_incomplete')
        online = {name: summarize(values) for name, values in groups.items()}
        fixed, adaptive = (frozen['selected'][key] for key in ('fixed', 'nonfixed'))
        # Reuse the existing engineering gate with freshly recomputed full-dev
        # quality and actual paired latency. Failed answers remain zero in F1.
        gate = deployment_gate({**online[adaptive], 'answer_f1': dev[adaptive]['answer_f1']},
                               {**online[fixed], 'answer_f1': dev[fixed]['answer_f1']}, online_n=30)
        return {**gate, 'reason': 'verified_full_dev_quality_and_paired_online_latency' if gate['deploy'] else 'recomputed_threshold_not_met',
                'dev': {n: dev[n] for n in names}, 'online': online}
    except (KeyError, TypeError, ValueError, OSError, IndexError, AttributeError) as exc:
        return {'deploy': False, 'default': 'fixed', 'reason': str(exc) or type(exc).__name__}


def deployable_default(policy, *, model_identity=None, index_manifest=None, runtime_config=None):
    return deployment_readiness(policy, model_identity=model_identity, index_manifest=index_manifest,
                                runtime_config=runtime_config)['deploy']


def visible_request(question, scope, documents, corpus_sha256):
    """Read canonical document metadata, never queries, answers or evidence."""
    from ragscope.indexing.identity import verify_artifacts
    from ragscope.routing.core import RoutingRequest
    if documents.get('sha256') != corpus_sha256 or not verify_artifacts([documents]):
        raise ValueError('路由文档来源与检索语料不一致')
    if not scope:
        return RoutingRequest(question, {'kind': 'global'}, {})
    selected = None
    with Path(documents['path']).open(encoding='utf-8') as handle:
        for line in handle:
            doc = json.loads(line)
            if doc['doc_id'] == scope:
                if selected is not None:
                    raise ValueError('资料范围在语料中重复')
                selected = doc.get('metadata', {})
    if selected is None:
        raise ValueError('资料范围不在已核验语料中')
    count = selected.get('reference_n_tokens')
    if type(count) is not int or count < 0:
        raise ValueError('缺少规范文档的真实 reference token 长度')
    return RoutingRequest(question, {'kind': 'document', 'doc_ids': [scope]},
        {'document_length_tokens': count, 'document_type': 'paper', 'language': 'en'})


def choose_route(policy, question, scope, index_entry, index_identity):
    """Return a method override; context/index/model choices remain fixed."""
    from ragscope.indexing.identity import verify_artifacts
    from ragscope.routing.campaign import validate_decisions
    from ragscope.routing.online_campaign import load_router, verified_body
    if policy.get('ready') is not True or policy.get('index_id') != index_entry['id']:
        raise ValueError('该实验路由未就绪或未绑定当前索引')
    if (index_identity.get('embedding') is None
            or not index_entry.get('embedding', {}).get('ready')
            or not index_entry.get('reranker', {}).get('ready')):
        raise ValueError('实验路由要求 BM25、Hybrid、Hybrid+Rerank 三条路径均就绪')
    if policy.get('expansion') != index_entry.get('expansion', 'none'):
        raise ValueError('路由上下文策略与冻结索引配置不一致')
    sources = [policy['protocol'], policy['decisions']]
    if not verify_artifacts(sources):
        raise ValueError('路由实验来源校验失败')
    protocol = verified_body(policy['protocol']['path'], 'protocol_hash')
    decisions = json.loads(Path(policy['decisions']['path']).read_text(encoding='utf-8'))
    name = policy['kind']
    validate_decisions(decisions, protocol, name)
    started = perf_counter()
    request = visible_request(question, scope, policy['documents'], index_identity['corpus_sha256'])
    metadata_ms = (perf_counter()-started)*1000
    if name == 'lstm':
        from ragscope.experiments.runtime import require_ram
        require_ram(512*1024**2, reserve_bytes=1024**3)
    started = perf_counter()
    router = load_router(name, Path(policy['protocol']['path']).parent)
    load_ms = (perf_counter()-started)*1000
    identity = getattr(router, 'identity', getattr(router, 'report', {}))
    if identity != decisions['router_identity']:
        raise ValueError('路由模型与已完成实验的冻结选择不一致')
    if name in {'linear', 'mlp', 'lstm'}:
        report = router.report
        if (report.get('data_kind') != 'real_generation_proxy'
                or set(report.get('train_query_ids', [])) != set(protocol['populations']['train']['query_ids'])
                or set(report.get('dev_query_ids', [])) != set(protocol['populations']['dev']['query_ids'])):
            raise ValueError('路由训练必须来自相同真实 train/dev 生成记录')
    started = perf_counter()
    decision = router.choose(request).to_dict()
    decision_ms = (perf_counter()-started)*1000
    method = 'hybrid+rerank' if decision['route'] == 'hybrid_rerank' else decision['route']
    return {**index_entry, 'method': method}, {
        'policy_id': policy['id'], 'kind': name, 'decision': decision,
        'visible_request': {'question': request.question, 'scope': request.scope, 'metadata': request.metadata},
        'protocol_hash': protocol['protocol_hash'], 'decision_hash': decisions['decision_hash'],
        'timings': {'routing_metadata_ms': metadata_ms, 'router_load_ms': load_ms, 'routing_ms': decision_ms},
    }
