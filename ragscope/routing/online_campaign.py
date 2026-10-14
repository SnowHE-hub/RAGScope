"""Frozen dev-only policy selection and resumable actual end-to-end comparison."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random
from time import perf_counter, time_ns

from filelock import FileLock

from ragscope.evaluation.v2 import score_generation
from ragscope.experiments.generation_campaign import CountedClient, load_campaign
from ragscope.experiments.ledger import RecordStore
from ragscope.experiments.runtime import require_ram
from ragscope.generation.context import pack_context
from ragscope.generation.pipeline import RAGPipeline
from ragscope.indexing.identity import artifact, atomic_json, signature, verify_artifacts
from .campaign import REQUIRED_ROUTERS, immutable_json, load_route_records
from .core import RoutingRequest
from .evaluate import deployment_gate, summarize


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verified_body(path, key):
    body = read_json(path)
    expected = body.pop(key)
    if signature(body) != expected:
        raise ValueError(f'Frozen routing hash differs: {Path(path).name}')
    return {**body, key: expected}


def select_dev(summaries, names, tolerance=.01):
    """Quality first, then recorded replay cost within the fixed tolerance."""
    valid = [name for name in names if summaries[name].get('answer_f1') is not None]
    if not valid:
        raise ValueError('Dev policy selection requires answerable questions')
    best = max(summaries[name]['answer_f1'] for name in valid)
    eligible = [name for name in valid if best-summaries[name]['answer_f1'] <= tolerance+1e-12]
    return min(eligible, key=lambda name: (summaries[name]['p50_latency_s'], names.index(name)))


def freeze_online(campaign, routing_dir, output, *, n=30):
    """Read only dev scores, after all seven policies' decisions are frozen."""
    routing_dir, output = Path(routing_dir), Path(output)
    protocol = verified_body(routing_dir/'ROUTING_MANIFEST.json', 'protocol_hash')
    if n != protocol['online_dev_n'] or n < 30:
        raise ValueError('Online sample must equal the preregistered minimum of at least 30')
    decisions, sources = {}, []
    for name in REQUIRED_ROUTERS:
        path = routing_dir/f'DECISIONS_{name}.json'
        decision = verified_body(path, 'decision_hash')
        if decision['protocol_hash'] != protocol['protocol_hash'] or decision['name'] != name:
            raise ValueError('Mandatory decision protocol differs')
        for split in ('dev', 'test'):
            ids = [row['query_id'] for row in decision['decisions'][split]]
            if len(ids) != len(set(ids)) or set(ids) != set(protocol['populations'][split]['query_ids']):
                raise ValueError('Mandatory decision population differs')
        decisions[name] = decision
        sources.append(artifact(path))
    records, record_sources = load_route_records(campaign, protocol, ('dev',))
    by_key = {(row['query_id'], row['path']): row for row in records}
    summaries = {}
    for name, saved in decisions.items():
        rows = []
        for item in saved['decisions']['dev']:
            decision = item['decision']
            row = by_key[item['query_id'], decision['route']]
            rows.append({**row, 'fallback': decision['fallback'],
                         'latency_s': row['total_latency_s']+decision['elapsed_s']})
        summaries[name] = summarize(rows)
    fixed = select_dev(summaries, list(REQUIRED_ROUTERS[:3]))
    adaptive = select_dev(summaries, list(REQUIRED_ROUTERS[3:]))
    ids = sorted(protocol['populations']['dev']['query_ids'])
    random.Random(protocol['seed']).shuffle(ids)
    ids = ids[:n]
    if len(ids) != n:
        raise ValueError('Insufficient frozen dev population')
    selected = {'fixed': fixed, 'nonfixed': adaptive}
    selected_models = []
    for name in selected.values():
        if name in {'linear', 'mlp', 'lstm'}:
            path = routing_dir/name/'manifest.json'
            receipt = read_json(path)
            if signature(receipt['identity']) != receipt['signature'] or not verify_artifacts(receipt['artifacts']):
                raise ValueError('Selected router weights changed')
            report = receipt['identity']['report']
            if (report.get('data_kind') != 'real_generation_proxy'
                    or set(report.get('train_query_ids', [])) != set(protocol['populations']['train']['query_ids'])
                    or set(report.get('dev_query_ids', [])) != set(protocol['populations']['dev']['query_ids'])):
                raise ValueError('Selected learned router must use the exact real train/dev proxy population')
            selected_models.extend([artifact(path), *receipt['artifacts']])
    body = {'version': 'online-routing-v2', 'protocol_hash': protocol['protocol_hash'],
            'campaign_manifest_hash': protocol['comparison']['manifest_hash'],
            'selected': selected, 'dev_summaries_for_selection': summaries,
            'selection_rule': 'dev_answerable_f1_within_0.01_then_replay_p50_then_fixed_order',
            'test_scores_used': False, 'query_ids': ids, 'n_per_policy': n,
            'seed': protocol['seed'], 'warmups_per_policy_per_session': 3,
            'model_id': protocol['comparison']['model_id'],
            'model_identity': protocol['comparison']['model_identity'],
            'generation_config': protocol['comparison']['generation'],
            'router_identities': {name: decisions[name]['router_identity'] for name in selected.values()},
            'sources': [artifact(routing_dir/'ROUTING_MANIFEST.json'), *sources, *record_sources, *selected_models],
            'latency_kind': 'actual_online', 'timing': 'router_choose + live_retrieval + shared_context + generation',
            'context_policy': '2048_reference_retrieval_then_common_all_three_qwen_chat_budget',
            'implementation': artifact(Path(__file__))}
    body['online_hash'] = signature(body)
    immutable_json(output/'ONLINE_MANIFEST.json', body)
    return body


def load_router(name, routing_dir):
    from .core import FixedRouter, RuleRouter
    if name.startswith('fixed_'):
        return FixedRouter(name.removeprefix('fixed_'))
    if name == 'rules':
        return RuleRouter()
    if name in {'linear', 'mlp'}:
        from .learned import TfidfRouter
        return TfidfRouter.load(Path(routing_dir)/name)
    if name == 'lstm':
        from .lstm import LSTMRouter
        return LSTMRouter.load(Path(routing_dir)/name)
    raise ValueError('Unknown frozen online policy')


def authorize_remote_gpu(client, verify_owned):
    """Bind only a verified owned GPU HTTP service; never expose local CUDA."""
    if client.num_gpu <= 0 or client.service_device_verified != 'gpu':
        raise ValueError('Online remote GPU client must identify a verified GPU service')
    verify_owned(client.identity)
    client.remote_gpu_authorizer = verify_owned


def run_online(campaign, routing_dir, output, client, budget, retriever, routers, *,
               common_clients, shared_context, evidence_units, max_tasks=None):
    """Caller holds the global generation lock and supplies guarded real clients."""
    output = Path(output)
    frozen = verified_body(output/'ONLINE_MANIFEST.json', 'online_hash')
    if not verify_artifacts(frozen['sources']+[frozen['implementation']]):
        raise ValueError('Frozen online provenance changed')
    manifest, inputs = load_campaign(campaign)
    if manifest['manifest_hash'] != frozen['campaign_manifest_hash']:
        raise ValueError('Generation campaign changed')
    if client.real_model is not True or client.identity != frozen['model_identity']:
        raise ValueError('Online model must match the fixed smallest main model')
    expected = [manifest['models'][name] for name in ('qwen_2b', 'qwen_4b', 'qwen_9b')]
    if [item.identity for item in common_clients] != expected:
        raise ValueError('Online packing requires all three frozen Qwen counters')
    if max_tasks is not None and max_tasks < 1:
        raise ValueError('max_tasks must be positive')
    names = list(frozen['selected'].values())
    if set(routers) != set(names):
        raise ValueError('Both and only the selected policies must run')
    for name, router in routers.items():
        identity = getattr(router, 'identity', getattr(router, 'report', {}))
        if identity != frozen['router_identities'][name]:
            raise ValueError('Loaded router differs from the frozen decisions')
    protocol = verified_body(Path(routing_dir)/'ROUTING_MANIFEST.json', 'protocol_hash')
    visible = {q['query_id']: q for q in protocol['populations']['dev']['visible_queries']}
    sources = {}
    for source in inputs.values():
        q = source['query']
        if q['query_id'] in frozen['query_ids']:
            if q['split'] != 'dev':
                raise ValueError('Online test leakage')
            if q['query_id'] in sources and sources[q['query_id']]['query'] != q:
                raise ValueError('Query differs between route inputs')
            sources[q['query_id']] = source
    if set(sources) != set(frozen['query_ids']):
        raise ValueError('Frozen online queries are missing')
    # Verify gold scoring data before any calls, but pass only visible request fields to policies/retrieval.
    empty = asdict(pack_context([]))
    for source in sources.values():
        score_generation(source['query'], {'error': {'kind': 'preflight'}, 'context_bundle': empty}, evidence_units=evidence_units)
    config = {'online_hash': frozen['online_hash'], 'retrieval_identity': retriever.identity,
              'model_identity': client.identity, 'context_models': expected}
    completed, stop_reason, consecutive_failures = 0, None, 0
    with FileLock(str(output/'online-writer.lock'), timeout=0):
        store = RecordStore(output/'records.jsonl', config)
        warmups = RecordStore(output/'warmups.jsonl', {**config, 'purpose': 'warmup'})
        tasks = []
        for index, qid in enumerate(frozen['query_ids']):
            # Alternate paired policy order, avoiding a fixed first-policy cache advantage.
            tasks.extend((name, qid) for name in (names if index % 2 == 0 else names[::-1]))
        pending = [(name, qid) for name, qid in tasks if f'{name}:{qid}' not in store.ids]
        session = str(time_ns())

        def execute(name, qid, purpose):
            request = RoutingRequest.from_query(visible[qid])
            query = sources[qid]['query']
            counted = CountedClient(client, budget, purpose, qid)
            started = perf_counter()
            live, decision, result = {}, None, None
            failure_phase = 'router_error'
            try:
                decision = routers[name].choose(request)
                failure_phase = 'retrieval_error'
                live = retriever.retrieve(request.question, request.scope, decision.route)
                if live.get('error'):
                    raise RuntimeError(str(live['error']))
                failure_phase = 'context_error'
                source = shared_context({'query': {'question': request.question, 'scope': request.scope},
                                         'context_bundle': live['context_bundle']}, common_clients)
                result = RAGPipeline(None, counted, use_cache=False, config=frozen['generation_config']).answer(
                    request.question, context_bundle=source['context_bundle'], scope=request.scope, formal_evaluation=True)
                generated = asdict(result)
            except Exception as exc:
                generated = {'answer': '', 'citations': [], 'abstain': False, 'real_model': False,
                    'model_identity': client.identity, 'error': {'kind': failure_phase, 'message': str(exc)},
                    'context_bundle': live.get('context_bundle') or empty, 'raw_output': ''}
            elapsed = perf_counter()-started
            scores = score_generation(query, generated, evidence_units=evidence_units)
            unanswerable = query['answerability'] == 'unanswerable'
            return {'query_id': qid, 'split': 'dev', 'policy': name,
                'selected_route': decision.route if decision else None,
                'router_decision': decision.to_dict() if decision else None,
                'fallback': decision.fallback if decision else False,
                'answer_f1': scores['token_f1'], 'answerability': query['answerability'],
                'correct_abstention': float(not generated.get('error') and scores['predicted_abstain']) if unanswerable else None,
                'wrong_abstain': bool(scores['wrong_abstention']),
                'evidence_coverage': scores['context_evidence']['continuous_coverage'],
                'error': generated.get('error'), 'scores': scores,
                'latency_s': elapsed, 'latency_kind': 'actual_online',
                'model_residency': 'same_2b_service_resident_after_separate_warmups',
                'real_model': generated.get('real_model') is True and counted.last_ticket is not None,
                'call_reserved': counted.last_ticket is not None,
                'generation_attempted': counted.last_ticket is not None and (generated.get('error') or {}).get('kind') not in {
                    'service_not_ready', 'device_policy_error'},
                'model_identity': client.identity, 'call_ticket': counted.last_ticket,
                'generation_record': generated, 'retrieval_record': live,
                'resources': generated.get('backend_metadata', {}).get('resources', {}),
                'session': session, 'purpose': purpose}

        def breaker(row):
            nonlocal consecutive_failures, stop_reason
            error = row.get('error') or {}
            if error.get('kind') in {'transport_error', 'timeout', 'oom', 'service_not_ready',
                                     'device_policy_error', 'retrieval_error', 'router_error', 'context_error'}:
                consecutive_failures += 1
            else:
                consecutive_failures = 0
            if error.get('kind') == 'generation_budget_exhausted' or error.get('kind') == 'oom' or consecutive_failures >= 3:
                stop_reason = {'kind': 'online_circuit_break', 'last_error': error,
                               'consecutive_failures': consecutive_failures}

        def resource_ready(name, qid, purpose):
            nonlocal stop_reason
            try:
                require_ram(0, reserve_bytes=1280*1024**2)
            except RuntimeError as exc:
                # This request has not started: no record, budget ticket or failure.
                stop_reason = {'kind': 'incomplete_resource', 'message': str(exc),
                               'reserve_bytes': 1280*1024**2, 'policy': name,
                               'query_id': qid, 'purpose': purpose}
                return False
            return True

        if pending:
            for name in names:
                for number, qid in enumerate(frozen['query_ids'][:3]):
                    if not resource_ready(name, qid, 'router_online_warmup'):
                        break
                    row = execute(name, qid, 'router_online_warmup')
                    warmups.add(f'{session}:{name}:{number}', row)
                    breaker(row)
                    if stop_reason:
                        break
                if stop_reason:
                    break
        for name, qid in pending:
            if stop_reason or (max_tasks is not None and completed >= max_tasks):
                break
            if not resource_ready(name, qid, 'router_online'):
                break
            row = execute(name, qid, 'router_online')
            store.add(f'{name}:{qid}', row)
            completed += 1
            breaker(row)
            print(json.dumps({'policy': name, 'query_id': qid, 'online_completed': completed,
                              'error': row['error']}), flush=True)
        groups = {}
        for name in names:
            rows = [row for row in store.records if row['policy'] == name]
            groups[name] = {'n_expected': frozen['n_per_policy'], 'n_pending': frozen['n_per_policy']-len(rows),
                            'n_generation_attempted': sum(row['generation_attempted'] for row in rows),
                            'n_pre_generation_failures': sum(not row['generation_attempted'] for row in rows),
                            'summary': summarize(rows) if rows else None}
        complete = all(group['n_pending'] == 0 for group in groups.values())
        if complete:
            fixed, nonfixed = frozen['selected']['fixed'], frozen['selected']['nonfixed']
            gate = deployment_gate(groups[nonfixed]['summary'], groups[fixed]['summary'], online_n=frozen['n_per_policy'])
            gate['online_subset_quality_ok'] = gate['quality_ok']
            full_fixed, full_nonfixed = (frozen['dev_summaries_for_selection'][name] for name in (fixed, nonfixed))
            gate['quality_ok'] = full_nonfixed['answer_f1'] >= full_fixed['answer_f1']-.01
            gate['quality_population'] = {'kind': 'all_frozen_dev', 'n': full_fixed['n'],
                'answerable_n': full_fixed['answer_f1_n'], 'fixed_f1': full_fixed['answer_f1'],
                'nonfixed_f1': full_nonfixed['answer_f1']}
            gate['actual_request_minimum_met'] = all(group['n_generation_attempted'] >= 30 for group in groups.values())
            gate['deploy'] = bool(gate['quality_ok'] and gate['latency_ok'] and gate['actual_request_minimum_met'])
            gate['default'] = 'router' if gate['deploy'] else 'fixed'
            gate['reason'] = 'full_dev_quality_and_actual_paired_latency' if gate['actual_request_minimum_met'] else 'fewer_than_30_actual_requests_per_policy'
        else:
            gate = {'deploy': False, 'default': 'fixed', 'reason': 'actual_online_comparison_incomplete'}
        status = ('COMPLETE' if complete else 'INCOMPLETE_RESOURCE'
                  if (stop_reason or {}).get('kind') == 'incomplete_resource' else 'PENDING')
        summary = {'status': status, 'online_hash': frozen['online_hash'],
            'groups': groups, 'deployment': gate, 'stop_reason': stop_reason, 'test_used': False,
            'load_timings_ms': getattr(retriever, 'load_timings_ms', {}),
            'warmup_records': len(warmups.records), 'budget': budget.state(),
            'records': artifact(store.path) if store.path.exists() else None,
            'note': 'Measured host/device sampling is in individual records; no inferred layer offload.'}
        atomic_json(output/'SUMMARY.json', summary)
        return summary
