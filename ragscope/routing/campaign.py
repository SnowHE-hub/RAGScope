"""Bind router experiments to frozen, counted real generation records."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import random

from ragscope.experiments.analysis_report import validated_rows
from ragscope.experiments.generation_campaign import load_campaign
from ragscope.indexing.identity import artifact, atomic_json, signature
from .core import ROUTES, RoutingRequest, build_proxy_labels, proxy_quality
from .evaluate import summarize

METHODS = {'R0': 'bm25', 'R2': 'hybrid', 'R3': 'hybrid_rerank'}
REQUIRED_ROUTERS = ('fixed_bm25', 'fixed_hybrid', 'fixed_hybrid_rerank', 'rules', 'linear', 'mlp', 'lstm')


def immutable_json(path, body):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != body:
            raise ValueError(f'Frozen routing artifact changed: {path.name}')
    else:
        atomic_json(path, body)


def request_query(source):
    """Visible metadata is derived only from the selected source document."""
    query = source['query']
    visible = RoutingRequest.from_query(query)
    length = source.get('document_length', {}).get('reference_n_tokens')
    if length is not None and query.get('scope', {}).get('kind') == 'document':
        visible.metadata.update(document_length_tokens=length, document_type='paper', language='en')
    return {'query_id': query['query_id'], 'split': query['split'],
            'question': visible.question, 'scope': visible.scope, 'metadata': visible.metadata}


def freeze_routing(campaign, output, *, model_id='qwen_2b', counts=None):
    """Read inputs, never outputs: use the manifest's train/dev IDs verbatim."""
    counts = counts or {'train': 200, 'dev': 60, 'test': 100}
    manifest, inputs = load_campaign(campaign)
    by_split = {}
    for split, experiment in [('train', 'ROUTE_TRAIN'), ('dev', 'ROUTE_DEV'), ('test', 'E2')]:
        tasks = [t for t in manifest['tasks'] if t['model_id'] == model_id and t['experiment_id'] == experiment]
        methods = ROUTES if split != 'test' else tuple(METHODS)
        members = [{t['query_id'] for t in tasks if t['method'] == method} for method in methods]
        if not members[0] or any(s != members[0] for s in members):
            raise ValueError(f'Missing aligned routing input paths for {split}')
        ids = sorted(members[0])
        if split == 'test':
            random.Random(manifest['seed']).shuffle(ids)
            ids = ids[:counts[split]]
        if len(ids) != counts[split]:
            raise ValueError(f'Frozen routing population differs: {split}')
        queries = {}
        for task in tasks:
            if task['query_id'] not in ids:
                continue
            source = inputs[task['input_id']]
            if source['query']['split'] != split:
                raise ValueError('Routing population split differs')
            query = request_query(source)
            if task['query_id'] in queries and queries[task['query_id']] != query:
                raise ValueError('Visible query metadata differs across routes')
            queries[task['query_id']] = query
        by_split[split] = {'query_ids': ids, 'visible_queries': [queries[qid] for qid in ids]}
    comparison = {'manifest_hash': manifest['manifest_hash'], 'model_id': model_id,
        'model_identity': manifest['models'][model_id], 'generation': manifest['generation_config'],
        'quality_policy': 'answerable_token_f1;unanswerable_correct_abstention_not_f1',
        'cost_policy': 'recorded_retrieval_total_plus_recorded_generation_total_replay_only'}
    protocol = {'version': 'routing-campaign-v2', 'comparison': comparison,
        'comparison_config_hash': signature(comparison), 'populations': by_split,
        'seed': manifest['seed'], 'source': artifact(Path(campaign)/'EXPERIMENT_MANIFEST.json'),
        'label_tolerance': .01, 'online_dev_n': 30,
        'deployment': 'Answerable dev F1 within .01 and actual online dev P50 >=10% faster; otherwise fixed.'}
    protocol['protocol_hash'] = signature(protocol)
    immutable_json(Path(output)/'ROUTING_MANIFEST.json', protocol)
    return protocol


def normalized_record(row, source, comparison_hash):
    scores = row['scores']
    f1 = scores['token_f1']
    unanswerable = source['query']['answerability'] == 'unanswerable'
    retrieval_ms = source.get('retrieval_timings', {}).get('total_ms')
    generation_ms = row.get('timings', {}).get('total_ms')
    if retrieval_ms is None or generation_ms is None or retrieval_ms < 0 or generation_ms < 0:
        raise ValueError('Router cost proxy requires actually measured retrieval and generation timing')
    record = {'query_id': row['query_id'], 'split': row['split'],
        'path': METHODS.get(row['method'], row['method']),
        'answer_f1': f1, 'answerability': 'unanswerable' if unanswerable else 'answerable',
        'correct_abstention': float(not row.get('error') and scores['predicted_abstain']) if unanswerable else None,
        'wrong_abstain': bool(scores['wrong_abstention']),
        'evidence_coverage': scores['context_evidence']['continuous_coverage'],
        'total_latency_s': (retrieval_ms+generation_ms)/1000, 'latency_kind': 'component_replay_estimate',
        'retrieval_ms': retrieval_ms, 'generation_ms': generation_ms,
        'error': row.get('error'), 'real_model': row['real_model'],
        'comparison_config_hash': comparison_hash, 'source_record_hash': row['record_hash'],
        'source_task_id': row['task_id'], 'source_call_ticket': row['call_ticket'],
        'context_hash': row['context_bundle']['context_hash']}
    proxy_quality(record)
    return record


def route_record_artifacts(campaign, validated_sources, experiment, model_id):
    """Use the audited journal owner, including separately frozen E2 campaigns."""
    name = f'{experiment}_{model_id}.jsonl'
    matches = [item for item in validated_sources if Path(item['path']).name == name]
    if len(matches) != 1:
        raise ValueError(f'Missing or ambiguous validated routing journal owner: {name}')
    journal = matches[0]
    path = Path(journal['path']).resolve()
    by_path = {Path(item['path']).resolve(): item for item in validated_sources}
    config_path = path.with_suffix('.config.json')
    if config_path not in by_path:
        raise ValueError('Validated routing journal is missing its config provenance')
    result = [journal, by_path[config_path]]
    owner = path.parent.parent
    if owner != Path(campaign).resolve():
        for filename in ('EXPERIMENT_MANIFEST.json', 'FROZEN_INPUTS.json'):
            if owner/filename not in by_path:
                raise ValueError('External routing journal is missing frozen owner provenance')
            result.append(by_path[owner/filename])
    return result


def load_route_records(campaign, protocol, splits):
    splits = tuple(splits)
    manifest, inputs = load_campaign(campaign)
    if manifest['manifest_hash'] != protocol['comparison']['manifest_hash']:
        raise ValueError('Routing source campaign changed')
    model_id = protocol['comparison']['model_id']
    experiments = {'train': 'ROUTE_TRAIN', 'dev': 'ROUTE_DEV', 'test': 'E2'}
    rows, validated_sources = validated_rows(campaign, manifest, inputs,
                                             experiments={experiments[split] for split in splits})
    out, sources = [], []
    for split in splits:
        wanted = set(protocol['populations'][split]['query_ids'])
        selected = [r for r in rows if r['model_id'] == model_id and r['experiment_id'] == experiments[split]
                    and r['query_id'] in wanted and METHODS.get(r['method'], r['method']) in ROUTES]
        pairs = Counter((r['query_id'], METHODS.get(r['method'], r['method'])) for r in selected)
        if pairs != Counter({(qid, route): 1 for qid in wanted for route in ROUTES}):
            raise ValueError(f'Routing {split} actual generation incomplete; expected {len(wanted)*3}, observed {len(selected)}')
        out.extend(normalized_record(r, inputs[r['input_id']], protocol['comparison_config_hash']) for r in selected)
        sources.extend(route_record_artifacts(campaign, validated_sources, experiments[split], model_id))
    return out, sources


def prepare_labels(campaign, output):
    output = Path(output)
    protocol = freeze_routing(campaign, output)
    records, sources = load_route_records(campaign, protocol, ('train', 'dev'))
    manifest, inputs = load_campaign(campaign)
    query_data = {source['query']['query_id']: source['query'] for source in inputs.values()}
    labels = []
    for split in ('train', 'dev'):
        pop = protocol['populations'][split]
        queries = [{**q, 'answerability': query_data[q['query_id']]['answerability']} for q in pop['visible_queries']]
        labels.extend(build_proxy_labels(queries, records, split=split, query_ids=pop['query_ids'],
                      expected_config_hash=protocol['comparison_config_hash'], tolerance=protocol['label_tolerance']))
    path = output/'labels.jsonl'
    text = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True)+'\n' for row in labels)
    if path.exists() and path.read_text(encoding='utf-8') != text:
        raise ValueError('Existing labels differ from frozen real records')
    if not path.exists():
        path.write_text(text, encoding='utf-8')
    immutable_json(output/'LABEL_PROVENANCE.json', {'protocol_hash': protocol['protocol_hash'],
        'labels': artifact(path), 'source_records': sources,
        'counts': dict(Counter(row['split'] for row in labels)),
        'quality_policy': protocol['comparison']['quality_policy']})
    return labels


def validate_decisions(saved, protocol, name):
    """A valid hash cannot substitute for the complete frozen population."""
    body = {key: value for key, value in saved.items() if key != 'decision_hash'}
    if (signature(body) != saved.get('decision_hash') or saved.get('name') != name
            or saved.get('protocol_hash') != protocol['protocol_hash']):
        raise ValueError('Decision population or identity changed')
    if set(saved.get('decisions', {})) != {'dev', 'test'}:
        raise ValueError('Decision population must contain dev and test')
    for split in ('dev', 'test'):
        population = protocol['populations'][split]
        expected = population['query_ids']
        visible = [query['query_id'] for query in population['visible_queries']]
        observed = [item['query_id'] for item in saved['decisions'][split]]
        if (len(expected) != len(set(expected)) or Counter(visible) != Counter(expected)
                or Counter(observed) != Counter(expected)):
            raise ValueError(f'Decision population differs from frozen {split} IDs')
        for item in saved['decisions'][split]:
            decision = item['decision']
            elapsed = decision.get('elapsed_s')
            if (decision.get('route') not in ROUTES or type(decision.get('fallback')) is not bool
                    or type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0):
                raise ValueError('Decision route, fallback or measured duration is invalid')
    return saved


def freeze_decisions(router, protocol, output, name):
    """No test generation metrics are loaded before decisions are committed."""
    path = Path(output)/f'DECISIONS_{name}.json'
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        return validate_decisions(saved, protocol, name)
    decisions = {split: [{'query_id': query['query_id'], 'decision': router.choose(RoutingRequest.from_query(query)).to_dict()}
                 for query in protocol['populations'][split]['visible_queries']] for split in ('dev', 'test')}
    body = {'name': name, 'protocol_hash': protocol['protocol_hash'], 'decisions': decisions,
            'router_identity': getattr(router, 'identity', getattr(router, 'report', {}))}
    body['decision_hash'] = signature(body)
    validate_decisions(body, protocol, name)
    atomic_json(path, body)
    return body


def evaluate_decisions(campaign, protocol, decisions, output):
    # Freeze every mandatory policy's test choices before looking at test route
    # scores; freezing only the current policy is insufficient for blind analysis.
    frozen = {}
    for name in REQUIRED_ROUTERS:
        path = Path(output)/f'DECISIONS_{name}.json'
        if not path.exists():
            raise ValueError(f'All mandatory policy decisions must be frozen before test analysis: {name}')
        saved = json.loads(path.read_text(encoding='utf-8'))
        frozen[name] = validate_decisions(saved, protocol, name)
    name = decisions['name']
    validate_decisions(decisions, protocol, name)
    path = Path(output)/f'DECISIONS_{name}.json'
    stored = frozen.get(name)
    if stored is None:
        stored = validate_decisions(json.loads(path.read_text(encoding='utf-8')), protocol, name)
    if stored != decisions:
        raise ValueError('Replay decisions differ from the frozen artifact')
    records, sources = load_route_records(campaign, protocol, ('dev', 'test'))
    by_key = {(r['query_id'], r['path']): r for r in records}
    groups = {}
    for split in ('dev', 'test'):
        rows = []
        for item in decisions['decisions'][split]:
            decision = item['decision']
            record = by_key[(item['query_id'], decision['route'])]
            rows.append({**record, 'selected_route': decision['route'], 'fallback': decision['fallback'],
                'latency_s': record['total_latency_s']+decision['elapsed_s'], 'router_decision': decision})
        groups[split] = {'summary': summarize(rows), 'rows': rows}
    result = {'name': decisions['name'], 'protocol_hash': protocol['protocol_hash'],
        'groups': groups, 'source_records': sources, 'deployment': 'PENDING_ACTUAL_ONLINE_DEV_VALIDATION'}
    immutable_json(Path(output)/f'REPLAY_{decisions["name"]}.json', result)
    return result
