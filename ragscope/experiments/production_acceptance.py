"""Fixed production populations, independent of a runnable manifest's size.

No model calls, result filtering, manifest repair or statistical inference.
The generic campaign harness intentionally remains usable with tiny unit fixtures.
"""
from collections import Counter
import json
from pathlib import Path
import random

from ragscope.experiments.analysis_report import validated_rows
from ragscope.experiments.generation_campaign import load_campaign
from ragscope.indexing.identity import artifact, atomic_json, file_hash, signature, verify_artifacts

MAIN_MODELS = ('qwen_2b', 'qwen_4b', 'qwen_9b')
SMALL_MODELS = MAIN_MODELS[:2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def subset(rows, n):
    rows = sorted(rows, key=lambda row: row['query_id'])
    random.Random(42).shuffle(rows)
    return rows[:n]


def population(corpus, *, squad=False):
    """Read trusted caller-selected corpus, never a manifest-selected substitute."""
    corpus = Path(corpus).resolve()
    receipt = read_json(corpus/'SPLIT_MANIFEST.json')
    wanted = {'test': 100} if squad else {'train': 1000, 'dev': 100, 'test': 200}
    result = {}
    for name in ['documents.jsonl'] + [f'queries_{split}.jsonl' for split in wanted]:
        require(file_hash(corpus/name) == receipt['files'][name]['sha256'], 'Frozen corpus artifact changed: '+name)
    for split, count in wanted.items():
        rows = [json.loads(line) for line in (corpus/f'queries_{split}.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        ids = [row['query_id'] for row in rows]
        declared = receipt['selected_query_ids'] if squad else receipt['query_ids'][split]
        require(len(ids) == len(set(ids)) == count and len(declared) == len(set(declared)) == count
                and set(ids) == set(declared), f'Production {split} population must contain exactly {count} frozen unique queries')
        require(all(row['split'] == split for row in rows), 'Frozen query split differs')
        result[split] = rows
    if not squad:
        require(len({r['query_id'] for rows in result.values() for r in rows}) == 1300, 'Cross-split query overlap')
    return result, artifact(corpus/'SPLIT_MANIFEST.json')


def expected_conditions(manifest, queries, squad_queries, scope):
    """Prescribed model/method Cartesian products, never inferred from tasks."""
    expected = {}

    def add(experiment, methods, models, rows):
        ids = None if rows is None else {q['query_id'] for q in rows}
        for method in methods:
            for model in models:
                expected[(experiment, model, method)] = ids

    if scope in {'main','e2'}:
        add('E2', ('R0', 'R1', 'R2', 'R3'), MAIN_MODELS, queries['test'])
    if scope == 'e2':
        return expected
    add('E3', ('oracle', 'no_context'), MAIN_MODELS, queries['test'])
    if scope == 'e3':
        return expected
    add('E4', ('hybrid',), MAIN_MODELS, squad_queries)
    add('E5', ('bm25', 'hybrid'), ('lfm',), subset(queries['test'], 100))
    add('E6', ('simple', 'structured'), SMALL_MODELS, subset(queries['test'], 100))
    for split, n in [('train', 200), ('dev', 60)]:
        add('ROUTE_'+split.upper(), ('bm25', 'hybrid', 'hybrid_rerank'), ('qwen_2b',), subset(queries[split], n))
    audit = manifest['provenance'].get('readiness', {}).get('source_artifacts', {}).get('E7')
    if audit is None:
        # All positions may be explicitly blocked; each still keeps target40.
        add('E7', ('start', 'middle', 'end'), SMALL_MODELS, None)
    else:
        fixtures = audit['fixtures']
        ids = [row['query_id'] for row in fixtures]
        answerable = [q for q in queries['test'] if q['answerability'] == 'answerable']
        require(audit['target_n'] == 40 and audit['actual_n'] == len(ids) and 0 < len(ids) <= 40
                and len(ids) == len(set(ids)), 'E7 must preserve its declared up-to40 source-feasible population')
        require(set(ids) <= {q['query_id'] for q in answerable}, 'E7 has non-answerable or foreign queries')
        rejected = audit.get('excluded_before_generation', [])
        excluded = [row['query_id'] for row in rejected]
        require(len(excluded) == len(set(excluded)) and not set(ids) & set(excluded)
                and all(row.get('reason') for row in rejected), 'E7 missing or duplicate pre-generation exclusion evidence')
        if not manifest['provenance']['spec'].get('e7_inputs'):
            ordered = subset(answerable, len(answerable))
            observed = set(ids) | set(excluded)
            require(observed == {q['query_id'] for q in ordered[:len(observed)]}, 'E7 source-feasibility selection changed')
            require(len(ids) == 40 or len(observed) == len(answerable), 'E7 fewer than40 without exhausting declared feasibility population')
        add('E7', ('start', 'middle', 'end'), SMALL_MODELS, [q for q in answerable if q['query_id'] in ids])
    return expected


def validate_production_manifest(manifest, inputs, queries, squad_queries, scope):
    spec = manifest['provenance']['spec']
    require(spec['main_models'] == list(MAIN_MODELS), 'Production requires all three main models')
    if scope == 'main':
        require(spec['small_models'] == list(SMALL_MODELS) and spec['smallest_model'] == 'qwen_2b'
                and spec['heterogeneous_model'] == 'lfm', 'Extension model assignment changed')
    require(manifest['seed'] == 42 and manifest['generation_config'] == {'max_tokens': 512, 'temperature': 0.0}
            and manifest['parse_repair_attempts'] == 0, 'Frozen generation protocol changed')
    expected = expected_conditions(manifest, queries, squad_queries, scope)
    canonical = {q['query_id']: q for rows in queries.values() for q in rows}
    canonical.update({q['query_id']: q for q in squad_queries})
    declarations, rebuilt, referenced = {}, [], set()
    for group in manifest['groups']:
        require(group.get('status') in {'READY', 'BLOCKED'}, 'Frozen group must be READY or evidence-backed BLOCKED')
        require(group['model_ids'] and len(group['model_ids']) == len(set(group['model_ids'])), 'Missing or duplicated group model')
        for model in group['model_ids']:
            key = group['experiment_id'], model, group['method']
            require(key in expected and key not in declarations, 'Unexpected or duplicated production condition: '+str(key))
            ids = expected[key]
            expected_n = 40 if ids is None else len(ids)
            require(group.get('expected_n') == expected_n, 'Declared denominator differs from production population: '+str(key))
            declarations[key] = group
            if group['status'] == 'BLOCKED':
                require(bool(group.get('reason')) and bool(group.get('evidence')) and verify_artifacts(group['evidence']), 'Blocked condition lacks valid evidence: '+str(key))
                require(not group.get('input_ids'), 'Blocked condition also declares runnable inputs')
                continue
            require(ids is not None, 'Runnable E7 lacks frozen feasibility audit')
            require(model in manifest['models'], 'Runnable model identity is missing')
            input_ids = group['input_ids']
            require(len(input_ids) == len(set(input_ids)) == expected_n, 'Missing or duplicate input population: '+str(key))
            actual_ids = []
            for iid in input_ids:
                source = inputs[iid]
                query = source['query']
                qid = query['query_id']
                require(qid in ids and query == canonical[qid], 'Input query differs from canonical frozen corpus')
                require(iid == f'{key[0]}:{key[2]}:{qid}', 'Input ID changed from predeclared condition')
                actual_ids.append(qid); referenced.add(iid)
                task = dict(experiment_id=key[0], method=key[2], model_id=model,
                            input_id=iid, query_id=qid, input_hash=source['input_hash'])
                task['task_id'] = signature(task)
                rebuilt.append(task)
            require(set(actual_ids) == ids, 'Production condition omits canonical questions')
    require(set(declarations) == set(expected), 'Missing mandatory production model/method groups')
    require(set(inputs) == referenced, 'Unowned or missing frozen generation inputs')
    require(len(manifest['tasks']) == len(rebuilt) and Counter(signature(t) for t in manifest['tasks']) == Counter(signature(t) for t in rebuilt), 'Frozen tasks differ from full prescribed group Cartesian product')
    require(set(manifest['models']) <= {key[1] for key in expected}, 'Unexpected model in production campaign')
    return expected, declarations


def accept_campaign(directory, corpus, *, scope='main'):
    """Return truthful state; a complete shard never grants main completion."""
    require(scope in {'main', 'e2', 'e3'}, 'Unknown acceptance scope')
    directory, corpus = Path(directory).resolve(), Path(corpus).resolve()
    result = {'schema_version': 'production-acceptance-v1', 'scope': scope,
              'scope_limitation': scope.upper()+' shard only; not main production acceptance' if scope != 'main' else 'Complete prescribed generation matrix',
              'status': 'PENDING', 'integrity_valid': None, 'audit_generation_calls': 0,
              'required_core_n': {'e2':2400,'e3':1200,'main':3600}[scope], 'n_observed': 0, 'n_failed': 0}
    if not (directory/'EXPERIMENT_MANIFEST.json').is_file():
        result['reason'] = 'Frozen production manifest not present; prerequisites still pending'
        if (directory/'FREEZE_READINESS.json').is_file():
            result['readiness'] = artifact(directory/'FREEZE_READINESS.json')
        return result
    try:
        queries, corpus_receipt = population(corpus)
        squad, squad_receipt = population(corpus/'squad_e4', squad=True) if scope == 'main' else ({'test': []}, None)
        manifest, inputs = load_campaign(directory)
        require(manifest['provenance']['corpus_manifest'] == corpus_receipt, 'Campaign references a different trusted corpus')
        require(Path(manifest['provenance']['spec']['corpus_dir']).resolve() == corpus, 'Campaign corpus path changed')
        if scope == 'main':
            require(manifest['provenance']['readiness']['source_artifacts']['squad_split'] == squad_receipt, 'E4 frozen corpus receipt differs')
        expected, declarations = validate_production_manifest(manifest, inputs, queries, squad['test'], scope)
        rows, sources = validated_rows(directory, manifest, inputs)
        observed = Counter((r['experiment_id'], r['model_id'], r['method']) for r in rows)
        failures = Counter((r['experiment_id'], r['model_id'], r['method']) for r in rows if r.get('error'))
        conditions = []
        for key, ids in sorted(expected.items()):
            blocked = declarations[key]['status'] == 'BLOCKED'
            n = len(ids) if ids is not None else 40
            got = observed[key]
            require(got <= n and (not blocked or got == 0), 'Observed records exceed condition ownership')
            status = 'BLOCKED' if blocked else 'COMPLETE' if got == n else 'RUNNING' if got else 'PENDING'
            conditions.append({'experiment_id': key[0], 'model_id': key[1], 'method': key[2],
                'status': status, 'n_expected': n, 'n_observed': got, 'n_failed': failures[key],
                'n_missing': 0 if blocked else n-got, 'n_blocked': n if blocked else 0,
                **({'reason': declarations[key]['reason'], 'evidence': declarations[key]['evidence']} if blocked else {})})
        missing, blocked = sum(c['n_missing'] for c in conditions), sum(c['n_blocked'] for c in conditions)
        status = ('RUNNING' if rows else 'PENDING') if missing else 'BLOCKED' if blocked else 'COMPLETE'
        result.update(status=status, integrity_valid=True, manifest_hash=manifest['manifest_hash'],
            n_expected=sum(c['n_expected'] for c in conditions), n_observed=len(rows),
            n_failed=sum(bool(r.get('error')) for r in rows), n_missing=missing, n_blocked=blocked,
            conditions=conditions, sources=[artifact(directory/'EXPERIMENT_MANIFEST.json'),
                artifact(directory/'FROZEN_INPUTS.json'), corpus_receipt]+([squad_receipt] if squad_receipt else [])+sources)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        result.update(status='FAILED_VALIDATION', integrity_valid=False, reason='Production integrity/population validation failed',
                      validation_error=f'{type(exc).__name__}: {exc}')
    return result


def write_acceptance(directory, corpus, output, *, scope='main'):
    result = accept_campaign(directory, corpus, scope=scope)
    result['receipt_hash'] = signature(result)
    atomic_json(Path(output), result)
    return result
