"""Frozen paper-disjoint splits and an explicit evaluator-only gold boundary."""
from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path

from ragscope.indexing.identity import artifact, signature

GOLD_FIELDS = {'answer', 'answer_refs', 'answerability', 'evidence_reference_sets',
               'gold_doc_id', 'gold_doc_id_evaluation_only', 'annotations', 'scores',
               'visible_context_sufficient', 'reference', 'references'}


def assert_inference_safe(value):
    if isinstance(value, dict):
        prohibited = set(value) & GOLD_FIELDS
        if prohibited:
            raise ValueError('Gold/evaluation field in inference input: ' + ','.join(sorted(prohibited)))
        for item in value.values():
            assert_inference_safe(item)
    elif isinstance(value, list):
        for item in value:
            assert_inference_safe(item)


def public_question(query):
    output = {'query_id': query['query_id'], 'question': query['question'],
              'scope': {'kind': 'document', 'doc_ids': [query['doc_id']]}}
    assert_inference_safe(output)
    return output


def select_paper_questions(queries, count, minimum_papers, seed, namespace):
    """Select by deterministic IDs, not answer difficulty, scores, or candidates."""
    grouped = defaultdict(list)
    for query in queries:
        grouped[query['doc_id']].append(query)
    papers = sorted(grouped, key=lambda p: signature([seed, namespace, p]))
    if len(papers) < minimum_papers:
        raise ValueError('Not enough eligible unseen papers')
    initial = min(len(papers), max(minimum_papers, count // 2))
    chosen = papers[:initial]
    while sum(len(grouped[p]) for p in chosen) < count and len(chosen) < len(papers):
        chosen.append(papers[len(chosen)])
    for paper in chosen:
        grouped[paper].sort(key=lambda q: signature([seed, namespace, q['query_id']]))
    result = []
    depth = 0
    while len(result) < count:
        before = len(result)
        for paper in chosen:
            if depth < len(grouped[paper]):
                result.append(grouped[paper][depth])
                if len(result) == count:
                    break
        if len(result) == before:
            raise ValueError('Not enough eligible questions')
        depth += 1
    if len({q['doc_id'] for q in result}) < minimum_papers:
        raise ValueError('Paper coverage requirement not met')
    return result


def read_split(run, split, *, purpose='inference'):
    """Inference always gets public questions. Holdout gold requires final lock."""
    run = Path(run)
    manifest = json.loads((run / 'SPLITS_V3.json').read_text('utf-8'))
    if signature({k: v for k, v in manifest.items() if k != 'manifest_hash'}) != manifest['manifest_hash']:
        raise ValueError('Split manifest hash differs')
    entry = manifest['splits'][split]
    if purpose == 'inference':
        ref = entry['questions']
    elif purpose in {'diagnosis', 'development', 'training', 'final_evaluation'}:
        if entry['role'] == 'sealed_test':
            if purpose != 'final_evaluation':
                raise PermissionError('Holdout gold is forbidden to training/selection/diagnosis')
            lock_path, authorization_path = run / 'CANDIDATE_LOCK.json', run / 'FINAL_EVALUATION_AUTHORIZATION.json'
            if not lock_path.is_file() or not authorization_path.is_file():
                raise PermissionError('Holdout gold needs candidate lock and one-round authorization')
            lock = json.loads(lock_path.read_text('utf-8'))
            auth = json.loads(authorization_path.read_text('utf-8'))
            if (lock.get('splits_manifest_hash') != manifest['manifest_hash'] or
                    auth.get('candidate_lock_sha256') != artifact(lock_path)['sha256'] or
                    auth.get('evaluation_round') != 1):
                raise PermissionError('Holdout final authorization differs from frozen candidates/splits')
        ref = entry['gold']
    else:
        raise ValueError('Unknown split access purpose')
    if artifact(ref['path']) != ref:
        raise ValueError('Frozen split contents changed')
    with Path(ref['path']).open(encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle]
    if purpose == 'inference':
        for row in rows:
            assert_inference_safe(row)
    return rows
