"""Freeze synthetic scope tests and source-only comparison/summary inputs."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from filelock import FileLock
from ragscope.experiments.runtime import configure, require_ram
from ragscope.indexing.identity import artifact, atomic_json, canonical_json, signature
from ragscope.v3.splits import assert_inference_safe


def save_lines(path, rows):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(canonical_json(row)+'\n')
        handle.flush(); os.fsync(handle.fileno())
    return artifact(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    runtime = Path(configure(cpu_only=True)['home']).resolve()
    run = Path(args.campaign).resolve()
    if run.parent != runtime/'runs' or not run.name.startswith('v3_'):
        raise ValueError('Wrong campaign')
    require_ram(2*1024**3)
    # Own isolated source-preparation stream; no model or parent generation journal writes.
    with FileLock(str(run/'auxiliary_splits.lock'), timeout=0):
        receipt = run/'AUXILIARY_SPLIT_INPUTS.json'
        if receipt.exists():
            saved = json.loads(receipt.read_text('utf-8'))
            if not args.resume:
                raise ValueError('Existing frozen auxiliary inputs require explicit resume')
            if any(artifact(ref['path']) != ref for ref in saved['outputs']):
                raise ValueError('Frozen auxiliary input changed')
            if artifact(saved['parent_split_manifest']['path']) != saved['parent_split_manifest']:
                raise ValueError('Parent primary split changed')
            parent = json.loads(Path(saved['parent_split_manifest']['path']).read_text('utf-8'))
            reserved = saved['reserved_paper_ids']
            forbidden = {paper for split in parent['splits'].values() for paper in split['doc_ids']}
            if len(reserved) != 50 or len(set(reserved)) != 50 or set(reserved) & forbidden:
                raise ValueError('Auxiliary papers overlap primary splits')
            scope_rows = [json.loads(l) for l in (run/'splits/H_scope.questions.jsonl').read_text('utf-8').splitlines()]
            source_rows = [json.loads(l) for l in (run/'H_tasks.source_documents.jsonl').read_text('utf-8').splitlines()]
            specified = [r for r in scope_rows if r['scope']['kind'] == 'document']
            ambiguous = [r for r in scope_rows if r['scope']['kind'] == 'global']
            if len(scope_rows) != 40 or len(specified) != 20 or len(ambiguous) != 20:
                raise ValueError('Scope test population changed')
            if {r['scope']['doc_ids'][0] for r in specified} != set(reserved[:20]):
                raise ValueError('Specified paper scope differs')
            if any(r['user_provided_title'] is not None or 'doc_ids' in r['scope'] for r in ambiguous):
                raise ValueError('Ambiguous input contains hidden paper anchor')
            if len(source_rows) != 30 or {r['source_id'] for r in source_rows} != set(reserved[20:]):
                raise ValueError('Task source paper population changed')
            result = {'status': 'PASS_INDEPENDENT_SCHEMA_CORRECT_VALIDATION', 'runner': artifact(__file__),
                'original_receipt': artifact(receipt), 'primary_doc_n': len(forbidden),
                'reserved_doc_n': 50, 'primary_overlap_n': len(set(reserved) & forbidden),
                'scope_n': 40, 'task_source_n': 30, 'outputs': saved['outputs'],
                'construction_v1_guard_field_error': 'paper_ids default was empty; actual manifest field is doc_ids',
                'human_review_status': 'NOT_REVIEWED', 'new_llm_requests': 0}
            atomic_json(run/'AUXILIARY_SPLIT_VALIDATION.json', result)
            print(json.dumps({k: v for k, v in result.items() if k not in ('runner', 'original_receipt', 'outputs')})); return 0
        parent_path = run/'SPLITS_V3.json'
        parent = json.loads(parent_path.read_text('utf-8'))
        reserved = parent['reserved_task_scope_papers']
        if len(reserved) != 50 or len(set(reserved)) != 50:
            raise ValueError('Expected 50 previously reserved disjoint papers')
        source_path = runtime/'corpora/20261014_v2/all_documents.jsonl'
        documents = {}
        with source_path.open(encoding='utf-8') as handle:
            for line in handle:
                doc = json.loads(line)
                if doc['doc_id'] in reserved:
                    documents[doc['doc_id']] = doc
        if set(documents) != set(reserved):
            raise ValueError('Reserved source papers missing')
        forbidden = {paper for split in parent['splits'].values() for paper in split['doc_ids']}
        if set(reserved) & forbidden:
            raise ValueError('Reserved paper overlaps primary splits')
        scope_questions, scope_gold = [], []
        for i, doc_id in enumerate(reserved[:20]):
            doc = documents[doc_id]
            for specified in (True, False):
                qid = 'H_SCOPE_'+signature([doc_id, specified])[:20]
                row = {'query_id': qid, 'question': 'What datasets did they use?',
                    'scope': {'kind': 'document', 'doc_ids': [doc_id]} if specified else {'kind': 'global'},
                    'user_provided_title': doc['title'] if specified else None,
                    'authorship': 'SYNTHETIC_RULE_CONSTRUCTION', 'human_review_status': 'NOT_REVIEWED'}
                assert_inference_safe(row)
                scope_questions.append(row)
                scope_gold.append({'query_id': qid, 'expected_scope_action': 'USE_SPECIFIED_DOCUMENT' if specified else 'ASK_FOR_DOCUMENT',
                    'construction_reason': 'explicit user-selected document' if specified else 'unanchored they with no title or selected paper over many papers',
                    'not_official_qasper': True, 'human_review_status': 'NOT_REVIEWED'})
        task_sources = []
        for doc_id in reserved[20:]:
            doc = documents[doc_id]
            # The teacher will choose auditable spans from this whole source; not QASPER gold.
            task_sources.append({'source_id': doc_id, 'title': doc['title'], 'text': doc['text'],
                'canonical_checksum': doc['canonical_checksum'], 'sections': doc['sections'],
                'source_revision': doc['source_revision'], 'language': doc['language'],
                'authorship': 'OFFICIAL_SOURCE_TEXT_ONLY_NO_QUESTION_ANSWERS'})
        evaluator = run/'splits/evaluator_only'
        evaluator.mkdir(parents=True, exist_ok=True)
        outputs = [save_lines(run/'splits/H_scope.questions.jsonl', scope_questions),
                   save_lines(evaluator/'H_scope.gold.jsonl', scope_gold),
                   save_lines(run/'H_tasks.source_documents.jsonl', task_sources)]
        result = {'status': 'PASS_SCOPE40_TASK_SOURCE30_FROZEN', 'runner': artifact(__file__),
            'parent_split_manifest': artifact(parent_path), 'source_documents': artifact(source_path),
            'scope_n': 40, 'scope_specified_n': 20, 'scope_ambiguous_n': 20, 'task_source_papers': 30,
            'paper_disjointness_checked': True, 'reserved_paper_ids': reserved, 'outputs': outputs,
            'H_tasks_status': 'PENDING_SOURCE_BACKED_REFERENCE_GENERATION',
            'H_scope_authorship': 'SYNTHETIC_RULE_NOT_OFFICIAL_QASPER',
            'operator_blindness': 'NOT_OPERATOR_BLIND; primary H_QA remains sealed',
            'human_review_status': 'NOT_REVIEWED', 'new_llm_requests': 0}
        atomic_json(receipt, result)
        print(json.dumps({k: v for k, v in result.items() if k not in ('outputs', 'reserved_paper_ids', 'source_documents', 'parent_split_manifest', 'runner')}))
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
