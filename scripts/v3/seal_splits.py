"""Seal new QASPER papers before v3 candidate selection; never print gold."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from filelock import FileLock
from ragscope.experiments.runtime import configure, require_ram
from ragscope.indexing.identity import artifact, atomic_json, canonical_json, signature
from ragscope.v3.splits import public_question, select_paper_questions


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            yield json.loads(line)


def identities(value, docs, queries):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'doc_id' and isinstance(item, str) and item.startswith('qasper:'):
                docs.add(item)
            elif key == 'doc_ids' and isinstance(item, list):
                docs.update(i for i in item if isinstance(i, str) and i.startswith('qasper:'))
            elif key == 'query_id' and isinstance(item, str):
                queries.add(item)
            elif isinstance(item, (dict, list)):
                identities(item, docs, queries)
    elif isinstance(value, list):
        for item in value:
            identities(item, docs, queries)


def write_jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        for row in values:
            handle.write(canonical_json(row) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    runtime = Path(configure(cpu_only=True)['home']).resolve()
    run = Path(args.campaign).resolve()
    if not run.is_relative_to(runtime / 'runs') or not run.name.startswith('v3_'):
        raise ValueError('Wrong campaign directory')
    require_ram(2 * 1024**3)
    identity = json.loads((run / 'CAMPAIGN_IDENTITY.json').read_text('utf-8'))
    if Path(identity['project']).resolve() != PROJECT:
        raise ValueError('Campaign repository mismatch')
    output = run / 'SPLITS_V3.json'
    with FileLock(str(run / 'writer.lock'), timeout=0):
        if output.exists():
            if not args.resume:
                raise FileExistsError('Sealed splits are immutable')
            saved = json.loads(output.read_text('utf-8'))
            for split in saved['splits'].values():
                for key in ('questions', 'gold'):
                    if artifact(split[key]['path']) != split[key]:
                        raise ValueError('Sealed split changed')
            print(json.dumps({'status': 'REUSED_SEALED_SPLITS', 'new_llm_requests': 0}))
            return 0
        corpus = runtime / 'corpora/20261014_v2'
        exposed, exposed_queries = set(), set()
        reasons, evidence = defaultdict(list), []
        for row in read_jsonl(corpus / 'documents.jsonl'):
            exposed.add(row['doc_id'])
            reasons[row['doc_id']].append('conservative exclusion: every v2 indexed/selected paper')
        # This exact first-source annotation was displayed during schema inspection.
        # It must be excluded even if it was not selected by v2.
        first_query = next(read_jsonl(corpus / 'all_queries_audit.jsonl'))
        exposed.add(first_query['doc_id'])
        exposed_queries.add(first_query['query_id'])
        reasons[first_query['doc_id']].append('v3 operator schema read exposed source line 1')
        sources = sorted({p for directory in (runtime / 'runs').glob('20261014_v2*')
                          if directory.is_dir() for p in directory.rglob('*.jsonl')})
        for path in sources:
            docs, qids = set(), set()
            for row in read_jsonl(path):
                identities(row, docs, qids)
            exposed.update(docs)
            exposed_queries.update(qids)
            for doc in docs:
                reasons[doc].append(str(path))
            evidence.append({'source': artifact(path), 'qasper_document_ids_found': len(docs), 'question_ids_found': len(qids)})
        # Equal titles or equal canonical texts never cross the new split boundary.
        titles, canonical, docs_meta = defaultdict(set), defaultdict(set), {}
        for row in read_jsonl(corpus / 'all_documents.jsonl'):
            doc = row['doc_id']
            title_key = re.sub(r'\W+', '', row['title'].casefold())
            titles[title_key].add(doc)
            canonical[row['canonical_checksum']].add(doc)
            docs_meta[doc] = {'title': row['title'], 'source_split': row['split'],
                              'canonical_checksum': row['canonical_checksum']}
        duplicate_docs = set()
        for group in list(titles.values()) + list(canonical.values()):
            if len(group) > 1:
                duplicate_docs.update(group)
        # Duplicate aliases are conservatively excluded, not assigned using gold.
        exposed.update(duplicate_docs)
        for doc in duplicate_docs:
            reasons[doc].append('same title or canonical text; conservatively exclude all aliases')
        eligible = []
        exclusions = Counter()
        for query in read_jsonl(corpus / 'all_queries_audit.jsonl'):
            reason = None
            if query['doc_id'] in exposed or query['query_id'] in exposed_queries:
                reason = 'prior_project_or_operator_exposure'
            elif query.get('metadata', {}).get('quarantine_reasons'):
                reason = 'text_annotation_quarantine'
            elif query['doc_id'] not in docs_meta:
                reason = 'missing_document'
            if reason:
                exclusions[reason] += 1
            else:
                eligible.append(query)
        hqa = select_paper_questions(eligible, 200, 70, 42, 'H_QA')
        hdocs = {q['doc_id'] for q in hqa}
        confirm = select_paper_questions([q for q in eligible if q['doc_id'] not in hdocs], 100, 40, 42, 'D_confirm')
        ddocs = {q['doc_id'] for q in confirm}
        reserve = sorted({q['doc_id'] for q in eligible} - hdocs - ddocs, key=lambda d: signature([42, 'task_scope_reserve', d]))[:50]
        db = sqlite3.connect(run / 'S0_EVIDENCE.sqlite')
        old_queries = [json.loads(row[0]) for row in db.execute('SELECT payload FROM queries')]
        db.close()
        pool = [q for q in old_queries if q['query_id'].startswith('qasper:')]
        lora_ids = {r['query_id'] for r in read_jsonl(run / 'LORA_100_PAIRS.jsonl') if r['group'] == 'NEW_ABSTENTION'}
        search = [q for q in pool if q['query_id'] in lora_ids]
        used = {q['query_id'] for q in search}
        buckets = defaultdict(list)
        for q in pool:
            if q['query_id'] not in used:
                types = sorted({a.get('answer_type', 'unknown') for a in q.get('metadata', {}).get('annotations', [])})
                buckets['unanswerable' if q['answerability'] == 'unanswerable' else '+'.join(types)].append(q)
        for values in buckets.values():
            values.sort(key=lambda q: signature([42, 'D_search', q['query_id']]))
        while len(search) < 120 and any(buckets.values()):
            for key in sorted(buckets):
                if buckets[key] and len(search) < 120:
                    search.append(buckets[key].pop(0))
        for q in search:
            if 'doc_id' not in q:
                q['doc_id'] = q['scope']['doc_ids'][0]
        manifest = {'schema_version': 1, 'campaign_id': run.name, 'seed': 42,
            'created_utc': datetime.now(timezone.utc).isoformat(), 'splits': {},
            'eligible_question_n': len(eligible), 'eligible_paper_n': len({q['doc_id'] for q in eligible}),
            'excluded_query_counts': exclusions, 'exposed_document_n': len(exposed),
            'exposure_sources': evidence, 'duplicate_screen': 'exact canonical hash and normalized title; conservative alias exclusion',
            'H_scope': {'status': 'PENDING_CONSTRUCTION', 'target_n': 40, 'authorship': 'SYNTHETIC_AI_NOT_HUMAN_REVIEWED'},
            'H_tasks': {'status': 'PENDING_CONSTRUCTION', 'maximum_n': 60, 'authorship': 'SYNTHETIC_AI_NOT_HUMAN_REVIEWED'},
            'reserved_task_scope_papers': reserve, 'human_review_status': 'NOT_REVIEWED',
            'exposure_scope': 'current local project artifacts and known tool reads; no access to other chats',
            'sealer_source': artifact(__file__), 'boundary_source': artifact(PROJECT / 'ragscope/v3/splits.py')}
        for name, queries, role in [('D_search', search, 'EXPOSED_DIAGNOSTIC'),
                                     ('D_confirm', confirm, 'development'), ('H_QA', hqa, 'sealed_test')]:
            public_path = run / 'splits' / (name + '.questions.jsonl')
            gold_path = run / 'splits/evaluator_only' / (name + '.gold.jsonl')
            write_jsonl(public_path, (public_question(q) for q in queries))
            write_jsonl(gold_path, queries)
            manifest['splits'][name] = {'n': len(queries), 'paper_n': len({q['doc_id'] for q in queries}),
                'query_ids': [q['query_id'] for q in queries], 'doc_ids': sorted({q['doc_id'] for q in queries}),
                'answerability_counts': Counter(q['answerability'] for q in queries), 'role': role,
                'questions': artifact(public_path), 'gold': artifact(gold_path)}
        if hdocs & ddocs or (hdocs | ddocs) & exposed:
            raise ValueError('Paper-disjoint holdout invariant failed')
        write_jsonl(run / 'EXPOSURE_LEDGER.jsonl', ({'doc_id': d, 'reasons': reasons[d],
            'policy': 'exclude entire paper from new D_confirm/H_QA'} for d in sorted(exposed)))
        manifest['exposure_ledger'] = artifact(run / 'EXPOSURE_LEDGER.jsonl')
        manifest['manifest_hash'] = signature(manifest)
        atomic_json(output, manifest)
        state = json.loads((run / 'STATUS.json').read_text('utf-8'))
        state.update(split_manifest=artifact(output), next_command_status='IMPLEMENT_NEXT',
            next_command='Implement S1 old-output metric audit/calibration; keep H_QA gold sealed until frozen final evaluator')
        atomic_json(run / 'STATUS.json', state)
        atomic_json(PROJECT / 'docs/execution/v3/STATUS.json', state)
        print(json.dumps({'status': 'SEALED', 'splits': {k: {'n': v['n'], 'papers': v['paper_n'],
            'answerability_counts': v['answerability_counts']} for k, v in manifest['splits'].items()},
            'eligible_papers': manifest['eligible_paper_n'], 'excluded_exposure_papers': len(exposed),
            'source_journals_scanned': len(sources), 'new_llm_requests': 0}))
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
