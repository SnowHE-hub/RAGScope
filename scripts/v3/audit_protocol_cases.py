"""Diagnose all old parse failures and empty-context schema semantics on CPU."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from filelock import FileLock
from ragscope.experiments.runtime import configure
from ragscope.generation.prompts import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE, visible_output_schema
from ragscope.indexing.identity import artifact, atomic_json, canonical_json
from ragscope.v3.protocol_diagnostics import recover_envelope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    runtime = Path(configure(cpu_only=True)['home']).resolve()
    run = Path(args.campaign).resolve()
    if not run.is_relative_to(runtime / 'runs') or not run.name.startswith('v3_'):
        raise ValueError('Wrong campaign')
    with FileLock(str(run / 'writer.lock'), timeout=0):
        output = run / 'PROTOCOL_CASE_AUDIT.json'
        if output.exists():
            saved = json.loads(output.read_text('utf-8'))
            if not args.resume or saved['runner'] != artifact(__file__):
                raise ValueError('Protocol audit identity changed')
            print(json.dumps({'status': 'REUSED_PROTOCOL_AUDIT'}))
            return 0
        db = sqlite3.connect(run / 'S0_EVIDENCE.sqlite')
        failures = []
        for (payload,) in db.execute("SELECT payload FROM records WHERE json_extract(payload,'$.failed')=1"):
            row = json.loads(payload)
            recovered = recover_envelope(row['raw_output'] or '')
            failures.append({'record_id': row['record_id'], 'query_id': row['query_id'],
                'model_id': row['model_id'], 'experiment_id': row['experiment_id'], 'method': row['method'],
                'source': row['source'], 'source_record_hash': row['source_record_hash'],
                'original_error': row['error'], 'original_failure_kept_in_primary_denominator': True,
                'recovery': recovered, 'interpretation': 'Envelope diagnosis, never original successful model output'})
        db.close()
        detail = run / 'PARSE_FAILURE_ENVELOPE_DIAGNOSTICS.jsonl'
        with detail.open('x', encoding='utf-8', newline='\n') as handle:
            for row in failures:
                handle.write(canonical_json(row) + '\n')
        empty = visible_output_schema([])
        result = {'status': 'PASS_DIAGNOSIS_NOT_NEW_GENERATION', 'runner': artifact(__file__),
            'implementation': artifact(PROJECT / 'ragscope/v3/protocol_diagnostics.py'),
            'failed_n': len(failures), 'lfm_failed_n': sum(r['model_id']=='lfm' for r in failures),
            'recovery_status_counts': Counter(r['recovery']['status'] for r in failures),
            'detail': artifact(detail), 'system_prompt': RAG_SYSTEM_PROMPT, 'user_template': RAG_USER_TEMPLATE,
            'empty_context_schema': empty, 'empty_schema_forces_abstain': False,
            'empty_schema_explanation': 'citations maxItems=0, no minItems; abstain remains an unconstrained boolean. The system prompt restricts answering to supplied evidence.',
            'method_mapping': {'R0':'bm25','R1':'dense','R2':'hybrid','R3':'hybrid_rerank'},
            'mapping_source': artifact(PROJECT / 'scripts/freeze_generation_v2.py'),
            'original_raw_records_modified': False, 'new_llm_requests': 0, 'human_review_status':'NOT_REVIEWED'}
        atomic_json(output, result)
        print(json.dumps({k: result[k] for k in ['status','failed_n','lfm_failed_n','recovery_status_counts','empty_schema_forces_abstain','new_llm_requests']}))
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
