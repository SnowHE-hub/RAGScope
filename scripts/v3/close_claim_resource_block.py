"""Close exhausted NLI attempts, retain coverage/unknowns, export blind review."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from filelock import FileLock
from ragscope.experiments.ledger import read_records
from ragscope.indexing.identity import artifact, atomic_json, canonical_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    args = parser.parse_args()
    run = Path(args.campaign).resolve()
    if run.parent != Path('D:/RAGScope_Runtime/runs').resolve() or not run.name.startswith('v3_'):
        raise ValueError('Wrong campaign')
    with FileLock(str(run/'claim_block_closure.lock'), timeout=0):
        target = run/'CLAIM_SCORING_RESOURCE_BLOCK.json'
        if target.exists():
            saved = json.loads(target.read_text('utf-8'))
            if saved['runner'] != artifact(__file__) or any(artifact(o['path']) != o for o in saved['outputs']):
                raise ValueError('Frozen closure changed')
            print(json.dumps({'status': 'REUSED_RESOURCE_BLOCK_CLOSURE'})); return 0
        attempts = []
        for name, handle, log, available in (
            ('claim_scores_cuda_v2', 80992, 'S1_CLAIM_METRICS_GPU_FULL.log', 3153125376),
            ('claim_scores_cuda_v3', 66941, 'S1_CLAIM_METRICS_GPU_FULL_streamed.log', 3206303744),
            ('claim_scores_cpu_v3', 24620, 'S1_CLAIM_METRICS_CPU_COMPATIBILITY.log', 2821029888)):
            answers, cache = run/name/'answers.jsonl', run/name/'relationships.jsonl'
            attempts.append({'directory': name, 'tool_session_id': handle, 'actual_exit_code': 1,
                'answers_scored': len(read_records(answers)), 'relationships': len(read_records(cache)),
                'guard_available_bytes': available, 'guard_required_bytes': 3221225472,
                'log': artifact(run/log), 'answers': artifact(answers), 'relationships_journal': artifact(cache)})
        cpu = {r['record_id']: r for r in read_records(run/'claim_scores_cpu_v3/answers.jsonl')}
        old = [json.loads(l) for l in (run/'RECOUNT_ALL_OLD.jsonl').read_text('utf-8').splitlines()]
        rows = []
        for original in old:
            rid = original['record_id']
            if rid in cpu:
                score = cpu[rid]
                rows.append({'record_id': rid, 'query_id': original['query_id'],
                    'status': 'ACTUAL_CPU_NLI_PROXY_PARTIAL', 'score_source_hash': score['record_hash'],
                    'claim_f1_proxy': score['claim_f1_proxy'], 'claim_f1_proxy_lower': score['claim_f1_proxy_lower'],
                    'claim_f1_proxy_upper': score['claim_f1_proxy_upper'], 'human_review_status': 'NOT_REVIEWED'})
            else:
                failed = original['failed'] or original['model_abstained']
                applicable = original['document_answerable']
                rows.append({'record_id': rid, 'query_id': original['query_id'],
                    'status': 'FAILED_OR_ABSTAINED_QUALITY_ZERO_BY_CONTRACT' if failed and applicable else 'NOT_RUN_RESOURCE_BLOCK',
                    'claim_f1_proxy': 0.0 if failed and applicable else None,
                    'claim_f1_proxy_lower': 0.0 if applicable else None,
                    'claim_f1_proxy_upper': 0.0 if failed and applicable else 1.0 if applicable else None,
                    'context_support_precision_proxy': None, 'citation_support_precision_proxy': None,
                    'citation_completeness_proxy': None, 'human_review_status': 'NOT_REVIEWED'})
        outputs = []
        path = run/'CLAIM_ALL_OLD_COVERAGE_AND_UNKNOWNS.jsonl'
        with path.open('x', encoding='utf-8', newline='\n') as handle:
            for row in rows:
                handle.write(canonical_json(row)+'\n')
        outputs.append(artifact(path))
        joins = {r['blind_id']: r for r in read_records(run/'BLIND_REVIEW_PRIVATE_JOIN.jsonl')}
        blind = read_records(run/'BLIND_REVIEW_PACKAGE.jsonl')
        path = run/'BLIND_REVIEW_NLI_RESOURCE_PARTIAL.jsonl'
        reviewed = 0
        with path.open('x', encoding='utf-8', newline='\n') as handle:
            for row in blind:
                score = cpu.get(joins[row['blind_id']]['record_id'])
                reviewed += score is not None
                metrics = {k: score[k] for k in ('claim_f1_proxy', 'claim_f1_proxy_lower', 'claim_f1_proxy_upper',
                    'context_support_precision_proxy', 'citation_support_precision_proxy', 'citation_completeness_proxy')} if score else None
                handle.write(canonical_json({'blind_id': row['blind_id'], 'sample_weight': row['sample_weight'],
                    'sampling_probability': row['sampling_probability'], 'metrics': metrics,
                    'status': 'ACTUAL_CPU_PROXY' if score else 'UNKNOWN_NOT_RUN_RESOURCE',
                    'human_review_status': 'NOT_REVIEWED'})+'\n')
        outputs.append(artifact(path))
        result = {'status': 'BLOCKED_RESOURCE_BOUNDED_ATTEMPTS_EXHAUSTED', 'runner': artifact(__file__),
            'recorded_utc': datetime.now(timezone.utc).isoformat(), 'attempts': attempts,
            'expected_old_answers': 5720, 'actual_cpu_partial_answers': len(cpu), 'all_old_coverage_rows': len(rows),
            'blind_expected': 120, 'blind_actual_cpu_scored': reviewed, 'outputs': outputs,
            'unscored_rows_are_not_model_results': True, 'gpu_and_cpu_scores_not_mixed': True,
            'automatic_release_gate_from_full_claim_metric': False,
            'human_review_status': 'NOT_REVIEWED', 'new_llm_requests': 0,
            'next_action': 'No more NLI reload; continue context-only judge and other independent S0-S6 work'}
        atomic_json(target, result)
        print(json.dumps({k: v for k, v in result.items() if k not in ('attempts', 'outputs', 'runner')}))
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
