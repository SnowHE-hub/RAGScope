"""Independently score the frozen S2 matrix and describe controlled differences."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from ragscope.experiments.ledger import read_records
from ragscope.indexing.identity import artifact, atomic_json, canonical_json
from ragscope.v3.audit import overlap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    args = parser.parse_args()
    run = Path(args.campaign).resolve()
    if run.parent != Path('D:/RAGScope_Runtime/runs').resolve() or not run.name.startswith('v3_'):
        raise ValueError('Wrong campaign')
    target = run/'S2_PROTOCOL_ANALYSIS.json'
    if target.exists():
        saved = json.loads(target.read_text('utf-8'))
        if saved['runner'] != artifact(__file__) or artifact(saved['source']['path']) != saved['source']:
            raise ValueError('Immutable S2 analysis identity differs')
        print(json.dumps({'status': 'REUSED_PROTOCOL_ANALYSIS'})); return 0
    rows = read_records(run/'S2_PROTOCOL_RESULTS.jsonl')
    labels = {r['query_id']: r for r in read_records(run/'splits/evaluator_only/S2_DIAGNOSTIC_LABELS.jsonl')}
    inputs = {r['input_id']: r for r in read_records(run/'S2_PROTOCOL_INPUTS.jsonl')}
    groups, scored, count_mismatches, token_mismatches = defaultdict(list), [], [], []
    for row in rows:
        if row.get('task_id') != 'S2_matrix' and not row.get('reused_actual_model_result'):
            continue
        entry = inputs[row['input_id']]
        label = labels[entry['query_id']]
        parsed, error = row['parsed_response'], row['error'] if row.get('reused_actual_model_result') else row['error_type']
        metadata = row['backend_metadata'] if row.get('reused_actual_model_result') else row['raw_response']['backend_metadata']
        failed = bool(error) or parsed is None
        abstained = parsed['abstain'] if parsed else None
        metrics = overlap(parsed['answer'], label['answer_refs']) if parsed and not failed and not abstained and label['document_answerable'] else None
        value = None if not label['document_answerable'] else 0.0 if failed or abstained else metrics['legacy_token_f1']
        if not row.get('reused_actual_model_result') and value != row['legacy_token_f1']:
            count_mismatches.append(row['record_id'])
        if metadata.get('token_count_match') is not True:
            token_mismatches.append(row['record_id'])
        result = {'record_id': row['record_id'], 'query_id': entry['query_id'], 'paper_ids': entry['scope']['doc_ids'],
            'model_id': row['model_id'], 'protocol': row['protocol'], 'context_kind': row['context_kind'],
            'seed': row.get('seed', 42), 'stratum': label['selection_stratum'],
            'document_answerable': label['document_answerable'], 'failed': failed, 'abstained': abstained,
            'legacy_token_f1': value, 'predicted_words': len(parsed['answer'].split()) if parsed else None,
            'actual_output_tokens': metadata.get('eval_count'), 'finish_reason': metadata.get('done_reason'),
            'actual_prompt_tokens': metadata.get('prompt_eval_count'), 'human_review_status': 'NOT_REVIEWED'}
        scored.append(result)
        groups[(row['model_id'], row['protocol'], row['context_kind'])].append(result)
    if len(scored) != 864 or count_mismatches or token_mismatches:
        raise ValueError('S2 matrix coverage or independent metric/token audit failed')
    summaries = []
    for (model, protocol, kind), values in sorted(groups.items()):
        applicable = [v for v in values if v['document_answerable']]
        summaries.append({'model_id': model, 'protocol': protocol, 'context_kind': kind, 'n': len(values),
            'answerable_n': len(applicable), 'legacy_word_f1': statistics.mean(v['legacy_token_f1'] for v in applicable),
            'failed_n': sum(v['failed'] for v in values), 'answerable_abstained_n': sum(v['abstained'] is True for v in applicable),
            'output_tokens_p50': statistics.median(v['actual_output_tokens'] for v in values),
            'predicted_words_p50': statistics.median(v['predicted_words'] for v in values if v['predicted_words'] is not None),
            'at_output_limit_n': sum(v['finish_reason'] == 'length' for v in values),
            'by_stratum': {s: {'n': len(part), 'legacy_word_f1': statistics.mean(v['legacy_token_f1'] for v in part if v['legacy_token_f1'] is not None) if any(v['legacy_token_f1'] is not None for v in part) else None}
                for s in ('unanswerable', 'yes_no', 'extractive', 'free_form') for part in [[v for v in values if v['stratum'] == s]]}})
    index = {(v['model_id'], v['protocol'], v['context_kind'], v['query_id'], v['seed']): v for v in scored}
    contrasts = []
    for model in ('qwen_2b', 'qwen_4b', 'qwen_9b'):
        for kind in ('retrieval_frozen', 'oracle_audited'):
            for left, right in (('G1', 'G0'), ('G2', 'G1')):
                pairs = [(index[(model, left, kind, qid, 42)], index[(model, right, kind, qid, 42)])
                    for qid in labels if labels[qid]['document_answerable']]
                contrasts.append({'model_id': model, 'context_kind': kind, 'contrast': left+'-'+right,
                    'applicable_n': len(pairs), 'mean_word_f1_delta': statistics.mean(a['legacy_token_f1']-b['legacy_token_f1'] for a,b in pairs),
                    'improved_n': sum(a['legacy_token_f1'] > b['legacy_token_f1'] for a,b in pairs),
                    'worse_n': sum(a['legacy_token_f1'] < b['legacy_token_f1'] for a,b in pairs)})
    detail = run/'S2_PROTOCOL_SCORED864.jsonl'
    with detail.open('x', encoding='utf-8', newline='\n') as handle:
        for row in scored: handle.write(canonical_json(row)+'\n')
    output = {'status': 'PASS_FROZEN864_WORD_PROTOCOL_AUDIT', 'runner': artifact(__file__),
        'source': artifact(run/'S2_PROTOCOL_RESULTS.jsonl'), 'detail': artifact(detail),
        'matrix_n': 864, 'new_matrix_n': 720, 'reused_G0_n': 144,
        'metric_mismatch_n': len(count_mismatches), 'actual_prompt_token_mismatch_n': len(token_mismatches),
        'groups': summaries, 'contrasts': contrasts, 'human_review_status': 'NOT_REVIEWED',
        'NLI_full_old': 'BLOCKED_RESOURCE', 'performance_claim': 'NONE; exposed diagnostic and word proxy only',
        'oracle_padding': 'PENDING_SEPARATE36', 'new_llm_requests_by_audit': 0}
    atomic_json(target, output)
    print(json.dumps({'status': output['status'], 'matrix_n': 864, 'contrasts': contrasts}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
