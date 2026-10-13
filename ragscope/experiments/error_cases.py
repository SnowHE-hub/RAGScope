"""Evidence-backed diagnostic flags; semantic or causal guesses stay uncertain."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from ragscope.evaluation.evidence import evidence_metrics
from ragscope.experiments.analysis_report import validated_rows
from ragscope.experiments.generation_campaign import load_campaign
from ragscope.indexing.identity import atomic_json, signature

CATEGORIES = ('document_not_retrieved', 'document_found_evidence_missing', 'packing_evidence_loss',
              'noise_or_order_uncertain', 'complete_evidence_low_overlap_uncertain', 'wrong_abstention',
              'missing_or_invalid_citation', 'json_error', 'oom_or_timeout')


def classify(row, query, evidence_units):
    scores, raw = row['scores'], row.get('raw_retrieval', [])
    flags = []
    def flag(category, certainty, reason):
        flags.append({'category': category, 'certainty': certainty, 'reason': reason})
    kind = (row.get('error') or {}).get('kind')
    if kind == 'parse_error':
        flag('json_error', 'observed', 'Persisted raw output did not satisfy the JSON parser.')
    if kind in ('oom', 'timeout'):
        flag('oom_or_timeout', 'observed', f'Persisted backend error kind={kind}.')
    if scores.get('wrong_abstention'):
        flag('wrong_abstention', 'observed', 'Model abstained on an answerable annotated question.')
    if not row.get('error') and not row.get('abstain') and (not row.get('citations') or row.get('invalid_citations')):
        flag('missing_or_invalid_citation', 'observed', 'No visible citation or an out-of-context citation ID; support correctness is a separate test.')
    answerable = query['answerability'] == 'answerable'
    context_full = scores.get('context_evidence', {}).get('full_reference_coverage')
    raw_spans = [span for piece in raw for span in piece.get('unit_spans', [])]
    raw_evidence = evidence_metrics(query.get('evidence_reference_sets', []), raw_spans,
                                    query['answerability'], evidence_units)
    selected = row.get('selected_retrieval')
    if selected is None and len(raw) <= 10:
        selected = raw
    selected_evidence = evidence_metrics(query.get('evidence_reference_sets', []),
        [span for piece in selected for span in piece.get('unit_spans', [])],
        query['answerability'], evidence_units) if selected is not None else None
    # No-context and Oracle deliberately do not perform retrieval: never call
    # those treatments retrieval failures simply because their raw list is empty.
    retrieval_treatment = row.get('experiment_id') not in ('E3', 'E7')
    if answerable and retrieval_treatment:
        target_doc = query.get('doc_id') or query.get('metadata', {}).get('gold_doc_id_evaluation_only')
        docs = {piece.get('doc_id') for piece in raw}
        if target_doc and target_doc not in docs:
            flag('document_not_retrieved', 'observed', 'Evaluation-only target document is absent from persisted retrieval candidates.')
        elif raw_evidence['full_reference_coverage'] == 0:
            flag('document_found_evidence_missing', 'observed', 'Target document is present but no full reference evidence set is covered by persisted candidates.')
        if selected_evidence and selected_evidence['full_reference_coverage'] == 1 and context_full == 0:
            flag('packing_evidence_loss', 'observed', 'A complete reference set exists before packing but is incomplete in the actual prompt context.')
        elif raw_evidence['full_reference_coverage'] == 1 and context_full == 0:
            flag('noise_or_order_uncertain', 'uncertain', 'Evidence is in raw candidates but missing after selection/packing; selected-top10 provenance is insufficient to assign this specifically to the token budget.')
    low_overlap = answerable and scores.get('token_f1') is not None and scores['token_f1'] < .5
    if low_overlap and context_full == 1 and not row.get('error') and not row.get('abstain'):
        flag('complete_evidence_low_overlap_uncertain', 'uncertain', 'All reference evidence is visible but token F1<0.5; paraphrases or annotation ambiguity need semantic review before calling the answer wrong.')
    if low_overlap and row.get('context_bundle', {}).get('pieces') and not row.get('error') and not any(f['category'] == 'noise_or_order_uncertain' for f in flags):
        flag('noise_or_order_uncertain', 'uncertain', 'Noise/order is a candidate explanation only; this individual result does not identify the cause. Consult the paired E7 intervention.')
    return {'flags': flags, 'raw_candidate_evidence': raw_evidence, 'selected_before_packing_evidence': selected_evidence,
            'semantic_answer_correctness': 'NOT_HUMAN_REVIEWED',
            'citation_support': 'NOT_ESTABLISHED_BY_VISIBLE_ID',
            'diagnostic_threshold': {'token_f1_low_overlap': .5, 'purpose': 'case_sampling_only_not_primary_success_metric'}}


def build_error_cases(campaign, output, evidence_units):
    manifest, inputs = load_campaign(campaign)
    rows, sources = validated_rows(campaign, manifest, inputs)
    cases = []
    for row in rows:
        query = inputs[row['input_id']]['query']
        diagnosis = classify(row, query, evidence_units)
        cases.append({'task_id': row['task_id'], 'query_id': row['query_id'], 'doc_id': row['doc_id'],
            'experiment_id': row['experiment_id'], 'method': row['method'], 'model_id': row['model_id'],
            'source_record_hash': row['record_hash'], 'question': query['question'],
            'answer': row.get('answer'), 'raw_output': row.get('raw_output'), 'error': row.get('error'),
            'answer_refs': query.get('answer_refs', []), 'scores': row['scores'], **diagnosis})
    counts = Counter(flag['category'] for case in cases for flag in case['flags'])
    selected = {}
    for category in CATEGORIES:
        candidates = [case for case in cases if any(f['category'] == category for f in case['flags'])]
        selected[category] = sorted(candidates, key=lambda c: signature([42, c['task_id']]))[:3]
    result = {'version': 'evidence-diagnostics-v1', 'manifest_hash': manifest['manifest_hash'],
        'status': 'COMPLETE' if len(rows) == len(manifest['tasks']) else 'INCOMPLETE',
        'n_expected': len(manifest['tasks']), 'n_observed': len(rows),
        'category_counts': {c: counts[c] for c in CATEGORIES}, 'cases': cases, 'selected_examples': selected,
        'sources': sources, 'selection': 'seed42 hash order; max3 per category; overlapping flags; no causal interpretation of uncertain labels'}
    atomic_json(Path(output)/'ERROR_CASES.json', result)
    lines = ['# 错误样例与未确认诊断', '',
        '分类允许重叠。uncertain表示待语义复核或缺少因果证据，不等同已确认答错。F1<0.5只用于抽样，不改变主指标。', '',
        '|类别|观察到的条数|', '|---|---:|']
    lines.extend(f'|{category}|{counts[category]}|' for category in CATEGORIES)
    for category, examples in selected.items():
        lines.extend(['', f'## {category}', ''])
        if not examples:
            lines.append('当前无符合条件的样例；未生成或虚构样例。')
        for case in examples:
            lines += [f"- {case['model_id']}/{case['experiment_id']}/{case['method']} — `{case['query_id']}`",
                      f"  问题：{case['question']}", f"  输出摘录：{str(case.get('raw_output', ''))[:450]}"]
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output)/'ERROR_CASES.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result
