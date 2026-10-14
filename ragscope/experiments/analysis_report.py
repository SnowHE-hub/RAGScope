"""Rebuild auditable generation tables from frozen tasks and hashed raw records.

No model calls. Incomplete groups retain their planned N but cannot acquire a
final estimate/CI. All multi-comparison intervals are labelled exploratory.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import json
from pathlib import Path

from ragscope.evaluation.v2 import cluster_bootstrap, paired_cluster_bootstrap, summarize_records, percentile
from ragscope.experiments.generation_campaign import load_campaign, iter_record_sources
from ragscope.experiments.ledger import read_records
from ragscope.indexing.identity import artifact, atomic_json, signature


def flatten_scores(row):
    row = copy.deepcopy(row)
    scores = row['scores']
    # Failures already have zero answer scores in production; refuse misleading
    # externally modified failure scores rather than silently accepting them.
    if row.get('error') and not scores.get('gold_unanswerable'):
        if any(scores.get(key) != 0 for key in ('token_f1', 'exact_match', 'rouge_l')):
            raise ValueError('Failure answer score must remain zero')
    for source, prefix in [('context_evidence', 'context'), ('citation_evidence', 'citation')]:
        for metric in ('continuous_coverage', 'full_unit_recall', 'full_reference_coverage'):
            scores[f'{prefix}_{metric}'] = scores.get(source, {}).get(metric)
    row['cluster_id'] = row.get('cluster_id') or row.get('query_metadata', {}).get('article_cluster_id') or row['doc_id']
    return row


def _key(row):
    return row['experiment_id'], row['model_id'], row['method']


def validated_rows(directory, manifest, inputs, *, experiments=None):
    """Validate real records, optionally without opening other experiments.

    Routing train/dev preparation must not read test-output journals merely to
    validate unrelated groups. Full reports retain the default complete scan.
    """
    directory = Path(directory).resolve()
    tasks = {task['task_id']: task for task in manifest['tasks']}
    selected_experiments = None if experiments is None else set(experiments)
    if selected_experiments is not None and (not selected_experiments or
            selected_experiments - {t['experiment_id'] for t in manifest['tasks']}):
        raise ValueError('Unknown or empty selected experiment population')
    seen, tickets, rows, sources = set(), set(), [], []
    for record_source in iter_record_sources(directory, manifest, inputs):
        source_directory = record_source['directory']
        source_manifest = record_source['manifest']
        owned = record_source['owned_task_ids']
        if source_directory != directory:
            sources.extend([artifact(source_directory/'EXPERIMENT_MANIFEST.json'),
                            artifact(source_directory/'FROZEN_INPUTS.json')])
        files = sorted({f"{t['experiment_id']}_{t['model_id']}.jsonl" for t in source_manifest['tasks']
                        if selected_experiments is None or t['experiment_id'] in selected_experiments})
        for name in files:
            path = source_directory/'records'/name
            if not path.exists():
                continue
            stored = read_records(path)
            config_path = path.with_suffix('.config.json')
            config = json.loads(config_path.read_text(encoding='utf-8'))
            if config['manifest_hash'] != source_manifest['manifest_hash']:
                raise ValueError('Record store belongs to another manifest')
            for row in stored:
                tid = row.get('task_id')
                if not row.get('record_hash') or row.get('config_hash') != signature(config):
                    raise ValueError('Missing record/config hash')
                if tid not in tasks or tid not in owned or tid in seen or row['record_id'] != tid:
                    raise ValueError('Unknown/duplicate/wrong-owner task record')
                if any(row.get(key) != value for key, value in tasks[tid].items()):
                    raise ValueError('Record differs from frozen task')
                source = inputs[row['input_id']]
                if row.get('real_model') is not True or row['model_identity'] != manifest['models'][row['model_id']]:
                    raise ValueError('Mock/stale model identity')
                if row['context_bundle'] != source['context_bundle'] or row['question'] != source['query']['question']:
                    raise ValueError('Frozen generation text differs')
                ticket = row.get('call_ticket', {}).get('call_number')
                if ticket is None or ticket in tickets:
                    raise ValueError('Missing/duplicate real request ticket')
                seen.add(tid); tickets.add(ticket)
                rows.append(flatten_scores(row))
            sources.extend([artifact(path), artifact(config_path)])
    return rows, sources


def _length_bin(row, bounds):
    value = row['context_bundle']['budget'].get('reference_tokens')
    if value is None:
        return 'unknown'
    if value == 0:
        return '0'
    previous = 0
    for upper in bounds:
        if upper and value <= upper:
            return f'{previous+1}-{upper}'
        previous = upper
    return f'>{bounds[-1]}'


def build_analysis(directory, config):
    manifest, inputs = load_campaign(directory)
    if config['cluster_resamples'] != 2000:
        raise ValueError('Frozen protocol requires 2000 cluster resamples')
    rows, sources = validated_rows(directory, manifest, inputs)
    indexed = defaultdict(list)
    expected = Counter(_key(t) for t in manifest['tasks'])
    for row in rows:
        indexed[_key(row)].append(row)
    groups = []
    for key, count in sorted(expected.items()):
        observed = indexed[key]
        item = dict(zip(('experiment_id', 'model_id', 'method'), key))
        item.update(status='COMPLETE' if len(observed) == count else 'INCOMPLETE',
                    n_expected=count, n_observed=len(observed), n_missing=count-len(observed),
                    observed_summary=summarize_records(observed), estimates=None)
        if item['status'] == 'COMPLETE':
            item['estimates'] = {metric: cluster_bootstrap(observed, metric, cluster_key='cluster_id',
                n_resamples=2000, seed=config['seed']) for metric in config['primary_metrics']}
        item['errors'] = dict(Counter(r['error']['kind'] for r in observed if r.get('error')))
        item['length_groups'] = []
        for label in sorted({_length_bin(r, config['reference_context_length_bins']) for r in observed}):
            subset = [r for r in observed if _length_bin(r, config['reference_context_length_bins']) == label]
            item['length_groups'].append({'reference_context_tokens': label, **summarize_records(subset)})
        item['document_length_groups'] = []
        for label in ('short', 'medium', 'long', 'unknown'):
            subset = [r for r in observed if r.get('document_length', {}).get('length_group', 'unknown') == label]
            if subset:
                item['document_length_groups'].append({'canonical_document_length_group': label,
                    'boundaries_reference_tokens': 'short<=512; medium513-4096; long>4096',
                    **summarize_records(subset)})
        item['evidence_answer_relationship'] = []
        for label, value in [('fully_covered', 1), ('not_fully_covered', 0)]:
            subset = [r for r in observed if r['scores']['context_full_reference_coverage'] == value]
            item['evidence_answer_relationship'].append({'context_group': label, **summarize_records(subset)})
        tokens = [r.get('backend_metadata', {}).get('prompt_eval_count') for r in observed]
        decode = [r.get('backend_metadata', {}).get('eval_duration') for r in observed]
        speeds = [r['backend_metadata']['eval_count']/(d/1e9) for r, d in zip(observed, decode)
                  if d and r.get('backend_metadata', {}).get('eval_count') is not None]
        item['performance'] = {'generation_timing_mode': 'generation_with_frozen_context',
            'retrieval_latency_included': False, 'first_token_ms': None,
            'first_token_limit': 'Nonstreaming calls have no observed first token event',
            'backend_input_tokens_p50': percentile([t for t in tokens if t is not None], .5),
            'decode_tokens_per_second_p50': percentile(speeds, .5), 'decode_measured_n': len(speeds)}
        measured = [r.get('resources', r.get('backend_metadata', {}).get('resources', {})) for r in observed]
        for field in ('peak_sampled_device_total_used_bytes', 'peak_sampled_service_tree_rss_bytes'):
            values = [m[field] for m in measured if m.get(field) is not None]
            item['performance'][field] = max(values) if values else None
        for field in ('client_request_wall_ms', 'telemetry_finalize_ms'):
            values = [r.get('backend_metadata', {}).get(field) for r in observed]
            item['performance'][field+'_p50'] = percentile([v for v in values if v is not None], .5)
        item['performance']['resource_measured_n'] = sum(m.get('status') == 'MEASURED_SAMPLED' for m in measured)
        item['performance']['wrapper_latency_includes_telemetry'] = True
        groups.append(item)
    blocked = [g for g in manifest['groups'] if g.get('status') == 'BLOCKED']
    comparisons = []

    def compare(left, right, label):
        left_rows, right_rows = indexed[left], indexed[right]
        output = {'contrast': label, 'left': left, 'right': right,
                  'inference': 'EXPLORATORY_MULTIPLE_COMPARISONS', 'statistics': None}
        if not expected[left] or not expected[right] or len(left_rows) != expected[left] or len(right_rows) != expected[right]:
            output['status'] = 'INCOMPLETE'
        else:
            # Existing paired primitive requires doc_id; use the verified upstream
            # article/paper cluster, preserving original paragraph IDs in rows.
            left_copy = [dict(r, doc_id=r['cluster_id']) for r in left_rows]
            right_copy = [dict(r, doc_id=r['cluster_id']) for r in right_rows]
            output.update(status='COMPLETE', statistics={metric: paired_cluster_bootstrap(
                left_copy, right_copy, metric, n_resamples=2000, seed=config['seed'])
                for metric in config['primary_metrics']})
        comparisons.append(output)

    for experiment, left, right in config['within_model_contrasts']:
        for model in sorted(manifest['models']):
            if (experiment, model, left) in expected or (experiment, model, right) in expected:
                compare((experiment, model, left), (experiment, model, right), f'{model}:{left}-{right}')
    scales = config['model_scale_comparison']
    for left_size, right_size in scales['pairs']:
        matches = {size: [model for model, identity in manifest['models'].items()
                          if f':{size}-' in identity['model'].lower()]
                   for size in (left_size, right_size)}
        if all(len(matches[size]) == 1 for size in matches):
            compare((scales['experiment'], matches[left_size][0], scales['method']),
                    (scales['experiment'], matches[right_size][0], scales['method']), f'{left_size}-{right_size}')
    return {'status': 'COMPLETE' if len(rows) == len(manifest['tasks']) and not blocked else 'INCOMPLETE_OR_BLOCKED',
        'manifest_hash': manifest['manifest_hash'], 'analysis_config': config, 'analysis_config_hash': signature(config),
        'n_expected_runnable': len(manifest['tasks']), 'n_observed': len(rows),
        'n_expected_blocked': sum(g['expected_n']*len(g['model_ids']) for g in blocked),
        'blocked_groups': blocked, 'groups': groups, 'comparisons': comparisons,
        'sources': [artifact(Path(directory)/'EXPERIMENT_MANIFEST.json'), artifact(Path(directory)/'FROZEN_INPUTS.json'), *sources],
        'limitations': ['Answer F1 measures textual overlap, not complete correctness.',
            'Citation visibility and gold evidence coverage are not faithfulness.',
            'Incomplete estimates are absent; observed summaries cannot replace the frozen denominator.',
            'All multiple contrasts are exploratory; no unadjusted significance claim.',
            'SQuAD clusters use upstream article; QASPER clusters use paper.',
            'Context-length groups describe supplied evidence; document-length groups separately use canonical source metadata.',
            'Resource peaks sampled on device include other processes; sampling may miss short peaks.']}


def markdown_table(report):
    lines = ['# RAGScope v2 逐行生成结果重建', '',
        f"状态：**{report['status']}**。已记录 {report['n_observed']} / {report['n_expected_runnable']} 个可运行任务；"
        f"另有 {report['n_expected_blocked']} 个任务明确受阻。", '',
        '下表 F1 只对可回答问题适用，失败记零；不可回答问题单列拒答指标。证据覆盖衡量实际发送的上下文，不能称为事实支持性。', '',
        '|实验|模型|方法|状态|已记录/计划|失败|文字 F1|完整证据覆盖|生成 P50 ms|',
        '|---|---|---|---|---:|---:|---:|---:|---:|']
    def fmt(value):
        return 'N/A' if value is None else f'{value:.4f}'
    for group in report['groups']:
        summary, estimates = group['observed_summary'], group['estimates']
        f1 = estimates['token_f1']['mean'] if estimates else None
        coverage = estimates['context_full_reference_coverage']['mean'] if estimates else None
        lines.append(f"|{group['experiment_id']}|{group['model_id']}|{group['method']}|{group['status']}|"
                     f"{group['n_observed']}/{group['n_expected']}|{summary['n_failed']}|{fmt(f1)}|{fmt(coverage)}|"
                     f"{fmt(summary['latency_ms']['p50'])}|")
    lines += ['', '同一篇论文/原始文章内的问题一起重采样，共 2,000 次。完整置信区间、按上下文长度分组、错误种类和所有探索性配对对照见同目录 ANALYSIS.json。',
              '生成时间从冻结上下文开始计时，不含检索。非流式调用没有实测首 token 时间。未完成组不生成最终均值或置信区间。', '',
              f"配置哈希：`{report['analysis_config_hash']}`；实验清单哈希：`{report['manifest_hash']}`。", '']
    return '\n'.join(lines)


def write_analysis(directory, config_path, output):
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    report = build_analysis(directory, config)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/'ANALYSIS.json', report)
    (output/'GENERATION_TABLES.md').write_text(markdown_table(report), encoding='utf-8')
    return report
