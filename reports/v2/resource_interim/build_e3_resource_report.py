"""Rebuild this E3-only resource report from sealed, validated local records."""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ragscope.experiments.analysis_report import validated_rows
from ragscope.experiments.generation_campaign import load_campaign
from ragscope.experiments.ledger import read_records
from ragscope.experiments.runtime import runtime_home
from ragscope.indexing.identity import artifact, atomic_json, signature


def main():
    output = Path(__file__).resolve().parent
    if (output/'RESOURCE_REPORT.md').exists():
        raise FileExistsError('Preserve the existing report and its signed source snapshot')
    campaign = runtime_home()/'runs/20261014_v2_generation_e3'
    manifest, inputs = load_campaign(campaign)
    rows, record_sources = validated_rows(campaign, manifest, inputs)
    if len(rows) != 1200 or len(manifest['tasks']) != 1200:
        raise ValueError('This E3-only report requires the complete fixed1200 population')
    analysis_path = campaign/'analysis_complete/ANALYSIS.json'
    analysis = json.loads(analysis_path.read_text(encoding='utf-8'))
    if analysis['status'] != 'COMPLETE' or analysis['manifest_hash'] != manifest['manifest_hash']:
        raise ValueError('Analysis scope/identity differs')
    estimates = {(g['model_id'], g['method']): g for g in analysis['groups']}
    grouped = defaultdict(list)
    for row in rows:
        if row['experiment_id'] != 'E3':
            raise ValueError('Non-E3 row in scoped report')
        grouped[row['model_id'], row['method']].append(row)
    summaries = []
    for key, observed in sorted(grouped.items()):
        measurements = [r.get('resources', r.get('backend_metadata', {}).get('resources', {})) for r in observed]
        if len(observed) != 200 or any(m.get('status') != 'MEASURED_SAMPLED' for m in measurements):
            raise ValueError('Missing formal rows or sampled resource measurements')
        result = {'model': key[0], 'method': key[1], 'n': len(observed),
                  'generation_p50_ms': statistics.median(r['latency_ms'] for r in observed)}
        for field in ('peak_sampled_device_total_used_bytes', 'peak_sampled_service_tree_rss_bytes'):
            result[field] = max(m[field] for m in measurements)
            if result[field] != estimates[key]['performance'][field]:
                raise ValueError('Resource peak differs from stored analysis')
        if result['generation_p50_ms'] != estimates[key]['observed_summary']['latency_ms']['p50']:
            raise ValueError('Recomputed generation median differs')
        result['median_decode_tokens_per_second'] = statistics.median(
            r['backend_metadata']['eval_count']/(r['backend_metadata']['eval_duration']/1e9) for r in observed)
        summaries.append(result)
    warmups, warmup_sources = [], []
    for path in sorted((campaign/'records').glob('warmup_*.jsonl')):
        actual = read_records(path)
        config = json.loads(path.with_suffix('.config.json').read_text(encoding='utf-8'))
        if config['manifest_hash'] != manifest['manifest_hash'] or config['purpose'] != 'warmup':
            raise ValueError('Warmup configuration identity differs')
        if len(actual) != 3 or any(not r['real_model'] or r.get('error') for r in actual):
            raise ValueError('Warmup scope/real model/error differs')
        for row in actual:
            if signature({k: v for k, v in row.items() if k != 'record_hash'}) != row['record_hash']:
                raise ValueError('Warmup raw row changed')
            if row['config_hash'] != signature(config) or row['model_identity'] != manifest['models'][config['model_id']]:
                raise ValueError('Warmup row model/config differs')
        first = actual[0]
        warmups.append({'model': config['model_id'], 'n': 3,
                        'first_request_wall_seconds': first['latency_ms']/1000,
                        'first_backend_load_seconds': first['backend_metadata']['load_duration']/1e9,
                        'first_record_hash': first['record_hash']})
        warmup_sources += [artifact(path), artifact(path.with_suffix('.config.json'))]
    if len(warmups) != 3:
        raise ValueError('Expected three model warmup groups')
    sources = [artifact(campaign/'EXPERIMENT_MANIFEST.json'), artifact(campaign/'FROZEN_INPUTS.json'),
               artifact(analysis_path), *record_sources, *warmup_sources]
    data = {'status': 'E3_SCOPE_COMPLETE_PROJECT_PENDING', 'recorded_utc': datetime.now(timezone.utc).isoformat(),
            'run_id': campaign.name, 'manifest_hash': manifest['manifest_hash'], 'groups': summaries,
            'warmups': warmups, 'sources': sources, 'source_snapshot_hash': signature(sources)}
    atomic_json(output/'RESOURCE_DATA.json', data)
    lines = ['# E3资源观测（全项目资源验收尚未完成）', '',
        '三个模型已串行完成1200条正式调用；每个模型另有3条预热，单独列出。'
        '下表从封存逐行记录重新计算，并与已有分析的P50及资源峰值逐项核对。E2、4K/8K诊断、在线路由和LoRA问答对照仍未计入。', '',
        '|模型|条件|正式调用|生成P50 ms|采样设备总显存峰值 GiB|采样服务树RSS峰值 GiB|解码速度中位数 token/s|',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"|{r['model']}|{r['method']}|{r['n']}|{r['generation_p50_ms']:.3f}|"
                     f"{r['peak_sampled_device_total_used_bytes']/1024**3:.3f}|"
                     f"{r['peak_sampled_service_tree_rss_bytes']/1024**3:.3f}|"
                     f"{r['median_decode_tokens_per_second']:.3f}|")
    lines += ['', '|模型|预热数|首条预热墙钟 s|后端报告该条加载 s|', '|---|---:|---:|---:|']
    for r in warmups:
        lines.append(f"|{r['model']}|3|{r['first_request_wall_seconds']:.3f}|{r['first_backend_load_seconds']:.3f}|")
    lines += ['', '生成P50从冻结上下文开始，包含客户端包装与遥测收尾，不含检索。首条预热墙钟含加载与推理，'
        '后端load_duration是其中一个计时量；它不是仅加载权重的独立实验。非流式调用没有实测首token事件，因此首token延迟为N/A。', '',
        '显存列为设备总使用量的采样峰值，包含可能同时存在的其他进程；RSS为本项目服务进程树的采样峰值。'
        '两者不是同一资源，也不是最低硬件需求。短暂峰值可能被采样遗漏，不能相减推出精确offload层数。GiB为1024³字节。', '',
        'CPU索引在该分片期间并行运行，墙钟时间属于当时共享资源条件下的描述性观测。'
        'No Context与Oracle输入和输出长度不同，不以更快直接判断回答更好；解码速度来自后端eval_count/eval_duration，未把加载或prefill计入解码吞吐。', '',
        '本报告不使用训练allocator allocated/reserved数字替代物理显存。LoRA训练资源与完整断点另见'
        '`docs/execution/TRAINING_REPORT.md`；最终磁盘增量、C盘例外、全部GPU释放和用户手动续跑通知仍须最终验收。', '',
        f"run_id：`{campaign.name}`；manifest：`{manifest['manifest_hash']}`；来源快照：`{data['source_snapshot_hash']}`。",
        '逐个字节计量、原始行及配置来源SHA见同目录RESOURCE_DATA.json。此报告未触发任何模型请求。', '']
    (output/'RESOURCE_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    atomic_json(output/'BUILD_RECEIPT.json', {'status': 'PASS_E3_ONLY', 'formal_rows': 1200,
        'warmup_rows': 9, 'model_requests': 0, 'sources_verified_by_existing_loader': True,
        'outputs': [artifact(output/'RESOURCE_DATA.json'), artifact(output/'RESOURCE_REPORT.md')],
        'implementation': artifact(Path(__file__))})
    print('PASS_E3_ONLY:1200 formal rows +9 separate warmups; no model calls')


if __name__ == '__main__':
    main()
