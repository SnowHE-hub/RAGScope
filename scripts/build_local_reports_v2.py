"""Read-only projections of existing verified analyses, never a final acceptance.

--prepare validates only a source specification and never opens experiment data.
--build is for the coordinator's final window, after all seven decisions freeze.
No model imports/calls, metric fitting, bootstrap reruns, or configuration choice.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragscope.experiments.ledger import read_records
from ragscope.indexing.identity import artifact, atomic_json, signature, verify_artifacts

FILES = ('RESOURCE_REPORT.md', 'TRAINING_REPORT.md', 'ROUTING_REPORT.md', 'RETRIEVAL_TABLES.md')
SOURCE_NAMES = ('analysis', 'retrieval_statistics', 'training_result', 'training_audit',
                'paired_summary', 'online_summary', 'online_manifest', 'gpu_r3_migration', 'gpu_r3_validation')
LIMITS = (
    '所有结果均为原运行条件的描述性观测；多重对照为探索性，不据未校正区间声称显著或因果。',
    '非流式调用的 TTFT 保持 N/A；device-total 为整个设备采样，包含其他进程且可能漏瞬时峰值。',
    'GPU R3 的 E8/ROUTE 成本与原 CPU E2 R3、CPU 在线部署不同，不能直接作为部署加速比。',
    '重放 latency 是已记录组件成本之和，actual_online 才是实际请求总耗时；加载与驻留条件分别保留。',
    '失败保留完整分母；不适用或缺失测量为 N/A/UNAVAILABLE，不补零来代替缺测。',
    '人工审核状态 NOT_REVIEWED；局部报告生成不代表最终验收或默认路由获准启用。',
)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def prepare(spec):
    if spec.get('version') != 'local_reports_v1' or not all(isinstance(spec.get(k), str) for k in ('campaign_dir', 'routing_dir')):
        raise ValueError('Expected local_reports_v1 and explicit campaign/routing directories')
    if set(spec.get('sources', {})) - set(SOURCE_NAMES):
        raise ValueError('Unknown source: do not silently add a new analysis')
    from ragscope.routing.campaign import REQUIRED_ROUTERS
    if set(spec.get('replays', {})) - {*REQUIRED_ROUTERS, 'laya'}:
        raise ValueError('Unknown replay policy')
    for entry in list(spec.get('sources', {}).values()) + list(spec.get('replays', {}).values()) + spec.get('command_receipts', []):
        if entry is not None and (not isinstance(entry, dict) or not {'path', 'bytes', 'sha256'} <= set(entry)):
            raise ValueError('Each supplied source requires its frozen path, bytes and SHA256')
    return {'status': 'PREPARED_NO_EXPERIMENT_READS', 'report_files': list(FILES),
            'unbound_sources': [key for key in SOURCE_NAMES if spec.get('sources', {}).get(key) is None],
            'model_calls': 0, 'experiment_sources_opened': 0,
            'required_build_gate': 'Coordinator --build after seven frozen policies and complete matrix'}


def references(value):
    """Find embedded artifact descriptors; hashes are validated, not trusted prose."""
    found = []
    if isinstance(value, dict):
        if {'path', 'bytes', 'sha256'} <= set(value):
            found.append({k: value[k] for k in ('path', 'bytes', 'sha256')})
        for item in value.values():
            found.extend(references(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(references(item))
    return list({(x['path'], x['sha256']): x for x in found}.values())


def source(entry, label, sources):
    if entry is None or not Path(entry['path']).is_file():
        return None
    if not verify_artifacts([entry]):
        raise ValueError('Frozen source hash changed: '+label)
    data = read(entry['path'])
    dependencies = references(data)
    if not verify_artifacts(dependencies):
        raise ValueError('Source dependency changed: '+label)
    sources.extend([entry, *dependencies])
    return data


def verified_store(path):
    path = Path(path)
    config = read(path.with_suffix('.config.json'))
    rows = read_records(path, repair_tail=False)
    if any(not row.get('record_hash') or row.get('config_hash') != signature(config) for row in rows):
        raise ValueError('RecordStore hash/config binding differs')
    if len({r['record_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate record ID')
    return config, rows


def frozen_policies(routing, sources):
    from ragscope.routing.campaign import REQUIRED_ROUTERS, validate_decisions
    from ragscope.routing.online_campaign import verified_body
    routing = Path(routing)
    protocol_path = routing/'ROUTING_MANIFEST.json'
    protocol = verified_body(protocol_path, 'protocol_hash')
    policies = {}
    for name in REQUIRED_ROUTERS:
        path = routing/f'DECISIONS_{name}.json'
        policies[name] = validate_decisions(read(path), protocol, name)
        sources.append(artifact(path))
    sources.append(artifact(protocol_path))
    return protocol, policies


def validate_main(spec, protocol, data, sources):
    from ragscope.experiments.analysis_report import validated_rows
    from ragscope.experiments.generation_campaign import load_campaign
    if data is None or data.get('status') != 'COMPLETE':
        raise ValueError('Final local build requires complete main ANALYSIS; use --prepare while pending')
    campaign = Path(spec['campaign_dir'])
    manifest, inputs = load_campaign(campaign)
    if (manifest['manifest_hash'] != data['manifest_hash']
            or manifest['manifest_hash'] != protocol['comparison']['manifest_hash']):
        raise ValueError('Analysis, routing and main matrix identity differ')
    rows, evidence = validated_rows(campaign, manifest, inputs)
    if len(rows) != len(manifest['tasks']) or len(rows) != data['n_observed'] or len(rows) != data['n_expected_runnable']:
        raise ValueError('Main matrix is incomplete or analysis denominator differs')
    counts = Counter((r['experiment_id'], r['model_id'], r['method']) for r in rows)
    if any(counts[g['experiment_id'], g['model_id'], g['method']] != g['n_observed'] or g['n_observed'] != g['n_expected'] for g in data['groups']):
        raise ValueError('Main group denominator differs')
    sources.extend([artifact(campaign/'EXPERIMENT_MANIFEST.json'), artifact(campaign/'FROZEN_INPUTS.json'), *evidence])
    return manifest


def scalar(value):
    if value is None:
        return 'N/A'
    return str(value).replace('|', '\\|').replace('\n', ' ')


def table(headers, rows):
    return '\n'.join(['|'+'|'.join(headers)+'|', '|'+'|'.join('---' for _ in headers)+'|'] +
                     ['|'+'|'.join(scalar(x) for x in row)+'|' for row in rows])


def block(title, data, keys=None):
    if data is None:
        return f'## {title}\n\n**UNAVAILABLE**：没有完整、已绑定来源；不以接口存在代替实验结果。\n'
    selected = data if keys is None else {key: data[key] for key in keys if key in data}
    return f'## {title}\n\n```json\n{json.dumps(selected, ensure_ascii=False, indent=2)}\n```\n'


def render_resources(analysis, data):
    rows = []
    for group in analysis['groups']:
        perf, observed = group['performance'], group['observed_summary']
        rows.append([group['experiment_id'], group['model_id'], group['method'], group['n_observed'],
            group['n_expected'], observed['n_failed'], perf.get('resource_measured_n'),
            observed['latency_ms'].get('p50'), observed['latency_ms'].get('p95'),
            perf.get('backend_input_tokens_p50'), perf.get('decode_tokens_per_second_p50'),
            perf.get('peak_sampled_device_total_used_bytes'), perf.get('peak_sampled_service_tree_rss_bytes'), 'N/A'])
    return ['# RAGScope v2 资源局部报告', '生成时间从冻结context开始，不包含检索；单位按原字段保留。',
        table(['实验','模型','方法','实际N','计划N','失败','资源测量N','P50 ms','P95 ms','输入tokens P50',
               'decode tokens/s P50','device-total峰值 bytes','服务树RSS峰值 bytes','TTFT'], rows),
        block('GPU R3迁移与可比性', data.get('gpu_r3_migration')),
        block('GPU R3完整性签收', data.get('gpu_r3_validation'), ('status','scope','n_total','n_failed','worker_exited','gpu_migration_complete'))]


def render_training(data):
    result, audit, paired = (data.get(k) for k in ('training_result','training_audit','paired_summary'))
    if result is not None and (result.get('status') != 'TRAIN_LOOP_COMPLETE' or audit is None or audit.get('status') != 'PASS_REAL_TRAINING_AUDIT'):
        result = None
    if result is not None and result['actual_optimizer_steps'] != audit['actual_optimizer_steps']:
        raise ValueError('LoRA result/audit step denominator differs')
    parts = ['# RAGScope v2 训练局部报告', '训练loss下降不是QA改善；训练完成与paired质量对照分别判定。',
        block('已完成训练结果', result), block('真实训练/恢复审计', audit)]
    if paired is None or paired.get('status') != 'COMPLETE':
        parts += [block('相同BF16 baseline/adapter对照', None),
                  block('已存在的未完成诊断（非质量结论）', paired, ('status','completed','pending','error','expected_n_per_condition'))]
    else:
        parts += [block('相同BF16 baseline/adapter完整对照', paired)]
    parts += ['两个BF16条件应共用后端、模板、context和精度。与Q4主矩阵的schema约束不同，跨协议差不能归因为精度或LoRA。',
              'optimizer step耗时不含模型加载、checkpoint I/O和dev评估；Torch allocator峰值不等于设备物理显存，不能与device-total直接合并。']
    return parts


def validate_paired(entry, data, sources):
    if data is None or data.get('status') != 'COMPLETE':
        return
    from ragscope.evaluation.v2 import summarize_records
    directory = Path(entry['path']).parent
    frozen_path = directory/'FROZEN_EVALUATION_INPUTS.json'
    frozen = read(frozen_path)
    if signature(frozen) != data['frozen_inputs_hash']:
        raise ValueError('Paired frozen input hash differs')
    ids = set(frozen['query_ids'])
    if len(ids) != data['expected_n_per_condition'] or len(ids) != 100:
        raise ValueError('Paired full100 denominator differs')
    for tag in ('baseline','adapted'):
        path = directory/(tag+'.jsonl')
        config, rows = verified_store(path)
        if (len(rows) != len(ids) or {r['query_id'] for r in rows} != ids
                or set(config['query_ids']) != ids or any(r.get('real_model') is not True for r in rows)
                or summarize_records(rows) != data[tag]):
            raise ValueError('Paired summary differs from complete real records')
        sources.extend([artifact(path), artifact(path.with_suffix('.config.json'))])
    sources.append(artifact(frozen_path))


def routing_data(spec, protocol, policies, sources):
    from ragscope.routing.evaluate import summarize
    result = {}
    for name in (*policies, 'laya'):
        binding = spec.get('replays', {}).get(name)
        data = source(binding, 'replay '+name, sources)
        if data is None:
            result[name] = None
            continue
        path = Path(binding['path'])
        if path.resolve() != (Path(spec['routing_dir'])/f'REPLAY_{name}.json').resolve():
            raise ValueError('Replay binding is outside its frozen policy directory')
        if data['name'] != name or data['protocol_hash'] != protocol['protocol_hash'] or not verify_artifacts(data['source_records']):
            raise ValueError('Replay provenance differs')
        for split in ('dev','test'):
            rows = data['groups'][split]['rows']
            expected = protocol['populations'][split]['query_ids']
            if (Counter(r['query_id'] for r in rows) != Counter(expected)
                    or data['groups'][split]['summary'] != summarize(rows)):
                raise ValueError('Replay population/summary differs')
        result[name] = {split: data['groups'][split]['summary'] for split in ('dev','test')}
        sources.extend([artifact(path), *data['source_records']])
    return result


def online_sections(online):
    if online is None or online.get('status') != 'COMPLETE':
        diagnostic = None if online is None else {key:online[key] for key in ('status','stop_reason','test_used') if key in online}
        return [block('真实在线paired dev完整结果', None), block('未完成在线诊断', diagnostic)]
    return [block('真实在线paired dev完整结果', online)]


def validate_online(data, sources):
    online, frozen = data.get('online_summary'), data.get('online_manifest')
    if online is None or online.get('status') != 'COMPLETE':
        return
    from ragscope.routing.evaluate import summarize
    if frozen is None or online['online_hash'] != frozen['online_hash'] or online.get('test_used') is not False:
        raise ValueError('Online paired provenance is absent or not dev-only')
    body = {k:v for k,v in frozen.items() if k != 'online_hash'}
    if signature(body) != frozen['online_hash']:
        raise ValueError('Online frozen body hash differs')
    path = Path(online['records']['path'])
    _, rows = verified_store(path)
    for name, item in online['groups'].items():
        selected = [r for r in rows if r['policy']==name]
        if (len(selected) != item['n_expected'] or len(selected) < 30 or item['n_pending']
                or {r['query_id'] for r in selected} != set(frozen['query_ids'])
                or summarize(selected) != item['summary']):
            raise ValueError('Online full paired population/summary differs')
    sources.extend([artifact(path), artifact(path.with_suffix('.config.json'))])


def render_retrieval(data):
    if data is None:
        return ['# RAGScope v2 检索局部表', block('完整检索统计', None)]
    rows, missing = [], []
    for name, run in data['runs'].items():
        if run['status'] != 'PASS':
            missing.append({'run_id':name, **run})
            continue
        for metric, values in run['metrics'].items():
            rows.append([name, run['n_total'], run['n_failed'], metric, values.get('mean'),
                         values.get('low'), values.get('high'), values.get('n_rows', 0 if values.get('mean') is None else None),
                         values.get('n_clusters'), values.get('n_resamples')])
    return ['# RAGScope v2 检索局部表', f"来源整体状态：{data['status']}；没有据test重新选型。",
        '下表直接读取现有bootstrap统计，不重新定义指标或选择对照。实际N含失败；证据指标对不可答为N/A，聚类数随适用指标变化。',
        table(['run_id','实际N','失败','指标','mean','95% low','95% high','适用N','cluster N','resamples'],rows),
        block('缺失组', missing), block('预声明探索性配对对照', data['comparisons']),
        block('原统计协议', data['protocol'])]


def validate_retrieval(data):
    if data is None:
        return
    if data.get('test_used_for_selection') is not False:
        raise ValueError('Retrieval source must retain dev-only selection')
    for run in data['runs'].values():
        if run['status'] != 'PASS':
            continue
        paths = [Path(a['path']) for a in run['artifacts'] if a['path'].endswith('.jsonl')]
        if len(paths) != 1:
            raise ValueError('Complete retrieval run needs one bound RecordStore')
        config, rows = verified_store(paths[0])
        if (len(rows) != run['n_total'] or Counter(r['query_id'] for r in rows) != Counter(config['query_ids'])
                or sum(bool(r.get('error')) for r in rows) != run['n_failed']):
            raise ValueError('Retrieval denominator differs from preserved records')


def command_evidence(spec, sources):
    commands = []
    for entry in spec.get('command_receipts', []):
        data = source(entry, 'actual command receipt', sources)
        if data is not None:
            commands.append({'receipt':entry, 'observed':{k:data[k] for k in
                ('command','argv','worker_command','observed_session_exit_code','exit_code','status') if k in data},
                'actual_command_available':any(k in data for k in ('command','argv','worker_command')),
                'missing_command_label':'UNAVAILABLE' if not any(k in data for k in ('command','argv','worker_command')) else None})
    return commands


def build(spec, output, *, command):
    prepare(spec)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a fresh directory; preserve previous reports and receipts')
    sources = []
    # Test-bearing summaries are not opened until the seven-policy freeze gate.
    protocol, policies = frozen_policies(spec['routing_dir'], sources)
    bindings = spec.get('sources', {})
    analysis = source(bindings.get('analysis'), 'analysis', sources)
    manifest = validate_main(spec, protocol, analysis, sources)
    data = {key:source(bindings.get(key), key, sources) for key in SOURCE_NAMES if key!='analysis'}
    validate_retrieval(data.get('retrieval_statistics'))
    validate_paired(bindings.get('paired_summary'), data.get('paired_summary'), sources)
    validate_online(data, sources)
    replay = routing_data(spec, protocol, policies, sources)
    commands = command_evidence(spec, sources)
    sections = {
        FILES[0]:render_resources(analysis,data), FILES[1]:render_training(data),
        FILES[2]:['# RAGScope v2 路由局部报告', '七种策略dev/test决策已冻结；train/dev代理标签不是绝对真实最优路由。',
                  *[block(name, value) for name,value in replay.items()],
                  *online_sections(data.get('online_summary')),
                  '仅记录已有部署门槛结论；冷应用与resident实验配置不同，此报告不启用UI默认。'],
        FILES[3]:render_retrieval(data.get('retrieval_statistics'))}
    unique = list({(s['path'],s['sha256']):s for s in sources}.values())
    provenance = ['## 来源与实际构建', f"清单身份：`{manifest['manifest_hash']}`；完整主矩阵N={analysis['n_observed']}。",
        block('本次实际只读构建命令', command), block('来源实际执行命令收据', commands if commands else None),
        *[f"- `{s['path']}` · SHA256 `{s['sha256']}` · {s['bytes']} bytes" for s in unique],
        '## 解释边界', *['- '+note for note in LIMITS]]
    output.mkdir(parents=True)
    for name, parts in sections.items():
        (output/name).write_text('\n\n'.join(parts+provenance)+'\n',encoding='utf-8')
    receipt = {'status':'LOCAL_REPORTS_WRITTEN_ACCEPTANCE_NOT_IMPLIED','time':datetime.now(timezone.utc).isoformat(),
        'manifest_hash':manifest['manifest_hash'],'n_observed':analysis['n_observed'],
        'sources':unique,'reports':[artifact(output/name) for name in FILES],
        'implementation':artifact(__file__),'command':command,'model_calls':0,'test_used_for_selection':False,
        'human_review':'NOT_REVIEWED','nonfixed_default_changed':False}
    atomic_json(output/'LOCAL_REPORTS_BUILD.json',receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--build',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    spec=read(args.spec)
    if args.prepare:
        result=prepare(spec)
    else:
        if args.output is None:raise ValueError('--build requires a fresh --output directory')
        result=build(spec,args.output,command=[sys.executable,*sys.argv])
    print(json.dumps({k:v for k,v in result.items() if k in ('status','report_files','unbound_sources','model_calls','experiment_sources_opened')}))


if __name__=='__main__':
    main()
