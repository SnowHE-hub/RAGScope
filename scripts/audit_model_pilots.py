"""Summarize actual compatibility pilots, without calling them quality results."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from ragscope.experiments.ledger import read_records
from ragscope.experiments.runtime import runtime_home
from ragscope.indexing.identity import artifact, atomic_json
from ragscope.generation.prompts import PROMPT_VERSION, OUTPUT_CONSTRAINT_VERSION

run = runtime_home()/'runs/20261014_v2'
records_path = run/'generation_pilots.jsonl'
all_rows = read_records(records_path)
groups = []
for size in ('2b', '4b', '9b', 'lfm'):
    path = run/f'pilot_{size}_summary.json'
    summary = json.loads(path.read_text(encoding='utf-8'))
    rows = [row for row in all_rows if row['result']['model_identity'] == summary['model_identity'] and row.get('prompt_version') == PROMPT_VERSION]
    latest = {row['query_id']: row for row in rows}
    if len(latest) != 20 or summary['status'] not in {'PASS', 'PASS_WITH_OUTPUT_FAILURES'}:
        raise ValueError(f'Incomplete final-setting pilot: {size}')
    results = [row['result'] for row in latest.values()]
    if any(row.get('split') != 'dev' or not row.get('ticket') or row.get('token_count_match') is not True for row in latest.values()):
        raise ValueError('Pilot lacks actual counted dev/token parity evidence')
    if any(r['error'] and not (r['error']['kind'] == 'parse_error' and r['backend_metadata'].get('done_reason') == 'length') for r in results):
        raise ValueError('Pilot has unresolved backend or token-budget failures')
    if any(not r.get('citation_constraint', {}).get('applied') or r['citation_constraint']['version'] != OUTPUT_CONSTRAINT_VERSION for r in results):
        raise ValueError('Pilot does not use the frozen output constraint')
    resources = [result.get('backend_metadata', {}).get('resources', {}) for result in results]
    def peak(field):
        values = [r[field] for r in resources if r.get(field) is not None]
        return max(values) if values else None
    groups.append({'size': size, 'model_identity': summary['model_identity'],
        'n_dev': len(latest), 'parse_or_transport_errors': sum(bool(r['error']) for r in results),
        'token_count_matches': sum(row['token_count_match'] for row in latest.values()),
        'responses_with_invalid_citation_ids': sum(bool(r.get('invalid_citations')) for r in results),
        'responses_with_any_visible_citation': sum(bool(r.get('citations')) for r in results),
        'responses_abstaining': sum(r.get('abstain', False) for r in results),
        'resource_measured_rows': sum(r.get('status') == 'MEASURED_SAMPLED' for r in resources),
        'peak_sampled_device_total_used_bytes': peak('peak_sampled_device_total_used_bytes'),
        'peak_sampled_service_tree_rss_bytes': peak('peak_sampled_service_tree_rss_bytes'),
        'peak_gpu_allocator_bytes': None, 'summary_source': artifact(path)})
report = {'status': 'PASS_COMPATIBILITY_NOT_QUALITY', 'groups': groups,
    'prompt_version': PROMPT_VERSION, 'output_constraint_version': OUTPUT_CONSTRAINT_VERSION,
    'all_pilot_attempts_retained': len(all_rows),
    'prior_or_current_attempt_errors': sum(bool(row['result']['error']) for row in all_rows),
    'records_source': artifact(records_path),
    'limitations': ['The same 20 dev questions are used; no test answers informed configuration.',
        'Compatibility PASS means load/completed-request/token-budget checks; it is not an answer/citation quality threshold.',
        'Invalid citation IDs are retained; no facts or citations were repaired to match references.',
        'Pilots are not the three warmups or formal quality/timing population.',
        'Sampled device-total VRAM includes all processes and can miss short peaks.',
        'Visible-ID enum constraints are engineering validation, not evidence-support quality.',
        'A completed output truncated at the common output limit is retained as a parse failure; it does not trigger tuning or cherry-picked retries.']}
atomic_json(run/'PILOT_AUDIT.json', report)
lines = ['# 真实模型预检核验', '',
    '最终设置下四个模型各完成20条相同dev问题。兼容性通过表示加载、请求完成与实际输入计数通过；解析失败单列保留，不表示回答正确。', '',
    '|模型|实际dev条数|解析/传输错误|token计数一致|含不可见引用ID的回答|有可见引用的回答|资源采样条数|',
    '|---|---:|---:|---:|---:|---:|---:|']
for g in groups:
    lines.append(f"|{g['model_identity']['model']}|{g['n_dev']}|{g['parse_or_transport_errors']}|{g['token_count_matches']}|"
                 f"{g['responses_with_invalid_citation_ids']}|{g['responses_with_any_visible_citation']}|{g['resource_measured_rows']}|")
lines += ['', f"全部配置累计保留{len(all_rows)}条预检尝试，其中{report['prior_or_current_attempt_errors']}条错误；没有从日志删除失败。",
    '主模型统一Q4_K_M、4096上下文、最多512输出、temperature=0、seed=42、后端think=false。纯文本别名使用原始语言权重，只省略视觉投影层。',
    '独立Ollama0.34.4服务绑定127.0.0.1:11435，模型位于D盘。LLAMA_ARG_CACHE_RAM=0关闭历史prompt RAM缓存，实际日志确认禁用。原服务未重启。',
    '旧缓存设置曾触发512MiB低内存保护；这些尝试与最终设置不合并延迟。最终每条均采样全设备显存及本服务进程树RSS，不是CUDA分配器精确峰值。',
    '输出协议rag-json-v2.2将引用ID限制在实际可见片段中，因而ID有效性属于工程约束；不得解读为引用支持度或事实忠实度提升。旧版本错误引用完整保留。LFM有1条输出达到统一512 token上限而无法解析，原输出保留，不重试挑选更好回答。',
    '所有统计来自 `D:/RAGScope_Runtime/runs/20261014_v2/PILOT_AUDIT.json` 及其逐文件哈希。正式E2/E3尚待完整检索上下文冻结。', '']
Path('docs/execution/PILOT_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False))
