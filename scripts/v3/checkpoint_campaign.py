"""Observe live workers and budget; save an accurate, resumable v3 checkpoint."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
import psutil
from ragscope.indexing.identity import atomic_json, canonical_json, signature
from ragscope.v3.budget import Budget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    args = parser.parse_args()
    run = Path(args.campaign).resolve()
    if run.parent != Path('D:/RAGScope_Runtime/runs').resolve() or not run.name.startswith('v3_'):
        raise ValueError('Wrong campaign')
    status_path = PROJECT / 'docs/execution/v3/STATUS.json'
    status = json.loads(status_path.read_text('utf-8'))
    live = []
    for process in psutil.process_iter(['pid', 'name', 'exe', 'cmdline', 'create_time']):
        try:
            info = process.info
            argv = info['cmdline'] or []
            if info['name'].lower() == 'python.exe' and len(argv) > 1 and argv[1].replace('\\', '/') in {
                    'scripts/v3/run_atomization.py', 'scripts/v3/resume_atomization.py',
                    'scripts/v3/calibrate_gpu_claim_judge.py', 'scripts/v3/score_claim_metrics.py',
                    'scripts/v3/run_context_judge.py'} and str(run).replace('\\', '/') in [a.replace('\\', '/') for a in argv]:
                live.append(info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    progress_path = run / 'ATOMIZATION_PROGRESS.json'
    progress = json.loads(progress_path.read_text('utf-8')) if progress_path.exists() else {}
    budget = Budget(run).snapshot()
    now = datetime.now(timezone.utc)
    status.update(updated_utc=now.isoformat(), updated_local=now.astimezone().isoformat(),
                  active_model_processes=live, budget_snapshot=budget,
                  new_llm_requests=budget['requests_reserved'], atomization_progress=progress,
                  last_local_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip())
    live_atom = any(p['cmdline'][1].replace('\\', '/').endswith('atomization.py') for p in live)
    live_scoring = any(p['cmdline'][1].replace('\\', '/').endswith('score_claim_metrics.py') for p in live)
    quotes_complete = progress.get('recorded_groups') == progress.get('expected_groups') == 558 and not live_atom
    status['tasks'].update({
        'S1.2_actual_atomic_claim_scoring': 'RUNNING_FULL_SCORING' if live_scoring else 'RUNNING_SEGMENTATION' if live_atom else 'SEGMENTATION_COMPLETE_WITH_10_RETAINED_FAILURES' if quotes_complete else 'SEGMENTATION_REQUIRES_TERMINAL_AUDIT',
        'S1.3_qa_style_nli_calibration': 'FAIL_QA_WRAPPER_GATE_RETAINED',
        'S1.3_narrow_qa_completion_calibration': 'PASS_80_CONSTRUCTED_TESTS_NOT_NATURAL_ACCURACY',
    })
    status['completed_evidence'].update(quote_groups_recorded=progress.get('recorded_groups', 0),
        quote_groups_valid=progress.get('valid_groups', 0), quote_groups_failed=progress.get('failures', 0),
        nli_qa_style_tests=72, nli_narrow_completion_tests=80)
    status['tasks']['S1.2_claim_support_citation_metrics'] = 'RUNNING_REAL_FULL_SCORING' if live_scoring else 'IMPLEMENTED_PENDING_REAL_FULL_SCORING'
    score_path = run/'claim_scores_cpu_v3/SUMMARY.json'
    if score_path.exists():
        scores = json.loads(score_path.read_text('utf-8'))
        status['completed_evidence'].update(actual_claim_score_answers=scores['answers'],
            actual_claim_score_relationships=scores['relationships'])
        if scores['answers'] == scores['expected'] == 5720:
            status['tasks']['S1.2_claim_support_citation_metrics'] = 'PASS_FULL_SCORING_WITH_UNCERTAINTY'
    if (run/'NLI_GPU_SCORING_CALIBRATION.json').exists():
        calibration = json.loads((run/'NLI_GPU_SCORING_CALIBRATION.json').read_text('utf-8'))
        status['tasks']['S1.3_actual_gpu_stack_calibration'] = calibration['status']
        status['completed_evidence']['actual_gpu_calibration_tests'] = sum(v['n'] for v in calibration['families'].values())
    if (run/'AUXILIARY_SPLIT_VALIDATION.json').exists():
        auxiliary = json.loads((run/'AUXILIARY_SPLIT_VALIDATION.json').read_text('utf-8'))
        if auxiliary['status'] == 'PASS_INDEPENDENT_SCHEMA_CORRECT_VALIDATION' and auxiliary['primary_overlap_n'] == 0:
            status['tasks']['S0.3_scope40_synthetic_holdout'] = 'PASS_FROZEN_SYNTHETIC_NOT_HUMAN_REVIEWED'
            status['tasks']['S0.3_task_source30_papers'] = 'PASS_SOURCE_INPUTS_REFERENCES_PENDING'
            status['completed_evidence'].update(H_scope=40, H_tasks_source_papers=30)
    if (run/'CONTEXT_JUDGEMENT_INPUTS.json').exists():
        context = json.loads((run/'CONTEXT_JUDGEMENT_INPUTS.json').read_text('utf-8'))
        status['tasks']['S1.4_context_only_sufficiency_inputs'] = 'PASS_INPUTS_ONLY_JUDGE_PENDING'
        status['completed_evidence']['context_only_judge_inputs'] = context['unique_question_context_n']
    if (run/'OLD_OUTPUT_ATTRIBUTION.json').exists():
        status['tasks']['S1.5_model_scale_length_and_lora_latency_decomposition'] = 'PASS_OLD_DESCRIPTIVE_DECOMPOSITION'
    if (run/'POLAR_PREFIX_DIAGNOSTIC.json').exists():
        status['tasks']['S1.5_yes_no_prefix_vs_word_score'] = 'PASS_DIAGNOSTIC_NOT_FACTUAL_ACCURACY'
    status['next_action'] = ('Observe the exact current CPU scoring worker/session24620; do not relaunch while live. '
        'After full scoring actually exits, inspect 5720 records, UNKNOWN bounds and natural blind subset, then '
        'run calibrated 9B context-only judging. H_tasks source-backed references remain pending. Then S2-S6.')
    status['next_command'] = (
        ". ./scripts/runtime_env.ps1\n"
        "& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' scripts/v3/checkpoint_campaign.py --campaign '" + str(run).replace('\\', '/') + "'")
    status['next_command_status'] = 'READ_ONLY_OBSERVATION_AND_CHECKPOINT; no model call'
    status['next_tool_call'] = {'tool': 'write_stdin', 'args': {'session_id': 24620,
        'chars': '', 'yield_time_ms': 1000, 'max_output_tokens': 1500}}
    status.pop('resume_command_after_verified_exit', None)
    observation = {'observed_utc': now.isoformat(), 'live_processes': live, 'budget': budget,
                   'atomization': progress, 'process_observation_not_terminal_exit_receipt': True}
    observation['observation_hash'] = signature(observation)
    with (run / 'CHECKPOINT_OBSERVATIONS.jsonl').open('a', encoding='utf-8', newline='\n') as handle:
        handle.write(canonical_json(observation) + '\n'); handle.flush(); os.fsync(handle.fileno())
    for destination in (status_path, run / 'STATUS.json'):
        atomic_json(destination, status)
    recovery = ('# v3 当前恢复点\n\n' + now.isoformat() + '\n\n'
        + 'Campaign `' + str(run) + '`；原训练 USER_MANUAL_RESUME，不自动恢复。\n\n'
        + '当前实际预算：' + str(budget['requests_reserved']) + ' 次请求；输出已扣除/保留 '
        + str(budget['output_tokens_charged_or_reserved']) + ' tokens；以 BUDGET.jsonl 为准。\n\n'
        + '已核验运行进程：`' + canonical_json(live) + '`。句柄未知时以 PID、创建时间、exe、argv 联合核验，不能因观察超时重启。\n\n'
        + '旧 V2 分割保存 83 组（含 3 个失败）；V3 实际 187 请求、183 成功、4 契约失败，另 371 组资源预检阻断而未发请求。V4 只续跑未发请求的组，所有原记录保留。\n\n'
        + 'NLI QA 校准：完整句 24/24，短片段 6/24，问题答案包装 12/24；后两者不得承担自动事实门槛，人工 NOT_REVIEWED。\n\n'
        + '参考答案不可见的窄 QA 完句规则实际 80/80 构造校准通过；其余语法/短片段仍 UNKNOWN，不宣称自然语料准确率。严禁与分割同时加载 CPU 裁判，以免再次耗尽内存余量。\n\n'
        + '下一条准确命令：\n```powershell\n' + status['next_command'] + '\n```\n\n'
        + 'V4实际终端session57682已exit2：558真实请求齐全，548有效、10契约失败保留；进程已不存在，私有服务卸载/退出。不得重跑分割。\n\n'
        + '实际GPU裁判210构造检查全部通过，session46755 exit0。评分12条冒烟因均UNKNOWN没有关系调用；原空缓存汇总bug session29339 exit1，已保存源与日志，修复后复用12条且actualexit0。\n\n'
        + 'GPU全量session80992/66941均actualexit1，分别在可用内存3153125376/3206303744字节触发3GiB守卫；其部分结果保留。流式修复后仍不足，GPU全量停止有界重试。\n\n'
        + '唯一CPU兼容全量正在session24620执行，输出 claim_scores_cpu_v3/；使用已校准CPU依赖栈与相同FP32模型，不混入GPU部分分数。只观察实际PID/创建时间/argv，不并行加载任何裁判。\n\n'
        + '40范围题已封存，30任务来源论文已冻结且与主分割241篇零交叉；H_tasks参考未生成。117个独立问题/上下文输入已冻结。\n\n'
        + '分割确认完整558组并实际退出后，依次执行（一个GPU模型，严禁并发）：\n```powershell\n'
        + ". ./scripts/runtime_env.ps1\n"
        + "& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' scripts/v3/run_context_judge.py --campaign 'D:/RAGScope_Runtime/runs/v3_20261015_014506' --resume\n```\n\n"
        + '每条实际终态和检查通过后才执行下一条。GPU工程门槛失败不能自动进入全量。S2–S6尚未执行，Goal active。\n')
    for destination in (PROJECT / 'docs/execution/v3/RECOVERY.md', run / 'RECOVERY.md'):
        destination.write_text(recovery, encoding='utf-8')
    print(canonical_json(observation))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
