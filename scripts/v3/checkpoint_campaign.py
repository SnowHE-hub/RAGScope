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
                    'scripts/v3/run_context_judge.py', 'scripts/v3/run_protocol_study.py',
                    'scripts/v3/build_task_references.py'} and str(run).replace('\\', '/') in [a.replace('\\', '/') for a in argv]:
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
    if (run/'CLAIM_SCORING_RESOURCE_BLOCK.json').exists():
        blocked = json.loads((run/'CLAIM_SCORING_RESOURCE_BLOCK.json').read_text('utf-8'))
        status['tasks']['S1.2_claim_support_citation_metrics'] = 'BLOCKED_RESOURCE_BOUNDED_ATTEMPTS_EXHAUSTED'
        status['tasks']['S1.3_natural_blind_review'] = 'PARTIAL_12_SCORE_ROWS_FULL_REFERENCE_SUPPORT_BLOCKED'
        status['completed_evidence']['actual_claim_score_answers'] = blocked['actual_cpu_partial_answers']
        status['completed_evidence']['actual_claim_score_relationships'] = blocked['attempts'][-1]['relationships']
        status['stages']['S1'] = 'BLOCKED_REQUIRED_FULL_CLAIM_METRICS_OTHER_DIAGNOSTICS_PASS'
    if (run/'CONTEXT_DIAGNOSIS.json').exists():
        diagnosis = json.loads((run/'CONTEXT_DIAGNOSIS.json').read_text('utf-8'))
        status['tasks']['S1.4_context_only_sufficiency_inputs'] = 'PASS_117_ACTUAL_CONTEXT_PROXY_JUDGEMENTS'
        status['tasks']['S1.5_new_refusal_context_assessment'] = 'PASS_23_INDEPENDENT_PROXY_VERDICTS_NOT_HUMAN'
        status['completed_evidence']['lora_refusal_context_counts'] = diagnosis['new_refusal_context_counts']
    if (run/'S2_PROTOCOL_ANALYSIS.json').exists():
        protocol = json.loads((run/'S2_PROTOCOL_ANALYSIS.json').read_text('utf-8'))
        status['completed_evidence'].update(S2_main_matrix_records=protocol['matrix_n'], S2_G0_reused=144,
            S2_actual_smoke_calls=72, S2_new_main_requests=720)
        status['tasks']['S2.1_actual_direct_pipeline_smoke'] = 'PASS_72_CALLS_36_EQUAL_PAIRS_NO_TOKEN_OR_CHANNEL_MISMATCH'
        status['tasks']['S2.2_G0_G3_frozen24_matrix'] = 'PASS_864_RECORDS_OLD_EXPOSED_DIAGNOSTIC_ONLY'
        status['tasks']['S2.2_oracle_padded36'] = 'PENDING'
        status['tasks']['S2.2_prior_extra6'] = 'PENDING_6_PRIOR_CALLS_DONE_6_ADDITIONAL_PLANNED'
        status['stages']['S2'] = 'RUNNING_REQUIRED_SUPPLEMENT_PENDING'
        status['current_stage'] = 'S2_AND_S0_AUXILIARY_REFERENCES'
    if (run/'H_TASKS_CONSTRUCTION.json').exists():
        construction = json.loads((run/'H_TASKS_CONSTRUCTION.json').read_text('utf-8'))
        status['tasks']['S0.3_scope_tasks_synthetic_holdout'] = 'PASS_SYNTHETIC_SOURCE_CONSTRUCTION_WITH_RETAINED_TEACHER_FAILURES'
        status['completed_evidence']['H_tasks_valid_tasks'] = construction['valid_task_n']
        status['completed_evidence']['H_tasks_teacher_failed_or_blocked_papers'] = construction['failed_or_blocked_papers']
    if (run/'H_TASKS_SOURCE_REPAIR.json').exists():
        repaired = json.loads((run/'H_TASKS_SOURCE_REPAIR.json').read_text('utf-8'))
        status['completed_evidence']['H_tasks_valid_tasks'] = repaired['valid_task_n']
        status['completed_evidence']['H_tasks_invalid_task_branches_retained'] = repaired['failed_task_n']
        status['tasks']['S0.3_scope_tasks_synthetic_holdout'] = 'PASS_40_SCOPE_20_SOURCE_VERIFIED_SYNTHETIC_TASKS_FAILURES_RETAINED'
        status['stages']['S0'] = 'PASS_WITH_SYNTHETIC_SOURCE_FAILURES_RETAINED'
    status['next_action'] = ('H_tasks teacher session71248 actually exited0; no model job remains. '
        '20 source-verified synthetic task references frozen in questions_v2/gold_v2, 29 invalid branches preserved. Finish S2 required Oracle padded36 '
        'and six extra prior-knowledge probes, then S3 retrieval funnel and core-first packing. No more old NLI reloads.')
    status['next_command'] = (
        ". ./scripts/runtime_env.ps1\n"
        "& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' scripts/v3/checkpoint_campaign.py --campaign '" + str(run).replace('\\', '/') + "'")
    status['next_command_status'] = 'READ_ONLY_OBSERVATION_AND_CHECKPOINT; no model call'
    status['next_tool_call'] = {'tool': 'exec_command', 'args': {'cmd':
        "Get-Content -LiteralPath 'D:/RAGScope_Runtime/runs/v3_20261015_014506/S2_PROTOCOL_INPUTS.jsonl' -TotalCount 1",
        'max_output_tokens': 2200}}
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
        + 'CPU兼容session24620也actualexit1；335实际评分/230关系判断保留。全量旧NLI已到有界阻断，不再加载。所有5720条覆盖/UNKNOWN及盲评120条部分包保存，未执行行不冒充模型结果。\n\n'
        + '40范围题已封存，30任务来源论文已冻结且与主分割241篇零交叉；H_tasks参考未生成。117个独立问题/上下文输入已冻结。\n\n'
        + '9B上下文独立校准59/60。session21014在104/117后资源exit1，6771只续跑13条actualexit0；117条完成。23新增拒答：YES8/NO12/UNKNOWN3，均为proxy，不是人评。\n\n'
        + 'S2输入冻结24题×2上下文，G0真实144复用。72新direct/pipeline调用session65647 exit0，36配对完全一致、无token/思考通道差异。主矩阵session84289 exit0，864完整记录（新720）独立计分验证；含6prior探针。\n\n'
        + 'H_tasks整篇来源session71248 actualexit0，30请求全部完成；初版16有效任务/20失败论文。一次零生成成本的空白格式修复和独立任务验证后，20有效任务、29失败分支保留；H_tasks.questions_v2/gold_v2为当前辅助人造任务集，不是官方QASPER或操作者盲测。无后台模型进程。\n\n'
        + '参考构建实际退出后，先核验 H_TASKS_CONSTRUCTION.json、全部失败与有效参考；再实现/执行尚未完成的S2 Oracle等长噪声36和额外prior6，之后S3–S6。不要重跑主矩阵或NLI。Goal active，execution_complete仍false。\n')
    for destination in (PROJECT / 'docs/execution/v3/RECOVERY.md', run / 'RECOVERY.md'):
        destination.write_text(recovery, encoding='utf-8')
    print(canonical_json(observation))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
