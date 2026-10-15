"""Synchronize terminal task state only after requirement-linked independent acceptance passes."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ragscope.indexing.identity import artifact,atomic_json,canonical_json
from ragscope.v3.budget import Budget


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args();run=Path(a.campaign).resolve()
    if run.parent!=Path('D:/RAGScope_Runtime/runs').resolve():raise ValueError('Wrong campaign')
    audit=json.loads((run/'INDEPENDENT_ACCEPTANCE_AUDIT.json').read_text('utf8'))
    acceptance=json.loads((run/'ACCEPTANCE_V3.json').read_text('utf8'))
    if audit['status']!='PASS_TERMINAL_EVIDENCE_AUDIT' or audit['issues'] or audit['missing_deliverables']:
        raise ValueError('Independent completion audit is not clean')
    if json.loads((run/'FINAL_RESOURCE_STATE.json').read_text('utf8'))['status']!='PASS_OWNED_RESOURCES_RELEASED':raise ValueError('Resources not released')
    status=json.loads((PROJECT/'docs/execution/v3/STATUS.json').read_text('utf8'))
    status.update({k:acceptance[k] for k in ('workflow_finished','execution_complete','performance_status','release_status','human_review_status')})
    status.update(updated_utc=datetime.now(timezone.utc).isoformat(),current_stage='TERMINAL_WITH_REQUIRED_BLOCKS',
        active_owned_processes=[],active_model_processes=[],budget_snapshot=Budget(run).snapshot(),
        new_llm_requests=Budget(run).snapshot()['requests_reserved'],
        next_action='No authorized campaign experiments remain. Preserve v2 default and sealed H_QA; original training USER_MANUAL_RESUME.',
        next_command='. ./scripts/runtime_env.ps1\n& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/v3/audit_campaign.py --campaign D:/RAGScope_Runtime/runs/v3_20261015_014506 --independent',
        next_command_status='READ_ONLY_REPRODUCTION_OF_FINAL_ACCEPTANCE',unresolved=acceptance['blocked_required_items'])
    status['stages']['S6']='TERMINAL_WITH_BLOCKED_REQUIRED_UNSEEN_COMPARISON'
    status['tasks'].update({'S6.1_candidate_lock':'NO_V3_CANDIDATE_ADMITTED',
        'S6.1_H_QA200':'BLOCKED_NO_ADMITTED_CANDIDATE','S6.2_H_scope40':'PASS_40_ACTUAL_SCOPE_INVARIANTS',
        'S6.2_H_tasks':'BLOCKED_NO_ADMITTED_TASK_WORKFLOW','S6.3_online180':'PASS_180_ACTUAL_ONLINE_AUDIT',
        'S6.3_application_cache':'PASS_REAL_APPLICATION_AND_SEPARATE_CACHE_TEST','S6.4_final_statistics':'BLOCKED_NO_FINAL_H_QA',
        'S6.UI':'PASS_REAL_API_AND_STREAMLIT_CONTROLS','S6.requirement_acceptance':'PASS_TERMINAL_EVIDENCE_AUDIT',
        'S6.resources':'PASS_OWNED_RESOURCES_RELEASED','S5.1_full_training_numeric_and_decode20':'PASS_FULL_SUPERVISION_NUMERIC_AND_DECODE20',
        'S5.2_runtime_independent_recount':'PASS_BASE_UNMERGED_MERGED_GATE_BLOCKED',
        'S1.3_calibration_and_natural_blind_inputs':'PASS_INPUTS_NATURAL_FULL_JUDGE_BLOCKED',
        'S1.3_natural_blind_review':'BLOCKED_FULL120_NATURAL_PROXY_AFTER_BOUNDED_ATTEMPTS'})
    status['completed_evidence'].update(S6_actual_online_formal_requests=180,S6_actual_online_warmups=6,S6_scope_requests=40,S6_real_application_answers=2,S6_v3_regression_passed=72)
    status.pop('next_tool_call',None)
    prior_commit=status.pop('last_local_commit',None)
    if prior_commit:status['local_commit_before_final_delivery']=prior_commit
    status['final_commit_authority']=str(run/'FINAL_GIT_COMMIT.json')+' and git rev-parse HEAD'
    for target in (PROJECT/'docs/execution/v3/STATUS.json',run/'STATUS.json'):atomic_json(target,status)
    recovery='''# v3最终恢复与复核

Campaign：D:/RAGScope_Runtime/runs/v3_20261015_014506。
workflow_finished=true，execution_complete=false，performance_status=INCONCLUSIVE，release_status=KEEP_V2_DEFAULT，human_review_status=NOT_REVIEWED。

180正式在线请求、6独立预热、40合成范围查询、两版本合计2真实应用回答与2答案缓存命中已完成；72v3功能回归通过。全部成功且身份一致的结果复用，不重跑旧矩阵、原子化、旧全量NLI或全库Dense。

必需受阻：旧完整claim/support/citation评分、120自然盲评全量proxy、完整块Dense、足够比较/总结标签及各3案例、adapter合并比较、无入围候选的正式H_QA/H_tasks配对及最终统计。失败、部分结果和UNKNOWN全部保留；不可替换裁判或挑已知子集追分。

H_QA200封存，未创建FINAL_EVALUATION_AUTHORIZATION.json；不能用未入围候选读gold。后续研究应使用新campaign和新holdout，当前不会继续调参。

本项目私有服务、11435/11437/8502及GPU/writer/orchestrator锁均已释放，精确PID/创建时间核验见FINAL_RESOURCE_STATE.json。原训练USER_MANUAL_RESUME，不自动恢复、重建定时器或修改日更/WSL。

下一条准确复核命令（不调用模型）：
```powershell
. ./scripts/runtime_env.ps1
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/v3/audit_campaign.py --campaign D:/RAGScope_Runtime/runs/v3_20261015_014506 --independent
```

启动和v2回退见LOCAL_RUN_V3.md。验收事实以ACCEPTANCE_V3.json及独立审计为准；Git提交本地保存，不push。
'''
    for target in (PROJECT/'docs/execution/v3/RECOVERY.md',run/'RECOVERY.md'):target.write_text(recovery,encoding='utf8')
    atomic_json(run/'FINAL_STATE_SYNCHRONIZATION.json',{'status':'PASS_TERMINAL_STATE_SYNC','runner':artifact(__file__),
        'acceptance':artifact(run/'ACCEPTANCE_V3.json'),'independent_audit':artifact(run/'INDEPENDENT_ACCEPTANCE_AUDIT.json'),
        'workflow_finished':True,'execution_complete':False})
    print(canonical_json({k:status[k] for k in ('workflow_finished','execution_complete','performance_status','release_status','human_review_status')}));return 0


if __name__=='__main__':raise SystemExit(main())
