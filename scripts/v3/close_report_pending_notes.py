"""Reconcile earlier stage reports with terminal S6 evidence, without rescoring answers."""
import argparse
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ragscope.experiments.ledger import read_records
from ragscope.indexing.identity import artifact,atomic_json,canonical_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args();run=Path(a.campaign).resolve()
    if run.parent!=Path('D:/RAGScope_Runtime/runs').resolve():raise ValueError('Wrong campaign')
    gate=json.loads((run/'CANDIDATE_LOCK.json').read_text('utf8'))
    if gate['status']!='NO_V3_CANDIDATE_ADMITTED':raise ValueError('Do not apply negative final note to a different candidate outcome')
    report=PROJECT/'docs/execution/v3';outputs=[]
    replacements={
        'GENERATION_PROTOCOL_DIAGNOSIS.md':{'后续仍需任务工作流和一次冻结终验，人工 NOT_REVIEWED。':'实际工作流与online180已完成；没有候选入围，正式holdout配对受阻，人工NOT_REVIEWED。'},
        'RETRIEVAL_FUNNEL.md':{'最终H_scope专项尚待冻结后执行。':'H_scope40已实际完成：20指定范围不越界、20缺对象澄清；合成范围策略成绩单列。',
            'core-first还需主要事实指标、一次holdout和真实在线成本。':'core-first主要事实差值仍UNKNOWN，实际online180没有效率收益，开发门槛未过；正式holdout配对受阻。'},
        'ROUTING_ABLATION.md':{'比较类仅5题，独立关键点标注及自然任务验证仍待补充；':'比较类仅5题；独立来源补充只2总结/1比较，仍不足以推广；',
            'H_tasks与H_scope仍封存于最终验证协议，不读它们来调参。':'H_scope40范围专项已完成；H_tasks最终比较受入围门槛阻断，不读参考来调参。',
            '最终online质量/成本、一次holdout、UI验收及人工审核状态尚需独立完成':'online180及工作台实连已完成，正式holdout配对受阻，人工审核NOT_REVIEWED'},
        'LORA_DIAGNOSIS.md':{'继续评估真正不同的工作流，保存每题完整效用向量。最终质量、支持性、引用及在线成本需按冻结候选和holdout协议验证；本报告不能替代S6。':'工作流180实际答案与240效用向量、online180及工作台实连均已完成。主要事实评分未知，没有候选满足开发门槛，正式holdout配对受阻；本LoRA诊断不替代质量终验。'}}
    for name,changes in replacements.items():
        path=report/name;text=path.read_text('utf8')
        for old,new in changes.items():text=text.replace(old,new)
        note='\n## 本轮最终门槛状态\n\n'
        if note not in text:
            text+=note+'真实online180、合成范围40和工作台实连完成。质量增益与非劣无法证明，效率门槛未过，保留v2默认；H_QA/H_tasks正式配对BLOCKED，execution_complete=false，human_review_status=NOT_REVIEWED。完整证据见QUALITY_SCORECARD.json与ACCEPTANCE_V3.json。\n'
        path.write_text(text,encoding='utf8');outputs.append(artifact(path))
    cold=[]
    for row in read_records(run/'S6_ONLINE_WARMUPS.jsonl'):
        response=(row.get('raw_response') or {}).get('raw_backend_response') or {}
        cold.append({'record_id':row['record_id'],'condition':row['condition'],
            'actual_warmup_total_seconds':row['wall_time_ms']/1000,
            'index_load_seconds_separate':row['index_load_seconds_separate'],
            'backend_model_load_seconds':response['load_duration']/1e9 if 'load_duration' in response else None,
            'interpretation':'Observed startup/warmup request; not a formal hot request or standalone full application cold-start SLA.'})
    atomic_json(run/'S6_COLD_START_OBSERVATIONS.json',{'status':'PASS_ACTUAL_WARMUP_METADATA_EXPORT','rows':cold,
        'source':artifact(run/'S6_ONLINE_WARMUPS.jsonl'),'formal_hot_requests':180,'new_model_calls':0})
    path=report/'QUALITY_SCORECARD.md';text=path.read_text('utf8')
    if 'S6_COLD_START_OBSERVATIONS.json' not in text:
        text+='\n冷启动/预热实测单列：`'+str(run/'S6_COLD_START_OBSERVATIONS.json')+'`，含6条实际预热总时、索引加载及后端模型加载元数据；不是冷启动SLA。\n'
        path.write_text(text,encoding='utf8')
    atomic_json(run/'FINAL_REPORT_RECONCILIATION.json',{'status':'PASS_STAGE_REPORTS_RECONCILED','outputs':outputs,'runner':artifact(__file__),'new_generation_requests':0})
    print(canonical_json({'status':'PASS_STAGE_REPORTS_RECONCILED','updated_reports':len(outputs)}));return 0


if __name__=='__main__':raise SystemExit(main())
