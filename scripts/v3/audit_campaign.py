"""Independent evidence audit; terminal workflow and complete execution are different."""
import argparse
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ragscope.experiments.ledger import read_records
from ragscope.indexing.identity import artifact,atomic_json,canonical_json,signature
from ragscope.v3.budget import Budget,STAGES
from ragscope.v3.splits import read_split
from ragscope.v3.auxiliary_public import read_auxiliary_public

DELIVERABLES = ['EXECUTIVE_SUMMARY_ZH.md','ROOT_CAUSE_VERDICTS.md','QUALITY_SCORECARD.md',
    'QUALITY_SCORECARD.json','GENERATION_PROTOCOL_DIAGNOSIS.md','RETRIEVAL_FUNNEL.md',
    'ROUTING_ABLATION.md','LORA_DIAGNOSIS.md','BEFORE_AFTER_CASES.md',
    'HUMAN_REVIEW_PACKAGE.md','ACCEPTANCE_V3.json','FINAL_RESOURCE_STATE.json']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True)
    p.add_argument('--independent',action='store_true',required=True);a=p.parse_args();run=Path(a.campaign).resolve()
    if run.parent!=Path('D:/RAGScope_Runtime/runs').resolve():raise ValueError('Wrong campaign')
    budget=Budget(run);state=budget._state();issues=[]
    if state['requests_reserved']>6000 or state['output_tokens_charged_or_reserved']>4000000:issues.append('Total budget exceeded')
    if any(state['stage_requests'][s]>limit for s,limit in STAGES.items()):issues.append('Stage budget exceeded')
    unresolved_tickets=sorted(set(state['reserved'])-set(state['finalized']))
    if unresolved_tickets:issues.append('Unfinalized budget tickets')
    reports=PROJECT/'docs/execution/v3';missing=[f for f in DELIVERABLES if not (reports/f).is_file()]
    if not (PROJECT/'LOCAL_RUN_V3.md').is_file():missing.append('LOCAL_RUN_V3.md')
    # Public holdout questions only. No evaluator gold is opened by this audit.
    public_counts={s:len(read_split(run,s,purpose='inference')) for s in ('D_search','D_confirm','H_QA')}
    public_counts.update({s:len(read_auxiliary_public(run,s)) for s in ('H_scope','H_tasks')})
    generation_counts={}
    for name,expected in [('S2_PROTOCOL_RESULTS.jsonl',864),('S4_WORKFLOW_GENERATION.jsonl',180),('S6_ONLINE_COST.jsonl',180)]:
        path=run/name
        if path.exists():
            rows=read_records(path);config=json.loads(path.with_suffix('.config.json').read_text('utf8'))
            if any(r.get('config_hash')!=signature(config) for r in rows):issues.append(name+' config mismatch')
            formal=[r for r in rows if r.get('task_id')=='S2_matrix' or r.get('reused_actual_model_result')] if name.startswith('S2_') else rows
            generation_counts[name]={'journal_rows_n':len(rows),'actual_formal_n':len(formal),'expected_formal_n':expected,'source':artifact(path)}
            if len(formal)!=expected:issues.append(name+' formal denominator mismatch')
        else:generation_counts[name]={'actual_formal_n':0,'expected_formal_n':expected,'source':None};issues.append(name+' missing required journal')
    acceptance_path=reports/'ACCEPTANCE_V3.json';acceptance=None
    if acceptance_path.exists():
        acceptance=json.loads(acceptance_path.read_text('utf8'))
        required=('workflow_finished','execution_complete','performance_status','release_status','human_review_status','requirements')
        if any(k not in acceptance for k in required):issues.append('Acceptance fields missing')
        if acceptance.get('human_review_status')!='NOT_REVIEWED':issues.append('No human review authority for changed status')
        for item in acceptance.get('requirements',[]):
            if item['status'] in {'PENDING','RUNNING'}:issues.append('Nonterminal requirement '+item['id'])
            if not item.get('evidence'):issues.append('Missing evidence '+item['id'])
            for ref in item.get('evidence',[]):
                if not Path(ref['path']).is_file() or artifact(ref['path'])!=ref:issues.append('Requirement evidence changed '+item['id'])
        blocked=[i['id'] for i in acceptance.get('requirements',[]) if i['status']=='BLOCKED']
        if blocked and acceptance.get('execution_complete'):issues.append('Required block cannot be execution_complete')
        ids={i['id'] for i in acceptance.get('requirements',[])}
        for clause in acceptance.get('source_requirement_lines',[]):
            controllers=clause.get('controlling_requirement_ids',[])
            if not controllers or any(controller not in ids for controller in controllers):
                issues.append('Unmapped master requirement line '+str(clause['master_line']))
    status='PASS_TERMINAL_EVIDENCE_AUDIT' if acceptance and not missing and not issues else 'INCOMPLETE_OR_INCONSISTENT_EVIDENCE'
    result={'status':status,'missing_deliverables':missing,'issues':issues,'public_split_counts':public_counts,
        'generation_counts':generation_counts,'budget':budget.snapshot(),'unfinalized_tickets':unresolved_tickets,
        'holdout_gold_read':False,'human_review_status':'NOT_REVIEWED',
        'execution_complete':acceptance.get('execution_complete',False) if acceptance else False,
        'verification_scope':'Artifact hashes, full listed journal hashes/denominators, public split seals, complete budget reservations and final terminal requirement receipts. It does not certify natural-language truth.',
        'own_auditor':artifact(__file__)}
    atomic_json(run/'INDEPENDENT_ACCEPTANCE_AUDIT.json',result)
    print(canonical_json(result));return 0 if status=='PASS_TERMINAL_EVIDENCE_AUDIT' else 3


if __name__=='__main__':raise SystemExit(main())
