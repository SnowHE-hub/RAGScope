"""Resume immutable v3 stages without rerunning completed or blocked experiments."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import psutil
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from filelock import FileLock
from ragscope.experiments.ledger import append_jsonl
from ragscope.experiments.runtime import configure
from ragscope.indexing.identity import artifact,atomic_json,canonical_json,signature
from ragscope.v3.budget import Budget


def verify_receipt_refs(value):
    if isinstance(value,dict):
        if {'path','bytes','sha256'}<=set(value):
            if artifact(value['path'])!={k:value[k] for k in ('path','bytes','sha256')}:
                raise ValueError('Terminal evidence reference changed: '+value['path'])
        else:
            for item in value.values():verify_receipt_refs(item)
    elif isinstance(value,list):
        for item in value:verify_receipt_refs(item)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',default='configs/v3/campaign.yaml')
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--stage',choices=['S0','S1','S2','S3','S4','S5','S6','acceptance'])
    p.add_argument('--until',choices=['S0','S1','S2','S3','S4','S5','S6','acceptance'],default='acceptance')
    a=p.parse_args();runtime=Path(configure(cpu_only=True)['home']).resolve()
    # JSON is a YAML subset; no extra parser/install is needed for the fixed plan.
    plan_path=(PROJECT/a.plan).resolve();plan=json.loads(plan_path.read_text('utf8'))
    run=Path(plan['campaign']).resolve();python=Path(plan['python']).resolve()
    if run.parent!=runtime/'runs' or not python.is_relative_to(runtime/'envs') or not python.is_file():raise ValueError('Invalid existing campaign/python')
    stages=plan['stage_order'];selected=[s for s in stages[:stages.index(a.until)+1] if not a.stage or s==a.stage]
    identity={'plan':artifact(plan_path),'master':artifact(PROJECT/'docs/execution/v3/MASTER_PLAN.md'),
              'steps':[{**s,'implementation':artifact(PROJECT/s['script'])} for s in plan['steps']],
              'budget_caps':plan['constraints'],
              'existing_stage_evidence':{s:artifact(run/name) for s,name in plan['existing_stage_evidence'].items()}}
    budget=Budget(run).snapshot();missing=[s for s,f in plan['existing_stage_evidence'].items() if s in selected and not (run/f).is_file()]
    if missing:raise ValueError('Existing stage evidence missing: '+','.join(missing))
    if a.dry_run:
        print(canonical_json({'status':'PASS_DRY_RUN','selected_stages':selected,'identity':identity,'budget':budget,
            'model_calls':0,'holdout_gold_reads':0,'available_GiB':psutil.virtual_memory().available/1024**3}));return 0
    execution_id=plan.get('execution_id','initial')
    if not execution_id.replace('_','').isalnum():raise ValueError('Invalid execution identity')
    directory=run/'campaign_execution'/execution_id;directory.mkdir(parents=True,exist_ok=True)
    lock_path=directory/'identity.json';history=directory/'commands.jsonl'
    with FileLock(str(run/'orchestrator.lock'),timeout=0):
        if lock_path.exists():
            if not a.resume:raise ValueError('Existing execution needs --resume')
            if json.loads(lock_path.read_text('utf8'))!=identity:raise ValueError('Execution config/source changed; use a new execution identity, not old records')
        else:atomic_json(lock_path,identity)
        previous=[json.loads(l) for l in history.read_text('utf8').splitlines()] if history.exists() else []
        exits=[]
        for step in plan['steps']:
            if step['stage'] not in selected:continue
            path=run/step['evidence']
            if path.is_file() and json.loads(path.read_text('utf8')).get('status')==step['required_status']:
                verify_receipt_refs(json.loads(path.read_text('utf8')))
                print(canonical_json({'step':step['id'],'event':'REUSED_VERIFIED_TERMINAL_EVIDENCE','evidence':artifact(path)}));continue
            attempts=sum(e.get('step')==step['id'] and e.get('event')=='FINISHED' for e in previous)
            if attempts>=3:
                print(canonical_json({'step':step['id'],'status':'BLOCKED_BOUNDED_ATTEMPTS_EXHAUSTED'}));exits.append(3);continue
            if step['requires_models'] and psutil.virtual_memory().available<3*1024**3:
                append_jsonl(history,{'step':step['id'],'event':'RESOURCE_DEFERRED','actual_exit_code':None,
                    'available_bytes':psutil.virtual_memory().available,'reason':'pre-batch3GiB floor; no model started'})
                exits.append(3);continue
            command=[str(python),str(PROJECT/step['script']),'--campaign',str(run),*step['args']]
            log=directory/('COMMAND_'+step['id']+'_'+str(attempts)+'.log')
            began=datetime.now(timezone.utc).isoformat();append_jsonl(history,{'step':step['id'],'event':'STARTED','argv':command,'identity_hash':signature(identity),'started_utc':began})
            with log.open('w',encoding='utf8') as handle:
                result=subprocess.run(command,cwd=PROJECT,stdout=handle,stderr=subprocess.STDOUT)
            receipt={'step':step['id'],'event':'FINISHED','argv':command,'actual_exit_code':result.returncode,
                'log':artifact(log),'identity_hash':signature(identity),'ended_utc':datetime.now(timezone.utc).isoformat()}
            append_jsonl(history,receipt);exits.append(result.returncode);print(canonical_json(receipt))
        return 0 if not any(exits) else 3


if __name__=='__main__':raise SystemExit(main())
