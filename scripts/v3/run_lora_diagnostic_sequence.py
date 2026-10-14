"""Serial isolated mandatory S5 jobs; record actual exits and continue independent checks."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from filelock import FileLock
from ragscope.experiments.ledger import RecordStore
from ragscope.experiments.runtime import configure
from ragscope.indexing.identity import artifact,atomic_json,canonical_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    jobs=[('merged24','run_lora_runtime_diagnostics.py',['--condition','merged','--resume']),
          ('supervision8','run_lora_supervision_smoke.py',['--resume'])]
    jobs += [('fixed64_'+c,'run_lora_fixed_token_benchmark.py',['--condition',c,'--resume']) for c in ('base','unmerged','merged')]
    jobs += [('independent_audit','audit_lora_runtime_diagnostics.py',[])]
    config={'runner':artifact(__file__),'jobs':jobs,
        'sources':[artifact(PROJECT/'scripts/v3'/s) for s in sorted({j[1] for j in jobs})],
        'python':str(runtime/'envs/train/Scripts/python.exe'),'retry_policy':'one dispatch per job, no automatic retry',
        'original_training_policy':'USER_MANUAL_RESUME'}
    with FileLock(str(run/'s5_sequence.lock'),timeout=0):
        store=RecordStore(run/'S5_SEQUENCE_EXITS.jsonl',config,max_records=len(jobs))
        if store.ids and not a.resume:raise ValueError('Existing sequence requires resume')
        for name,script,extra in jobs:
            if name in store.ids:continue
            if name=='fixed64_merged' and not (run/'S5_MERGED_MODEL_MANIFEST.json').exists():
                store.add(name,{'status':'BLOCKED_PREREQUISITE_MERGED_COPY','actual_exit_code':None});continue
            command=[config['python'],str(PROJECT/'scripts/v3'/script),'--campaign',str(run),*extra]
            log=run/('S5_SEQUENCE_'+name+'.log')
            if log.exists():raise ValueError('Unreconciled dispatch log; preserve instead of rerunning')
            print(canonical_json({'event':'S5_DISPATCH','job':name,'command':command}),flush=True)
            with log.open('x',encoding='utf8') as out:
                result=subprocess.run(command,cwd=PROJECT,stdout=out,stderr=subprocess.STDOUT)
            row={'status':'ACTUAL_EXIT_0' if result.returncode==0 else 'ACTUAL_FAILURE_RETAINED',
                'actual_exit_code':result.returncode,'command':command,'log':artifact(log)}
            store.add(name,row);print(canonical_json({'event':'S5_TERMINAL','job':name,**row}),flush=True)
        failed=[r['record_id'] for r in store.records if r['status']!='ACTUAL_EXIT_0']
        atomic_json(run/'S5_SEQUENCE_COMPLETION.json',{'status':'COMPLETE_WITH_BLOCKS' if failed else 'PASS_ALL_DISPATCHES',
            'failed_or_blocked_jobs':failed,'jobs':artifact(store.path),'original_training_restarted':False})
        return 2 if failed else 0


if __name__=='__main__':raise SystemExit(main())
