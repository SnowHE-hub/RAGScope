"""Actual matched60 workflow generation in serial fresh-process blocks of20."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'scripts'))
from freeze_generation_v2 import model_client_factory
from filelock import FileLock
from ragscope.experiments.ledger import RecordStore,read_records
from ragscope.experiments.runtime import configure,require_ram
from ragscope.generation.pipeline import RAGPipeline
from ragscope.indexing.identity import artifact,atomic_json,canonical_json,signature
from ragscope.v3.budget import Budget
from ragscope.v3.generation_protocols import ProtocolClient,protocol_messages,protocol_schema
from ragscope.v3.local_runtime import ManagedOllama,counted_chat
from ragscope.v3.records import canonical_payload


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--worker',action='store_true')
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    manifest_path=run/'S4_WORKFLOW_CONTEXT_MANIFEST.json';manifest=json.loads(manifest_path.read_text('utf8'))
    if manifest['status']!='PASS_FROZEN_WORKFLOW_CONTEXTS' or artifact(manifest['inputs']['path'])!=manifest['inputs']:
        raise ValueError('Workflow inputs changed or incomplete')
    entries=[json.loads(l) for l in Path(manifest['inputs']['path']).read_text('utf8').splitlines()]
    if len(entries)!=180:raise ValueError('Workflow matrix incomplete')
    config={'manifest':artifact(manifest_path),'runner':artifact(__file__),'record_schema':artifact(PROJECT/'ragscope/v3/records.py'),
        'protocol':'G0','max_output_tokens':512,'num_ctx':4096,'seed':42,'temperature':0,
        'formal_input_n':180,'fresh_process_block_size':20,'answer_cache':False,'human_review_status':'NOT_REVIEWED'}
    if a.dry_run:
        print(canonical_json({'formal_requests':180,'warmups_expected':9,'stage_cap':800,'protected_S6':1800,
            'config':config,'gold_read':False}));return 0
    journal=run/'S4_WORKFLOW_GENERATION.jsonl'
    if journal.exists() and not a.resume:raise ValueError('Existing results require resume')
    if not a.worker:
        with FileLock(str(run/'s4_sequence.lock'),timeout=0):
            for segment in range(12):
                rows=read_records(journal);done={r['record_id'] for r in rows}
                pending=[e for e in entries if 'S4:workflow:'+e['input_id'] not in done]
                if not pending:break
                log=run/('S4_WORKFLOW_SEGMENT_'+str(time.time_ns())+'.log')
                cmd=[sys.executable,str(Path(__file__).resolve()),'--campaign',str(run),'--resume','--worker']
                with log.open('x',encoding='utf8') as out:result=subprocess.run(cmd,cwd=PROJECT,stdout=out,stderr=subprocess.STDOUT)
                atomic_json(log.with_suffix('.exit.json'),{'actual_exit_code':result.returncode,'command':cmd,'log':artifact(log)})
                print(canonical_json({'event':'S4_SEGMENT_TERMINAL','segment':segment,'actual_exit_code':result.returncode,
                    'formal_records':len(read_records(journal))}),flush=True)
                if result.returncode:return result.returncode
            rows=read_records(journal)
            result={'status':'PASS_180_ACTUAL_WORKFLOW_GENERATIONS' if len(rows)==180 else 'INCOMPLETE',
                'formal_n':len(rows),'questions_n':60,'records':artifact(journal),'budget':Budget(run).snapshot(),
                'router_training_gate':'PENDING_INDEPENDENT_UTILITY_VECTOR','human_review_status':'NOT_REVIEWED'}
            atomic_json(run/'S4_WORKFLOW_GENERATION_COMPLETION.json',result)
            return 0 if len(rows)==180 else 2
    require_ram(2*1024**3)
    with FileLock(str(run/'writer.lock'),timeout=0):
        store=RecordStore(journal,config,max_records=180)
        warms=RecordStore(run/'S4_WORKFLOW_WARMUPS.jsonl',config,max_records=20)
        pending=[e for e in entries if 'S4:workflow:'+e['input_id'] not in store.ids][:20]
        if not pending:return 0
        dev={json.loads(l)['query_id'] for l in (runtime/'corpora/20261014_v2/queries_dev.jsonl').read_text('utf8').splitlines()}
        original,_=model_client_factory(manifest['generator_spec'],dev)
        if original.identity!=manifest['generator_identity']:raise ValueError('Frozen generator changed')
        budget=Budget(run)
        with ManagedOllama(run,runtime/'tools/ollama-0.34.4/ollama.exe',num_ctx=4096) as service:
            for kind,e in [('warmup',pending[0])]+[('formal',r) for r in pending]:
                rid='S4:workflow:'+e['input_id'] if kind=='formal' else 'S4:workflow_warmup:'+str(len(warms.ids))
                require_ram(2*1024**3)
                client=ProtocolClient(original,service,protocol='G0');messages=protocol_messages(e['workflow_question'],e['context_bundle'],'G0')
                response={};parsed=None;error=None;began=time.perf_counter()
                try:
                    response=counted_chat(run,service,client,request_id=rid,stage='S4_task_workflows_router',messages=messages,
                        max_tokens=512,temperature=0,output_schema=protocol_schema(e['context_bundle'],'G0'),purpose='workflow_'+kind)
                    output=RAGPipeline.parse_output(response['content']);parsed={k:output[k] for k in ('answer','citations','abstain')}
                    if client.last_metadata.get('token_count_match') is not True:raise RuntimeError('Prompt token mismatch')
                    visible={p['chunk_id'] for p in e['context_bundle']['pieces']}
                    if any(c not in visible for c in parsed['citations']):raise ValueError('Invisible citation')
                except Exception as exc:
                    with budget.lock:
                        if rid not in budget._state()['reserved']:raise
                    error={'kind':type(exc).__name__,'message':str(exc)}
                    if not response:response={'raw_backend_response':client.raw_response,'backend_metadata':client.last_metadata}
                row=canonical_payload({'question_id':e['query_id'],'dataset_role':e['dataset_role'],'workflow':e['workflow'],
                    'question':e['question'],'exposure_status':'EXPOSED_DIAGNOSTIC' if e['dataset_role']=='D_search' else 'DEVELOPMENT_VALIDATION',
                    'task_type_source':e['task_type_source'],'task':e['task'],'user_scope':e['scope'],
                    'model_identity':client.identity,'prompt_hash':signature(messages),
                    'actual_prompt_tokens':client.last_metadata.get('prompt_eval_count'),
                    'context_intervals':[{'doc_id':p['doc_id'],'start':p['canonical_start'],'end':p['canonical_end']} for p in e['context_bundle']['pieces']],
                    'packing_drops':e['context_bundle']['truncations'],'workflow_audit':e['workflow_audit'],
                    'raw_response':response,'parsed_response':parsed,'success':error is None,'error_type':error,
                    'stop_reason':client.last_metadata.get('done_reason'),'output_tokens':client.last_metadata.get('eval_count'),
                    'wall_time_ms':(time.perf_counter()-began)*1000,'measured_resource':client.last_metadata.get('resources'),
                    'budget_ticket':rid,'actual_sent_payload':client.sent_payload,
                    'proxy_unknowns':['claim_f1_proxy','support_proxy','citation_completeness_proxy','task_keypoint_coverage'],
                    'cost_scope':'actual_frozen_workflow_generation_not_online_retrieval_E2E'},campaign_id=run.name,task_id='S4_workflow_'+kind)
                (store if kind=='formal' else warms).add(rid,row)
                print(canonical_json({'event':'S4_GENERATION','formal_records':len(store.ids),'workflow':e['workflow'],'error':error}),flush=True)
                if error and error['kind']!='ValueError':raise RuntimeError('Recorded backend failure; stop bounded block')
    return 0


if __name__=='__main__':raise SystemExit(main())
