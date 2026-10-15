"""Matched real baseline/task-action answers for the valid source-backed development tasks."""
import argparse
import json
from pathlib import Path
import sys
import time
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'scripts'))
from freeze_generation_v2 import model_client_factory
from filelock import FileLock
from ragscope.experiments.ledger import RecordStore
from ragscope.experiments.runtime import configure,require_ram
from ragscope.generation.pipeline import RAGPipeline
from ragscope.indexing.identity import artifact,atomic_json,canonical_json,signature
from ragscope.v3.budget import Budget
from ragscope.v3.generation_protocols import ProtocolClient,protocol_messages,protocol_schema
from ragscope.v3.local_runtime import ManagedOllama,counted_chat
from ragscope.v3.records import canonical_payload


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    manifest_path=run/'S4_DEVELOPMENT_TASK_CONTEXT_MANIFEST.json';manifest=json.loads(manifest_path.read_text('utf8'))
    if manifest['status']!='PASS_FROZEN_SOURCE_TASK_CONTEXTS' or artifact(manifest['inputs']['path'])!=manifest['inputs']:raise ValueError('Task contexts not frozen')
    entries=[json.loads(l) for l in Path(manifest['inputs']['path']).read_text('utf8').splitlines()]
    config={'runner':artifact(__file__),'inputs':artifact(manifest_path),'max_output_tokens':512,'temperature':0,'seed':42,
        'protocol':'G0','answer_cache':False,'metric_version':'independent source-keypoint scoring follows generation','human_review_status':'NOT_REVIEWED'}
    require_ram(2*1024**3)
    with FileLock(str(run/'writer.lock'),timeout=0):
        store=RecordStore(run/'S4_DEVELOPMENT_TASK_GENERATION.jsonl',config,max_records=len(entries))
        warm=RecordStore(run/'S4_DEVELOPMENT_TASK_WARMUP.jsonl',config,max_records=1)
        if (store.ids or warm.ids) and not a.resume:raise ValueError('Existing task results need resume')
        pending=[e for e in entries if 'S4:source_task:'+e['input_id'] not in store.ids]
        dev={json.loads(l)['query_id'] for l in (runtime/'corpora/20261014_v2/queries_dev.jsonl').read_text('utf8').splitlines()}
        original,_=model_client_factory(manifest['generator_spec'],dev)
        if original.identity!=manifest['generator_identity']:raise ValueError('Pinned generator differs')
        budget=Budget(run)
        if pending:
            with ManagedOllama(run,runtime/'tools/ollama-0.34.4/ollama.exe',num_ctx=4096) as service:
                tasks=([('warmup',pending[0])] if not warm.ids else [])+[('formal',e) for e in pending]
                for kind,e in tasks:
                    require_ram(2*1024**3);client=ProtocolClient(original,service,protocol='G0')
                    rid='S4:source_task:'+e['input_id'] if kind=='formal' else 'S4:source_task:warmup'
                    messages=protocol_messages(e['question'],e['context_bundle'],'G0');response={};parsed=None;error=None;began=time.perf_counter()
                    try:
                        response=counted_chat(run,service,client,request_id=rid,stage='S4_task_workflows_router',messages=messages,
                            max_tokens=512,temperature=0,output_schema=protocol_schema(e['context_bundle'],'G0'),purpose='source_task_'+kind)
                        output=RAGPipeline.parse_output(response['content']);parsed={k:output[k] for k in ('answer','citations','abstain')}
                        if client.last_metadata.get('token_count_match') is not True:raise RuntimeError('Task prompt count mismatch')
                        if any(c not in {p['chunk_id'] for p in e['context_bundle']['pieces']} for c in parsed['citations']):raise ValueError('Invisible citation')
                    except Exception as exc:
                        with budget.lock:
                            if rid not in budget._state()['reserved']:raise
                        error={'kind':type(exc).__name__,'message':str(exc)}
                        if not response:response={'raw_backend_response':client.raw_response,'backend_metadata':client.last_metadata}
                    row=canonical_payload({'question_id':e['query_id'],'dataset_role':'DEVELOPMENT_SYNTHETIC','task_kind':e['task_kind'],
                        'condition':e['condition'],'input_id':e['input_id'],'exposure_status':'EXPOSED_DEVELOPMENT_AI_AUTHORED',
                        'user_scope':e['scope'],'task_type_source':'SYNTHETIC_AI_SOURCE_VERIFIED_NOT_HUMAN','model_identity':client.identity,
                        'prompt_hash':signature(messages),'actual_prompt_tokens':client.last_metadata.get('prompt_eval_count'),
                        'context_intervals':[{'doc_id':p['doc_id'],'start':p['canonical_start'],'end':p['canonical_end']} for p in e['context_bundle']['pieces']],
                        'retriever_candidates':e['retriever_candidates'],'packing_drops':e['context_bundle']['truncations'],
                        'raw_response':response,'parsed_response':parsed,'success':error is None,'error_type':error,
                        'stop_reason':client.last_metadata.get('done_reason'),'output_tokens':client.last_metadata.get('eval_count'),
                        'wall_time_ms':(time.perf_counter()-began)*1000,'measured_resource':client.last_metadata.get('resources'),
                        'budget_ticket':rid,'actual_sent_payload':client.sent_payload,
                        'proxy_unknowns':['claim_f1_proxy','support_proxy','citation_completeness_proxy','source_keypoint_entailment'],
                        'cost_scope':'actual frozen task-context generation; not onlineE2E'},campaign_id=run.name,task_id='S4_source_task_'+kind)
                    (store if kind=='formal' else warm).add(rid,row)
                    print(canonical_json({'event':'SOURCE_TASK_GENERATION','formal_records':len(store.ids),'condition':e['condition'],'error':error}),flush=True)
                    if error and error['kind']!='ValueError':raise RuntimeError('Task backend failure retained; stop bounded batch')
        result={'status':'PASS_MATCHED_SOURCE_TASK_GENERATIONS','formal_n':len(store.ids),'questions_n':manifest['questions_n'],
            'warmups':len(warm.ids),'records':artifact(store.path),'contexts':artifact(manifest_path),'human_review_status':'NOT_REVIEWED','budget':budget.snapshot()}
        atomic_json(run/'S4_DEVELOPMENT_TASK_GENERATION_COMPLETION.json',result);print(canonical_json(result));return 0


if __name__=='__main__':raise SystemExit(main())
