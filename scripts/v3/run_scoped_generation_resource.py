"""Single bounded streaming fix for the partially completed D_confirm study."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'scripts'));sys.path.insert(0,str(PROJECT/'scripts/v3'))
import run_scoped_generation as base
from ragscope.indexing.identity import artifact,atomic_json,canonical_json,signature
from ragscope.experiments.runtime import configure,require_ram
from ragscope.experiments.ledger import RecordStore,read_records
from ragscope.v3.splits import read_split


def worker(run,runtime,segment):
    from filelock import FileLock
    manifest_path=run/'S3_SCOPED_CONTEXT_MANIFEST.json';manifest=json.loads(manifest_path.read_text('utf8'))
    if artifact(manifest['inputs']['path'])!=manifest['inputs']:raise ValueError('Frozen input changed')
    entries=[];i=0
    with Path(manifest['inputs']['path']).open(encoding='utf8') as f:
        for l in f:
            e=json.loads(l)
            if e['dataset_role']!='D_confirm':continue
            if segment*20<=i<(segment+1)*20:entries.append(e)
            i+=1
    if i!=200:raise ValueError('D_confirm paired population changed')
    labels={q['query_id']:q for q in read_split(run,'D_confirm',purpose='development')}
    original_path=run/'S3_SCOPED_GENERATION_D_confirm.jsonl'
    config={'inputs':artifact(manifest_path),'runner':artifact(__file__),'original_results':artifact(original_path),
        'fix':artifact(run/'S3_D_CONFIRM_RESOURCE_FIX.json'),'protocol':artifact(PROJECT/'ragscope/v3/generation_protocols.py'),
        'max_records_per_fresh_process':20,'seed':42,'temperature':0,'max_output_tokens':512,'num_ctx':4096,'answer_cache':False}
    with FileLock(str(run/'writer.lock'),timeout=0):
        store=RecordStore(run/'S3_SCOPED_GENERATION_D_confirm_resource.jsonl',config,max_records=200)
        warm=RecordStore(run/'S3_SCOPED_WARMUPS_D_confirm_resource.jsonl',config,max_records=20)
        for prior in read_records(original_path):
            payload=dict(prior)
            for key in ('record_hash','config_hash','recorded_at'):payload['original_'+key]=payload.pop(key)
            payload['raw_inference_unchanged']=True
            store.add(prior['record_id'],payload)
        pending=[e for e in entries if 'S3:D_confirm:'+e['input_id'] not in store.ids]
        if not pending:return 0
        dev={json.loads(l)['query_id'] for l in (runtime/'corpora/20261014_v2/queries_dev.jsonl').read_text('utf8').splitlines()}
        original,_=base.model_client_factory(manifest['generator_spec'],dev)
        budget=base.Budget(run)
        with base.ManagedOllama(run,runtime/'tools/ollama-0.34.4/ollama.exe',num_ctx=4096) as service:
            for kind,e in [('warmup',pending[0])]+[('formal',x) for x in pending]:
                require_ram(2*1024**3)
                rid='S3:D_confirm:'+e['input_id'] if kind=='formal' else 'S3:resource_warmup:D_confirm:'+str(segment)
                client=base.ProtocolClient(original,service,protocol='G0');messages=base.protocol_messages(e['question'],e['context_bundle'],'G0')
                response,parsed,error={},None,None;began=time.perf_counter()
                try:
                    response=base.counted_chat(run,service,client,request_id=rid,stage='S3_retrieval_packing',messages=messages,
                        max_tokens=512,output_schema=base.protocol_schema(e['context_bundle'],'G0'),purpose='S3_D_confirm_resource_'+kind)
                    result=base.RAGPipeline.parse_output(response['content']);parsed={k:result[k] for k in ('answer','citations','abstain')}
                    if client.last_metadata.get('token_count_match') is not True:raise RuntimeError('Actual prompt count mismatch')
                    visible={p['chunk_id'] for p in e['context_bundle']['pieces']}
                    if any(c not in visible for c in parsed['citations']):raise ValueError('Citation not visible')
                except Exception as exc:
                    with budget.lock:
                        if rid not in budget._state()['reserved']:raise
                    error={'kind':type(exc).__name__,'message':str(exc)}
                    if not response:response={'raw_backend_response':client.raw_response,'backend_metadata':client.last_metadata}
                q=labels[e['query_id']];answerable=q['answerability']=='answerable'
                metrics=base.overlap(parsed['answer'],base.reference_texts(q)) if parsed and not error and not parsed['abstain'] and answerable else None
                row=base.canonical_payload({'question_id':e['query_id'],'dataset_role':'D_confirm','exposure_status':'DEVELOPMENT_VALIDATION',
                    'task_type_source':'UNLABELED_OFFICIAL_QASPER','user_scope':e['scope'],'condition':e['condition'],'input_id':e['input_id'],
                    'prompt_hash':signature(messages),'model_identity':client.identity,'actual_prompt_tokens':client.last_metadata.get('prompt_eval_count'),
                    'context_intervals':[{'doc_id':p['doc_id'],'start':p['canonical_start'],'end':p['canonical_end']} for p in e['context_bundle']['pieces']],
                    'retriever_candidates':e['retriever_candidates'],'rank_scores':[p['score'] for p in e['retriever_candidates']],
                    'packing_drops':e['context_bundle']['truncations'],'raw_response':response,'parsed_response':parsed,
                    'stop_reason':client.last_metadata.get('done_reason'),'output_tokens':client.last_metadata.get('eval_count'),
                    'success':error is None,'error_type':error,'wall_time_ms':(time.perf_counter()-began)*1000,
                    'measured_resource':client.last_metadata.get('resources'),'budget_ticket':rid,'actual_sent_payload':client.sent_payload,
                    'metric_version':'v3-independent-word-overlap; primary semantics pending','proxy_unknowns':['claim_f1_proxy','support_proxy','citation_completeness_proxy'],
                    'document_answerable':answerable,'legacy_token_f1':None if not answerable else metrics['legacy_token_f1'] if metrics else 0.0,
                    'word_metrics':metrics,'purpose':kind},campaign_id=run.name,task_id='S3_scoped_'+kind)
                (store if kind=='formal' else warm).add(rid,row)
                print(canonical_json({'event':'RESOURCE_GENERATION','segment':segment,'formal':len(store.ids),'kind':kind,'failure':bool(error)}),flush=True)
                if error and error['kind']!='ValueError':raise RuntimeError('Recorded backend failure; no automatic retry')
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);p.add_argument('--worker',action='store_true');p.add_argument('--segment',type=int);a=p.parse_args()
    runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    require_ram(2*1024**3)
    if a.worker:return worker(run,runtime,a.segment)
    for segment in range(10):
        with (run/('S3_D_CONFIRM_RESOURCE_SEGMENT_'+str(segment)+'.log')).open('ab') as log:
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--campaign',str(run),'--worker','--segment',str(segment)],
                cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
        if result.returncode:
            atomic_json(run/'S3_D_CONFIRM_RESOURCE_FIX_FAILURE.json',{'status':'FIX_TERMINAL_FAILURE','segment':segment,'actual_worker_exit':result.returncode,'automatic_retry':False});return result.returncode
    path=run/'S3_SCOPED_GENERATION_D_confirm_resource.jsonl';rows=read_records(path)
    if len(rows)!=200:raise ValueError('Paired confirmation incomplete')
    warmups=len(read_records(run/'S3_SCOPED_WARMUPS_D_confirm.jsonl'))+len(read_records(run/'S3_SCOPED_WARMUPS_D_confirm_resource.jsonl'))
    atomic_json(run/'S3_SCOPED_GENERATION_D_confirm_COMPLETION.json',{'status':'PASS_COMPLETE_PAIRED_GENERATION','split':'D_confirm',
        'questions_n':100,'formal_records':200,'warmups':warmups,'records':artifact(path),
        'context_manifest':artifact(run/'S3_SCOPED_CONTEXT_MANIFEST.json'),'budget':base.Budget(run).snapshot(),
        'primary_metrics':'PENDING','human_review_status':'NOT_REVIEWED','resource_fix':True})
    print(canonical_json({'status':'PASS_COMPLETE_PAIRED_GENERATION','records':200,'warmups':warmups}));return 0


if __name__=='__main__':raise SystemExit(main())
