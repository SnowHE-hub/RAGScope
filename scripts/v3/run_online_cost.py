"""Actual30 questions ×2 policies ×3 repeats, with uncached live embedding and hot serial models."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import psutil
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'scripts'))
from freeze_generation_v2 import model_client_factory
from filelock import FileLock
from ragscope.experiments.ledger import RecordStore
from ragscope.experiments.runtime import configure,require_ram
from ragscope.generation.context import pack_context
from ragscope.generation.pipeline import RAGPipeline
from ragscope.indexing.identity import artifact,atomic_json,canonical_json,signature
from ragscope.v3.budget import Budget
from ragscope.v3.core_packing import pack_core_first
from ragscope.v3.generation_protocols import ProtocolClient,protocol_messages,protocol_schema
from ragscope.v3.hierarchical_runtime import load_hierarchy
from ragscope.v3.online_runtime import OnlineOwnedOllama,UncachedOnlineEmbedding
from ragscope.v3.private_embeddings import TAG
from ragscope.v3.records import canonical_payload
from ragscope.v3.splits import read_split


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--dry-run',action='store_true');p.add_argument('--max-formal',type=int,default=180);a=p.parse_args()
    runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs' or not 1<=a.max_formal<=180:raise ValueError('Invalid campaign/bound')
    questions=sorted(read_split(run,'D_confirm',purpose='inference'),key=lambda q:signature([42,'S6_online30',q['query_id']]))[:30]
    frozen=json.loads((run/'S3_SCOPED_CONTEXT_MANIFEST.json').read_text('utf8'))
    config={'runner':artifact(__file__),'online_runtime':artifact(PROJECT/'ragscope/v3/online_runtime.py'),
        'hierarchy':artifact(PROJECT/'ragscope/v3/hierarchical_runtime.py'),'core_packing':artifact(PROJECT/'ragscope/v3/core_packing.py'),
        'query_ids':[q['query_id'] for q in questions],'conditions':['B0_policy_neighbor_bridge','C1_policy_core_first_bridge'],
        'repeats':3,'seed':42,'reference_budget':2048,'output_budget':512,'answer_cache':False,'embedding_cache':False,
        'num_ctx':4096,'threads':4,'model_residency':'one encoder+one2B generator; serialized requests, resident during hot tests',
        'interpretation':'D_confirm development efficiency screen; primary quality admission currently unproved; not unseen-test evidence',
        'baseline_qualification':frozen['baseline_qualification'],'human_review_status':'NOT_REVIEWED'}
    if a.dry_run:print(canonical_json({'formal_requests':180,'warmup_requests':2,'config':config}));return 0
    require_ram(2*1024**3);index_start=time.perf_counter();engine=load_hierarchy(run);index_load=time.perf_counter()-index_start
    dev={json.loads(l)['query_id'] for l in (runtime/'corpora/20261014_v2/queries_dev.jsonl').read_text('utf8').splitlines()}
    original,_=model_client_factory(frozen['generator_spec'],dev)
    chunks={c['chunk_id']:c for c in engine.chunks};budget=Budget(run)
    with FileLock(str(run/'writer.lock'),timeout=0):
        store=RecordStore(run/'S6_ONLINE_COST.jsonl',config,max_records=180)
        warm=RecordStore(run/'S6_ONLINE_WARMUPS.jsonl',config,max_records=12)
        if (store.ids or warm.ids) and not a.resume:raise ValueError('Existing online measurements require resume')
        tasks=[]
        for repeat in range(3):
            for q in questions:
                conditions=sorted(config['conditions'],key=lambda c:signature([42,'paired_order',repeat,q['query_id'],c]))
                tasks.extend((q,c,repeat) for c in conditions)
        pending=[t for t in tasks if f'S6:online:{t[2]}:{t[0]["query_id"]}:{t[1]}' not in store.ids][:a.max_formal]
        if pending:
            with OnlineOwnedOllama(run,runtime/'tools/ollama-0.34.4/ollama.exe',num_ctx=4096) as service:
                encoder=UncachedOnlineEmbedding(run,service)
                if encoder.identity!=engine.embedding_identity:raise ValueError('Online/index embedding identity mismatch')
                planned=[('warmup',questions[0],c,len(warm.ids)+i) for i,c in enumerate(config['conditions'])]+[('formal',*t) for t in pending]
                for kind,q,condition,repeat in planned:
                    if psutil.virtual_memory().available<1024**3:raise RuntimeError('Resident emergency memory floor; retain completed requests')
                    if kind=='formal':residency=service.residency([TAG,original.identity['model']])
                    else:residency=None
                    rid=f'S6:online:{repeat}:{q["query_id"]}:{condition}' if kind=='formal' else f'S6:online_warmup:{repeat}'
                    client=ProtocolClient(original,service,protocol='G0')
                    budget.reserve(request_id=rid,stage='S6_final_validation',max_output_tokens=512,purpose='online_'+kind,model_identity=client.identity)
                    began=time.perf_counter();raw={};parsed=None;error=None;timings={};bundle=None;ranked=[];issued=False
                    try:
                        vector,encoder_seconds=encoder.query(q['question'],rid);timings['query_encoding_seconds']=encoder_seconds
                        t=time.perf_counter();result=engine.search(q['question'],scope=q['scope'],query_vector=vector,embedding_identity=encoder.identity)
                        ranked=result['candidates'];timings['retrieval_seconds']=time.perf_counter()-t;timings['rerank_seconds']=0.0
                        t=time.perf_counter()
                        if condition.startswith('B0'):bundle=pack_context(ranked[:10],max_tokens=2048,scope=q['scope'],expansion='neighbor',chunk_lookup=chunks)
                        else:bundle,_=pack_core_first(ranked[:10],max_tokens=2048,scope=q['scope'],chunk_lookup=chunks)
                        context=asdict(bundle);messages=protocol_messages(q['question'],context,'G0');schema=protocol_schema(context,'G0')
                        if original.prompt_token_counter(messages)+512>4096:raise ValueError('Actual online packing exceeds shared model budget')
                        timings['packing_seconds']=time.perf_counter()-t;t=time.perf_counter();issued=True
                        text=client.chat(messages,max_tokens=512,temperature=0,output_schema=schema)
                        timings['generation_request_seconds']=time.perf_counter()-t
                        output=RAGPipeline.parse_output(text);parsed={k:output[k] for k in ('answer','citations','abstain')}
                        if client.last_metadata.get('token_count_match') is not True:raise RuntimeError('Actual online prompt counter mismatch')
                        if any(c not in {p['chunk_id'] for p in context['pieces']} for c in parsed['citations']):raise ValueError('Online citation invisible')
                        raw={'content':text,'raw_backend_response':client.raw_response,'backend_metadata':client.last_metadata}
                        timings['prefill_seconds']=client.last_metadata.get('prompt_eval_duration',0)/1e9 if client.last_metadata.get('prompt_eval_duration') is not None else None
                        timings['decode_seconds']=client.last_metadata.get('eval_duration',0)/1e9 if client.last_metadata.get('eval_duration') is not None else None
                        t=time.perf_counter();canonical_json({'answer':parsed,'context':context});timings['serialization_seconds']=time.perf_counter()-t
                    except Exception as exc:error={'kind':type(exc).__name__,'message':str(exc)}
                    elapsed=time.perf_counter()-began
                    budget.finalize(rid,output_tokens=client.last_metadata.get('eval_count') if issued else 0,outcome='FAILED' if error else 'BACKEND_RESPONSE')
                    context=asdict(bundle) if bundle is not None else None
                    row=canonical_payload({'question_id':q['query_id'],'dataset_role':'D_confirm','condition':condition,'repeat':repeat,
                        'exposure_status':'DEVELOPMENT_VALIDATION','user_scope':q['scope'],'task_type_source':'OFFICIAL_QASPER_UNLABELED',
                        'model_identity':client.identity,'prompt_hash':signature(messages) if bundle else None,
                        'actual_prompt_tokens':client.last_metadata.get('prompt_eval_count'),
                        'context_intervals':[{'doc_id':p['doc_id'],'start':p['canonical_start'],'end':p['canonical_end']} for p in context['pieces']] if context else [],
                        'retriever_candidates':ranked,'packing_drops':context['truncations'] if context else [],
                        'raw_response':raw,'parsed_response':parsed,'success':error is None,'error_type':error,
                        'stop_reason':client.last_metadata.get('done_reason'),'output_tokens':client.last_metadata.get('eval_count'),
                        'wall_time_ms':elapsed*1000,'timings':timings,'first_token_seconds':None,
                        'measured_resource':client.last_metadata.get('resources'),'residency_before':residency,
                        'index_load_seconds_separate':index_load,'budget_ticket':rid,'generator_call_issued':issued,
                        'actual_sent_payload':client.sent_payload,'answer_cache':False,'embedding_cache':False,
                        'cost_scope':'actual live encoder+retrieval+packing+generation+serialization; experimental budget/fsync and residency safety check excluded',
                        'proxy_unknowns':['claim_f1_proxy','support_proxy','citation_completeness_proxy']},campaign_id=run.name,task_id='S6_online_'+kind)
                    (store if kind=='formal' else warm).add(rid,row)
                    print(canonical_json({'event':'ONLINE_REQUEST','formal_n':len(store.ids),'kind':kind,'condition':condition,'wall_seconds':elapsed,'error':error}),flush=True)
                    if error and error['kind']!='ValueError':raise RuntimeError('Online backend/count failure retained; stop bounded run')
        result={'status':'PASS_180_ACTUAL_ONLINE_REQUESTS' if len(store.ids)==180 else 'PARTIAL_BOUNDED_ONLINE',
            'formal_n':len(store.ids),'warmup_n':len(warm.ids),'records':artifact(store.path),'budget':budget.snapshot(),
            'quality_admission':'NOT_PROVED','human_review_status':'NOT_REVIEWED'}
        atomic_json(run/'S6_ONLINE_COST_COMPLETION.json',result);print(canonical_json(result));return 0


if __name__=='__main__':raise SystemExit(main())
