"""Final220 answers: reuse exact old requests; generate only changed task requests."""
import argparse
import json
from pathlib import Path
import sys
import time
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'scripts'))
from finish_experiments import original_client,questions,load_vectors
from ragscope.experiments.runtime import configure
from ragscope.indexing.identity import atomic_json,artifact,signature


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--run',required=True)
    parser.add_argument('--resident8k',action='store_true')
    args=parser.parse_args();configure(cpu_only=True);source,run=Path(args.source),Path(args.run)
    from ragscope.experiments.ledger import read_records,RecordStore
    from ragscope.v3.full_retrieval import FullCorpusRetriever
    from ragscope.v3.deployment import deployment_context,DeploymentClient
    from ragscope.v3.finish_runtime import answer_messages
    from ragscope.v3.local_runtime import ManagedOllama
    from ragscope.v3.budget import Budget
    from ragscope.generation.prompts import visible_output_schema
    from ragscope.generation.pipeline import RAGPipeline
    old=read_records(run/'FINAL_GENERATION.jsonl')
    if len(old)!=440:raise ValueError('Finish initial440 rows before deploying task-context extension')
    original=original_client(run.parents[1],'qwen_2b');engine=FullCorpusRetriever(source,run);vectors=load_vectors(run)
    public=questions(source);selection=json.loads((run/'RETRIEVAL_SELECTION.json').read_text('utf8'))
    suffix='_RESIDENT8K' if args.resident8k else ''
    prior=old+(read_records(run/'DEPLOYMENT_FINAL_ANSWERS.jsonl') if args.resident8k else [])
    old_by_query={}
    for row in prior:old_by_query.setdefault(row['query_id'],[]).append(row)
    config={'implementation':artifact(__file__),'deployment':artifact(PROJECT/'ragscope/v3/deployment.py'),
        'ordinary_protocol':artifact(PROJECT/'ragscope/v3/finish_runtime.py'),'index':artifact(run/'FULL_DENSE_INDEX.json'),
        'previous440':artifact(run/'FINAL_GENERATION.jsonl'),'prior_deployment':artifact(run/'DEPLOYMENT_FINAL_ANSWERS.jsonl') if args.resident8k else None,
        'ordinary_ctx':8192,'complex_ctx':8192,'complex_output':768,
        'selection_basis':'Dev QA regressions retained originalG0; whole-paper source coverage motivated8K task contexts before H_tasks gold read',
        'reuse':'Exact request payload equality with a successful pinned original2B response; reused requests are not new calls'}
    atomic_json(run/('DEPLOYMENT_CONFIGURATION'+suffix+'.json'),{'protocol':config,
        'source_artifacts':[artifact(PROJECT/p) for p in ('ragscope/v3/deployment.py','ragscope/v3/full_workbench.py')],
        'frozen_before_task_gold_read':True,'initial_configuration':artifact(run/'FINAL_CONFIGURATION.json')})
    store=RecordStore(run/('DEPLOYMENT_FINAL_ANSWERS'+suffix+'.jsonl'),config,max_records=220,real_only=True);budget=Budget(source)
    with ManagedOllama(run,run.parents[1]/'tools/ollama-0.34.4/ollama.exe',num_ctx=8192,startup_ram_bytes=512*1024**2) as service:
        for split in ('H_QA','H_tasks'):
            for q in public[split]:
                rid='deployment'+suffix+':'+q['query_id']
                if rid in store.ids:continue
                context,retrieval,kind,ctx,limit,effective=deployment_context(engine,q['question'],q['scope'],vectors[q['query_id']],original,selection,q.get('task_kind'))
                messages=answer_messages(q['question'],context,condition=effective,kind=kind);schema=visible_output_schema(context['pieces'])
                payload={'model':original.model,'messages':messages,'stream':False,'think':False,'keep_alive':'5m',
                    'options':{'num_predict':limit,'temperature':0,'seed':42,'num_ctx':ctx,'num_gpu':999},'format':schema}
                matching=[r for r in old_by_query[q['query_id']] if r['success'] and r['actual_sent_payload']==payload]
                if matching:
                    row=dict(matching[-1])
                    for key in ('record_id','record_hash','recorded_at','config_hash'):row.pop(key,None)
                    row.update(condition='FINAL',reuse_source_record_id=matching[-1]['record_id'],reuse_source_record_hash=matching[-1]['record_hash'],
                        new_model_call=False,deployment_task_kind=kind,source_character_coverage=retrieval.get('source_character_coverage',{}))
                    store.add(rid,row);print(json.dumps({'event':'DEPLOYMENT_REUSED','n':len(store.ids),'split':split}),flush=True);continue
                client=DeploymentClient(original,service,ctx);service.verify()
                budget.reserve(request_id=rid,stage='S6_final_validation',max_output_tokens=limit,purpose='deployment_changed_task_request',model_identity=client.identity)
                start=time.perf_counter();error=None;parsed=None;raw=None
                try:
                    raw=client.chat(messages,max_tokens=limit,temperature=0,output_schema=schema)
                    out=RAGPipeline.parse_output(raw);parsed={k:out[k] for k in ('answer','citations','abstain')}
                    if client.sent_payload!=payload:raise ValueError('Deployed payload differs from reuse comparison')
                    if client.last_metadata.get('token_count_match') is not True:raise ValueError('Extended prompt count mismatch')
                    if any(c not in {p['chunk_id'] for p in context['pieces']} for c in parsed['citations']):raise ValueError('Invisible citation')
                except Exception as exc:error={'kind':type(exc).__name__,'message':str(exc)}
                elapsed=time.perf_counter()-start;budget.finalize(rid,output_tokens=client.last_metadata.get('eval_count'),outcome='FAILED' if error else 'BACKEND_RESPONSE')
                store.add(rid,{'query_id':q['query_id'],'question':q['question'],'scope':q['scope'],'split':split,'condition':'FINAL','model':'qwen_2b',
                    'parsed':parsed,'raw_text':raw,'success':error is None,'error':error,'context':context,'elapsed_s':elapsed,'backend':client.last_metadata,
                    'actual_sent_payload':client.sent_payload,'raw_backend_response':client.raw_response,'model_identity':client.identity,'real_model':True,
                    'new_model_call':True,'budget_ticket':rid,'deployment_task_kind':kind,'source_character_coverage':retrieval.get('source_character_coverage',{})})
                print(json.dumps({'event':'DEPLOYMENT_GENERATED','n':len(store.ids),'split':split,'ctx':ctx,'error':error}),flush=True)
        client=None
    atomic_json(run/('DEPLOYMENT_ANSWERS_COMPLETION'+suffix+'.json'),{'status':'COMPLETE','n':len(store.ids),'new_calls':sum(r['new_model_call'] for r in store.records),
        'reused_exact_requests':sum(not r['new_model_call'] for r in store.records),'failures':sum(not r['success'] for r in store.records),'records':artifact(store.path)})


if __name__=='__main__':main()
