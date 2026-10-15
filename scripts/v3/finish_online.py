"""Real uncached hot online comparisons:30 questions per policy plus warmups."""
import argparse
import json
from pathlib import Path
import sys
import time
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'scripts'))
from finish_experiments import original_client
from ragscope.experiments.runtime import configure
from ragscope.indexing.identity import artifact,signature,atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--run',required=True)
    parser.add_argument('--resident8k',action='store_true')
    args=parser.parse_args();configure(cpu_only=True)
    source,run=Path(args.source),Path(args.run)
    suffix='_RESIDENT8K' if args.resident8k else ''
    import numpy as np
    from ragscope.experiments.ledger import RecordStore
    from ragscope.v3.splits import read_split
    from ragscope.v3.full_retrieval import FullCorpusRetriever
    from ragscope.v3.finish_runtime import ReusableEncoder,evidence_bundle,answer_messages,resolve_policy
    from ragscope.v3.online_runtime import OnlineOwnedOllama,UncachedOnlineEmbedding
    from ragscope.v3.generation_protocols import ProtocolClient
    from ragscope.v3.deployment import deployment_context,deployment_kind
    from ragscope.v3.budget import Budget
    from ragscope.generation.prompts import visible_output_schema
    from ragscope.generation.pipeline import RAGPipeline
    selection=json.loads((run/'GENERATION_SELECTION.json').read_text('utf8'))
    qa_policy=selection.get('qa_policy','FINAL')
    retrieval=json.loads((run/'RETRIEVAL_SELECTION.json').read_text('utf8'))['selected']['options']
    questions=sorted(read_split(source,'D_confirm'),key=lambda q:signature([42,'finish_online30',q['query_id']]))[:30]
    engine=FullCorpusRetriever(source,run);budget=Budget(source)
    config={'questions':[q['query_id'] for q in questions],'conditions':['B0','FINAL'],'final_model':selection['model'],
        'query_embedding_cache':False,'answer_cache':False,'runner':artifact(__file__),'final_configuration':artifact(run/('DEPLOYMENT_CONFIGURATION'+suffix+'.json')),
        'timing':'Live query encoding+retrieval+packing+generation+parsing. Excludes budget/journal fsync and pre-request ownership check. Hot models; cold loads retained in warmup rows.'}
    store=RecordStore(run/('FINAL_ONLINE'+suffix+'.jsonl'),config,max_records=60,real_only=True)
    warm=RecordStore(run/('FINAL_ONLINE_WARMUPS'+suffix+'.jsonl'),config,max_records=2,real_only=True)
    with OnlineOwnedOllama(run,run.parents[1]/'tools/ollama-0.34.4/ollama.exe',num_ctx=4096,startup_ram_bytes=512*1024**2) as service:
        encoder=UncachedOnlineEmbedding(run,service)
        if encoder.identity!=engine.embedding_identity:raise ValueError('Online encoder identity differs')
        for condition,model in [('B0','qwen_2b'),('FINAL',selection['model'])]:
            original=original_client(run.parents[1],model);client=ProtocolClient(original,service,protocol='G0')
            for index,q in [(-1,questions[0])]+list(enumerate(questions)):
                rid=f'finish:online:{signature(config)[:12]}:{condition}:{index}'
                destination=warm if index<0 else store
                if rid in destination.ids:continue
                complex_task=condition=='FINAL' and deployment_kind(q['question']) in {'summary','comparison'}
                ctx,limit=(8192,768) if complex_task else (8192 if condition=='FINAL' and args.resident8k else 4096,512)
                client.num_ctx=ctx;client.identity['num_ctx']=ctx
                service.verify();budget.reserve(request_id=rid,stage='S6_final_validation',max_output_tokens=limit,purpose='finish_online_warmup' if index<0 else 'finish_online_formal',model_identity=client.identity)
                started=time.perf_counter();error=None;parsed=None;context=None;issued=False;timings={};client.last_metadata={}
                try:
                    vector,seconds=encoder.query(q['question'],rid);timings['embedding_s']=seconds
                    tick=time.perf_counter()
                    if condition=='FINAL':
                        choices=json.loads((run/'RETRIEVAL_SELECTION.json').read_text('utf8'))
                        context,ranked,kind,ctx,limit,effective=deployment_context(engine,q['question'],q['scope'],vector,original,choices)
                    else:
                        effective='B0';context,ranked,kind=evidence_bundle(engine,q['question'],q['scope'],vector,condition='B0',retrieval={'candidate_depth':50})
                    messages=answer_messages(q['question'],context,condition=effective,kind=kind)
                    timings['retrieval_packing_s']=time.perf_counter()-tick;tick=time.perf_counter();issued=True
                    text=client.chat(messages,max_tokens=limit,temperature=0,output_schema=visible_output_schema(context['pieces']))
                    parsed=RAGPipeline.parse_output(text);timings['generation_s']=time.perf_counter()-tick
                    if client.last_metadata.get('token_count_match') is not True:raise ValueError('Prompt count mismatch')
                except Exception as exc:error={'kind':type(exc).__name__,'message':str(exc)}
                elapsed=time.perf_counter()-started
                budget.finalize(rid,output_tokens=client.last_metadata.get('eval_count') if issued else 0,outcome='FAILED' if error else 'BACKEND_RESPONSE')
                destination.add(rid,{'query_id':q['query_id'],'condition':condition,'model':model,'model_identity':client.identity,'real_model':True,
                    'parsed':parsed,'context':context,'success':error is None,'error':error,'elapsed_s':elapsed,'timings':timings,
                    'backend':client.last_metadata,'actual_sent_payload':client.sent_payload,'raw_backend_response':client.raw_response,
                    'query_embedding_cache':False,'answer_cache':False,'budget_ticket':rid})
                print(json.dumps({'event':'ONLINE','formal_n':len(store.ids),'condition':condition,'elapsed_s':elapsed,'error':error}),flush=True)
            client.unload()
    summary={}
    for condition in ('B0','FINAL'):
        rows=[r for r in store.records if r['condition']==condition]
        summary[condition]={'n':len(rows),'failures':sum(not r['success'] for r in rows),
            'p50_s':float(np.median([r['elapsed_s'] for r in rows])),'p95_s':float(np.percentile([r['elapsed_s'] for r in rows],95))}
    atomic_json(run/('FINAL_ONLINE_SUMMARY'+suffix+'.json'),{'groups':summary,'records':artifact(store.path),'warmups':artifact(warm.path),'config':config})
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
