"""Bounded development checks and mandatory final same-question comparisons."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'scripts'))
from ragscope.experiments.runtime import configure
from ragscope.indexing.identity import artifact,atomic_json,signature


def read_jsonl(path):
    with Path(path).open(encoding='utf8') as handle:return [json.loads(line) for line in handle if line.strip()]


def questions(source):
    from ragscope.v3.splits import read_split
    from ragscope.v3.auxiliary_public import read_auxiliary_public
    return {name:read_split(source,name,purpose='inference') for name in ('D_search','D_confirm','H_QA')} | {'H_tasks':read_auxiliary_public(source,'H_tasks')}


def units(source, selected=None):
    output={}
    with (source.parents[1]/'corpora/20261014_v2/all_documents.jsonl').open(encoding='utf8') as handle:
        for line in handle:
            document=json.loads(line)
            if selected is None or document['doc_id'] in selected:
                output.update({u['unit_id']:u for u in document['evidence_units']})
    return output


def prepare(source,run):
    import numpy as np
    from ragscope.v3.local_runtime import ManagedOllama
    from ragscope.v3.finish_runtime import ReusableEncoder
    from ragscope.indexing.local_embedding import QUERY_INSTRUCTION
    groups=questions(source);rows=[q for group in groups.values() for q in group]
    identity=json.loads((source/'S3_PRIVATE_EMBEDDING_SMOKE.json').read_text())['model_identity']
    with ManagedOllama(run,run.parents[1]/'tools/ollama-0.34.4/ollama.exe',num_ctx=4096,startup_ram_bytes=512*1024**2) as service:
        encoder=ReusableEncoder(run,source,service,identity)
        vectors=encoder.embed([QUERY_INSTRUCTION+q['question'] for q in rows])
    np.savez(run/'QUERY_VECTORS.npz',query_ids=np.asarray([q['query_id'] for q in rows]),vectors=vectors)
    atomic_json(run/'PUBLIC_INPUTS.json',groups)
    print('PUBLIC_QUERY_VECTORS_READY',len(rows),flush=True)


def load_vectors(run):
    import numpy as np
    data=np.load(run/'QUERY_VECTORS.npz',allow_pickle=False)
    return dict(zip(data['query_ids'].tolist(),data['vectors']))


def screen(source,run):
    import numpy as np
    from ragscope.v3.full_retrieval import FullCorpusRetriever
    from ragscope.v3.finish_runtime import evidence_bundle
    from ragscope.v3.splits import read_split
    from ragscope.evaluation.evidence import evidence_metrics
    from ragscope.experiments.ledger import RecordStore
    engine=FullCorpusRetriever(source,run);vectors=load_vectors(run);public=questions(source)
    gold={s:{q['query_id']:q for q in read_split(source,s,purpose='development')} for s in ('D_search','D_confirm')}
    evidence=units(source,{p for s in gold.values() for q in s.values() for p in q['scope']['doc_ids']})
    recipes=[('B0',{}),('CORE',{}),('DENSE75',{'alpha':.75}),('SPARSE75',{'alpha':.25}),
             ('META',{'metadata_weight':.25,'document_weight':.15}),
             ('DENSE_META',{'alpha':.75,'metadata_weight':.15,'document_weight':.25})]
    summary=[]
    store=RecordStore(run/'DEV_RETRIEVAL.jsonl',{'recipes':recipes,'implementation':artifact(__file__),
        'retriever':artifact(PROJECT/'ragscope/v3/full_retrieval.py'),'packer':artifact(PROJECT/'ragscope/v3/finish_runtime.py')},max_records=1320)
    for name,recipe in recipes:
        for split in ('D_search','D_confirm'):
            scores=[]
            for q in public[split]:
                rid=f'{name}:{split}:{q["query_id"]}'
                context,result,kind=evidence_bundle(engine,q['question'],q['scope'],vectors[q['query_id']],condition='B0' if name=='B0' else 'FINAL',retrieval=recipe)
                g=gold[split][q['query_id']]
                score=evidence_metrics(g['evidence_reference_sets'],context['unit_spans'],g['answerability'],evidence)
                if rid not in store.ids:store.add(rid,{'query_id':q['query_id'],'recipe':name,'split':split,'evidence':score,'context_hash':context['context_hash']})
                scores.append(score)
            row={'recipe':name,'options':recipe,'split':split,'n':len(scores)}
            for metric in ('full_reference_coverage','continuous_coverage'):
                values=[s[metric] for s in scores if s[metric] is not None]
                row[metric]=float(np.mean(values)) if values else None
            summary.append(row);print(json.dumps(row),flush=True)
    candidates=[r for r in summary if r['split']=='D_search' and r['recipe']!='B0']
    selected=max(candidates,key=lambda r:(r['full_reference_coverage'],r['continuous_coverage']))
    global_summary=[]
    global_recipes=[('GLOBAL_CORE',{}),('GLOBAL_META',{'metadata_weight':.25,'document_weight':.15}),
                    ('GLOBAL_DOC',{'alpha':.75,'metadata_weight':.15,'document_weight':.5}),
                    ('GLOBAL_DOC_STRONG',{'metadata_weight':.25,'document_weight':1.})]
    global_store=RecordStore(run/'GLOBAL_DEV_RETRIEVAL.jsonl',{'recipes':global_recipes,'runner':artifact(__file__),
        'retriever':artifact(PROJECT/'ragscope/v3/full_retrieval.py')},max_records=480)
    for name,recipe in global_recipes:
        scores=[];hits=[]
        for q in public['D_search']:
            context,result,kind=evidence_bundle(engine,q['question'],{'kind':'global'},vectors[q['query_id']],condition='FINAL',retrieval=recipe)
            g=gold['D_search'][q['query_id']]
            score=evidence_metrics(g['evidence_reference_sets'],context['unit_spans'],g['answerability'],evidence)
            hit=bool(set(g['scope']['doc_ids']) & {c['doc_id'] for c in result['candidates'][:10]})
            rid=f'{name}:{q["query_id"]}'
            if rid not in global_store.ids:global_store.add(rid,{'query_id':q['query_id'],'recipe':name,'evidence':score,'document_hit_at10':hit})
            scores.append(score);hits.append(hit)
        row={'recipe':name,'options':recipe,'document_hit_at10':float(np.mean(hits))}
        for metric in ('full_reference_coverage','continuous_coverage'):
            row[metric]=float(np.mean([s[metric] for s in scores if s[metric] is not None]))
        global_summary.append(row);print(json.dumps(row),flush=True)
    global_selected=max(global_summary,key=lambda r:(r['full_reference_coverage'],r['continuous_coverage'],r['document_hit_at10']))
    atomic_json(run/'RETRIEVAL_SELECTION.json',{'selected':selected,'all':summary,'selection_source':'D_search only; D_confirm descriptive check, not admission gate',
        'global_selected':global_selected,'global_all':global_summary,'corpus_papers':len(engine.by_doc),'full_dense_manifest':artifact(run/'FULL_DENSE_INDEX.json')})


def original_client(runtime, model):
    from freeze_generation_v2 import default_spec,model_client_factory,RUNTIME_POLICY
    spec=default_spec(runtime,'20261014_v2')['models'][model]
    spec['client_options']={'backend_runtime_config':RUNTIME_POLICY}
    dev={q['query_id'] for q in read_jsonl(runtime/'corpora/20261014_v2/queries_dev.jsonl')}
    return model_client_factory(spec,dev)[0]


def generate_jobs(source,run,jobs,name):
    import numpy as np
    import psutil
    from ragscope.v3.full_retrieval import FullCorpusRetriever
    from ragscope.v3.finish_runtime import evidence_bundle,answer_messages,resolve_policy
    from ragscope.v3.local_runtime import ManagedOllama
    from ragscope.v3.generation_protocols import ProtocolClient
    from ragscope.v3.budget import Budget
    from ragscope.generation.prompts import visible_output_schema
    from ragscope.generation.pipeline import RAGPipeline
    from ragscope.experiments.ledger import RecordStore
    engine=FullCorpusRetriever(source,run);vectors=load_vectors(run)
    selected=json.loads((run/'RETRIEVAL_SELECTION.json').read_text())['selected']['options']
    qa_policy=json.loads((run/'GENERATION_SELECTION.json').read_text()).get('qa_policy','FINAL') if (run/'GENERATION_SELECTION.json').exists() else 'FINAL'
    config={'jobs':[(j['query_id'],c,m,s) for j,c,m,s in jobs], 'retrieval':selected,
        'runner':artifact(__file__),'runtime':artifact(PROJECT/'ragscope/v3/finish_runtime.py'),
        'retriever':artifact(PROJECT/'ragscope/v3/full_retrieval.py'),
        'full_index':artifact(run/'FULL_DENSE_INDEX.json'),'output_tokens':512,'qa_policy':qa_policy,
        'baseline':'V2 fixed Hybrid/neighbor/G0 policy transferred to same expanded corpus and GPU embedding space; not original CPU deployment timing',
        'test_interpretation':'User-authorized final validation. No claim of operator-blind or human-reviewed evaluation.'}
    store=RecordStore(run/(name+'.jsonl'),config,max_records=len(jobs),real_only=True);budget=Budget(source)
    clients={m:original_client(run.parents[1],m) for m in {job[2] for job in jobs}}
    with ManagedOllama(run,run.parents[1]/'tools/ollama-0.34.4/ollama.exe',num_ctx=4096,startup_ram_bytes=512*1024**2) as service:
        for model in sorted(clients):
            client=ProtocolClient(clients[model],service,protocol='G0')
            for q,condition,mid,split in jobs:
                if mid!=model:continue
                rid=f'finish:{name}:{signature(config)[:12]}:{condition}:{mid}:{q["query_id"]}'
                if rid in store.ids:continue
                if psutil.virtual_memory().available<512*1024**2:raise RuntimeError('Resident generation memory floor; completed records retained')
                service.verify();context=None;parsed=None;error=None;raw=None;issued=False
                client.last_metadata={};client.sent_payload=None
                budget.reserve(request_id=rid,stage='S6_final_validation',max_output_tokens=512,purpose=name,model_identity=client.identity)
                started=time.perf_counter()
                try:
                    effective=resolve_policy(q['question'],q['scope'],condition,qa_policy,q.get('task_kind'))
                    context,retrieval,kind=evidence_bundle(engine,q['question'],q['scope'],vectors[q['query_id']],condition=effective,
                        retrieval={'candidate_depth':50} if effective=='B0' else selected,explicit_task=q.get('task_kind'))
                    messages=answer_messages(q['question'],context,condition=effective,kind=kind)
                    if clients[model].prompt_token_counter(messages)+512>4096:raise ValueError('Actual prompt exceeds 4096 token window')
                    issued=True
                    raw=client.chat(messages,max_tokens=512,temperature=0,output_schema=visible_output_schema(context['pieces']))
                    output=RAGPipeline.parse_output(raw);parsed={k:output[k] for k in ('answer','citations','abstain')}
                    if any(c not in {p['chunk_id'] for p in context['pieces']} for c in parsed['citations']):raise ValueError('Invisible citation')
                    if client.last_metadata.get('token_count_match') is not True:raise ValueError('Actual prompt token count mismatch')
                except Exception as exc:error={'kind':type(exc).__name__,'message':str(exc)}
                elapsed=time.perf_counter()-started
                budget.finalize(rid,output_tokens=client.last_metadata.get('eval_count') if issued else 0,outcome='FAILED' if error else 'BACKEND_RESPONSE')
                row={'query_id':q['query_id'],'question':q['question'],'scope':q['scope'],'split':split,'condition':condition,'model':mid,
                    'task_kind':q.get('task_kind'),'parsed':parsed,'raw_text':raw,'success':error is None,'error':error,
                    'context':context,'elapsed_s':elapsed,'backend':client.last_metadata,'actual_sent_payload':client.sent_payload,'real_model':True,
                    'raw_backend_response':getattr(client,'raw_response',None),'model_identity':client.identity,'budget_ticket':rid}
                store.add(rid,row)
                print(json.dumps({'event':name,'n':len(store.ids),'total':len(jobs),'model':mid,'condition':condition,'elapsed_s':elapsed,'error':error}),flush=True)
            client.unload()


def development(source,run):
    import numpy as np
    from ragscope.v3.splits import read_split
    from ragscope.evaluation.generation_metrics import token_f1
    from ragscope.experiments.ledger import read_records
    public=questions(source)
    subset=sorted(public['D_confirm'],key=lambda q:signature([42,'finish_dev20',q['query_id']]))[:20]
    jobs=[(q,c,m,'D_confirm') for m,c in [('qwen_2b','B0'),('qwen_2b','FINAL'),('qwen_4b','FINAL'),('qwen_9b','FINAL')] for q in subset]
    generate_jobs(source,run,jobs,'DEV_GENERATION')
    gold={q['query_id']:q for q in read_split(source,'D_confirm',purpose='development')}
    groups={}
    for r in read_records(run/'DEV_GENERATION.jsonl'):
        g=gold[r['query_id']];key=r['model']+':'+r['condition']
        p=r['parsed'] or {};f1=None if g['answerability']=='unanswerable' else (token_f1(p.get('answer',''),g['answer_refs']) if r['success'] and not p.get('abstain') else 0.)
        groups.setdefault(key,[]).append({'f1':f1,'elapsed':r['elapsed_s'],'failure':not r['success'],'abstain':p.get('abstain',False)})
    summary={key:{'n':len(rows),'f1':float(np.mean([r['f1'] for r in rows if r['f1'] is not None])),
        'p50_s':float(np.median([r['elapsed'] for r in rows])),'failures':sum(r['failure'] for r in rows)} for key,rows in groups.items()}
    candidates=[key for key in summary if key.endswith(':FINAL') and key.split(':')[0] in {'qwen_2b','qwen_4b'}]
    best=max(candidates,key=lambda key:summary[key]['f1'])
    # Prefer smaller/faster model within one F1 point of the highest dev mean.
    eligible=[key for key in candidates if summary[key]['f1']>=summary[best]['f1']-.01]
    chosen=min(eligible,key=lambda key:summary[key]['p50_s']).split(':')[0]
    atomic_json(run/'GENERATION_SELECTION.json',{'model':chosen,'condition':'FINAL','summary':summary,
        'rule':'Best dev20 mean F1 among2B/4B, then fastest within0.01. Small exposed development check, not test proof.',
        '9B_default_exclusion':'Measured separately; its weight plus the resident encoder and context exceed the practical8GB online budget.'})
    print(json.dumps(summary),flush=True)


def formal(source,run):
    public=questions(source);chosen=json.loads((run/'GENERATION_SELECTION.json').read_text())['model']
    lock={'retrieval':json.loads((run/'RETRIEVAL_SELECTION.json').read_text())['selected'],
        'generation':json.loads((run/'GENERATION_SELECTION.json').read_text()),'created_before_holdout_gold_read':True,
        'source_artifacts':[artifact(PROJECT/path) for path in ('ragscope/v3/finish_runtime.py','ragscope/v3/full_retrieval.py','scripts/v3/finish_experiments.py')],
        'user_override':'Original 10.1 admission gate explicitly removed; H_QA200 and H_tasks20 must execute',
        'split_manifest':artifact(source/'SPLITS_V3.json')}
    if not (run/'FINAL_CONFIGURATION.json').exists():atomic_json(run/'FINAL_CONFIGURATION.json',lock)
    jobs=[]
    for split in ('H_QA','H_tasks'):
        for q in public[split]:
            for condition,model in [('B0','qwen_2b'),('FINAL',chosen)]:jobs.append((q,condition,model,split))
    generate_jobs(source,run,jobs,'FINAL_GENERATION')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--run',required=True)
    parser.add_argument('--mode',choices=['prepare','screen','dev','formal','all'],default='all')
    args=parser.parse_args();configure(cpu_only=True)
    source,run=Path(args.source).resolve(),Path(args.run).resolve()
    for mode,operation in [('prepare',prepare),('screen',screen),('dev',development),('formal',formal)]:
        if args.mode in {mode,'all'}:
            if mode=='prepare' and (run/'QUERY_VECTORS.npz').exists():continue
            if mode=='screen' and (run/'RETRIEVAL_SELECTION.json').exists():continue
            if mode=='dev' and (run/'GENERATION_SELECTION.json').exists():continue
            operation(source,run)


if __name__=='__main__':main()
