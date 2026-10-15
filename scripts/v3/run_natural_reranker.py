"""Actual natural-development reranking on frozen scoped hybrid candidates."""
import argparse
from dataclasses import asdict
import gc
import inspect
import json
from pathlib import Path
import sys
import time
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'scripts/v3'))
from calibrate_bge_reranker import TorchPairScorer
from calibrate_qwen_reranker import PREFIX,SUFFIX,INSTRUCTION
from filelock import FileLock
from ragscope.experiments.ledger import RecordStore,read_records
from ragscope.experiments.runtime import configure,require_ram
from ragscope.generation.context import pack_context
from ragscope.indexing.identity import artifact,atomic_json,canonical_json
from ragscope.retrieval.reranker import Reranker
from ragscope.v3.gpu import GpuSession
from ragscope.v3.rerank_budget import RerankBudget
from ragscope.v3.retrieval_diagnostics import top2_ranking


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True)
    p.add_argument('--model',choices=['bge','qwen'],required=True);p.add_argument('--split',choices=['D_search','D_confirm'],default='D_search')
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    frozen_path=run/'S3_SCOPED_CONTEXT_MANIFEST.json';frozen=json.loads(frozen_path.read_text('utf8'))
    if artifact(frozen['inputs']['path'])!=frozen['inputs']:raise ValueError('Frozen candidate source changed')
    wanted=set(frozen['initial_D_search_query_ids']) if a.split=='D_search' else None
    entries=[]
    with Path(frozen['inputs']['path']).open(encoding='utf8') as f:
        for line in f:
            e=json.loads(line)
            if e['dataset_role']==a.split and e['condition']=='B0_policy_neighbor_bridge' and (wanted is None or e['query_id'] in wanted):entries.append(e)
    if len(entries)!=(40 if a.split=='D_search' else 100):raise ValueError('Natural development input population changed')
    paper_records={r['doc_id']:r for r in read_records(run/'S3_SCOPED_VECTOR_PAPERS.jsonl')}
    lookups={}
    for pid in {e['scope']['doc_ids'][0] for e in entries}:
        source=paper_records[pid]['source']
        if artifact(source['path'])!=source:raise ValueError('Canonical paper source changed')
        lookups[pid]={c['chunk_id']:c for c in json.loads(Path(source['path']).read_text('utf8'))['chunks']}
    gate=json.loads((run/('S3_BGE_RELEVANCE_CALIBRATION.json' if a.model=='bge' else 'S3_QWEN_RERANKER_CALIBRATION.json')).read_text('utf8'))
    if not gate['status'].startswith('PASS_'):raise ValueError('Constructed relevance gate did not pass')
    qwen=json.loads((run/'QWEN_RERANKER_MODEL.json').read_text('utf8'))
    modeldir=Path(qwen['path']) if a.model=='qwen' else runtime/'models/bge-reranker-base/2cfc18c9415c912f9d8155881c133215df768a70'
    files=[artifact(modeldir/name) for name in ('model.safetensors','config.json','tokenizer.json')]
    config={'runner':artifact(__file__),'candidate_manifest':artifact(frozen_path),'model':a.model,'model_files':files,
        'model_revision':modeldir.name,'device':'cuda:0','dtype':'bfloat16' if a.model=='qwen' else 'float32',
        'split':a.split,'query_ids':[e['query_id'] for e in entries],'candidate_k':50,'rank_k':10,
        'reference_budget':2048,'packing':'neighbor','batch_size':4,'score_text':'canonical body only, same candidate pool',
        'bge_pair_max':512,'bge_overlap':32,'qwen_complete_input_max':8192,
        'prefix':PREFIX if a.model=='qwen' else None,'suffix':SUFFIX if a.model=='qwen' else None,
        'instruction':INSTRUCTION if a.model=='qwen' else None,
        'host_preload_minimum_with_reserve_bytes':3*1024**3,'resident_emergency_minimum_bytes':1024**3,
        'gold_read':False,'human_review_status':'NOT_REVIEWED'}
    if a.dry_run:print(canonical_json({'config':config,'question_n':len(entries),'candidate_pairs':sum(len(e['retriever_candidates'][:50]) for e in entries),'new_generation_requests':0}));return 0
    require_ram(2*1024**3)
    with FileLock(str(run/'writer.lock'),timeout=0):
        journal=run/('S3_NATURAL_RERANK_'+a.model+'_'+a.split+'.jsonl')
        store=RecordStore(journal,config,max_records=len(entries))
        if store.ids and not a.resume:raise ValueError('Existing scores require resume')
        pending=[e for e in entries if 'S3:natural:'+a.model+':'+a.split+':'+e['query_id'] not in store.ids]
        if pending:
            with GpuSession(run) as gpu:
                import torch
                import psutil
                from transformers import AutoTokenizer,AutoModelForCausalLM,AutoModelForSequenceClassification
                torch.set_num_threads(4)
                tokenizer=AutoTokenizer.from_pretrained(modeldir,local_files_only=True,padding_side='left' if a.model=='qwen' else 'right')
                loader=AutoModelForCausalLM if a.model=='qwen' else AutoModelForSequenceClassification
                model=loader.from_pretrained(modeldir,local_files_only=True,dtype=torch.bfloat16 if a.model=='qwen' else torch.float32,
                    device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
                if a.model=='bge':reranker=Reranker('BAAI/bge-reranker-base',revision=modeldir.name,device='cuda:0',model=TorchPairScorer(model,tokenizer),tokenizer=tokenizer)
                else:
                    if 'logits_to_keep' not in inspect.signature(model.forward).parameters:raise ValueError('Final-token scoring unavailable')
                    yes,no=tokenizer.convert_tokens_to_ids('yes'),tokenizer.convert_tokens_to_ids('no')
                    if tokenizer.encode('yes',add_special_tokens=False)!=[yes] or tokenizer.encode('no',add_special_tokens=False)!=[no]:raise ValueError('Official token IDs invalid')
                    prefix=tokenizer.encode(PREFIX,add_special_tokens=False);suffix=tokenizer.encode(SUFFIX,add_special_tokens=False)
                budget=RerankBudget(run)
                try:
                    for e in pending:
                        gpu.check()
                        if psutil.virtual_memory().available<1024**3:raise RuntimeError('Resident emergency RAM guard; retain completed scores')
                        candidates=e['retriever_candidates'][:50];rid='S3:natural:'+a.model+':'+a.split+':'+e['query_id']
                        if len({c['chunk_id'] for c in candidates})!=len(candidates):raise ValueError('Duplicate candidate IDs')
                        started=time.perf_counter();windows=[]
                        if a.model=='bge':
                            n=sum(len(reranker._windows(e['question'],c['text'])) for c in candidates)
                            budget.reserve(rid,n,config['model_files']);ranked=reranker.rerank(e['question'],candidates,top_k=10)
                            windows=reranker.last_window_records;alternate=top2_ranking(candidates,windows,k=10)
                            conditions={'bge_max':ranked,'bge_top2':alternate}
                        else:
                            encoded=[]
                            for c in candidates:
                                body='<Instruct>: '+INSTRUCTION+'\n<Query>: '+e['question']+'\n<Document>: '+c['text']
                                ids=prefix+tokenizer.encode(body,add_special_tokens=False)+suffix
                                if len(ids)>8192:raise ValueError('Full natural input exceeds limit; no truncation')
                                encoded.append(ids)
                            n=len(encoded);budget.reserve(rid,n,config['model_files']);scored=[]
                            for offset in range(0,n,4):
                                batch=encoded[offset:offset+4];width=max(map(len,batch));pad=tokenizer.pad_token_id
                                inputs=torch.tensor([[pad]*(width-len(ids))+ids for ids in batch],device='cuda:0')
                                attention=torch.tensor([[0]*(width-len(ids))+[1]*len(ids) for ids in batch],device='cuda:0')
                                with torch.inference_mode():
                                    logits=model(input_ids=inputs,attention_mask=attention,use_cache=False,logits_to_keep=1).logits[:,-1,[no,yes]].float()
                                    scores=torch.softmax(logits,dim=-1)[:,1].cpu().tolist()
                                for i,score in enumerate(scores):
                                    c=candidates[offset+i];scored.append({**c,'rerank_score':score})
                                    windows.append({'chunk_id':c['chunk_id'],'complete_input_tokens':len(batch[i]),'score':score,'truncated':False})
                                del inputs,attention,logits
                            conditions={'qwen_yes_no':sorted(scored,key=lambda c:(-c['rerank_score'],c['chunk_id']))[:10]}
                        torch.cuda.synchronize();wall=time.perf_counter()-started
                        lookup=lookups[e['scope']['doc_ids'][0]]
                        bundles={name:asdict(pack_context(ranking,max_tokens=2048,scope=e['scope'],expansion='neighbor',chunk_lookup=lookup)) for name,ranking in conditions.items()}
                        store.add(rid,{'query_id':e['query_id'],'question':e['question'],'scope':e['scope'],'dataset_role':a.split,
                            'model_identity':config['model_files'],'real_model':True,'candidate_n':len(candidates),'actual_pair_n':n,
                            'candidate_ids':[c['chunk_id'] for c in candidates],'rankings':conditions,'contexts':bundles,'windows':windows,
                            'rerank_wall_seconds':wall,'resource_after':{'host_available_bytes':psutil.virtual_memory().available,
                                'gpu_allocated_bytes':torch.cuda.memory_allocated()},'human_review_status':'NOT_REVIEWED','gold_read':False})
                        print(canonical_json({'event':'NATURAL_RERANK','model':a.model,'split':a.split,'records':len(store.ids),'candidate_n':len(candidates),'pairs':n}),flush=True)
                        gc.collect();torch.cuda.empty_cache()
                finally:
                    if a.model=='bge':del reranker
                    del model,tokenizer;gc.collect();torch.cuda.empty_cache()
        result={'status':'PASS_NATURAL_RERANK_EXECUTION' if len(store.ids)==len(entries) else 'INCOMPLETE',
            'model':a.model,'split':a.split,'questions_n':len(store.ids),'records':artifact(journal),'new_generation_requests':0,
            'actual_rerank_pairs':sum(r['actual_pair_n'] for r in store.records),'quality_gate':'PENDING_OFFLINE_EVIDENCE_AUDIT','human_review_status':'NOT_REVIEWED'}
        atomic_json(run/('S3_NATURAL_RERANK_'+a.model+'_'+a.split+'_COMPLETION.json'),result)
        print(canonical_json(result));return 0


if __name__=='__main__':raise SystemExit(main())
