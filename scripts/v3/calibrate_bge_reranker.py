"""Real old-BGE direction, window, mapping and max/top2 relevance fixtures."""
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import sys
import numpy as np
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'scripts/v3'))
from calibrate_qwen_reranker import fixtures
from filelock import FileLock
from ragscope.experiments.ledger import RecordStore
from ragscope.experiments.runtime import configure,require_ram
from ragscope.indexing.identity import artifact,atomic_json,canonical_json
from ragscope.retrieval.reranker import Reranker
from ragscope.v3.gpu import GpuSession
from ragscope.v3.rerank_budget import RerankBudget
from ragscope.v3.retrieval_diagnostics import top2_ranking


class TorchPairScorer:
    def __init__(self,model,tokenizer):self.model,self.tokenizer,self.max_length=model,tokenizer,512
    def predict(self,pairs,batch_size=4,show_progress_bar=False):
        import torch
        result=[]
        for offset in range(0,len(pairs),batch_size):
            batch=pairs[offset:offset+batch_size]
            inputs=self.tokenizer([p[0] for p in batch],[p[1] for p in batch],padding=True,truncation=False,return_tensors='pt')
            if inputs['input_ids'].shape[1]>512:raise ValueError('BGE full pair exceeds window budget')
            with torch.inference_mode():
                logits=self.model(**{k:v.to('cuda:0') for k,v in inputs.items()}).logits.float().reshape(-1)
                result.extend(torch.sigmoid(logits).cpu().tolist())
        return np.asarray(result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    require_ram(2*1024**3)
    model_path=runtime/'models/bge-reranker-base/2cfc18c9415c912f9d8155881c133215df768a70'
    files=[artifact(model_path/name) for name in ('model.safetensors','config.json','tokenizer.json')]
    config={'model_files':files,'revision':model_path.name,'runner':artifact(__file__),'fixtures':fixtures(),
        'device':'cuda:0','dtype':'float32','batch_size':4,'max_pair_length':512,'window_overlap':32,'aggregation':['max','top2_mean'],
        'torch':importlib.metadata.version('torch'),'transformers':importlib.metadata.version('transformers'),
        'truth_source':'CONSTRUCTED_RELEVANCE_NOT_HUMAN','gate_min_correct':11}
    with FileLock(str(run/'writer.lock'),timeout=0):
        store=RecordStore(run/'S3_BGE_RELEVANCE_CALIBRATION.jsonl',config,max_records=12)
        if store.ids and not a.resume:raise ValueError('Existing results need --resume')
        pending=[f for f in fixtures() if 'S3:bge_fixture:'+f['fixture_id'] not in store.ids]
        if pending:
            with GpuSession(run) as gpu:
                import torch
                from transformers import AutoTokenizer,AutoModelForSequenceClassification
                torch.set_num_threads(4)
                tokenizer=AutoTokenizer.from_pretrained(str(model_path),local_files_only=True)
                model=AutoModelForSequenceClassification.from_pretrained(str(model_path),local_files_only=True,
                    dtype=torch.float32,device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
                reranker=Reranker('BAAI/bge-reranker-base',revision=model_path.name,device='cuda:0',model=TorchPairScorer(model,tokenizer),tokenizer=tokenizer)
                budget=RerankBudget(run)
                try:
                    for f in pending:
                        require_ram(2*1024**3);gpu.check()
                        candidates=[{'chunk_id':'relevant','text':f['relevant']},{'chunk_id':'irrelevant','text':f['irrelevant']}]
                        pair_n=sum(len(reranker._windows(f['query'],c['text'])) for c in candidates)
                        rid='S3:bge_fixture:'+f['fixture_id'];budget.reserve(rid,pair_n,{'files':files,'dtype':'float32','device':'cuda:0'})
                        ranked=reranker.rerank(f['query'],candidates,top_k=2)
                        alternative=top2_ranking(candidates,reranker.last_window_records,k=2)
                        if any(w['pair_tokens']>512 for w in reranker.last_window_records):raise ValueError('Window budget violation')
                        store.add(rid,{**f,'max_ranking':ranked,'top2_ranking':alternative,'windows':reranker.last_window_records,
                            'correct_max':ranked[0]['chunk_id']=='relevant','correct_top2':alternative[0]['chunk_id']=='relevant',
                            'actual_model_pairs':pair_n,'human_review_status':'NOT_REVIEWED'})
                        print(canonical_json({'event':'BGE_CALIBRATION','records':len(store.ids),'correct_max':store.records[-1]['correct_max']}),flush=True)
                finally:
                    del reranker,model,tokenizer;gc.collect();torch.cuda.empty_cache()
        correct=sum(r['correct_max'] for r in store.records);top2=sum(r['correct_top2'] for r in store.records)
        result={'status':'PASS_CONSTRUCTED_BGE_GATE' if len(store.ids)==12 and correct>=11 else 'FAIL_CONSTRUCTED_BGE_GATE',
            'record_n':len(store.ids),'correct_max_n':correct,'correct_top2_n':top2,'new_rerank_pairs':sum(r['actual_model_pairs'] for r in store.records),
            'records':artifact(store.path),'human_review_status':'NOT_REVIEWED','natural_retrieval_quality':'NOT_ESTABLISHED_BY_FIXTURES',
            'new_generation_requests':0,'device_comparability':'max/top2 use identical GPU backend/FP32 weights/window scores'}
        atomic_json(run/'S3_BGE_RELEVANCE_CALIBRATION.json',result);print(canonical_json(result));return 0 if len(store.ids)==12 else 2


if __name__=='__main__':raise SystemExit(main())
