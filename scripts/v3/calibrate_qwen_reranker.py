"""Run 12 constructed relevance pairs using pinned official yes/no logits."""
import argparse
import gc
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys
import time
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from filelock import FileLock
from ragscope.experiments.ledger import RecordStore
from ragscope.experiments.runtime import configure,require_ram
from ragscope.indexing.identity import artifact,atomic_json,canonical_json
from ragscope.v3.gpu import GpuSession
from ragscope.v3.rerank_budget import RerankBudget

PREFIX='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
INSTRUCTION='Given a web search query, retrieve relevant passages that answer the query'


def fixtures():
    rows=[
        ('What dataset did the experiment use?','The experiment used dataset Delta.','The experiment was presented in 2020.'),
        ('Did the method increase accuracy?','The method did not increase accuracy.','The authors used a batch size of 16.'),
        ('What was the accuracy?','The accuracy was 84 percent.','The paper was published in 2019.'),
        ('Which of Alpha and Beta scored higher?','Alpha scored 10 and Beta scored 12; higher is better.','Alpha and Beta attended a conference.'),
        ('What is the learning rate?','The learning rate was 0.001.','The experiment used 128 samples.'),
        ('What language was evaluated?','Only English was evaluated.','The code was released in May.'),
        ('How many training samples were used?','There were 128 training samples.','The accuracy was 90 percent.'),
        ('What are the limitations?','The method requires labeled data and was not tested on other languages.','The method uses two hidden layers.'),
        ('What is the main contribution?','We introduce a method that propagates sentiment through discourse relations.','The acknowledgments thank the conference organizers.'),
        ('What is the chemical formula of water?','The chemical formula of water is H2O.','Water freezes at zero degrees Celsius.'),
        ('Which planet is known as the Red Planet?','Mars is known as the Red Planet.','Jupiter has a large red spot.'),
        ('What dataset did the final experiment use?',('Background material concerns conference logistics. '*150)+'The final experiment used dataset Sigma.','The final experiment reported an accuracy of 91 percent.'),
    ]
    return [{'fixture_id':'S3:qwen_reranker:'+str(i),'query':q,'relevant':yes,'irrelevant':no,
             'truth_source':'CONSTRUCTED_UNAMBIGUOUS_RELEVANCE_NOT_QASPER_NOT_HUMAN'} for i,(q,yes,no) in enumerate(rows)]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    require_ram(2*1024**3)
    with FileLock(str(run/'writer.lock'),timeout=0):
        manifest_path=run/'QWEN_RERANKER_MODEL.json';manifest=json.loads(manifest_path.read_text('utf8'))
        if any(artifact(x['path'])['sha256']!=x['sha256'] for x in manifest['files']):raise ValueError('Pinned files changed')
        config={'model':artifact(manifest_path),'runner':artifact(__file__),'fixtures':fixtures(),
            'device':'cuda:0','dtype':'bfloat16','attention':'sdpa','max_complete_tokens':8192,
            'official_prefix':PREFIX,'official_suffix':SUFFIX,'instruction':INSTRUCTION,
            'gate_min_correct':11,'torch':importlib.metadata.version('torch'),'transformers':importlib.metadata.version('transformers'),
            'score':'softmax of final-token [no,yes] logits; no autoregressive generation','human_review_status':'NOT_REVIEWED'}
        store=RecordStore(run/'S3_QWEN_RERANKER_CALIBRATION.jsonl',config,max_records=12)
        pending=[x for x in fixtures() if x['fixture_id'] not in store.ids]
        if store.ids and not a.resume:raise ValueError('Existing results require --resume')
        started=time.monotonic();pair_budget=RerankBudget(run)
        if pending:
            with GpuSession(run) as gpu:
                import torch
                from transformers import AutoTokenizer,AutoModelForCausalLM
                torch.set_num_threads(4)
                tokenizer=AutoTokenizer.from_pretrained(manifest['path'],local_files_only=True,padding_side='left')
                yes=tokenizer.convert_tokens_to_ids('yes');no=tokenizer.convert_tokens_to_ids('no')
                if yes==no or tokenizer.encode('yes',add_special_tokens=False)!=[yes] or tokenizer.encode('no',add_special_tokens=False)!=[no]:raise ValueError('yes/no token IDs do not match official scoring')
                model=AutoModelForCausalLM.from_pretrained(manifest['path'],local_files_only=True,
                    dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
                if 'logits_to_keep' not in inspect.signature(model.forward).parameters:raise ValueError('Backend cannot select official final-token logits without full output allocation')
                prefix=tokenizer.encode(PREFIX,add_special_tokens=False);suffix=tokenizer.encode(SUFFIX,add_special_tokens=False)
                try:
                    for f in pending:
                        require_ram(2*1024**3);gpu.check();results={}
                        for kind in ('relevant','irrelevant'):
                            body='<Instruct>: '+INSTRUCTION+'\n<Query>: '+f['query']+'\n<Document>: '+f[kind]
                            middle=tokenizer(body,padding=False,truncation=False,return_attention_mask=False)['input_ids']
                            ids=prefix+middle+suffix
                            if len(ids)>8192:raise ValueError('Constructed complete input exceeds bound; never silently truncate')
                            pair_budget.reserve(f['fixture_id']+':'+kind,1,config['model'])
                            inputs=torch.tensor([ids],dtype=torch.long,device='cuda:0')
                            with torch.inference_mode():
                                output=model(input_ids=inputs,attention_mask=torch.ones_like(inputs),use_cache=False,logits_to_keep=1)
                                logits=output.logits[0,-1,[no,yes]].float()
                                probability=torch.softmax(logits,dim=-1)[1].item()
                            results[kind]={'score':probability,'no_yes_logits':logits.cpu().tolist(),'complete_input_tokens':len(ids),'truncated':False}
                            del inputs,output,logits
                        store.add(f['fixture_id'],{**f,'scores':results,'correct':results['relevant']['score']>results['irrelevant']['score'],
                            'human_review_status':'NOT_REVIEWED','real_model':True,'model_identity':manifest['revision']})
                        print(canonical_json({'event':'RERANKER_CALIBRATION','record_n':len(store.ids),'correct':store.records[-1]['correct']}),flush=True)
                finally:
                    del model,tokenizer;gc.collect();torch.cuda.empty_cache()
        correct=sum(x['correct'] for x in store.records)
        result={'status':'PASS_CONSTRUCTED_RELEVANCE_GATE' if len(store.ids)==12 and correct>=11 else 'FAIL_CONSTRUCTED_RELEVANCE_GATE',
            'record_n':len(store.ids),'correct_n':correct,'expected_n':12,'new_rerank_pairs':24,
            'records':artifact(store.path),'elapsed_seconds':time.monotonic()-started,'new_generation_requests':0,
            'human_review_status':'NOT_REVIEWED','natural_retrieval_quality':'NOT_YET_EVALUATED'}
        atomic_json(run/'S3_QWEN_RERANKER_CALIBRATION.json',result);print(canonical_json(result));return 0 if len(store.ids)==12 else 2


if __name__=='__main__':raise SystemExit(main())
