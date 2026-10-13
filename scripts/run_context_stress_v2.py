"""Freeze or execute only the prespecified smallest-model4K/8K dev diagnosis."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import os
import time
from filelock import FileLock

from freeze_generation_v2 import default_spec, model_client_factory, verified_corpus, read_jsonl, RUNTIME_POLICY
from run_generation_campaign import GuardedClient, verify_owned_service
from ragscope.experiments.context_stress import freeze_stress, load_stress, run_window, WINDOWS
from ragscope.experiments.generation_campaign import GenerationBudget
from ragscope.experiments.ledger import read_records
from ragscope.experiments.runtime import configure, runtime_home, require_ram
from ragscope.generation.llm_client import LocalLLMClient
from ragscope.generation.resources import MeasuredClient
from ragscope.indexing.identity import artifact, atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=['freeze','run'])
    parser.add_argument('--window',type=int,choices=WINDOWS)
    parser.add_argument('--campaign',default='20261014_v2')
    args=parser.parse_args()
    configure(cpu_only=args.mode=='freeze')
    home=runtime_home(); spec=default_spec(home,args.campaign); run=home/'runs'/args.campaign
    directory=home/'runs'/(args.campaign+'_context_stress')
    queries,_,corpus_receipt=verified_corpus(spec['corpus_dir'])
    spec['models']['qwen_2b'].setdefault('client_options',{})['backend_runtime_config']=RUNTIME_POLICY
    base,pilot=model_client_factory(spec['models']['qwen_2b'],{q['query_id'] for q in queries['dev']})
    clients={w:LocalLLMClient(**{**base.identity,'num_ctx':w},prompt_token_counter=base.prompt_token_counter) for w in WINDOWS}
    if args.mode=='freeze':
        documents={d['doc_id']:d for d in read_jsonl(Path(spec['corpus_dir'])/'documents.jsonl')}
        receipt=freeze_stress(directory,queries['dev'],documents,clients,
            [corpus_receipt,artifact(Path(spec['corpus_dir'])/'documents.jsonl'),
             artifact(Path(spec['corpus_dir'])/'queries_dev.jsonl'),pilot['summary'],pilot['records'],
             artifact(Path(spec['models']['qwen_2b']['source_manifest'])),
             artifact(Path(spec['models']['qwen_2b']['tokenizer_dir'])/'manifest.json'),
             artifact(Path(spec['models']['qwen_2b']['renderer_source'])),artifact(Path(__file__))])
        print(json.dumps({'status':'FROZEN','query_n':len(receipt['query_ids']),'manifest_hash':receipt['manifest_hash'],'generation_calls':0}));return 0
    if args.window is None: raise ValueError('Run exactly one --window at a time')
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    lease=run/'gpu_lease.json'; service=run/'generation_service.json'
    verify_owned_service(service,lease,clients[args.window].identity)
    try: require_ram(2*1024**3)
    except Exception as exc:
        manifest,_=load_stress(directory)
        evidence=directory/f'RESOURCE_GUARD_{args.window}_{time.time_ns()}.json'
        atomic_json(evidence,{'status':'BLOCKED','kind':'preload_host_ram_guard','reason':str(exc),'model_loaded':False,'generation_calls':0})
        atomic_json(directory/f'SUMMARY_{args.window}.json',{'status':'BLOCKED','window':args.window,
            'manifest_hash':manifest['manifest_hash'],'n_expected':len(manifest['query_ids']),
            'n_observed':len(read_records(directory/'records'/f'window_{args.window}.jsonl')),
            'reason':'Host RAM pre-load guard; not a model OOM','evidence':artifact(evidence),
            'quality_ranking_eligible':False})
        raise
    with FileLock(str(home/'runs/generation-client.lock'),timeout=0):
        measured=MeasuredClient(GuardedClient(clients[args.window],service,lease),service,gpu_device=0,interval=1.0)
        result=run_window(directory,args.window,measured,GenerationBudget(run/'generation_budget.json'))
    print(json.dumps(result));return 0 if result['status']=='COMPLETE' else 2


if __name__=='__main__': raise SystemExit(main())
