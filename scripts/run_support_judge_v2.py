"""Freeze or run a bounded local support proxy; never starts a model service."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import os

from filelock import FileLock
from freeze_generation_v2 import model_client_factory, verified_corpus
from run_generation_campaign import GuardedClient, verify_owned_service
from ragscope.experiments.generation_campaign import GenerationBudget, load_campaign
from ragscope.experiments.runtime import configure, runtime_home
from ragscope.experiments.support_judge import freeze_sample, load_sample, run_judge
from ragscope.generation.resources import MeasuredClient


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-dir',required=True)
    parser.add_argument('--output-dir')
    parser.add_argument('--judge-model',choices=['qwen_4b','qwen_9b'],default='qwen_4b')
    parser.add_argument('--sample-size',type=int,default=200)
    parser.add_argument('--run',action='store_true',help='Run only an already-frozen sample after the complete E2 matrix exists')
    args=parser.parse_args()
    directory=Path(args.output_dir) if args.output_dir else Path(args.campaign_dir)/'support_judge'
    if not args.run:
        result=freeze_sample(args.campaign_dir,directory,args.judge_model,args.sample_size)
        print(json.dumps({'status':'SAMPLE_FROZEN','n':len(result['sample']),'sample_hash':result['sample_hash']}));return 0
    configure(cpu_only=False); os.environ['CUDA_VISIBLE_DEVICES']='0'
    sample=load_sample(directory);manifest,_=load_campaign(args.campaign_dir)
    if Path(sample['campaign_dir']).resolve()!=Path(args.campaign_dir).resolve():
        raise ValueError('Judge sample belongs to another campaign')
    spec=manifest['provenance']['spec'];rootrun=runtime_home()/'runs'/spec['campaign']
    queries,_,_=verified_corpus(spec['corpus_dir'])
    client,_=model_client_factory(spec['models'][sample['judge_model_id']],{q['query_id'] for q in queries['dev']})
    service=rootrun/'generation_service.json';lease=rootrun/'gpu_lease.json'
    verify_owned_service(service,lease,client.identity)
    budget=GenerationBudget(rootrun/'generation_budget.json')
    with FileLock(str(runtime_home()/'runs/generation-client.lock'),timeout=0):
        measured=MeasuredClient(GuardedClient(client,service,lease),service,gpu_device=0,interval=1.0)
        result=run_judge(directory,measured,budget)
    print(json.dumps(result,ensure_ascii=False));return 0 if result['n_pending']==0 else 2


if __name__=='__main__':
    raise SystemExit(main())
