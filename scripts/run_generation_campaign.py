"""Run one frozen model against an already-ready owned local service.

This command starts/downloads no model service; server lifecycle and release remain
with the root operator. Its service owns the global GPU lock.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import os

from filelock import FileLock
from freeze_generation_v2 import model_client_factory, read_json, verified_corpus
from ragscope.experiments.generation_campaign import GenerationBudget, load_campaign, run_model
from ragscope.experiments.runtime import configure, runtime_home
from ragscope.generation.llm_client import GenerationError
from ragscope.generation.resources import MeasuredClient
from ragscope.training.run import verify_lease


def verify_owned_service(service_path, lease_path, identity):
    import psutil
    verify_lease(lease_path)
    service=read_json(service_path)
    if service.get('status')!='READY' or service.get('base_url')!=identity['base_url'] or service.get('backend_version')!=identity['backend_version']:
        raise GenerationError('service_not_ready','Owned service receipt is not ready or differs from frozen backend')
    env=service.get('environment',{})
    if env.get('OLLAMA_MAX_LOADED_MODELS')!='1' or env.get('OLLAMA_NUM_PARALLEL')!='1' or env.get('OLLAMA_NO_CLOUD')!='1':
        raise GenerationError('service_not_ready','Owned service serial/local-only configuration is not verified')
    if env.get('LLAMA_ARG_CACHE_RAM')!='0' or identity.get('backend_runtime_config')!={'prompt_ram_cache_mib':0,'prompt_cache_policy':'backend_history_disabled'}:
        raise GenerationError('service_not_ready','Prompt RAM-cache runtime differs from frozen disabled policy')
    try:
        process=psutil.Process(service['pid'])
        if abs(process.create_time()-service['creation_time'])>.01 or Path(process.exe()).resolve()!=Path(service['binary']).resolve():
            raise GenerationError('service_not_ready','Owned service process identity changed')
    except psutil.NoSuchProcess as exc:
        raise GenerationError('service_not_ready','Owned service process exited') from exc
    return service


class GuardedClient:
    def __init__(self,client,service,lease):
        self.client,self.service,self.lease=client,service,lease
    def __getattr__(self,key): return getattr(self.client,key)
    def chat(self,messages,**kwargs):
        verify_owned_service(self.service,self.lease,self.identity)
        return self.client.chat(messages,**kwargs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-dir',required=True)
    parser.add_argument('--model-id',required=True)
    parser.add_argument('--lease')
    parser.add_argument('--service-receipt')
    parser.add_argument('--budget')
    parser.add_argument('--max-tasks',type=int)
    args=parser.parse_args()
    configure(cpu_only=False)
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    directory=Path(args.campaign_dir)
    manifest,_=load_campaign(directory)
    spec=manifest['provenance']['spec']; rootrun=runtime_home()/'runs'/spec['campaign']
    lease=Path(args.lease) if args.lease else rootrun/'gpu_lease.json'
    service=Path(args.service_receipt) if args.service_receipt else rootrun/'generation_service.json'
    queries,units,_=verified_corpus(spec['corpus_dir'])
    _,squad_units,_=verified_corpus(Path(spec['corpus_dir'])/'squad_e4'); units.update(squad_units)
    client,_=model_client_factory(spec['models'][args.model_id],{row['query_id'] for row in queries['dev']})
    if client.identity!=manifest['models'][args.model_id]:
        raise ValueError('Frozen model identity changed')
    verify_owned_service(service,lease,client.identity)
    budget=GenerationBudget(Path(args.budget) if args.budget else rootrun/'generation_budget.json')
    # All local generation campaign clients serialize across model/run boundaries;
    # the owned service separately holds runs/gpu.lock for its complete lifecycle.
    with FileLock(str(runtime_home()/'runs/generation-client.lock'),timeout=0):
        measured=MeasuredClient(GuardedClient(client,service,lease),service,gpu_device=0,interval=1.0)
        results=run_model(directory,args.model_id,measured,budget,
                          evidence_units=units,max_tasks=args.max_tasks)
    return 0 if all(row['n_pending']==0 for row in results.values()) else 2


if __name__=='__main__':
    raise SystemExit(main())
