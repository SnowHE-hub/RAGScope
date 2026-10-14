"""Download only the permitted pinned Qwen 0.6B reranker, anonymously to D."""
import argparse
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'scripts/v3'))
from download_metric_models import metadata,download_file,SMALL_FILES
from ragscope.experiments.runtime import configure,require_disk,require_ram
from ragscope.indexing.identity import artifact,atomic_json,canonical_json
from filelock import FileLock


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args()
    runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    require_ram(2*1024**3)
    with FileLock(str(run/'reranker_download.lock'),timeout=0):
        repo='Qwen/Qwen3-Reranker-0.6B';api_path=run/'source_package/Qwen__Qwen3-Reranker-0.6B.hf_api.json'
        if api_path.exists():api=json.loads(api_path.read_text('utf8'))
        else:
            api,_=metadata(repo);atomic_json(api_path,api)
        revision=api['sha'];siblings={x['rfilename']:x for x in api['siblings']}
        if 'model.safetensors' not in siblings:raise ValueError('Official single weight file unavailable')
        chosen=sorted(SMALL_FILES & set(siblings))+['model.safetensors']
        size=sum(x.get('size',x.get('lfs',{}).get('size',0)) or 0 for x in (siblings[n] for n in chosen))
        require_disk(size)
        target=runtime/'models/v3_reranker/Qwen__Qwen3-Reranker-0.6B'/revision;target.mkdir(parents=True,exist_ok=True)
        outputs=[]
        for name in chosen:
            print(canonical_json({'event':'RERANKER_DOWNLOAD','file':name,'revision':revision}),flush=True)
            outputs.append(download_file('https://huggingface.co/'+repo+'/resolve/'+revision+'/'+name,target/name,siblings[name]))
        result={'status':'PASS_PINNED_DOWNLOAD','model':repo,'revision':revision,'path':str(target),
            'files':outputs,'source_api':artifact(api_path),'runner':artifact(__file__),'new_llm_calls':0,
            'intended_scoring':'official yes/no last-token logits; never ordinary sequence classification head'}
        atomic_json(run/'QWEN_RERANKER_MODEL.json',result);print(canonical_json({'status':result['status'],'path':str(target),'bytes':sum(x['bytes'] for x in outputs)}));return 0


if __name__=='__main__':raise SystemExit(main())
