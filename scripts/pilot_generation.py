"""Real 20-dev pilots; every request counted, retained and compared to backend tokens."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from dataclasses import asdict
import json
import os
import random
import time
import requests
from ragscope.experiments.runtime import configure,runtime_home,validate_gpu_lease
from ragscope.experiments.ledger import append_jsonl,read_records
from ragscope.experiments.generation_campaign import GenerationBudget,CountedClient
from ragscope.generation.llm_client import LocalLLMClient
from ragscope.generation.pipeline import RAGPipeline
from ragscope.generation.prompts import PROMPT_VERSION
from ragscope.generation.resources import MeasuredClient
from ragscope.generation.token_budget import Qwen35PromptCounter,LFMPromptCounter
from ragscope.indexing.identity import atomic_json,signature

parser=argparse.ArgumentParser(); parser.add_argument('--size',choices=['2b','4b','9b','lfm'],required=True); parser.add_argument('--limit',type=int,default=20); parser.add_argument('--attempt',type=int,choices=[1,2,3],default=1); parser.add_argument('--summarize-only',action='store_true')
args=parser.parse_args(); configure(cpu_only=True); root=runtime_home(); run=root/'runs/20261014_v2'
validate_gpu_lease(run/'gpu_lease.json')
os.environ['CUDA_VISIBLE_DEVICES']='0'
if args.size=='lfm':
    source=json.loads((root/'models/generation/lfm2.5_q4_k_m.json').read_text())
    counter=LFMPromptCounter(root/'models/tokenizers/LFM2.5-1.2B-Instruct',root/'models/ollama/blobs/sha256-58e51bf3ff10a5f7ed85bdbcb131428218f367ffd9fd30915cd73e31d2edb904')
else:
    source=json.loads((root/'models/generation/text_only'/f'qwen3.5_{args.size}-q4_K_M.json').read_text())
    counter=Qwen35PromptCounter(root/'models/tokenizers'/f'Qwen3.5-{args.size.upper()}',root/'downloads/ollama_renderer_0.34.4/qwen35.go')
weight=next(a for a in source['artifacts'] if a['media_type']=='application/vnd.ollama.image.model')
client=LocalLLMClient(source['model'],revision=source['manifest_sha256'],weight_sha256=weight['sha256'],quantization='Q4_K_M',backend_version='0.34.4',chat_template_hash=counter.template_hash,prompt_token_counter=counter,num_gpu=999,num_ctx=4096,think=False,service_device_verified='gpu',timeout=180,backend_runtime_config={'prompt_ram_cache_mib':0,'prompt_cache_policy':'backend_history_disabled'})
client=MeasuredClient(client,run/'generation_service.json')
budget=GenerationBudget(run/'generation_budget.json')
rows=read_records(root/'runs/20261014_v2_retrieval/records/E1_R512_bm25_none.jsonl')
random.Random(42).shuffle(rows); rows=rows[:20]
path=run/'generation_pilots.jsonl'; existing=read_records(path) if path.exists() else []
done={row['record_id'] for row in existing}
for row in ([] if args.summarize_only else rows[:args.limit]):
    prior=[r for r in existing if r['query_id']==row['query_id'] and r.get('prompt_version')==PROMPT_VERSION and r['result']['model_identity']==client.identity]
    if any((not r['result']['error'] or (r['result']['error']['kind']=='parse_error' and r['result']['backend_metadata'].get('done_reason')=='length')) and r.get('token_count_match') for r in prior): continue
    if len(prior)>=3: continue  # bounded per input/config, failed attempts remain in the ledger
    attempt=len(prior)+1
    rid=signature({'pilot_version':1,'prompt_version':PROMPT_VERSION,'model':client.identity,'query':row['query_id'],'context':row['context_bundle']['context_hash'], 'attempt':attempt})
    if rid in done: continue
    counted=CountedClient(client,budget,'pilot',row['query_id'])
    result=RAGPipeline(None,counted,use_cache=False).answer(row['question'],context_bundle=row['context_bundle'],scope=row['scope'],formal_evaluation=True)
    record=dict(record_id=rid,model_size=args.size,prompt_version=PROMPT_VERSION,query_id=row['query_id'],split='dev',purpose='pilot',attempt=attempt,invocation_label=args.attempt,ticket=counted.last_ticket,result=asdict(result),recorded_at=time.time())
    actual=result.backend_metadata.get('prompt_eval_count'); expected=result.backend_metadata.get('actual_input_tokens')
    record['token_count_match']=actual==expected if actual is not None else None
    append_jsonl(path,record)
    print(json.dumps({'model':source['model'],'query':row['query_id'],'error':result.error,'actual_tokens':actual,'counter_tokens':expected,'match':record['token_count_match'],'seconds':result.latency_ms/1000,'output':result.raw_output[:150]}),flush=True)
    if result.error and result.error['kind'] in ('oom','timeout','transport_error'): break
    if record['token_count_match'] is False: break
all_rows=[r for r in read_records(path) if r['model_size']==args.size and r.get('prompt_version')==PROMPT_VERSION and r['result']['model_identity']==client.identity]
latest={r['query_id']:r for r in all_rows}
completed = len(latest)==20 and all(r['token_count_match'] and (not r['result']['error'] or (r['result']['error']['kind']=='parse_error' and r['result']['backend_metadata'].get('done_reason')=='length')) for r in latest.values())
output_failures = sum(bool(r['result']['error']) for r in latest.values())
atomic_json(run/f'pilot_{args.size}_summary.json',{'model_identity':client.identity,'counter_identity':counter.identity,'prompt_version':PROMPT_VERSION,'records':len(all_rows),'errors_all_attempts':sum(bool(r['result']['error']) for r in all_rows),'token_count_matches':sum(r['token_count_match'] is True for r in all_rows),'expected_n':20,'unique_queries':len(latest),'output_failures_latest':output_failures,'status':('PASS_WITH_OUTPUT_FAILURES' if output_failures else 'PASS') if completed else 'INCOMPLETE_OR_FAILED','interpretation':'Compatibility checks actual completed requests and token parity; parse failures remain model-quality failures, never retried for a better answer.'})
