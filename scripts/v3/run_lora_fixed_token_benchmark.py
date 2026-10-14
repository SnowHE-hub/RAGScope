"""Separate fixed64-token benchmark with observed first-token timing; not answer quality."""
import argparse
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import sys
import time
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'scripts/v3'))
from run_lora_runtime_diagnostics import selected_inputs
from filelock import FileLock
from ragscope.experiments.ledger import RecordStore
from ragscope.experiments.runtime import configure,require_ram
from ragscope.indexing.identity import artifact,atomic_json,canonical_json,verify_artifacts
from ragscope.training.run import load_training_model
from ragscope.v3.budget import Budget
from ragscope.v3.generation_protocols import protocol_messages
from ragscope.v3.gpu import GpuSession


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True)
    p.add_argument('--condition',choices=['base','unmerged','merged'],required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    require_ram(2*1024**3)
    training=runtime/'runs/20261014_v2_lora/train_2b_1024_seed42'
    cfg=json.loads((training/'steps.config.json').read_text('utf8'))['config']
    checkpoint=training/'checkpoints'/json.loads((training/'checkpoints/best.json').read_text('utf8'))['checkpoint']
    inputs=selected_inputs(run)[:3]
    config={'runner':artifact(__file__),'contexts':artifact(run/'S3_SCOPED_CONTEXTS.jsonl'),
        'condition':a.condition,'query_ids':[r['query_id'] for r in inputs],'min_new_tokens':64,'max_new_tokens':64,
        'adapter_manifest':artifact(checkpoint/'manifest.json'),'timing':'synchronous streamer first generated token; prompt callback excluded',
        'quality_metrics':'NOT_APPLICABLE_FORCED_LENGTH_BENCHMARK','human_review_status':'NOT_REVIEWED'}
    with FileLock(str(run/'writer.lock'),timeout=0):
        store=RecordStore(run/('S5_FIXED64_'+a.condition+'.jsonl'),config,max_records=4)
        if store.ids and not a.resume:raise ValueError('Existing benchmark requires resume')
        tasks=[('warmup',inputs[0])]+[('formal',e) for e in inputs]
        pending=[(kind,e) for kind,e in tasks if 'S5:fixed64:'+a.condition+':'+(e['query_id'] if kind=='formal' else 'warmup') not in store.ids]
        if not pending:return 0
        with GpuSession(run):
            import torch
            import transformers
            from transformers.generation.streamers import BaseStreamer
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            torch.set_num_threads(4)
            start=time.perf_counter()
            if a.condition=='merged':
                manifest=json.loads((run/'S5_MERGED_MODEL_MANIFEST.json').read_text('utf8'))
                if not verify_artifacts(manifest['artifacts']):raise ValueError('Merged copy checksum failed')
                model=transformers.Qwen3_5ForConditionalGeneration.from_pretrained(run/'S5_MERGED_MODEL',local_files_only=True,
                    dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='eager')
                tokenizer=transformers.AutoTokenizer.from_pretrained(run/'S5_MERGED_MODEL',local_files_only=True)
            else:
                model,tokenizer,_=load_training_model(runtime/'models/Qwen3.5-2B'/cfg['paths'][0]['revision'],cfg['paths'][0],cfg)
                set_peft_model_state_dict(model,load_file(str(checkpoint/'adapter/adapter_model.safetensors')))
            model.gradient_checkpointing_disable();model.eval();load_s=time.perf_counter()-start
            class TokenClock(BaseStreamer):
                def __init__(self):self.prompt_seen=False;self.first=None;self.generated_n=0
                def put(self,value):
                    if not self.prompt_seen:self.prompt_seen=True;return
                    torch.cuda.synchronize()
                    if self.first is None:self.first=time.perf_counter()
                    self.generated_n+=value.numel()
                def end(self):pass
            budget=Budget(run)
            try:
                for kind,e in pending:
                    rid='S5:fixed64:'+a.condition+':'+(e['query_id'] if kind=='formal' else 'warmup')
                    identity={'model':'Qwen3.5-2B','condition':a.condition,'dtype':'bfloat16','backend':'transformers-eager'}
                    ids=tokenizer.apply_chat_template(protocol_messages(e['question'],e['context_bundle'],'G0'),tokenize=True,
                        add_generation_prompt=True,enable_thinking=False,return_dict=False)
                    budget.reserve(request_id=rid,stage='S5_optional_lora',max_output_tokens=64,purpose='fixed_token_'+kind,model_identity=identity)
                    clock=TokenClock();began=time.perf_counter();new=[];error=None
                    try:
                        tensor=torch.tensor([ids],device='cuda:0');ctx=model.disable_adapter() if a.condition=='base' else nullcontext()
                        with ctx,torch.inference_mode():
                            out=model.generate(input_ids=tensor,attention_mask=torch.ones_like(tensor),do_sample=False,
                                min_new_tokens=64,max_new_tokens=64,use_cache=True,streamer=clock)
                        torch.cuda.synchronize();ended=time.perf_counter();new=out[0,len(ids):].cpu().tolist();del out,tensor
                        if len(new)!=64 or clock.generated_n!=64 or clock.first is None:raise ValueError('Fixed-token/streamer count mismatch')
                    except Exception as exc:ended=time.perf_counter();error={'kind':type(exc).__name__,'message':str(exc)}
                    budget.finalize(rid,output_tokens=len(new) if new else None,outcome='failed' if error else 'success')
                    record={'purpose':kind,'condition':a.condition,'query_id':e['query_id'],'actual_prompt_tokens':len(ids),
                        'output_ids':new,'actual_output_tokens':len(new),'total_generate_seconds':ended-began,
                        'first_token_seconds':clock.first-began if clock.first else None,
                        'decode_after_first_token_seconds':ended-clock.first if clock.first else None,
                        'tokens_per_second_after_first':63/(ended-clock.first) if clock.first else None,
                        'load_seconds_separate':load_s,'model_identity':identity,'budget_ticket':rid,'error':error,
                        'cost_scope':'fixed64 synthetic microbenchmark; excludes online retrieval; callback synchronization overhead included',
                        'human_review_status':'NOT_REVIEWED','answer_quality':'NOT_APPLICABLE'}
                    store.add(rid,record);print(canonical_json({k:record[k] for k in ('condition','purpose','first_token_seconds','tokens_per_second_after_first','error')}),flush=True)
                    gc.collect();torch.cuda.empty_cache()
                    if error:raise RuntimeError('Benchmark failure retained; stop bounded batch')
            finally:
                del model,tokenizer;gc.collect();torch.cuda.empty_cache()
        atomic_json(run/('S5_FIXED64_'+a.condition+'_COMPLETION.json'),{'record_n':len(store.ids),'formal_n':3,'warmup_n':1,
            'status':'PASS_FIXED_TOKEN_EXECUTION','records':artifact(store.path)});return 0


if __name__=='__main__':raise SystemExit(main())
