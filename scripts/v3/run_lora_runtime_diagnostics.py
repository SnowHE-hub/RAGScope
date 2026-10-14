"""Same-BF16 fixed24 development runtime and adapter merge diagnostics."""
import argparse
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import sys
import time
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from filelock import FileLock
from ragscope.experiments.ledger import RecordStore
from ragscope.experiments.runtime import configure,require_ram
from ragscope.generation.pipeline import RAGPipeline
from ragscope.indexing.identity import artifact,atomic_json,canonical_json,signature
from ragscope.training.checkpoint import CheckpointStore
from ragscope.training.run import load_training_model
from ragscope.v3.budget import Budget
from ragscope.v3.generation_protocols import protocol_messages
from ragscope.v3.gpu import GpuSession
from ragscope.v3.records import canonical_payload


def selected_inputs(run):
    rows=[]
    with (run/'S3_SCOPED_CONTEXTS.jsonl').open(encoding='utf8') as f:
        for line in f:
            row=json.loads(line)
            if row['dataset_role']=='D_confirm' and row['condition']=='B0_policy_neighbor_bridge':rows.append(row)
    rows=sorted(rows,key=lambda x:x['query_id'])[:24]
    if len(rows)!=24:raise ValueError('Require fixed24 development contexts')
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True)
    p.add_argument('--condition',required=True,choices=['base','unmerged','merged'])
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    training=runtime/'runs/20261014_v2_lora/train_2b_1024_seed42'
    immutable=json.loads((training/'steps.config.json').read_text('utf8'));cfg=immutable['config']
    checkpoint_store=CheckpointStore(training/'checkpoints',immutable)
    checkpoint=checkpoint_store.root/json.loads((checkpoint_store.root/'best.json').read_text('utf8'))['checkpoint']
    checkpoint_store.verify(checkpoint)
    modeldir=runtime/'models/Qwen3.5-2B'/cfg['paths'][0]['revision']
    inputs=selected_inputs(run)
    config={'runner':artifact(__file__),'context_source':artifact(run/'S3_SCOPED_CONTEXTS.jsonl'),
        'query_ids':[e['query_id'] for e in inputs],'condition':a.condition,
        'model_manifest':artifact(modeldir/'model_manifest.json'),'adapter_manifest':artifact(checkpoint/'manifest.json'),
        'max_new_tokens':512,'do_sample':False,'seed':42,'dtype':'bfloat16','backend':'transformers-local-eager',
        'json_constraint':'prompt_only','eos_policy':'actual_unchanged_generation_config',
        'cache':True,'gradient_checkpointing_inference':False,
        'merge_gate':{'max_abs_logits':0.08,'relative_l2':0.01,'probes':3},
        'task_role':'development_runtime_diagnostic_not_final_quality','human_review_status':'NOT_REVIEWED'}
    if a.dry_run:
        print(canonical_json({'config':config,'formal_calls':24,'warmups':1,'gold_read':False}));return 0
    require_ram(2*1024**3)
    with FileLock(str(run/'writer.lock'),timeout=0):
        store=RecordStore(run/('S5_RUNTIME_'+a.condition+'.jsonl'),config,max_records=24)
        warm=RecordStore(run/('S5_RUNTIME_'+a.condition+'_WARMUP.jsonl'),config,max_records=1)
        if (store.ids or warm.ids) and not a.resume:raise ValueError('Existing run requires resume')
        pending=[e for e in inputs if 'S5:runtime:'+a.condition+':'+e['query_id'] not in store.ids]
        if not pending:return 0
        with GpuSession(run) as gpu:
            import torch
            import psutil
            from peft import get_peft_model_state_dict,set_peft_model_state_dict
            from safetensors.torch import load_file
            torch.set_num_threads(4)
            began=time.perf_counter()
            model,tokenizer,observed=load_training_model(modeldir,cfg['paths'][0],cfg)
            adapter=load_file(str(checkpoint/'adapter/adapter_model.safetensors'))
            set_peft_model_state_dict(model,adapter)
            if any(not torch.equal(v.detach().cpu(),adapter[k]) for k,v in get_peft_model_state_dict(model).items()):
                raise ValueError('Adapter load differs from checkpoint')
            del adapter
            model.gradient_checkpointing_disable();model.config.use_cache=True
            if hasattr(model.config,'text_config'):model.config.text_config.use_cache=True
            model.eval();load_seconds=time.perf_counter()-began
            observation={'condition':a.condition,'load_seconds':load_seconds,'modules':observed,
                'tokenizer_eos_id':tokenizer.eos_token_id,'tokenizer_eos_token':tokenizer.eos_token,
                'model_generation_config':model.generation_config.to_dict(),
                'model_text_eos_id':getattr(getattr(model.config,'text_config',model.config),'eos_token_id',None),
                'module_training':model.training,'use_cache':model.config.use_cache,
                'host_available_bytes':psutil.virtual_memory().available,'gpu_allocated_bytes':torch.cuda.memory_allocated(),
                'stop_bug_status':'HYPOTHESIS_UNTIL_ACTUAL_OUTPUT_IDS_CHECKED'}
            atomic_json(run/('S5_RUNTIME_'+a.condition+'_LOAD.json'),observation)
            print(canonical_json({'event':'S5_LOADED','condition':a.condition,'load_seconds':load_seconds,
                'tokenizer_eos':tokenizer.eos_token_id,'generation_eos':model.generation_config.eos_token_id}),flush=True)
            if a.condition=='merged':
                probes=[]
                for e in inputs[:3]:
                    ids=tokenizer.apply_chat_template(protocol_messages(e['question'],e['context_bundle'],'G0'),
                        tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
                    tokens=torch.tensor([ids],device='cuda:0')
                    with torch.inference_mode():logits=model(input_ids=tokens,use_cache=False).logits[0,-1].float().cpu()
                    probes.append((ids,logits));del tokens
                model=model.merge_and_unload(safe_merge=True).eval()
                checks=[]
                for ids,before in probes:
                    tokens=torch.tensor([ids],device='cuda:0')
                    with torch.inference_mode():after=model(input_ids=tokens,use_cache=False).logits[0,-1].float().cpu()
                    checks.append({'max_abs':float((after-before).abs().max()),
                        'relative_l2':float((after-before).norm()/before.norm().clamp_min(1e-12)),
                        'argmax_equal':int(after.argmax())==int(before.argmax())});del tokens
                passed=all(c['max_abs']<=.08 and c['relative_l2']<=.01 for c in checks)
                atomic_json(run/'S5_MERGE_LOGIT_GATE.json',{'checks':checks,'passed':passed,'config':config})
                if not passed:raise ValueError('Merged function exceeds frozen numerical tolerance')
                merged_dir=run/'S5_MERGED_MODEL'
                if merged_dir.exists():raise ValueError('Do not overwrite merged snapshot')
                model.save_pretrained(merged_dir,safe_serialization=True);tokenizer.save_pretrained(merged_dir)
                atomic_json(run/'S5_MERGED_MODEL_MANIFEST.json',{'artifacts':[artifact(f) for f in merged_dir.rglob('*') if f.is_file()],
                    'source_model':config['model_manifest'],'source_adapter':config['adapter_manifest'],'new_copy_only':True})
                del probes;gc.collect();torch.cuda.empty_cache()
            identity={'model':'Qwen3.5-2B','base_revision':cfg['paths'][0]['revision'],
                'condition':a.condition,'dtype':'bfloat16','adapter_manifest':config['adapter_manifest']}
            budget=Budget(run)
            tasks=([('warmup',inputs[0])] if not warm.ids else [])+[('formal',e) for e in pending]
            try:
                for kind,e in tasks:
                    gpu.check()
                    rid='S5:runtime:'+a.condition+':'+(e['query_id'] if kind=='formal' else 'warmup')
                    messages=protocol_messages(e['question'],e['context_bundle'],'G0')
                    ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                        enable_thinking=False,return_dict=False)
                    if len(ids)+512>4096:raise ValueError('Frozen prompt exceeds generation budget')
                    budget.reserve(request_id=rid,stage='S5_optional_lora',max_output_tokens=512,purpose=kind,model_identity=identity)
                    start=time.perf_counter();error=None;parsed=None;new_ids=[];raw=''
                    torch.cuda.reset_peak_memory_stats()
                    ctx=model.disable_adapter() if a.condition=='base' else nullcontext()
                    try:
                        tensor=torch.tensor([ids],device='cuda:0')
                        with ctx,torch.inference_mode():
                            result=model.generate(input_ids=tensor,attention_mask=torch.ones_like(tensor),
                                do_sample=False,max_new_tokens=512,use_cache=True)
                        torch.cuda.synchronize();new_ids=result[0,len(ids):].cpu().tolist()
                        raw=tokenizer.decode(new_ids,skip_special_tokens=True)
                        del tensor,result
                        try:
                            output=RAGPipeline.parse_output(raw);parsed={k:output[k] for k in ('answer','citations','abstain')}
                            visible={p['chunk_id'] for p in e['context_bundle']['pieces']}
                            if any(c not in visible for c in parsed['citations']):raise ValueError('Invisible citation')
                        except Exception as exc:error={'kind':'FORMAT_OR_CITATION','message':str(exc)}
                    except Exception as exc:error={'kind':type(exc).__name__,'message':str(exc)}
                    elapsed=time.perf_counter()-start
                    budget.finalize(rid,output_tokens=len(new_ids) if new_ids else None,outcome='failed' if error else 'success')
                    eos=model.generation_config.eos_token_id;effective_eos=[eos] if isinstance(eos,int) else eos or []
                    imend=[i for i,v in enumerate(new_ids) if v==tokenizer.eos_token_id]
                    payload=canonical_payload({'question_id':e['query_id'],'paper_ids':e['scope']['doc_ids'],
                        'dataset_role':'D_confirm','exposure_status':'DEVELOPMENT_VALIDATION','condition':a.condition,
                        'user_scope':e['scope'],'task_type_source':'UNLABELED_OFFICIAL_QASPER','model_identity':identity,
                        'prompt_hash':signature(messages),'actual_prompt_tokens':len(ids),
                        'context_intervals':[{'doc_id':p['doc_id'],'start':p['canonical_start'],'end':p['canonical_end']} for p in e['context_bundle']['pieces']],
                        'raw_response':raw,'parsed_response':parsed,'success':error is None,'error_type':error,
                        'output_tokens':len(new_ids),'actual_output_ids':new_ids,
                        'stop_reason':'effective_eos' if new_ids and new_ids[-1] in effective_eos else 'length' if len(new_ids)==512 else 'UNKNOWN',
                        'tokenizer_eos_positions':imend,'tokens_after_first_tokenizer_eos':len(new_ids)-imend[0]-1 if imend else None,
                        'wall_time_ms':elapsed*1000,'first_token_ms':None,'prefill_ms':None,'decode_ms':None,
                        'generation_tokens_per_second_including_prefill':len(new_ids)/elapsed,
                        'measured_resource':{'gpu_allocated_peak':torch.cuda.max_memory_allocated(),'gpu_reserved_peak':torch.cuda.max_memory_reserved(),
                            'host_available_after':psutil.virtual_memory().available,'process_rss_after':psutil.Process().memory_info().rss},
                        'load_seconds_separate':load_seconds,'budget_ticket':rid,
                        'proxy_unknowns':['claim_f1_proxy','support_proxy','citation_completeness_proxy']},campaign_id=run.name,task_id='S5_runtime')
                    (store if kind=='formal' else warm).add(rid,payload)
                    print(canonical_json({'event':'S5_RUNTIME','condition':a.condition,'formal_records':len(store.ids),
                        'output_tokens':len(new_ids),'tokens_after_im_end':payload['tokens_after_first_tokenizer_eos'],'error':error}),flush=True)
                    gc.collect();torch.cuda.empty_cache()
                    if error and error['kind']!='FORMAT_OR_CITATION':raise RuntimeError('Backend failure retained; bounded batch stopped')
            finally:
                del model,tokenizer;gc.collect();torch.cuda.empty_cache()
        atomic_json(run/('S5_RUNTIME_'+a.condition+'_COMPLETION.json'),{'formal_n':len(store.ids),'warmup_n':len(warm.ids),
            'status':'PASS_RUNTIME_EXECUTED' if len(store.ids)==24 else 'INCOMPLETE','records':artifact(store.path),'budget':budget.snapshot()})
        return 0


if __name__=='__main__':raise SystemExit(main())
