"""Eight source-verified short training fixtures; one optimizer step and exact resume."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from filelock import FileLock
from ragscope.experiments.ledger import RecordStore
from ragscope.experiments.runtime import configure,require_ram
from ragscope.generation.pipeline import RAGPipeline
from ragscope.indexing.identity import artifact,atomic_json,canonical_json,signature
from ragscope.training.run import load_training_model,train_loop
from ragscope.v3.audit import reference_texts
from ragscope.v3.budget import Budget
from ragscope.v3.gpu import GpuSession
from ragscope.v3.splits import read_split


def fixtures(runtime,run):
    source=runtime/'runs/20261014_v2_lora/prepared_2b_1024_v2/train_examples.jsonl'
    rows=[json.loads(l) for l in source.read_text('utf8').splitlines()]
    queries={q['query_id']:q for q in map(json.loads,(runtime/'corpora/20261014_v2/queries_train.jsonl').read_text('utf8').splitlines())}
    sealed_docs={d for q in read_split(run,'H_QA',purpose='inference') for d in q['scope']['doc_ids']}
    selected=[];checks=[]
    for row in sorted(rows,key=lambda r:r['query_id']):
        if row['answerability']!='answerable' or row['total_tokens']>512 or row['supervised_tokens']>160:continue
        target=RAGPipeline.parse_output(row['messages'][-1]['content']);q=queries[row['query_id']]
        matching=[r for r in q['metadata']['annotations'] if r['annotation_id']==row['annotation_id']]
        if not matching or target['abstain'] or not target['citations']:continue
        if target['answer'].strip() not in [r.strip() for r in reference_texts(q)]:continue
        annotation=matching[0]
        # Prepared v2 training contexts use an explicit train: namespace.
        if not all(c.startswith('train:') for c in target['citations']):continue
        if not {c.removeprefix('train:') for c in target['citations']}<=set(annotation['unit_ids']):continue
        if row['doc_id'] in sealed_docs:raise ValueError('Training fixture overlaps holdout paper')
        selected.append(row);checks.append({'query_id':row['query_id'],'annotation_id':row['annotation_id'],
            'reference_answer_matches':True,'citations_in_paired_annotation':True,
            'truth_source':'OFFICIAL_SOURCE_ANNOTATION_NOT_HUMAN_REVIEW_BY_OPERATOR'})
        if len(selected)==8:break
    if len(selected)!=8:raise ValueError('Fewer than eight eligible source-verified fixtures')
    return selected,checks,source


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    examples,checks,source=fixtures(runtime,run)
    training=runtime/'runs/20261014_v2_lora/train_2b_1024_seed42'
    cfg=json.loads((training/'steps.config.json').read_text('utf8'))['config'];cfg={**cfg,'max_steps':1}
    config={'runner':artifact(__file__),'source':artifact(source),'query_ids':[r['query_id'] for r in examples],
        'checks':checks,'steps':1,'generation_probes':2,'max_output_tokens':256,'config':cfg,
        'purpose':'SUPERVISION_CHAIN_TEST_NOT_GENERALIZATION','human_review_status':'NOT_REVIEWED'}
    if a.dry_run:print(canonical_json(config));return 0
    require_ram(2*1024**3)
    with FileLock(str(run/'writer.lock'),timeout=0):
        output=run/'S5_SUPERVISION_SMOKE';output.mkdir(exist_ok=True)
        store=RecordStore(output/'generation_probes.jsonl',config,max_records=4)
        if store.ids and not a.resume:raise ValueError('Existing probes require resume')
        if (output/'COMPLETION.json').exists():return 0
        with GpuSession(run):
            import torch
            from peft import get_peft_model_state_dict
            torch.set_num_threads(4)
            modeldir=runtime/'models/Qwen3.5-2B'/cfg['paths'][0]['revision']
            model,tokenizer,observed=load_training_model(modeldir,cfg['paths'][0],cfg)
            initial={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
            def losses():
                model.eval();values=[]
                with torch.inference_mode():
                    for row in examples:
                        batch={k:torch.tensor([row[k]],device='cuda:0') for k in ('input_ids','attention_mask','labels')}
                        values.append(float(model(**batch).loss));del batch;torch.cuda.empty_cache()
                return values
            def probes(phase):
                model.eval();budget=Budget(run)
                for row in examples[:2]:
                    rid='S5:smoke:'+phase+':'+row['query_id']
                    if rid in store.ids:continue
                    identity={'model':'Qwen3.5-2B','dtype':'bfloat16','phase':phase,'training_config_hash':signature(cfg)}
                    budget.reserve(request_id=rid,stage='S5_optional_lora',max_output_tokens=256,purpose='supervision_smoke',model_identity=identity)
                    new=[];error=None;raw=''
                    try:
                        ids=torch.tensor([row['input_ids'][:row['prompt_tokens']]],device='cuda:0')
                        with torch.inference_mode():out=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=False,
                            max_new_tokens=256,use_cache=True)
                        new=out[0,row['prompt_tokens']:].cpu().tolist();raw=tokenizer.decode(new,skip_special_tokens=True)
                        del ids,out
                    except Exception as exc:error={'kind':type(exc).__name__,'message':str(exc)}
                    budget.finalize(rid,output_tokens=len(new) if new else None,outcome='failed' if error else 'success')
                    store.add(rid,{'phase':phase,'query_id':row['query_id'],'raw_response':raw,'output_ids':new,
                        'output_tokens':len(new),'error':error,'model_identity':identity,'budget_ticket':rid})
                    if error:raise RuntimeError('Smoke generation failure retained')
                    gc.collect();torch.cuda.empty_cache()
            before_path=output/'BEFORE.json'
            if before_path.exists():before=json.loads(before_path.read_text('utf8'))['losses']
            else:
                before=losses();probes('before');atomic_json(before_path,{'losses':before,'config':config})
            has_checkpoint=(output/'training/checkpoints/latest.json').exists()
            result=train_loop(model,tokenizer,examples,cfg,output/'training',resume=has_checkpoint,
                empty_cache_after_microbatch=True)
            after=losses();probes('after')
            updated=get_peft_model_state_dict(model)
            changed=[k for k,v in updated.items() if not torch.equal(v.detach().cpu(),initial[k])]
            # Re-entering the existing loop verifies adapter/optimizer/scheduler/RNG/data position exactly.
            restored=train_loop(model,tokenizer,examples,cfg,output/'training',resume=True,
                empty_cache_after_microbatch=True)
            resume_receipt=json.loads((output/'training/resume_from_step_000001.json').read_text('utf8'))
            summary={'status':'PASS_SUPERVISION_CHAIN' if changed and all(resume_receipt['verification'].values()) else 'FAIL_SUPERVISION_CHAIN',
                'fixture_n':8,'optimizer_steps':result['actual_optimizer_steps'],'before_losses':before,'after_losses':after,
                'mean_loss_change':statistics.mean(after)-statistics.mean(before),'changed_tensor_n':len(changed),
                'changed_tensor_names':changed,'trainable_parameter_n':observed['trainable_parameters'],
                'resume_verification':resume_receipt['verification'],'probe_records':artifact(store.path),
                'generation_quality':'TRAINING_FIXTURES_ONLY_NOT_GENERALIZATION','config':config,'human_review_status':'NOT_REVIEWED'}
            atomic_json(output/'COMPLETION.json',summary);print(canonical_json({k:summary[k] for k in ('status','fixture_n','optimizer_steps','mean_loss_change','changed_tensor_n')}))
            del model,tokenizer;gc.collect();torch.cuda.empty_cache()
        return 0 if summary['status']=='PASS_SUPERVISION_CHAIN' else 2


if __name__=='__main__':raise SystemExit(main())
