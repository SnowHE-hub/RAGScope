"""Run fixed100 same-BF16 baseline/adapter comparison after full context freeze."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import os
import random

from filelock import FileLock
from freeze_generation_v2 import read_json, verified_corpus
from ragscope.evaluation.v2 import summarize_records
from ragscope.experiments.generation_campaign import GenerationBudget,load_campaign
from ragscope.experiments.runtime import configure,runtime_home
from ragscope.indexing.identity import atomic_json,signature,artifact
from ragscope.training.checkpoint import CheckpointStore
from ragscope.training.evaluate import evaluate_paired
from ragscope.training.run import verify_lease,load_training_model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run',required=True)
    parser.add_argument('--campaign-dir',required=True)
    parser.add_argument('--model-dir',required=True)
    parser.add_argument('--output-dir')
    args=parser.parse_args()
    configure(cpu_only=False);os.environ['CUDA_VISIBLE_DEVICES']='0'
    training=Path(args.training_run);directory=Path(args.output_dir) if args.output_dir else training/'paired_test100'
    result=read_json(training/'training_result.json')
    if result.get('status')!='TRAIN_LOOP_COMPLETE' or not result.get('best_checkpoint_available'):
        raise ValueError('Comparison requires completed bounded training and dev-selected best checkpoint')
    immutable=read_json(training/'steps.config.json');config=immutable['config']
    store=CheckpointStore(training/'checkpoints',immutable)
    best=store.root/read_json(store.root/'best.json')['checkpoint'];store.verify(best)
    manifest,inputs=load_campaign(args.campaign_dir)
    selected={task['query_id']:inputs[task['input_id']] for task in manifest['tasks'] if task['experiment_id']=='E2' and task['method']=='R2'}
    ids=sorted(selected);random.Random(42).shuffle(ids);ids=ids[:100]
    if len(ids)!=100:raise ValueError('Frozen main matrix has fewer than the prescribed100 comparison queries')
    frozen=[selected[qid] for qid in ids]
    specification=manifest['provenance']['spec'];queries,units,_=verified_corpus(specification['corpus_dir'])
    train_docs={query['doc_id'] for query in queries['train']}
    audit={'manifest_hash':manifest['manifest_hash'],'query_ids':ids,'inputs':frozen,
        'rule':'query_id sorted then seed42 shuffle first100 E2/R2 contexts; no outputs read',
        'adapter_checkpoint':str(best),'adapter_manifest':artifact(best/'manifest.json'),
        'precision':'same BF16 base weights, identical backend/template, adapter toggled',
        'json_constraint':'prompt_only for both BF16 conditions; distinct from Q4 schema-constrained main matrix'}
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/'FROZEN_EVALUATION_INPUTS.json'
    if path.exists() and read_json(path)!=audit:raise ValueError('Immutable paired evaluation inputs changed')
    atomic_json(path,audit)
    rootrun=runtime_home()/'runs'/specification['campaign'];verify_lease(rootrun/'gpu_lease.json')
    with FileLock(str(runtime_home()/'runs/gpu.lock'),timeout=0),FileLock(str(directory/'writer.lock'),timeout=0):
        model,tokenizer,observed=load_training_model(args.model_dir,config['paths'][config['selected_path_index']],config)
        from peft import get_peft_model_state_dict,set_peft_model_state_dict
        from safetensors.torch import load_file
        import torch
        adapter=load_file(str(best/'adapter/adapter_model.safetensors'))
        expected=get_peft_model_state_dict(model)
        if set(adapter)!=set(expected) or any(adapter[key].shape!=expected[key].shape for key in adapter):
            raise ValueError('Best adapter structure differs from actual language modules')
        set_peft_model_state_dict(model,adapter)
        if any(not torch.equal(value.detach().cpu(),adapter[key]) for key,value in get_peft_model_state_dict(model).items()):
            raise ValueError('Best adapter did not load exactly')
        receipt=read_json(Path(args.model_dir)/'model_manifest.json')
        weights=[row for row in receipt['artifacts'] if row['path'].endswith('.safetensors')]
        identity={'model':receipt['model'],'revision':receipt['revision'],'weight_sha256':signature(weights),
            'backend_version':observed['transformers'],'backend':'transformers-local-bf16','quantization':'BF16',
            'tokenizer_template_hash':signature(tokenizer.chat_template),'torch':observed['torch'],'peft':observed['peft']}
        budget=GenerationBudget(rootrun/'generation_budget.json')
        try:
            rows=evaluate_paired(model,tokenizer,identity,frozen,directory,evidence_units=units,budget=budget,
                expected_n=100,train_doc_ids=train_docs,adapter_identity=artifact(best/'adapter/adapter_model.safetensors'))
            summary={'status':'COMPLETE','expected_n_per_condition':100,'frozen_inputs_hash':signature(audit),
                'baseline':summarize_records(rows['baseline']),'adapted':summarize_records(rows['adapted']),
                'identity':identity,'adapter':artifact(best/'adapter/adapter_model.safetensors')}
            atomic_json(directory/'SUMMARY.json',summary);print(json.dumps(summary),flush=True)
        finally:
            del model
            torch.cuda.empty_cache()
    return 0


if __name__=='__main__':raise SystemExit(main())
