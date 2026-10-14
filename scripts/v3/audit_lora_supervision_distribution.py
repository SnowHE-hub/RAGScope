"""CPU-only all-row supervision checks plus 20 actual tokenizer decodes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from tokenizers import Tokenizer
from ragscope.experiments.runtime import configure,require_ram
from ragscope.generation.pipeline import RAGPipeline
from ragscope.indexing.identity import artifact,atomic_json,canonical_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args()
    runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    require_ram(2*1024**3)
    audit=json.loads((run/'SUPERVISION_20_AUDIT.json').read_text('utf8'))
    source=audit['sources'][0]
    if artifact(source['path'])!=source:raise ValueError('Original training examples changed')
    rows=[json.loads(l) for l in Path(source['path']).read_text('utf8').splitlines()]
    selected={x['query_id'] for x in audit['checks']}
    tok_path=runtime/'models/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc/tokenizer.json'
    tokenizer=Tokenizer.from_file(str(tok_path));eos=audit['tokenizer_eos_id']
    cases=[];failures=[];policies=Counter();counts=Counter()
    for row in rows:
        ids,labels,mask=row['input_ids'],row['labels'],row['attention_mask'];prompt=row['prompt_tokens']
        policies[row['evidence_policy']]+=1
        checks={'lengths_equal':len(ids)==len(labels)==len(mask)==row['total_tokens'],
            'prompt_not_supervised':all(x==-100 for x in labels[:prompt]),
            'padding_not_supervised':all(label==-100 for label,active in zip(labels,mask) if not active),
            'supervised_ids_match_input':all(i==l for i,l in zip(ids,labels) if l!=-100),
            'supervised_token_count':sum(l!=-100 for l in labels)==row['supervised_tokens'],
            'tokenizer_eos_supervised':any(i==eos and l==eos for i,l in zip(ids,labels)),
            'within_original_1024':len(ids)<=1024}
        target=row['messages'][-1]['content'];visible=set(re.findall(r'^\[([^\]\n]+)\]',row['messages'][1]['content'],re.MULTILINE))
        try:
            parsed=RAGPipeline.parse_output(target)
            checks['target_json_valid']=True
            checks['target_citations_visible']=all(c in visible for c in parsed['citations'])
            counts['target_abstain']+=int(parsed['abstain'])
        except Exception as exc:
            parsed=None;checks['target_json_valid']=False;checks['target_citations_visible']=False
        if not all(checks.values()):failures.append({'query_id':row['query_id'],'checks':checks})
        if row['query_id'] in selected:
            decoded=tokenizer.decode([i for i,l in zip(ids,labels) if l!=-100],skip_special_tokens=False)
            cases.append({'query_id':row['query_id'],'checks':checks,'decoded_supervised_target':decoded,
                'original_assistant_target':target,'assistant_content_present_exactly':target in decoded,
                'tokenizer_end_marker_present':'<|im_end|>' in decoded,
                'supervised_eos_positions':[i for i,(token,label) in enumerate(zip(ids,labels)) if token==eos and label==eos],
                'prompt_tokens':prompt,'supervised_tokens':row['supervised_tokens'],
                'citation_factual_support':'UNKNOWN_NOT_ESTABLISHED_BY_ID_VISIBILITY'})
    if len(rows)!=972 or len(cases)!=20:raise ValueError('Supervision population changed')
    pairs=[json.loads(l) for l in (run/'LORA_100_PAIRS.jsonl').read_text('utf8').splitlines()]
    inputs=[x['baseline']['actual_input_tokens'] for x in pairs if x['baseline']['actual_input_tokens'] is not None]
    result={'status':'PASS_SOURCE_DIAGNOSIS_RUNTIME_MICRO_PENDING' if not failures and all(c['assistant_content_present_exactly'] for c in cases) else 'FAIL_SUPERVISION_DIAGNOSIS',
        'train_n':len(rows),'all_row_numeric_or_target_failure_n':len(failures),'failures':failures,
        'decoded_cases':cases,'evidence_policy_counts':dict(policies),'target_abstain_n':counts['target_abstain'],
        'training_prompt_tokens':{'p50':statistics.median(x['prompt_tokens'] for x in rows),'max':max(x['prompt_tokens'] for x in rows)},
        'training_supervised_tokens':{'p50':statistics.median(x['supervised_tokens'] for x in rows),'max':max(x['supervised_tokens'] for x in rows)},
        'training_assistant_token_fraction_mean':statistics.mean(x['supervised_tokens']/x['total_tokens'] for x in rows),
        'paired_evaluation_actual_prompt_tokens':{'n':len(inputs),'p50':statistics.median(inputs),'max':max(inputs)},
        'source_train':source,'source_tokenizer':artifact(tok_path),'source_pairs':artifact(run/'LORA_100_PAIRS.jsonl'),
        'human_review_status':'NOT_REVIEWED','new_model_calls':0,
        'verdicts':['Target masking/EOS supervision and visible citation IDs checked; this does not establish semantic support.',
            'Training evidence policy and deployment prompt-length differences describe distribution shift, not a causal proof of LoRA regression.',
            'Old BF16 generate used eval/use_cache=True with no explicit EOS override; actual runtime generation_config/EOS still needs GPU S5 microdiagnosis.']}
    atomic_json(run/'S5_SUPERVISION_DISTRIBUTION_DIAGNOSIS.json',result)
    print(canonical_json({k:result[k] for k in ('status','train_n','all_row_numeric_or_target_failure_n','evidence_policy_counts','target_abstain_n','training_prompt_tokens','training_supervised_tokens','paired_evaluation_actual_prompt_tokens')}))
    return 0 if result['status'].startswith('PASS') else 2


if __name__=='__main__':raise SystemExit(main())
