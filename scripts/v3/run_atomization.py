"""Counted local claim segmentation, retaining every error and original text."""
import argparse
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(PROJECT/'scripts'))
from filelock import FileLock
from freeze_generation_v2 import model_client_factory
from ragscope.experiments.runtime import configure, require_ram
from ragscope.experiments.ledger import RecordStore, read_records
from ragscope.indexing.identity import artifact, atomic_json, signature
from ragscope.v3.local_runtime import ManagedOllama, counted_chat
from ragscope.v3.quote_atoms import QUOTE_SYSTEM, quote_schema, quote_spans


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--max-groups',type=int)
    args=parser.parse_args()
    runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(args.campaign).resolve()
    if not run.is_relative_to(runtime/'runs') or not run.name.startswith('v3_'):raise ValueError('Wrong campaign')
    require_ram(2*1024**3)
    with FileLock(str(run/'writer.lock'),timeout=0):
        with (run/'ATOMIZATION_INPUTS.jsonl').open(encoding='utf-8') as h:groups=[json.loads(line) for line in h]
        old=json.loads((runtime/'runs/20261014_v2_generation/EXPERIMENT_MANIFEST.json').read_text('utf-8'))
        with (runtime/'corpora/20261014_v2/queries_dev.jsonl').open(encoding='utf-8') as h:dev={json.loads(l)['query_id'] for l in h}
        original,pilot=model_client_factory(old['provenance']['spec']['models']['qwen_4b'],dev)
        config={'source_inputs':artifact(run/'ATOMIZATION_INPUTS.jsonl'),'prompt':QUOTE_SYSTEM,'prompt_version':'quote-only-v2',
            'model_base_identity':original.identity,'num_ctx':8192,'max_tokens':4096,'temperature':0.0,
            'implementation':artifact(PROJECT/'ragscope/v3/quote_atoms.py'), 'runtime_implementation':artifact(PROJECT/'ragscope/v3/local_runtime.py'),
            'runner':artifact(__file__),'pilot':pilot,'truth_status':'SEGMENTATION_NOT_JUDGEMENT','human_review_status':'NOT_REVIEWED'}
        parent_path=run/'ATOMIZATION_QUOTES_V2.jsonl'
        parent_config_path=parent_path.with_suffix('.config.json')
        parent_config=json.loads(parent_config_path.read_text('utf-8'))
        for key,value in config.items():
            if key!='runner' and parent_config.get(key)!=value:
                raise ValueError('Parent segmentation semantic identity differs; do not reuse')
        config['reused_parent']={'journal':artifact(parent_path),'configuration':artifact(parent_config_path),
                                'only_change':'stop on consecutive backend failures, not independent quote-contract failures'}
        store=RecordStore(run/'ATOMIZATION_QUOTES_V3.jsonl',config,max_records=len(groups))
        for number,row in enumerate(read_records(parent_path),1):
            if row.get('config_hash')!=signature(parent_config):raise ValueError('Parent record config differs')
            body={k:v for k,v in row.items() if k not in ['record_id','config_hash','record_hash','recorded_at']}
            body['inherited_source']={'path':str(parent_path),'line':number,'record_hash':row['record_hash']}
            store.add(row['record_id'],body)
        pending=[g for g in groups if g['group_id'] not in store.ids]
        if args.max_groups is not None:
            if args.max_groups<1:raise ValueError('Invalid group limit')
            pending=pending[:args.max_groups]
        if not pending:print(json.dumps({'status':'ALL_SEGMENTATION_RECORDS_PRESENT'}));return 0
        binary=runtime/'tools/ollama-0.34.4/ollama.exe'
        errors=0;consecutive_backend_failures=0
        with ManagedOllama(run,binary) as service:
            client=service.client(original,num_ctx=8192)
            for group in pending:
                body={k:group[k] for k in ['query_id','question','texts']}
                messages=[{'role':'system','content':QUOTE_SYSTEM},{'role':'user','content':json.dumps(body,ensure_ascii=False)}]
                schema=quote_schema([t['source_id'] for t in group['texts']])
                rid='atom_quote_v2:'+group['group_id']
                result={};error=None;validated=[]
                try:
                    result=counted_chat(run,service,client,request_id=rid,stage='S0_S1_evidence_metrics',
                        messages=messages,max_tokens=4096,purpose='old_answer_quote_segmentation',output_schema=schema)
                    consecutive_backend_failures=0
                    parsed=json.loads(result['content'])
                    returned=parsed['sources'];expected={t['source_id']:t['text'] for t in group['texts']}
                    if len(returned)!=len(expected) or {r['source_id'] for r in returned}!=set(expected):raise ValueError('Missing/duplicate sources')
                    for source in returned:
                        validated.append({'source_id':source['source_id'],'units':quote_spans(expected[source['source_id']],source['units'])})
                except Exception as exc:
                    error={'kind':type(exc).__name__,'message':str(exc)};errors+=1
                    if getattr(exc,'kind',None) in {'oom','transport_error','timeout','device_policy_error','service_not_ready'}:
                        consecutive_backend_failures+=1
                store.add(group['group_id'],{'query_id':group['query_id'],'input_hash':group['input_hash'],
                    'model_response':result,'segments':validated if error is None else None,'error':error,
                    'human_review_status':'NOT_REVIEWED','real_model':bool(result.get('model_identity')),
                    'original_texts_unchanged':True})
                atomic_json(run/'ATOMIZATION_PROGRESS.json',{'expected_groups':len(groups),'recorded_groups':len(store.ids),
                    'valid_groups':sum(r['error'] is None for r in store.records),'failures':sum(r['error'] is not None for r in store.records),
                    'last_group_id':group['group_id'],'service_receipt':str(service.path),'new_llm_calls_budget_authority':'BUDGET.jsonl'})
                print(json.dumps({'event':'ATOMIZATION_PROGRESS','completed':len(store.ids),'expected':len(groups),'errors_this_invocation':errors}),flush=True)
                if consecutive_backend_failures>=3:raise RuntimeError('Three consecutive backend failures; retain all results')
        print(json.dumps({'status':'SEGMENTATION_BATCH_TERMINAL','recorded':len(store.ids),'expected':len(groups),'failures':errors}))
        return 0 if errors==0 else 2


if __name__=='__main__':raise SystemExit(main())
