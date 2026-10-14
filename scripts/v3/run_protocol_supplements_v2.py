"""Frozen 36 Oracle padding calls and six remaining prior-knowledge probes."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(PROJECT/'scripts'))
from freeze_generation_v2 import model_client_factory
from filelock import FileLock
from ragscope.experiments.ledger import RecordStore, read_records
from ragscope.experiments.runtime import configure, require_ram
from ragscope.generation.context import pack_context, reference_token_count, render_context
from ragscope.generation.pipeline import RAGPipeline
from ragscope.indexing.identity import artifact, atomic_json, canonical_json, signature
from ragscope.v3.audit import overlap
from ragscope.v3.budget import Budget
from ragscope.v3.generation_protocols import ProtocolClient, protocol_messages, protocol_schema
from ragscope.v3.local_runtime import ManagedOllama, counted_chat
from ragscope.v3.oracle_padding import padded_context
from ragscope.v3.splits import assert_inference_safe


def prepare(run, runtime, study):
    receipt_path = run/'S2_SUPPLEMENT_INPUTS.json'
    if receipt_path.exists():
        saved = json.loads(receipt_path.read_text('utf-8'))
        if saved['preparation_runner'] != artifact(PROJECT/'scripts/v3/run_protocol_supplements.py') or saved['padding_implementation'] != artifact(PROJECT/'ragscope/v3/oracle_padding.py'):
            raise ValueError('Frozen supplement implementation changed')
        if any(artifact(x['path']) != x for x in saved['outputs']):
            raise ValueError('Frozen supplement inputs changed')
        return saved
    labels = {x['query_id']:x for x in read_records(run/'splits/evaluator_only/S2_DIAGNOSTIC_LABELS.jsonl')}
    source = {(x['query_id'],x['context_kind']):x for x in read_records(run/'S2_PROTOCOL_INPUTS.jsonl')}
    selected = []
    for category in ('yes_no', 'extractive', 'free_form'):
        eligible = [q for q in study['query_ids'] if labels[q]['selection_stratum'] == category]
        selected += sorted(eligible, key=lambda q:signature([42,'S2_padding_predeclared',q]))[:4]
    if len(selected) != 12:
        raise ValueError('Expected four cases in each answerable stratum')
    paper_ids = {source[q,'oracle_audited']['scope']['doc_ids'][0] for q in selected}
    docs_path = runtime/'corpora/20261014_v2/all_documents.jsonl'
    docs = {}
    with docs_path.open(encoding='utf-8') as handle:
        for line in handle:
            doc = json.loads(line)
            if doc['doc_id'] in paper_ids:
                docs[doc['doc_id']] = doc
    inputs, audits = [], []
    with sqlite3.connect('file:'+str(run/'S0_EVIDENCE.sqlite')+'?mode=ro', uri=True) as db:
        for q in selected:
            clean, retrieval = source[q,'oracle_audited'], source[q,'retrieval_frozen']
            label = json.loads(db.execute('SELECT payload FROM queries WHERE query_id=?',(q,)).fetchone()[0])
            gold_ids = {u for group in label['evidence_reference_sets'] for u in group}
            doc = docs[clean['scope']['doc_ids'][0]]
            bundle, audit = padded_context(clean['question'], clean['context_bundle'], retrieval['context_bundle'], doc, gold_ids, clean['scope'])
            for piece in bundle['pieces']:
                if doc['text'][piece['canonical_start']:piece['canonical_end']] != piece['text']:
                    raise ValueError('Padding canonical coordinates are not exact source text')
            entry = {k:clean[k] for k in ('query_id','question','scope')}
            entry.update(context_kind='oracle_padded', context_bundle=bundle, input_id='S2:oracle_padded:'+q)
            assert_inference_safe(entry)
            entry['input_hash'] = signature(entry)
            inputs.append(entry)
            audits.append({'query_id':q, **audit, 'selection_stratum':labels[q]['selection_stratum'],
                'clean_reference_tokens':reference_token_count(render_context(clean['context_bundle'])),
                'retrieval_reference_tokens':reference_token_count(render_context(retrieval['context_bundle'])),
                'gold_exclusion_check':not bool(set(audit['noise_unit_ids']) & gold_ids)})
    for i,(question,expected) in enumerate((('What is the chemical formula of water?','H2O'),('How many days are in a week?','7'))):
        scope = {'kind':'document','doc_ids':['v3_prior:'+str(i)]}
        inputs.append({'query_id':'extra_prior:'+str(i), 'input_id':'S2:extra_prior:'+str(i),
            'question':question,'scope':scope,'context_kind':'prior',
            'context_bundle':asdict(pack_context([],scope=scope)), 'expected_answer':expected})
    path = run/'S2_SUPPLEMENT_INPUTS.jsonl'
    with path.open('x',encoding='utf-8',newline='\n') as handle:
        for row in inputs:
            handle.write(canonical_json(row)+'\n')
    saved = {'status':'FROZEN_BEFORE_NEW_OUTPUTS', 'question_n':12,'padding_calls':36,'prior_calls':6,
        'selection':'four per answerable stratum; deterministic hash; no model outcomes used',
        'study':artifact(run/'S2_PROTOCOL_STUDY_INPUTS.json'), 'source_documents':artifact(docs_path),
        'preparation_runner':artifact(__file__),'padding_implementation':artifact(PROJECT/'ragscope/v3/oracle_padding.py'),
        'outputs':[artifact(path)],'padding_audits':audits,'human_review_status':'NOT_REVIEWED'}
    atomic_json(receipt_path,saved)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',required=True)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    runtime = Path(configure(cpu_only=True)['home']).resolve()
    run = Path(args.campaign).resolve()
    if run.parent != runtime/'runs' or not run.name.startswith('v3_'):
        raise ValueError('Wrong campaign')
    require_ram(2*1024**3)
    with FileLock(str(run/'writer.lock'),timeout=0):
        study_path = run/'S2_PROTOCOL_STUDY_INPUTS.json'
        study = json.loads(study_path.read_text('utf-8'))
        if any(artifact(x['path']) != x for x in study['outputs']):
            raise ValueError('Main frozen study inputs changed')
        receipt = prepare(run,runtime,study)
        if args.prepare_only:
            print(canonical_json({'status':receipt['status'],'audits':receipt['padding_audits']}));return 0
        config = {'inputs':artifact(run/'S2_SUPPLEMENT_INPUTS.json'), 'runner':artifact(__file__),
            'protocol':study['protocol_implementation'],'runtime':artifact(PROJECT/'ragscope/v3/local_runtime.py'),
            'protocol_id':'G1','num_ctx':4096,'max_output_tokens':512,'seed':42,'temperature':0,
            'models':{k:v['identity'] for k,v in study['models'].items()},'answer_cache':False,
            'resource_guard_fix':artifact(run/'S2_SUPPLEMENT_RESOURCE_GUARD_FIX.json'),
            'first_attempt_records':artifact(run/'S2_SUPPLEMENT_RESULTS.jsonl')}
        if artifact(study['protocol_implementation']['path']) != study['protocol_implementation']:
            raise ValueError('Frozen protocol changed')
        path = run/'S2_SUPPLEMENT_RESULTS_v2.jsonl'
        if path.exists() and not args.resume:
            raise ValueError('Existing journal requires --resume')
        store = RecordStore(path,config,max_records=42)
        for previous in read_records(run/'S2_SUPPLEMENT_RESULTS.jsonl'):
            store.add(previous['record_id'], {**previous, 'reused_actual_result': True, 'reuse_source': artifact(run/'S2_SUPPLEMENT_RESULTS.jsonl')})
        rows = read_records(run/'S2_SUPPLEMENT_INPUTS.jsonl')
        labels = {x['query_id']:x for x in read_records(run/'splits/evaluator_only/S2_DIAGNOSTIC_LABELS.jsonl')}
        dev = {json.loads(l)['query_id'] for l in (runtime/'corpora/20261014_v2/queries_dev.jsonl').read_text('utf-8').splitlines()}
        budget = Budget(run)
        for mid,model in study['models'].items():
            pending = [x for x in rows if 'S2:supplement:'+mid+':'+x['input_id'] not in store.ids]
            if not pending:
                continue
            original,_ = model_client_factory(model['spec'],dev)
            if original.identity != model['identity']:
                raise ValueError('Pinned generator identity changed')
            # All three tokenizers share this frozen bundle; check before any model call.
            for entry in pending:
                if original.prompt_token_counter(protocol_messages(entry['question'],entry['context_bundle'],'G1'))+512 > 4096:
                    raise ValueError('Supplement exceeds complete prompt budget')
            with ManagedOllama(run,runtime/'tools/ollama-0.34.4/ollama.exe',num_ctx=4096) as service:
                for entry in pending:
                    require_ram(2*1024**3)
                    rid = 'S2:supplement:'+mid+':'+entry['input_id']
                    client = ProtocolClient(original,service,protocol='G1')
                    messages = protocol_messages(entry['question'],entry['context_bundle'],'G1')
                    if entry['context_kind'] == 'prior':
                        messages[0]['content'] = ('Answer this general factual question using your prior knowledge. '
                            'No evidence or citations are available or required; citations must be []. '
                            'Return JSON with answer, citations and abstain; abstain only when uncertain.')
                    response,parsed,error = {},None,None
                    started = time.perf_counter()
                    try:
                        response = counted_chat(run,service,client,request_id=rid,stage='S2_generation_protocol',
                            messages=messages,max_tokens=512,purpose='S2_'+entry['context_kind'],
                            output_schema=protocol_schema(entry['context_bundle'],'G1'))
                        parsed = RAGPipeline.parse_output(response['content'])
                        visible = {p['chunk_id'] for p in entry['context_bundle']['pieces']}
                        if client.last_metadata.get('token_count_match') is not True:
                            raise RuntimeError('Actual template token count differs')
                        if any(c not in visible for c in parsed['citations']):
                            raise ValueError('Output citation not visible')
                        parsed = {k:parsed[k] for k in ('answer','citations','abstain')}
                    except Exception as exc:
                        with budget.lock:
                            if rid not in budget._state()['reserved']:
                                raise
                        error = {'kind':type(exc).__name__,'message':str(exc)}
                        if not response:
                            response = {'raw_backend_response':client.raw_response,'backend_metadata':client.last_metadata}
                    refs = labels[entry['query_id']]['answer_refs'] if entry['context_kind']=='oracle_padded' else [entry['expected_answer']]
                    metrics = overlap(parsed['answer'],refs) if parsed and not error and not parsed['abstain'] else None
                    store.add(rid, {'campaign_id':run.name,'task_id':'S2_supplement','question_id':entry['query_id'],
                        'input_id':entry['input_id'],'model_id':mid,'model_identity':client.identity,
                        'dataset_role':'EXPOSED_DIAGNOSTIC' if entry['context_kind']=='oracle_padded' else 'CONSTRUCTED_RULE_DIAGNOSTIC',
                        'protocol':'G1','seed':42,'context_kind':entry['context_kind'],'user_scope':entry['scope'],
                        'prompt_hash':signature(messages),'actual_sent_payload':client.sent_payload,
                        'actual_prompt_tokens':client.last_metadata.get('prompt_eval_count'),
                        'raw_response':response,'parsed_response':parsed,'error_type':error,'success':error is None,
                        'stop_reason':client.last_metadata.get('done_reason'),'output_tokens':client.last_metadata.get('eval_count'),
                        'wall_time_ms':(time.perf_counter()-started)*1000,'measured_resource':client.last_metadata.get('resources'),
                        'budget_ticket':rid,'legacy_token_f1':metrics['legacy_token_f1'] if metrics else 0.0,
                        'word_metrics':metrics,'human_review_status':'NOT_REVIEWED'})
                    print(canonical_json({'event':'S2_SUPPLEMENT_PROGRESS','records':len(store.ids),'model':mid,'kind':entry['context_kind']}),flush=True)
                    if error and error['kind'] not in {'ValueError','JSONDecodeError'}:
                        raise RuntimeError('Backend or token mismatch recorded; stop without automatic repeat')
        atomic_json(run/'S2_SUPPLEMENT_COMPLETION.json',{'status':'TERMINAL','record_n':len(store.ids),
            'expected_n':42,'records':artifact(store.path),'budget':budget.snapshot(),'human_review_status':'NOT_REVIEWED'})
        print(canonical_json({'status':'SUPPLEMENT_TERMINAL','records':len(store.ids)}));return 0


if __name__ == '__main__':
    raise SystemExit(main())
