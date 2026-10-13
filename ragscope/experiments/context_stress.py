"""Bounded 4K/8K resource diagnosis; no retrieval/answer-quality selection."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

from filelock import FileLock
from ragscope.experiments.generation_campaign import CountedClient
from ragscope.experiments.ledger import RecordStore, read_records
from ragscope.generation.context import pack_context, render_context, render_piece, validate_bundle
from ragscope.generation.pipeline import RAGPipeline
from ragscope.generation.prompts import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE, PROMPT_VERSION, OUTPUT_CONSTRAINT_VERSION
from ragscope.indexing.identity import artifact, atomic_json, signature, verify_artifacts

WINDOWS = (4096, 8192)
OUTPUT_RESERVE = 512
MAX_UNUSED_INPUT = 64


def messages(question, context):
    return [{'role': 'system', 'content': RAG_SYSTEM_PROMPT},
            {'role': 'user', 'content': RAG_USER_TEMPLATE.format(question=question, context=context)}]


def prefix_piece(document, end):
    return {'chunk_id': 'window-prefix:'+document['doc_id'], 'doc_id': document['doc_id'],
            'title': document['title'], 'text': document['text'][:end],
            'canonical_start': 0, 'canonical_end': end,
            'unit_spans': [{'unit_id': u['unit_id'], 'doc_id': document['doc_id'],
                'start': max(0, u['canonical_start']), 'end': min(end, u['canonical_end'])}
                for u in document['evidence_units'] if u['canonical_start'] < end and u['canonical_end'] > 0]}


def near_window_context(query, document, window, counter):
    if window not in WINDOWS:
        raise ValueError('Only predeclared4K/8K windows are permitted')
    scope=query['scope']; doc_id=document['doc_id']
    exact_scope=scope==doc_id or (isinstance(scope,dict) and scope.get('kind')=='document'
                                and scope.get('doc_ids')==[doc_id])
    if query['split'] != 'dev' or query['doc_id'] != doc_id or not exact_scope:
        raise ValueError('Context must be the specified dev paper')
    target = window-OUTPUT_RESERVE

    def count(end):
        return counter(messages(query['question'], render_piece(prefix_piece(document, end))))

    if count(len(document['text'])) < target-MAX_UNUSED_INPUT:
        raise ValueError('Canonical paper is too short to approach the real8K/4K input budget')
    lo, hi, best = 1, len(document['text']), 0
    while lo <= hi:
        mid = (lo+hi)//2
        if count(mid) <= target:
            best, lo = mid, mid+1
        else:
            hi = mid-1
    if not best:
        raise ValueError('Prompt headers alone exceed window')
    piece = prefix_piece(document, best)
    bundle = pack_context([piece], max_tokens=window*4, scope=query['scope'])
    actual = counter(messages(query['question'], render_context(bundle)))
    if not target-MAX_UNUSED_INPUT <= actual <= target or len(bundle.pieces) != 1 or bundle.pieces[0]['text'] != piece['text']:
        raise ValueError('Packed input did not retain exact source or near-window token occupancy')
    return {'context_bundle': asdict(bundle), 'expected_actual_input_tokens': actual,
            'output_reserve': OUTPUT_RESERVE, 'window': window,
            'unused_input_tokens': target-actual, 'canonical_document_checksum': document['canonical_checksum']}


def freeze_stress(directory, queries, documents, clients, source_artifacts):
    directory = Path(directory)
    if set(clients) != set(WINDOWS) or not verify_artifacts(source_artifacts):
        raise ValueError('Two verified window identities and source artifacts are required')
    if any(clients[w].identity.get('num_ctx')!=w for w in WINDOWS) or len({signature({k:v for k,v in clients[w].identity.items() if k!='num_ctx'}) for w in WINDOWS})!=1:
        raise ValueError('Window conditions must share the identical model/backend except num_ctx')
    if len(queries) != 100 or len({q['query_id'] for q in queries}) != 100 or any(q['split']!='dev' for q in queries):
        raise ValueError('Select only from the full frozen100 dev population')
    ordered = sorted(queries, key=lambda q: q['query_id'])
    random.Random(42).shuffle(ordered)
    selected, inputs, audit = [], {}, []
    for q in ordered:
        if len(selected) == 20:
            audit.append({'query_id':q['query_id'], 'status':'NOT_EVALUATED_CAP20', 'reason':'Predeclared20-query cap already filled'})
            continue
        safe_query = {key:q[key] for key in ('query_id','question','doc_id','scope','split')}
        try:
            variants = {window: near_window_context(safe_query, documents[q['doc_id']], window,
                        clients[window].prompt_token_counter) for window in WINDOWS}
        except (ValueError, KeyError) as exc:
            audit.append({'query_id':q['query_id'], 'status':'EXCLUDED_SOURCE_FEASIBILITY', 'reason':str(exc)})
            continue
        for window, variant in variants.items():
            iid = f'{window}:{q["query_id"]}'
            inputs[iid] = {'query':safe_query, **variant}
        selected.append(q['query_id'])
        audit.append({'query_id':q['query_id'], 'status':'SELECTED', 'reason':'Both exact source prefixes fit real window budgets within64 tokens'})
    if not selected:
        raise ValueError('No source-feasible dev query for both fixed windows')
    manifest = {'schema_version':'context-stress-v1','seed':42,'windows':list(WINDOWS),
        'model_id':'qwen_2b','identities':{str(w):clients[w].identity for w in WINDOWS},
        'query_ids':selected,'selection_audit':audit,'source_artifacts':source_artifacts,
        'generation_config':{'max_tokens':512,'temperature':0.0},
        'prompt_version':PROMPT_VERSION,'prompt_hash':signature([RAG_SYSTEM_PROMPT,RAG_USER_TEMPLATE]),
        'output_constraint_version':OUTPUT_CONSTRAINT_VERSION,'warmups_per_window':3,
        'selection_rule':'seed42 dev source feasibility only; same paired queries; no answer labels or generated outcomes',
        'interpretation':'Resource capability diagnosis only; excluded from main answer-quality ranking',
        'input_hashes':{iid:signature(source) for iid,source in inputs.items()},
        'implementation':artifact(Path(__file__))}
    manifest['manifest_hash']=signature(manifest)
    directory.mkdir(parents=True,exist_ok=True)
    target=directory/'STRESS_MANIFEST.json'
    if target.exists():
        if json.loads(target.read_text())!=manifest or json.loads((directory/'FROZEN_INPUTS.json').read_text())!=inputs:
            raise ValueError('Immutable diagnostic changed')
    else:
        atomic_json(directory/'FROZEN_INPUTS.json',inputs); atomic_json(target,manifest)
    return manifest


def load_stress(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'STRESS_MANIFEST.json').read_text())
    expected=manifest.pop('manifest_hash')
    if signature(manifest)!=expected:
        raise ValueError('Diagnostic manifest hash changed')
    manifest['manifest_hash']=expected
    if not verify_artifacts(manifest['source_artifacts']+[manifest['implementation']]):
        raise ValueError('Diagnostic source or implementation artifact changed')
    inputs=json.loads((directory/'FROZEN_INPUTS.json').read_text())
    if {iid:signature(source) for iid,source in inputs.items()}!=manifest['input_hashes']:
        raise ValueError('Diagnostic frozen input changed')
    if len(manifest['query_ids'])!=len(set(manifest['query_ids'])) or not 0<len(manifest['query_ids'])<=20:
        raise ValueError('Diagnostic query population changed')
    if set(inputs)!={f'{w}:{qid}' for w in WINDOWS for qid in manifest['query_ids']}:
        raise ValueError('Missing paired diagnostic input')
    for source in inputs.values():
        validate_bundle(source['context_bundle'],scope=source['query']['scope'])
        count=source['expected_actual_input_tokens']
        if source['query']['split']!='dev' or not source['window']-512-64<=count<=source['window']-512:
            raise ValueError('Diagnostic population or near-window occupancy changed')
    return manifest, inputs


def run_window(directory, window, client, budget, *, max_tasks=None):
    directory=Path(directory); manifest,inputs=load_stress(directory)
    if window not in WINDOWS or client.real_model is not True or client.identity!=manifest['identities'][str(window)]:
        raise ValueError('Real diagnostic model/window identity differs')
    stop_path=directory/f'STOP_{window}.json'
    if stop_path.exists():
        return json.loads((directory/f'SUMMARY_{window}.json').read_text())
    stop_reason=None; consecutive=0; completed=0
    with FileLock(str(directory/'writer.lock'),timeout=0):
        store=RecordStore(directory/'records'/f'window_{window}.jsonl',
            {'manifest_hash':manifest['manifest_hash'],'window':window,'purpose':'context_window_stress'},real_only=True,max_records=20)
        pending=[qid for qid in manifest['query_ids'] if qid not in store.ids]
        warmups=RecordStore(directory/'records'/f'warmup_{window}.jsonl',
            {'manifest_hash':manifest['manifest_hash'],'window':window,'purpose':'warmup'},real_only=True)
        sequence=[('warmup',pending[0],f'{time.time_ns()}:{i}') for i in range(3)] if pending else []
        sequence += [('formal',qid,qid) for qid in pending]
        for purpose,qid,rid in sequence:
            if purpose=='formal' and max_tasks is not None and completed>=max_tasks: break
            source=inputs[f'{window}:{qid}']; query=source['query']
            counted=CountedClient(client,budget,'context_window_stress_'+purpose,qid)
            pipeline=RAGPipeline(None,counted,use_cache=False,config=manifest['generation_config'])
            result=pipeline.answer(query['question'],context_bundle=source['context_bundle'],scope=query['scope'],formal_evaluation=True)
            if counted.last_ticket is None:
                stop_reason=result.error or {'kind':'request_not_started'}; break
            row={**asdict(result),'query_id':qid,'doc_id':query['doc_id'],'split':'dev','window':window,
                 'purpose':purpose,'expected_actual_input_tokens':source['expected_actual_input_tokens'],
                 'resources':result.backend_metadata.get('resources',{}),'call_ticket':counted.last_ticket,
                 'quality_ranking_eligible':False}
            (warmups if purpose=='warmup' else store).add(rid,row)
            completed+=int(purpose=='formal')
            kind=result.error.get('kind') if result.error else None
            consecutive=consecutive+1 if kind in {'oom','transport_error','timeout','service_not_ready'} else 0
            metadata=result.backend_metadata
            parity_bad=metadata.get('prompt_eval_count') is not None and metadata.get('token_count_match') is not True
            if kind=='oom' or consecutive>=3 or parity_bad:
                stop_reason={'kind':'backend_circuit_break' if not parity_bad else 'token_count_mismatch',
                             'last_error':result.error,'consecutive_failures':consecutive}
                break
            print(json.dumps({'window':window,'purpose':purpose,'completed_this_session':completed,'error':result.error}),flush=True)
        summary={'status':'COMPLETE' if len(store.records)==len(manifest['query_ids']) else 'BLOCKED' if stop_reason else 'RUNNING',
            'window':window,'manifest_hash':manifest['manifest_hash'],'n_expected':len(manifest['query_ids']),
            'n_observed':len(store.records),'n_failed':sum(bool(r.get('error')) for r in store.records),
            'n_pending':len(manifest['query_ids'])-len(store.records),'warmup_n':len(warmups.records),
            'warmup_failed':sum(bool(r.get('error')) for r in warmups.records),
            'stop_reason':stop_reason,'quality_ranking_eligible':False,
            'records':[artifact(p) for p in (store.path,warmups.path) if p.exists()]}
        atomic_json(directory/f'SUMMARY_{window}.json',summary)
        if stop_reason: atomic_json(stop_path,stop_reason)
        return summary
