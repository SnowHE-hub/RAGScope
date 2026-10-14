"""Freeze only the unchanged E2 core matrix once its four test sources are ready.

No SQuAD, simple-chunk, E7, ROUTE or auxiliary R3 source is required. This does
not execute models or shorten the core population to currently available rows.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import copy
import json

from freeze_generation_v2 import (RUNTIME_POLICY, default_spec, document_lengths,
    model_client_factory, read_json, read_jsonl, shared_context, verified_context_rows, verified_corpus)
from ragscope.experiments.generation_campaign import freeze_campaign, load_campaign
from ragscope.experiments.runtime import configure, runtime_home
from ragscope.generation.prompts import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE, PROMPT_VERSION, OUTPUT_CONSTRAINT_VERSION
from ragscope.indexing.identity import artifact, atomic_json, file_hash, signature

MAIN_MODELS = ['qwen_2b','qwen_4b','qwen_9b']
METHODS = [('R0','bm25'),('R1','dense'),('R2','hybrid'),('R3','hybrid_rerank')]


def shard_spec(spec):
    spec=copy.deepcopy(spec)
    spec.pop('external_campaigns',None)  # Sibling shards must never nest.
    spec['output_dir']=str(Path(spec['output_dir']).with_name(spec['campaign']+'_generation_e2'))
    if spec['main_models']!=MAIN_MODELS:
        raise ValueError('E2 requires all three original main Qwen models in their frozen order')
    if any(spec.get('model_blocks',{}).get(mid) for mid in MAIN_MODELS) or any(spec.get('blocks',{}).get('E2:'+method) for method,_ in METHODS):
        raise ValueError('E2 core shard cannot omit or block any of its twelve conditions')
    return spec


def selected_sources(spec):
    retrieval=Path(spec['retrieval_run'])
    selection_path=retrieval/'CANDIDATE_SELECTION.json'
    context_path=retrieval/'CONTEXT_SELECTION.json'
    selection,context=read_json(selection_path),read_json(context_path)
    if selection.get('split')!='dev' or selection.get('method')!='hybrid':
        raise ValueError('E2 structure selection must come from frozen Hybrid dev results')
    if context.get('split')!='dev' or context.get('candidate_selection_sha256')!=file_hash(selection_path):
        raise ValueError('E2 context strategy must match the frozen dev candidate selection')
    group=selection['selected']['structured']; expansion=context['selected'][group]
    if expansion not in {'none','neighbor','parent'}:
        raise ValueError('E2 requires the declared dev context expansion')
    sources={method:Path(spec.get('context_sources',{}).get('E2:'+method,
        retrieval/'records'/f'TEST_CONTEXT_{group}_{backend}_{expansion}.jsonl')) for method,backend in METHODS}
    return sources,{'candidate_selection':artifact(selection_path),'context_selection':artifact(context_path),
                    'structured_group':group,'expansion':expansion}


def test_population(spec):
    corpus=Path(spec['corpus_dir']); manifest=read_json(corpus/'SPLIT_MANIFEST.json')
    path=corpus/'queries_test.jsonl'
    if file_hash(path)!=manifest['files']['queries_test.jsonl']['sha256']:
        raise ValueError('Frozen QASPER test population changed')
    queries=read_jsonl(path); ids=[q['query_id'] for q in queries]
    if len(ids)!=200 or len(set(ids))!=200 or set(ids)!=set(manifest['query_ids']['test']):
        raise ValueError('E2 requires exactly all200 unique original test queries')
    return queries


def source_readiness(spec):
    """Read provenance/records only; do not inspect weights or load tokenizers."""
    spec=shard_spec(spec)
    result={'status':'PENDING','scope':'E2_ONLY','expected_tasks':2400,'expected_n_per_condition':200,
        'generation_calls':0,'models_loaded':False,'weights_or_tokenizers_loaded':False,'pending':[],
        'sources':{},'out':spec['output_dir']}
    try:
        queries=test_population(spec)
        paths,selection=selected_sources(spec)
        result['selection']=selection
    except (OSError,ValueError,KeyError) as exc:
        result['pending'].append({'item':'population_or_selection','reason':str(exc)})
        return result
    for method,path in paths.items():
        try:
            source,audit=verified_context_rows(path,queries)
            # verified_context_rows can intentionally select subsets for other
            # studies; a core-only test source must have precisely this full set.
            raw=read_jsonl(path)
            if len(raw)!=200 or {r['query_id'] for r in raw}!={q['query_id'] for q in queries}:
                raise ValueError('E2 source must contain exactly the full200 test rows, no extras')
            result['sources'][method]={'status':'READY','n':len(source),'artifact':audit}
        except (OSError,ValueError,KeyError) as exc:
            result['sources'][method]={'status':'PENDING','path':str(path),'reason':str(exc)}
            result['pending'].append({'item':'E2:'+method,'reason':str(exc)})
    if not result['pending']:
        result['status']='SOURCE_RECORDS_READY_MODELS_NOT_CHECKED'
    return result


def freeze_e2(spec, *, client_factory=model_client_factory):
    spec=shard_spec(spec); out=Path(spec['output_dir'])
    readiness=source_readiness(spec)
    atomic_json(out/'SOURCE_READINESS.json',readiness)
    if readiness['pending']:
        return readiness
    queries,_,corpus_artifact=verified_corpus(spec['corpus_dir'])
    if len(queries['test'])!=200 or len({q['query_id'] for q in queries['test']})!=200:
        raise ValueError('E2 requires all200 test queries')
    receipt=read_json(spec['service_receipt'])
    if receipt.get('environment',{}).get('LLAMA_ARG_CACHE_RAM')!='0':
        raise ValueError('E2 requires the final audited backend cache policy')
    clients,models,pilots={},{},{}
    for mid in MAIN_MODELS:
        options=spec['models'][mid].setdefault('client_options',{})
        if options.get('backend_runtime_config',RUNTIME_POLICY)!=RUNTIME_POLICY:
            raise ValueError('E2 runtime policy differs from the frozen protocol')
        options['backend_runtime_config']=RUNTIME_POLICY
        client,pilot=client_factory(spec['models'][mid],{q['query_id'] for q in queries['dev']})
        clients[mid],models[mid],pilots[mid]=client,client.identity,pilot
    paths,selection=selected_sources(spec)
    lengths=document_lengths(spec['corpus_dir']); groups,inputs,sources=[],{},{}
    for method,_ in METHODS:
        source_rows,audit=verified_context_rows(paths[method],queries['test'])
        if audit!=readiness['sources'][method]['artifact']:
            raise ValueError('E2 retrieval source changed during freeze')
        sources['E2:'+method]=audit
        ids=[]
        for row in source_rows:
            source=shared_context(row,list(clients.values()))
            query=source['query']; doc_id=query.get('doc_id') or query.get('metadata',{}).get('gold_doc_id_evaluation_only')
            source['document_length']=lengths.get(doc_id,{'doc_id':doc_id,'reference_n_tokens':None,'length_group':None,
                'reason':'Canonical document length metadata unavailable; never substitute retrieved context length'})
            iid='E2:'+method+':'+query['query_id']; ids.append(iid); inputs[iid]=source
        groups.append({'experiment_id':'E2','method':method,'model_ids':MAIN_MODELS,
            'expected_n':200,'input_ids':ids,'status':'READY'})
    snapshot=out/'SERVICE_RUNTIME_SNAPSHOT.json'
    if not snapshot.exists():atomic_json(snapshot,receipt)
    manifest=freeze_campaign(out,models,groups,inputs,provenance={
        'spec':spec,'corpus_manifest':corpus_artifact,'model_pilots':pilots,
        'retrieval_sources':sources,**selection,
        'backend_runtime_config':RUNTIME_POLICY,'service_runtime_snapshot':artifact(snapshot),
        'prompt_version':PROMPT_VERSION,'prompt_hash':signature([RAG_SYSTEM_PROMPT,RAG_USER_TEMPLATE]),
        'output_constraint_version':OUTPUT_CONSTRAINT_VERSION,
        'citation_visibility_interpretation':'Citation ID visibility constrained by backend schema; not a model faithfulness score',
        'latency_population':'Formal runs only, measured under frozen runtime policy; pilot latencies excluded',
        'shard_rule':'Exactly full-main E2 task/input semantics; no auxiliary population and no model outputs used.'})
    if len(manifest['tasks'])!=2400 or any(t['experiment_id']!='E2' for t in manifest['tasks']):
        raise ValueError('Invalid production E2 shard task count')
    load_campaign(out)
    result={'status':'FROZEN','scope':'E2_ONLY','manifest_hash':manifest['manifest_hash'],
        'tasks':2400,'per_model':800,'test_n':200,'output':str(out),'generation_calls':0,
        'source_readiness':artifact(out/'SOURCE_READINESS.json')}
    atomic_json(out/'FREEZE_READINESS.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',default='20261014_v2')
    parser.add_argument('--spec')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args(); configure(cpu_only=True)
    spec=read_json(args.spec) if args.spec else default_spec(runtime_home(),args.campaign)
    if args.check_only:
        result=source_readiness(spec)
        atomic_json(Path(shard_spec(spec)['output_dir'])/'SOURCE_READINESS.json',result)
    else:
        result=freeze_e2(spec)
    print(json.dumps(result,ensure_ascii=False))
    return 2 if result.get('pending') else 0


if __name__=='__main__':raise SystemExit(main())
