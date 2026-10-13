"""Freeze the identical E3 core shard before retrieval-dependent experiments."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import copy
from dataclasses import asdict
import json

from freeze_generation_v2 import (RUNTIME_POLICY, default_spec, document_lengths, model_client_factory,
    read_json, shared_context, verified_corpus)
from ragscope.experiments.generation_campaign import freeze_campaign, oracle_context
from ragscope.experiments.runtime import runtime_home
from ragscope.generation.context import pack_context
from ragscope.generation.prompts import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE, PROMPT_VERSION, OUTPUT_CONSTRAINT_VERSION
from ragscope.indexing.identity import artifact, atomic_json, signature


def freeze_e3(spec):
    spec=copy.deepcopy(spec)
    spec.pop('external_campaigns',None)  # Shards never reference other campaigns.
    out=Path(spec['output_dir']).with_name(spec['campaign']+'_generation_e3')
    spec['output_dir']=str(out)
    queries,units,corpus_artifact=verified_corpus(spec['corpus_dir'])
    if len(queries['test'])!=200 or spec['main_models']!=['qwen_2b','qwen_4b','qwen_9b']:
        raise ValueError('E3 shard must preserve all200 original test questions and three main Qwen models')
    receipt=read_json(spec['service_receipt'])
    if receipt.get('environment',{}).get('LLAMA_ARG_CACHE_RAM')!='0':
        raise ValueError('E3 requires the final audited backend cache policy')
    clients,models,pilots={}, {}, {}
    for model_id in spec['main_models']:
        spec['models'][model_id].setdefault('client_options',{})['backend_runtime_config']=RUNTIME_POLICY
        client,pilot=model_client_factory(spec['models'][model_id],{q['query_id'] for q in queries['dev']})
        clients[model_id],models[model_id],pilots[model_id]=client,client.identity,pilot
    lengths=document_lengths(spec['corpus_dir'])
    groups,inputs=[],{}
    for method in ('oracle','no_context'):
        ids=[]
        for query in queries['test']:
            bundle=oracle_context(query,units) if method=='oracle' else pack_context([])
            source=shared_context({'query':query,'context_bundle':asdict(bundle)},list(clients.values()))
            doc_id=query.get('doc_id') or query.get('metadata',{}).get('gold_doc_id_evaluation_only')
            source['document_length']=lengths.get(doc_id,{'doc_id':doc_id,'reference_n_tokens':None,'length_group':None,
                'reason':'Canonical document length metadata unavailable; never substitute retrieved context length'})
            input_id='E3:'+method+':'+query['query_id']
            ids.append(input_id); inputs[input_id]=source
        groups.append({'experiment_id':'E3','method':method,'model_ids':spec['main_models'],
                       'expected_n':200,'input_ids':ids,'status':'READY'})
    snapshot=out/'SERVICE_RUNTIME_SNAPSHOT.json'
    if not snapshot.exists():atomic_json(snapshot,receipt)
    manifest=freeze_campaign(out,models,groups,inputs,provenance={
        'spec':spec,'corpus_manifest':corpus_artifact,'model_pilots':pilots,
        'backend_runtime_config':RUNTIME_POLICY,'service_runtime_snapshot':artifact(snapshot),
        'prompt_version':PROMPT_VERSION,'prompt_hash':signature([RAG_SYSTEM_PROMPT,RAG_USER_TEMPLATE]),
        'output_constraint_version':OUTPUT_CONSTRAINT_VERSION,
        'citation_visibility_interpretation':'Citation ID visibility constrained by backend schema; not a model faithfulness score',
        'oracle_unanswerable_definition':'Empty context by experimental construction, not detected answerability',
        'latency_population':'Formal runs only, measured under frozen runtime policy; pilot latencies excluded',
        'shard_rule':'Exactly the full-main E3 task/input semantics; no retrieval selection or model outputs used.'})
    summary={'status':'FROZEN','manifest_hash':manifest['manifest_hash'],'tasks':len(manifest['tasks']),
             'per_model':400,'test_n':200,'output':str(out),'generation_calls':0}
    atomic_json(out/'FREEZE_READINESS.json',summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',default='20261014_v2')
    parser.add_argument('--spec')
    args=parser.parse_args()
    print(json.dumps(freeze_e3(read_json(args.spec) if args.spec else default_spec(runtime_home(),args.campaign))))
