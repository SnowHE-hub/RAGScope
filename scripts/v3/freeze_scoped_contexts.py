"""Freeze matched hybrid/neighbor and hybrid/core-first development contexts."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'scripts'))
from freeze_generation_v2 import model_client_factory
from ragscope.experiments.ledger import read_records
from ragscope.experiments.runtime import configure,require_ram
from ragscope.generation.context import pack_context
from ragscope.indexing.identity import artifact,atomic_json,canonical_json,signature
from ragscope.v3.compressed_index import CompressedBM25
from ragscope.v3.core_packing import pack_core_first
from ragscope.v3.generation_protocols import protocol_messages
from ragscope.v3.scoped_retrieval import rank_scoped
from ragscope.v3.splits import read_split,assert_inference_safe


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args()
    runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    require_ram(2*1024**3)
    complete_path=run/'S3_SCOPED_VECTOR_COMPLETION.json';complete=json.loads(complete_path.read_text('utf8'))
    if complete['status']!='PASS_COMPACT_PAPER_GIVEN_VECTORS':raise ValueError('Compact alternative incomplete')
    records=read_records(Path(complete['records']['path']));paper_records={x['doc_id']:x for x in records}
    query_path=run/'S3_DEV_QUERY_VECTORS.json';query_meta=json.loads(query_path.read_text('utf8'))
    recipe=json.loads(Path(complete['recipe']['path']).read_text('utf8'))
    if query_meta['model_identity']!=recipe['embedding']:raise ValueError('Scoped/query embedding spaces differ')
    if artifact(query_meta['vectors']['path'])!=query_meta['vectors']:raise ValueError('Query vector matrix changed')
    query_matrix=np.load(query_meta['vectors']['path'],allow_pickle=False)
    query_vectors=dict(zip(query_matrix['query_ids'].tolist(),query_matrix['vectors']))
    sparse_meta=json.loads((run/'S3_CHUNK_SPARSE_INDICES.json').read_text('utf8'))['variants']['R512_body']
    if artifact(sparse_meta['path'])!=sparse_meta:raise ValueError('Sparse index manifest changed')
    sparse_manifest=json.loads(Path(sparse_meta['path']).read_text('utf8'))
    if any(artifact(x['path'])!=x for x in sparse_manifest['artifacts']):raise ValueError('Sparse index artifacts changed')
    directory=Path(sparse_meta['path']).parent;sparse=CompressedBM25.load(directory)
    corpus_rows={}
    with (directory/'chunks.jsonl').open(encoding='utf8') as f:
        for i,line in enumerate(f):
            x=json.loads(line)
            if x['doc_id'] in paper_records:corpus_rows[x['chunk_id']]=i
    study=json.loads((run/'S2_PROTOCOL_STUDY_INPUTS.json').read_text('utf8'))
    dev={json.loads(l)['query_id'] for l in (runtime/'corpora/20261014_v2/queries_dev.jsonl').read_text('utf8').splitlines()}
    client,_=model_client_factory(study['models']['qwen_2b']['spec'],dev)
    if client.identity!=study['models']['qwen_2b']['identity']:raise ValueError('Pinned 2B generator identity changed')
    papers={}
    for pid,r in paper_records.items():
        if artifact(r['source']['path'])!=r['source'] or artifact(r['vectors']['path'])!=r['vectors']:raise ValueError('Paper source/vector changed')
        chunks=json.loads(Path(r['source']['path']).read_text('utf8'))['chunks'];vectors=np.load(r['vectors']['path'],allow_pickle=False)
        if vectors['chunk_ids'].tolist()!=[x['chunk_id'] for x in chunks]:raise ValueError('Paper vectors mapped to wrong chunk IDs')
        papers[pid]=(chunks,vectors['vectors'])
    destination=run/'S3_SCOPED_CONTEXTS.jsonl'
    if destination.exists():raise ValueError('Preserve frozen contexts')
    selected=[];token_max=0;model_budget_failures=[]
    with destination.open('x',encoding='utf8',newline='\n') as out:
        for split in ('D_search','D_confirm'):
            questions=read_split(run,split,purpose='inference')
            if split=='D_search':selected=sorted((q['query_id'] for q in questions),key=lambda q:signature([42,'S3_initial40',q]))[:40]
            for q in questions:
                pid=q['scope']['doc_ids'][0];chunks,vectors=papers[pid]
                rows=[corpus_rows[x['chunk_id']] for x in chunks]
                ranking=rank_scoped(chunks,rows,sparse.scores(q['question']),query_vectors[q['query_id']],vectors)
                lookup={x['chunk_id']:x for x in chunks}
                for condition in ('B0_policy_neighbor_bridge','C1_policy_core_first_bridge'):
                    if condition.startswith('B0'):
                        bundle=pack_context(ranking[:10],max_tokens=2048,scope=q['scope'],expansion='neighbor',chunk_lookup=lookup);packing=None
                    else:bundle,packing=pack_core_first(ranking[:10],max_tokens=2048,scope=q['scope'],chunk_lookup=lookup)
                    context=asdict(bundle);assert_inference_safe(context)
                    tokens=client.prompt_token_counter(protocol_messages(q['question'],context,'G0'));token_max=max(tokens,token_max)
                    feasible=tokens+512<=4096
                    if not feasible:model_budget_failures.append({'query_id':q['query_id'],'condition':condition,'prompt_tokens':tokens})
                    entry={**q,'dataset_role':split,'condition':condition,'context_bundle':context,'packing_audit':packing,
                        'retriever_candidates':ranking,'actual_generator_prompt_tokens_preflight':tokens,'model_budget_feasible':feasible}
                    entry['input_id']='S3:'+condition+':'+q['query_id'];entry['input_hash']=signature(entry)
                    out.write(canonical_json(entry)+'\n')
    result={'status':'PASS_FROZEN_COMPACT_CONTEXTS' if not model_budget_failures else 'FAIL_PREFLIGHT_WITH_CASES_RETAINED',
        'inputs':artifact(destination),'initial_D_search_query_ids':selected,'conditions':['B0_policy_neighbor_bridge','C1_policy_core_first_bridge'],
        'reference_budget':2048,'generator_model_identity':client.identity,'generator_spec':study['models']['qwen_2b']['spec'],
        'protocol':'G0','maximum_actual_prompt_tokens':token_max,'model_budget_failures':model_budget_failures,
        'source_vectors':artifact(complete_path),'query_vectors':artifact(query_path),'sparse_manifest':sparse_meta,
        'runner':artifact(__file__),'ranking_implementation':artifact(PROJECT/'ragscope/v3/scoped_retrieval.py'),
        'packing_implementation':artifact(PROJECT/'ragscope/v3/core_packing.py'),'H_QA_read':False,'question_gold_read':False,
        'new_generation_requests':0,'human_review_status':'NOT_REVIEWED',
        'baseline_qualification':'Frozen v2 retrieval/generation policy on new corpus, with explicitly pinned GPU embedding backend bridge; not original CPU latency reproduction',
        'full_corpus_dense_status':'BLOCKED','not_a_release_candidate_until_development_guards_pass':True}
    atomic_json(run/'S3_SCOPED_CONTEXT_MANIFEST.json',result);print(canonical_json({k:result[k] for k in ('status','maximum_actual_prompt_tokens','model_budget_failures','baseline_qualification')}))
    return 0 if not model_budget_failures else 2


if __name__=='__main__':raise SystemExit(main())
