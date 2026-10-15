"""Real query vectors and matched baseline/task-action contexts for valid synthetic development tasks."""
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
from ragscope.indexing.identity import artifact,atomic_json,canonical_json
from ragscope.indexing.local_embedding import QUERY_INSTRUCTION
from ragscope.v3.compressed_index import CompressedBM25
from ragscope.v3.generation_protocols import protocol_messages
from ragscope.v3.local_runtime import ManagedOllama
from ragscope.v3.private_embeddings import PrivateEmbeddingClient
from ragscope.v3.scoped_retrieval import rank_scoped
from ragscope.v3.splits import assert_inference_safe
from ragscope.v3.task_workflows import named_comparison_queries,summary_paragraph_context
from ragscope.v3.workflows import balanced_context


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args()
    runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    destination=run/'S4_DEVELOPMENT_TASK_CONTEXTS.jsonl'
    if destination.exists():raise ValueError('Preserve frozen task contexts')
    require_ram(2*1024**3)
    public_path=run/'S4_DEVELOPMENT_TASK_QUESTIONS_v2.jsonl';questions=[json.loads(l) for l in public_path.read_text('utf8').splitlines()]
    for q in questions:assert_inference_safe(q)
    if not questions:raise ValueError('No source-verified task questions')
    selected_docs={q['scope']['doc_ids'][0] for q in questions}
    texts=list(dict.fromkeys([q['question'] for q in questions]+[text for q in questions if q['task_kind']=='compare' for text in named_comparison_queries(q['question'])]))
    with ManagedOllama(run,runtime/'tools/ollama-0.34.4/ollama.exe',num_ctx=4096) as service:
        embed=PrivateEmbeddingClient(run,service);matrix=embed.embed([QUERY_INSTRUCTION+t for t in texts]);embedding_identity=embed.identity
    vectors=dict(zip(texts,matrix))
    vector_path=run/'S4_DEVELOPMENT_TASK_QUERY_VECTORS.npz';np.savez(vector_path,texts=np.asarray(texts),vectors=matrix)
    records={r['doc_id']:r for r in read_records(run/'S3_SCOPED_VECTOR_PAPERS.jsonl')};papers={};documents={}
    for pid in selected_docs:
        r=records[pid]
        if artifact(r['source']['path'])!=r['source'] or artifact(r['vectors']['path'])!=r['vectors']:raise ValueError('Scoped source/vector changed')
        chunks=json.loads(Path(r['source']['path']).read_text('utf8'))['chunks'];array=np.load(r['vectors']['path'],allow_pickle=False)
        if array['chunk_ids'].tolist()!=[c['chunk_id'] for c in chunks]:raise ValueError('Scoped vector mapping changed')
        papers[pid]=(chunks,array['vectors'])
    with (runtime/'corpora/20261014_v2/all_documents.jsonl').open(encoding='utf8') as f:
        for line in f:
            doc=json.loads(line)
            if doc['doc_id'] in selected_docs:documents[doc['doc_id']]=doc
    frozen=json.loads((run/'S3_SCOPED_CONTEXT_MANIFEST.json').read_text('utf8'))
    sparse_meta=json.loads((run/'S3_CHUNK_SPARSE_INDICES.json').read_text('utf8'))['variants']['R512_body'];directory=Path(sparse_meta['path']).parent
    sparse=CompressedBM25.load(directory);rowids={}
    with (directory/'chunks.jsonl').open(encoding='utf8') as f:
        for i,line in enumerate(f):
            c=json.loads(line)
            if c['doc_id'] in selected_docs:rowids[c['chunk_id']]=i
    dev={json.loads(l)['query_id'] for l in (runtime/'corpora/20261014_v2/queries_dev.jsonl').read_text('utf8').splitlines()}
    generator,_=model_client_factory(frozen['generator_spec'],dev)
    entries=[];maximum=0
    for q in questions:
        pid=q['scope']['doc_ids'][0];chunks,array=papers[pid];indices=[rowids[c['chunk_id']] for c in chunks]
        def rank(text):return rank_scoped(chunks,indices,sparse.scores(text),vectors[text],array)
        ranking=rank(q['question']);lookup={c['chunk_id']:c for c in chunks}
        base=pack_context(ranking[:10],max_tokens=2048,scope=q['scope'],expansion='neighbor',chunk_lookup=lookup)
        if q['task_kind']=='summarize':
            candidate,audit=summary_paragraph_context(documents[pid],ranking,scope=q['scope'])
        else:
            subqueries=named_comparison_queries(q['question'])
            candidate,audit=balanced_context([(str(i),rank(text)[:10]) for i,text in enumerate(subqueries)],scope=q['scope'])
            audit.update(subqueries=subqueries,query_embedding_identity=embedding_identity)
        for condition,bundle,notes in (('B0_policy_neighbor_bridge',base,None),('C1_explicit_task_action',candidate,audit)):
            context=asdict(bundle);assert_inference_safe(context);count=generator.prompt_token_counter(protocol_messages(q['question'],context,'G0'));maximum=max(maximum,count)
            if count+512>4096:raise ValueError('Task prompt exceeds shared output/context budget')
            entries.append({**q,'condition':condition,'context_bundle':context,'packing_audit':notes,'actual_generator_prompt_tokens_preflight':count,
                'input_id':'S4_TASK:'+condition+':'+q['query_id'],'retriever_candidates':ranking})
    destination.write_text(''.join(canonical_json(e)+'\n' for e in entries),encoding='utf8')
    result={'status':'PASS_FROZEN_SOURCE_TASK_CONTEXTS','questions_n':len(questions),'formal_inputs_n':len(entries),
        'inputs':artifact(destination),'public_questions':artifact(public_path),'query_vectors':artifact(vector_path),
        'embedding_identity':embedding_identity,'maximum_prompt_tokens':maximum,'reference_budget':2048,'output_budget':512,
        'generator_spec':frozen['generator_spec'],'generator_identity':generator.identity,'runner':artifact(__file__),
        'task_workflows':artifact(PROJECT/'ragscope/v3/task_workflows.py'),'gold_read':False,'holdout_gold_read':False,
        'human_review_status':'NOT_REVIEWED','new_generation_requests':0}
    atomic_json(run/'S4_DEVELOPMENT_TASK_CONTEXT_MANIFEST.json',result);print(canonical_json({k:result[k] for k in ('status','questions_n','formal_inputs_n','maximum_prompt_tokens')}));return 0


if __name__=='__main__':raise SystemExit(main())
