"""Freeze real local/compare/section-summary actions for the preselected60 dev questions."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'scripts'))
from freeze_generation_v2 import model_client_factory
from ragscope.experiments.ledger import read_records
from ragscope.experiments.runtime import configure,require_ram
from ragscope.generation.context import pack_context
from ragscope.indexing.identity import artifact,atomic_json,canonical_json,signature
from ragscope.v3.compressed_index import CompressedBM25
from ragscope.v3.generation_protocols import protocol_messages
from ragscope.v3.splits import assert_inference_safe
from ragscope.v3.workflows import balanced_context,clarification_action,comparison_subqueries,summary_context


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args()
    runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    require_ram(2*1024**3)
    output=run/'S4_WORKFLOW_CONTEXTS.jsonl'
    if output.exists():raise ValueError('Do not overwrite frozen workflow inputs')
    label_manifest=json.loads((run/'S4_DEVELOPMENT_TASK_LABELS.json').read_text('utf8'))
    wanted=set(label_manifest['selected_query_ids'])
    labels={r['query_id']:r for r in map(json.loads,(run/'S4_DEVELOPMENT_TASK_LABELS.jsonl').read_text('utf8').splitlines()) if r['query_id'] in wanted}
    contexts={}
    with (run/'S3_SCOPED_CONTEXTS.jsonl').open(encoding='utf8') as f:
        for line in f:
            row=json.loads(line)
            if row['query_id'] in wanted and row['condition']=='B0_policy_neighbor_bridge':contexts[row['query_id']]=row
    if len(contexts)!=60 or len(labels)!=60:raise ValueError('Workflow60 inputs incomplete')
    papers={r['doc_id']:r for r in read_records(run/'S3_SCOPED_VECTOR_PAPERS.jsonl')}
    pidset={e['scope']['doc_ids'][0] for e in contexts.values()}
    chunks={}
    for pid in pidset:
        source=papers[pid]['source']
        if artifact(source['path'])!=source:raise ValueError('Paper source changed')
        chunks[pid]=json.loads(Path(source['path']).read_text('utf8'))['chunks']
    sparse_meta=json.loads((run/'S3_CHUNK_SPARSE_INDICES.json').read_text('utf8'))['variants']['R512_body']
    if artifact(sparse_meta['path'])!=sparse_meta:raise ValueError('Sparse manifest changed')
    sparse_directory=Path(sparse_meta['path']).parent;sparse=CompressedBM25.load(sparse_directory)
    rowids={}
    with (sparse_directory/'chunks.jsonl').open(encoding='utf8') as f:
        for i,line in enumerate(f):
            row=json.loads(line)
            if row['doc_id'] in pidset:rowids[row['chunk_id']]=i
    frozen=json.loads((run/'S3_SCOPED_CONTEXT_MANIFEST.json').read_text('utf8'))
    dev={json.loads(l)['query_id'] for l in (runtime/'corpora/20261014_v2/queries_dev.jsonl').read_text('utf8').splitlines()}
    client,_=model_client_factory(frozen['generator_spec'],dev)
    if client.identity!=frozen['generator_model_identity']:raise ValueError('Generator changed')
    entries=[];maximum=0
    for qid in label_manifest['selected_query_ids']:
        e=contexts[qid];label=labels[qid];assert_inference_safe(e)
        pid=e['scope']['doc_ids'][0];paper=chunks[pid];ranking=e['retriever_candidates']
        workflow_data=[]
        local=pack_context(ranking[:10],max_tokens=2048,scope=e['scope'])
        workflow_data.append(('W_local',local,{'subqueries':[e['question']],'action':'one scoped hybrid retrieval, compact unexpanded evidence'},e['question']))
        subqueries=comparison_subqueries(e['question']);groups=[]
        for index,query in enumerate(subqueries):
            scores=sparse.scores(query)
            ranked=sorted(paper,key=lambda c:(-scores[rowids[c['chunk_id']]],c['chunk_id']))[:10]
            groups.append((f'operand_{index+1}',[{**c,'score':float(scores[rowids[c['chunk_id']]])} for c in ranked]))
        compare,audit=balanced_context(groups,scope=e['scope'],max_tokens=2048)
        audit.update(subqueries=subqueries,ambiguous_operand_fallback=len(subqueries)==1,
            retrieval='existing global BM25 weights restricted to the user-selected paper; no new embedding model')
        compare_q=e['question']+'\nCompare only the requested aspects. State each side and its experimental conditions. If one side lacks evidence, say so.'
        workflow_data.append(('W_compare',compare,audit,compare_q))
        rankids={c['chunk_id'] for c in ranking}
        fullranking=ranking+[{**c,'score':0.0} for c in sorted(paper,key=lambda c:c['canonical_start']) if c['chunk_id'] not in rankids]
        summary,audit=summary_context(fullranking,scope=e['scope'],max_tokens=2048)
        summary_q=e['question']+'\nOrganize the supported key points across the selected sections. Do not add information absent from the evidence.'
        workflow_data.append(('W_summary',summary,audit,summary_q))
        for workflow,bundle,audit,workflow_question in workflow_data:
            context=asdict(bundle);assert_inference_safe(context)
            count=client.prompt_token_counter(protocol_messages(workflow_question,context,'G0'));maximum=max(maximum,count)
            if count+512>4096:raise ValueError('Shared model context budget exceeded')
            entry={'query_id':qid,'question':e['question'],'workflow_question':workflow_question,'scope':e['scope'],
                'dataset_role':e['dataset_role'],'workflow':workflow,'task':label['task'],'task_labels':label['task_labels'],
                'task_type_source':'RULE_PROXY_NOT_HUMAN','context_bundle':context,'workflow_audit':audit,
                'clarification_action':clarification_action(e['question'],e['scope']),
                'actual_generator_prompt_tokens_preflight':count,'input_id':'S4:'+workflow+':'+qid}
            entry['input_hash']=signature(entry);entries.append(entry)
    with output.open('x',encoding='utf8',newline='\n') as f:
        for entry in entries:f.write(canonical_json(entry)+'\n')
    result={'status':'PASS_FROZEN_WORKFLOW_CONTEXTS','questions_n':60,'formal_inputs_n':180,'maximum_prompt_tokens':maximum,
        'inputs':artifact(output),'labels':artifact(run/'S4_DEVELOPMENT_TASK_LABELS.json'),
        'source_contexts':artifact(run/'S3_SCOPED_CONTEXT_MANIFEST.json'),'sparse_manifest':sparse_meta,
        'generator_spec':frozen['generator_spec'],'generator_identity':client.identity,
        'runner':artifact(__file__),'workflow_implementation':artifact(PROJECT/'ragscope/v3/workflows.py'),
        'reference_budget':2048,'output_budget':512,'conditions':['W_local','W_compare','W_summary'],
        'W_clarify':'specified scope legally delegates to W_local; reuse its actual answer in utility vector; ambiguous cases require no model call',
        'new_model_calls':0,'H_QA_read':False,'gold_read':False,'human_review_status':'NOT_REVIEWED',
        'class_limitations':label_manifest['selected_task_counts']}
    atomic_json(run/'S4_WORKFLOW_CONTEXT_MANIFEST.json',result);print(canonical_json(result));return 0


if __name__=='__main__':raise SystemExit(main())
