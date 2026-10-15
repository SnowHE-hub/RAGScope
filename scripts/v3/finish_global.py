"""Full corpus inventory and1200 real global retrieval comparisons on final questions."""
import argparse
import json
from pathlib import Path
import sys
import time
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'scripts'))
from finish_experiments import units,load_vectors
from ragscope.experiments.runtime import configure
from ragscope.indexing.identity import artifact,atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--run',required=True)
    args=parser.parse_args();configure(cpu_only=True)
    source,run=Path(args.source),Path(args.run)
    import numpy as np
    from ragscope.v3.full_retrieval import FullCorpusRetriever
    from ragscope.v3.finish_runtime import evidence_bundle,resolve_policy
    from ragscope.v3.splits import read_split
    from ragscope.experiments.ledger import RecordStore
    from ragscope.evaluation.evidence import evidence_metrics
    engine=FullCorpusRetriever(source,run);vectors=load_vectors(run)
    target_ids=set()
    with (run.parents[1]/'corpora/20261014_v2/documents.jsonl').open(encoding='utf8') as handle:
        for line in handle:target_ids.add(json.loads(line)['doc_id'])
    missing=sorted(target_ids-set(engine.by_doc))
    if missing or len(target_ids)!=509:raise ValueError('Original509 target corpus not fully indexed')
    norms=[]
    for begin in range(0,len(engine.vectors),512):
        block=engine.vectors[begin:begin+512]
        if not np.isfinite(block).all() or not np.allclose(np.linalg.norm(block,axis=1),1,atol=1e-4):raise ValueError('Missing full dense vector')
    inventory={'status':'PASS','original_target_papers':len(target_ids),'target_papers_missing':missing,
        'total_papers':len(engine.by_doc),'chunk_rows':len(engine.chunks),'dense_shape':list(engine.vectors.shape),
        'body_bm25_rows':engine.chunk_sparse.weights.shape[0],'metadata_bm25_rows':engine.metadata_sparse.weights.shape[0],
        'document_dense_shape':list(engine.document_vectors.shape),'all_vectors_finite_unit_norm':True,
        'full_dense_manifest':artifact(run/'FULL_DENSE_INDEX.json')}
    atomic_json(run/'FULL_CORPUS_VALIDATION.json',inventory)
    atomic_json(run/'WORKBENCH_CATALOG.json',engine.documents)
    public=read_split(source,'H_QA');gold={q['query_id']:q for q in read_split(run,'H_QA',purpose='final_evaluation')}
    evidence=units(source,{pid for q in gold.values() for pid in q['scope']['doc_ids']})
    selected=json.loads((run/'RETRIEVAL_SELECTION.json').read_text('utf8'))['global_selected']['options']
    qa_policy=json.loads((run/'GENERATION_SELECTION.json').read_text('utf8')).get('qa_policy','FINAL')
    config={'runner':artifact(__file__),'retriever':artifact(PROJECT/'ragscope/v3/full_retrieval.py'),
        'conditions':['B0','FINAL'],'methods':['bm25','dense','hybrid'],'questions':[q['query_id'] for q in public],
        'selected':selected,'scope':'global: remove user-specified document IDs before retrieval, gold only in evaluation'}
    store=RecordStore(run/'FINAL_GLOBAL_RETRIEVAL.jsonl',config,max_records=1200)
    summary={}
    for condition in ('B0','FINAL'):
        for method in ('bm25','dense','hybrid'):
            key=condition+':'+method;scores=[]
            for q in public:
                started=time.perf_counter()
                context,result,kind=evidence_bundle(engine,q['question'],{'kind':'global'},vectors[q['query_id']],condition=resolve_policy(q['question'],{'kind':'global'},condition,qa_policy),
                    retrieval={'method':method,**({'candidate_depth':50} if condition=='B0' else selected)})
                elapsed=time.perf_counter()-started;g=gold[q['query_id']]
                metric=evidence_metrics(g['evidence_reference_sets'],context['unit_spans'],g['answerability'],evidence)
                relevant=set(g['scope']['doc_ids']);top10={c['doc_id'] for c in result['candidates'][:10]}
                row={'query_id':q['query_id'],'condition':condition,'method':method,'doc_hit_at10':bool(relevant&top10),
                    'evidence':metric,'elapsed_s':elapsed,'retrieved_doc_ids':[c['doc_id'] for c in result['candidates'][:10]],
                    'context_hash':context['context_hash']}
                rid=key+':'+q['query_id']
                if rid not in store.ids:store.add(rid,row)
                scores.append(row)
            applicable=[r['evidence']['full_reference_coverage'] for r in scores if r['evidence']['full_reference_coverage'] is not None]
            summary[key]={'n':len(scores),'document_hit_at10':float(np.mean([r['doc_hit_at10'] for r in scores])),
                'full_evidence_coverage':float(np.mean(applicable)),'evidence_n':len(applicable),
                'p50_lookup_pack_s':float(np.median([r['elapsed_s'] for r in scores]))}
            print(json.dumps({key:summary[key]}),flush=True)
    atomic_json(run/'FINAL_GLOBAL_SUMMARY.json',{'groups':summary,'inventory':inventory,'records':artifact(store.path),
        'interpretation':'Full1585-paper retrieval, including all509 original papers. No answer generation in this test.'})


if __name__=='__main__':main()
