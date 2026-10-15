"""Actual completed-index lookup probes; query embeddings frozen, costs not onlineE2E."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ragscope.experiments.runtime import configure,require_ram
from ragscope.indexing.identity import artifact,atomic_json,canonical_json,signature
from ragscope.v3.hierarchical_runtime import load_hierarchy
from ragscope.v3.splits import read_split


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args()
    runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    require_ram(2*1024**3);engine=load_hierarchy(run)
    queries=sorted(read_split(run,'D_search',purpose='inference'),key=lambda q:signature([42,'hierarchy20',q['query_id']]))[:20]
    meta=json.loads((run/'S3_DEV_QUERY_VECTORS.json').read_text('utf8'))
    if artifact(meta['vectors']['path'])!=meta['vectors'] or meta['model_identity']!=engine.embedding_identity:raise ValueError('Frozen query vectors differ')
    data=np.load(meta['vectors']['path'],allow_pickle=False);vectors=dict(zip(data['query_ids'].tolist(),data['vectors']))
    rows=[]
    for q in queries:
        for condition in ('specified','ambiguous','identifiable_mechanism_probe'):
            scope=q['scope'] if condition=='specified' else {'kind':'global'}
            start=time.perf_counter();result=engine.search(q['question'],scope=scope,query_vector=vectors[q['query_id']],
                embedding_identity=meta['model_identity'],visible_object_identified=condition=='identifiable_mechanism_probe')
            elapsed=time.perf_counter()-start
            if condition=='specified' and (result['document_lookup_performed'] or any(c['doc_id'] not in q['scope']['doc_ids'] for c in result['candidates'])):raise ValueError('Specified scope leaked')
            if condition=='ambiguous' and result['action']!='CLARIFY_DOCUMENT':raise ValueError('Missing object did not clarify')
            if condition=='identifiable_mechanism_probe' and not result['global_bypass_used']:raise ValueError('Global escape path missing')
            if len({c['chunk_id'] for c in result['candidates']})!=len(result['candidates']):raise ValueError('Duplicate merged source IDs')
            rows.append({'query_id':q['query_id'],'condition':condition,'result':result,'wall_seconds':elapsed,
                'source_kind':'actual real-index mechanism test with frozen query vector, not independent product accuracy',
                'visible_identifiability_ground_truth':'NOT_ASSERTED' if condition=='identifiable_mechanism_probe' else None,
                'gold_read':False,'human_review_status':'NOT_REVIEWED'})
    path=run/'S3_HIERARCHY_REAL_LOOKUPS.jsonl'
    if path.exists():raise ValueError('Preserve previous real lookup probes')
    path.write_text(''.join(canonical_json(r)+'\n' for r in rows),encoding='utf8')
    result={'status':'PASS_60_REAL_INDEX_MECHANISM_LOOKUPS','query_n':20,'lookup_n':60,'records':artifact(path),
        'new_generation_requests':0,'new_embedding_requests':0,'holdout_gold_read':False,'full_chunk_dense_status':'BLOCKED',
        'human_review_status':'NOT_REVIEWED','quality_claim':'NONE; explicit-identifiability branch forced only to exercise real fusion and bypass',
        'latency_kind':'actual retrieval with frozen query vectors; not onlineE2E'}
    atomic_json(run/'S3_HIERARCHY_REAL_LOOKUPS.json',result);print(canonical_json(result));return 0


if __name__=='__main__':raise SystemExit(main())
