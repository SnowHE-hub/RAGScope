"""Dev-only matched sparse screening; gold never enters retrieval or packing."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ragscope.experiments.runtime import configure,require_ram
from ragscope.generation.context import pack_context
from ragscope.indexing.identity import artifact,atomic_json,canonical_json
from ragscope.v3.compressed_index import CompressedBM25
from ragscope.v3.retrieval_diagnostics import coverage
from ragscope.v3.splits import read_split,assert_inference_safe


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args()
    runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    source_path=run/'S3_CHUNK_SPARSE_INDICES.json';source=json.loads(source_path.read_text('utf8'))
    public=read_split(run,'D_search',purpose='inference');gold={q['query_id']:q for q in read_split(run,'D_search',purpose='development')}
    wanted={d for q in public for d in q['scope']['doc_ids']};needed={u for q in gold.values() for g in q['evidence_reference_sets'] for u in g};units={}
    with (runtime/'corpora/20261014_v2/all_documents.jsonl').open(encoding='utf8') as f:
        for l in f:
            for u in json.loads(l)['evidence_units']:
                if u['unit_id'] in needed:units[u['unit_id']]=u
    path=run/'S3_SPARSE_CANDIDATE_SCREEN.jsonl'
    if path.exists():raise ValueError('Preserve completed screen')
    summaries={}
    with path.open('x',encoding='utf8',newline='\n') as out:
        for variant,pointer in source['variants'].items():
            require_ram(2*1024**3)
            if artifact(pointer['path'])!=pointer:raise ValueError('Sparse manifest changed')
            manifest=json.loads(Path(pointer['path']).read_text('utf8'));directory=Path(pointer['path']).parent
            if any(artifact(x['path'])!=x for x in manifest['artifacts']):raise ValueError('Sparse artifacts changed')
            index=CompressedBM25.load(directory);bydoc=defaultdict(list);lookup={}
            with (directory/'chunks.jsonl').open(encoding='utf8') as f:
                for i,l in enumerate(f):
                    chunk=json.loads(l)
                    if chunk['doc_id'] in wanted:
                        bydoc[chunk['doc_id']].append((i,chunk));lookup[chunk['chunk_id']]=chunk
            rows=[]
            for q in public:
                assert_inference_safe(q)
                scores=index.scores(q['question']);scoped=bydoc[q['scope']['doc_ids'][0]]
                ranked=sorted(scoped,key=lambda x:(-scores[x[0]],x[1]['chunk_id']))[:50]
                candidates=[{**c,'score':float(scores[i])} for i,c in ranked]
                selected=candidates[:10]
                bundle=asdict(pack_context(selected,max_tokens=2048,scope=q['scope'],expansion='neighbor',chunk_lookup=lookup))
                assert_inference_safe(bundle)
                # Only now consult evaluator reference units, after inference actions are fixed.
                refs=gold[q['query_id']]['evidence_reference_sets']
                metrics={k:coverage(v,refs,units) for k,v in {'candidate':candidates,'selected':selected,'packed':bundle['pieces']}.items()} if refs else None
                row={'query_id':q['query_id'],'variant':variant,'dataset_role':'EXPOSED_DIAGNOSTIC_D_SEARCH',
                    'question':q['question'],'user_scope':q['scope'],'retriever_candidates':candidates,'context_bundle':bundle,
                    'metrics':metrics,'applicable':bool(refs),'source_index':pointer,'human_review_status':'NOT_REVIEWED',
                    'new_model_calls':0,'retrieval_method':'bm25_only_screen_not_hybrid_quality'}
                out.write(canonical_json(row)+'\n');rows.append(row)
            applicable=[x for x in rows if x['applicable']]
            summaries[variant]={'n':len(rows),'applicable_n':len(applicable),'packed':{
                k:statistics.mean(x['metrics']['packed'][k] for x in applicable) for k in ('continuous_coverage','full_unit_recall','full_reference_coverage')},
                'candidate_full_reference_coverage':statistics.mean(x['metrics']['candidate']['full_reference_coverage'] for x in applicable)}
            del index,bydoc,lookup,rows
    metadata=[v for v in summaries if v.endswith('_metadata')]
    priority={'R512_metadata':0,'F512_O0_metadata':1,'P256_1024_metadata':2}
    selected=max(metadata,key=lambda v:(summaries[v]['packed']['full_reference_coverage'],summaries[v]['packed']['full_unit_recall'],-priority[v]))
    result={'status':'PASS_FOUR_SPARSE_DEV_SCREENS','groups':summaries,'selected_metadata_for_dense':selected,
        'selection_rule':'full reference coverage, then full unit recall, then simplest R512/F512/P256 tie preference',
        'next_dense_variants':['R512_body',selected],'inputs':artifact(source_path),'records':artifact(path),'new_model_calls':0,
        'D_confirm_read':False,'H_QA_read':False,'human_review_status':'NOT_REVIEWED',
        'limitations':['Cheap sparse-only selection for bounded next experiment; it does not prove hybrid or answer-quality superiority.',
            'All four variants use same 2048 reference budget and legacy neighbor packing. Failures/inapplicable cases retained.']}
    atomic_json(run/'S3_SPARSE_CANDIDATE_SCREEN.json',result);print(canonical_json(result));return 0


if __name__=='__main__':raise SystemExit(main())
