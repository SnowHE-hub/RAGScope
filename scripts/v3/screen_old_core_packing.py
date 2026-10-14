"""Zero-generation matched packing ablation on preserved old retrieval rankings."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import statistics
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ragscope.experiments.runtime import configure
from ragscope.indexing.identity import artifact,atomic_json,canonical_json
from ragscope.v3.audit import journal_rows
from ragscope.v3.core_packing import pack_core_first
from ragscope.v3.retrieval_diagnostics import coverage


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args()
    runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    out=run/'S3_CORE_FIRST_OLD_SCREEN.jsonl'
    if out.exists():raise ValueError('Preserve existing screening')
    base=runtime/'runs/20261014_v2_retrieval/records'
    groups=['E1_R512_hybrid_neighbor','E1_R512_hybrid_rerank_neighbor','E8_R512_hybrid_neighbor','E8_R512_hybrid_rerank_neighbor_gpu']
    config=json.loads((base/'E8_R512_hybrid_rerank_neighbor_gpu.config.json').read_text())
    manifest=next(x for x in config['index_artifacts'] if x['path'].endswith('chunk_manifest.json'))
    if artifact(manifest['path'])!=manifest:raise ValueError('Original chunk manifest changed')
    chunk_art=next(x for x in json.loads(Path(manifest['path']).read_text())['artifacts'] if x['path'].endswith('chunks.jsonl'))
    if artifact(chunk_art['path'])!=chunk_art:raise ValueError('Original chunks changed')
    lookup={x['chunk_id']:x for x in (json.loads(l) for l in Path(chunk_art['path']).read_text('utf8').splitlines())}
    qs={}
    with sqlite3.connect('file:'+str(run/'S0_EVIDENCE.sqlite')+'?mode=ro',uri=True) as db:
        qs={qid:json.loads(payload) for qid,payload in db.execute('SELECT query_id,payload FROM queries')}
    with (runtime/'corpora/20261014_v2/queries_dev.jsonl').open(encoding='utf8') as f:
        for l in f:
            q=json.loads(l);qs[q['query_id']]=q
    needed={u for q in qs.values() for g in q['evidence_reference_sets'] for u in g}
    units={}
    with (runtime/'corpora/20261014_v2/all_documents.jsonl').open(encoding='utf8') as f:
        for l in f:
            for u in json.loads(l)['evidence_units']:
                if u['unit_id'] in needed:units[u['unit_id']]=u
    summaries={}
    with out.open('x',encoding='utf8',newline='\n') as f:
        for group in groups:
            pairs=[];failures=[]
            for line,row in journal_rows(base/(group+'.jsonl')):
                refs=qs[row['query_id']]['evidence_reference_sets']
                item={'group':group,'query_id':row['query_id'],'source_line':line,'source_record_hash':row['record_hash'],
                    'dataset_role':'EXPOSED_DIAGNOSTIC','error':row['error'],'answerable':bool(refs),'human_review_status':'NOT_REVIEWED'}
                if not row['error']:
                    chunks=[]
                    for c in row['selected_retrieval']:
                        if c['chunk_id'] not in lookup:raise ValueError('Scoped chunk missing from shared original corpus index')
                        source=lookup[c['chunk_id']]
                        if source['canonical_start']!=c['canonical_start'] or source['canonical_end']!=c['canonical_end']:raise ValueError('Canonical chunk identity mismatch')
                        chunks.append({**source,**c})
                    budget=row['context_bundle']['budget']['max_reference_tokens']
                    bundle,audit=pack_core_first(chunks,max_tokens=budget,scope=row['scope'],chunk_lookup=lookup)
                    item.update(context_bundle=asdict(bundle),packing_audit=audit,reference_budget=budget)
                    if refs:
                        item['old']=coverage(row['context_bundle']['pieces'],refs,units)
                        item['new']=coverage(asdict(bundle)['pieces'],refs,units)
                        pairs.append(item)
                else:failures.append(item)
                f.write(canonical_json(item)+'\n')
            summaries[group]={'applicable_n':len(pairs),'failures_retained_n':len(failures)}
            for key in ('continuous_coverage','full_unit_recall','full_reference_coverage'):
                summaries[group][key]={'old':statistics.mean(x['old'][key] for x in pairs),
                    'new':statistics.mean(x['new'][key] for x in pairs),
                    'delta':statistics.mean(x['new'][key]-x['old'][key] for x in pairs)}
            summaries[group]['improved_case_n']=sum(x['new']['continuous_coverage']>x['old']['continuous_coverage'] for x in pairs)
            summaries[group]['worsened_case_n']=sum(x['new']['continuous_coverage']<x['old']['continuous_coverage'] for x in pairs)
    result={'status':'PASS_MATCHED_OFFLINE_SCREEN','groups':summaries,'results':artifact(out),
        'source_chunks':chunk_art,'packing_implementation':artifact(PROJECT/'ragscope/v3/core_packing.py'),
        'new_llm_calls':0,'new_retrieval_calls':0,'human_review_status':'NOT_REVIEWED',
        'conclusion_scope':'Original ranked chunks and reference budgets held fixed. Evidence coverage only, no answer-quality claim.'}
    atomic_json(run/'S3_CORE_FIRST_OLD_SCREEN.json',result);print(canonical_json(result));return 0


if __name__=='__main__':raise SystemExit(main())
