"""Reconstruct old funnel losses from immutable canonical spans and real scores."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import statistics
import sys

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from ragscope.experiments.runtime import configure
from ragscope.indexing.identity import artifact,atomic_json,canonical_json
from ragscope.v3.audit import journal_rows
from ragscope.v3.retrieval_diagnostics import coverage,top2_ranking

GROUPS=['E1_R512_hybrid_none','E1_R512_hybrid_neighbor','E1_R512_hybrid_rerank_neighbor',
        'E8_R512_hybrid_neighbor','E8_R512_hybrid_rerank_neighbor_gpu']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);args=p.parse_args()
    runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(args.campaign).resolve()
    if run.parent!=runtime/'runs' or not run.name.startswith('v3_'):raise ValueError('Wrong campaign')
    destination=run/'S3_OLD_RETRIEVAL_FUNNEL.jsonl'
    if destination.exists():raise ValueError('Audit already exists; do not overwrite historical evidence')
    old=runtime/'runs/20261014_v2_retrieval/records'
    paths=[old/(g+'.jsonl') for g in GROUPS]
    query_ids=set()
    for path in paths:
        for _,row in journal_rows(path):query_ids.add(row['query_id'])
    queries={}
    with sqlite3.connect('file:'+str(run/'S0_EVIDENCE.sqlite')+'?mode=ro',uri=True) as db:
        for q in query_ids:
            found=db.execute('SELECT payload FROM queries WHERE query_id=?',(q,)).fetchone()
            if found:queries[q]=json.loads(found[0])
    # Only legacy exposed dev labels, never the new H_QA labels.
    dev_path=runtime/'corpora/20261014_v2/queries_dev.jsonl'
    with dev_path.open(encoding='utf8') as handle:
        for line in handle:
            q=json.loads(line)
            if q['query_id'] in query_ids:queries[q['query_id']]=q
    if set(queries)!=query_ids:raise ValueError('Some legacy query gold is unavailable')
    needed={u for q in queries.values() for group in q['evidence_reference_sets'] for u in group}
    units={};doc_path=runtime/'corpora/20261014_v2/all_documents.jsonl'
    with doc_path.open(encoding='utf8') as handle:
        for line in handle:
            doc=json.loads(line)
            for u in doc['evidence_units']:
                if u['unit_id'] in needed:units[u['unit_id']]=u
    if set(units)!=needed:raise ValueError('Missing canonical evidence units')
    summaries={};independent_mismatches=[];window_checks=defaultdict(int)
    with destination.open('x',encoding='utf8',newline='\n') as out:
        for group,path in zip(GROUPS,paths):
            rows=[]
            for number,row in journal_rows(path):
                q=queries[row['query_id']];refs=q['evidence_reference_sets'];error=row.get('error')
                item={'group':group,'query_id':q['query_id'],'dataset_role':'EXPOSED_DIAGNOSTIC',
                    'source_path':str(path),'source_line':number,'source_record_hash':row['record_hash'],
                    'scope':row['scope'],'failure':error,'answerable':bool(refs),'human_review_status':'NOT_REVIEWED',
                    'document_candidate_stage':'NOT_PRESENT_IN_V2; chunk-object presence reported separately'}
                if refs and not error:
                    stages={'candidate':row['raw_retrieval'],'selected':row['selected_retrieval'],'packed':row['context_bundle']['pieces']}
                    item['coverage']={k:coverage(v,refs,units) for k,v in stages.items()}
                    item['correct_paper_in_chunk_candidates']=any(x['doc_id']==q['doc_id'] for x in row['raw_retrieval'])
                    item['candidate_count']=len(stages['candidate'])
                    item['packing_loss_continuous']=item['coverage']['selected']['continuous_coverage']-item['coverage']['packed']['continuous_coverage']
                    item['candidate_to_selected_loss_continuous']=item['coverage']['candidate']['continuous_coverage']-item['coverage']['selected']['continuous_coverage']
                    for key,value in item['coverage']['packed'].items():
                        recorded=row['scores']['context'].get(key)
                        if recorded is None or abs(value-recorded)>1e-8:independent_mismatches.append({'group':group,'query_id':q['query_id'],'metric':key,'new':value,'old':recorded})
                    windows=row.get('rerank_windows',[])
                    if windows:
                        item['top2_mean_selected_coverage']=coverage(top2_ranking(stages['candidate'],windows),refs,units)
                        item['window_score_source']='actual preserved model scores; same row device/backend/weights; no new inference'
                        byid=defaultdict(list)
                        for w in windows:
                            byid[w['chunk_id']].append(w);window_checks['window_n']+=1
                            window_checks['token_budget_violations']+=int(w['pair_tokens']>512)
                        for c in stages['candidate']:
                            ws=sorted(byid[c['chunk_id']],key=lambda w:w['start']);length=c['canonical_end']-c['canonical_start']
                            window_checks['candidate_n']+=1
                            window_checks['full_char_coverage_violations']+=int(ws[0]['start']!=0 or ws[-1]['end']!=length or any(a['end']<b['start'] for a,b in zip(ws,ws[1:])))
                        if row['selected_retrieval']:
                            scores=[x['rerank_score'] for x in row['selected_retrieval']]
                            window_checks['descending_score_violations']+=int(any(a<b for a,b in zip(scores,scores[1:])))
                out.write(canonical_json(item)+'\n');rows.append(item)
            applicable=[r for r in rows if r['answerable'] and not r['failure']]
            summary={'record_n':len(rows),'failure_n':sum(bool(r['failure']) for r in rows),'answerable_n':sum(r['answerable'] for r in rows),
                'successful_applicable_n':len(applicable),'scope_kinds':sorted({r['scope']['kind'] for r in rows})}
            for stage in ('candidate','selected','packed'):
                summary[stage]={key:statistics.mean(r['coverage'][stage][key] for r in applicable) if applicable else None for key in ('continuous_coverage','full_unit_recall','full_reference_coverage')}
            summary['packing_loss_case_n']=sum(r['packing_loss_continuous']>1e-9 for r in applicable)
            summary['candidate_to_selected_loss_case_n']=sum(r['candidate_to_selected_loss_continuous']>1e-9 for r in applicable)
            summary['correct_paper_in_chunk_candidates_n']=sum(r['correct_paper_in_chunk_candidates'] for r in applicable)
            reranked=[r for r in applicable if 'top2_mean_selected_coverage' in r]
            if reranked:
                summary['top2_vs_max_selected_continuous_delta']=statistics.mean(r['top2_mean_selected_coverage']['continuous_coverage']-r['coverage']['selected']['continuous_coverage'] for r in reranked)
            summaries[group]=summary
    receipt={'status':'PASS_INDEPENDENT_SPAN_RECONSTRUCTION' if not independent_mismatches else 'FAIL_METRIC_RECONCILIATION',
        'groups':summaries,'independent_metric_mismatch_n':len(independent_mismatches),'mismatches':independent_mismatches,
        'window_checks':dict(window_checks),'outputs':[artifact(destination)],'source_journals':[artifact(x) for x in paths],
        'source_dev':artifact(dev_path),'source_documents':artifact(doc_path),'runner':artifact(__file__),
        'human_review_status':'NOT_REVIEWED','new_llm_calls':0,'new_reranker_calls':0,
        'limitations':['Old global questions have no supplied title/document; not uniquely resolved by dataset gold.',
            'Top2 comparison is only rank-stage coverage on identical real window scores; no claim of end-to-end benefit.',
            'Separate CPU/GPU groups never pooled. Failed and inapplicable rows retained.']}
    atomic_json(run/'S3_OLD_RETRIEVAL_FUNNEL.json',receipt)
    print(canonical_json({'status':receipt['status'],'groups':summaries,'window_checks':dict(window_checks)}))
    return 0 if not independent_mismatches else 2


if __name__=='__main__':raise SystemExit(main())
