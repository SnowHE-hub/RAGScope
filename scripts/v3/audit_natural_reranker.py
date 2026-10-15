"""Offline source-span funnel audit after natural candidate ranking is frozen."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ragscope.experiments.ledger import read_records
from ragscope.indexing.identity import artifact,atomic_json,canonical_json
from ragscope.v3.retrieval_diagnostics import coverage
from ragscope.v3.splits import read_split


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True)
    p.add_argument('--split',choices=['D_search','D_confirm'],default='D_search')
    p.add_argument('--models',nargs='+',choices=['bge','qwen'],default=['bge','qwen']);a=p.parse_args();run=Path(a.campaign).resolve()
    if run.parent!=Path('D:/RAGScope_Runtime/runs').resolve():raise ValueError('Wrong campaign')
    frozen=json.loads((run/'S3_SCOPED_CONTEXT_MANIFEST.json').read_text('utf8'))
    if artifact(frozen['inputs']['path'])!=frozen['inputs']:raise ValueError('Frozen candidate source changed')
    wanted=set(frozen['initial_D_search_query_ids']) if a.split=='D_search' else None;inputs={}
    with Path(frozen['inputs']['path']).open(encoding='utf8') as f:
        for line in f:
            e=json.loads(line)
            if e['dataset_role']==a.split and e['condition']=='B0_policy_neighbor_bridge' and (wanted is None or e['query_id'] in wanted):inputs[e['query_id']]=e
    labels={q['query_id']:q for q in read_split(run,a.split,purpose='development') if q['query_id'] in inputs}
    units={};needed={u for q in labels.values() for g in q['evidence_reference_sets'] for u in g}
    with Path('D:/RAGScope_Runtime/corpora/20261014_v2/all_documents.jsonl').open(encoding='utf8') as f:
        for line in f:
            for u in json.loads(line)['evidence_units']:
                if u['unit_id'] in needed:units[u['unit_id']]=u
    rows={};sources=[]
    for model in a.models:
        completion=run/('S3_NATURAL_RERANK_'+model+'_'+a.split+'_COMPLETION.json')
        if not completion.exists():raise ValueError('Missing terminal natural '+model+' scores')
        result=json.loads(completion.read_text('utf8'))
        if result['status']!='PASS_NATURAL_RERANK_EXECUTION' or artifact(result['records']['path'])!=result['records']:raise ValueError('Incomplete/changed scorer results')
        rows[model]={r['query_id']:r for r in read_records(Path(result['records']['path']))}
        recipe=json.loads(Path(result['records']['path']).with_suffix('.config.json').read_text('utf8'))
        if recipe['candidate_manifest']!=artifact(run/'S3_SCOPED_CONTEXT_MANIFEST.json'):raise ValueError('Scorer source identity changed')
        if set(rows[model])!=set(inputs):raise ValueError('Natural scorer denominator mismatch')
        sources.append(artifact(completion))
    details=[];groups=defaultdict(list)
    for qid,e in inputs.items():
        q=labels[qid];refs=q['evidence_reference_sets']
        baseline={'hybrid':{'ranking':e['retriever_candidates'][:10],'bundle':e['context_bundle']}}
        for model in rows:
            r=rows[model][qid]
            if r['candidate_ids']!=[c['chunk_id'] for c in e['retriever_candidates'][:50]]:raise ValueError('Scorer candidate pool mismatch')
            if len({w['chunk_id'] for w in r['windows']})!=r['candidate_n']:raise ValueError('Score mapping misses candidates')
            if model=='bge' and any(w['pair_tokens']>512 for w in r['windows']):raise ValueError('BGE window token overflow')
            if model=='qwen' and any(w['truncated'] or w['complete_input_tokens']>8192 for w in r['windows']):raise ValueError('Qwen input truncated or overflow')
            for condition,ranking in r['rankings'].items():baseline[condition]={'ranking':ranking,'bundle':r['contexts'][condition]}
        candidate=coverage(e['retriever_candidates'][:50],refs,units) if refs else None
        per={}
        for name,value in baseline.items():
            rank=coverage(value['ranking'],refs,units) if refs else None
            pack=coverage(value['bundle']['pieces'],refs,units) if refs else None
            item={'query_id':qid,'paper_id':q['doc_id'],'candidate_coverage':candidate,'rank_top10_coverage':rank,'packed_coverage':pack}
            groups[name].append(item);per[name]=item
        details.append({'query_id':qid,'conditions':per})
    summaries={}
    for name,values in groups.items():
        applicable=[v for v in values if v['candidate_coverage'] is not None]
        summaries[name]={'question_n':len(values),'evidence_applicable_n':len(applicable)}
        for stage in ('candidate_coverage','rank_top10_coverage','packed_coverage'):
            summaries[name][stage]={k:statistics.mean(v[stage][k] for v in applicable) for k in ('continuous_coverage','full_unit_recall','full_reference_coverage')}
    base=summaries['hybrid']['packed_coverage'];delta={name:g['packed_coverage']['full_reference_coverage']-base['full_reference_coverage'] for name,g in summaries.items() if name!='hybrid'}
    best=max(delta,key=delta.get)
    result={'status':'PASS_INDEPENDENT_NATURAL_RERANK_FUNNEL','split':a.split,'groups':summaries,
        'packed_full_reference_delta':delta,'best_by_predeclared_evidence_metric':best,
        'D_confirm_trigger':a.split=='D_search' and delta[best]>=.02,
        'development_screen_not_release_gate':True,'generation_quality':'UNMEASURED','human_review_status':'NOT_REVIEWED',
        'sources':sources,'details':details,'baseline_contexts':artifact(run/'S3_SCOPED_CONTEXT_MANIFEST.json'),
        'limitations':['Scope explicitly given; not unresolved full-corpus lookup.',
            'BGE uses FP32/512 windows; Qwen uses BF16/full input <=8192. Cross-model comparison does not isolate architecture.',
            'Evidence spans are an official-reference proxy; extra valid source evidence may be unannotated.',
            'Ranking time is not online end-to-end response time.']}
    atomic_json(run/('S3_NATURAL_RERANK_'+a.split+'_ANALYSIS.json'),result)
    print(canonical_json({k:result[k] for k in ('status','groups','packed_full_reference_delta','best_by_predeclared_evidence_metric','D_confirm_trigger')}));return 0


if __name__=='__main__':raise SystemExit(main())
