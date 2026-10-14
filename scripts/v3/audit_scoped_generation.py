"""Independent paired word/coverage audit; missing claim metrics block promotion."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
import numpy as np
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ragscope.experiments.ledger import read_records
from ragscope.generation.pipeline import RAGPipeline
from ragscope.indexing.identity import artifact,atomic_json,canonical_json
from ragscope.v3.audit import overlap,reference_texts
from ragscope.v3.records import FIELDS
from ragscope.v3.retrieval_diagnostics import coverage
from ragscope.v3.splits import read_split


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);p.add_argument('--split',choices=['D_search','D_confirm'],required=True);a=p.parse_args()
    run=Path(a.campaign).resolve()
    if run.parent!=Path('D:/RAGScope_Runtime/runs').resolve():raise ValueError('Wrong campaign')
    completion_path=run/('S3_SCOPED_GENERATION_'+a.split+'_COMPLETION.json')
    completion=json.loads(completion_path.read_text('utf8'))
    if completion['status']!='PASS_COMPLETE_PAIRED_GENERATION' or artifact(completion['records']['path'])!=completion['records']:raise ValueError('Generation not terminal complete')
    rows=read_records(Path(completion['records']['path']));wanted={x['question_id'] for x in rows}
    manifest=json.loads(Path(completion['context_manifest']['path']).read_text('utf8'));inputs={}
    if artifact(manifest['inputs']['path'])!=manifest['inputs']:raise ValueError('Context inputs changed')
    with Path(manifest['inputs']['path']).open(encoding='utf8') as f:
        for l in f:
            e=json.loads(l)
            if e['query_id'] in wanted and e['dataset_role']==a.split:inputs[e['input_id']]=e
    labels={q['query_id']:q for q in read_split(run,a.split,purpose='development') if q['query_id'] in wanted}
    needed={u for q in labels.values() for group in q['evidence_reference_sets'] for u in group};units={}
    with Path('D:/RAGScope_Runtime/corpora/20261014_v2/all_documents.jsonl').open(encoding='utf8') as f:
        for l in f:
            for u in json.loads(l)['evidence_units']:
                if u['unit_id'] in needed:units[u['unit_id']]=u
    groups=defaultdict(list);pairs=defaultdict(dict);mismatches=[]
    for row in rows:
        if not set(FIELDS)<=set(row):raise ValueError('Canonical required fields missing')
        e=inputs[row['input_id']];q=labels[row['question_id']];parsed=row['parsed_response'];failed=bool(row['error_type'])
        if not failed:
            observed=RAGPipeline.parse_output(row['raw_response']['content'])
            if {k:observed[k] for k in ('answer','citations','abstain')}!=parsed:raise ValueError('Raw/stored parser mismatch')
        answerable=q['answerability']=='answerable'
        score=None if not answerable else overlap(parsed['answer'],reference_texts(q))['legacy_token_f1'] if parsed and not failed and not parsed['abstain'] else 0.0
        if score!=row['legacy_token_f1']:mismatches.append(row['record_id'])
        if row['actual_prompt_tokens'] is not None and row['actual_prompt_tokens']!=e['actual_generator_prompt_tokens_preflight']:raise ValueError('Actual prompt tokens differ from freeze')
        refs=q['evidence_reference_sets'];cov=coverage(e['context_bundle']['pieces'],refs,units) if refs else None
        item={'query_id':q['query_id'],'paper_id':q['doc_id'],'condition':row['condition'],'legacy_token_f1':score,
            'coverage':cov,'failure':failed,'abstain':parsed['abstain'] if parsed else None,'wall_time_ms':row['wall_time_ms'],
            'source_record_hash':row['record_hash']}
        groups[row['condition']].append(item);pairs[q['query_id']][row['condition']]=item
    names=manifest['conditions'];paired=[]
    for qid,pair in pairs.items():
        if set(pair)!=set(names):raise ValueError('Missing paired output')
        b,c=pair[names[0]],pair[names[1]]
        if b['legacy_token_f1'] is not None:paired.append({'query_id':qid,'paper_id':b['paper_id'],'delta':c['legacy_token_f1']-b['legacy_token_f1']})
    paper_deltas=defaultdict(list)
    for x in paired:paper_deltas[x['paper_id']].append(x['delta'])
    keys=sorted(paper_deltas);rng=np.random.default_rng(42);bootstrap=[]
    for _ in range(2000):
        selected=rng.choice(keys,size=len(keys),replace=True)
        bootstrap.append(statistics.mean(d for k in selected for d in paper_deltas[k]))
    summaries={}
    for name,g in groups.items():
        applicable=[x for x in g if x['legacy_token_f1'] is not None];ev=[x for x in g if x['coverage'] is not None]
        summaries[name]={'n':len(g),'answerable_n':len(applicable),'failure_n':sum(x['failure'] for x in g),
            'answerable_abstain_n':sum(x['abstain'] is True for x in applicable),
            'legacy_token_f1':statistics.mean(x['legacy_token_f1'] for x in applicable),
            'evidence_metrics':{k:statistics.mean(x['coverage'][k] for x in ev) for k in ('continuous_coverage','full_unit_recall','full_reference_coverage')},
            'generation_wall_p50_ms':statistics.median(x['wall_time_ms'] for x in g),
            'latency_kind':'FROZEN_CONTEXT_GENERATION_NOT_ONLINE_E2E'}
    result={'status':'PASS_INDEPENDENT_PAIRED_DIAGNOSTIC' if not mismatches else 'FAIL_RECOUNT',
        'split':a.split,'groups':summaries,'score_mismatch_n':len(mismatches),'word_delta':statistics.mean(x['delta'] for x in paired),
        'word_delta_paper_cluster_ci95':np.quantile(bootstrap,[.025,.975]).tolist(),'applicable_questions':len(paired),'papers':len(keys),
        'paired_details':paired,'bootstrap_repeats':2000,'claim_f1_proxy':None,'support_proxy':None,'citation_completeness_proxy':None,
        'promotion_status':'BLOCKED_UNMEASURED_PRIMARY_METRICS','baseline_qualification':manifest['baseline_qualification'],
        'source_completion':artifact(completion_path),'human_review_status':'NOT_REVIEWED',
        'limitations':['Development diagnostic, not frozen unseen-test confirmation.',
            'Word overlap and coverage do not establish factual correctness or citation support.',
            'Per-stage/frozen-context generation times are not online E2E performance.']}
    atomic_json(run/('S3_SCOPED_GENERATION_'+a.split+'_ANALYSIS.json'),result)
    print(canonical_json({k:result[k] for k in ('status','groups','word_delta','word_delta_paper_cluster_ci95','promotion_status')}));return 0 if not mismatches else 2


if __name__=='__main__':raise SystemExit(main())
