"""All100-question development claim gate; UNKNOWN cannot be discarded to admit a candidate."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
import numpy as np
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ragscope.experiments.ledger import read_records
from ragscope.indexing.identity import artifact,atomic_json,canonical_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args();run=Path(a.campaign).resolve()
    if run.parent!=Path('D:/RAGScope_Runtime/runs').resolve():raise ValueError('Wrong campaign')
    completion_path=run/'S3_D_CONFIRM_CLAIM_COMPLETION.json';done=json.loads(completion_path.read_text('utf8'))
    if done['answer_rows_n']!=200 or artifact(done['scores']['path'])!=done['scores']:raise ValueError('Development claim population changed')
    rows=read_records(Path(done['scores']['path']));groups=defaultdict(list);pairs=defaultdict(dict)
    for row in rows:
        groups[row['condition']].append(row)
        if row['answerable']:pairs[row['query_id']][row['condition']]=row
    names=['B0_policy_neighbor_bridge','C1_policy_core_first_bridge']
    if set(groups)!=set(names) or any(len(g)!=100 for g in groups.values()) or any(set(p)!=set(names) for p in pairs.values()):raise ValueError('Paired denominator inconsistent')
    summaries={}
    for name,group in groups.items():
        applicable=[r for r in group if r['answerable']];metrics=[r['metrics'] for r in applicable]
        points=[m['claim_f1_proxy'] for m in metrics]
        summaries[name]={'question_n':len(group),'answerable_n':len(applicable),'failures_n':sum(r['failure'] for r in group),
            'answerable_abstentions_n':sum(r['abstain'] is True for r in applicable),
            'claim_f1_point_known_questions':sum(v is not None for v in points),
            'claim_f1_mean':statistics.mean(points) if all(v is not None for v in points) else None,
            'claim_f1_mean_lower':statistics.mean(m['claim_f1_proxy_lower'] for m in metrics),
            'claim_f1_mean_upper':statistics.mean(m['claim_f1_proxy_upper'] for m in metrics)}
        for field in ('context_support_precision_proxy','citation_support_precision_proxy','citation_completeness_proxy'):
            values=[m[field] for m in metrics if field in m and m[field]['n']>0]
            summaries[name][field]={'applicable_questions':len(values),'unknown_units':sum(v['unknown'] for v in values),
                'mean':statistics.mean(v['value'] for v in values) if values and all(v['value'] is not None for v in values) else None,
                'mean_lower':statistics.mean(v['lower'] for v in values) if values else None,
                'mean_upper':statistics.mean(v['upper'] for v in values) if values else None}
    paper_deltas=defaultdict(list)
    for pair in pairs.values():
        b,c=pair[names[0]],pair[names[1]];bm,cm=b['metrics'],c['metrics']
        paper_deltas[b['paper_id']].append({'lower':cm['claim_f1_proxy_lower']-bm['claim_f1_proxy_upper'],
            'upper':cm['claim_f1_proxy_upper']-bm['claim_f1_proxy_lower']})
    keys=sorted(paper_deltas);rng=np.random.default_rng(42);low=[];high=[]
    for _ in range(2000):
        selected=rng.choice(keys,len(keys),replace=True);values=[v for pid in selected for v in paper_deltas[pid]]
        low.append(statistics.mean(v['lower'] for v in values));high.append(statistics.mean(v['upper'] for v in values))
    b,c=summaries[names[0]],summaries[names[1]]
    delta=c['claim_f1_mean']-b['claim_f1_mean'] if b['claim_f1_mean'] is not None and c['claim_f1_mean'] is not None else None
    gate=delta is not None and delta>=.02
    result={'status':'PASS_INDEPENDENT_DEVELOPMENT_CLAIM_GATE_AUDIT','groups':summaries,'paired_answerable_n':len(pairs),'papers_n':len(keys),
        'claim_f1_delta_mean':delta,'claim_f1_delta_uncertainty_and_paper_bootstrap95':[float(np.quantile(low,.025)),float(np.quantile(high,.975))],
        'bootstrap_repeats':2000,'quality_candidate_admitted':gate,
        'quality_gate_reason':'UNKNOWN primary claims prevent proving required+0.02; all applicable questions retained' if delta is None else 'Required+0.02 primary point gain not met' if not gate else 'Development primary point threshold met; final and other guards still required',
        'efficiency_candidate_status':'NOT_ESTABLISHED_REQUIRES_ACTUAL_ONLINE_AND_QUALITY_NONINFERIORITY',
        'human_review_status':'NOT_REVIEWED','source_completion':artifact(completion_path),
        'old_full_claim_metric_operation_restarted':False,'no_known_subset_selection':True,
        'note':'Uncertainty envelope combines relationUNKNOWN bounds with2000 paper-cluster resamples; it is not an ordinary point-estimate CI.'}
    atomic_json(run/'S3_D_CONFIRM_CLAIM_GATE.json',result)
    print(canonical_json({k:result[k] for k in ('status','groups','claim_f1_delta_mean','quality_candidate_admitted','quality_gate_reason')}));return 0


if __name__=='__main__':raise SystemExit(main())
