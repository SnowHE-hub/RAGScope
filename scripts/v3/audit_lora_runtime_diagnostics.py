"""Independent fixed-development LoRA runtime recount, retaining every failure."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ragscope.experiments.ledger import read_records
from ragscope.indexing.identity import artifact,atomic_json,canonical_json
from ragscope.v3.audit import overlap,reference_texts
from ragscope.v3.splits import read_split


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args()
    run=Path(a.campaign).resolve();labels={q['query_id']:q for q in read_split(run,'D_confirm',purpose='development')}
    groups={};pairs={};sources=[]
    for condition in ('base','unmerged','merged'):
        path=run/('S5_RUNTIME_'+condition+'.jsonl')
        if not path.exists():continue
        rows=read_records(path);sources.append(artifact(path));pairs[condition]={r['question_id']:r for r in rows}
        words=[];failures=0;refusals=0
        for row in rows:
            q=labels[row['question_id']];parsed=row['parsed_response'];failed=not row['success'];failures+=failed
            if q['answerability']=='answerable':
                refusals+=bool(parsed and parsed['abstain'])
                words.append(0 if failed or not parsed or parsed['abstain'] else overlap(parsed['answer'],reference_texts(q))['legacy_token_f1'])
        latency=[r['wall_time_ms']/1000 for r in rows];tokens=[r['output_tokens'] for r in rows]
        after=[r['tokens_after_first_tokenizer_eos'] for r in rows if r['tokens_after_first_tokenizer_eos'] is not None]
        groups[condition]={'formal_n':len(rows),'answerable_n':len(words),'failures_n':failures,'answerable_refusals_n':refusals,
            'word_f1':statistics.mean(words) if words else None,'natural_generation_p50_seconds':statistics.median(latency) if latency else None,
            'output_tokens_mean':statistics.mean(tokens) if tokens else None,'output_tokens_p50':statistics.median(tokens) if tokens else None,
            'stop_counts':dict(Counter(r['stop_reason'] for r in rows)),
            'after_im_end_mean':statistics.mean(after) if after else None,'after_im_end_max':max(after) if after else None,
            'claim_f1_proxy':None,'support_proxy':None,'citation_completeness_proxy':None,'human_review_status':'NOT_REVIEWED',
            'cost_scope':'actual_frozen_context_BF16_generation_not_online_retrieval_E2E','decode_speed':'UNKNOWN_PREFILL_NOT_SEPARATED'}
    deltas={}
    for other in ('unmerged','merged'):
        if other not in pairs or 'base' not in pairs:continue
        ids=sorted(set(pairs['base'])&set(pairs[other]));token_change=[];time_change=[]
        for qid in ids:
            b,c=pairs['base'][qid],pairs[other][qid]
            token_change.append(c['output_tokens']-b['output_tokens']);time_change.append((c['wall_time_ms']-b['wall_time_ms'])/1000)
        deltas[other]={'matched_n':len(ids),'mean_token_change':statistics.mean(token_change) if ids else None,
            'mean_generation_seconds_change':statistics.mean(time_change) if ids else None}
    complete=len(groups)==3 and all(g['formal_n']==24 for g in groups.values())
    if complete and len({tuple(sorted(pairs[c])) for c in pairs})!=1:raise ValueError('Conditions use different questions')
    result={'status':'PASS_INDEPENDENT_RUNTIME_RECOUNT' if complete else 'PARTIAL_RUNTIME_RECOUNT',
        'groups':groups,'paired_differences':deltas,'sources':sources,'primary_quality_gate':'NOT_ESTABLISHED',
        'latency_causality':'Output length association alone does not identify per-token adapter overhead',
        'human_review_status':'NOT_REVIEWED','new_model_requests':0}
    atomic_json(run/'S5_RUNTIME_INDEPENDENT_ANALYSIS.json',result);print(canonical_json(result));return 0


if __name__=='__main__':raise SystemExit(main())
