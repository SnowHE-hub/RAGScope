"""Freeze visible-input development task labels without hidden answers."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ragscope.indexing.identity import atomic_json,artifact,canonical_json,signature
from ragscope.v3.splits import read_split,assert_inference_safe
from ragscope.v3.task_taxonomy import label_question


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args();run=Path(a.campaign).resolve()
    if run.parent!=Path('D:/RAGScope_Runtime/runs').resolve():raise ValueError('Wrong campaign')
    output=run/'S4_DEVELOPMENT_TASK_LABELS.jsonl'
    if output.exists():raise ValueError('Freeze labels once; do not overwrite after outcomes')
    rows=[]
    for split in ('D_search','D_confirm'):
        for q in read_split(run,split,purpose='inference'):
            assert_inference_safe(q)
            rows.append({**q,'dataset_role':split,**label_question(q['question'],q['scope'])})
    selected=[]
    for task in ('compare','summarize','explain','fact','other'):
        available=sorted((r for r in rows if r['task']==task),key=lambda r:signature([42,'S4_before_outcome60',r['query_id']]))
        selected.extend(r['query_id'] for r in available[:min(20,60-len(selected))])
    if len(selected)<60:
        remaining=sorted((r for r in rows if r['query_id'] not in selected),key=lambda r:signature([42,'S4_fill60',r['query_id']]))
        selected.extend(r['query_id'] for r in remaining[:60-len(selected)])
    with output.open('x',encoding='utf8',newline='\n') as f:
        for r in rows:f.write(canonical_json({**r,'selected_for_workflow60':r['query_id'] in selected})+'\n')
    summary={'status':'PASS_RULE_PROXY_LABELS_FROZEN','question_n':len(rows),'selected_n':len(selected),'selected_query_ids':selected,
        'population_task_counts':dict(Counter(r['task'] for r in rows)),
        'selected_task_counts':dict(Counter(r['task'] for r in rows if r['query_id'] in selected)),
        'labels':artifact(output),'taxonomy':artifact(PROJECT/'ragscope/v3/task_taxonomy.py'),'H_QA_read':False,'gold_read':False,
        'human_review_status':'NOT_REVIEWED','new_model_calls':0,'router_training_gate':'NOT_EVALUATED; workflow utility first',
        'limitations':['Rule labels are unreviewed proxies; rare comparison/summary classes cannot be fabricated.',
            'Evidence extent remains heuristic until retrieval; context sufficiency is never inferred from question length.']}
    atomic_json(run/'S4_DEVELOPMENT_TASK_LABELS.json',summary);print(canonical_json(summary));return 0


if __name__=='__main__':raise SystemExit(main())
