"""Two bounded20-question controls; retain the original80 development outputs."""
import argparse
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'scripts'))
from finish_experiments import generate_jobs,questions
from ragscope.experiments.runtime import configure
from ragscope.indexing.identity import signature,atomic_json,artifact


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--run',required=True)
    args=parser.parse_args();configure(cpu_only=True);source,run=Path(args.source),Path(args.run)
    import numpy as np
    from ragscope.experiments.ledger import read_records
    from ragscope.v3.splits import read_split
    from ragscope.evaluation.generation_metrics import token_f1
    subset=sorted(questions(source)['D_confirm'],key=lambda q:signature([42,'finish_dev20',q['query_id']]))[:20]
    jobs=[(q,c,'qwen_2b','D_confirm') for c in ('CONTROL','ADAPT') for q in subset]
    generate_jobs(source,run,jobs,'QA_PROMPT_CONTROL')
    gold={q['query_id']:q for q in read_split(source,'D_confirm',purpose='development')}
    summary=json.loads((run/'GENERATION_SELECTION_DEV80.json').read_text('utf8'))['summary']
    for condition in ('CONTROL','ADAPT'):
        rows=[r for r in read_records(run/'QA_PROMPT_CONTROL.jsonl') if r['condition']==condition]
        scores=[]
        for row in rows:
            g=gold[row['query_id']];p=row['parsed'] or {}
            if g['answerability']=='answerable':scores.append(token_f1(p.get('answer',''),g['answer_refs']) if row['success'] and not p.get('abstain') else 0.)
        summary['qwen_2b:'+condition]={'n':len(rows),'f1':float(np.mean(scores)),'p50_s':float(np.median([r['elapsed_s'] for r in rows])),'failures':sum(not r['success'] for r in rows)}
    candidates=[key for key in summary if key.split(':')[0] in {'qwen_2b','qwen_4b'}]
    best=max(candidates,key=lambda key:summary[key]['f1'])
    eligible=[key for key in candidates if summary[key]['f1']>=summary[best]['f1']-.01]
    chosen=min(eligible,key=lambda key:summary[key]['p50_s'])
    model,policy=chosen.split(':')
    result={'model':model,'condition':'FINAL','qa_policy':policy,'summary':summary,
        'rule':'Best exposed dev20 F1 among baseline and tested controls; fastest within0.01. No H_QA gold used.',
        'controls':artifact(run/'QA_PROMPT_CONTROL.jsonl'),'original80_retained':artifact(run/'DEV_GENERATION.jsonl')}
    atomic_json(run/'GENERATION_SELECTION.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
