"""One zero-generation development compatibility projection using full source paragraphs."""
import argparse
import copy
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'scripts/v3'))
from repair_task_reference_whitespace import source_span
from ragscope.experiments.ledger import read_records
from ragscope.indexing.identity import artifact,atomic_json,canonical_json
from ragscope.v3.task_references import validate_reference


def source_paragraph(source,quote):
    literal=source_span(source,quote);start=source.find(literal)
    lo=source.rfind('\n\n',0,start);lo=lo+2 if lo>=0 else 0
    hi=source.find('\n\n',start+len(literal));hi=len(source) if hi<0 else hi
    expanded=source[lo:hi]
    if len(expanded)>4096:raise ValueError('Complete source paragraph exceeds frozen4096-character bound')
    return expanded


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args();run=Path(a.campaign).resolve()
    if run.parent!=Path('D:/RAGScope_Runtime/runs').resolve():raise ValueError('Wrong campaign')
    receipt=run/'S4_DEVELOPMENT_TASK_PARAGRAPH_COMPATIBILITY.json'
    if receipt.exists():
        saved=json.loads(receipt.read_text('utf8'))
        if saved['runner']!=artifact(__file__) or any(artifact(x['path'])!=x for x in saved['outputs']):raise ValueError('Compatibility identity changed')
        print(canonical_json({'status':'REUSED_COMPLETED_COMPATIBILITY'}));return 0
    papers={p['source_id']:p for p in map(json.loads,(run/'S4_DEVELOPMENT_TASK_SOURCE_PAPERS.jsonl').read_text('utf8').splitlines())}
    teachers=read_records(run/'S4_DEVELOPMENT_TASK_TEACHER.jsonl');valid=[];failed=[]
    for row in teachers:
        pid=row['record_id'];paper=papers[pid]
        try:parsed=json.loads(row['raw_response']['content'])
        except Exception as exc:failed.append({'paper_id':pid,'task_kind':'ALL','error':str(exc)});continue
        for field,kind in (('summary','summarize'),('comparison','compare')):
            if parsed.get(field) is None:failed.append({'paper_id':pid,'task_kind':kind,'error':'NULL_TASK_NOT_FABRICATED'});continue
            task=copy.deepcopy(parsed[field]);original_quotes=[p['quote'] for p in task['keypoints']];repair='NONE'
            try:
                try:public,gold=validate_reference(paper,task,kind)
                except ValueError:
                    for point in task['keypoints']:point['quote']=source_paragraph(paper['text'],point['quote'])
                    # Identical paragraphs are one source unit; preserve dropped facets as evidence notes.
                    seen=set();unique=[];duplicates=[]
                    for point in task['keypoints']:
                        if point['quote'] in seen:duplicates.append(point)
                        else:unique.append(point);seen.add(point['quote'])
                    task['keypoints']=unique
                    public,gold=validate_reference(paper,task,kind);repair='FULL_SOURCE_PARAGRAPH_NO_WORD_REWRITE'
                    gold['deduplicated_same_paragraph_facets']=duplicates
                qid=public['query_id'].replace('H_TASK_','D_TASK_',1);public['query_id']=gold['query_id']=qid
                public['dataset_role']=gold['dataset_role']='DEVELOPMENT_SYNTHETIC'
                gold.update(source_representation=repair,original_teacher_quotes=original_quotes,teacher_record_hash=row['record_hash'])
                valid.append({'public':public,'gold':gold})
            except Exception as exc:failed.append({'paper_id':pid,'task_kind':kind,'error':str(exc),
                'original_teacher_quotes':original_quotes,'teacher_record_hash':row['record_hash']})
    outputs=[]
    for path,rows in ((run/'S4_DEVELOPMENT_TASK_QUESTIONS_v2.jsonl',[v['public'] for v in valid]),
        (run/'S4_DEVELOPMENT_TASK_GOLD_v2.jsonl',[v['gold'] for v in valid]),
        (run/'S4_DEVELOPMENT_TASK_COMPATIBILITY_FAILURES.jsonl',failed)):
        path.write_text(''.join(canonical_json(row)+'\n' for row in rows),encoding='utf8');outputs.append(artifact(path))
    result={'status':'COMPATIBILITY_PROJECTION_COMPLETE','runner':artifact(__file__),'original_teacher':artifact(run/'S4_DEVELOPMENT_TASK_TEACHER.jsonl'),
        'outputs':outputs,'valid_task_n':len(valid),'task_counts':{k:sum(v['public']['task_kind']==k for v in valid) for k in ('compare','summarize')},
        'retained_failed_task_n':len(failed),'new_generation_requests':0,'human_review_status':'NOT_REVIEWED',
        'bounded_policy':'one full-source paragraph compatibility branch; never rewrite teacher answer/entities; retain original failures; no teacher retries',
        'holdout_gold_read':False,'word_or_punctuation_rewrite':False}
    atomic_json(receipt,result);print(canonical_json({k:result[k] for k in ('status','valid_task_n','task_counts','retained_failed_task_n')}));return 0


if __name__=='__main__':raise SystemExit(main())
