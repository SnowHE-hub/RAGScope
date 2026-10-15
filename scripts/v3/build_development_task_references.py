"""Six development papers, whole-source AI keypoints, exact-quote checks and retained failures."""
import argparse
import copy
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'scripts'));sys.path.insert(0,str(PROJECT/'scripts/v3'))
from freeze_generation_v2 import model_client_factory
from repair_task_reference_whitespace import source_span
from filelock import FileLock
from ragscope.experiments.ledger import RecordStore,read_records
from ragscope.experiments.runtime import configure,require_ram
from ragscope.indexing.identity import artifact,atomic_json,canonical_json,signature
from ragscope.v3.budget import Budget
from ragscope.v3.local_runtime import ManagedOllama,counted_chat
from ragscope.v3.splits import read_split,assert_inference_safe
from ragscope.v3.task_references import TEACHER_SYSTEM,schema,validate_reference


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    selected=sorted({q['scope']['doc_ids'][0] for q in read_split(run,'D_search',purpose='inference')},key=lambda pid:signature([42,'S4_whole_source_dev6',pid]))[:6]
    papers=[]
    with (runtime/'corpora/20261014_v2/all_documents.jsonl').open(encoding='utf8') as f:
        for line in f:
            doc=json.loads(line)
            if doc['doc_id'] in selected:papers.append({'source_id':doc['doc_id'],'title':doc['title'],'text':doc['text'],
                'sections':doc['sections'],'canonical_checksum':doc['canonical_checksum'],'dataset_role':'DEVELOPMENT_SYNTHETIC_D_SEARCH'})
    papers=sorted(papers,key=lambda paper:selected.index(paper['source_id']))
    if len(papers)!=6:raise ValueError('Six development papers not found')
    source_path=run/'S4_DEVELOPMENT_TASK_SOURCE_PAPERS.jsonl'
    if source_path.exists():
        if [json.loads(l) for l in source_path.read_text('utf8').splitlines()]!=papers:raise ValueError('Development source changed')
    else:source_path.write_text(''.join(canonical_json(p)+'\n' for p in papers),encoding='utf8')
    if a.prepare_only:print(canonical_json({'papers_n':6,'source':artifact(source_path),'new_model_requests':0,'holdout_gold_read':False}));return 0
    require_ram(2*1024**3)
    with FileLock(str(run/'writer.lock'),timeout=0):
        old=json.loads((runtime/'runs/20261014_v2_generation/EXPERIMENT_MANIFEST.json').read_text('utf8'))
        dev={json.loads(l)['query_id'] for l in (runtime/'corpora/20261014_v2/queries_dev.jsonl').read_text('utf8').splitlines()}
        original,pilot=model_client_factory(old['provenance']['spec']['models']['qwen_4b'],dev)
        config={'source':artifact(source_path),'runner':artifact(__file__),'teacher_protocol':artifact(PROJECT/'ragscope/v3/task_references.py'),
            'model_base_identity':original.identity,'pilot':pilot,'num_ctx':32768,'max_output_tokens':4096,
            'temperature':0,'source_scope':'WHOLE_PAPER_NEVER_TRUNCATED','human_review_status':'NOT_REVIEWED',
            'repair':'one deterministic whitespace-only branch; original invalid teacher quotes retained'}
        journal=run/'S4_DEVELOPMENT_TASK_TEACHER.jsonl';store=RecordStore(journal,config,max_records=6)
        if store.ids and not a.resume:raise ValueError('Existing teacher requires resume')
        pending=[p for p in papers if p['source_id'] not in store.ids];budget=Budget(run)
        for paper in pending:
            # Independent service lifecycle bounds the large whole-paper request.
            require_ram(2*1024**3)
            with ManagedOllama(run,runtime/'tools/ollama-0.34.4/ollama.exe',num_ctx=32768) as service:
                client=service.client(original,num_ctx=32768)
                messages=[{'role':'system','content':TEACHER_SYSTEM},{'role':'user','content':json.dumps(
                    {'paper_id':paper['source_id'],'title':paper['title'],'whole_paper':paper['text']},ensure_ascii=False)}]
                tokens=original.prompt_token_counter(messages)
                if tokens+4096>32768:
                    store.add(paper['source_id'],{'status':'BLOCKED_FULL_SOURCE_TOO_LONG','raw_response':None,'tasks':[],
                        'failed_branches':[{'kind':'whole_paper','error':'full input exceeds bound; never truncate'}],'actual_prompt_tokens':tokens,
                        'new_generation_requests':0,'human_review_status':'NOT_REVIEWED'});continue
                rid='S4:development_teacher:'+paper['source_id'];response={};error=None;tasks=[];failures=[]
                try:
                    response=counted_chat(run,service,client,request_id=rid,stage='S4_task_workflows_router',messages=messages,
                        max_tokens=4096,temperature=0,output_schema=schema(),purpose='development_whole_paper_keypoints')
                    parsed=json.loads(response['content'])
                    for field,kind in (('summary','summarize'),('comparison','compare')):
                        if parsed.get(field) is None:
                            failures.append({'kind':kind,'status':'NULL_TASK_NOT_FABRICATED'});continue
                        task=copy.deepcopy(parsed[field]);original_quotes=[p['quote'] for p in task['keypoints']]
                        try:
                            public,gold=validate_reference(paper,task,kind)
                            repair=False
                        except Exception as original_error:
                            failures.append({'kind':kind,'status':'ORIGINAL_VALIDATION_FAILED','error':str(original_error),'original_task':task})
                            try:
                                for point in task['keypoints']:point['quote']=source_span(paper['text'],point['quote'])
                                public,gold=validate_reference(paper,task,kind);repair=True
                            except Exception as exc:
                                failures.append({'kind':kind,'status':'WHITESPACE_COMPATIBILITY_FAILED','error':str(exc)});continue
                        qid=public['query_id'].replace('H_TASK_','D_TASK_',1);public['query_id']=gold['query_id']=qid
                        public['dataset_role']='DEVELOPMENT_SYNTHETIC';gold['dataset_role']='DEVELOPMENT_SYNTHETIC'
                        gold['source_format_repair']=repair;gold['original_teacher_quotes']=original_quotes
                        assert_inference_safe(public);tasks.append({'public':public,'gold':gold})
                except Exception as exc:
                    with budget.lock:
                        if rid not in budget._state()['reserved']:raise
                    error={'kind':type(exc).__name__,'message':str(exc)}
                store.add(paper['source_id'],{'status':'SOURCE_TASKS_PROCESSED' if error is None else 'FAILED_TEACHER_RETAINED',
                    'raw_response':response,'tasks':tasks,'failed_branches':failures,'actual_prompt_tokens':tokens,
                    'budget_ticket':rid,'error':error,'human_review_status':'NOT_REVIEWED'})
                print(canonical_json({'event':'DEV_TASK_TEACHER','papers':len(store.ids),'valid_tasks':len(tasks),'retained_failed_branches':len(failures)}),flush=True)
                if error:raise RuntimeError('Teacher backend/parse failure retained; stop bounded batch')
        tasks=[t for r in store.records for t in r['tasks']];outputs=[]
        for path,rows in ((run/'S4_DEVELOPMENT_TASK_QUESTIONS.jsonl',[t['public'] for t in tasks]),
            (run/'S4_DEVELOPMENT_TASK_GOLD.jsonl',[t['gold'] for t in tasks])):
            if path.exists():
                if [json.loads(l) for l in path.read_text('utf8').splitlines()]!=rows:raise ValueError('Do not alter established development task labels')
            else:path.write_text(''.join(canonical_json(row)+'\n' for row in rows),encoding='utf8')
            outputs.append(artifact(path))
        result={'status':'PASS_DEVELOPMENT_SOURCE_REFERENCES_WITH_FAILURES_RETAINED','paper_n':len(store.ids),'valid_task_n':len(tasks),
            'task_counts':{kind:sum(t['public']['task_kind']==kind for t in tasks) for kind in ('compare','summarize')},
            'retained_failed_branch_n':sum(len(r['failed_branches']) for r in store.records),'outputs':outputs,'teacher':artifact(journal),
            'human_review_status':'NOT_REVIEWED','operator_blindness':'NOT_OPERATOR_BLIND','not_official_qasper':True,
            'holdout_gold_read':False,'new_teacher_request_n':sum('budget_ticket' in r for r in store.records)}
        atomic_json(run/'S4_DEVELOPMENT_TASK_CONSTRUCTION.json',result);print(canonical_json(result));return 0


if __name__=='__main__':raise SystemExit(main())
