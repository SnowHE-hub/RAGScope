"""Actual40 sealed synthetic scope requests using real uncached query encoding/index search."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import psutil
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from filelock import FileLock
from ragscope.experiments.ledger import RecordStore
from ragscope.experiments.runtime import configure,require_ram
from ragscope.indexing.identity import artifact,atomic_json,canonical_json
from ragscope.v3.auxiliary_public import read_auxiliary_public
from ragscope.v3.hierarchical_runtime import load_hierarchy
from ragscope.v3.local_runtime import ManagedOllama
from ragscope.v3.online_runtime import UncachedOnlineEmbedding
from ragscope.v3.records import canonical_payload


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    questions=read_auxiliary_public(run,'H_scope');require_ram(2*1024**3);engine=load_hierarchy(run)
    config={'runner':artifact(__file__),'scope_questions':artifact(run/'splits/H_scope.questions.jsonl'),
        'query_ids':[q['query_id'] for q in questions],'expected_n':40,'metrics':'explicit user scope invariants, no factual answer-quality claim',
        'embedding_cache':False,'human_review_status':'NOT_REVIEWED','authorship':'SYNTHETIC_RULE_NOT_OFFICIAL_QASPER'}
    with FileLock(str(run/'writer.lock'),timeout=0):
        store=RecordStore(run/'S6_H_SCOPE_REQUESTS.jsonl',config,max_records=40)
        if store.ids and not a.resume:raise ValueError('Existing scope records need resume')
        pending=[q for q in questions if q['query_id'] not in store.ids]
        if pending:
            with ManagedOllama(run,runtime/'tools/ollama-0.34.4/ollama.exe',num_ctx=4096) as service:
                encoder=UncachedOnlineEmbedding(run,service)
                for q in pending:
                    if psutil.virtual_memory().available<1024**3:raise RuntimeError('Frozen scope cohort emergency floor')
                    start=time.perf_counter();vector,_=encoder.query(q['question'],'S6:scope:'+q['query_id'])
                    result=engine.search(q['question'],scope=q['scope'],query_vector=vector,embedding_identity=encoder.identity)
                    specified=q['scope']['kind']=='document'
                    passed=(result['action']=='RETRIEVE' and not result['document_lookup_performed'] and
                            all(c['doc_id'] in q['scope']['doc_ids'] for c in result['candidates'])) if specified else (
                            result['action']=='CLARIFY_DOCUMENT' and not result['candidates'])
                    store.add(q['query_id'],canonical_payload({'question_id':q['query_id'],'dataset_role':'H_scope',
                        'exposure_status':'FROZEN_SYNTHETIC_SCOPE_ONLY','user_scope':q['scope'],
                        'task_type_source':'SYNTHETIC_RULE','model_identity':encoder.identity,
                        'retriever_candidates':result['candidates'],'scope_result':result,'scope_invariant_pass':passed,
                        'success':True,'wall_time_ms':(time.perf_counter()-start)*1000,'new_generation_requests':0,
                        'gold_read':False,'authorship':'SYNTHETIC_RULE_NOT_OFFICIAL_QASPER'},campaign_id=run.name,task_id='S6_scope'))
        rows=store.records
        result={'status':'PASS_40_ACTUAL_SCOPE_INVARIANTS' if len(rows)==40 and all(r['scope_invariant_pass'] for r in rows) else 'FAIL_SCOPE_INVARIANTS',
            'request_n':len(rows),'specified_n':sum(r['user_scope']['kind']=='document' for r in rows),
            'clarification_n':sum(r['scope_result']['action']=='CLARIFY_DOCUMENT' for r in rows),
            'source_papers_n':20,'new_generation_requests':0,'actual_uncached_embedding_requests':len(rows),
            'records':artifact(store.path),'human_review_status':'NOT_REVIEWED','holdout_gold_read':False,
            'interpretation':'Scope policy on40 synthetic cases; not QA factual correctness, document identification accuracy, or unseen-candidate quality.'}
        atomic_json(run/'S6_H_SCOPE_COMPLETION.json',result);print(canonical_json(result));return 0 if result['status'].startswith('PASS') else 2


if __name__=='__main__':raise SystemExit(main())
