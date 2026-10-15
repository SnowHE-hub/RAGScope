"""Real loopback API gates and separate answer-cache test, using one development input."""
import argparse
import json
from pathlib import Path
import sys
import time
import requests
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ragscope.experiments.ledger import read_records
from ragscope.indexing.identity import artifact,atomic_json,canonical_json
from ragscope.v3.splits import read_split


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True)
    p.add_argument('--test-id',default='initial');a=p.parse_args()
    run=Path(a.campaign).resolve()
    if run.parent!=Path('D:/RAGScope_Runtime/runs').resolve():raise ValueError('Wrong campaign')
    if not a.test_id.replace('_','').isalnum():raise ValueError('Invalid immutable application test identity')
    target=run/('S6_APPLICATION_CACHE_VERIFICATION.json' if a.test_id=='initial' else 'S6_APPLICATION_CACHE_VERIFICATION_'+a.test_id+'.json')
    if target.exists():
        old=json.loads(target.read_text('utf8'))
        if old['runner']!=artifact(__file__):raise ValueError('Application test identity changed')
        print(canonical_json({'status':'REUSED_APPLICATION_TEST','original_status':old['status']}));return 0 if old['status'].startswith('PASS') else 3
    session=requests.Session();session.trust_env=False;deadline=time.monotonic()+45;state=None
    while time.monotonic()<deadline:
        try:
            response=session.get('http://127.0.0.1:11437/api/state',timeout=2);response.raise_for_status();state=response.json();break
        except requests.RequestException:time.sleep(.5)
    if state is None:raise RuntimeError('Explicit local service did not become ready')
    if state['campaign']!=run.name:raise ValueError('Service campaign mismatch')
    qid=read_records(run/'S6_ONLINE_COST.jsonl')[0]['question_id']
    q=next(q for q in read_split(run,'D_confirm',purpose='inference') if q['query_id']==qid)
    pid=q['scope']['doc_ids'][0];events=[]
    def post(name,payload):
        started=time.perf_counter();response=session.post('http://127.0.0.1:11437/api/answer',json=payload,timeout=180)
        result=response.json();events.append({'case':name,'http_status':response.status_code,
            'actual_http_seconds':time.perf_counter()-started,'response':result});return result
    try:
        empty=post('empty',{'question':'','doc_id':pid})
        missing=post('missing_object',{'question':q['question'],'doc_id':''})
        invalid=post('invalid_document',{'question':q['question'],'doc_id':'missing-local-paper'})
        none=post('no_evidence',{'question':'zzqvvxxzznotaword123456789','doc_id':pid})
        uncached=post('cache_miss_real_model',{'question':q['question'],'doc_id':pid,'condition':'B0','answer_cache':True})
        cached=post('cache_hit_application_only',{'question':q['question'],'doc_id':pid,'condition':'B0','answer_cache':True}) if uncached.get('action')=='ANSWER' else None
        passes={'empty':empty['action']=='EMPTY_QUESTION','missing':missing['action']=='CLARIFY_DOCUMENT',
            'invalid_document':invalid['action']=='INVALID_DOCUMENT','no_evidence':none['action']=='NO_EVIDENCE',
            'uncached_real_model':uncached.get('action')=='ANSWER' and uncached.get('answer_cache_hit') is False,
            'cache_hit':bool(cached and cached.get('answer_cache_hit') and cached['record_id']==uncached['record_id'] and cached['answer']==uncached['answer'])}
        result={'status':'PASS_REAL_APPLICATION_AND_SEPARATE_CACHE_TEST' if all(passes.values()) else 'PARTIAL_APPLICATION_TEST_RETAINED',
            'runner':artifact(__file__),'controller':artifact(PROJECT/'ragscope/v3/workbench.py'),
            'test_id':a.test_id,'checks':passes,'requests':events,'input_query_id':qid,
            'human_review_status':'NOT_REVIEWED','dataset_role':'D_confirm_application_supplement',
            'primary_online_measurement':False,'new_generation_requests_authority':'BUDGET.jsonl',
            'interpretation':'One real answer reused by cache; cache speed is application behavior, not uncached model improvement.'}
        atomic_json(target,result);print(canonical_json({k:result[k] for k in ('status','checks','input_query_id')}))
        return 0 if all(passes.values()) else 3
    finally:
        Path(state['stop_file']).write_text('Own workbench verification finished; release own service.',encoding='utf8')
        session.close()


if __name__=='__main__':raise SystemExit(main())
