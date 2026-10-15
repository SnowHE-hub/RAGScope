"""Real local API acceptance, separate from frozen quality experiments."""
import argparse
import json
from pathlib import Path
import sys
import time
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ragscope.experiments.runtime import configure
from ragscope.indexing.identity import artifact,atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True)
    args=parser.parse_args();configure(cpu_only=True);run=Path(args.run)
    import requests
    from ragscope.v3.splits import read_split
    from ragscope.v3.auxiliary_public import read_auxiliary_public
    from ragscope.experiments.ledger import read_records
    source=Path(json.loads((run/'FULL_DENSE_INDEX.json').read_text('utf8'))['source_campaign'])
    session=requests.Session();session.trust_env=False
    state=session.get('http://127.0.0.1:11437/api/state',timeout=30);state.raise_for_status();state=state.json()
    assert state['papers']==1585 and state['chunks']==19673 and state['index_status']=='COMPLETE'
    compact={r['doc_id'] for r in read_records(source/'S3_SCOPED_VECTOR_PAPERS.jsonl')}
    query=next(q for q in read_split(source,'H_QA') if q['scope']['doc_ids'][0] not in compact)
    summary=next(q for q in read_auxiliary_public(source,'H_tasks') if q['task_kind']=='summarize')
    cases=[('empty',{'question':'','doc_ids':[]},'EMPTY_QUESTION'),
           ('unknown_document',{'question':'What is the method?','doc_ids':['not-indexed']},'INVALID_DOCUMENT'),
           ('ambiguous_paper',{'question':'What is the method of this paper?','doc_ids':[]},'CLARIFY_DOCUMENT'),
           ('outside_old141',{'question':query['question'],'doc_ids':query['scope']['doc_ids']},'ANSWER'),
           ('global',{'question':'Explain how cross-lingual word embeddings are used for transfer learning.','doc_ids':[]},'ANSWER'),
           ('cross_paper',{'question':'Compare the objectives and methods of the two selected papers, clearly distinguishing them.','doc_ids':[query['scope']['doc_ids'][0],summary['scope']['doc_ids'][0]],'task':'comparison'},'ANSWER'),
           ('summary',{'question':summary['question'],'doc_ids':summary['scope']['doc_ids'],'task':'summary'},'ANSWER'),
           ('cache_fill',{'question':'how did they ask if a tweet was racist?','doc_ids':['qasper:1608.08738'],'answer_cache':True},'ANSWER'),
           ('cache_hit',{'question':'how did they ask if a tweet was racist?','doc_ids':['qasper:1608.08738'],'answer_cache':True},'ANSWER')]
    results=[]
    for name,payload,expected in cases:
        start=time.perf_counter();response=session.post('http://127.0.0.1:11437/api/answer',json=payload,timeout=180)
        response.raise_for_status();result=response.json();assert result['action']==expected,(name,result)
        if expected=='ANSWER':
            pieces=result['context_bundle']['pieces'];visible={p['chunk_id'] for p in pieces}
            assert set(result['citations'])<=visible
            if payload['doc_ids']:assert {p['doc_id'] for p in pieces}<=set(payload['doc_ids'])
            if name=='cross_paper':assert {p['doc_id'] for p in pieces}==set(payload['doc_ids'])
            if name=='cache_hit':assert result['answer_cache_hit'] is True
        results.append({'case':name,'expected':expected,'payload':payload,'result':result,'http_wall_s':time.perf_counter()-start})
        print(json.dumps({'case':name,'action':result['action'],'seconds':results[-1]['http_wall_s']}),flush=True)
    destination=run/'API_REAL_ACCEPTANCE.json'
    atomic_json(destination,{'status':'PASS','cases':results,'case_n':len(results),'state':state,
        'implementation':artifact(__file__),'quality_interpretation':'Application function acceptance, not additional frozen QA benchmark or human factual validation'})


if __name__=='__main__':main()
