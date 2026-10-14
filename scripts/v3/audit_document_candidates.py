"""Real BM25 document candidates on exposed dev questions, with scope disclosed."""
import argparse
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ragscope.experiments.runtime import configure,require_ram
from ragscope.indexing.identity import artifact,atomic_json,canonical_json
from ragscope.indexing.sparse_index import BM25Index
from ragscope.v3.splits import read_split


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args()
    runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    require_ram(2*1024**3)
    receipt=json.loads((run/'S3_DOCUMENT_BM25_INDEX.json').read_text('utf8'))
    for x in receipt['outputs']:
        if artifact(x['path'])!=x:raise ValueError('Document index changed')
    index=BM25Index.load(receipt['outputs'][0]['path'])
    questions=read_split(run,'D_search',purpose='inference')
    wanted={d for q in questions for d in q['scope']['doc_ids']};titles={}
    with (runtime/'corpora/20261014_v2/all_documents.jsonl').open(encoding='utf8') as f:
        for line in f:
            doc=json.loads(line)
            if doc['doc_id'] in wanted:titles[doc['doc_id']]=doc['title']
    path=run/'S3_DOCUMENT_CANDIDATES_D_SEARCH.jsonl'
    if path.exists():raise ValueError('Preserve existing candidate audit')
    sums={mode:{k:0 for k in (1,5,10)} for mode in ('raw_unresolved','explicit_title_anchored')}
    with path.open('x',encoding='utf8',newline='\n') as f:
        for q in questions:
            if len(q['scope']['doc_ids'])!=1:raise ValueError('Development source does not have one visible selected paper')
            paper=q['scope']['doc_ids'][0]
            for mode in sums:
                question=q['question'] if mode=='raw_unresolved' else 'Paper title: '+titles[paper]+'\nQuestion: '+q['question']
                candidates=index.search(question,k=10)
                rank=next((i+1 for i,x in enumerate(candidates) if x['doc_id']==paper),None)
                for k in sums[mode]:sums[mode][k]+=int(rank is not None and rank<=k)
                row={'query_id':q['query_id'],'dataset_role':'EXPOSED_DIAGNOSTIC','condition':mode,
                    'query':question,'paper_ids_visible_in_original_user_scope':q['scope']['doc_ids'],
                    'title_source':'original source document selected by visible input scope' if mode!='raw_unresolved' else None,
                    'candidate_document_ids':[x['doc_id'] for x in candidates],'scores':[x['score'] for x in candidates],
                    'selected_paper_rank':rank,'human_review_status':'NOT_REVIEWED'}
                f.write(canonical_json(row)+'\n')
    result={'status':'PASS_REAL_DOCUMENT_CANDIDATE_AUDIT','questions_n':len(questions),'retrieval_requests':2*len(questions),
        'corpus_papers':receipt['paper_n'],'recall':{m:{str(k):v/len(questions) for k,v in values.items()} for m,values in sums.items()},
        'candidates':artifact(path),'index':artifact(run/'S3_DOCUMENT_BM25_INDEX.json'),'new_llm_requests':0,
        'human_review_status':'NOT_REVIEWED','limitations':['Anchor condition explicitly changes input; never compared as a fix to unanchored E8.',
            'Raw generic questions may have no uniquely intended paper; dataset-paper match is diagnostic only.',
            'PAPER_GIVEN deployment bypasses global document lookup; this measures the anchored global stage only.',
            'Document Dense and full chunk bypass not yet built; no end-to-end quality claim.']}
    atomic_json(run/'S3_DOCUMENT_CANDIDATES_D_SEARCH.json',result);print(canonical_json(result));return 0


if __name__=='__main__':raise SystemExit(main())
