"""Build the real source-only document BM25 stage, preserving all old indices."""
import argparse
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from filelock import FileLock
import pandas as pd
from ragscope.experiments.runtime import configure,require_ram,require_disk
from ragscope.indexing.identity import artifact,atomic_json,canonical_json,signature
from ragscope.indexing.sparse_index import BM25Index


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    require_ram(2*1024**3);require_disk(20*1024**2)
    with FileLock(str(run/'document_bm25_build.lock'),timeout=0):
        source=runtime/'corpora/20261014_v2/all_documents.jsonl'
        config={'source':artifact(source),'runner':artifact(__file__),'bm25_implementation':artifact(PROJECT/'ragscope/indexing/sparse_index.py'),
            'retrieval_text':'original title + original abstract paragraphs + original section titles',
            'document_only_stage':True,'question_or_answer_gold_read':False}
        directory=runtime/'indexes'/run.name/'document_bm25'/signature(config);directory.mkdir(parents=True,exist_ok=True)
        receipt=run/'S3_DOCUMENT_BM25_INDEX.json'
        if receipt.exists():
            saved=json.loads(receipt.read_text('utf8'))
            if not a.resume or saved['config']!=config or any(artifact(x['path'])!=x for x in saved['outputs']):raise ValueError('Frozen document index changed')
            print(canonical_json({'status':'REUSED_ACTUAL_DOCUMENT_BM25','paper_n':saved['paper_n']}));return 0
        rows=[]
        with source.open(encoding='utf8') as f:
            for line in f:
                d=json.loads(line)
                abstract='\n'.join(u['text'] for u in d['evidence_units'] if u.get('section_id')=='abstract' and u['kind']=='paragraph')
                text='Title: '+d['title']+'\nAbstract: '+abstract+'\nSections: '+' | '.join(s['title'] for s in d['sections'])
                rows.append({'chunk_id':'document:'+d['doc_id'],'doc_id':d['doc_id'],'text':text})
        if len({x['doc_id'] for x in rows})!=len(rows):raise ValueError('Duplicate paper IDs')
        index_path=directory/'bm25.json'
        if index_path.exists():raise ValueError('Unreceipted index retained; independently verify before reuse')
        index=BM25Index(str(index_path));index.build(pd.DataFrame(rows))
        loaded=BM25Index.load(str(index_path))
        # Identity/lookup integrity check, not retrieval-quality evidence.
        if len(loaded.chunks)!=len(rows) or loaded.chunks!=index.chunks:raise ValueError('Index roundtrip changed document identity/text')
        result={'status':'PASS_REAL_DOCUMENT_BM25_BUILD','paper_n':len(rows),'config':config,'outputs':[artifact(index_path)],
            'new_llm_requests':0,'dense_document_stage':'NOT_BUILT','quality_status':'NOT_YET_EVALUATED',
            'generation_evidence':'document titles/abstracts used only for candidate selection; body evidence fetched separately'}
        atomic_json(receipt,result);print(canonical_json({'status':result['status'],'paper_n':len(rows),'index_path':str(index_path)}));return 0


if __name__=='__main__':raise SystemExit(main())
