"""Independent bounded ledger for non-generative query/window scorer pairs."""
from pathlib import Path
from filelock import FileLock
from ragscope.experiments.ledger import append_jsonl,read_records,now
from ragscope.indexing.identity import signature


class RerankBudget:
    def __init__(self,run):
        self.path=Path(run)/'RERANK_PAIR_BUDGET.jsonl'
        self.lock=FileLock(str(self.path.with_suffix('.lock')),timeout=0)

    def reserve(self,request_id,pairs,identity):
        if not isinstance(pairs,int) or isinstance(pairs,bool) or pairs<1 or not identity:raise ValueError('Invalid pair reservation')
        with self.lock:
            rows=read_records(self.path)
            if any(x['request_id']==request_id for x in rows):raise ValueError('Pair request already reserved; reconcile instead of repeat')
            if sum(x['pairs'] for x in rows)+pairs>200000:raise RuntimeError('Rerank pair budget exhausted')
            row={'request_id':request_id,'pairs':pairs,'identity':identity,'recorded_at':now(),'limit':200000}
            row['record_hash']=signature(row);append_jsonl(self.path,row)
            return row
