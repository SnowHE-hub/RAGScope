"""Same calibrated relation semantics, with a bounded resident-input reservation."""
import time
from ragscope.experiments.runtime import require_ram
from ragscope.indexing.identity import signature


class ResidentCachedJudge:
    def __init__(self,nli,store,*,completion_gate,gpu=None):
        if nli.maximum!=512 or nli.batch_size>4 or nli.threshold!=.8:
            raise ValueError('Resident reservation requires the calibrated512-token/batch4/.8 stack')
        self.nli,self.store,self.completion_gate,self.gpu=nli,store,completion_gate,gpu
        self.index={r['record_id']:r['result'] for r in store.records};self.last_gpu_check=0

    def __call__(self,premise,hypothesis):
        if not premise['eligible'] or not hypothesis['eligible']:return {'verdict':'UNKNOWN','reason':'fragment_or_composite_not_calibrated'}
        if not self.completion_gate and any(p.get('reason','').startswith('mechanical_') for p in (premise,hypothesis)):
            return {'verdict':'UNKNOWN','reason':'qa_completion_calibration_failed'}
        rid=signature([premise['text'],hypothesis['text']])
        if rid not in self.index:
            # Model loading already required >=3GiB free. The bounded512/64-window
            # relation inputs reserve64MiB additional plus the existing1GiB margin.
            require_ram(64*1024**2)
            if self.gpu and time.monotonic()-self.last_gpu_check>=10:self.gpu.check();self.last_gpu_check=time.monotonic()
            result=self.nli.classify(premise['text'],hypothesis['text'])
            self.store.add(rid,{'premise_hash':signature(premise['text']),'hypothesis_hash':signature(hypothesis['text']),
                'premise_chars':len(premise['text']),'hypothesis_chars':len(hypothesis['text']),
                'result':result,'human_review_status':'NOT_REVIEWED'})
            self.index[rid]=result
        return self.index[rid]
