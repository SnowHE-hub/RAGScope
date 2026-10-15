from ragscope.v3.resident_claim_judge import ResidentCachedJudge


def test_resident_guard_preserves_relation_cache_and_fragment_uncertainty(monkeypatch):
    sizes=[];monkeypatch.setattr('ragscope.v3.resident_claim_judge.require_ram',lambda size:sizes.append(size))
    class NLI:
        maximum=512;batch_size=4;threshold=.8
        def classify(self,premise,hypothesis):return {'verdict':'SUPPORTED_PROXY','encoder_tokens':7}
    class Store:
        records=[]
        def add(self,rid,row):self.records.append({'record_id':rid,**row})
    judge=ResidentCachedJudge(NLI(),Store(),completion_gate=True)
    premise={'text':'The method improves accuracy.','eligible':True};hypothesis={'text':'Accuracy improves.','eligible':True}
    assert judge(premise,hypothesis)['verdict']=='SUPPORTED_PROXY'
    assert judge(premise,hypothesis)['verdict']=='SUPPORTED_PROXY'
    assert sizes==[64*1024**2]
    assert judge(premise,{'text':'yes','eligible':False})['verdict']=='UNKNOWN'
