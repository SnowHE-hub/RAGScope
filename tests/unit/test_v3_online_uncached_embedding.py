import numpy as np
from ragscope.v3.online_runtime import UncachedOnlineEmbedding


def test_each_live_query_posts_even_when_text_repeats(monkeypatch,tmp_path):
    calls=[]
    class Tokenizer:
        def encode(self,text):return [1,2]
    class Client:
        tokenizer=Tokenizer();identity={'fixture':'not a real experiment'}
        def __init__(self,*args):pass
    class Response:
        def raise_for_status(self):pass
        def json(self):return {'embeddings':[np.ones(1024).tolist()],'prompt_eval_count':2}
    class Session:
        def post(self,url,**kwargs):calls.append(kwargs['json']);return Response()
    class Service:
        base_url='http://fixture.invalid';session=Session()
    monkeypatch.setattr('ragscope.v3.online_runtime.PrivateEmbeddingClient',Client)
    encoder=UncachedOnlineEmbedding(tmp_path,Service())
    first,_=encoder.query('same question','1');second,_=encoder.query('same question','2')
    assert len(calls)==2 and all(c['truncate'] is False for c in calls)
    assert np.linalg.norm(first)==np.linalg.norm(second)==1
    assert len((tmp_path/'ONLINE_EMBEDDING_REQUESTS.jsonl').read_text().splitlines())==4
