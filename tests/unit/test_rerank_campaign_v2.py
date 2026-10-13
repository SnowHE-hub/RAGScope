"""Offline fixtures exercise R3 accounting; no true quality results are claimed."""
import copy
import json
import pytest

from ragscope.experiments.rerank_campaign import restore_candidates, rerank_record, verified_source, frozen_jobs
from ragscope.experiments.ledger import RecordStore
from ragscope.indexing.identity import signature, artifact, atomic_json, file_hash


def material(n=55):
    chunks=[]
    for i in range(n):
        text=f"fixture evidence number {i}"
        start=i*100
        chunks.append({"chunk_id":f"c{i}","doc_id":"d","text":text,"canonical_start":start,
                       "canonical_end":start+len(text),"unit_spans":[{"unit_id":f"u{i}","doc_id":"d","start":start,"end":start+len(text)}]})
    raw=[{k:c[k] for k in ('chunk_id','doc_id','canonical_start','canonical_end','unit_spans')}|
         {'rank':i+1,'score':1/(i+1)} for i,c in enumerate(chunks[:50])]
    query={"query_id":"q","question":"Which fixture?","scope":{"kind":"paper","doc_ids":["d"]},
           "split":"test","doc_id":"d","answerability":"answerable","evidence_reference_sets":[["u49"]]}
    row={**query,"raw_retrieval":raw,"record_id":"q","record_hash":"fixturehash","embedding_identity":{"fixture":True},
         "index_signature":"fixtureindex","timings":{"query_encoding_ms":2.,"retrieval_ms":3.}}
    units={f'u{i}':{"unit_id":f'u{i}',"doc_id":"d","canonical_start":c['canonical_start'],"canonical_end":c['canonical_end'],"text":c['text']}
           for i,c in enumerate(chunks)}
    return chunks,row,query,units


class FixtureReranker:
    identity={"offline_fixture":True}
    last_window_records=[]
    def rerank(self,question,candidates,top_k=10):
        self.last_window_records=[{"chunk_id":c['chunk_id'],"start":0,"end":len(c['text']),"pair_tokens":20,"score":i}
                                  for i,c in enumerate(candidates)]
        return [dict(c,rerank_score=float(i),rerank_windows=[self.last_window_records[i]])
                for i,c in reversed(list(enumerate(candidates)))][:top_k]


def test_restore_requires_all50_not_old_top10():
    chunks,row,_,_=material()
    restored,n=restore_candidates(row,chunks)
    assert len(restored)==50 and n==55 and restored[49]['text']==chunks[49]['text']
    row['raw_retrieval']=row['raw_retrieval'][:10]
    with pytest.raises(ValueError,match='full Hybrid50'):restore_candidates(row,chunks)


def test_short_scope_is_actual_available_count_without_padding():
    chunks,row,_,_=material(7)
    restored,n=restore_candidates(row,chunks)
    assert len(restored)==n==7


@pytest.mark.parametrize('mutation',['rank','coordinates','duplicate','scope','nan'])
def test_candidate_integrity_fail_closed(mutation):
    chunks,row,_,_=material()
    if mutation=='rank':row['raw_retrieval'][0]['rank']=3
    if mutation=='coordinates':row['raw_retrieval'][0]['canonical_end']+=1
    if mutation=='duplicate':row['raw_retrieval'][1]=copy.deepcopy(row['raw_retrieval'][0])
    if mutation=='scope':row['scope']['doc_ids']=['another_document']
    if mutation=='nan':row['raw_retrieval'][0]['score']=float('nan')
    with pytest.raises(ValueError):restore_candidates(row,chunks)


def test_rerank49_moves_to_selected_first_and_metrics_use_selected():
    chunks,row,query,units=material()
    result=rerank_record(row,query,chunks,[],units,FixtureReranker(),'none')
    assert result['raw_retrieval'][0]['chunk_id']=='c0'
    assert result['selected_retrieval'][0]['chunk_id']=='c49'
    assert len(result['selected_retrieval'])==10 and len(result['rerank_windows'])==50
    assert result['scores']['raw_top10']['full_reference_coverage']==1
    assert result['scores']['context']['full_reference_coverage']==1
    assert not result['real_model']
    assert result['timings']['total_ms']==pytest.approx(5+result['timings']['materialization_ms'])
    assert 'not online' in result['latency_interpretation']


def test_failure_stays_in_row_and_zero_applicable_metrics():
    chunks,row,query,units=material()
    class Failing(FixtureReranker):
        def rerank(self,*args,**kwargs):raise RuntimeError('fixture failure')
    result=rerank_record(row,query,chunks,[],units,Failing(),'none')
    assert result['error']['type']=='RuntimeError'
    assert len(result['raw_retrieval'])==50 and result['selected_retrieval']==[]
    assert result['scores']['continuous_coverage']==0


def test_global_scope_does_not_use_gold_document_to_filter():
    chunks,row,query,units=material()
    chunks[49]['doc_id']='outside_gold'
    chunks[49]['unit_spans'][0]['doc_id']='outside_gold'
    row['raw_retrieval'][49]['doc_id']='outside_gold'
    row['scope']={'kind':'global'}
    restored,_=restore_candidates(row,chunks)
    assert restored[49]['doc_id']=='outside_gold'


def test_source_config_and_hash_validation(tmp_path):
    chunks,row,query,_=material()
    config={'method':'hybrid','group':{'id':'STRUCT'},'expansion':'none',
            'retrieval':{'candidate_depth':50,'final_k':10,'context':{'max_tokens':2048}},'index_identity':{'fixture':True}}
    row.update(real_model=True,index_signature=signature(config['index_identity']),error=None)
    path=tmp_path/'source.jsonl'
    store=RecordStore(path,config)
    row.pop('record_hash')
    store.add('q',row)
    job={'source':path,'queries':[query],'group_id':'STRUCT','expansion':'none'}
    assert len(verified_source(job)[1])==1
    job['queries'][0]['question']='changed question'
    with pytest.raises(ValueError,match='frozen input'):verified_source(job)


def test_frozen_jobs_reuse_parent_route_ids_and_verify_manifest(tmp_path):
    corpus=tmp_path/'corpora/demo'; corpus.mkdir(parents=True)
    files={}
    populations={}
    for split,n in [('train',230),('dev',100),('test',200),('global_test',100)]:
        rows=[{'query_id':f'{split}-{i:03d}'} for i in range(n)]
        path=corpus/f'queries_{split}.jsonl'
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
        files[path.name]={'sha256':file_hash(path)}
        populations[split]=[r['query_id'] for r in rows]
    atomic_json(corpus/'SPLIT_MANIFEST.json',{'files':files})
    run=tmp_path/'runs/demo_retrieval'; run.mkdir(parents=True)
    selection=run/'CANDIDATE_SELECTION.json'
    atomic_json(selection,{'split':'dev','method':'hybrid','selected':{'structured':'STRUCT'}})
    atomic_json(run/'CONTEXT_SELECTION.json',{'split':'dev','candidate_selection_sha256':file_hash(selection),'selected':{'STRUCT':'none'}})
    route={'query_ids':{'train':list(reversed(populations['train']))[:200],'dev':list(reversed(populations['dev']))[:60]},
           'source':artifact(corpus/'SPLIT_MANIFEST.json'),'seed':42}
    route['manifest_hash']=signature(route)
    routepath=tmp_path/'runs/demo/ROUTING_QUERY_MANIFEST.json';atomic_json(routepath,route)
    jobs=frozen_jobs(tmp_path,'demo')
    assert [len(job['queries']) for job in jobs]==[100,200,100,200,60]
    assert [q['query_id'] for q in jobs[3]['queries']]==route['query_ids']['train']
    assert jobs[4]['destination'].name=='ROUTE_CONTEXT_DEV_STRUCT_hybrid_rerank_none.jsonl'
    route['query_ids']['train'][0]='altered'
    atomic_json(routepath,route)
    with pytest.raises(ValueError,match='hash-verified'):frozen_jobs(tmp_path,'demo')


def test_local_reranker_manifest_identity_and_path_enforced(tmp_path,monkeypatch):
    from ragscope.retrieval import verified_reranker as loader
    folder=tmp_path/'models/bge-reranker-base'/loader.REVISION
    folder.mkdir(parents=True)
    names=('config.json','model.safetensors','sentencepiece.bpe.model','special_tokens_map.json','tokenizer.json','tokenizer_config.json')
    items=[]
    for name in names:
        path=folder/name;path.write_text('offline fixture '+name,encoding='utf-8');items.append(artifact(path))
    monkeypatch.setattr(loader,'WEIGHT_SHA256',file_hash(folder/'model.safetensors'))
    info={'model':'BAAI/bge-reranker-base','revision':loader.REVISION,'status':'VERIFIED','files':items}
    manifest=folder/'download_manifest.json';atomic_json(manifest,info)
    assert loader.verified_checkpoint(tmp_path)[0]==folder
    outside=tmp_path/'tokenizer.json';outside.write_bytes((folder/'tokenizer.json').read_bytes())
    info['files']=[artifact(outside) if p['path']==str((folder/'tokenizer.json').resolve()) else p for p in items]
    atomic_json(manifest,info)
    with pytest.raises(ValueError,match='checkpoint'):loader.verified_checkpoint(tmp_path)


def test_unpermitted_loader_does_not_import_torch(tmp_path,monkeypatch):
    import sys
    from ragscope.retrieval import verified_reranker as loader
    monkeypatch.setattr(loader,'verified_checkpoint',lambda home:(tmp_path,{'offline_fixture':True}))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    before='torch' in sys.modules
    with pytest.raises(ValueError,match='CPU-only'):loader.load_verified_reranker(tmp_path)
    assert ('torch' in sys.modules)==before
