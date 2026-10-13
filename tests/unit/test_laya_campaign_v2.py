"""Fixed-input Laya accounting tests; never import its actual SDK/model."""
import json
import pytest

from ragscope.routing.laya_campaign import freeze_visible_inputs,decisions_to_protocol
from ragscope.indexing.identity import atomic_json,artifact,file_hash,signature


def source(tmp_path):
    corpus=tmp_path/'corpora/demo';corpus.mkdir(parents=True)
    doc={'doc_id':'d','metadata':{'reference_n_tokens':900},'evidence_units':[]}
    files={}
    path=corpus/'documents.jsonl';path.write_text(json.dumps(doc)+'\n',encoding='utf-8')
    files[path.name]={'sha256':file_hash(path)}
    ids={}
    for split,n in [('train',200),('dev',60),('test',100)]:
        rows=[{'query_id':f'{split}{i}','question':f'Question {i}?','split':split,'doc_id':'d',
               'scope':{'kind':'document','doc_ids':['d']},'answerability':'answerable','evidence_reference_sets':[['secret_gold']],
               'answers':['secret answer'],'metadata':{'oracle_path':'secret','answer_type':'hidden'}} for i in range(n)]
        path=corpus/f'queries_{split}.jsonl'
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
        files[path.name]={'sha256':file_hash(path)}
        ids[split]=[r['query_id'] for r in rows]
    atomic_json(corpus/'SPLIT_MANIFEST.json',{'files':files})
    route={'source':artifact(corpus/'SPLIT_MANIFEST.json'),'query_ids':ids}
    route['manifest_hash']=signature(route)
    atomic_json(tmp_path/'runs/demo/ROUTING_QUERY_MANIFEST.json',route)
    return corpus


def test_frozen_visible_inputs_strip_all_gold_and_preserve360(tmp_path):
    source(tmp_path)
    result=freeze_visible_inputs(tmp_path,'demo')
    assert result['expected_decisions']==1080
    assert [len(result['populations'][s]['visible_queries']) for s in ['train','dev','test']]==[200,60,100]
    query=result['populations']['test']['visible_queries'][0]
    assert query['metadata']=={'document_length_tokens':900,'document_type':'paper','language':'en'}
    assert set(query)=={'query_id','question','split','scope','metadata'}
    assert 'secret' not in json.dumps(result['populations'])
    path=tmp_path/'runs/demo_laya/INPUTS.json';before=path.read_bytes()
    assert freeze_visible_inputs(tmp_path,'demo')==result and path.read_bytes()==before


def test_changed_frozen_corpus_fails_before_any_model(tmp_path):
    corpus=source(tmp_path)
    (corpus/'queries_test.jsonl').write_text('{}\n',encoding='utf-8')
    with pytest.raises(ValueError,match='corpus input'):freeze_visible_inputs(tmp_path,'demo')


def test_probability_binding_requires_real_complete_population(tmp_path):
    source(tmp_path)
    inputs=freeze_visible_inputs(tmp_path,'demo')
    directory=tmp_path/'runs/demo_laya'
    atomic_json(directory/'SUMMARY.json',{'status':'PENDING','input_hash':inputs['input_hash']})
    with pytest.raises(ValueError,match='incomplete'):decisions_to_protocol(directory,{},tmp_path/'output')
