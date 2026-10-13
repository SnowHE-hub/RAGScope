"""Offline provenance and full-matrix freeze tests; no services/model calls."""
import copy
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from ragscope.experiments.ledger import RecordStore, append_jsonl
from ragscope.generation.context import pack_context
from ragscope.indexing.identity import file_hash, signature, atomic_json

spec = importlib.util.spec_from_file_location('freeze_v2',Path(__file__).parents[2]/'scripts/freeze_generation_v2.py')
freeze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freeze)


def query(qid='q', split='test'):
    return {'query_id':qid,'question':'Question?','doc_id':'d','scope':{'kind':'document','doc_ids':['d']},
            'split':split,'answerability':'answerable','answer_refs':['source'],'evidence_reference_sets':[['u']]}


def unit(uid='u', text='source', start=0):
    return {'unit_id':uid,'doc_id':'d','canonical_start':start,'canonical_end':start+len(text),'text':text,'kind':'paragraph'}


def write_corpus(path, squad=False):
    path.mkdir(parents=True)
    documents=[{'doc_id':'d','evidence_units':[unit()]}]
    payload={'documents.jsonl':documents,'queries_test.jsonl':[query()]}
    if not squad:
        payload.update({'queries_train.jsonl':[query('train','train')],'queries_dev.jsonl':[query('dev','dev')]})
    for name, rows in payload.items():
        (path/name).write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
    manifest={'files':{name:{'sha256':file_hash(path/name)} for name in payload}}
    if squad: manifest['selected_query_ids']=['q']
    else: manifest['query_ids']={'test':['q'],'train':['train'],'dev':['dev']}
    atomic_json(path/'SPLIT_MANIFEST.json',manifest)


def test_corpus_supports_qasper_and_separate_squad_manifest(tmp_path):
    write_corpus(tmp_path/'qasper'); write_corpus(tmp_path/'squad',True)
    for directory in ['qasper','squad']:
        queries, units, artifact=freeze.verified_corpus(tmp_path/directory)
        assert queries['test'][0]['query_id']=='q' and units['u']['text']=='source'
        assert artifact['sha256']
    (tmp_path/'qasper'/'queries_test.jsonl').write_text('{}\n')
    with pytest.raises(ValueError,match='changed'):
        freeze.verified_corpus(tmp_path/'qasper')


def test_document_length_uses_canonical_metadata_and_fixed_boundaries(tmp_path):
    rows=[{'doc_id':str(n),'metadata':{'reference_n_tokens':n,'reference_tokenizer':'cl100k_base'}} for n in [512,513,4096,4097]]
    (tmp_path/'documents.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    lengths=freeze.document_lengths(tmp_path)
    assert [lengths[str(n)]['length_group'] for n in [512,513,4096,4097]]==['short','medium','medium','long']
    assert lengths['4097']['reference_n_tokens']==4097
    rows[0]['metadata']['length_group']='long'
    (tmp_path/'documents.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError,match='disagrees'):freeze.document_lengths(tmp_path)


def test_readiness_lists_every_mandatory_group_when_models_pending(tmp_path):
    settings=freeze.default_spec(tmp_path,'run')
    write_corpus(Path(settings['corpus_dir'])); write_corpus(Path(settings['corpus_dir'])/'squad_e4',True)
    result=freeze.freeze_all(settings)
    assert result['status']=='PENDING'
    counts={exp:sum(g['experiment_id']==exp for g in result['groups']) for exp in ['E2','E3','E4','E5','E6','E7','ROUTE_TRAIN','ROUTE_DEV']}
    assert counts=={'E2':4,'E3':2,'E4':1,'E5':2,'E6':2,'E7':3,'ROUTE_TRAIN':3,'ROUTE_DEV':3}
    assert not (Path(settings['output_dir'])/'EXPERIMENT_MANIFEST.json').exists()
    assert result['routing_query_ids']=={'train':['train'],'dev':['dev']}


def test_explicit_block_requires_existing_evidence(tmp_path):
    assert freeze.explicit_block({'reason':'not sufficient'}) is None
    with pytest.raises(ValueError,match='exist'):
        freeze.explicit_block({'reason':'offline fixture','evidence':[str(tmp_path/'absent')]})
    evidence=tmp_path/'guard.json'; evidence.write_text('{}')
    value=freeze.explicit_block({'reason':'offline fixture','evidence':[str(evidence)]})
    assert value['status']=='BLOCKED' and value['evidence'][0]['sha256']==file_hash(evidence)


def test_context_loader_checks_record_config_and_query_scope(tmp_path):
    bundle=asdict(pack_context([freeze.unit_piece(unit(),'c:')],token_counter=len))
    path=tmp_path/'rows.jsonl'; config={'identity':'test'}
    store=RecordStore(path,config)
    q=query()
    store.add('q',{**q,'context_bundle':bundle,'raw_retrieval':[],'error':None})
    rows,audit=freeze.verified_context_rows(path,[q])
    assert rows[0]['query']==q and audit['records']['sha256']
    with pytest.raises(ValueError,match='scope'):
        freeze.verified_context_rows(path,[{**q,'scope':{'kind':'global'}}])
    with pytest.raises(ValueError,match='omit'):
        freeze.verified_context_rows(path,[query('other')])


def test_common_budget_repacked_once_and_identical_for_models():
    q=query(); source={'query':q,'context_bundle':asdict(pack_context([freeze.unit_piece(unit(),'c:')],token_counter=len))}
    first=SimpleNamespace(model='a',num_ctx=600,prompt_token_counter=lambda messages:len(messages[-1]['content']))
    second=SimpleNamespace(model='b',num_ctx=601,prompt_token_counter=lambda messages:len(messages[-1]['content']))
    # Fixed prompt fits, but character-count test counter rejects evidence.
    base=first.prompt_token_counter(freeze.messages(q['question'],''))
    first.num_ctx=base+512; second.num_ctx=base+512
    result=freeze.shared_context(source,[first,second])
    assert not result['context_bundle']['pieces'] and result['context_bundle']['truncations']
    assert result['full_prompt_tokens_by_model']['a']==result['full_prompt_tokens_by_model']['b']
    first.num_ctx=1
    with pytest.raises(ValueError,match='before evidence'):
        freeze.shared_context(source,[first])


def test_pilot_requires_raw_twenty_distinct_dev_and_backend_count_agreement(tmp_path):
    counter=SimpleNamespace(identity={'counter':'fixture'})
    class Counter:
        identity={'counter':'fixture'}
        def __call__(self,messages): return 10
    counter=Counter(); client=SimpleNamespace(identity={'model':'unit-fixture'})
    summary=tmp_path/'summary.json'; rows=tmp_path/'rows.jsonl'
    atomic_json(summary,{'status':'PASS','expected_n':20,'unique_queries':20,'model_identity':client.identity,'counter_identity':counter.identity})
    dev_ids={str(i) for i in range(20)}
    bundle=asdict(pack_context([],token_counter=len))
    prompt=freeze.messages('Question?', '')
    for qid in dev_ids:
        append_jsonl(rows,{'query_id':qid,'split':'dev','purpose':'pilot','ticket':{'call_number':1},
            'result':{'real_model':True,'model_identity':client.identity,'prompt':prompt,'question':'Question?','context_bundle':bundle,
                      'prompt_version':freeze.PROMPT_VERSION,'citation_constraint':{'applied':True},
                      'output_schema':freeze.visible_output_schema(bundle['pieces']),
                      'error':None,'backend_metadata':{'prompt_eval_count':10}}})
    assert freeze.verify_pilot(client,counter,summary,rows,dev_ids)['verified_n']==20
    previous=next(row for row in freeze.read_records(rows) if row['query_id']=='0')
    truncated=copy.deepcopy(previous)
    truncated['result']['error']={'kind':'parse_error','message':'Actual completion reached the output limit'}
    truncated['result']['backend_metadata']['done_reason']='length'
    append_jsonl(rows,truncated)
    audit=freeze.verify_pilot(client,counter,summary,rows,dev_ids)
    assert audit['verified_n']==20 and audit['valid_output_n']==19 and len(audit['output_failures'])==1
    transport=copy.deepcopy(truncated);transport['result']['error']['kind']='timeout'
    append_jsonl(rows,transport)
    with pytest.raises(ValueError,match='Pilot failure'):freeze.verify_pilot(client,counter,summary,rows,dev_ids)
    bad={'query_id':'0','split':'dev','purpose':'pilot','ticket':{'call_number':2},
         'result':{'real_model':True,'model_identity':client.identity,'prompt':[], 'error':None,
                   'backend_metadata':{'prompt_eval_count':11}}}
    append_jsonl(rows,bad)
    with pytest.raises(ValueError,match='mismatch'):
        freeze.verify_pilot(client,counter,summary,rows,dev_ids)


def test_position_fixtures_exclude_all_reference_units_and_preserve_sources():
    units={u['unit_id']:u for u in [unit(text='source.'),unit('alternate','alternative source.',10),unit('n1',('alpha noise '*15).strip()+'.',100),unit('n2',('beta noise '*15).strip()+'.',400)]}
    q=query();q['evidence_reference_sets'].append(['alternate'])
    client=SimpleNamespace(num_ctx=4096,prompt_token_counter=lambda messages:10)
    report=freeze.make_position_fixtures([q],units,[client],n=40)
    assert report['actual_n']==1 and report['target_n']==40
    fixture=report['fixtures'][0]
    assert {p['unit_spans'][0]['unit_id'] for p in fixture['noise_pieces']}=={'n1','n2'}
    assert fixture['evidence_pieces'][0]['text']=='source.'


def test_full_blocked_matrix_freezes_explicit_denominators_without_fake_tasks(tmp_path):
    settings=freeze.default_spec(tmp_path,'run')
    atomic_json(Path(settings['service_receipt']),{'environment':{'LLAMA_ARG_CACHE_RAM':'0'}})
    write_corpus(Path(settings['corpus_dir'])); write_corpus(Path(settings['corpus_dir'])/'squad_e4',True)
    retrieval=Path(settings['retrieval_run']); retrieval.mkdir(parents=True)
    atomic_json(retrieval/'CANDIDATE_SELECTION.json',{'split':'dev','method':'hybrid','selected':{'simple':'F512_O64','structured':'R512'}})
    atomic_json(retrieval/'CONTEXT_SELECTION.json',{'split':'dev','selected':{'F512_O64':'none','R512':'neighbor'},
        'candidate_selection_sha256':file_hash(retrieval/'CANDIDATE_SELECTION.json')})
    evidence=tmp_path/'real_guard_fixture.json'; evidence.write_text('{"fixture_only":true}')
    settings['model_blocks']={mid:{'reason':'offline test unavailable fixture','evidence':[str(evidence)]}
                             for mid in settings['main_models']+[settings['heterogeneous_model']]}
    result=freeze.freeze_all(settings)
    assert result['status']=='FROZEN' and not result['pending']
    manifest=freeze.read_json(Path(settings['output_dir'])/'EXPERIMENT_MANIFEST.json')
    assert not manifest['tasks'] and all(group['status']=='BLOCKED' for group in manifest['groups'])
    assert len([g for g in manifest['groups'] if g['experiment_id']=='E2'])==12


def test_full_ready_matrix_frozen_once_with_common_contexts(tmp_path):
    settings=freeze.default_spec(tmp_path,'run')
    atomic_json(Path(settings['service_receipt']),{'environment':{'LLAMA_ARG_CACHE_RAM':'0'}})
    corpus=Path(settings['corpus_dir']); write_corpus(corpus); write_corpus(corpus/'squad_e4',True)
    units=[unit(text='source.'),unit('n1',('alpha noise '*15).strip()+'.',100),unit('n2',('beta noise '*15).strip()+'.',400)]
    (corpus/'documents.jsonl').write_text(json.dumps({'doc_id':'d','evidence_units':units})+'\n')
    cm=freeze.read_json(corpus/'SPLIT_MANIFEST.json'); cm['files']['documents.jsonl']['sha256']=file_hash(corpus/'documents.jsonl'); atomic_json(corpus/'SPLIT_MANIFEST.json',cm)
    # Both tiny fixture corpora share a source ID; make that source identical.
    (corpus/'squad_e4'/'documents.jsonl').write_text(json.dumps({'doc_id':'d','evidence_units':[units[0]]})+'\n')
    sm=freeze.read_json(corpus/'squad_e4'/'SPLIT_MANIFEST.json'); sm['files']['documents.jsonl']['sha256']=file_hash(corpus/'squad_e4'/'documents.jsonl'); atomic_json(corpus/'squad_e4'/'SPLIT_MANIFEST.json',sm)
    retrieval=Path(settings['retrieval_run']); retrieval.mkdir(parents=True)
    atomic_json(retrieval/'CANDIDATE_SELECTION.json',{'split':'dev','method':'hybrid','selected':{'simple':'F512_O64','structured':'R512'}})
    atomic_json(retrieval/'CONTEXT_SELECTION.json',{'split':'dev','selected':{'F512_O64':'none','R512':'neighbor'},
        'candidate_selection_sha256':file_hash(retrieval/'CANDIDATE_SELECTION.json')})
    settings['models']['lfm']={'source_manifest':'offline-lfm-fixture'}
    def factory(model_spec,dev_ids):
        identity={'model':model_spec['source_manifest'],'revision':'offline','weight_sha256':'offline-fixture','backend_version':'fixture'}
        return SimpleNamespace(identity=identity,model=identity['model'],num_ctx=4096,prompt_token_counter=lambda messages:10),{'offline_test_only':True}
    conditions={'E2':['R0','R1','R2','R3'],'E4':['hybrid'],'E5':['bm25','hybrid'],'E6':['simple','structured'],
                'ROUTE_TRAIN':['bm25','hybrid','hybrid_rerank'],'ROUTE_DEV':['bm25','hybrid','hybrid_rerank']}
    for exp,methods in conditions.items():
        for method in methods:
            split='train' if exp=='ROUTE_TRAIN' else 'dev' if exp=='ROUTE_DEV' else 'test'
            q=query(split if split!='test' else 'q',split)
            path=tmp_path/(exp+'_'+method+'.jsonl')
            store=RecordStore(path,{'experiment':exp,'method':method})
            store.add(q['query_id'],{**q,'context_bundle':asdict(pack_context([freeze.unit_piece(units[0],'c:')])),
                                    'raw_retrieval':[],'error':None})
            settings['context_sources'][exp+':'+method]=str(path)
    result=freeze.freeze_all(settings,client_factory=factory)
    assert result['status']=='FROZEN' and not result['pending']
    manifest=freeze.read_json(Path(settings['output_dir'])/'EXPERIMENT_MANIFEST.json')
    assert len(manifest['tasks'])==39
    e2=[task for task in manifest['tasks'] if task['experiment_id']=='E2' and task['method']=='R0']
    assert len({task['input_hash'] for task in e2})==1
    assert freeze.freeze_all(settings,client_factory=factory)['manifest_hash']==result['manifest_hash']
