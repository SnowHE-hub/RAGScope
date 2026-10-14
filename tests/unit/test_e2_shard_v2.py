"""Offline core-shard contracts; no model, network or experiment calls."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import freeze_generation_v2 as full
import freeze_e2_v2 as e2
import freeze_e3_v2 as e3
from ragscope.experiments.generation_campaign import freeze_campaign,load_campaign,iter_record_sources,run_model,GenerationBudget
from ragscope.experiments.ledger import RecordStore
from ragscope.generation.context import pack_context
from ragscope.indexing.identity import atomic_json,file_hash,artifact


def fixture(tmp_path,n=200):
    spec=full.default_spec(tmp_path,'run');corpus=Path(spec['corpus_dir']);corpus.mkdir(parents=True)
    unit={'unit_id':'u','doc_id':'d','canonical_start':0,'canonical_end':7,'text':'source.','kind':'paragraph'}
    populations={split:[{'query_id':f'{split}{i}','question':f'Question{i}?','doc_id':'d','scope':'d',
        'split':split,'answerability':'answerable','answer_refs':['source'], 'evidence_reference_sets':[['u']]} for i in range(count)]
        for split,count in [('train',1),('dev',1),('test',n)]}
    (corpus/'documents.jsonl').write_text(json.dumps({'doc_id':'d','text':'source.','evidence_units':[unit],
        'metadata':{'reference_n_tokens':2,'reference_tokenizer':'cl100k_base'}})+'\n')
    for split,rows in populations.items():
        (corpus/f'queries_{split}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    atomic_json(corpus/'SPLIT_MANIFEST.json',{'files':{p.name:{'sha256':file_hash(p)} for p in corpus.glob('*.jsonl')},
        'query_ids':{split:[q['query_id'] for q in rows] for split,rows in populations.items()}})
    atomic_json(Path(spec['service_receipt']),{'environment':{'LLAMA_ARG_CACHE_RAM':'0'}})
    retrieval=Path(spec['retrieval_run']);retrieval.mkdir(parents=True)
    atomic_json(retrieval/'CANDIDATE_SELECTION.json',{'split':'dev','method':'hybrid','selected':{'simple':'F512_O0','structured':'R512'}})
    atomic_json(retrieval/'CONTEXT_SELECTION.json',{'split':'dev','selected':{'F512_O0':'neighbor','R512':'neighbor'},
        'candidate_selection_sha256':file_hash(retrieval/'CANDIDATE_SELECTION.json')})
    paths,_=e2.selected_sources(spec)
    for method,path in paths.items():
        store=RecordStore(path,{'method':method,'offline_fixture':True})
        for query in populations['test']:
            store.add(query['query_id'],{**query,'raw_retrieval':[],'selected_retrieval':[],
                'context_bundle':asdict(pack_context([full.unit_piece(unit,'chunk:')])),'error':None})
    return spec,populations,paths


def factory(spec,dev_ids):
    name=spec['source_manifest']
    identity={'model':name,'revision':'fixture','weight_sha256':'not-real-weights','backend_version':'fixture'}
    return SimpleNamespace(identity=identity,model=name,num_ctx=4096,
        prompt_token_counter=lambda messages:len(messages[-1]['content'])//4+10),{'offline_fixture':True}


def test_only_full_e2_sources_are_required_and_identical_tasks_are_frozen(tmp_path):
    spec,queries,paths=fixture(tmp_path)
    assert not (Path(spec['corpus_dir'])/'squad_e4').exists()
    ready=e2.source_readiness(spec)
    assert ready['status']=='SOURCE_RECORDS_READY_MODELS_NOT_CHECKED'
    assert set(ready['sources'])=={'R0','R1','R2','R3'}
    result=e2.freeze_e2(spec,client_factory=factory)
    assert result['status']=='FROZEN' and result['tasks']==2400 and result['per_model']==800
    manifest,inputs=load_campaign(result['output'])
    assert len(inputs)==800 and not manifest['provenance'].get('external_campaigns')
    assert {t['experiment_id'] for t in manifest['tasks']}=={'E2'}
    assert all(len(r['full_prompt_tokens_by_model'])==3 for r in inputs.values())
    assert e2.freeze_e2(spec,client_factory=factory)['manifest_hash']==result['manifest_hash']
    inherited=full.default_spec(tmp_path,'run')
    assert inherited['external_campaigns']==[{'manifest':artifact(Path(result['output'])/'EXPERIMENT_MANIFEST.json')}]


def test_missing_or_partial_r3_cannot_freeze_or_inspect_models(tmp_path):
    spec,_,paths=fixture(tmp_path)
    lines=paths['R3'].read_text().splitlines()
    paths['R3'].write_text('\n'.join(lines[:199])+'\n')
    def forbidden(*args):raise AssertionError('Not-ready source loaded model metadata')
    result=e2.freeze_e2(spec,client_factory=forbidden)
    assert result['status']=='PENDING' and result['pending'][0]['item']=='E2:R3'
    assert not (Path(result['out'])/'EXPERIMENT_MANIFEST.json').exists()
    paths['R3'].unlink()
    assert e2.source_readiness(spec)['status']=='PENDING'


def test_short_population_wrong_selection_or_blocked_core_is_rejected(tmp_path):
    spec,_,_=fixture(tmp_path,n=199)
    assert 'all200' in e2.source_readiness(spec)['pending'][0]['reason']
    wrong=copy.deepcopy(spec);wrong['main_models'].pop()
    with pytest.raises(ValueError,match='three'):e2.shard_spec(wrong)
    spec['blocks']['E2:R3']={'reason':'Not a runnable subset'}
    with pytest.raises(ValueError,match='omit or block'):e2.shard_spec(spec)


def test_main_imports_both_shards_with_exact_all3600_core_task_ownership(tmp_path,monkeypatch):
    spec,queries,paths=fixture(tmp_path)
    e2result=e2.freeze_e2(spec,client_factory=factory)
    monkeypatch.setattr(e3,'model_client_factory',factory)
    e3result=e3.freeze_e3(spec)
    assert len(full.default_spec(tmp_path,'run')['external_campaigns'])==2
    main=copy.deepcopy(spec)  # Simulates an explicit spec saved before either shard existed.
    assert not main.get('external_campaigns')
    # Auxiliaries are explicitly blocked test fixtures; they are not prerequisites
    # of the actual E2 shard. Main still declares every required condition.
    corpus=Path(spec['corpus_dir']);squad=corpus/'squad_e4';squad.mkdir()
    (squad/'documents.jsonl').write_bytes((corpus/'documents.jsonl').read_bytes())
    q={**queries['test'][0],'query_id':'squad-test'}
    (squad/'queries_test.jsonl').write_text(json.dumps(q)+'\n')
    atomic_json(squad/'SPLIT_MANIFEST.json',{'files':{p.name:{'sha256':file_hash(p)} for p in squad.glob('*.jsonl')},'selected_query_ids':[q['query_id']]})
    evidence=tmp_path/'offline_guard.json';atomic_json(evidence,{'fixture_only':True})
    block={'reason':'Offline contract fixture only','evidence':[str(evidence)]}
    main['model_blocks']['lfm']=block
    for key in ['E4:hybrid','E5:bm25','E5:hybrid','E6:simple','E6:structured','E7']:
        main['blocks'][key]=block
    for split in ('TRAIN','DEV'):
        for method in ('bm25','hybrid','hybrid_rerank'):main['blocks']['ROUTE_'+split+':'+method]=block
    assert full.freeze_all(main,client_factory=factory)['status']=='FROZEN'
    manifest,inputs=load_campaign(main['output_dir'])
    assert len(manifest['provenance']['external_campaigns'])==2
    sources=list(iter_record_sources(main['output_dir'],manifest,inputs))
    assert len(sources)==3 and [len(s['owned_task_ids']) for s in sources]==[0,1200,2400]
    assert len(set.union(*(s['owned_task_ids'] for s in sources)))==3600
    em,ei=load_campaign(e2result['output']); assert all(inputs[k]==v for k,v in ei.items())
    assert [t for t in manifest['tasks'] if t['experiment_id']=='E2']==em['tasks']
    class NeverCalled:
        real_model=True
        identity=manifest['models']['qwen_2b']
        def chat(self,*args,**kwargs):raise AssertionError('External tasks must never run in the main directory')
    budget=GenerationBudget(tmp_path/'budget.json')
    summaries=run_model(main['output_dir'],'qwen_2b',NeverCalled(),budget,evidence_units={})
    assert budget.state()['reserved_calls']==0
    assert summaries['E2']['n_pending']==800 and summaries['E3']['n_pending']==400


def test_mismatching_external_input_rejected_before_main_manifest_written(tmp_path):
    spec,_,_=fixture(tmp_path)
    result=e2.freeze_e2(spec,client_factory=factory)
    manifest,inputs=load_campaign(result['output'])
    # freeze_campaign expects unhashed input objects; alter a retained provenance field.
    raw={k:{field:value for field,value in row.items() if field not in {'input_hash','input_id'}} for k,row in inputs.items()}
    next(iter(raw.values()))['retrieval_timings']={'changed':1}
    parent=tmp_path/'bad-main'
    provenance={**manifest['provenance'],'external_campaigns':[{'manifest':artifact(Path(result['output'])/'EXPERIMENT_MANIFEST.json')}]}
    with pytest.raises(ValueError,match='External task semantics'):
        freeze_campaign(parent,manifest['models'],manifest['groups'],raw,provenance=provenance)
    assert not (parent/'EXPERIMENT_MANIFEST.json').exists()


def test_e2_execution_cli_has_no_squad_dependency(tmp_path,monkeypatch):
    import run_generation_campaign as runner
    calls=[]
    corpus=tmp_path/'corpus';corpus.mkdir()
    home=tmp_path/'runtime';(home/'runs').mkdir(parents=True)
    identity={'model':'offline-fixture'}
    manifest={'tasks':[{'experiment_id':'E2','model_id':'qwen_2b'}],
        'models':{'qwen_2b':identity},'provenance':{'spec':{'campaign':'offline',
            'corpus_dir':str(corpus),'models':{'qwen_2b':{}}}}}
    monkeypatch.setattr(runner,'load_campaign',lambda path:(manifest,{}))
    monkeypatch.setattr(runner,'configure',lambda **kwargs:None)
    monkeypatch.setattr(runner,'runtime_home',lambda:home)
    def load(path):
        calls.append(Path(path));assert Path(path)==corpus
        return {'dev':[{'query_id':'dev'}]}, {}, {}
    monkeypatch.setattr(runner,'verified_corpus',load)
    monkeypatch.setattr(runner,'model_client_factory',lambda *args:(SimpleNamespace(identity=identity),{}))
    monkeypatch.setattr(runner,'verify_owned_service',lambda *args:{})
    monkeypatch.setattr(runner,'MeasuredClient',lambda client,*args,**kwargs:client)
    monkeypatch.setattr(runner,'run_model',lambda *args,**kwargs:{'E2':{'n_pending':0}})
    monkeypatch.setattr(sys,'argv',['run_generation_campaign.py','--campaign-dir',str(tmp_path/'campaign'),'--model-id','qwen_2b'])
    assert runner.main()==0 and calls==[corpus]
