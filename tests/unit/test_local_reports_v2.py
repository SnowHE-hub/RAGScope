"""Offline report contracts only; fixtures are not real experiment results."""
import json

import pytest

from scripts import build_local_reports_v2 as reports
from ragscope.experiments.ledger import RecordStore
from ragscope.indexing.identity import artifact


def spec():
    return {'version':'local_reports_v1','campaign_dir':'not-opened','routing_dir':'not-opened',
            'sources':{name:None for name in reports.SOURCE_NAMES}}


def test_prepare_never_reads_experiments(monkeypatch):
    monkeypatch.setattr(reports,'read',lambda *args:pytest.fail('Preparation opened experiment data'))
    result=reports.prepare(spec())
    assert result['experiment_sources_opened']==0 and result['model_calls']==0
    assert len(result['unbound_sources'])==9


def test_missing_seventh_freeze_prevents_test_analysis_read(tmp_path,monkeypatch):
    def gate(*args):raise ValueError('seventh decision unavailable')
    monkeypatch.setattr(reports,'frozen_policies',gate)
    monkeypatch.setattr(reports,'source',lambda *args:pytest.fail('Test source opened before freeze'))
    with pytest.raises(ValueError,match='seventh decision'):
        reports.build(spec(),tmp_path/'output',command=['fixture'])
    assert not (tmp_path/'output').exists()


def test_bound_source_and_nested_dependencies_are_verified(tmp_path):
    raw=tmp_path/'raw.json';raw.write_text('{"fixture":1}')
    summary=tmp_path/'summary.json';summary.write_text(json.dumps({'sources':[artifact(raw)],'status':'COMPLETE'}))
    bound=artifact(summary);sources=[]
    assert reports.source(bound,'fixture',sources)['status']=='COMPLETE'
    assert len(sources)==2
    raw.write_text('{"fixture":2}')
    with pytest.raises(ValueError,match='dependency changed'):
        reports.source(bound,'fixture',[])
    assert reports.source(None,'missing',[]) is None
    assert 'UNAVAILABLE' in reports.block('missing',None)


def test_resource_missing_measurement_is_na_and_failure_denominator_preserved():
    analysis={'groups':[{'experiment_id':'fixture','model_id':'model','method':'R0',
        'n_observed':10,'n_expected':10,'observed_summary':{'n_failed':2,'latency_ms':{'p50':1,'p95':2}},
        'performance':{'resource_measured_n':0}}]}
    text='\n'.join(reports.render_resources(analysis,{}))
    assert '|10|10|2|0|' in text and 'TTFT' in text and 'N/A' in text
    assert 'UNAVAILABLE' in text


def test_incomplete_paired_does_not_render_quality_estimate():
    text='\n'.join(reports.render_training({'paired_summary':{
        'status':'INCOMPLETE','completed':{'baseline':1,'adapted':0},
        'pending':{'baseline':99,'adapted':100},'baseline':{'answer_f1':0.98765}}}))
    assert 'UNAVAILABLE' in text and '99' in text and '0.98765' not in text


def test_retrieval_full_denominator_is_checked_from_recordstore(tmp_path):
    store=RecordStore(tmp_path/'run.jsonl',{'query_ids':['one','two']})
    store.add('one',{'query_id':'one','error':None})
    store.add('two',{'query_id':'two','error':{'kind':'fixture'}})
    data={'test_used_for_selection':False,'runs':{'run':{'status':'PASS','n_total':2,'n_failed':1,
        'artifacts':[artifact(store.path)]}}}
    reports.validate_retrieval(data)
    data['runs']['run']['n_total']=1
    with pytest.raises(ValueError,match='denominator'):
        reports.validate_retrieval(data)


def test_unbound_replay_does_not_read_an_unapproved_path(monkeypatch):
    monkeypatch.setattr(reports,'read',lambda *args:pytest.fail('Unbound replay opened'))
    assert reports.routing_data(spec(),{}, {'fixed_bm25':{}},[]) == {'fixed_bm25':None,'laya':None}


def test_partial_online_summary_does_not_become_final_metric_table():
    text='\n'.join(reports.online_sections({'status':'PENDING','test_used':False,
        'groups':{'fixture':{'summary':{'answer_f1':0.98765}}}}))
    assert 'UNAVAILABLE' in text and 'PENDING' in text and '0.98765' not in text


def test_fixture_build_writes_only_four_reports_and_refuses_overwrite(tmp_path,monkeypatch):
    analysis=tmp_path/'analysis.json';analysis.write_text(json.dumps({'status':'COMPLETE','groups':[],
        'n_observed':0,'manifest_hash':'explicit-offline-fixture'}))
    config=spec();config['sources']['analysis']=artifact(analysis)
    monkeypatch.setattr(reports,'frozen_policies',lambda *args:({},{}))
    monkeypatch.setattr(reports,'validate_main',lambda *args:{'manifest_hash':'explicit-offline-fixture'})
    output=tmp_path/'fixture_reports'
    receipt=reports.build(config,output,command=['offline-fixture'])
    assert {p.name for p in output.iterdir()}==set(reports.FILES)|{'LOCAL_REPORTS_BUILD.json'}
    assert receipt['model_calls']==0 and receipt['human_review']=='NOT_REVIEWED'
    assert all('UNAVAILABLE' in (output/name).read_text() for name in reports.FILES)
    with pytest.raises(FileExistsError):
        reports.build(config,output,command=['offline-fixture'])
