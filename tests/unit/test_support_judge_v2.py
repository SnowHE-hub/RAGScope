"""Offline contract tests only; no model service or test answer quality evidence."""
import copy
import json
from dataclasses import asdict

import pytest

from ragscope.experiments.generation_campaign import GenerationBudget, freeze_campaign
from ragscope.experiments.ledger import RecordStore, read_records
from ragscope.experiments.support_judge import (freeze_sample, run_judge, load_sample,
    judge_messages, parse_judgment, SCHEMA, verified_e2_records)
from ragscope.generation.context import pack_context
from ragscope.generation.llm_client import LocalLLMClient, GenerationError

IDENTITY={'model':'offline-fixture-4b','revision':'fixture','weight_sha256':'no-real-weight',
          'backend_version':'offline-test-only'}


class OfflineJudge:
    """Satisfies the runtime contract solely inside tmp_path unit tests."""
    real_model=True; identity=IDENTITY; num_ctx=4096
    prompt_token_counter=staticmethod(lambda messages:10)
    def __init__(self,outputs=None):
        self.outputs=list(outputs or []);self.calls=[];self.last_metadata={}
    def chat(self,messages,**kwargs):
        self.calls.append((messages,kwargs));self.last_metadata={'prompt_eval_count':10}
        result=self.outputs.pop(0) if self.outputs else '{"verdict":"supported","unsupported_claims":[],"rationale":"Offline test fixture"}'
        if isinstance(result,Exception):raise result
        return result


def fixture_campaign(tmp_path,n=4,write_rows=True,source_failure=False):
    piece={'chunk_id':'c','doc_id':'d','text':'Source fact.','canonical_start':0,'canonical_end':12,'unit_spans':[]}
    inputs={str(i):{'query':{'query_id':str(i),'question':'Question?','doc_id':'d','scope':'d','split':'test',
        'answerability':'answerable' if i%2 else 'unanswerable','answer_refs':['HIDDEN_REFERENCE'],
        'evidence_reference_sets':[['HIDDEN_LABEL']]},'context_bundle':asdict(pack_context([piece],token_counter=len))} for i in range(n)}
    groups=[{'experiment_id':'E2','method':method,'model_ids':['qwen_4b'],'input_ids':list(inputs)} for method in ['R0','R2']]
    manifest=freeze_campaign(tmp_path/'campaign',{'qwen_4b':IDENTITY},groups,inputs)
    if write_rows:
        store=RecordStore(tmp_path/'campaign/records/E2_qwen_4b.jsonl',
            {'manifest_hash':manifest['manifest_hash'],'model_id':'qwen_4b','experiment_id':'E2'},real_only=True)
        for task in manifest['tasks']:
            store.add(task['task_id'],{**task,'real_model':True,'model_identity':IDENTITY,'call_ticket':{'call_number':1},
                'answer':'Source fact.','raw_output':'{}','citations':['c'],'abstain':False,
                'context_bundle':inputs[task['input_id']]['context_bundle'],
                'error':{'kind':'timeout'} if source_failure else None})
    return manifest,inputs


def test_sample_is_balanced_seeded_quality_independent_and_immutable(tmp_path):
    manifest,_=fixture_campaign(tmp_path,8,False)
    sample=freeze_sample(tmp_path/'campaign',tmp_path/'judge',n=12)
    assert len(sample['sample'])==12 and {row['sample_n'] for row in sample['strata']}=={3}
    assert sample==freeze_sample(tmp_path/'campaign',tmp_path/'judge',n=12)
    assert all(row['task_id'] in {t['task_id'] for t in manifest['tasks']} for row in sample['sample'])
    with pytest.raises(ValueError,match='Immutable'):freeze_sample(tmp_path/'campaign',tmp_path/'judge',n=11)
    with pytest.raises(ValueError,match='bounded'):freeze_sample(tmp_path/'campaign',tmp_path/'x',n=201)
    with pytest.raises(ValueError,match='Qwen'):freeze_sample(tmp_path/'campaign',tmp_path/'x','qwen_2b')
    path=tmp_path/'judge/SAMPLE_MANIFEST.json';sample['seed']=43;path.write_text(json.dumps(sample))
    with pytest.raises(ValueError,match='hash'):load_sample(tmp_path/'judge')


def test_prompt_contains_only_cited_actual_evidence_no_gold_or_model_labels(tmp_path):
    manifest,inputs=fixture_campaign(tmp_path)
    rows=verified_e2_records(tmp_path/'campaign',manifest);row=copy.deepcopy(next(iter(rows.values())))
    row['citations']=['c','missing'];row['invalid_citations']=['other']
    prompt=judge_messages(inputs[row['input_id']],row)
    payload=json.loads(prompt[1]['content'])
    assert payload['cited_source_excerpts']==[{'citation_id':'c','text':'Source fact.'}]
    assert payload['invalid_citation_ids']==['other','missing']
    assert 'HIDDEN' not in str(prompt) and 'offline-fixture' not in str(prompt)
    assert 'answer_refs' not in str(prompt) and 'answerability' not in str(prompt)


def test_actual_judge_contract_counted_schema_parity_resume_and_human_export(tmp_path):
    fixture_campaign(tmp_path,2)
    sample=freeze_sample(tmp_path/'campaign',tmp_path/'judge',n=4)
    client=OfflineJudge();budget=GenerationBudget(tmp_path/'budget',limit=10,core_reserved=0)
    result=run_judge(tmp_path/'judge',client,budget)
    assert result['status']=='COMPLETE' and result['n_actual_judge_calls']==4 and result['n_same_model_bias']==4
    assert budget.state()['reserved_calls']==4 and len(client.calls)==4
    assert client.calls[0][1]['output_schema']==SCHEMA
    assert run_judge(tmp_path/'judge',client,budget)['n_pending']==0 and len(client.calls)==4
    rows=read_records(tmp_path/'judge/judgments.jsonl')
    assert all(row['prompt_hash'] and row['source_record_hash'] and row['call_ticket'] for row in rows)
    export=json.loads((tmp_path/'judge/HUMAN_REVIEW.json').read_text())
    assert len(export['examples'])==4 and all(x['human_review_status']=='NOT_REVIEWED' and x['human_verdict'] is None for x in export['examples'])


def test_source_failures_stay_unknown_without_judge_calls_or_replacement(tmp_path):
    fixture_campaign(tmp_path,2,source_failure=True);freeze_sample(tmp_path/'campaign',tmp_path/'judge',n=4)
    client=OfflineJudge();budget=GenerationBudget(tmp_path/'budget',limit=10,core_reserved=0)
    result=run_judge(tmp_path/'judge',client,budget)
    assert result['verdict_counts']=={'unknown':4} and result['n_expected']==4
    assert result['n_actual_judge_calls']==0 and budget.state()['reserved_calls']==0
    assert all(not row['real_model'] for row in read_records(tmp_path/'judge/judgments.jsonl'))


def test_incomplete_matrix_blocks_all_calls(tmp_path):
    fixture_campaign(tmp_path,2,False);freeze_sample(tmp_path/'campaign',tmp_path/'judge',n=4)
    client=OfflineJudge();budget=GenerationBudget(tmp_path/'budget',limit=10,core_reserved=0)
    with pytest.raises(FileNotFoundError):run_judge(tmp_path/'judge',client,budget)
    assert client.calls==[]


def test_parse_failure_transport_failure_and_token_mismatch_never_become_supported(tmp_path):
    fixture_campaign(tmp_path,2);freeze_sample(tmp_path/'campaign',tmp_path/'judge',n=4)
    client=OfflineJudge(['invalid JSON',TimeoutError('offline'),'{"verdict":"supported"}'])
    budget=GenerationBudget(tmp_path/'budget',limit=10,core_reserved=0)
    result=run_judge(tmp_path/'judge',client,budget)
    assert result['verdict_counts']=={'unknown':3,'supported':1} and budget.state()['reserved_calls']==4
    other=tmp_path/'different';other.mkdir();fixture_campaign(other,1);freeze_sample(other/'campaign',other/'judge',n=2)
    class Mismatch(OfflineJudge):
        def chat(self,*args,**kwargs):
            answer=super().chat(*args,**kwargs);self.last_metadata={'prompt_eval_count':11};return answer
    result=run_judge(other/'judge',Mismatch(),budget)
    assert result['verdict_counts']=={'unknown':2}


def test_input_budget_no_truncation_and_call_budget_leave_pending(tmp_path):
    fixture_campaign(tmp_path,2);freeze_sample(tmp_path/'campaign',tmp_path/'judge',n=4)
    client=OfflineJudge();client.num_ctx=520
    budget=GenerationBudget(tmp_path/'budget',limit=1,core_reserved=0)
    result=run_judge(tmp_path/'judge',client,budget)
    assert result['n_actual_judge_calls']==0 and result['n_failed_or_unassessable']==4
    other=tmp_path/'other';other.mkdir();fixture_campaign(other,2);freeze_sample(other/'campaign',other/'judge',n=4)
    result=run_judge(other/'judge',OfflineJudge(),budget)
    assert result['n_pending']==3 and result['stop_reason']['kind']=='generation_budget_exhausted'


def test_circuit_break_and_output_parser(tmp_path):
    fixture_campaign(tmp_path,3);freeze_sample(tmp_path/'campaign',tmp_path/'judge',n=6)
    result=run_judge(tmp_path/'judge',OfflineJudge([GenerationError('oom','offline fixture')]),
        GenerationBudget(tmp_path/'budget',limit=10,core_reserved=0))
    assert result['n_pending']==5 and result['stop_reason']['kind']=='backend_circuit_break'
    for raw in ['[]','{"verdict":"yes","rationale":"x","unsupported_claims":[]}',
                '{"verdict":"supported","rationale":"x","unsupported_claims":"bad"}']:
        with pytest.raises(ValueError):parse_judgment(raw)


def test_local_backend_passes_frozen_judge_schema_without_changing_rag_default(monkeypatch):
    calls=[]
    def request(self,path,payload):
        calls.append(payload);return {'message':{'content':'{}'}}
    monkeypatch.setattr(LocalLLMClient,'_request',request)
    client=LocalLLMClient('offline-fixture')
    client.chat([],output_schema=SCHEMA);assert calls[-1]['format']==SCHEMA
    client.chat([]);assert calls[-1]['format']['required']==['answer','citations','abstain']
