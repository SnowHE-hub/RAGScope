"""Offline only: deterministic window packing and retained failure contracts."""
from dataclasses import asdict
import json

import pytest

from ragscope.experiments.context_stress import freeze_stress, load_stress, near_window_context, run_window, WINDOWS
from ragscope.experiments.generation_campaign import GenerationBudget
from ragscope.generation.llm_client import GenerationError
from ragscope.indexing.identity import artifact


class OfflineWindowFixture:
    real_model=True
    supports_output_schema=True
    json_constraint=True
    prompt_token_counter=staticmethod(lambda messages: sum(len(m['content']) for m in messages)//4)
    def __init__(self,window,oom=False):
        self.identity={'model':'offline-window-fixture','revision':'fixture','weight_sha256':'fixture',
                       'backend_version':'fixture','num_ctx':window}
        self.calls=0;self.oom=oom;self.last_metadata={}
    def chat(self,messages,**kwargs):
        self.calls+=1
        if self.oom: raise GenerationError('oom','Offline fixture OOM')
        count=self.prompt_token_counter(messages)
        self.last_metadata={'actual_input_tokens':count,'prompt_eval_count':count,'token_count_match':True}
        return '{"answer":"","citations":[],"abstain":true}'


def inputs():
    text='Canonical scientific paper paragraph. '*2000
    doc={'doc_id':'paper','title':'Title','text':text,'canonical_checksum':'fixture',
         'evidence_units':[{'unit_id':'u','canonical_start':0,'canonical_end':len(text)}]}
    queries=[{'query_id':f'q{i:03d}','question':'Which property?','doc_id':'paper','scope':'paper',
              'split':'dev','answer_refs':['MUST_NOT_ENTER_DIAGNOSTIC'],'answerability':'answerable'} for i in range(100)]
    return queries,doc


def frozen(tmp_path):
    queries,doc=inputs()
    evidence=tmp_path/'source.json'; evidence.write_text('synthetic source receipt')
    clients={w:OfflineWindowFixture(w) for w in WINDOWS}
    run=tmp_path/'run'
    manifest=freeze_stress(run,queries,{'paper':doc},clients,[artifact(evidence)])
    return run,manifest,clients


def test_windows_really_grow_and_preserve_source_coordinates():
    qs,doc=inputs()
    variants={w:near_window_context(qs[0],doc,w,OfflineWindowFixture.prompt_token_counter) for w in WINDOWS}
    for window,row in variants.items():
        p=row['context_bundle']['pieces'][0]
        assert p['text']==doc['text'][p['canonical_start']:p['canonical_end']]
        assert window-512-64<=row['expected_actual_input_tokens']<=window-512
    assert variants[8192]['expected_actual_input_tokens']>variants[4096]['expected_actual_input_tokens']*2


def test_short_source_and_test_split_are_rejected():
    qs,doc=inputs()
    with pytest.raises(ValueError,match='too short'):
        near_window_context(qs[0],{**doc,'text':'short'},8192,OfflineWindowFixture.prompt_token_counter)
    with pytest.raises(ValueError,match='dev paper'):
        near_window_context({**qs[0],'split':'test'},doc,4096,OfflineWindowFixture.prompt_token_counter)
    structured={**qs[0],'scope':{'kind':'document','doc_ids':['paper']}}
    assert near_window_context(structured,doc,4096,OfflineWindowFixture.prompt_token_counter)['window']==4096
    with pytest.raises(ValueError,match='dev paper'):
        near_window_context({**structured,'scope':{'kind':'global'}},doc,4096,OfflineWindowFixture.prompt_token_counter)


def test_selection_caps20_all_candidates_audited_no_answer_labels(tmp_path):
    run,manifest,clients=frozen(tmp_path)
    assert len(manifest['query_ids'])==20 and len(manifest['selection_audit'])==100
    assert sum(x['status']=='NOT_EVALUATED_CAP20' for x in manifest['selection_audit'])==80
    _,packed=load_stress(run)
    assert len(packed)==40 and 'MUST_NOT_ENTER_DIAGNOSTIC' not in json.dumps(packed)
    assert all(set(row['query'])=={'query_id','question','doc_id','scope','split'} for row in packed.values())


def test_failed_warmup_preserved_and_circuit_cannot_repeat(tmp_path):
    run,manifest,clients=frozen(tmp_path)
    client=OfflineWindowFixture(8192,oom=True)
    budget=GenerationBudget(tmp_path/'budget.json')
    result=run_window(run,8192,client,budget)
    assert result['status']=='BLOCKED' and result['n_pending']==20 and result['warmup_n']==1
    assert budget.state()['reserved_calls']==1
    assert run_window(run,8192,client,budget)==result and client.calls==1


def test_resume_keeps_formal_ids_and_counts_separate_warmups(tmp_path):
    run,manifest,clients=frozen(tmp_path)
    client=clients[4096];budget=GenerationBudget(tmp_path/'budget.json')
    first=run_window(run,4096,client,budget,max_tasks=1)
    assert first['n_observed']==1 and first['warmup_n']==3
    second=run_window(run,4096,client,budget)
    assert second['status']=='COMPLETE' and second['n_observed']==20 and second['warmup_n']==6
    assert client.calls==26 and budget.state()['reserved_calls']==26


def test_corrupt_input_is_rejected_before_call(tmp_path):
    run,manifest,clients=frozen(tmp_path)
    path=run/'FROZEN_INPUTS.json'; rows=json.loads(path.read_text());next(iter(rows.values()))['expected_actual_input_tokens']=1
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='input changed'):
        run_window(run,4096,clients[4096],GenerationBudget(tmp_path/'budget.json'))
    assert clients[4096].calls==0
