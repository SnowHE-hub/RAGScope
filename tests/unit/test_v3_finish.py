import numpy as np
import pytest
from types import SimpleNamespace
from ragscope.v3.full_retrieval import FullCorpusRetriever
from ragscope.v3.finish_runtime import task_kind, answer_messages, evidence_bundle,resolve_policy
from ragscope.v3.practical_quality import fact_checks
from ragscope.v3.full_workbench import validate_full_request
from ragscope.v3.deployment import deployment_kind,deployment_context


def tiny_engine():
    engine=FullCorpusRetriever.__new__(FullCorpusRetriever)
    engine.embedding_identity={'model':'test-only'}
    engine.documents=[{'doc_id':str(i),'title':f'Paper {i}'} for i in range(3)]
    engine.doc_rows={str(i):i for i in range(3)}
    engine.chunks=[]
    for i,text in enumerate(['Objective and method.','Results improve accuracy.','Conclusions and limitations.']):
        engine.chunks.append({'chunk_id':f'c{i}','doc_id':str(i),'text':text,
            'canonical_start':0,'canonical_end':len(text),'unit_spans':[]})
    engine.by_doc={str(i):[i] for i in range(3)}
    engine.vectors=np.zeros((3,1024),dtype=np.float32)
    engine.vectors[0,0]=1;engine.vectors[1,1]=1;engine.vectors[2,2]=1
    engine.document_vectors=engine.vectors.copy()
    engine.chunk_sparse=SimpleNamespace(scores=lambda q:np.array([3.,2.,1.]))
    engine.metadata_sparse=SimpleNamespace(scores=lambda q:np.array([1.,2.,3.]))
    engine.document_sparse=SimpleNamespace(scores=lambda q:np.array([1.,2.,3.]))
    return engine


def test_full_dense_can_retrieve_last_paper_globally_and_in_scope():
    e=tiny_engine();q=e.vectors[2]
    for scope in (None,{'doc_ids':['2'],'kind':'document'}):
        result=e.search('result',scope=scope,query_vector=q,embedding_identity=e.embedding_identity,method='dense')
        assert result['candidates'][0]['doc_id']=='2'
        assert result['full_chunk_dense_status']=='COMPLETE'


def test_multiple_papers_are_balanced_and_never_escape_scope():
    e=tiny_engine()
    result=e.search('result',scope={'doc_ids':['0','2']},query_vector=e.vectors[2],
        embedding_identity=e.embedding_identity,balanced=True,k=2)
    assert {r['doc_id'] for r in result['candidates']}=={'0','2'}
    assert not result['document_lookup_performed']


def test_embedding_space_mismatch_and_unknown_scope_are_errors():
    e=tiny_engine()
    with pytest.raises(ValueError,match='space'):
        e.search('query',query_vector=e.vectors[0],embedding_identity={})
    with pytest.raises(ValueError,match='not indexed'):
        e.search('query',scope={'doc_ids':['missing']},method='bm25')


def test_task_prompt_uses_question_only_and_keeps_sources_as_data():
    assert task_kind('Why does this method help?')=='explanation'
    assert task_kind('Compare the two methods.')=='comparison'
    assert task_kind('Summarize the paper.')=='summary'
    assert task_kind('How many examples?')=='fact'
    context={'pieces':[{'chunk_id':'c0','text':'A supported fact.'}]}
    messages=answer_messages('Is X absent?',context,condition='FINAL',kind='fact')
    assert 'negative answer is not abstention' in messages[0]['content']
    assert '[c0]' in messages[1]['content']


def test_baseline_and_final_contexts_preserve_exact_source_text():
    e=tiny_engine()
    for condition in ('B0','FINAL'):
        context,result,kind=evidence_bundle(e,'What is stated?',{'doc_ids':['2']},e.vectors[2],condition=condition)
        assert context['pieces'][0]['text']==e.chunks[2]['text']
        assert context['pieces'][0]['doc_id']=='2'


def test_fact_checks_flag_literal_number_and_polarity_conflicts_without_claiming_truth():
    pieces=[{'chunk_id':'p','text':'No. The experiment uses 20 examples.'}]
    check=fact_checks('Did it work?','Yes, using 30 examples.',['No'],pieces,['p'])
    assert check['yes_no_reference_conflict'] is True
    assert check['numbers_not_in_context']==['30']
    assert check['factual_correctness']=='UNKNOWN'


def test_application_allows_global_and_multiple_papers_but_rejects_gold_fields():
    assert validate_full_request({'question':'Which methods use attention?','doc_ids':[]}) is None
    assert validate_full_request({'question':'Compare the papers.','doc_ids':['a','b'],'task':'comparison'}) is None
    with pytest.raises(ValueError,match='Gold'):
        validate_full_request({'question':'q','answer_refs':['secret']})


def test_final_policy_preserves_successful_scoped_qa_and_uses_new_global_and_complex_workflows():
    assert resolve_policy('How many examples?',{'doc_ids':['p']},'FINAL','SAFE_QA_TASKS')=='B0'
    assert resolve_policy('Which approaches use attention?',{'kind':'global'},'FINAL','SAFE_QA_TASKS')=='CONTROL'
    assert resolve_policy('Summarize this paper.',{'doc_ids':['p']},'FINAL','SAFE_QA_TASKS')=='FINAL'
    assert resolve_policy('Compare the findings.',{'doc_ids':['p','q']},'FINAL','SAFE_QA_TASKS')=='FINAL'
    assert resolve_policy('Summarize this paper.',{'doc_ids':['p']},'B0','SAFE_QA_TASKS')=='B0'


def test_deployment_distinguishes_baseline_name_questions_from_comparison_tasks():
    assert deployment_kind('Which models is the approach compared to?')=='fact'
    assert deployment_kind('Does it outperform the baseline?')=='fact'
    assert deployment_kind('Compare the two methods and their limitations.')=='comparison'


def test_long_summary_context_balances_papers_and_reports_real_source_coverage():
    e=tiny_engine();original=SimpleNamespace(prompt_token_counter=lambda messages:sum(len(m['content'].split()) for m in messages))
    selection={'selected':{'options':{}},'global_selected':{'options':{}}}
    context,result,kind,window,limit,policy=deployment_context(e,'Summarize the papers.',{'doc_ids':['0','2']},e.vectors[0],original,selection)
    assert {p['doc_id'] for p in context['pieces']}=={'0','2'}
    assert result['source_character_coverage']=={'0':1.,'2':1.}
    assert window==8192 and limit==768
