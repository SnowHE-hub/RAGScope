"""Functional gates only; no mock outputs are counted as model-quality evidence."""
import pytest
from ragscope.v3.workbench import validate_request, Workbench


def test_missing_inputs_are_resolved_before_model_access():
    assert validate_request({'question':' ','doc_id':''})['action']=='EMPTY_QUESTION'
    assert validate_request({'question':'Compare the methods','doc_id':''})['action']=='CLARIFY_DOCUMENT'
    assert validate_request({'question':'What data?','doc_id':'paper'}) is None
    with pytest.raises(ValueError,match='Gold/evaluation'):
        validate_request({'question':'What data?','doc_id':'paper','answer':'leaked gold'})


def test_no_evidence_does_not_reserve_or_call_model():
    class Sparse:
        def scores(self,question):return [0,0]
    class Engine:
        by_doc={'paper':[0,1]}
        scoped_vectors={'paper':'functional_fixture_not_real_vectors'}
        chunk_sparse=Sparse()
    work=object.__new__(Workbench);work.engine=Engine();work.cache={}
    work.store=type('Store',(),{'config_hash':'functional-fixture'})()
    result=work.answer({'question':'unmatchedword','doc_id':'paper'})
    assert result['action']=='NO_EVIDENCE' and result['new_llm_requests']==0
    assert work.answer({'question':'valid','doc_id':'missing'})['action']=='INVALID_DOCUMENT'


def test_incomplete_hybrid_does_not_silently_generate_with_sparse_fallback():
    work=object.__new__(Workbench)
    work.engine=type('Engine',(),{'by_doc':{'paper':[]},'scoped_vectors':{}})()
    result=work.answer({'question':'What datasets?','doc_id':'paper'})
    assert result['action']=='HYBRID_UNAVAILABLE' and result['new_llm_requests']==0


def test_actual_streamlit_empty_and_missing_scope_controls():
    from streamlit.testing.v1 import AppTest
    from pathlib import Path
    app=Path(__file__).resolve().parents[2]/'ui/chat_app_v3.py'
    at=AppTest.from_file(str(app),default_timeout=30).run()
    assert not at.exception
    at.button[0].click().run()
    assert at.warning[0].value=='请输入问题。'
    at.text_input[1].set_value('Compare the methods').run()
    at.button[0].click().run()
    assert at.warning[0].value=='请先指定要分析的论文或文档。'


def test_actual_streamlit_unavailable_service_has_no_mock_fallback():
    from streamlit.testing.v1 import AppTest
    from pathlib import Path
    import socket
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1',11437))==0:
            pytest.skip('Unavailable-service case requires no live local workbench')
    at=AppTest.from_file(str(Path(__file__).resolve().parents[2]/'ui/chat_app_v3.py'),default_timeout=30).run()
    at.text_input[0].set_value('paper').run();at.text_input[1].set_value('What data?').run()
    at.button[0].click().run()
    assert not at.exception
    assert '不会自动下载或改用Mock' in at.error[0].value
