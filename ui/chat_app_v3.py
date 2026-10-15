"""v3 local workbench reuses v2 result validation; no model launch/download in UI."""
from pathlib import Path
import json
import os
import sys
import urllib.error
import urllib.request
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ragscope.v3.workbench import validate_request
from ui.chat_app import load_result_file, result_view
from ragscope.generation.llm_client import _NoRedirect


def request_answer(payload):
    validation=validate_request(payload)
    if validation:return validation
    # Fixed loopback destination: never accept a URL supplied through the page.
    request=urllib.request.Request('http://127.0.0.1:11437/api/answer',
        data=json.dumps(payload).encode('utf8'),headers={'Content-Type':'application/json'})
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),_NoRedirect())
    try:
        with opener.open(request,timeout=180) as response:return json.load(response)
    except urllib.error.HTTPError as exc:
        try:return json.load(exc)
        except (ValueError, UnicodeError):return {'action':'EXECUTION_ERROR','message':'本地服务返回了无法读取的错误。'}
    except urllib.error.URLError:
        return {'action':'MODEL_UNAVAILABLE','message':'本地问答服务未就绪。请按启动说明手动启动；页面不会自动下载或改用Mock。'}


def main():
    import streamlit as st
    st.set_page_config(page_title='RAGScope v3',page_icon='🔎',layout='wide')
    st.title('RAGScope v3 · 本地诊断工作台')
    st.caption('默认保留v2策略 · core-first仅供手动开发诊断 · 人工审核尚未完成')
    run=Path(os.environ.get('RAGSCOPE_V3_CAMPAIGN','D:/RAGScope_Runtime/runs/v3_20261015_014506'))
    tabs=st.tabs(['资料问答','实际实验记录','诊断结论'])
    with tabs[0]:
        pid=st.text_input('论文范围（doc_id）')
        question=st.text_input('问题')
        condition=st.selectbox('处理策略',['B0：保留v2策略','C1：core-first开发诊断'])
        cache=st.checkbox('答案缓存（应用测试，独立于正式在线实验）',value=False)
        if st.button('运行',type='primary'):
            result=request_answer({'question':question,'doc_id':pid,
                'condition':'B0' if condition.startswith('B0') else 'C1_DIAGNOSTIC','answer_cache':cache})
            st.session_state['v3_result']=result
        if 'v3_result' in st.session_state:
            r=st.session_state['v3_result']
            if r['action']=='ANSWER':
                result_view(st,{**r,'timings':{'total_ms':r.get('application_seconds',0)*1000}})
                st.caption('答案缓存命中' if r.get('answer_cache_hit') else '实际本地模型请求；不计为冻结测试成绩')
            elif r['action']=='NO_EVIDENCE':st.warning(r['answer'])
            elif r['action'] in {'EMPTY_QUESTION','CLARIFY_DOCUMENT'}:st.warning(r['message'])
            else:st.error(r['message'])
    with tabs[1]:
        paths=[str(p) for p in [run/'S6_ONLINE_COST.jsonl',run/'S3_SCOPED_GENERATION_D_confirm_resource.jsonl'] if p.is_file()]
        if paths:
            chosen=st.selectbox('逐题记录',paths)
            if st.button('读取并校验'):
                try:
                    _,rows=load_result_file(chosen);st.session_state['v3_rows']=rows
                except Exception as exc:st.error('记录校验失败：'+str(exc))
            rows=st.session_state.get('v3_rows',[])
            if rows:
                st.caption(f'实际已保存{len(rows)}条，以下是历史回放')
                i=st.number_input('记录序号',min_value=1,max_value=len(rows),value=1)
                row=rows[int(i)-1];st.json({k:row.get(k) for k in ('question_id','condition','parsed_response','error_type','wall_time')})
        else:st.info('尚无可核验记录。')
    with tabs[2]:
        for name in ('EXECUTIVE_SUMMARY_ZH.md','ROOT_CAUSE_VERDICTS.md'):
            path=ROOT/'docs/execution/v3'/name
            if path.is_file():st.markdown(path.read_text('utf8'))
        st.caption('支持性与语义评分为AI代理；未知结果不作为正确或错误计分。')


if __name__=='__main__':main()
