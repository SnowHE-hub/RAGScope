"""Full-corpus local workbench; UI never starts or downloads a model."""
import json
from pathlib import Path
import sys
import urllib.request
import urllib.error
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ragscope.generation.llm_client import _NoRedirect
from ragscope.experiments.ledger import read_records


def api(path,payload=None):
    request=urllib.request.Request('http://127.0.0.1:11437'+path,
        data=json.dumps(payload).encode('utf8') if payload is not None else None,
        headers={'Content-Type':'application/json'})
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),_NoRedirect())
    with opener.open(request,timeout=180 if payload is not None else 5) as response:return json.load(response)


def main():
    import streamlit as st
    st.set_page_config(page_title='RAGScope',page_icon='🔎',layout='wide')
    st.title('RAGScope · 本地论文问答')
    config_path=ROOT/'configs/v3/default.json'
    if not config_path.exists():st.info('最终配置尚未生成。');return
    config=json.loads(config_path.read_text('utf8'));run=Path(config['run'])
    if config.get('release_status')!='VERIFIED':st.caption('完整索引已接入，最终对照与运行验收正在执行。')
    try:
        state=api('/api/state');catalog=state['documents']
        st.caption(f"完整索引：{state['papers']} 篇 / {state['chunks']} 块 · 模型：{state['model']}")
    except (urllib.error.URLError,ValueError,KeyError):
        state=None;catalog=json.loads((run/'WORKBENCH_CATALOG.json').read_text('utf8')) if (run/'WORKBENCH_CATALOG.json').exists() else []
        st.warning('问答服务尚未启动。按 LOCAL_RUN_V3.md 启动服务后刷新；已保存实验仍可查看。')
    tabs=st.tabs(['资料问答','实验结果','运行说明'])
    with tabs[0]:
        titles={d['doc_id']:d['title'] for d in catalog}
        ids=st.multiselect('论文范围（留空搜索全库；选择多篇可进行跨论文比较）',list(titles),
            format_func=lambda pid:titles[pid]+' · '+pid,max_selections=10)
        question=st.text_area('问题',placeholder='例如：Which datasets are used to evaluate this method?')
        kinds={'自动识别':'auto','简短事实':'fact','解释':'explanation','比较':'comparison','总结':'summary'}
        task=st.selectbox('回答方式',list(kinds))
        methods={'混合检索':'hybrid','向量检索':'dense','关键词检索':'bm25'}
        method=st.selectbox('检索方式',list(methods))
        cached=st.checkbox('复用相同问题的答案',value=False)
        if st.button('生成答案',type='primary',disabled=state is None):
            with st.spinner('正在检索资料并生成答案…'):
                try:st.session_state['full_answer']=api('/api/answer',{'question':question,'doc_ids':ids,'task':kinds[task],'method':methods[method],'answer_cache':cached})
                except (urllib.error.URLError,ValueError) as exc:st.error('请求失败：'+str(exc))
        result=st.session_state.get('full_answer')
        if result:
            if result['action']=='ANSWER':
                st.markdown(result['answer'])
                if result['abstain']:st.info('模型判断当前证据不足。')
                timing_label='原生成耗时' if result.get('answer_cache_hit') else '耗时'
                st.caption(f"{timing_label} {result['application_seconds']:.2f} 秒 · {result['model']} · {'答案缓存' if result.get('answer_cache_hit') else '本地模型生成'}")
                coverage=result.get('source_character_coverage',{})
                if coverage:
                    st.caption('实际输入的原文比例：'+'；'.join(titles.get(pid,pid)+f' {value:.0%}' for pid,value in coverage.items()))
                cited=set(result['citations'])
                st.write('引用证据')
                for piece in result['context_bundle']['pieces']:
                    if piece['chunk_id'] in cited:
                        with st.expander(titles.get(piece['doc_id'],piece['doc_id'])+' · '+piece['chunk_id']):st.write(piece['text'])
                with st.expander('全部实际输入证据'):
                    for piece in result['context_bundle']['pieces']:
                        st.caption(piece['doc_id']+' · '+piece['chunk_id']);st.write(piece['text'])
                st.caption('记录编号：'+result['record_id'])
            else:st.warning(result.get('message','请求未完成。'))
    with tabs[1]:
        summary=run/('FINAL_SUMMARY_RESIDENT8K.json' if (run/'FINAL_SUMMARY_RESIDENT8K.json').exists() else 'FINAL_SUMMARY.json')
        if summary.exists():st.json(json.loads(summary.read_text('utf8')))
        answer_path=run/('DEPLOYMENT_FINAL_ANSWERS_RESIDENT8K.jsonl' if (run/'DEPLOYMENT_FINAL_ANSWERS_RESIDENT8K.jsonl').exists() else 'DEPLOYMENT_FINAL_ANSWERS.jsonl')
        sources={'最终部署答案（220题）':answer_path,
                 '基线与4K初版（440条原始记录）':run/'FINAL_GENERATION.jsonl'}
        available={label:path for label,path in sources.items() if path.exists()}
        path=available[st.selectbox('记录来源',list(available))] if available else run/'missing.jsonl'
        if path.exists():
            if st.button('读取并校验记录'):
                try:
                    st.session_state['checked_rows']=read_records(path)
                    st.session_state['checked_path']=str(path)
                except ValueError:st.warning('结果正在写入或校验未通过，请稍后刷新；不显示未完整保存的记录。')
            rows=st.session_state.get('checked_rows',[]) if st.session_state.get('checked_path')==str(path) else []
            if rows:st.caption(f'已校验 {len(rows)} 条真实实验记录；以下为历史回放。')
            if rows:
                index=st.number_input('记录序号',min_value=1,max_value=len(rows),value=1)
                row=rows[int(index)-1]
                st.write(row['question']);st.json({k:row.get(k) for k in ('split','condition','model','parsed','error','elapsed_s','new_model_call','reuse_source_record_id')})
        report=ROOT/'reports/v3/FINISH_REPORT_ZH.md'
        if report.exists():st.markdown(report.read_text('utf8'))
    with tabs[2]:
        st.markdown((ROOT/'LOCAL_RUN_V3.md').read_text('utf8'))


if __name__=='__main__':main()
