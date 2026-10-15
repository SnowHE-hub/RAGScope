"""Build concise Chinese delivery from verified current artifacts, without inference."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ragscope.experiments.runtime import configure
from ragscope.experiments.ledger import read_records
from ragscope.indexing.identity import artifact,atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);a=p.parse_args()
    configure(cpu_only=True);run=Path(a.run)
    def load(name):return json.loads((run/name).read_text('utf8'))
    score=load('FINAL_SUMMARY_RESIDENT8K.json');online=load('FINAL_ONLINE_SUMMARY_RESIDENT8K.json')
    glob=load('FINAL_GLOBAL_SUMMARY.json');target=load('TARGET_509_COVERAGE.json')
    api=load('API_REAL_ACCEPTANCE.json');ui=load('UI_REAL_COMPONENT_INTEGRATION.json')
    offline=load('UI_OFFLINE_FINAL.json');release=load('RESOURCE_RELEASE_FINAL.json')
    assert target['target_papers']==509 and not target['target_missing'] and target['total_papers']==1585
    assert api['status']=='PASS' and api['case_n']==9 and ui['exception_n']==0
    assert ui['status']=='PASS_REAL_STREAMLIT_API_INTEGRATION' and offline['verified_history_rows']==220
    assert release['status']=='PASS' and release['gpu_lock_free'] and not release['ports_listening']
    tree=ET.parse(run/'FINAL_PYTEST_DELIVERY.xml');suites=list(tree.getroot().iter('testsuite'))
    tests=sum(int(s.attrib.get('tests',0)) for s in suites)
    failures=sum(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0)) for s in suites)
    assert tests==82 and not failures
    records=read_records(run/'FINAL_SCORES_RESIDENT8K.jsonl');assert len(records)==440
    grouped=defaultdict(list)
    for row in records:grouped[row['split']+':'+row['condition']].append(row)
    for group,rows in grouped.items():
        assert len(rows)==(200 if group.startswith('H_QA:') else 20)
        for metric in ('token_f1','exact_match','rouge_l','semantic_cosine','evidence_full_coverage'):
            values=[r['metrics'][metric] for r in rows if r['metrics'][metric] is not None]
            expected=score['groups'][group][metric]
            assert (not values and expected is None) or abs(sum(values)/len(values)-expected)<1e-12
    answers=read_records(run/'DEPLOYMENT_FINAL_ANSWERS_RESIDENT8K.jsonl');assert len(answers)==220
    assert all(r['real_model'] and r['actual_sent_payload']['options']['num_ctx']==8192 for r in answers)
    assert all(r['backend']['token_count_match'] is True for r in answers)
    online_rows=read_records(run/'FINAL_ONLINE_RESIDENT8K.jsonl');assert len(online_rows)==60
    assert all(not r['answer_cache'] and not r['query_embedding_cache'] and r['success'] for r in online_rows)
    assert all(r['actual_sent_payload']['options']['num_ctx']==8192 for r in online_rows if r['condition']=='FINAL')
    hot_loads=[r['backend'].get('load_duration',0)/1e9 for r in online_rows if r['condition']=='FINAL']
    assert max(hot_loads)<1,'Repeated model reload remains'
    app_errors=[];app_n=0
    for path in run.glob('APPLICATION_*.jsonl'):
        for row in read_records(path):
            app_n+=1
            if not row['success']:app_errors.append({'record_id':row['record_id'],'source':str(path),'error':row['error']})
    atomic_json(run/'APPLICATION_RETAINED_FAILURES.json',{'model_request_n':app_n,'retained_failure_n':len(app_errors),'failures':app_errors,
        'fixed_case':'Multi-paper citation overflow; cap4 citations/1600 answer characters. Same9 API cases passed after fix.'})
    q0,q1=score['groups']['H_QA:B0'],score['groups']['H_QA:FINAL']
    t0,t1=score['groups']['H_tasks:B0'],score['groups']['H_tasks:FINAL']
    o0,o1=online['groups']['B0'],online['groups']['FINAL']
    g0,g1=glob['groups']['B0:hybrid'],glob['groups']['FINAL:hybrid']
    paired={}
    for split in ('H_QA','H_tasks'):
        base={r['query_id']:r for r in grouped[split+':B0']};final={r['query_id']:r for r in grouped[split+':FINAL']}
        assert set(base)==set(final)
        diffs=[final[q]['metrics']['token_f1']-base[q]['metrics']['token_f1'] for q in base if base[q]['metrics']['token_f1'] is not None]
        paired[split]={'applicable_n':len(diffs),'improved':sum(x>1e-12 for x in diffs),'worse':sum(x < -1e-12 for x in diffs),'tied':sum(abs(x)<=1e-12 for x in diffs)}
    atomic_json(run/'PAIRED_FINAL_CHANGES.json',paired)
    report=PROJECT/'reports/v3/FINISH_REPORT_ZH.md'
    report.parent.mkdir(parents=True,exist_ok=True)
    body=f'''# RAGScope v3 本地交付结果（2026-10-15）

已完成全库索引、可运行工作台、正式同题对照、质量评价和资源释放。收益集中在全库证据覆盖及总结/比较；普通问答有小幅描述性提升，不能宣称全面优于基线。浏览器视觉验收被权限策略拒绝，改用真实API与Streamlit组件集成验收，未绕过限制。

## 实际交付

- 原509篇目标缺失0篇；完整扩展库1585篇、19673块，文档层与块层BM25/Dense/Hybrid均可用。复用13045个有效向量，新增6628个，未下载新权重。
- 默认已有Qwen3.5-2B Q4_K_M及Qwen3-Embedding-0.6B Q8_0。生成后端统一驻留8K；普通问答保留2K证据/512输出，总结比较最多6K证据/768输出。支持全库、指定论文和多篇范围，显示真实引用与原文覆盖。
- H_QA200与H_tasks20均完成基线/最终同题对照，440条最终评分，0生成失败。原440条初版、24条长上下文新增、196条统一驻留新增均保留；最终220条中24条严格相同请求复用，不冒充新调用。
- 全库检索1200条、在线60条加2预热完成；82项回归通过，真实API9场景通过，真实组件完成问答/引用显示，离线可校验220条历史。

## 同题结果

|指标|V2策略迁移基线|最终配置|
|---|---:|---:|
|H_QA文字F1（170可答）|{q0['token_f1']:.4f}|{q1['token_f1']:.4f}|
|H_QA语义余弦|{q0['semantic_cosine']:.4f}|{q1['semantic_cosine']:.4f}|
|H_tasks文字F1（20题）|{t0['token_f1']:.4f}|{t1['token_f1']:.4f}|
|H_tasks语义余弦|{t0['semantic_cosine']:.4f}|{t1['semantic_cosine']:.4f}|
|任务标注关键点在实际上下文中可见|{t0['source_keypoint_context_coverage']:.2%}|{t1['source_keypoint_context_coverage']:.2%}|
|全库Hybrid完整证据覆盖（170适用）|{g0['full_evidence_coverage']:.2%}|{g1['full_evidence_coverage']:.2%}|
|在线P50|{o0['p50_s']:.3f}秒|{o1['p50_s']:.3f}秒|
|在线P95|{o0['p95_s']:.3f}秒|{o1['p95_s']:.3f}秒|

4K/8K交替导致约7.9秒反复重载，已通过统一驻留消除；问题版本P95为10.132秒，修复后{ o1['p95_s']:.3f}秒，正式热请求最大模型加载{max(hot_loads):.3f}秒。相对基线并非全面提速，复杂请求读取更多证据，P95仍较高。任务生成P50约{t0['p50_seconds']:.2f}→{t1['p50_seconds']:.2f}秒，输出也更长。

全库Hybrid文档命中@10由40.5%降至38%，纯Dense由32.5%降至27%；未隐藏这些退步。默认按开发集冻结的Hybrid保留，未按最终成绩另挑方法。H_QA的配对F1变化：{paired['H_QA']['improved']}改善、{paired['H_QA']['worse']}退步、{paired['H_QA']['tied']}持平；这是描述性验证，不是显著性证明。

## 评价与边界

EM/F1/ROUGE-L、Qwen embedding参考余弦、实体/数字/否定/比较字面检查、引用可见性和证据覆盖均逐题保存。语义余弦不是BERTScore或事实正确率；引用ID合法由schema约束，也不证明支持关系。事实真值未知保留UNKNOWN，不删失败或未知分母。

不可答30题的正确拒答仅3/30，两方案均10%；yes/no参考极性冲突提示均9条，仍需复核。任务最终有{t1['number_flags']}条数值未在上下文字面出现的提示，不能直接等同事实错误。人工事实审核未完成。LoRA和学习路由不默认启用。

基线是V2固定Hybrid/neighbor/G0迁移到同一1585篇和GPU embedding空间，不能冒称原CPU部署速度。H_QA是新200题，与旧v2测试集不同。H_tasks是AI构造且原文引句校验的20题，非官方或人审真值；本轮为明确授权的最终验证，不声称操作者盲测。

应用验收曾保留{len(app_errors)}条JSON截断失败，限制多篇回答引用后同9场景通过。浏览器安全策略拒绝访问127.0.0.1:8502；仅声明组件/API集成通过，未伪造浏览器截图或视觉验收。

## 使用与证据

准确启动/退出见[LOCAL_RUN_V3.md]({(PROJECT/'LOCAL_RUN_V3.md').as_posix()})。模型、环境、缓存、数据、索引和结果均复用D盘；未push、上传、付费、改驱动或重启WSL。自有模型与服务已释放，原训练保持USER_MANUAL_RESUME，由用户手动续跑。

运行目录：`{run.as_posix()}`。主要来源：FINAL_SUMMARY_RESIDENT8K.json、FINAL_SCORES_RESIDENT8K.jsonl、DEPLOYMENT_FINAL_ANSWERS_RESIDENT8K.jsonl、FINAL_GLOBAL_SUMMARY.json、FINAL_ONLINE_SUMMARY_RESIDENT8K.json、API_REAL_ACCEPTANCE.json、UI_REAL_COMPONENT_INTEGRATION.json、FINAL_PYTEST_DELIVERY.xml、RESOURCE_RELEASE_FINAL.json。旧query缓存24条已按原后端SHA逐位恢复，新在线编码不写旧缓存，恢复前版本亦保留。所有旧报告、断点和原始结果保留。
'''
    report.write_text(body,encoding='utf8')
    proofs={name:artifact(run/name) for name in ('FULL_DENSE_INDEX.json','FULL_CORPUS_VALIDATION.json','TARGET_509_COVERAGE.json',
        'FINAL_SUMMARY_RESIDENT8K.json','FINAL_SCORES_RESIDENT8K.jsonl','DEPLOYMENT_ANSWERS_COMPLETION_RESIDENT8K.json',
        'FINAL_GLOBAL_SUMMARY.json','FINAL_ONLINE_SUMMARY_RESIDENT8K.json','API_REAL_ACCEPTANCE.json',
        'UI_REAL_COMPONENT_INTEGRATION.json','UI_OFFLINE_FINAL.json','FINAL_PYTEST_DELIVERY.xml','RESOURCE_RELEASE_FINAL.json','QUERY_CACHE_RESTORATION.json')}
    for name in ('DEPLOYMENT_CONFIGURATION_RESIDENT8K.json','APPLICATION_REVISION.json','APPLICATION_RETAINED_FAILURES.json','BUDGET_CONTINUATION.json'):
        proofs[name]=artifact(run/name)
    requirements=[('全部目标论文检索','PASS'),('可运行工作台','PASS_COMPONENT_API_BROWSER_RESTRICTED'),('经实测默认配置','PASS'),
        ('V2与最终同题对照','PASS'),('H_QA200与H_tasks20','PASS'),('可运行质量评价','PASS_WITH_UNKNOWN_TRUTH_RETAINED'),
        ('代码测试与运行文档','PASS'),('简洁中文结论','PASS')]
    acceptance={'execution_complete':True,'requirements':[{'requirement':k,'status':v} for k,v in requirements],
        'required_incomplete':[],'browser_visual_verified':False,'browser_limitation':'Automatic approval denied browser access; real component/API alternative completed without bypass',
        'performance_status':'MEASURED_MIXED_GAINS_NOT_UNIVERSAL','human_review':'NOT_REVIEWED','proofs':proofs,
        'report':artifact(report),'paired':paired,'original_training':'USER_MANUAL_RESUME'}
    atomic_json(run/'FINISH_ACCEPTANCE.json',acceptance)
    atomic_json(PROJECT/'reports/v3/FINISH_ACCEPTANCE.json',acceptance)
    config=load('DEPLOYMENT_CONFIGURATION_RESIDENT8K.json')
    atomic_json(PROJECT/'configs/v3/default.json',{'run':str(run),'source':load('FULL_DENSE_INDEX.json')['source_campaign'],
        'release_status':'VERIFIED','verification':'REAL_API_AND_STREAMLIT_COMPONENTS','browser_visual_verified':False,
        'default_model':'qwen_2b','backend_context':8192,'default_method':'hybrid','host':'127.0.0.1','api_port':11437,'ui_port':8502,
        'deployment_configuration':artifact(run/'DEPLOYMENT_CONFIGURATION_RESIDENT8K.json'),'application_revision':artifact(run/'APPLICATION_REVISION.json')})
    print(json.dumps({'status':'DELIVERY_VERIFIED_WITH_BROWSER_LIMITATION','report':str(report),'requirements':len(requirements),'tests':tests,'paired':paired},ensure_ascii=False))


if __name__=='__main__':main()
