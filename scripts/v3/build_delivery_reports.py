"""Source-backed final readout; blocked holdout and insufficient task cases stay explicit."""
import argparse
import json
from pathlib import Path
import statistics
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ragscope.experiments.ledger import read_records
from ragscope.indexing.identity import artifact,atomic_json,canonical_json
from ragscope.v3.budget import Budget


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args();run=Path(a.campaign).resolve()
    if run.parent!=Path('D:/RAGScope_Runtime/runs').resolve():raise ValueError('Wrong campaign')
    reports=PROJECT/'docs/execution/v3'
    def load(name):return json.loads((run/name).read_text('utf8'))
    word=load('S3_SCOPED_GENERATION_D_confirm_ANALYSIS.json');claim=load('S3_D_CONFIRM_CLAIM_GATE.json')
    online=load('S6_ONLINE_COST_ANALYSIS.json');gate=load('CANDIDATE_LOCK.json')
    scope=load('S6_H_SCOPE_COMPLETION.json')
    application_name='S6_APPLICATION_CACHE_VERIFICATION_final_scope_guard.json'
    application=load(application_name)
    if online['formal_n']!=180 or not scope['status'].startswith('PASS') or not application['status'].startswith('PASS'):
        raise ValueError('Real independent cost/scope/application evidence required')
    sources={name:artifact(run/name) for name in ['S3_SCOPED_GENERATION_D_confirm_ANALYSIS.json','S3_D_CONFIRM_CLAIM_GATE.json',
        'S6_ONLINE_COST_ANALYSIS.json','CANDIDATE_LOCK.json','S6_H_SCOPE_COMPLETION.json',application_name]}
    scorecard={'status':'INCONCLUSIVE_KEEP_V2_DEFAULT','developer_confirmation100':{
        'word_metrics':word['groups'],'word_delta':word['word_delta'],'word_paper_ci95':word['word_delta_paper_cluster_ci95'],
        'claim_support_citation':claim['groups'],'claim_delta':claim['claim_f1_delta_mean'],
        'claim_uncertainty_envelope95':claim['claim_f1_delta_uncertainty_and_paper_bootstrap95']},
        'uncached_online30x3':online,'scope40_synthetic':scope,
        'application_cache_separate':{'status':application['status'],'checks':application['checks'],'actual_answers_all_versions':2,
            'actual_final_version_answers':1,'cache_hits_all_versions':2},
        'H_QA200':{'status':gate['H_QA_status'],'actual_answers':0,'quality':None,'support':None,'citation':None,'abstention':None},
        'H_tasks20':{'status':gate['H_tasks_status'],'actual_final_answers':0},
        'C2':'NOT_RUN_GATE_NO_ADMITTED_ROUTER_OR_ADAPTER','release_status':'KEEP_V2_DEFAULT',
        'human_review_status':'NOT_REVIEWED','execution_complete':False,'sources':sources,
        'interpretation':'Development comparisons only; no unseen quality or efficiency gain confirmed. UNKNOWN retained; no arbitrary combined score.'}
    atomic_json(reports/'QUALITY_SCORECARD.json',scorecard)
    names=['B0_policy_neighbor_bridge','C1_policy_core_first_bridge'];b,c=[word['groups'][n] for n in names]
    ob,oc=[online['groups'][n] for n in names]
    lines=['# v3质量与成本记分卡','',
        '结论：没有候选通过开发入围门槛，默认保留v2。以下开发结果不是新holdout发布证明；人工审核NOT_REVIEWED。','',
        '## 同题开发确认：100题、50篇论文','',
        '|指标|B0保留策略|C1 core-first|','|---|---:|---:|',
        f'|可答文字F1（92题）|{b["legacy_token_f1"]:.6f}|{c["legacy_token_f1"]:.6f}|',
        f'|可答题拒答数|{b["answerable_abstain_n"]}|{c["answerable_abstain_n"]}|',
        f'|格式/调用失败|{b["failure_n"]}|{c["failure_n"]}|',
        '|主要claim F1均值|UNKNOWN|UNKNOWN|','|支持性均值|UNKNOWN|UNKNOWN|','|引用完整性均值|UNKNOWN|UNKNOWN|','',
        f'文字差值{word["word_delta"]:+.6f}，2000次论文聚类95%区间{word["word_delta_paper_cluster_ci95"]}，跨零。',
        '主要claim F1只14/13个可答问题有确定点值；其余未知不丢弃，完整上下界见JSON。不能把已知子集当总体。','',
        '## 实际在线：30题×每条件3次','',
        '|指标|B0保留策略|C1 core-first|','|---|---:|---:|',
        '|实际正式请求|90|90|',f'|P50秒|{ob["online_p50_seconds"]:.4f}|{oc["online_p50_seconds"]:.4f}|',
        f'|P95秒|{ob["online_p95_seconds"]:.4f}|{oc["online_p95_seconds"]:.4f}|',
        '|调用/格式失败|0|0|','',
        '每次真实重新编码问题、检索、打包、生成、序列化。答案和embedding应用缓存关闭；6次独立预热不并入180正式请求。',
        '编码器与一个2B生成器同驻留GPU，所有请求串行。两条件均使用已核验的GPU embedding桥接，不能冒充原CPU部署成本复现。',
        f'P50比值{gate["declined_candidate"]["online_p50_ratio"]:.4f}、P95比值{gate["declined_candidate"]["online_p95_ratio"]:.4f}，没有达到效率收益。',
        '完整论文聚类区间、可答/不可答及失败分母见JSON。缓存命中应用测试另列，不用它宣称模型变快；非流式TTFT为N/A。','',
        '## 范围与正式留出','',
        'H_scope40真实范围策略测试通过：20指定论文不越界，20缺对象澄清；合成规则案例、非事实QA成绩。',
        'H_QA200/H_tasks20最终配对为BLOCKED：任务书10.1禁止未入围候选进入正式比较，没有创建最终gold读取授权。',
        '必须的未见题质量、支持性、引用及拒答比较尚未完成，所以execution_complete=false；不把开发比较或仅基线跑分替代它。','',
        '## 可复核来源','']
    lines.extend(f'- `{ref["path"]}` SHA256 `{ref["sha256"]}`' for ref in sources.values())
    (reports/'QUALITY_SCORECARD.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    # Human package includes original responses and exact source coordinates; private aliases map only to conditions.
    package=read_records(run/'HUMAN_REVIEW_PACKAGE.jsonl');mapping=read_records(run/'HUMAN_REVIEW_PRIVATE_MAPPING.jsonl')
    details=word['paired_details'];deltas={r['query_id']:r['delta'] for r in details}
    # Export script assigns sorted query IDs to case IDs, so resolve through the original answer hashes instead of guessing order.
    complete=load('S3_SCOPED_GENERATION_D_confirm_COMPLETION.json');original=read_records(Path(complete['records']['path']))
    byhash={r['record_hash']:r for r in original};bycase={r['case_id']:r for r in package};cases=[]
    for private in mapping:
        aliases=private['private_mapping']
        if set(aliases)!={'A','B'}:raise ValueError('Invalid frozen blind alias mapping')
        matched={v['condition']:byhash[v['source_record_hash']] for v in aliases.values()}
        if set(matched)!=set(names):continue
        qid=matched[names[0]]['question_id']
        if qid not in deltas:continue
        cases.append({'query_id':qid,'delta':deltas[qid],'package':bycase[private['case_id']],
                      'aliases':{v['condition']:k for k,v in aliases.items()}})
    case_md=['# 真实前后案例','',
        '原答案、引用和原文均来自已保存的开发确认配对；按文字差值选例用于说明，不改变完整实验分母，不证明事实质量变化。人工NOT_REVIEWED。','']
    selected=[]
    for kind,subset in [('文字改善',sorted([r for r in cases if r['delta']>0],key=lambda r:-r['delta'])[:3]),
                        ('文字退步',sorted([r for r in cases if r['delta']<0],key=lambda r:r['delta'])[:3])]:
        case_md.extend(['## '+kind,''])
        for item in subset:
            selected.append(item['query_id']);case=item['package'];variants={v['alias']:v for v in case['variants']}
            case_md.extend(['### '+item['query_id'],'',case['question'],'',f'文字差值：{item["delta"]:+.6f}。',
                '这是参考表达匹配变化；上下文组织同时变化，不能直接认定答案更正确。',''])
            for name in names:
                v=variants[item['aliases'][name]]
                case_md.extend([name+'：','',v['answer'] or '无合法答案','',f'实际引用：{v["citations"]}',''])
                for piece in v['context']:
                    if piece['citation_id'] in (v['citations'] or []):
                        case_md.extend([f'原文 `{piece["citation_id"]}`，源坐标{piece["canonical_start"]}–{piece["canonical_end"]}（节选，完整片段见审核包）：','',piece['text'][:600].rstrip(),'']);break
            case_md.extend(['完整原文与A/B回答：`'+str(run/'HUMAN_REVIEW_PACKAGE.jsonl')+'`，'+case['case_id'],''])
    case_md.extend(['## 比较与总结样本缺口','',
        '独立来源开发诊断只有2条总结和1条比较，没有达到每类至少3条真实配对案例的要求。',
        '该交付项为BLOCKED_VALID_SOURCE_TASK_COVERAGE；原整篇教师6请求、17失败分支及兼容后9失败任务保留，不生成虚构补例。',
        '已存在的全部3条前后结果及证据位置见TASK_KEYPOINT_DIAGNOSIS.md和原6条答案journal。它们未证明任务工作流泛化收益。',''])
    if len(selected)!=6:raise ValueError('Could not resolve real six before/after examples; preserve source alias schema and repair once')
    (reports/'BEFORE_AFTER_CASES.md').write_text('\n'.join(case_md),encoding='utf8')
    atomic_json(run/'BEFORE_AFTER_CASES_RECEIPT.json',{'status':'PARTIAL_CASES_TASK_CLASS_MINIMUM_BLOCKED',
        'actual_word_improvement_cases':3,'actual_word_regression_cases':3,'selected_query_ids':selected,
        'source':artifact(run/'HUMAN_REVIEW_PACKAGE.jsonl'),'comparison_valid_pairs':1,'summary_valid_pairs':2,
        'human_review_status':'NOT_REVIEWED','report':artifact(reports/'BEFORE_AFTER_CASES.md')})
    budget=Budget(run).snapshot()
    summary=['# RAGScope v3结论','',
        '**本轮没有证明v3整体优于v2，默认保持v2。** 已完成真实诊断、开发对照、180次在线实验、40次范围专项和工作台实连；必需的正式留出配对及部分事实评分受阻，execution_complete=false。','',
        '文字F1不是答对题比例。旧5720条重算与原文字算法无差异；补充语义信号仍不足以替代事实支持判断。主要NLI事实评分有大量UNKNOWN，不能换裁判或挑已知子集宣称提升。','',
        '9B在固定模板与pipeline对照中未发现通道/token错配；提示协议影响随模型变化。现有证据不支持把规模倒挂认定为某个已确认模板bug。','',
        '重排在确认集反而丢失打包证据，拒绝部署。core-first在100题开发确认上文字F1约+0.01235，区间跨零；在线P50没有改善，P95更慢，未满足入围门槛。','',
        '任务类型路由仍值得研究，但现有比较/总结标签覆盖不足，分类映射没有超过最佳固定工作流，因此没有继续训练新router。LoRA存在新增拒答、训练/部署分布差异及更长、更慢的输出；监督链路可更新与恢复，固定样本未显示无限续写。合并数值门槛失败，不增加正式训练轮数。','',
        '工程成果包括范围明确时跳过文档定位、缺对象澄清、真实全库BM25旁路、核心优先打包、逐题不可变身份/预算、真实未缓存在线计时及本地诊断工作台。全库块Dense只完成部分向量，不能作为完整索引使用。','',
        '无候选满足任务书10.1开发门槛，H_QA200仍封存，没有跑未入围配置的正式配对；不能用开发成绩或基线单独成绩替代它。审核包已导出，人工状态NOT_REVIEWED。','',
        f'当前预算：{budget["requests_reserved"]}/6000生成请求，已扣除或预留{budget["output_tokens_charged_or_reserved"]}/4000000输出token；编码/NLI另记账。全部失败与预热保留。','',
        '启动与回退见项目根目录LOCAL_RUN_V3.md，逐项证据见QUALITY_SCORECARD.json、ACCEPTANCE_V3.json。训练恢复策略USER_MANUAL_RESUME，只释放本项目资源，不自动恢复原训练或定时器。','']
    (reports/'EXECUTIVE_SUMMARY_ZH.md').write_text('\n'.join(summary),encoding='utf8')
    atomic_json(run/'DELIVERY_REPORT_BUILD_RECEIPT.json',{'status':'PASS_SOURCE_BACKED_READOUT_BUILD','runner':artifact(__file__),
        'sources':sources,'outputs':[artifact(reports/f) for f in ['QUALITY_SCORECARD.json','QUALITY_SCORECARD.md','BEFORE_AFTER_CASES.md','EXECUTIVE_SUMMARY_ZH.md']],
        'human_review_status':'NOT_REVIEWED','new_generation_requests':0})
    print(canonical_json({'status':'PASS_SOURCE_BACKED_READOUT_BUILD','budget':budget}));return 0


if __name__=='__main__':raise SystemExit(main())
