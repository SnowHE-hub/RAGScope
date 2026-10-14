"""Evidence-bounded H01-H12 snapshot; pending work stays pending."""
import argparse
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ragscope.indexing.identity import artifact,atomic_json,canonical_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args();run=Path(a.campaign).resolve()
    original_path=PROJECT/'docs/execution/v3/HYPOTHESES.jsonl'
    original=[json.loads(l) for l in original_path.read_text('utf8').splitlines()]
    updates={
        'H01':('PARTIAL_MEASUREMENT_EFFECT_FACTUAL_TRUTH_UNRESOLVED','官方兼容重算无差异；短yes/no参考和篇幅影响文字分数。替代语义代理不能判断数字、否定和引用支持；完整事实指标仍阻断。',['RECOUNT_ALL_OLD.json','S2_PROTOCOL_ANALYSIS.json']),
        'H02':('NO_MISMATCH_IN_FIXED_SMOKE_GENERAL_CAUSE_UNRESOLVED','固定72次direct/pipeline、36对一致，未发现通道/token错配。G1/G3影响随模型变化；不把固定样本反证推广为所有9B场景无包装问题。',['S2_PROTOCOL_ANALYSIS.json']),
        'H03':('CONFIRMED_TASK_INTERPRETATION_LIMIT','strict-grounded空上下文衡量材料依赖行为，不直接衡量已有知识。新增两类简单常识探针三模型可回答，但不能代表知识广度。',['S2_SUPPLEMENT_ANALYSIS.json']),
        'H04':('CONTROLLED_PADDING_EFFECT_MODEL_DEPENDENT','12题×3模型等长干扰保持证据逐字；2B/4B平均文字分下降、9B平均变化小。长度/噪声混杂存在，规模有限、两题证据位置缺失有明确替代。',['S2_SUPPLEMENT_ANALYSIS.json']),
        'H05':('SCOPE_INPUT_LIMIT_DIAGNOSED_CAUSAL_SHARE_UNKNOWN','原E8缺少论文锚定不能全归因于算法。新文档检索明确区分原始问句与可见题名输入，后者不能冒充同任务提升；H_scope最终专项尚待执行。',['S3_DOCUMENT_CANDIDATES_D_SEARCH.json','S3_DOCUMENT_DENSE_CANDIDATES.json']),
        'H06':('OLD_WINDOW_COUNTEREVIDENCE_NEW_NATURAL_RERANK_PENDING','500记录/9519窗口未发现512预算、字符覆盖或分数方向违例；max/top2复用旧实际窗口分数未改变所选覆盖。BGE11/12、新Qwen12/12仅夹具校准，不是自然检索提升。',['S3_OLD_RETRIEVAL_FUNNEL.json','S3_BGE_RELEVANCE_CALIBRATION.json','S3_QWEN_RERANKER_CALIBRATION.json']),
        'H07':('MIXED_EFFECT_D_CONFIRM_PRIMARY_GAIN_UNPROVEN','旧400行core-first效果混合；新D_search40文字提升约6.37pp，但D_confirm100只约1.24pp且论文聚类区间跨零。证据覆盖改善不能替代事实收益，未通过发布门槛。',['S3_CORE_FIRST_OLD_SCREEN.json','S3_SCOPED_GENERATION_D_confirm_ANALYSIS.json']),
        'H08':('OLD_COLLAPSE_CONFIRMED_NEW_TASK_UTILITY_PENDING','原代理标签下学习路由塌缩事实保留；多数类占比变化不能直接称过拟合。新60题实际工作流正在检验，未完成全文总结任务覆盖，不触发新分类器训练。',['S4_WORKFLOW_CONTEXT_MANIFEST.json']),
        'H09':('DISTRIBUTION_SHIFT_DESCRIBED_CAUSAL_MECHANISM_UNRESOLVED','训练短干净证据与部署长检索存在分布差异；旧23新增拒答的独立充分性proxy为YES8/NO12/UNKNOWN3。不能全部判错，尚未通过部署噪声训练因果试验。',['S5_SUPERVISION_DISTRIBUTION_DIAGNOSIS.json','LORA_23_REFUSAL_CONTEXT_PROXY.jsonl']),
        'H10':('SUPERVISION_CHAIN_COUNTEREVIDENCE_EOS_NORMAL_ON_FIXED48','972行数值检查和20实际解码通过。8夹具两步兼容测试更新228张量、loss下降、完整恢复通过；单步warmup零学习率测试失败保留。新基础/adapter48条正常结束，EOS编号不同未造成大量续写。',['S5_SUPERVISION_SMOKE_v2/COMPLETION.json','S5_RUNTIME_INDEPENDENT_ANALYSIS.json']),
        'H11':('LENGTH_AND_DECODE_COST_SUPPORTED_MERGE_GATE_FAILED','新24题输出中位67→171；固定64-token微基准显示未合并adapter解码更慢。三条合并logit探针超过事先容差，首选token相同不足证明函数等价，不能称合并解决延迟。',['S5_RUNTIME_INDEPENDENT_ANALYSIS.json','S5_FIXED64_base.jsonl','S5_FIXED64_unmerged.jsonl','S5_MERGE_LOGIT_GATE.json']),
        'H12':('INCONCLUSIVE_KEEP_V2_FINAL_ACCEPTANCE_PENDING','工程链路验证与开发观察均不能替代一次锁定holdout、支持性/引用和真实在线成本。当前主指标未知，S6及最终交付尚未完成；默认v2，人工NOT_REVIEWED。',['STATUS.json'])}
    if (run/'S4_WORKFLOW_UTILITY_ANALYSIS.json').exists():
        utility=json.loads((run/'S4_WORKFLOW_UTILITY_ANALYSIS.json').read_text('utf8'))
        updates['H08']=('OLD_COLLAPSE_CONFIRMED_NEW_MAPPING_NOT_BETTER',
            f'180次真实工作流生成及240项效用齐全，预标映射相对最佳固定短问答代理差{utility["mapping_minus_best_fixed_proxy"]:+.6f}；没有总结标签且关键点未知，不触发新router训练。事后最佳是不可部署上界，不是分类准确率。',
            ['S4_WORKFLOW_UTILITY_ANALYSIS.json','S4_WORKFLOW_CONTEXT_MANIFEST.json'])
    rows=[];lines=['# 根因假设 H01–H12：证据与未解项','',
        '## 当前结论','',
        '已经确认测量口径限制、训练/部署分布差异以及LoRA输出长度和逐token开销变化。固定探针未发现pipeline通道/token错配；监督链路自检可更新并恢复。完整事实评分、自然新重排、工作流效用和最终holdout仍有受阻或待执行项目，不能宣称v3整体更好。','',
        '本文件是当前审计快照，workflow_finished=false、execution_complete=false、performance_status=INCONCLUSIVE、release_status=KEEP_V2_DEFAULT、human_review_status=NOT_REVIEWED。原假设账本不覆盖，更新保存在独立投影。','']
    for old in original:
        code,note,names=updates[old['id']];evidence=[];gaps=[]
        for name in names:
            path=run/name
            if path.exists():evidence.append(artifact(path))
            else:gaps.append(str(path))
        row={**old,'original_hypothesis_source':artifact(original_path),'current_verdict':code,
            'current_interpretation':note,'current_evidence':evidence,'source_access_gaps':gaps,
            'human_review_status':'NOT_REVIEWED'};rows.append(row)
        lines += [f'## {old["id"]}：{old["hypothesis"]}','',f'结论：`{code}`。{note}','',
            '原反证：'+old['counterevidence'],'',
            '证据：'+('；'.join('`'+e['path']+'`' for e in evidence) if evidence else 'UNKNOWN：无可读当前证据文件。'),'']
        if gaps:lines += ['证据缺口：'+'；'.join('`'+g+'`' for g in gaps),'']
    target=PROJECT/'docs/execution/v3/ROOT_CAUSE_VERDICTS.md';target.write_text('\n'.join(lines)+'\n',encoding='utf8')
    projection=run/'HYPOTHESES_CURRENT_VERDICTS.jsonl';projection.write_text(''.join(canonical_json(r)+'\n' for r in rows),encoding='utf8')
    atomic_json(run/'ROOT_CAUSE_REPORT_SOURCE_NOTES.json',{'report':artifact(target),'projection':artifact(projection),
        'original_hypotheses':artifact(original_path),'human_review_status':'NOT_REVIEWED','report_state':'INTERIM_FINAL_ACCEPTANCE_PENDING'})
    print(str(target));return 0


if __name__=='__main__':raise SystemExit(main())
