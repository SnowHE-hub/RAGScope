# 任务工作流与路由消融

## 当前类别映射没有超过最佳固定工作流，不训练新router

60题×3工作流的180次实际生成已完成，9次独立预热全部计入预算。预标类别映射相对最佳固定W_local的混合短问答代理效用差为 -0.012080。依据阶段门槛，本轮不训练新线性/MLP/LSTM/Laya分类器；该负结果不是“问题分类永远无用”的证明。

## 控制条件与类别不足

所有工作流使用同一已验证2B Q4生成器、2048参考上下文预算、512输出预算和G0 JSON schema。题目在结果之前按可见问题/用户scope规则选定；事实28、解释22、比较5、其他5，没有总结标签。规则是AI/工程代理，未由操作者人工确认。52题可回答、8题不可回答。

W_local单次scoped hybrid检索、无扩展紧凑证据；W_compare最多两个明确操作数子查询，用既有BM25权重在指定论文内各检索并平衡预算，不清楚双方时保留原问题并标记fallback；W_summary从原文章节选择最多三个证据组，明确报告未覆盖章节，不宣称全篇覆盖。工作流指令是实验变量。

## 完整结果保留失败，澄清动作不伪造新请求

|工作流|实际新正式生成|可答文字F1|失败|冻结生成P50毫秒|
|---|---:|---:|---:|---:|
|W_clarify|0|0.363394|0|1577.139|
|W_compare|60|0.340127|0|1699.794|
|W_local|60|0.363394|0|1577.139|
|W_summary|60|0.360794|0|1481.815|

全部60题已指定论文，所以W_clarify依法委托W_local，效用向量引用同一实际响应，不额外发60次模型请求；其表中时间也不是独立实测。真正缺对象的澄清行为在独立范围专项验证。

上表成本仅为冻结证据后的实际生成，不是包括检索、重排、模型冷启动的在线E2E。文字F1不衡量事实支持或总结要点完整性；三路径都无格式失败也不代表答案正确。

## 三种参考策略不能混为分类准确率

预标映射：fact/explain/other→W_local，compare→W_compare，summarize→W_summary；在看到结果之前冻结。最佳固定工作流：同60题短问答代理效用最高的W_local。事后最佳路径：每题从三个实际结果挑最高代理分，仅是不可部署的hindsight上界。

事后上界的混合代理效用均值为 0.419707。该效用把可答文字F1与不可答正确拒答0/1用于开发选择，不能称整体事实准确率，也不能推断分类器能达到该上界。每题四条完整效用项、原记录hash与证据覆盖均保存。

## 停止条件和未完成的任务覆盖

当前预标策略未超过最佳固定策略，同时比较/总结关键点、claim/support/citation代理仍未知；分类训练为 NOT_RUN_GATE_PRIMARY_TASK_UTILITY_UNVERIFIED。没有通过效用门槛就不增加复杂模型或重新学习“最快检索器”标签。

在没有总结标签的短QA上执行W_summary不能验证全文总结。比较类仅5题，独立关键点标注及自然任务验证仍待补充；不把“框架支持”写成已完成，不将功能测试当质量收益。H_tasks与H_scope仍封存于最终验证协议，不读它们来调参。

手动工作流是可审计实验动作，不是通过发布门槛的自动产品配置。默认保持v2；最终online质量/成本、一次holdout、UI验收及人工审核状态尚需独立完成。

## 复核来源与命令

实际生成：`scripts/v3/run_workflow_generation.py --campaign <campaign> --resume`；独立重算：`scripts/v3/audit_workflow_utility.py --campaign <campaign>`；报告：`scripts/v3/report_workflow_diagnostics.py --campaign <campaign>`。resume不会重发已完成且身份一致的180条。

- `D:\RAGScope_Runtime\runs\v3_20261015_014506\S4_WORKFLOW_UTILITY_ANALYSIS.json` SHA256 `69fcbf41df9fb32db3d328575b3fe9f5cedde5f67863fb5379b396b003056887`
- `D:\RAGScope_Runtime\runs\v3_20261015_014506\S4_WORKFLOW_UTILITY_VECTORS.jsonl` SHA256 `e2e0dcb97e12d98d4984a3fe32fc8d0df6fe4d65c40a20dcaa2a1540d7f6e5e7`

