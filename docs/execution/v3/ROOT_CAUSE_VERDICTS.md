# 根因假设 H01–H12：证据与未解项

## 当前结论

已经确认测量口径限制、训练/部署分布差异以及LoRA输出长度和逐token开销变化。固定探针未发现pipeline通道/token错配；监督链路自检可更新并恢复。完整事实评分、自然新重排、工作流效用和最终holdout仍有受阻或待执行项目，不能宣称v3整体更好。

本文件是当前审计快照，workflow_finished=false、execution_complete=false、performance_status=INCONCLUSIVE、release_status=KEEP_V2_DEFAULT、human_review_status=NOT_REVIEWED。原假设账本不覆盖，更新保存在独立投影。

## H01：文字代理受到答案篇幅与短参考影响

结论：`PARTIAL_MEASUREMENT_EFFECT_FACTUAL_TRUTH_UNRESOLVED`。官方兼容重算无差异；短yes/no参考和篇幅影响文字分数。替代语义代理不能判断数字、否定和引用支持；完整事实指标仍阻断。

原反证：官方兼容重算5720条与legacy差异0，未发现多参考实现bug；整体替代语义代理也下降但短参考同样受限

证据：`D:\RAGScope_Runtime\runs\v3_20261015_014506\RECOUNT_ALL_OLD.json`；`D:\RAGScope_Runtime\runs\v3_20261015_014506\S2_PROTOCOL_ANALYSIS.json`

## H02：9B模板、通道、stop或schema可能不适配

结论：`NO_MISMATCH_IN_FIXED_SMOKE_GENERAL_CAUSE_UNRESOLVED`。固定72次direct/pipeline、36对一致，未发现通道/token错配。G1/G3影响随模型变化；不把固定样本反证推广为所有9B场景无包装问题。

原反证：同上下文仍有部分类型9B文字表现更好；参数量本身不是因果证据

证据：`D:\RAGScope_Runtime\runs\v3_20261015_014506\S2_PROTOCOL_ANALYSIS.json`

## H03：严格材料限制的NoContext不能衡量已有知识

结论：`CONFIRMED_TASK_INTERPRETATION_LIMIT`。strict-grounded空上下文衡量材料依赖行为，不直接衡量已有知识。新增两类简单常识探针三模型可回答，但不能代表知识广度。

原反证：schema只约束空引用列表，不强制abstain=true

证据：`D:\RAGScope_Runtime\runs\v3_20261015_014506\S2_SUPPLEMENT_ANALYSIS.json`

## H04：Oracle空材料可能泄露不可答状态，干净短上下文混杂检索收益

结论：`CONTROLLED_PADDING_EFFECT_MODEL_DEPENDENT`。12题×3模型等长干扰保持证据逐字；2B/4B平均文字分下降、9B平均变化小。长度/噪声混杂存在，规模有限、两题证据位置缺失有明确替代。

原反证：标注证据不是完整事实真值；尚未完成等长噪声对照

证据：`D:\RAGScope_Runtime\runs\v3_20261015_014506\S2_SUPPLEMENT_ANALYSIS.json`

## H05：部分全库题没有可唯一确定的论文对象

结论：`SCOPE_INPUT_LIMIT_DIAGNOSED_CAUSAL_SHARE_UNKNOWN`。原E8缺少论文锚定不能全归因于算法。新文档检索明确区分原始问句与可见题名输入，后者不能冒充同任务提升；H_scope最终专项尚待执行。

原反证：尚不能将81个未找齐证据题全部归因于范围问题

证据：`D:\RAGScope_Runtime\runs\v3_20261015_014506\S3_DOCUMENT_CANDIDATES_D_SEARCH.json`；`D:\RAGScope_Runtime\runs\v3_20261015_014506\S3_DOCUMENT_DENSE_CANDIDATES.json`

## H06：重排窗口聚合、排序或打包可能移走正确证据

结论：`OLD_WINDOW_COUNTEREVIDENCE_NEW_NATURAL_RERANK_PENDING`。500记录/9519窗口未发现512预算、字符覆盖或分数方向违例；max/top2复用旧实际窗口分数未改变所选覆盖。BGE11/12、新Qwen12/12仅夹具校准，不是自然检索提升。

原反证：原BGE已有512窗口，不能断言全被截断

证据：`D:\RAGScope_Runtime\runs\v3_20261015_014506\S3_OLD_RETRIEVAL_FUNNEL.json`；`D:\RAGScope_Runtime\runs\v3_20261015_014506\S3_BGE_RELEVANCE_CALIBRATION.json`；`D:\RAGScope_Runtime\runs\v3_20261015_014506\S3_QWEN_RERANKER_CALIBRATION.json`

## H07：邻块先占预算可能挤掉核心证据

结论：`MIXED_EFFECT_D_CONFIRM_PRIMARY_GAIN_UNPROVEN`。旧400行core-first效果混合；新D_search40文字提升约6.37pp，但D_confirm100只约1.24pp且论文聚类区间跨零。证据覆盖改善不能替代事实收益，未通过发布门槛。

原反证：改善8题退步7题，净一题不能概括全部；未定位每次预算丢弃

证据：`D:\RAGScope_Runtime\runs\v3_20261015_014506\S3_CORE_FIRST_OLD_SCREEN.json`；`D:\RAGScope_Runtime\runs\v3_20261015_014506\S3_SCOPED_GENERATION_D_confirm_ANALYSIS.json`

## H08：文字/成本标签、设备条件与多数类影响路由塌缩

结论：`OLD_COLLAPSE_CONFIRMED_NEW_MAPPING_NOT_BETTER`。180次真实工作流生成及240项效用齐全，预标映射相对最佳固定短问答代理差-0.012080；没有总结标签且关键点未知，不触发新router训练。事后最佳是不可部署上界，不是分类准确率。

原反证：MLP/LSTM训练dev准确率差恰与多数类占比变化相同，不足证明过拟合；换任务尚未验证收益

证据：`D:\RAGScope_Runtime\runs\v3_20261015_014506\S4_WORKFLOW_UTILITY_ANALYSIS.json`；`D:\RAGScope_Runtime\runs\v3_20261015_014506\S4_WORKFLOW_CONTEXT_MANIFEST.json`

## H09：干净短Oracle训练与真实检索分布不一致引起拒答偏移

结论：`DISTRIBUTION_SHIFT_DESCRIBED_CAUSAL_MECHANISM_UNRESOLVED`。训练短干净证据与部署长检索存在分布差异；旧23新增拒答的独立充分性proxy为YES8/NO12/UNKNOWN3。不能全部判错，尚未通过部署噪声训练因果试验。

原反证：23新增拒答尚未按实际材料充分性判定，不能自动称错误拒答

证据：`D:\RAGScope_Runtime\runs\v3_20261015_014506\S5_SUPERVISION_DISTRIBUTION_DIAGNOSIS.json`；`D:\RAGScope_Runtime\runs\v3_20261015_014506\LORA_23_REFUSAL_CONTEXT_PROXY.jsonl`

## H10：监督mask、EOS、引用或截断可能存在缺陷

结论：`SUPERVISION_CHAIN_COUNTEREVIDENCE_EOS_NORMAL_ON_FIXED48`。972行数值检查和20实际解码通过。8夹具两步兼容测试更新228张量、loss下降、完整恢复通过；单步warmup零学习率测试失败保留。新基础/adapter48条正常结束，EOS编号不同未造成大量续写。

原反证：20样本数值mask检查通过且含tokenizer EOS，不能称mask已坏；generation_config公开固定revision端点404不证明漏下载

证据：`D:\RAGScope_Runtime\runs\v3_20261015_014506\S5_SUPERVISION_SMOKE_v2\COMPLETION.json`；`D:\RAGScope_Runtime\runs\v3_20261015_014506\S5_RUNTIME_INDEPENDENT_ANALYSIS.json`

## H11：输出长度或结束/adapter后端造成LoRA延迟增加

结论：`LENGTH_AND_DECODE_COST_SUPPORTED_MERGE_GATE_FAILED`。新24题输出中位67→171；固定64-token微基准显示未合并adapter解码更慢。三条合并logit探针超过事先容差，首选token相同不足证明函数等价，不能称合并解决延迟。

原反证：重复4gram平均下降；EOS/TTFT/decode速度原始未记录，不能补造；长度不是全部原因

证据：`D:\RAGScope_Runtime\runs\v3_20261015_014506\S5_RUNTIME_INDEPENDENT_ANALYSIS.json`；`D:\RAGScope_Runtime\runs\v3_20261015_014506\S5_FIXED64_base.jsonl`；`D:\RAGScope_Runtime\runs\v3_20261015_014506\S5_FIXED64_unmerged.jsonl`；`D:\RAGScope_Runtime\runs\v3_20261015_014506\S5_MERGE_LOGIT_GATE.json`

## H12：功能可靠性未转化为可发布质量/效率增益

结论：`INCONCLUSIVE_KEEP_V2_FINAL_ACCEPTANCE_PENDING`。工程链路验证与开发观察均不能替代一次锁定holdout、支持性/引用和真实在线成本。当前主指标未知，S6及最终交付尚未完成；默认v2，人工NOT_REVIEWED。

原反证：结构切块在旧题存在描述性收益，但不能推广到新holdout

证据：`D:\RAGScope_Runtime\runs\v3_20261015_014506\STATUS.json`

