# v3 评分契约（S1 当前版本）

本文件定义后续评分，不代表所有指标已实际计算。人工审核始终为 NOT_REVIEWED。

旧文字基准保留 legacy_token_f1。答案按小写、去英文标点、去 a/an/the 后的词语计重合；跨合法参考答案取最佳 F1。只评分解析后的 answer，不将 JSON、引用或其他字段混入。可答题的请求/格式失败和拒答记零，全部可答题保留在分母；不可答题的答案 F1 为 N/A，拒答单列。

已从本地官方 QASPER v0.3 发布包保存纯 evaluator，并对全部 5,720 条旧答案（主矩阵 5,520，加 LoRA 200）重新计算 qasper_compatible_answer_f1。此字段遵循官方文字计算和参考处理，但保留本轮“不可答 N/A、拒答另列”的报告约定，不能冒称官方包含 Unanswerable 文本的全人群总分。实际样本与 legacy F1 的差异数为 0。空归一化参考的边界差异有单独功能测试，未在实际样本制造提升。

证据覆盖按 canonical 文本区间求交集：每个参考单元算覆盖比例，合并重复可见区间；按完整集合、完整单元、连续覆盖的次序选择最佳合法参考集合。完整集合覆盖只是标注证据全部可见，不等于答案正确、事实有支持或上下文充分。visible_context_sufficient 在尚未完成问题/上下文评判时保持 UNKNOWN。

实际输出 tokens 优先 backend.actual_output_tokens（BF16），其次 backend.eval_count（Ollama）。completion_tokens 可能是参考 tokenizer 计数，不能冒充后端输出长度。S0 初版投影的 BF16 fallback 计数已在 S1 明确更正，原投影及其收据保留。首 token 和停止原因缺记录时保持 null/NOT_RECORDED。

BERTScore 英文默认 RoBERTa-large/17层在GPU和两种CPU缓存调度下分别完成64、832、232对后触发3GiB内存守卫，原配置记BLOCKED_RESOURCE，部分结果和失败原样保留。官方实现分发版本固定0.3.13（包内部__version__仍写0.3.12，两者均记录），慢tokenizer、无IDF、无baseline rescale。

唯一编码器方法替代复用已下载的MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli，按官方BERTScore的base架构注册选择9层，固定FP32/CPU/encoder batch4/外层8对。它是NLI微调编码器的独立资源替代，不能冒充英文默认，也不是与NLI独立的第二个模型裁判。实际完成6342对、5720旧答案，342932 encoder tokens、0截断；所有模型同配置，原答案未改写。它只衡量参考语义接近度，不是事实真值。模型/层选择在查看全量得分前固定，依据内存约束而非更高得分。

后续将独立报告 claim_reference_precision_proxy、key_fact_recall_proxy、claim_f1_proxy、context_support_precision_proxy、citation_support_precision_proxy、citation_completeness_proxy 与关键实体/数值/否定/关系冲突。没有任意加权单一总分；未评判和 UNKNOWN 不从总体消失。

裁判用60组规则构造对照进行工程校准；输入构造不是人类真值审核。V1中“购买”不排除以后“卖出”的歧义已在判分前另立V2，原文件保留。实际NLI完成130项关系判断（含引用错配），130/130、60/60组通过；满足本轮90%工程门槛，不代表自然问题或人评准确率。120条自然样本按失败、拒答和答案类型固定分层抽样，模型/方案名字只在私有join文件；自然样本的AI复核和所有人工审核仍待完成。

新 H_QA 答案封存于 evaluator_only，选型/训练/诊断读取被拒绝；正式评估需要绑定冻结候选与单轮授权。该 API 边界不是操作系统权限隔离，不把它描述成 Full Access 下不可绕过的安全沙箱。

后续实际 QA 校准发现，短答案原片段仅 6/24、问题与答案拼接仅 12/24 正确，完整陈述 24/24；前两种包装禁止直接承担自动事实门槛。参考不可见的窄极性/数量完句规则在独立 80 个构造测试上 80/80，通过仅限规则支持范围的工程校准；自然语料准确率及人工仍未验证。完句只用于 NLI 辅助输入，旧答案与文字/BERTScore 输入保持原样。其他片段、不独立复合句和裁判不确定均保留 UNKNOWN；点估计 null，完整总体另报上下界，不把 UNKNOWN 移出分母。
