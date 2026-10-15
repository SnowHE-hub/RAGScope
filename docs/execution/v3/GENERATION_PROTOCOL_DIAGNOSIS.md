# 生成协议与模型规模诊断

## 没有发现直接调用与pipeline的通道或token错配；协议影响依模型而异

已完成72次direct/pipeline冒烟、864条固定问题协议记录和42条补充记录。更简洁的提示对4B/9B文字代理有帮助，对2B没有一致帮助；去除schema增加了输出/引用失败。结果不支持统一更换协议，也不证明9B事实能力必然弱于2B。主要事实评分仍有资源阻断。

## 样本、分母与控制条件

固定24题，每类不可答/yes-no/抽取/自由回答各6题；每题使用干净Oracle和冻结真实检索两种材料。G0复用144条真实结果，新主矩阵720条，总计864条。G3有3个种子，因此组n=72是24题的重复测量，不能当作72个独立问题。

G0保留原schema与greedy；G1仅加强最小充分回答；G2在G1基础上取消schema；G3在G1基础上用固定官方模型卡文本配置及三个预先种子。全部保留失败分母；可答题文字F1、不可答拒答分开，不能把文字相似度称事实正确率。

逐条审计的metric mismatch=0、实际prompt token mismatch=0。旧已见题仅作诊断，不能作为新holdout发布证明。

## 协议对照的完整组结果

|模型|材料|协议|记录数|可答F1|失败|可答拒答|输出token中位数|
|---|---|---|---:|---:|---:|---:|---:|
|qwen_2b|oracle_audited|G0|24|0.608320|0|3|55.0|
|qwen_2b|retrieval_frozen|G0|24|0.522908|0|1|65.0|
|qwen_2b|oracle_audited|G1|24|0.581310|0|2|56.0|
|qwen_2b|retrieval_frozen|G1|24|0.481895|0|0|79.0|
|qwen_2b|oracle_audited|G2|24|0.082716|13|1|40.5|
|qwen_2b|retrieval_frozen|G2|24|0.324725|6|0|66.0|
|qwen_2b|oracle_audited|G3|72|0.512299|0|5|53.0|
|qwen_2b|retrieval_frozen|G3|72|0.312628|0|2|67.0|
|qwen_4b|oracle_audited|G0|24|0.521613|0|2|57.0|
|qwen_4b|retrieval_frozen|G0|24|0.293717|0|3|105.0|
|qwen_4b|oracle_audited|G1|24|0.623152|0|2|55.5|
|qwen_4b|retrieval_frozen|G1|24|0.446065|0|0|82.0|
|qwen_4b|oracle_audited|G2|24|0.569184|1|2|55.5|
|qwen_4b|retrieval_frozen|G2|24|0.319081|8|0|77.0|
|qwen_4b|oracle_audited|G3|72|0.556609|0|5|56.0|
|qwen_4b|retrieval_frozen|G3|72|0.415718|0|1|91.0|
|qwen_9b|oracle_audited|G0|24|0.417982|0|3|59.0|
|qwen_9b|retrieval_frozen|G0|24|0.211752|0|2|96.5|
|qwen_9b|oracle_audited|G1|24|0.470670|0|3|58.0|
|qwen_9b|retrieval_frozen|G1|24|0.276619|0|2|74.0|
|qwen_9b|oracle_audited|G2|24|0.349287|5|2|58.0|
|qwen_9b|retrieval_frozen|G2|24|0.000000|19|2|69.0|
|qwen_9b|oracle_audited|G3|72|0.481451|0|8|59.0|
|qwen_9b|retrieval_frozen|G3|72|0.284523|0|5|73.5|

同一材料下，提示词改变是受控实验变量；不同材料下Oracle和检索同时有证据范围、长度和噪声差异，不能只归因于召回。主协议结果只说明本机固定Q4后端及模板下的表现，不是跨架构能力排名。

## 干净Oracle与等长干扰仍需分开解释

补充实际36条Oracle padded生成，覆盖12个可答问题×3模型；证据逐字保留，干扰来自同论文的非标注证据段。最大长度相对差2.5668%；2个原检索不含证据的位置无法复刻，使用预先声明的中间位置并保留标记。

|模型|padded−clean可答文字F1|题数|
|---|---:|---:|
|qwen_2b|-0.035220|12|
|qwen_4b|-0.150512|12|
|qwen_9b|-0.004175|12|

这些结果说明上下文组织是混杂因素；只有12题，标注之外的段落可能仍有相关信息，未经过操作者人工相关性审核。不能由padded变化推断一般噪声鲁棒性。

## No Context的任务限制与已有知识探针

原strict-grounded任务要求只依据材料回答，空材料下拒答不能直接解释为“模型没有知识”。schema允许空引用，但不强制abstain=true；限制主要来自任务提示。补充允许已有知识的H2O/一周天数探针，共6次三模型实际回答均命中字面答案，只证明这两个简单问题，不代表广泛知识成绩。

## 尚未解决的根因和选型结果

直接/pipeline的72次冒烟中36对输入和输出一致，未观察到token或thinking通道错配。该反证只覆盖固定夹具及模板；没有足够依据声称所有9B场景都不存在包装问题。

G2的失败保留，不以“能解析出JSON外壳”冒充合法引用或可靠事实答案。官方采样配置不保证本机更好，不能以哪个种子最高来挑选。未新增9B BF16全矩阵，也不将未运行的精度桥接写成已验证。

D_confirm上的core-first配对100题已另完成；文字F1增量约+0.01235、论文聚类95%区间跨零，主要claim/support/citation代理未知。当前没有可据此发布的统一新生成协议，保留v2默认。实际工作流与online180已完成；没有候选入围，正式holdout配对受阻，人工NOT_REVIEWED。

## 可复核来源

报告只读取已完成审计，无新模型调用。若重新运行报告，不重跑主矩阵或补充模型请求。

- `D:\RAGScope_Runtime\runs\v3_20261015_014506\S2_PROTOCOL_ANALYSIS.json` SHA256 `d535460de0fed50c4b43aee42a93465264f253a4612eaa39f6cc0b012fb098a8`
- `D:\RAGScope_Runtime\runs\v3_20261015_014506\S2_SUPPLEMENT_ANALYSIS.json` SHA256 `a92b6d37be3c34c1c8821167996961a97fbad71a71b1fb0c458c92528e352749`

## 本轮最终门槛状态

真实online180、合成范围40和工作台实连完成。质量增益与非劣无法证明，效率门槛未过，保留v2默认；H_QA/H_tasks正式配对BLOCKED，execution_complete=false，human_review_status=NOT_REVIEWED。完整证据见QUALITY_SCORECARD.json与ACCEPTANCE_V3.json。
