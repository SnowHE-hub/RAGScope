# v3质量与成本记分卡

结论：没有候选通过开发入围门槛，默认保留v2。以下开发结果不是新holdout发布证明；人工审核NOT_REVIEWED。

## 同题开发确认：100题、50篇论文

|指标|B0保留策略|C1 core-first|
|---|---:|---:|
|可答文字F1（92题）|0.395329|0.407682|
|可答题拒答数|2|1|
|格式/调用失败|0|0|
|主要claim F1均值|UNKNOWN|UNKNOWN|
|支持性均值|UNKNOWN|UNKNOWN|
|引用完整性均值|UNKNOWN|UNKNOWN|

文字差值+0.012354，2000次论文聚类95%区间[-0.014965764379557723, 0.040795451139721085]，跨零。
主要claim F1只14/13个可答问题有确定点值；其余未知不丢弃，完整上下界见JSON。不能把已知子集当总体。

## 实际在线：30题×每条件3次

|指标|B0保留策略|C1 core-first|
|---|---:|---:|
|实际正式请求|90|90|
|P50秒|1.1406|1.1469|
|P95秒|1.6026|1.9960|
|调用/格式失败|0|0|

每次真实重新编码问题、检索、打包、生成、序列化。答案和embedding应用缓存关闭；6次独立预热不并入180正式请求。
编码器与一个2B生成器同驻留GPU，所有请求串行。两条件均使用已核验的GPU embedding桥接，不能冒充原CPU部署成本复现。
P50比值1.0055、P95比值1.2454，没有达到效率收益。
完整论文聚类区间、可答/不可答及失败分母见JSON。缓存命中应用测试另列，不用它宣称模型变快；非流式TTFT为N/A。

## 范围与正式留出

H_scope40真实范围策略测试通过：20指定论文不越界，20缺对象澄清；合成规则案例、非事实QA成绩。
H_QA200/H_tasks20最终配对为BLOCKED：任务书10.1禁止未入围候选进入正式比较，没有创建最终gold读取授权。
必须的未见题质量、支持性、引用及拒答比较尚未完成，所以execution_complete=false；不把开发比较或仅基线跑分替代它。

## 可复核来源

- `D:\RAGScope_Runtime\runs\v3_20261015_014506\S3_SCOPED_GENERATION_D_confirm_ANALYSIS.json` SHA256 `f21125118b3320526bc8824eae772bf7d9619964e7568069e1e02d3a6bcd44f0`
- `D:\RAGScope_Runtime\runs\v3_20261015_014506\S3_D_CONFIRM_CLAIM_GATE.json` SHA256 `220c0a07166117468b2a571aab15ac85a40a5492cd2bd0ea9e17bd4ce2b128d0`
- `D:\RAGScope_Runtime\runs\v3_20261015_014506\S6_ONLINE_COST_ANALYSIS.json` SHA256 `3a97747931b4fd1296b2dbc51872445a060aca25a174ca2ec9545d1ef474e60e`
- `D:\RAGScope_Runtime\runs\v3_20261015_014506\CANDIDATE_LOCK.json` SHA256 `0f7087935607dbd6a92f6439e97bf056d37be9cc04debaed6a48d38d99a52a7d`
- `D:\RAGScope_Runtime\runs\v3_20261015_014506\S6_H_SCOPE_COMPLETION.json` SHA256 `ea3e57568a893d8a9349af1e9f9ea6d48b47eca5b33a9e87785805afb9e4618f`
- `D:\RAGScope_Runtime\runs\v3_20261015_014506\S6_APPLICATION_CACHE_VERIFICATION_final_scope_guard.json` SHA256 `3a8f6f25fa88d8bb2814e03a5072bc9ce14e56e7312becb4fc7f2ae6d176f2af`

冷启动/预热实测单列：`D:\RAGScope_Runtime\runs\v3_20261015_014506\S6_COLD_START_OBSERVATIONS.json`，含6条实际预热总时、索引加载及后端模型加载元数据；不是冷启动SLA。
