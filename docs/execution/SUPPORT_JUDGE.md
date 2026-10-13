# 支持性代理判分

实现版本 cited-support-proxy-v1。当前只完成离线契约测试，尚无真实 judge 结果；真实执行必须等待冻结且完整的 E2 主矩阵。

固定从 E2 task IDs 按生成模型、检索方法、可回答性分层，以 seed42 的轮转抽样选至多200条。抽样不读取答案或分数，样本清单不可变；源模型失败保留 unknown、不补样。判分模型固定为已冻结、通过真实预检的 Qwen4B（必要时在执行前固定9B）。每次真实调用采用统一 GenerationBudget/CountedClient，并验证实际后端输入token数与完整聊天模板计数一致。

Rubric只判断生成回答是否得到它所引用的实际发送片段支持。提示不含 gold答案、证据标签或生成模型身份，记录原文、解析、提示哈希、模型身份、来源记录哈希、同权重自评偏差。失败、JSON不合法和token差异保留unknown；引用片段过长则明确无法判分，不截断后冒充完整支持。自动分层比例不等同于自然分布幻觉率，也不称人工真值。

先冻结：`D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/run_support_judge_v2.py --campaign-dir D:/RAGScope_Runtime/runs/20261014_v2_generation`

在root完成资源交接并已有完整E2记录后，原命令加`--run`。默认写入campaign下support_judge的SAMPLE_MANIFEST.json、SOURCE_RECORDS_MANIFEST.json、judgments.jsonl、SUMMARY.json、HUMAN_REVIEW.json。后者保留完整实际上下文，人工字段为空并明确NOT_REVIEWED。没有启动、下载或恢复模型服务的隐含动作。

离线判分与冻结回归19项通过（2026-10-14）；测试客户仅在临时目录模拟协议，不构成模型质量证据。
