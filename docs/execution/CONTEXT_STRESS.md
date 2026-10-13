# 4K / 8K 上下文资源诊断

本诊断落实任务书 S3.5 的有界压力测试，只使用最小主模型 Qwen3.5-2B Q4_K_M，最多20个开发问题，固定4K/8K两条件，不参与主问答质量排名。

真实冻结目录：`D:/RAGScope_Runtime/runs/20261014_v2_context_stress`。manifest语义哈希为 `e87d506a9eab594eac0f088317d094df315c82097bc3eaa3ce0a56cbb04589c9`。

100个dev问题全部按 seed42 顺序检查原文长度可行性。只有8题的指定论文足够长，同时满足两个窗口；其余92题均在任何模型调用前排除，逐题原因保留在manifest。`CANONICAL_LENGTH_AUDIT.json` 另保存全部100题的实际聊天模板全文token数，范围1895～11699，其中仅8题达到8K窗口的输入长度要求。没有按答案或生成结果选题，没有补抽test题。

两条件使用相同8题和同一规范化论文的原文前缀，保留规范化坐标、evidence unit交集、标题与引用标签。发送内容不含标准答案字段、answerability或gold evidence标签。4K的8条实际输入全部3584token，8K全部7680token，各预留512输出；因此实际输入长度随窗口增加，绝非只改 `num_ctx`。JSON schema、prompt v2.2、权重、量化、seed42、greedy512完全一致，仅窗口及其原文前缀长度不同。

执行命令（须先得到调度器的当前资源交接并启动自有11435服务）：

```powershell
. ./scripts/runtime_env.ps1 -Gpu
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/run_context_stress_v2.py --mode run --window 4096
# 完成并正常卸载4K模型、核验资源后，再执行第二条。
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/run_context_stress_v2.py --mode run --window 8192
```

每条件三个单列warmup和8个正式请求，全部进入同一全局generation budget。逐行保存实际输入token计数与后端parity、输出、解析错误、资源采样、resident context length和调用票据。OOM一次、连续后端错误三次或实际token计数不匹配即停止该条件并保留原因，不重复已执行题目，不新增窗口或模型。主机RAM加载前保护与实际模型OOM分开记录。最后卸载模型、退出项目服务并验证GPU释放。

当前状态：输入已冻结，6项离线契约测试通过，尚未开始真实GPU请求。CPU batch benchmark已正常完成，但一次实际加载前检查未通过；保存于 `PRE_SERVICE_RESOURCE_GUARD_1791913367123377000.json`。`require_ram(2GiB)` 实际表示新增2GiB并保留默认1GiB余量，因此有效可用内存门槛为3GiB，当次主机可用约2.53GiB。没有加载模型、启动GPU服务或产生模型OOM。`EXECUTION_READINESS.json` 保持PENDING，等待structured/ROUTE完成后正常CPU资源交接，或自然恢复到实际guard要求；不降低门槛、不连续重试。Embedding权重仍驻留，不假设CPU模型已卸载。

准备阶段曾发现新的scope guard只支持字符串，而冻结数据使用document/doc_ids结构；零模型调用，100个工程guard拒绝及旧实现SHA保存于 `FREEZE_SCOPE_VALIDATION_FAILURE.json`。已修复两种合法单论文scope，仍拒绝global scope，复测并重新冻结成功。该准备错误不计作模型失败或源长度不足。
