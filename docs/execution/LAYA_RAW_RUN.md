# Laya真实概率预先冻结

当前：**输入已冻结，真实SDK推理尚未执行**。`scripts/run_laya_v2.py`默认仅准备输入，无Torch/Laya导入或模型调用。实际已读取固定corpus与父任务`ROUTING_QUERY_MANIFEST.json`，保存train200/dev60/test100共360个仅可见请求；三循环候选顺序共1080个预定决策。输入位于`D:/RAGScope_Runtime/runs/20261014_v2_laya/INPUTS.json`，input_hash `b82d4b56253f1773672fc35a49402b38409256a6e892777cd55fa64c20b270f5`。

可见输入通过父任务同一个`request_query`函数导出，仅question、用户scope和文档长度/类型/语言。answers、answerability、证据、oracle/最佳路径等字段不会进入输入。冻结与推理不打开任何generation records或labels文件；后续dev校准与test质量评分仍需真实三路径标签，不能据上游校准描述声明本任务校准完成。

官方safetensors header实际仅读取21,536字节，张量元素421,293,830（含buffer），绝大部分F16；权重payload842,587,666字节。SDK先构造FP32参数再加载权重，同时存活的张量存储地址空间估计2,527,762,986字节；checkpoint mmap页可被回收，这不是RSS下界。预计预算3.5GiB，默认预载要求5GiB空闲，运行保留1.25GiB。它是静态估算而非测量，原凭据`D:/RAGScope_Runtime/runs/20261014_v2_routing/LAYA_READINESS.json`保持不改，准确解释见`LAYA_LOAD_MEMORY_SOURCE_AUDIT.json`及本目录`LAYA_LOAD_MEMORY_AUDIT.md`。当前tokenizer配置不触发SDK修复条件；加载后仍记录实际hash。

资源顺序固定为structured+ROUTE完整边界→embedding退出→R3五组660条完成且CPU进程退出→22调用4K/8K压力诊断完成且GPU服务退出→Laya→恢复CPU embedding。Laya窗口的准确命令：

```powershell
. ./scripts/runtime_env.ps1
& D:/RAGScope_Runtime/envs/models/Scripts/python.exe -u scripts/run_laya_v2.py --campaign 20261014_v2 --run
```

仅CPU四线程、batch1、max_len512、eager、不compile。最多3次实际加载兼容失败，3个dev真实预热，逐个顺序保存原始decision/probabilities、输入hash、fallback、错误、实际RSS；连续3个后端失败停止。半写入尾行按现有RecordStore保存恢复，已完成记录不会重复推理。`records.jsonl`必须1080条后才有SUMMARY COMPLETE；这仍只代表概率执行完成，quality_scoring保持PENDING_REAL_ROUTE_LABELS。

主执行器另授权唯一条件式资源备选：fresh RAM 4.5–5 GiB、无其他本项目模型常驻时，上述CLI自动用精确身份验证的独立CPU子进程执行同一SDK/FP32加载与原3题dev预热；低于4.5 GiB不尝试。父进程0.25秒采样、加载上限180秒、运行余量1.25 GiB；仅首次WARMUP授权前可停止自有无结果加载进程，此后只能保存结果再合作退出。一次真实资源/加载失败保留证据并停止备选，不能以原3次兼容性上限重试这一备选。真实加载仍未执行。

正式完整generation协议冻结后，`decisions_to_protocol`校验同一360人群和每个可见请求，将默认候选顺序的dev/test决策绑定protocol_hash并写`DECISIONS_laya.json`，不会重跑模型或读取路径质量。候选顺序研究保留全部三次输出，不根据test结果选择顺序。

新增3项离线测试实际通过，3.26秒，日志`D:/RAGScope_Runtime/logs/laya_campaign_input_tests.log`；覆盖360人数/1080预算、gold字段不进输入、重复冻结字节一致、来源变动拒绝、未完成真实人口拒绝绑定。测试不涉及实际SDK推理。

与R3及统一runner联测共40 passed，10.02秒，`D:/RAGScope_Runtime/logs/router_rerank_runner_final_tests.log`；所有真实模型工作仍需上述资源门槛与交接。
