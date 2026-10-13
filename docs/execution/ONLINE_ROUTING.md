# 在线路由验证

当前状态：实现与离线契约测试通过，实际在线实验尚未开始。最终 Hybrid dev 检索选型、全部路由标签和七个策略决策尚未就绪，不能把重放耗时当作在线结果。

运行入口：

```powershell
. ./scripts/runtime_env.ps1
& D:/RAGScope_Runtime/envs/models/Scripts/python.exe scripts/run_online_routing_v2.py --campaign D:/RAGScope_Runtime/runs/20261014_v2_generation --routing-dir D:/RAGScope_Runtime/runs/20261014_v2_routing --mode freeze
& D:/RAGScope_Runtime/envs/models/Scripts/python.exe scripts/run_online_routing_v2.py --campaign D:/RAGScope_Runtime/runs/20261014_v2_generation --routing-dir D:/RAGScope_Runtime/runs/20261014_v2_routing --mode run
```

`freeze` 要求 fixed BM25、fixed Hybrid、fixed Hybrid+rerank、rules、linear、MLP、LSTM 七个策略的 dev/test 决策已全部冻结；仅从 dev 实际生成记录选出最佳固定和最佳非固定策略。质量优先，answerable F1 在 0.01 内以 dev 重放 P50 较低者优先。同一组 30 个 dev ID 按 seed42 冻结，不按答案质量抽样。test 分数不参与选择。该步骤不加载模型、访问服务或产生生成调用。

`run` 校验本项目独立生成服务的 PID、创建时间、二进制、GPU lease、单卡串行配置和已关闭的历史 prompt RAM 缓存。三个 Qwen counter、2B 实际生成模型和已冻结身份必须一致。CPU reranker 加载严格核验的 BGE 本地文件，不自动下载或启动服务。正式运行需模型环境已包含各个应用依赖，CPU embedding 服务已由主调度器就绪；这些运行前提尚不能视为满足。

每个问题实际调用路由器，然后执行 live retrieval。检索先完成固定 2048 reference-token context，再按与正式矩阵相同的 `shared_context` 函数，使用三 Qwen 真实聊天模板共同预算重新封装。完整计时包含路由、检索、context packing 和 2B 生成；评分及写盘不算用户请求延时。模型、索引、路由器初次加载单列，正式测量前每策略有三条独立 warmup 记录。两策略逐问题配对并交替先后顺序；查询 embedding 缓存状态逐条保留，不宣称冷缓存耗时。

输出在路由目录的 `online/`：不可变 `ONLINE_MANIFEST.json`、`records.jsonl`、`warmups.jsonl`、`SUMMARY.json`。每次新进程恢复待办任务时重新做三次 warmup，所有真实请求（包括失败和 warmup）均进入全局 10,000 次计数。已有正式行不重复生成。检索或路由失败仍占 30 条分母，但 `generation_attempted=false`、无生成 ticket，不能称为真实模型输出。实际生成错误保留原始输出及 ticket。OOM 一次或连续三个后端/检索/路由错误触发保留断点退出。

每策略分别报告已记录行数、实际生成尝试数、生成前失败数。两组都记录30行仍不等于30次真实请求；任一策略少于30次实际生成尝试时，部署判定不通过。质量门槛使用完整冻结60题dev的answerable F1，要求非固定策略不低于固定策略减0.01；延迟门槛使用30题实际在线配对测量，要求P50至少降低10%。30题子集的质量另列，不代替完整dev质量。模型/路由器/index/reranker加载单列冷启动时间，在线比较边界是warmup后的驻留服务。CPU reranker加载前调用require_ram(2GiB)，即新增2GiB且保留默认1GiB余量，实际要求至少3GiB可用主机内存。这些是工程门槛，不等于统计显著性。未完成或未达标时保留固定策略，不根据test再选择。

验证：`tests/unit/test_online_routing_v2.py` 共9项通过，涵盖dev-only选择、七策略先冻结、分母保留、全局计数、断点恢复、模型/counter身份、来源篡改拒绝、选中学习路由器必须来自实际train/dev代理标签，以及CPU-only客户端对独立GPU服务的lease校验。测试临时目录的离线客户端仅用于功能测试，没有写入真实实验目录。
