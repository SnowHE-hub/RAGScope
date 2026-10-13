# E0 legacy Dense/Hybrid重建执行记录

2026-10-14，Australia/Sydney。主执行器确认正式LoRA训练稳定、可用RAM满足保护门槛后交回轻量E0运行许可。真实BGE-small仅CPU完成19,328/22,254条编码后，在第151批durable commit后触发1.25GiB保护，状态PENDING_RESOURCE；**Dense/Hybrid200题正式指标尚未产出**。实际worker42092已确认退出，全部151批向量SHA复核通过，剩余2,926条；没有并载Laya或BGE-base。

原项目`data/indexes/chroma`只声明cosine、384维、22,054条embedding，没有模型revision/权重SHA。SQLite只以read-only immutable连接检查；队列仅保留55条向量，不足以验证完整向量空间。原EmbeddingService源码指定`BAAI/bge-small-en-v1.5`但不固定revision，因此原历史权重身份记UNKNOWN，不用猜测的新query向量搜索旧库。

已在`D:/RAGScope_Runtime/runs/20261014_v2_legacy/reconstructed`保存原源码zip展开副本、原8个索引文件的逐个原件/副本SHA、完全相同22,054 legacy chunks、同seed42的200问题与BM25 top20。全部200题top10 ID均与已完成E0.BM25逐题记录完全一致。原`rerun/`及其CSV、原索引没有改写。证据为`PROVENANCE.json`和上级`e0_source_inspection.json`。

已核验并下载[官方BGE-small模型](https://huggingface.co/BAAI/bge-small-en-v1.5/tree/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a)，MIT，revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`。safetensors实测133,466,304字节，SHA `3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad`与官方LFS一致；11个必要文件共134,505,940字节，清单位于`D:/RAGScope_Runtime/models/bge-small-en-v1.5/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/model_manifest.json`。下载没有导入Torch、读取Hub token或加载模型。

重建标签固定为`legacy_reconstructed_fixed_revision`。同legacy文本使用固定官方CLS/L2 CPU模型；查询加原源码前缀。显式保存截断前token数、实际编码token数和超512的每条位置，复现的legacy截断不能混入v2严格预算公平比较。重建使用exact NumPy cosine，代替未知历史权重的HNSW；Hybrid仍为每路20候选、alpha0.5、RRF60、final10。所有偏差必须随结果一起报告，不将变化归因于单个因素。

实现为`ragscope/experiments/legacy_reconstruction.py`：128文本分批缓存、batch_size8串行、CPU四线程，逐批sha、输入哈希、token日志和恢复状态；官方模型最多2次实际后端兼容失败；运行前可用RAM至少1792MiB，逐批提交后保留1280MiB（主任务最新要求1.25GiB）。实际worker自身每秒记录RSS与hostavailable，避免将venv启动器当模型。结果保存到新campaign `20261014_v2_legacy_reconstructed`，Dense/Hybrid各200条，失败保留，原BM25已完成记录保持不变。当前编码尚未完成，没有生成该campaign的PASS结果。

本次初次实际执行命令（原日志保留）：

```powershell
. ./scripts/runtime_env.ps1
& 'D:/RAGScope_Runtime/envs/models/Scripts/python.exe' -u -m ragscope.experiments.legacy_reconstruction --home 'D:/RAGScope_Runtime' --stage all *> 'D:/RAGScope_Runtime/logs/e0_dense_hybrid_reconstruction.log'
```

如果仅继续已完成编码后的指标计算，用core环境同模块`--stage evaluate`，不加载任何模型。AST语法已验证；`tests/unit/test_legacy_reconstruction_v2.py`实际6 passed in1.99s，日志`D:/RAGScope_Runtime/logs/e0_reconstruction_unit_tests.log`。覆盖原20候选边界/同分顺序、双路分数累加、非法向量拒绝、CPU约束先于模型导入；不把这些小夹具当成真实检索成绩。

资源暂停证据位于`D:/RAGScope_Runtime/runs/20261014_v2_legacy_reconstructed/RESOURCE_PAUSE.json`：退出前hostavailable1,240,330,240字节，小于保护值1,342,177,280；backend_failures仍0。主执行器安排2B400题完成后的模型切换窗口恢复，不反复重载，不停止主CPU矩阵。收到handoff且1792MiB预载保护通过后的下一条准确命令：

```powershell
. ./scripts/runtime_env.ps1
& D:/RAGScope_Runtime/envs/models/Scripts/python.exe -u -m ragscope.experiments.legacy_reconstruction --home D:/RAGScope_Runtime --stage all *> D:/RAGScope_Runtime/logs/e0_dense_hybrid_reconstruction_resume2.log
```
