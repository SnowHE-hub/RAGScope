# E0 legacy Dense/Hybrid重建执行记录

2026-10-14，Australia/Sydney。**E0固定权重重建已完成：Dense与Hybrid各200条真实结果，均0失败。** 全部22,254条文本（22,054 chunks和200 queries）在CPU编码完成。独立审计从400条记录重新计算指标，核验全部174批向量哈希、原8个索引及副本哈希未变；实际模型worker37108已退出，资源已交回主执行器安排的4B窗口。没有并载Laya或BGE-base。

原项目`data/indexes/chroma`只声明cosine、384维、22,054条embedding，没有模型revision/权重SHA。SQLite只以read-only immutable连接检查；队列仅保留55条向量，不足以验证完整向量空间。原EmbeddingService源码指定`BAAI/bge-small-en-v1.5`但不固定revision，因此原历史权重身份记UNKNOWN，不用猜测的新query向量搜索旧库。

已在`D:/RAGScope_Runtime/runs/20261014_v2_legacy/reconstructed`保存原源码zip展开副本、原8个索引文件的逐个原件/副本SHA、完全相同22,054 legacy chunks、同seed42的200问题与BM25 top20。全部200题top10 ID均与已完成E0.BM25逐题记录完全一致。原`rerun/`及其CSV、原索引没有改写。证据为`PROVENANCE.json`和上级`e0_source_inspection.json`。

已核验并下载[官方BGE-small模型](https://huggingface.co/BAAI/bge-small-en-v1.5/tree/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a)，MIT，revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`。safetensors实测133,466,304字节，SHA `3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad`与官方LFS一致；11个必要文件共134,505,940字节，清单位于`D:/RAGScope_Runtime/models/bge-small-en-v1.5/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/model_manifest.json`。下载没有导入Torch、读取Hub token或加载模型。

重建标签固定为`legacy_reconstructed_fixed_revision`。同legacy文本使用固定官方CLS/L2 CPU模型；查询加原源码前缀。显式保存截断前token数、实际编码token数和超512的每条位置，复现的legacy截断不能混入v2严格预算公平比较。重建使用exact NumPy cosine，代替未知历史权重的HNSW；Hybrid仍为每路20候选、alpha0.5、RRF60、final10。所有偏差必须随结果一起报告，不将变化归因于单个因素。

实现为`ragscope/experiments/legacy_reconstruction.py`：128文本分批缓存、batch_size8串行、CPU四线程，逐批sha、输入哈希、token日志和恢复状态；官方模型最多2次实际后端兼容失败；运行前可用RAM至少1792MiB，逐批提交后保留1280MiB（主任务最新要求1.25GiB）。实际worker自身每秒记录RSS与hostavailable，避免将venv启动器当模型。结果保存到新campaign `20261014_v2_legacy_reconstructed`，Dense/Hybrid各200条，失败保留，原BM25已完成记录保持不变。两项执行的PASS仅证明本项真实重建完成，不代表S2.6其他实验或整个任务完成。

| 方法 | N / 失败 | Recall@5 / Hit@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| Dense（固定版本legacy重建） | 200 / 0 | 0.870 | 0.905 | 0.761560 | 0.796696 |
| Hybrid（固定版本legacy重建） | 200 / 0 | 0.915 | 0.950 | 0.797012 | 0.834733 |

以上为原SQuAD问题和legacy相关性定义，不能直接当作v2证据覆盖，也不能据此归因切块或权重带来的提升。逐条结果为`dense.jsonl`、`hybrid.jsonl`，各自附`.config.json`和`summaries/*.json`。独立复算凭据为`D:/RAGScope_Runtime/runs/20261014_v2_legacy_reconstructed/INDEPENDENT_VALIDATION.json`；固定执行源码副本及哈希位于`reconstructed/embeddings_15b3a476b1e72a01/execution_source.py`和`execution_source_provenance.json`。

实际33,360,000个模型参数；8/22,054个chunk按legacy规则截断，200个query均未截断。174个持久化批次的实际编码用时合计1845.551秒；第二次进程包含加载、恢复验证、剩余编码和评估共464.362秒，其中模型加载58.282秒。该第二次时长不代表全程总用时。两次进程一秒采样的峰值模型RSS为747,855,872字节，最小hostavailable为1,239,101,440字节；共2次编码进程、1次正常资源暂停、0次后端失败。

本次初次实际执行命令（原日志保留）：

```powershell
. ./scripts/runtime_env.ps1
& 'D:/RAGScope_Runtime/envs/models/Scripts/python.exe' -u -m ragscope.experiments.legacy_reconstruction --home 'D:/RAGScope_Runtime' --stage all *> 'D:/RAGScope_Runtime/logs/e0_dense_hybrid_reconstruction.log'
```

如果仅继续已完成编码后的指标计算，用core环境同模块`--stage evaluate`，不加载任何模型。AST语法已验证；`tests/unit/test_legacy_reconstruction_v2.py`实际6 passed in1.99s，日志`D:/RAGScope_Runtime/logs/e0_reconstruction_unit_tests.log`。覆盖原20候选边界/同分顺序、双路分数累加、非法向量拒绝、CPU约束先于模型导入；不把这些小夹具当成真实检索成绩。

首次进程在第151批提交后正常资源暂停，19,328/22,254条已编码，实际worker42092退出。历史证据位于`D:/RAGScope_Runtime/runs/20261014_v2_legacy_reconstructed/RESOURCE_PAUSE.json`：退出前hostavailable1,240,330,240字节，小于保护值1,342,177,280；backend_failures仍0。主执行器安排2B400题完成并卸载后的模型切换窗口恢复，全部151批向量SHA通过后继续余下2,926条，没有停止主CPU矩阵。以下恢复命令已真实执行并退出0，无需再运行：

```powershell
. ./scripts/runtime_env.ps1
& D:/RAGScope_Runtime/envs/models/Scripts/python.exe -u -m ragscope.experiments.legacy_reconstruction --home D:/RAGScope_Runtime --stage all 2>&1 | Tee-Object -FilePath D:/RAGScope_Runtime/logs/e0_dense_hybrid_reconstruction_resume2.log
```

补充的描述性统计侧车`E0_ARTICLE_CLUSTER_CI.json`已完成。全部200题严格回映到冻结原始`train-v2.0.json`中的146篇文章，核对标题、段落、问题及合法答案，并保存官方QA ID。第一次精确文本检查发现原始段落首部换行；归档旧loader第67行确实调用`str.strip()`，故只复现该已知边界处理，没有模糊匹配或折叠内部空白。初次失败日志保留，修正后8项映射测试PASS（1.52秒）。400条模型记录的SHA没有变化。

每次整篇抽样并保留该篇全部问题，2000次、seed42、95% percentile CI，统计量仍为问题加权平均。Dense Recall@5 CI为[0.820388, 0.916667]，Hybrid为[0.876236, 0.950739]；Dense MRR CI为[0.712409, 0.808694]，Hybrid为[0.751456, 0.839825]。其余预定指标同存JSON。这里只提供各方法描述性区间，不进行新的显著性检验、不从区间重叠推导因果改善。

复现命令：`D:/RAGScope_Runtime/envs/core/Scripts/python.exe -m ragscope.experiments.legacy_statistics --home D:/RAGScope_Runtime --campaign 20261014_v2`。分析、输入/原行SHA、当时源码副本和日志凭据位于同目录`E0_ARTICLE_CLUSTER_CI.provenance.json`，未新增模型调用。
