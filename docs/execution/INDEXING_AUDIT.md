# S2.1 / S2.4 索引身份、embedding 与重排核验

日期：2026-10-14（客户端 Australia/Sydney）。本子任务只运行离线小型测试，没有加载真实模型、分配 CUDA 或接触原 GPU 程序。

## 任务书问题的本机状态

| 问题编号 | 本机确认 | 修复范围 |
|---|---|---|
| 7 | 旧 DenseIndex.build 只看 collection.count()>0 便跳过；固定文件名无法确认当前配置 | `ragscope/indexing/dense_index.py` 以规范化内容 digest、完整 embedding identity、维数、距离度量、条数生成 manifest；不匹配时报错并保留原库；无 manifest 的旧库不当成已验证的新库 |
| 9 | 旧 embedding cache key 只有模型名字和文本，不含 revision/参数；生成缓存问题由另一子任务处理 | `ragscope/indexing/embedding.py` 改用主任务 ArrayCache 的 SHA-256 身份目录、float32 npy + 校验清单；身份含模型/tokenizer revision、pooling、归一化、query instruction、预算、precision。真实模型解析到 commit hash 时优先使用 commit hash |
| 13 | 旧 Hybrid 按 2k 扩候选；旧 Reranker 直接把所有长片段交给 max_length=512 的 CrossEncoder | Hybrid 默认两路各50，RRF rank从1、constant60、alpha0.5、稳定同分排序；Reranker.retrieve 默认50→10。每个 query+window 使用真实 tokenizer 并包含 special tokens，完整窗口逐一评分后 windowed_max，保存跨度/token数/得分 |
| 15 | 旧 embedding diskcache 保存 numpy 对象可能涉及 pickle 与环境兼容；模型导入过早 | 新 cache 不读取旧 pickle/diskcache，不删除原数据；SentenceTransformer/CrossEncoder/Chroma 均延迟至显式构造且支持注入测试对象。模块 import 无模型下载。NLTK 导入下载已在 CHUNKING_AUDIT 中处理 |

旧 `scripts/build_indexes.py` 的工厂/命名由主任务单独维护；此表不把本模块修复冒充该入口的全部修复。`DenseRetriever.retrieve` 调用必须传 `embedding_identity=self.embedding_service.identity`，主任务已经整合。

## 关键行为

- `EmbeddingService.embed` 在缓存或真实模型调用之前核对完整输入（含 special tokens），超限明确报错；`embed_query` 先加已冻结的 query instruction 再计数。自动 SentenceTransformer prompt 显式禁用，避免重复或隐含前缀。BGE512 的 chunk 正文通常应预留 special-token 余量，不能把512正文静默截成510。
- DenseIndex.build 可恢复本任务创建的 building 状态；读取已存在 chunk ID、正文和 payload hash，只补缺失行。完整 manifest 只在最终条数正确时原子写入。错误、不同模型、不同正文不会触发覆盖旧库。
- DenseIndex.search 强制 query vector space identity 与 manifest 精确一致，同维度的不同模型仍被拒绝。输出恢复完整 chunk payload（含父块与证据 spans），再次校验内容。
- Reranker 默认 CPU。问题本身耗尽窗口预算时明确失败，不偷偷截问题；长文档窗口覆盖到末尾，max 分数聚合。默认保留最多10项；小于50项的实际语料自然返回不足50，不虚构候选。相同候选 ID 会报错，避免窗口记录归属不清。
- 每条候选记录带 `rerank_aggregation=windowed_max` 和窗口详情。这是工程上统一的评分策略，不能与不切窗 reranker 的延迟或分数不加说明地混合。

## 已运行测试

命令：

```powershell
. ./scripts/runtime_env.ps1
$env:COVERAGE_FILE='D:/RAGScope_Runtime/runs/indexing_unit.coverage'
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -m coverage run --source=ragscope -m pytest tests/unit/test_indexing_v2.py -q -p no:cacheprovider
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -m coverage report -m ragscope/indexing/embedding.py ragscope/indexing/dense_index.py ragscope/retrieval/hybrid_retriever.py ragscope/retrieval/reranker.py
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -m coverage json -o 'D:/RAGScope_Runtime/runs/indexing_coverage.json'
```

最终实际结果：20 passed，退出码0。针对本次四个模块的覆盖率：DenseIndex94%、EmbeddingService97%、Hybrid100%、Reranker89%，合计94%。机器报告保存在 `D:/RAGScope_Runtime/runs/indexing_coverage.json`（该 JSON 还列出测试期间被观察到的其他 ragscope 模块，本表仅对应本子任务负责范围）。

验证包含：相同文本去重编码；npy缓存复用/不同revision隔离/维数拒绝；完整query预算；无身份模型拒绝；NaN与零向量拒绝；manifest/条数/正文不匹配；受控中断后只补缺失批次；原有无manifest库保留；同维异空间拒绝；父块与证据字段不丢；窗口覆盖末尾；query预算；候选第50项重排进入前10；空候选/超长问题/重复ID/非有限分数。

初次 coverage 用四个 dotted module 作为 source 时出现 NumPy `cannot load module more than once per process` 的采集失败；普通 NumPy import、普通 pytest 均成功，改成单一 `--source=ragscope` 后采集成功。没有把该失败计为模型实验或略过功能断言。

所有模型都是明确的测试对象；测试不能证明 BGE 或其他实际权重的检索效果。真实加载、版本锁定、长窗口 CPU 吞吐与正式50→10实验仍由后续模型预检和 campaign 记录给出。实际 CrossEncoder/Chroma 构造分支未在本轮离线单元测试中冒充已集成验证。

## 官方接口核验

执行时读取了 [SentenceTransformer 官方接口](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html)、[CrossEncoder 官方接口](https://www.sbert.net/docs/package_reference/cross_encoder/model.html) 与 [BAAI/bge-reranker-base 官方模型卡](https://huggingface.co/BAAI/bge-reranker-base)。据其接口使用显式 device、revision、trust_remote_code=False、local_files_only，且将完整输入长度核验放在可能截断的模型调用之前。具体正式模型 revision 由主任务模型 manifest 冻结，此文不虚构已下载权重 hash。
