## 18. 附录与个人速查

### 附录 A：一页项目技术速查表

| 面试官问 | 可直接给出的准确回答 | 证据入口 |
|---|---|---|
| 系统是什么 | 本地论文RAG工作台+可复核实验平台 | 第1/4/14章 |
| 最新默认入口 | run_ui_v3.ps1 → full_workbench UI → 本地11437 API | LOCAL_RUN_V3.md |
| 用什么模型 | 2B Q4写答案、.6B Q8生成1024维检索向量 | default.json与模型清单 |
| 论文规模 | 1585篇/19673块；原509目标缺失0 | FULL_DENSE_INDEX/TARGET_509 |
| 默认检索后端 | NumPy精确mmap向量+压缩BM25/RRF，非当前Chroma | full_retrieval.py |
| 三种模型是否混淆 | embedding编码，router选流程，generator写答案 | 第7/9/10章 |
| 当前有智能学习路由吗 | 没有；研究做过，默认是范围+任务规则 | 第10章 |
| 当前有LoRA吗 | 训练/恢复/同精度对照做过，默认不加载 | 第11章 |
| 普通QA改善多少 | 同170可答题F1 .3904→.4065，描述性小幅提升 | 第12章最终表 |
| 总结比较如何 | 20题F1 .4274→.4582，语义余弦.8125→.8594 | 同上，非人审真值 |
| 最突出短板 | 不可答仅3/30正确拒答；全库准确性仍弱 | 原始评分与检索 |
| 速度如何 | 热在线30题FINAL P50约1.187s/P95约1.988s | FINAL_ONLINE_RESIDENT8K |
| 95%准确率是真的吗 | 不是；旧.95是Recall@10，不是答案正确率 | V1/E0原始逐题 |
| 工程规模与测试 | HEAD405个.py/44275行；487静态测试函数；v3定向82通过 | repository_snapshot/XML |
| 能否说全面胜出 | 不能；保留退步项、失败与UNKNOWN | 第16章 |

**两分钟陈述示例。** 我把RAGScope从短段落检索实验扩展成长论文问答与评测平台，建立规范化原文坐标和可恢复索引，接通完整1585篇论文的全库、单篇和多篇问答。系统用BM25与1024维Dense融合，再按问题范围组织证据，通过本地量化2B生成结构化回答和可查看的引用。我用同题对照把检索覆盖、文字重合、语义接近、事实支持代理和真实在线成本拆开，发现重排、学习路由、LoRA并不自动改善质量，因此没有默认开启。最终总结比较在20题上改善，普通QA仅小幅上升，正确拒答和全库定位仍是短板。这些边界和原始失败都保存在账本里。

### 附录 B：完整受控仓库地图

以下列出本次HEAD的全部572个受控文件，不把环境/模型/缓存列进源码树。未跟踪的个人任务包、旧根架构说明、`rerun/`与`svg/`保留，但不计入这份受控文件地图；它们的初始状态保存在repository_snapshot。当前新增 `docs/architecture.md/html` 与 `_architecture/`属于本次文档交付，单列于树后。

```text
<!--REPOSITORY_TREE-->
```

本次文档资产：`docs/architecture.md`为单一正文来源；`docs/architecture.html`由它生成；`docs/_architecture/`含轻量统计、证据核验、离线转换、图源、预览和验收收据。阅读时无需运行脚本或访问网络。

### 附录 C：重要Python文件与函数索引

| 文件/位置 | 关键符号 | 读它可以回答什么 |
|---|---|---|
| `ragscope/schema.py:38` | Document / EvidenceUnit / Query / Chunk / ContextBundle | 各层数据身份、坐标和字段 |
| `ragscope/ingestion/qasper.py:38` | make_document / convert_question / duplicate_groups / freeze_splits | 数据怎样规范化、隔离、映射与冻结 |
| `ragscope/ingestion/cleaner.py` | clean_text / clean_documents / evidence_units | 原文结构与legacy清洗区别 |
| `ragscope/ingestion/chunkers.py:17` | TokenizerAdapter | 真实tokenizer与离线码点计数的区别 |
| `ragscope/ingestion/chunkers.py:175` | FixedSizeChunker | token预算、空白边界与overlap |
| `ragscope/ingestion/chunkers.py:208` | SentenceAwareChunker | 句子合并与超长句fallback |
| `ragscope/ingestion/chunkers.py:273` | StructureRecursiveChunker / SemanticChunker | 结构切块为何不是相似度切块 |
| `ragscope/ingestion/chunkers.py` | ParentChildChunker / _assemble | 父子映射、chunk ID和证据投影 |
| `ragscope/indexing/embedding.py` | EmbeddingService | 固定BGE、显式设备、真实长度限制 |
| `ragscope/indexing/local_embedding.py:15` | HFJsonTokenizer / LocalEmbeddingService | 旧本地CPU编码协议与query指令 |
| `ragscope/indexing/array_index.py:14` | ArrayCache / ArrayIndex | 哈希缓存、npy及V2精确检索 |
| `ragscope/indexing/dense_index.py` | DenseIndex | Chroma旧/兼容路径与manifest校验 |
| `ragscope/indexing/identity.py` | artifact / signature / atomic_json / verify_artifacts | 字节身份与原子写入 |
| `ragscope/v3/compressed_index.py:12` | CompressedBM25 / CanonicalIndex | BM25稀疏矩阵与规范块检索 |
| `ragscope/v3/hierarchical_runtime.py` | load_hierarchy / DocumentSparse | 文档层与旧scoped资产加载 |
| `ragscope/v3/full_retrieval.py:11` | FullCorpusRetriever / search | 当前全库/范围/RRF/文档先验 |
| `ragscope/v3/finish_runtime.py` | ReusableEncoder / task_kind / evidence_bundle / answer_messages | query编码、基础任务证据与Prompt |
| `ragscope/v3/deployment.py:10` | deployment_kind / DeploymentClient / deployment_context | 当前普通与复杂任务真实分支 |
| `ragscope/v3/core_packing.py:6` | pack_core_first | 保核心、限邻居，避免挤占 |
| `ragscope/generation/context.py:97` | pack_context / validate_bundle / render_context | 最终实际材料、预算、坐标去重 |
| `ragscope/generation/prompts.py` | visible_output_schema | 引用枚举约束的能力与局限 |
| `ragscope/generation/llm_client.py:41` | LocalLLMClient / chat / _request | 本地HTTP、设备授权、token一致性 |
| `ragscope/generation/pipeline.py:165` | RAGPipeline.parse_output | JSON合法性，非事实裁判 |
| `ragscope/v3/full_workbench.py:50` | FullWorkbench.answer | 当前问答总编排与应用缓存/日志 |
| `scripts/v3/serve_full_workbench.py` | main / Handler | API入口、启动校验和正常退出 |
| `ui/full_workbench.py` | api / main | 用户范围、任务、引用与历史回放 |
| `ragscope/generation/resources.py` | RequestResourceSampler | 性能/资源采样的测量边界 |
| `ragscope/evaluation/generation_metrics.py` | exact_match / token_f1 / rouge_l | 规范化、多参考与LCS公式 |
| `ragscope/evaluation/evidence.py` | evidence_metrics / merge_intervals | 证据覆盖而非文档命中 |
| `ragscope/v3/practical_quality.py` | fact_checks / numbers / polarity | 字面疑点，真值仍UNKNOWN |
| `ragscope/v3/nli.py` / `claim_metrics.py` | NLI与主张评分原语 | 高级事实评价研究，非默认在线 |
| `ragscope/routing/core.py:118` | build_proxy_labels / RuleRouter | 三路代理标签与规则 |
| `ragscope/routing/learned.py:32` | TfidfRouter | linear/MLP的特征与持久化 |
| `ragscope/routing/lstm.py:24` | LSTM路由实现 | 小序列分类器，而非生成器 |
| `ragscope/routing/laya_adapter.py:39` | Laya适配层 | 真实choice、置信阈值和fallback |
| `ragscope/training/data.py:39` | prepare_examples | train/dev隔离、assistant-only监督 |
| `ragscope/training/run.py:63` | discover_attention_targets / train_loop | LoRA目标、优化与训练循环 |
| `ragscope/training/checkpoint.py:26` | checkpoint状态保存/恢复 | 原子提交、RNG和样本位置 |
| `ragscope/training/evaluate.py:89` | paired评估 | 同BF16只切adapter的对照 |
| `ragscope/experiments/ledger.py` | RecordStore / read_records / append_jsonl | 逐条账本、哈希和续写 |
| `ragscope/experiments/runtime.py` | configure / require_ram / validate_gpu_lease | 子进程路径、内存与当前boot租约 |
| `scripts/v3/complete_dense.py` | main | 补齐完整矩阵、复用与位图恢复 |
| `scripts/v3/finish_experiments.py` | screen / original_client / questions | 收尾开发选型与固定客户端 |
| `scripts/v3/finish_evaluation.py` | 最终评分入口 | 答案重合/语义/证据/规则汇总 |
| `scripts/v3/finish_online.py` | 在线专项入口 | 热请求与缓存控制 |

这是导航索引，不意味着每个列出的模块都进入默认链路。完整当前调用顺序以第14章为准，完整受控路径以附录B为准。

### 附录 D：模型与运行环境定位

完整模型/环境表见第13章。所有相对 `models/`、`envs/`、`tools/` 路径均相对于 `D:/RAGScope_Runtime`，不是项目根。核心定位如下：

| 资产 | 实际定位/身份 |
|---|---|
| 当前生成manifest | `D:/RAGScope_Runtime/models/generation/text_only/qwen3.5_2b-q4_K_M.json` |
| 当前embedding | `D:/RAGScope_Runtime/models/qwen3-embedding-0.6b/Qwen3-Embedding-0.6B-Q8_0.gguf` |
| 私有Ollama | `D:/RAGScope_Runtime/tools/ollama-0.34.4/ollama.exe` |
| 当前默认解释器 | `D:/RAGScope_Runtime/envs/core/Scripts/python.exe`，Python3.12.13 |
| 训练解释器 | `D:/RAGScope_Runtime/envs/train/Scripts/python.exe`，BF16/CUDA研究 |
| 源依赖身份 | `configs/v3/dependencies.lock.json`及各env的实际metadata/锁文件 |
| 模型/训练证据核验 | `docs/_architecture/training_routing_evidence.json` |

“模型文件总和”“目录逻辑大小”“硬链接去重逻辑大小”“磁盘物理分配”“进程RAM”“设备VRAM”应分别统计。此前运行目录去重逻辑约32.35GB是历史整个运行资产口径，不是当前向量矩阵，也不是程序运行必需一次装入内存。该历史存储数字不能当成本次2026-10-15重新扫描的最新占用。

### 附录 E：数据与实验结果索引

<!--EXPERIMENT_INDEX-->

### 附录 F：V1—V3变化汇总

| 维度 | V1 | V2 | V3当前 |
|---|---|---|---|
| 主要语料 | SQuAD段落/arXiv摘要 | 509篇QASPER及独立技术/SQuAD | 同源全1585篇、原509全覆盖 |
| 证据对象 | 文档/chunk命中为主 | 独立EvidenceUnit与canonical区间 | 继续复用并记录完整实际上下文 |
| 索引 | BGE384+Chroma/BM25 | Qwen1024数组/范围索引 | mmap全块+压缩BM25+文档先验 |
| 切块 | 代码多种、实际短文难区分 | 长论文结构/父子/overlap受控实验 | R512默认，其他候选与历史保留 |
| 生成 | 有客户端/UI，早期无完整质量评测 | 四模型、Oracle/无材料/位置/切块对照 | 2B普通QA保守策略+任务证据组织 |
| 路由/训练 | 无已证实部署收益 | 三路代理、linear/MLP/LSTM/Laya；BF16 LoRA | 机制诊断、默认不用学习路由/adapter |
| 评价 | 检索Recall/MRR/nDCG | 答案F1/EM/ROUGE、覆盖与统计 | 另有BERTScore/NLI研究、最终余弦和规则 |
| 界面 | chat_app.py | chat_app_v2.py | full_workbench.py及独立API |
| 实测主要结论 | 短文本Hybrid召回较高 | 复杂模块有负结果，2B在所测协议更稳 | 总结比较改善较清晰，普通QA小增，拒答弱 |

### 附录 G：尚未解决的技术问题

1. 如何在不依赖题目给定论文的条件下，提高完整证据覆盖，并兼顾文档定位？
2. 如何校准“材料足够回答”与“需要拒答”，避免可答/不可答两端失衡？
3. 自动评价如何正确惩罚数字、否定和比较对象错误，同时认可正确复述？
4. 哪些LoRA退步来自输入分布、监督目标、停止条件、adapter开销或引用协议，各占多少？
5. 对“局部事实/跨段比较/全文总结”的语义分类，在足够监督数据下是否值得训练？旧三路标签不能回答这个问题。
6. 如何在固定预算内选择互补证据，而不只优化单块相关排序？
7. 多篇/中文/极长论文、并发与取消流程的真实用户验证还缺多少？
8. 新的独立题集和人工审核如何建立，避免在已经反复分析的题上继续选型？

这些是待研究问题，不是本次文档任务偷偷新增的开发目标。第16章记录了各项现有证据、影响和处理状态。

### 附录 H：面试前优先学习清单

**第一优先级：能手算。** 给两路排名算一次加权RRF；给参考/预测词集合算F1；给原文区间算交集覆盖；算1024维float32×19673行的存储；写出LoRA参数量r(m+n)与alpha/r缩放；说清8K窗口如何扣除prompt和输出预留。

**第二优先级：能沿代码讲。** 从UI payload追到FullWorkbench、deployment_context、search、pack_context、chat和parse；指出每个环节读取哪一份manifest、为什么不允许旧向量空间混入；解释为什么当前不是通过旧RAGPipeline.answer全包编排。

**第三优先级：能读一条失败。** 打开旧LoRA拒答题、一个全库漏证据题、一个重排后打包丢失题、一个数字规则疑点，分别检查原文、实际上下文、模型输出和参考。不要只背均值。

**第四优先级：能承认证据边界。** 把统计显著、工程容差、配置冻结、盲测、人工事实正确率和组件测试区分开。回答“为什么没有启用”时用具体实验，不用“高级技术不适合小数据”这种未经验证的总解释。

**第五优先级：扩展理论但不冒认实现。** 进一步学习ANN/HNSW与精确搜索取舍、cross-encoder训练目标、对比学习embedding、KL/校准与拒答、QLoRA的NF4/分页优化器、多证据组合优化。它们能帮助思考替代方案，但不应写成RAGScope已经实现。

### 附录 I：本手册的证据与生成方式

Markdown是HTML的单一正文来源，六幅SVG直接内嵌同一文本；HTML包含完整目录、响应式样式、系统中文字体与打印样式，无CDN、外部字体、在线图表或运行脚本依赖。

`repository_snapshot.json`保存源码计数/工作区初态和源码SHA；`experiment_audit.json`保存105个实验与配置来源、分母和逐条复算；`training_routing_evidence.json`保存训练/路由/环境的核验；`REFERENCE_STYLE_AUDIT.json`保存真实QuantMind范例结构；`CONSISTENCY_CHECK.json`保存MD→HTML章节、表格、代码、SVG、锚点和内容一致性；最终 `CONTENT_AUDIT.json`逐项对应用户要求，并声明未知事项。

本次只做资料核验和文档编写。没有把未进行的人评、未运行的模型、未完成的合并或被拒绝的应用浏览器验收写成完成；历史源码、模型、数据、报告和断点保留。
