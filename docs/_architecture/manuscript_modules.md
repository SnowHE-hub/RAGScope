## 6. Chunking 模块

### 6.1 共用约束比切块器名字更重要

所有当前切块器从同一份canonical文本切片，而不是先解码token再猜回原文。`BaseChunker._assemble`生成包含文档内容身份、切块配置和 `[start,end)` 的确定性chunk_id，并保存 `doc_id`、`chunk_index`、`prev_chunk_id`、`next_chunk_id`、`parent_id`、`unit_spans` 与tokenizer身份。原文变化、tokenizer变化、参数变化都会改变相应身份。

`[代码]` `ragscope/ingestion/chunkers.py` 的 `BaseChunker._assemble`。这是证据评分、邻块扩展、断点复用与引用可追溯共同依赖的基础；不能只存一段文本和一个自增ID就宣称具备相同可复核性。

### 6.2 各种方法怎样切

| 方法 | 实际算法与参数 | 优点/代价 | 在项目中的状态 |
|---|---|---|---|
| Fixed Size | 用真实tokenizer计数，寻找不超预算的字符终点；优先空白边界；可配置重叠 | 稳定易控；可能割裂语义 | `FixedSizeChunker:175`；V1主路径、V2/V3实验 |
| Sentence Aware | 正则句界分段后合并至max_tokens；超长句退固定切分；0/1句重叠，放不下时减少重叠 | 尽量不断句；句界不是完美语言学解析 | `SentenceAwareChunker:208`；有测试与长文对照 |
| Structure Recursive | 先按空行、标题、列表、代码fence等结构分段；超长结构退句子/固定切分；再合并，overlap0 | 保留结构；不是模型理解语义后切界 | `StructureRecursiveChunker:273`；当前R512来源 |
| Semantic兼容名 | `SemanticChunker = StructureRecursiveChunker` | 兼容旧调用名 | `chunkers.py:295`；**没有相邻embedding相似度阈值切界** |
| Parent-Child | 先建结构父块，在每个父块内部切小子块，给子块写parent_id | 小块定位、大块补语境；父块可能占掉预算 | `ParentChildChunker`；独立实验，当前默认不用 |
| Neighbor扩展 | 命中后加入同文档的前后块，再去重/预算筛选 | 补前后条件；可能挤掉后续高排名证据 | `pack_context(expansion='neighbor')`；普通指定论文问答仍用 |

当前R512使用Qwen embedding tokenizer、max_tokens512、结构overlap0。V3还准备F512_O0与P256_1024（child256、overlap32、parent1024），是候选研究来源，不是三种同时运行在每次用户请求上。

### 6.3 Overlap与token预算的细节

`_fit_end`以字符边界搜索并重新计token；BPE token数并非严格可加，因此代码不宣称找到全局最大填充。`_overlap_start`保证尾部重叠不超预算；若重叠阻止向前推进则减少重叠并记事件。Sentence方式的重叠单位是一句，不应写成“64token”。

检索块大小使用embedding tokenizer，ContextBundle默认参考计数器是 `cl100k_base`，生成器还会用自身聊天模板验证总输入。三个计数服务三个边界：**能编码、便于公平打包、能完整推理**。512块与2048上下文不能简单按“正好放四块”计算，标题、引用标签和分隔符也占token。

### 6.4 原始短文本为何几乎不需要切块

`[文档]` 旧架构核验记录22004/22029篇只有一块，约99.89%；原文token中位149、p99约372，大多低于400/450/512阈值。`[实测]` 原始切块资产的可复查路径及重建口径见第12章。这个观察只能支持“旧语料难以区分长文切块策略”，不能支持“所有任务的切块都无意义”。

长论文把问题变成多个证据分散在不同位置。即使文档命中，关键对照实验或限制条件也可能位于另一块。因此V2/V3新增证据单位坐标、邻块、父子和结构对照，关注真正送入模型的材料，而不只统计有无命中doc_id。

### 6.5 边界情况与可学习点

代码fence被当成结构边界保留，但超长代码仍可能被切开；正则句界可能误判缩写、公式和列表。结构标题只是线索，不是完整的论文逻辑树。保留所有原文坐标、记录fallback与超预算跳过，能暴露这些问题；它不保证块划分最优。`[推断]` 结构方式在本项目的优势主要是可解释、确定性与避免破坏原文，而不是已证明具有最强语义边界质量。

## 7. Embedding 与索引模块

<!--DIAGRAM:offline-->

### 7.1 旧BGE与当前Qwen是不同向量空间

V1使用 `BAAI/bge-small-en-v1.5` 的SentenceTransformer方式与Chroma cosine索引；该家族small向量维度384。旧权重具体revision没有完整保存，因此V2的E0重建固定官方版本并独立命名，不能冒称逐权重复现原始Dense部署。当前 `EmbeddingService` 会要求显式device、固定revision、真实tokenizer和输入长度，使用ArrayCache保存身份校验的npy。`[代码]` `ragscope/indexing/embedding.py`。

V2/V3主要使用 `Qwen/Qwen3-Embedding-0.6B`，当前全库空间是官方GGUF **Q8_0 /1024维/单位L2/float32输出/Ollama0.34.4/GPU**。词表和tokenizer revision固定为 `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`；权重SHA为 `06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439`。`[实测]` V3FINAL `FULL_DENSE_INDEX.json.embedding`。

模型元数据原生窗口32768，而本地embedding请求主动设4096，且 `truncate=False`。二者不是矛盾：**模型理论可接收长度与项目为资源/一致性选择的运行长度不同**。官方定义可查[Qwen embedding模型卡](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)；这里的实际维度、窗口和权重以本机清单为准。

### 7.2 文档编码与查询编码

文档端编码既有R512原文块；query端在问题前加：

```text
Instruct: Given a question about a scientific paper, retrieve relevant passages that answer the query
Query:<用户问题>
```

编码输出先检查形状 `(batch,1024)`、有限值与非零范数，再做L2归一化。检索只允许查询identity与索引embedding完全一致；不能把CPU旧服务或另一种量化模型的向量混进来，即使维度也为1024。`[代码]` `ragscope/indexing/local_embedding.py:12`、`ReusableEncoder`、`FullCorpusRetriever.search`。

语义评分的答案/参考编码不加这个检索问题前缀；因此“同一个embedding模型”并不等于“相同输入协议”。当前GGUF元数据包含pooling_type3，但代码只把pooling身份记录为后端模型定义；不能声称Python手写复现了HF模型全部pooling细节。

### 7.3 当前具体有哪些索引

| 层次 | Sparse | Dense | 当前用途 |
|---|---|---|---|
| 文档层1585篇 | 标题+摘要+章节名的BM25 | 对应文档表示矩阵 | 全库论文候选与文档先验 |
| R512正文19673块 | CompressedBM25 body | V3FINAL `vectors.npy` | 当前主块检索 |
| R512 metadata | 标题/章节前缀BM25 | 候选实现与构建状态另存 | 当前全库融合使用metadata BM25信号，不默认替代body Dense |
| 其他切块/早期scoped | 单独manifest与缓存 | 早期逐论文NPZ、候选矩阵 | 历史实验、共享/复用来源 |
| V1 | pickle BM25 | Chroma持久集合 | 旧入口/E0核验，不是最新默认数据库 |

`load_hierarchy`校验既有文档层、body sparse与早期scoped制品；`FullCorpusRetriever`再接上完整mmap矩阵，并验证向量/块行序、metadata sparse行序和模型身份。代码仍加载历史层的部分制品作为共享数据，不代表运行范围仍被早期141篇限制。`[代码]` `ragscope/v3/hierarchical_runtime.py`、`full_retrieval.py:11`。

### 7.4 压缩BM25与精确Dense搜索

`CompressedBM25.build`先用CountVectorizer生成词频，计算长度、IDF与BM25权重，再存成CSC稀疏矩阵。查询通过词表定位列，只把命中词的权重累加到相关行。正文表面看是BM25，工程上已不必每次重建一个巨大Python词频对象。`[代码]` `compressed_index.py:12`。

Dense矩阵形状 `(19673,1024)`，float32主体约80.58MB（含npy头80580736字节），并不是几十GB。`mmap_mode='r'`按文件映射加载；计算仍要访问参与运算的行，并不是零内存开销。全库矩阵点积是精确穷举，当前没有FAISS/HNSW近似检索的默认路径，因此不要在简历写“当前使用ANN向量数据库加速百万级检索”。

### 7.5 缓存身份、批量编码与断点

`ArrayCache`目录按完整model_identity哈希分隔，文件key又包含文本+模型身份；每个npy旁边有SHA/路径/identity清单。读取校验实际字节而不只相信文件名。模型名相同但revision、tokenizer、量化、后端等身份不同，不应复用。`[代码]` `ragscope/indexing/array_index.py:14`。

补全脚本 `scripts/v3/complete_dense.py` 采用16条一批、预分配mmap与完成位图；先复用通过校验的缓存，再编码缺失项。写入顺序是向量flush→缓存→完成位图原子替换→批次回执。恢复时已完成行要通过有限值与单位范数检查，行ID哈希必须一致。

本轮复用13045、新计算6628，合计19673；1585篇全部覆盖，原509目标缺失0。`[实测]` FULL_DENSE_INDEX、FULL_CORPUS_VALIDATION、TARGET_509_COVERAGE及批次记录。这个数字是索引完成度，不是检索正确率。

### 7.6 内存工程的实际处理

编码器驻留后，再为每批要求几GiB额外内存会重复计入已驻留开销。补全入口缩小启动请求量，保留系统reserve，并在可用内存低或每512个新向量时受控卸载**自有编码器**，让后端回收主机内存；可用内存低于512MiB时停止，保存断点。没有靠修改WSL内存上限、重启系统或结束未知用户任务完成。

当前 `ReusableEncoder.embed(cached=False)` 只跳过读缓存，仍会写新向量；不能把这个参数理解成“绝不写缓存”。最终在线实验改用专门 `UncachedOnlineEmbedding`，避免覆盖旧query缓存；受影响的24条缓存已按冻结数据逐位恢复，原SHA核验保留。`[代码]` `finish_runtime.py`、`online_runtime.py`；`[实测]` `QUERY_CACHE_RESTORATION.json`。这是需要读实现才能发现的缓存语义差异。

## 8. Retrieval 检索模块

<!--DIAGRAM:retrieval-->

### 8.1 最新融合公式与查询范围

`FullCorpusRetriever.search(question, scope, query_vector, embedding_identity, method, alpha, metadata_weight, document_weight, k, candidate_depth, balanced)` 是当前检索入口。`method`为bm25/dense/hybrid。默认方法hybrid；普通指定问答候选深度50，默认search返回至多50块。它先对允许的chunk行分别排序，再用加权RRF合并：

```text
chunk_score(c) = (1-alpha)/(60+rank_body_bm25)
               + alpha/(60+rank_body_dense)
               + metadata_weight/(60+rank_metadata_bm25)
               + document_weight * document_score(doc(c))
```

不在某路候选中的项没有该项贡献。`document_score`由文档BM25与Dense排名融合而来；纯Dense不加metadata sparse。当前全库普通问答选metadata_weight=.25、document_weight=1.0、alpha默认.5；指定复杂任务选alpha=.75。

**关键区别**：当前全库并不是“先取10篇，再只搜索这10篇”。它在全库块候选上融合，再加文档先验；Top文档仅是额外信号。早期 `HierarchicalRetriever` 的多级流程是另一份实现，不能统一描述成同一个硬级联架构。

### 8.2 全库、单篇、多篇

- **全库**：scope为空，参与行是19673块；计算文档候选，给候选块加入文档先验，返回论文候选供展示。
- **单篇**：scope中有明确doc_id，参与Dense乘法仅为该论文的行；跳过全库文档定位，不拿其他论文补答案。
- **多篇**：只在用户选定论文集合内检索；复杂工作流开启balanced，将各论文候选轮询交织，避免单篇占满前列。轮询并不保证每篇最终都有足够预算。

`FullWorkbench.answer`会拒绝不存在的论文；未选论文而问题含英文“this paper/the paper/this study”时返回CLARIFY_DOCUMENT。这个检查是有限的英文规则，不是完整指代消解；中文“这篇论文”或其他表达可能不被同一guard识别。`[代码]` `full_workbench.py:50`。

### 8.3 从“找到了”到“真的送给模型”

检索返回候选不等于候选全部进入上下文。`pack_context`按顺序处理候选：scope检查→邻块/父块扩展→同ID/同文档同文本去重→减去已覆盖canonical区间→加入标签标题后重计token→参考预算及模型预算检查→保留片段或记录跳过原因。

超预算候选通常是**整段跳过并继续看后续候选**，不是静默把末尾截掉。重叠移除可能生成带区间后缀的新引用ID并保留source_chunk_id，所有坐标同步投影。ContextBundle保存去重、truncations、unit_spans和context_hash，因此能查“证据检索到了，却为何没有实际输入”。`[代码]` `generation/context.py:97`。

普通指定问答保留旧neighbor打包，先处理一个核心及邻居再看下个核心，可能让邻居挤掉其他证据；新 `pack_core_first`先保核心，再给邻居使用不超过总预算25%的剩余空间，并验证核心未变。它属于部分新工作流，不可写成所有当前问答均使用core-first。

### 8.4 为什么总结不能只搜“summarize this paper”

泛泛的总结问题本身没有具体术语，直接按相关性可能只重复命中摘要和引言。当前指定论文summary先从原文块顺序中找起始块、方法相关标题、实验结果相关标题、结论/局限末端，再补剩余块；多论文交织，最多6144参考token，并实际检查8192总窗口可容纳768输出。

这是结构关键词与顺序启发式，不是训练了“知道全篇重点”的摘要规划器。UI显示source_character_coverage，覆盖低时不能声称读完全文；即使高覆盖也不证明重点理解正确。`[代码]` `deployment.py:25`。

### 8.5 重排负结果应该怎样解释

CrossEncoder通常在候选内估计问题-文本相关性，而本项目长论文题可能需要多段互补证据。重排改变顺序后，固定预算下进入模型的块也改变；邻块扩展可能进一步放大排序差异。**覆盖下降是可观察结果；“模型不擅长跨段证据”只是可能原因之一。** 还需控制候选池、token长度、章节、多段需求才能作因果判断。

历史V1 Hybrid+Rerank与Hybrid的Recall@10可同为.95，但MRR下降；V2/V3实际证据覆盖也出现退步，详见第12章。当前FullCorpusRetriever没有CrossEncoder调用，默认关闭重排。不能把“代码保留reranker”写成“在线有重排优化”。

### 8.6 用一道真实题模拟路径

最终H_QA中有题 `Which news corpus is used?`，范围论文为 `qasper:2002.11402`。查询编码加科学论文检索指令；scope限定该篇；hybrid=.5召回候选；G0/neighbor打包最多2048参考token；生成返回JSON，答案是私有英文新闻语料并提及Wikipedia标题。原始parsed与实际payload均在 `DEPLOYMENT_FINAL_ANSWERS_RESIDENT8K.jsonl`。

这个例子证明整条链路实际运行过，**不作为完美引用的示范**：引用ID合法不自动证明每个陈述都由对应段落支持。核验引用应把answer、citation pieces与参考原文同时阅读。正文没有为了展示而编造一段“理想模型输出”。

## 9. Generation 生成模块

### 9.1 模型与后端的分工

真实生成比较使用Qwen3.5-2B/4B/9B及V2的LFM2.5-1.2B；当前默认tag为 `ragscope-qwen3.5:2b-q4_K_M`，量化Q4_K_M，不加载LoRA。Qwen embedding .6B独立用于编码。模型权重、tokenizer、模板和后端版本清单见第13章及附录D。

当前服务由 `scripts/v3/serve_full_workbench.py` 创建私有Ollama0.34.4进程，后端11435、API11437、UI8502，均loopback。原系统11434服务不是当前工作台的默认调用目标。生成请求使用 `stream=False,think=False,temperature=0,seed=42`；这些设置降低随机性，不等于所有硬件/后端结果逐位确定。

### 9.2 Prompt与Chat Template怎样连接

`answer_messages`决定系统指令和用户证据；普通指定问答采用G0，要求仅用所给上下文、把资料视为数据、证据不足时拒答、输出JSON。新summary/comparison指令要求区分方法、条件、结论和局限；未提供的局限不应编造。`finish_runtime.py`。

消息列表送入 `DeploymentClient`，它继承ProtocolClient/CapturedClient/LocalLLMClient，保存actual_sent_payload与raw backend response。`original_client`从已固定的V2模型包构建token计数器；chat template由本地模型/包模板渲染，模型身份记录chat_template_hash。**Python中的system/user文本不是完整模型输入；角色包装也要计数。** `scripts/v3/finish_experiments.py`、`ragscope/generation/llm_client.py:113`。

### 9.3 JSON schema与引用检查

```json
{"answer":"根据证据形成的回答", "citations":["本次上下文里可见的ID"], "abstain":false}
```

这是输出结构示意，不是某次实测答案。schema只允许answer/citations/abstain三个字段，引用元素枚举本次实际piece IDs；无piece时maxItems=0。多篇输入额外限制最多4条引用、answer最多1600字符，并要求按论文清晰作答，以降低768输出token内JSON截断风险。

模型输出经 `RAGPipeline.parse_output:165` 检查类型、JSON与abstain布尔值，FullWorkbench再检查引用在实际上下文内，以及token_count_match必须为True。当前并没有一个强模型逐条判断这些引用蕴含了答案。合法率高具有schema构造因素，不能宣传成“引用100%正确”。

### 9.4 长度预算与任务差异

| 默认任务 | 证据策略 | 参考证据token | 输出token上限 | 后端窗口 |
|---|---|---:|---:|---:|
| 单篇事实/解释 | B0普通hybrid+neighbor、G0 | 2048 | 512 | 8192 |
| 全库普通问题 | 文档先验/metadata融合，core-first打包 | 2048 | 512 | 8192 |
| 总结/比较或多篇 | 结构/范围平衡，原文片段 | 6144 | 768 | 8192 |

`LocalLLMClient.chat`先检查完整prompt+输出预留≤窗口，并比较后端prompt_eval_count与本地计数。6144是参考token上限，不是实际必定塞满；超长片段被跳过。长证据增加prefill与KV内存、输出更长增加decode耗时，仍可能发生注意力分散或比较方向混乱。`[代码]` `deployment.py:25`、`llm_client.py:113`；`[实测]` 最终payload与token_count_match记录。

### 9.5 错误、拒答与降级的真实行为

“不可答”应是有意义的 `abstain=true`，不是请求超时或解析失败。当前应用把异常写进独立APPLICATION账本并返回EXECUTION_ERROR；无效输入、未知论文和有限英文指代分别返回INPUT_ERROR/INVALID_DOCUMENT/CLARIFY_DOCUMENT。客户端无隐藏重试，错误不会悄悄改用Mock。

服务离线时UI可读保存的catalog与实验历史，但生成按钮禁用；这是界面离线降级，不是模型回答降级。索引或源哈希不符时启动拒绝，不自动重建旧文件。早期Workbench曾有BM25零分NO_EVIDENCE预判，**新FullWorkbench没有相同的零分前置分支**，不要跨版本合并行为。

最终220条答案0生成失败，不代表应用永不报错；真实应用验收保留过1次多篇JSON截断，随后限制输出/引用并复测场景通过。30不可答仅3正确拒答，说明prompt要求没有自动转化成可靠行为。Mock仅为功能替身，`real_model=False`，正式 `RAGPipeline`会拒绝用它产出模型成绩。
