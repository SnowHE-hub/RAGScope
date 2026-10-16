## 17. 面试问题与个人学习清单

这些问题用于练习把原理、实际实现和证据边界连起来。回答时先说“我做了什么”，再说“哪些条件下有效”，最后承认尚未验证的部分。下面的源码路径绑定本次手册版本；没有定位到稳定函数行号的条目只列相关模块，不编造行号。实验目录采用第0章定义的 `V3FINAL`、`V3EARLY`、`V2RUN`；LoRA与路由原始目录见第11章和附录E。

### Q01 怎样在两分钟内介绍这个项目？

- **面试官问法：** “RAGScope解决什么问题？与普通聊天机器人有什么区别？”
- **通俗答案：** 它帮助我阅读本地论文：先找原文，再根据原文回答，并把引用展示出来。我同时建设了实验记录，能追踪每道题到底检索到什么、输入模型什么、回答是否失败。
- **技术答案要点：** 外部论文语料、结构切块、BM25与Dense融合、范围和任务控制、量化本地生成、JSON引用及逐条实验账本组成可复核链路。检索与生成解耦，真实模型运行和功能测试分开。
- **项目真实例子：** [实测] 当前覆盖1585篇/19673块，支持全库、单篇和多篇范围；原509篇目标缺失0。它尚不是已经验证可靠的事实裁判，也不默认支持任意PDF上传、OCR或图表理解。
- **源码位置：** `ragscope/v3/full_retrieval.py`、`ragscope/v3/full_workbench.py`、`ui/full_workbench.py`、`configs/v3/default.json`。
- **还需补充知识：** RAG与参数化知识的边界、实际用户需求评估、从本地实验系统走向生产服务所需的鉴权与并发能力。
- **现有证据是否足够：** 足够说明系统已实现并实际运行；不足以承诺任意论文、任意问题都能准确回答。

### Q02 一次用户提问怎样沿真实代码执行？

- **面试官问法：** “不要只画框图，按实际调用链讲一下。”
- **通俗答案：** 页面收集问题与论文范围，服务检查范围，检索相关材料并控制长度，模型写出答案，最后校验格式和引用，再展示原文。
- **技术答案要点：** 当前入口是FullWorkbench而非把历史RAGPipeline全流程直接套入；查询编码与索引身份必须一致，任务策略决定普通或复杂上下文，生成客户端保存实际请求，解析复用旧Pipeline的输出检查。
- **项目真实例子：** [实测] 最终真实API验收包含全库问题、超出旧141篇范围的论文、多篇比较、总结、无效输入及缓存场景。组件测试也真正请求过模型，浏览器视觉验收另有权限限制，不能混为一谈。
- **源码位置：** `ui/full_workbench.py` → `scripts/v3/serve_full_workbench.py` → `ragscope/v3/full_workbench.py` → `full_retrieval.py` / `deployment.py` / `finish_runtime.py` → `ragscope/generation/llm_client.py`；解析为 `RAGPipeline.parse_output`。
- **还需补充知识：** HTTP请求生命周期、故障传播、为什么同步本地服务不能直接宣称支持高并发。
- **现有证据是否足够：** 足够追踪当前默认链路；不支持生产级吞吐或多用户隔离的结论。

### Q03 为什么早期Recall接近0.95，现在答案F1却只有约0.4？

- **面试官问法：** “功能增加这么多，怎么分数反而低了？”
- **通俗答案：** 以前测的是能不能找回正确资料，现在测的是能不能读懂长论文并写出参考答案。找到书与答对考试不是同一个难度，也不是同一个分数。
- **技术答案要点：** V1短段落文档Recall@k与V3长论文生成token F1的对象、数据分布和评价阶段均不同。长文还需要多证据覆盖、预算打包和生成，误差会逐层累积。
- **项目真实例子：** [实测] V3最终同题对照中，普通可答QA F1为0.3904→0.4065，总结/比较0.4274→0.4582；这些才是相同条件下的有效比较，不拿它们与SQuAD文档Recall作升降百分比。
- **源码位置：** `ragscope/evaluation/retrieval_metrics.py`、`ragscope/evaluation/generation_metrics.py`、`ragscope/v3/practical_quality.py`；原始结果 `V3FINAL/FINAL_SUMMARY_RESIDENT8K.json`。
- **还需补充知识：** 检索指标与端到端指标、难度迁移、指标量纲及比较条件。
- **现有证据是否足够：** 足够证明两类数字不可直接比较，并描述本轮小幅增益；不足以证明项目整体已达到高可靠质量。

### Q04 为什么要切块，块越小是不是越精确？

- **面试官问法：** “直接把整篇论文向量化不好吗？chunk size怎么选？”
- **通俗答案：** 整篇像一本书的简介，容易丢掉具体细节；小块更容易定位，但太小又会把原因、条件和结果拆散。所以块大小是定位精度与上下文完整性的取舍。
- **技术答案要点：** 切块影响向量语义粒度、BM25长度分布、候选冗余和上下文预算。不能只按检索命中选chunk size，还要检查实际输入的完整参考证据覆盖与生成质量。
- **项目真实例子：** [代码/实测] 当前R512由结构切分、Qwen tokenizer、max_tokens512、overlap0构建；另有固定、句子及父子候选。V1语料约99.89%一文一块，因此旧短文实验难以验证长论文切分策略。
- **源码位置：** `ragscope/ingestion/chunkers.py` 的 `FixedSizeChunker`、`SentenceAwareChunker`、`StructureRecursiveChunker`、`ParentChildChunker`。
- **还需补充知识：** token长度分布、检索粒度与答案所需证据粒度、长文分层表示。
- **现有证据是否足够：** 足够说明当前算法与研究取舍；不足以宣称512是所有任务的最优大小。

### Q05 项目里的“Semantic Chunker”真的用语义相似度切界吗？

- **面试官问法：** “你怎么计算相邻段落的语义边界？”
- **通俗答案：** 当前没有做那种计算。代码中的兼容名称叫Semantic，但实际按段落、标题、列表等结构切分，太长再按句子或固定长度切。
- **技术答案要点：** 结构启发式与embedding相似度阈值切界是不同算法。名称不能作为实现证据；需要指出真正的分段规则以及fallback。
- **项目真实例子：** [代码] `SemanticChunker = StructureRecursiveChunker`。没有相邻句向量距离、动态阈值或边界模型，不能在简历写“实现相似度驱动语义切块”。
- **源码位置：** `ragscope/ingestion/chunkers.py` 的 `StructureRecursiveChunker` 与 `SemanticChunker`兼容别名。
- **还需补充知识：** 真正语义分段的距离度量、阈值选择、主题转折评估及其额外编码成本。
- **现有证据是否足够：** 足够回答当前真实实现；不具备相似度驱动切块的实验依据。

### Q06 Overlap有什么作用，为什么还要保存原文坐标？

- **面试官问法：** “相邻块重复了，怎么避免重复证据占满窗口？”
- **通俗答案：** 重叠避免一句话被边界切坏，但搜索后不能把同一段反复塞给模型。原文坐标相当于给每段标页码和位置，方便去重、引用与检查覆盖。
- **技术答案要点：** 块保存canonical `[start,end)`、文档和内容身份；打包时减去已覆盖区间并重新投影证据span。token边界与字符边界不同，BPE也不是简单可加，必须按实际文本重新计数。
- **项目真实例子：** [代码] ContextBundle保留去重、跳过原因、unit_spans与context_hash；部分重叠移除后生成新区间引用ID并保留source_chunk_id，而不是复制文本后丢掉原始位置。
- **源码位置：** `ragscope/ingestion/chunkers.py` 的 `BaseChunker._assemble`、`_overlap_start`；`ragscope/generation/context.py` 的 `pack_context`。
- **还需补充知识：** 区间并集/差集、Unicode坐标、不同tokenizer下重计数与证据映射。
- **现有证据是否足够：** 足够说明可追溯和去重机制；不能保证引用区间中的文字一定蕴含答案。

### Q07 Embedding模型与生成模型有什么区别？

- **面试官问法：** “为什么需要两个Qwen模型？一个不能全部做完吗？”
- **通俗答案：** 一个负责把文字变成适合搜索的坐标，另一个负责读材料后写答案。就像图书馆的索引员与回答问题的人，工作目标不同。
- **技术答案要点：** embedding输出固定维连续表征，用于相似度；生成器逐token输出词表分布。训练目标、输入协议、缓存、模型身份及评估指标均不同。
- **项目真实例子：** [实测] 当前Qwen3-Embedding-0.6B Q8_0输出1024维归一化向量；Qwen3.5-2B Q4_K_M生成JSON。BGE-small历史384维不能直接放进当前1024维索引。
- **源码位置：** `ragscope/indexing/local_embedding.py`、`ragscope/v3/finish_runtime.py` 的 `ReusableEncoder`、`ragscope/generation/llm_client.py`。
- **还需补充知识：** 双塔表征学习、对比学习、生成式语言建模与embedding任务的区别。
- **现有证据是否足够：** 足够说明角色和实际身份；没有自行训练基础embedding或生成模型的证据。

### Q08 查询与文档为什么要用不同的编码输入？

- **面试官问法：** “query instruction有什么用？文档也加同一句吗？”
- **通俗答案：** 问题和答案段落写法不同。给编码器说明“我要找能回答科学论文问题的段落”，可以让它按检索任务理解问题；原文段落则保持相应文档端协议。
- **技术答案要点：** 指令式embedding常采用非对称query/document输入。前缀是编码协议的一部分，必须随模型身份和缓存记录，不能随意改变后还复用旧向量。
- **项目真实例子：** [代码] 查询前加 `Instruct: Given a question about a scientific paper, retrieve relevant passages that answer the query\nQuery:`；当前正文Dense为R512 body。答案语义相似度编码不加这个检索问题前缀。
- **源码位置：** `configs/v2/retrieval.json` 的 `query_instruction`；`ragscope/indexing/local_embedding.py`、`ragscope/v3/practical_quality.py`。
- **还需补充知识：** 非对称检索、任务指令敏感性、同模型不同输入协议的校准。
- **现有证据是否足够：** 足够回答实际输入；本轮没有单独前缀消融，不能声称该前缀贡献了具体提升。

### Q09 为什么向量归一化后可以直接做点积？

- **面试官问法：** “你用的是余弦距离还是内积？二者什么时候等价？”
- **通俗答案：** 如果把每个向量长度都缩成1，比较点积就只是在比较方向，相当于余弦相似度。
- **技术答案要点：** `cos(q,d)=q·d/(||q||||d||)`；单位L2归一化时分母为1。先检查维度、有限值和非零范数，否则NaN或零向量会破坏排序。
- **项目真实例子：** [代码/实测] 最终矩阵19673×1024、float32，全部有限且单位L2；FullCorpusRetriever在允许行上做NumPy矩阵乘，当前没有默认FAISS/HNSW近似搜索。
- **源码位置：** `ragscope/v3/full_retrieval.py`、`ragscope/v3/finish_runtime.py`；`V3FINAL/FULL_DENSE_INDEX.json`、`FULL_CORPUS_VALIDATION.json`。
- **还需补充知识：** 高维空间、归一化与向量模长信息、精确搜索和近似最近邻的成本差异。
- **现有证据是否足够：** 足够说明当前实现；不支持百万级近似检索或召回损失的性能结论。

### Q10 为什么缓存不能只按“模型名+文本”存？

- **面试官问法：** “同一个Qwen模型，为什么还要记录那么多哈希？”
- **通俗答案：** 同名模型也可能换权重、量化、分词器或运行方式。旧坐标与新坐标混用后，搜索还能跑，但结果可能悄悄错掉。
- **技术答案要点：** 缓存身份包含revision、权重SHA、tokenizer、量化、后端、归一化等；索引还要固定chunk行序与语料身份。完整性校验应读实际文件字节，不只看文件名。
- **项目真实例子：** [实测] 补全索引复用13045个通过身份校验的向量、新算6628个；一次在线实验误写24条旧query缓存后已按冻结向量逐位恢复，并保存原SHA核验。`cached=False`在ReusableEncoder中并不等于绝不写缓存。
- **源码位置：** `ragscope/indexing/array_index.py` 的 `ArrayCache`；`scripts/v3/complete_dense.py`；`ragscope/v3/online_runtime.py` 的 `UncachedOnlineEmbedding`；`QUERY_CACHE_RESTORATION.json`。
- **还需补充知识：** 内容寻址、缓存失效、数据版本与计算身份、确定性和数值近似的边界。
- **现有证据是否足够：** 足够说明已发生的问题和修复；不能保证不同后端仅因维度相同就兼容。

### Q11 BM25为什么不只是统计关键词出现次数？

- **面试官问法：** “TF-IDF和BM25有什么区别？k1、b控制什么？”
- **通俗答案：** 生僻而关键的词应更重要；重复太多次不应无限加分；长文也不能只因为词多就占优势。BM25把这几件事放进同一个打分公式。
- **技术答案要点：** IDF体现词的区分度；词频饱和由k1调节；长度归一由b调节。TF-IDF是相关思想，但当前正文检索不是路由TF-IDF分类器。
- **项目真实例子：** [代码] 当前压缩BM25使用k1=1.5、b=.75；按词表列累加预计算权重，负IDF按平均值修正。标识符或精确数据集名可能适合稀疏信号，但不是所有题都胜过Dense。
- **源码位置：** `ragscope/v3/compressed_index.py` 的 `CompressedBM25`；`configs/v2/retrieval.json`；历史 `ragscope/retrieval/bm25_retriever.py`。
- **还需补充知识：** BM25推导、词频饱和、IDF变体、分词和领域词表的影响。
- **现有证据是否足够：** 足够说明参数与实现；没有本轮k1/b系统寻优的证据。

### Q12 为什么Hybrid用RRF，而不是直接把BM25和余弦分数相加？

- **面试官问法：** “融合公式是什么？alpha有什么含义？”
- **通俗答案：** 两个检索器的分数刻度不一样，直接相加容易被数值更大的那一路主导。RRF先看排名，再给靠前结果加分。
- **技术答案要点：** 当前 `Σ w_i/(60+rank_i)`，rank从1开始；alpha是Dense权重，另一路为1-alpha。未入某路候选的对象没有该路贡献；常数60不能说是本轮发现的最优值。
- **项目真实例子：** [代码] 普通指定论文QA alpha=.5；复杂指定任务用.75；全库还加入metadata BM25和文档先验。它们是不同任务配置，不是全项目只有一个融合权重。
- **源码位置：** `ragscope/v3/full_retrieval.py`；`ragscope/retrieval/hybrid_retriever.py`；`ragscope/v3/deployment.py`。
- **还需补充知识：** 排名融合、分数归一化、RRF常数与候选深度的联动。
- **现有证据是否足够：** 足够解释公式和当前配置；Hybrid的互补原理不等于每道题都提升。

### Q13 当前全库检索是不是“先找10篇，再在这10篇里找段落”？

- **面试官问法：** “你做的是哪种分层检索？第一阶段错了怎么办？”
- **通俗答案：** 当前默认不是这种硬筛选。块候选仍从全库产生，论文层结果只是给相关论文中的块增加先验信号。
- **技术答案要点：** 区分硬级联候选裁剪与软文档先验融合。前者第一阶段漏文档后无法恢复；后者仍有全库块候选，但先验可能改变排序，也会有误导风险。
- **项目真实例子：** [代码/实测] FullCorpusRetriever在19673块上融合并加document_score；早期HierarchicalRetriever是另一实现。最终Hybrid完整证据覆盖11.18%→17.65%，但doc_hit@10为40.5%→38%，不能只报有利指标。
- **源码位置：** `ragscope/v3/full_retrieval.py`、`ragscope/v3/hierarchical_runtime.py`；`V3FINAL/FINAL_GLOBAL_SUMMARY.json`。
- **还需补充知识：** 级联误差、先验校准、候选多样性、文档命中与多段证据覆盖的关系。
- **现有证据是否足够：** 足够说明最新默认架构及结果；不足以称全库检索准确性已解决。

### Q14 Cross-Encoder与双塔Dense有什么不同？

- **面试官问法：** “既然重排更精细，为什么不用它直接搜索全库？”
- **通俗答案：** Dense可以提前给所有段落算坐标，提问时只算一次问题；Cross-Encoder要把问题和每个候选一起重新阅读，更细但更贵。
- **技术答案要点：** 双塔独立编码，支持预计算；交叉编码器允许query与document token交互，最终分依赖这对输入，不能预先为每个文档存一个固定相关性分数。一般只在候选集上重排。
- **项目真实例子：** [代码/实测] BGE-reranker用于V2真实重排研究，另保留Qwen3-Reranker研究资产；当前FullCorpusRetriever不调用Cross-Encoder。
- **源码位置：** `ragscope/retrieval/reranker.py`、`ragscope/retrieval/verified_reranker.py`、`ragscope/v3/full_retrieval.py`。
- **还需补充知识：** cross-attention与序列分类目标、候选深度和token截断、reranker分数校准。
- **现有证据是否足够：** 足够解释角色与项目状态；资产存在不等于候选重排已用于默认服务。

### Q15 重排为什么反而降低了证据覆盖？

- **面试官问法：** “是你代码写错，还是模型不行？”
- **通俗答案：** 重排器可能把单段很相关的内容排前面，却挤掉回答所需的另一段。模型最终能读的材料有长度上限，排序变了，实际证据也会变。
- **技术答案要点：** 候选相关性、集合互补性、邻块扩展与固定预算打包不是同一个目标。需要分别控制候选池、顺序、长度、章节和多证据需求，才能定位因果。
- **项目真实例子：** [实测] V2路线重放dev60中，Hybrid证据覆盖0.767857、F1 .429777；重排覆盖.663690、F1 .370988。覆盖下降是实测，“模型不懂跨段关系”只是推断，不能直接归咎外部模型bug。
- **源码位置：** `ragscope/retrieval/verified_reranker.py`、`ragscope/generation/context.py`；`20261014_v2_routing/REPLAY_fixed_hybrid*.json`。
- **还需补充知识：** 集合选择、多样性、证据互补性评价、相关性与可回答性的差别。
- **现有证据是否足够：** 足够解释为何默认关闭；不足以确定唯一退步原因或证明所有重排无效。

### Q16 检索命中了证据，为什么模型还是答不出来？

- **面试官问法：** “你怎么区分retrieval failure和generation failure？”
- **通俗答案：** 搜到的段落不一定真正进入模型；进入了也不代表信息完整；材料完整也不代表模型读对。所以必须把每个环节分别记录。
- **技术答案要点：** 候选召回→实际ContextBundle→完整参考证据覆盖→生成输出→事实支持是不同判定。Oracle context用于去掉检索误差，但只隔离特定条件，不能变成部署成绩。
- **项目真实例子：** [代码/实测] pack_context可因预算整段跳过，neighbor也可能挤掉后续核心；ContextBundle保存跳过原因和unit_spans。当前未用模型事实判断把所有答案自动判真。
- **源码位置：** `ragscope/generation/context.py` 的 `pack_context`；`ragscope/evaluation/evidence.py`；`ragscope/v3/practical_quality.py`。
- **还需补充知识：** 分阶段错误归因、必要与充分证据、oracle实验的控制变量。
- **现有证据是否足够：** 足够定位一些失败层次；自然语言事实支持仍有UNKNOWN，不能把全部错误完整自动归因。

### Q17 Parent-Child与邻块扩展解决同一个问题吗？

- **面试官问法：** “为什么做两种扩展？当前哪种在用？”
- **通俗答案：** 两者都补语境，但父子方式找小块后返回它所属的大段，邻块方式补前后内容；补得太多都会占掉别的证据位置。
- **技术答案要点：** parent_id描述预定义父块；prev/next描述同文档相邻块。扩展后的预算、去重和证据覆盖必须重算。更大上下文不是无成本的质量保证。
- **项目真实例子：** [代码] 项目保留P256_1024等父子候选，当前默认R512没有父块替换；普通指定论文QA仍用neighbor，部分新工作流采用core-first并限制邻居剩余预算。
- **源码位置：** `ragscope/ingestion/chunkers.py` 的 `ParentChildChunker`；`ragscope/generation/context.py`；`ragscope/v3/finish_runtime.py`。
- **还需补充知识：** 层级片段检索、窗口扩展与信息冗余、预算约束下的证据选择。
- **现有证据是否足够：** 足够区分实现与默认状态；不能把父子检索写成最新每道题都执行的核心步骤。

### Q18 总结与事实问题为什么不能共享完全相同的检索流程？

- **面试官问法：** “把top-k调大不就能总结了吗？”
- **通俗答案：** 事实题常只需要一个局部答案，总结要覆盖方法、实验和局限。泛泛的总结问题容易反复搜到摘要，top-k更大也未必覆盖整篇。
- **技术答案要点：** 任务所需证据范围不同。当前summary按原文顺序、方法/实验/结论等结构线索选块，多篇交织，预算6144参考token、768输出；这不是训练出来的摘要规划器。
- **项目真实例子：** [实测] 最终20道总结/比较题关键点进入上下文的比例57.45%→100%，文字F1 .4274→.4582；参考内容由AI构造并校验原文，引入小样本和非人工审核局限，100%也不是答案正确率。
- **源码位置：** `ragscope/v3/deployment.py` 的 `deployment_context`；`ragscope/v3/full_workbench.py`；`V3FINAL/FINAL_SUMMARY_RESIDENT8K.json`。
- **还需补充知识：** 摘要覆盖与显著性、跨文档平衡、claim/keypoint评估及人工标注。
- **现有证据是否足够：** 足够证明小规模任务的证据与代理分数改善；不足以推广为所有论文的高质量全文总结。

### Q19 JSON schema和引用枚举能消除幻觉吗？

- **面试官问法：** “引用合法率100%，是不是答案都有依据？”
- **通俗答案：** 它能要求模型按格式写，并只引用给过的编号；但模型仍可能引用对了编号、解释错了内容。
- **技术答案要点：** 结构约束、引用可见性、引用蕴含性是三层。schema控制语法和可选ID；实际事实支持需要把具体claim与对应证据比较，甚至人工审核。
- **项目真实例子：** [代码/实测] 当前输出只有answer/citations/abstain，多篇限制最多4引用与1600字符；应用曾保留1次JSON截断失败，修正后复测通过。未宣称引用enum实现了自动事实裁判。
- **源码位置：** `ragscope/v3/full_workbench.py`、`ragscope/generation/pipeline.py` 的 `RAGPipeline.parse_output`、`ragscope/v3/practical_quality.py`。
- **还需补充知识：** 约束解码、schema表达能力、claim-to-evidence entailment、引用精确率与完整率。
- **现有证据是否足够：** 足够证明格式与可见性检查；不足以证明全部引用支持全部答案陈述。

### Q20 为什么写了“证据不足就拒答”，模型仍会勉强回答？

- **面试官问法：** “你们的拒答效果怎么样？怎么评？”
- **通俗答案：** 提示词只是要求，模型未必能准确判断自己有没有足够证据。要专门测不可答题，不能把格式失败当拒答，也不能把所有拒答都当正确。
- **技术答案要点：** 不可答的answer F1保留N/A，单独评correct abstention；可答但拒答还要结合当前材料充分性，不能只凭全论文有答案判定模型胡乱拒答。
- **项目真实例子：** [实测] 最新H_QA30道不可答仅3道正确拒答，10%没有改善。旧LoRA在92道可答题中新拒答23次，但其中一些检索材料不足，人工充分性审核未完成。
- **源码位置：** `ragscope/evaluation/generation_metrics.py`、`ragscope/routing/core.py` 的 `proxy_quality`、`ragscope/v3/practical_quality.py`。
- **还需补充知识：** 选择性预测、置信度校准、answerability与evidence sufficiency、拒答precision/recall的成本取舍。
- **现有证据是否足够：** 足够确认拒答仍是主要短板；没有可靠拒答阈值或事实安全保证。

### Q21 Q4、Q8和BF16分别表示什么，为什么不能直接比较它们的效果？

- **面试官问法：** “你证明了量化导致模型变差吗？”
- **通俗答案：** 它们是保存或计算数值的不同精细程度。更低位数通常更省空间，但不能只看到两次分数不同，就说一定是量化造成，因为后端、提示和格式约束也可能变了。
- **技术答案要点：** GGUF Q4_K_M/Q8_0是量化方案，不表示激活和KV cache全是4/8位；BF16 LoRA用Transformers eager，Q4部署用Ollama且有schema。要隔离量化需要同权重来源、协议、任务、上下文及后端尽可能匹配。
- **项目真实例子：** [代码/实测] 默认生成Q4、embedding Q8，真实LoRA训练BF16；同BF16基础/adapter比较能隔离adapter开关，而跨Q4/BF16的差异不能全部归因于精度。
- **源码位置：** `ragscope/training/run.py` 的 `load_training_model`；`ragscope/training/evaluate.py` 的 `BF16ModelClient`；`RUNTIME/models/generation/text_only/`清单。
- **还需补充知识：** 量化误差、分组缩放、混合精度、KV cache精度与校准数据。
- **现有证据是否足够：** 足够说明实际表示和实验混杂；没有单因素量化质量损失结论。

### Q22 GPU显存为什么不等于模型文件大小？

- **面试官问法：** “1.27GB的2B模型为什么仍需要更多运行内存？”
- **通俗答案：** 文件只是模型参数。运行还要放中间结果、上下文记忆和计算工作区，训练又要加梯度、优化器和保存状态，所以不能按文件大小算显存。
- **技术答案要点：** 推理包含权重、KV cache、激活和临时workspace；训练包含反向激活、adapter梯度和optimizer state。Torch allocated/reserved、物理VRAM、进程RSS是不同口径，Windows还可能涉及共享内存。
- **项目真实例子：** [实测] BF16 LoRA峰值allocator allocated约7.61GB、reserved约8.28GB，原报告明确不是物理显存计量。最新服务只串行调用一个生成模型与独立encoder，不借此宣称模型体积等于驻留开销。
- **源码位置：** `ragscope/training/run.py` 的 `train_loop`资源记录；`ragscope/v3/local_runtime.py`、`ragscope/v3/finish_runtime.py`；LoRA `TRAINING_AUDIT.json`。
- **还需补充知识：** KV cache随层数/序列长度增长的估算、allocator缓存、prefill/decode资源差异。
- **现有证据是否足够：** 足够解释测量边界；没有把所有瞬时峰值和物理分页行为完整剖析的证据。

### Q23 为什么统一8K窗口反而解决了一次严重卡顿？

- **面试官问法：** “窗口变大不是更慢吗？你们究竟优化了什么？”
- **通俗答案：** 问题不只是读得多，而是普通题和总结题切换不同窗口时，服务反复重新加载模型。统一驻留设置后，避免了这部分重复等待。
- **技术答案要点：** 后端模型驻留配置变化可能触发reload；区分实际输入长度与num_ctx配置，区分冷启动与热请求，查看backend load_duration才能定位卡顿，而不只比较总时长。
- **项目真实例子：** [实测] 混用4K/8K版本FINAL在线P95约10.13秒；统一8192后约1.99秒。但相同运行条件的基线P95约1.48秒，所以结论是修复重载卡顿，不是最终配置全面更快。
- **源码位置：** `ragscope/v3/deployment.py`、`scripts/v3/finish_online.py`；`V3FINAL/FINAL_ONLINE_SUMMARY_RESIDENT8K.json`及旧在线日志。
- **还需补充知识：** 冷热分布、模型生命周期、后端缓存与配置键、P95在小样本中的不稳定性。
- **现有证据是否足够：** 足够定位这次重载问题和修复效果；不足以给出长期生产SLA或TTFT承诺。

### Q24 为什么分多个Python环境，core没有Torch怎么用GPU？

- **面试官问法：** “多环境是不是把依赖管理搞复杂了？”
- **通俗答案：** 页面、CPU历史模型和新训练库需要不同依赖。页面只发送本地请求，真正的GPU计算由另一个模型服务进程执行，所以页面环境不必安装Torch。
- **技术答案要点：** core负责协调/UI；models保留CPU Torch2.9.1与Transformers4.57.6；train使用cu128 Torch和Transformers5.19.0/PEFT。metrics环境通过只读.pth复用底层依赖，不是各复制一套完整Torch。
- **项目真实例子：** [资产核验] 七个环境均Python3.12.13。train当前47直接包，对应 `requirements.r3_gpu.lock.txt`；旧准备文档46包与旧lock属于早期状态，不能当当前完整锁。
- **源码位置：** `scripts/runtime_env.ps1`；`configs/v3/dependencies.lock.json`；`RUNTIME/envs/*/pyvenv.cfg`与环境锁；`scripts/v3/serve_full_workbench.py`。
- **还需补充知识：** 进程边界、ABI/版本兼容、虚拟环境和物理文件复用、可重复安装的锁定策略。
- **现有证据是否足够：** 足够描述当前依赖与分工；.pth借用意味着底层环境更新仍会影响metrics，不能称完全独立隔离。

### Q25 LoRA和QLoRA有什么区别，你真正做了哪一种？

- **面试官问法：** “用了低位模型就叫QLoRA吗？”
- **通俗答案：** LoRA是在冻结模型上学习一小组修正参数；QLoRA还把训练底座量化来节省内存。项目的训练底座是BF16，因此是LoRA；部署模型量化并不让这次训练自动变成QLoRA。
- **技术答案要点：** 低秩更新 `W+(alpha/r)BA`，参数规模约r(m+n)而非mn。QLoRA需要量化训练底座与相应数值计算流程；本项目没有以GGUF Q4文件反向训练。
- **项目真实例子：** [实测] Qwen3.5-2B BF16底座，r8/alpha16/dropout.05，114个目标模块、228个A/B张量，共3691008个可训练参数。adapter约14.8MB，不代表训练只占14.8MB内存。
- **源码位置：** `configs/v2/lora.json`；`ragscope/training/run.py` 的 `discover_attention_targets`、`load_training_model`；LoRA `model_observed.json`。
- **还需补充知识：** 矩阵低秩近似、rank与适配容量、量化训练、optimizer内存。
- **现有证据是否足够：** 足够说明真实训练方法；没有QLoRA实训、量化adapter发布或从零训练底座的证据。

### Q26 LoRA数据和训练配置如何保证不是“随便跑了一下”？

- **面试官问法：** “训练多少条、多少步、学了哪些参数？”
- **通俗答案：** 我冻结数据和配置，先保证每条训练样本的答案与证据配对完整，再做有上限的训练，并保存每步结果和恢复状态。
- **技术答案要点：** QASPER train最多1000、dev loss最多60；超1024完整样本丢弃而不截答案。实际972训练/59验证，microbatch1×累积8、AdamW1e-4、warmup.03、seed42、梯度裁剪1、gradient checkpointing。一轮实际ceil(972/8)=122步，未跑满配置上限200。
- **项目真实例子：** [实测] train394篇、dev31篇，文档交集0；841可答参考证据、131真实首段负例；最佳dev loss为step122的.135663。0.8B仅是配置后备路径，不能算已训练第二个模型。
- **源码位置：** `ragscope/training/data.py`、`prepare.py`、`run.py`；`20261014_v2_lora/prepared_2b_1024_v2/`与 `train_2b_1024_seed42/`。
- **还需补充知识：** 样本筛除偏差、有效batch、优化器步与microbatch区别、验证集选择偏差。
- **现有证据是否足够：** 足够证明一次具体可复核训练完成；规模有限，不足以代表充分训练或最优超参数。

### Q27 Assistant-only mask为什么重要，怎样验证？

- **面试官问法：** “你是不是把问题和参考材料也当目标训练了？”
- **通俗答案：** 训练应该主要教模型怎样回答，而不是要求它把给定材料照抄一遍。因此输入材料仍参与理解，但不会作为需要预测得分的目标部分。
- **技术答案要点：** prompt label置−100、assistant和结束标记保留真实token ID；使用实际chat template检查prompt是完整序列的精确前缀，避免凭字符串长度猜assistant起点。
- **项目真实例子：** [实测] 对972/59样本实际input_ids/labels/attention_mask重新核验，长度与mask通过；V3还解码20个监督目标并做固定8夹具两步学习smoke。证明监督链路有效，不证明真实问答泛化提升。
- **源码位置：** `ragscope/training/data.py:12` 的 `encode_assistant_only`；`MASK_AND_SPLIT_AUDIT.json`；`V3EARLY/S5_SUPERVISION_DISTRIBUTION_DIAGNOSIS.json`。
- **还需补充知识：** teacher forcing、交叉熵ignore_index、模板边界与EOS监督。
- **现有证据是否足够：** 足够反驳“整体mask完全失效”；不排除其他训练分布或标注质量问题。

### Q28 完整checkpoint恢复为什么不只是保存adapter？

- **面试官问法：** “断电后重跑能真的从同一步继续吗？”
- **通俗答案：** adapter只存学到的参数；要接着训练，还要记住优化器、学习率、随机状态以及已经读过哪些样本，否则只能算从相似权重重新开始。
- **技术答案要点：** adapter、optimizer、scheduler、Python/NumPy/Torch/CUDA RNG、data_order/data_position、step/epoch一起保存；临时目录写全并fsync、manifest校验后原子提交；恢复逐项精确比较。
- **项目真实例子：** [实测] step5与10均留下RESTORED_VERIFIED，adapter/optimizer/scheduler/RNG/顺序/位置一致。当前保留最新step120/121/122与best122；恢复收据存在不等于所有历史checkpoint都永久留存。
- **源码位置：** `ragscope/training/checkpoint.py` 的 `CheckpointStore`；`ragscope/training/run.py` 的 `train_loop`、`_restore_rng`、`_state_equal`。
- **还需补充知识：** 原子写入、故障一致性、确定性训练、梯度累积中的安全保存边界。
- **现有证据是否足够：** 足够证明两次实际完整恢复；不能承诺任意库版本/硬件变更后仍位级复现。

### Q29 验证loss下降，为什么LoRA最终问答反而变差？

- **面试官问法：** “是不是过拟合？你怎么证明？”
- **通俗答案：** loss衡量给定正确前文时预测目标词的能力，真实问答还要求检索、选择证据、判断拒答并完整生成。loss变好不代表整个任务都变好，也不能仅凭这两个数字断言过拟合。
- **技术答案要点：** 区分teacher-forced dev loss与free-generation质量；分布、输出长度、拒答校准、引用验证和参考表达都可能影响。因果结论需要匹配分布和协议的消融。
- **项目真实例子：** [实测] 同BF16100题、92可答的token F1 .464889→.426683；新增23次可答拒答是负向贡献之一，但EM .184783→.217391。训练prompt中位350.5token、旧评测2147.5，存在描述性分布差异，尚未证明唯一成因。
- **源码位置：** `ragscope/training/evaluate.py`；`V3EARLY/LORA_OLD_RECOUNT.json`、`S5_SUPERVISION_DISTRIBUTION_DIAGNOSIS.json`。
- **还需补充知识：** 自回归暴露偏差、分布迁移、拒答校准、多个质量指标与过拟合诊断。
- **现有证据是否足够：** 足够确认这次F1负结果；不足以证明“LoRA普遍有害”或锁定单一原因。

### Q30 LoRA为什么变慢，合并后有没有验证成功？

- **面试官问法：** “是不是EOS错了导致一直生成？merge不就解决了吗？”
- **通俗答案：** 我分别检查了输出是否更长、每个token是否更慢和停止是否正常。结果支持长度增加与未合并修正层开销；合并探针没有过预定数值容差，所以没有宣称合并成功。
- **技术答案要点：** 自然停止24题与固定64-token微基准解决不同问题；排除warmup、同步计时、区分首token与后续decode。merge需验证数值差异，不只看能否调用函数。
- **项目真实例子：** [实测] 新24题两分支全部正常EOS；输出token中位67→171。3题固定64输出的后续速度12.5634→8.5900 token/s。merge的max_abs .50–.625超过.08、relativeL2 .033–.050超过.01，未保存合并快照，也未跑合并正式对照。
- **源码位置：** `scripts/v3/run_lora_runtime_diagnostics.py`；`V3EARLY/S5_FIXED64_*.jsonl`、`S5_MERGE_LOGIT_GATE.json`。
- **还需补充知识：** prefill/decode、低秩分支计算、BF16舍入、logits容差与行为等价。
- **现有证据是否足够：** 足够支持特定微基准机制与合并失败事实；3题不能成为在线P95或所有输入结论。

### Q31 路由分类器训练的“正确标签”从哪里来？

- **面试官问法：** “你怎么知道一题该用BM25还是Hybrid？”
- **通俗答案：** 不是凭人工感觉给标签，而是让同一个生成模型在三条检索路线都答一遍，再按答案代理质量和成本规则选。但这仍只是该实验条件下的标签，不是绝对真理。
- **技术答案要点：** 三路线同配置真实生成；可答用F1、不可答用正确拒答；质量差≤.01时优先低耗时，再看证据覆盖。标签依赖生成器、上下文、prompt和代理指标，不能叫客观最优路线。
- **项目真实例子：** [实测] train200标签128 BM25/42 Hybrid/30重排，dev60为31/23/6；固定2B Q4生成条件。测试数据不用于fit或选路由器。
- **源码位置：** `ragscope/routing/core.py` 的 `build_proxy_labels`、`proxy_quality`、`validate_training_splits`；`configs/v2/routing.json`。
- **还需补充知识：** 代价敏感学习、弱监督标签、选择偏差与代理目标错配。
- **现有证据是否足够：** 足够复查标签来源；不足以把分类accuracy解释为最终答案正确率。

### Q32 线性、MLP、LSTM都试了，为什么没采用？

- **面试官问法：** “换成更强分类器会不会解决？”
- **通俗答案：** 这批数据里三者在开发和测试上都几乎没有分流，实际全部选BM25。换复杂模型不是自动补救，先要确认标签、样本和任务目标是否合适。
- **技术答案要点：** linear是TF-IDF logistic，MLP为SVD64+hidden32，LSTM为词向量64+hidden64；训练上限20轮、patience3。多数类train64%、dev51.667%，MLP/LSTM恰好等于多数类基线，不能只拿train/dev差就下过拟合结论。
- **项目真实例子：** [实测] 三者dev60/test100全部选BM25，F1等于固定BM25 .380598/.459276。规则在线P50仅较固定Hybrid快5.04%，未达10%门槛，默认仍固定。V3也没有训练“事实/比较/总结”的新语义分类器，旧负结果不能回答那个新目标是否有效。
- **源码位置：** `ragscope/routing/learned.py` 的 `TfidfRouter`；`ragscope/routing/lstm.py` 的 `LSTMRouter`；`ROUTE/train_diagnostics/ACCEPTANCE.json`、`online/SUMMARY.json`。
- **还需补充知识：** 多数类坍缩、类不平衡、分类校准、工作流效用上界与分类任务定义。
- **现有证据是否足够：** 足够解释当前停用；不支持“LSTM或SLM永远不能有效分类问题”的泛化判断。

### Q33 Laya做了什么，候选顺序为什么重要？

- **面试官问法：** “Laya是不是你训练的路由大模型？效果怎样？”
- **通俗答案：** 用的是固定现成英语模型，让它在三条路线说明中选择；没有训练一个新的项目专属Laya。模型会受候选展示顺序影响，低置信时又常回退Hybrid，所以没有证明值得默认增加它。
- **技术答案要点：** 固定权重与SDK revision、CPU eager、max_len512、阈值.5、三种顺序；保存原始概率与回退后路线。区分原始choice改变、阈值后route改变、校准误差和最终QA效用。
- **项目真实例子：** [实测] 360题×3=1080决策、错误0、回退646；默认顺序dev回退95%、test96%；142/360题最终路线受顺序影响。不能事后挑测试F1更高的顺序改正式选型。Jev仅为付费API被排除的配置项，无真实本地实验。
- **源码位置：** `ragscope/routing/laya_adapter.py` 的 `LayaRouter`；`laya_campaign.py`、`laya_validation.py`；`LAYA/RAW_ACCEPTANCE.json`、`ROUTE/LAYA_PERMUTATION_CALIBRATION.json`。
- **还需补充知识：** choice模型位置偏差、概率校准、ECE/Brier与任务代理标签的限制。
- **现有证据是否足够：** 足够证明真实跑过且未显示稳定收益；不能把1080决策算成1080次新QA生成。

### Q34 怎样防止数据泄漏？索引里有测试论文算泄漏吗？

- **面试官问法：** “训练和测试是否读过同一篇论文？测试答案会不会进入索引？”
- **通俗答案：** 检索系统需要能搜索测试论文原文，这本身是开卷任务的正常条件；危险的是把测试问题、答案或看过结果后的调参信息混入训练和选择过程。
- **技术答案要点：** 按论文划分train/dev/test；训练器核验doc/query隔离；索引正文与QA参考答案分开；配置在holdout gold读取前冻结；区分语料可见、标签可见与开发反馈泄漏。
- **项目真实例子：** [实测] LoRA394 train篇与31 dev篇交集0；路由验证拒绝train/dev论文交集；V3FINAL保存配置冻结和split manifest。总结/比较参考由AI辅助构造并校验原文，人工尚未审核，题目构造偏差仍存在。
- **源码位置：** `ragscope/training/prepare.py`、`ragscope/training/evaluate.py` 的 `validate_comparison_inputs`；`ragscope/routing/core.py`；`V3FINAL/FINAL_CONFIGURATION.json`。
- **还需补充知识：** 检索评测的可见性协议、文档级划分、同源近重复与测试集反复使用风险。
- **现有证据是否足够：** 足够说明已实现的隔离与冻结；不能证明预训练语料零污染或所有潜在近重复都不存在。

### Q35 文字F1、语义相似度和事实正确率分别告诉我们什么？

- **面试官问法：** “不逐字一样也可能正确，为什么还用F1？加语义评分就解决了吗？”
- **通俗答案：** F1看答案词语与参考重合，能发现一些明显偏离，但会误伤正确改写。语义相似度更容忍换说法，却可能把数字错、否定错的答案也判得很像，因此还要核对事实与证据。
- **技术答案要点：** EM严格匹配，token F1衡量precision/recall重合，ROUGE-L看最长公共子序列，答案embedding cosine衡量表征接近。NLI/claim-support又是另一层自动代理，存在误判；引用可见性是格式属性。
- **项目真实例子：** [实测] 最终总结/比较语义cosine .8125→.8594，不能称“事实正确率85.94%”，也不是BERTScore。最新规则还保留数字、极性等flags与UNKNOWN，flag不自动等于事实错误，人审未完成。
- **源码位置：** `ragscope/evaluation/generation_metrics.py`、`ragscope/v3/practical_quality.py`；`reports/v3/METRIC_SPEC.md`；`V3FINAL/FINAL_SCORES_RESIDENT8K.jsonl`。
- **还需补充知识：** 多参考答案、语义评价鲁棒性、NLI域迁移、事实支持的人审协议与一致性。
- **现有证据是否足够：** 足够说明指标及结果的含义；不足以输出可信的整体自动事实正确率。

### Q36 怎样证明实验结论可复核，又不把测试数量当成果？

- **面试官问法：** “这些分数能重算吗？P95、置信区间和失败怎么处理？”
- **通俗答案：** 不能只留一张好看的汇总表。我保存每道题的输入、答案、失败、配置和模型身份，让分数能从原记录重算；实验规模说明测得多，不等于答得好。
- **技术答案要点：** 固定同题配对、保留失败分母和N/A；正式请求、warmup、精确响应复用与重放分开计数；P50/P95注明冷热与计时范围。bootstrap是对既有样本重采样估计不确定性，存在同论文聚类时要考虑聚类抽样，不能制造新独立样本。
- **项目真实例子：** [实测] 最终评分为基线220+最终220，部分最终响应是payload精确一致的复用，不冒充新调用；在线30+30用无答案/查询缓存路线。普通QA170可答题25提高/17下降/128持平；82定向回归通过只证明所覆盖行为，不是82道真实质量题，更不是所有功能零缺陷。
- **源码位置：** `ragscope/experiments/ledger.py`、`scripts/v3/finalize_delivery.py`、`scripts/v3/finish_evaluation.py`、`scripts/v3/finish_online.py`、`ragscope/v3/online_runtime.py`；最终原始评分、Junit与验收文件。
- **还需补充知识：** 配对检验、bootstrap/聚类bootstrap、置信区间与工程门槛、统计显著性和实际效益、完整可重复环境。
- **现有证据是否足够：** 足够复算已保存的描述性结果和检查配置；没有本轮所有增益均统计显著、长期P95稳定或独立人工质量审核完成的证据。简历可写真实工程覆盖与具体同题增益，不应写“全链路准确率95%”或“LoRA显著提升”。
