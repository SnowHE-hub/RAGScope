# S2 CPU 检索实验报告

执行日期：2026-10-14（Australia/Sydney）。子campaign：`20261014_v2_retrieval`。
BM25全11组及有界上下文扩展对照已完成；真实CPU Dense/Hybrid全11组也已正常结束，2,200条、0失败。最终dev-only候选冻结为简单F512_O0和结构R512；这与BM25初筛简单候选F512_O64不同，主对照须使用正式冻结候选。上下文扩展选择及正式全库神经索引正在继续，不能将初筛test上下文称为全部正式矩阵完成。

此前为真实LoRA训练暂停自身worker/停止自身CPU服务，训练正常退出并明确handover后恢复。暂停/恢复凭据为`PLANNED_TRAINING_RAM_PAUSE.json`、`TRAINING_RAM_RESUME.json`；两次在途HTTP重置保留为有界恢复2/3，无已完成查询丢失。当前矩阵worker已正常exit0，receipt更新COMPLETE。

## 冻结输入与公平性

使用S1.5冻结的36篇dev论文、100个问题，92个可回答、8个不可回答。test未用于选型。
原始数据及切分身份见`D:/RAGScope_Runtime/corpora/20261014_v2/SPLIT_MANIFEST.json`。

切块使用实际Qwen3-Embedding-0.6B tokenizer，revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`，tokenizer文件SHA-256 `def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a`。
所有块都再次按真实tokenizer计数，并验证canonical源文本全覆盖、源片段与输出相等、父子包含关系。没有静默截断。
`cl100k_base`仅用于统一context 2048-token预算；标题和引用标签计入预算。

BM25使用`rank_bm25.BM25Okapi`，k1=1.5、b=0.75、epsilon=0.25；词法分词为alnum-cjk-v1。
IDF在相同冻结dev全文库上建立，查询限制为用户指定论文。先取最多50候选、再前10个按序打包。父子组先按子块检索再扩父、去重和预算打包。
查询顺序seed42打乱，预热3次。时延为真实检索加context构建端到端，不含数据下载和索引构建。
全部CPU-only；本执行器未启动或请求GPU模型，也未修改原GPU任务。

## BM25实测矩阵

每行100个问题、0失败。证据指标分母都是92，8个不可回答的证据召回为N/A。
“整组”表示至少一个合法参考证据集合全部覆盖；“连续”是参考单元字符范围的真实覆盖。
所有值来自`D:/RAGScope_Runtime/runs/20261014_v2_retrieval/E1_bm25_SUMMARY.json`及对应`records`逐行JSONL。

| group | 子块数 | 父块数 | context整组覆盖 | context连续覆盖 | 完整单元召回 | P50 ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| F256_O0 | 696 | 0 | 0.521739 | 0.721533 | 0.583514 | 13.278 | 16.402 |
| F256_O32 | 786 | 0 | 0.586957 | 0.736286 | 0.646014 | 14.616 | 21.837 |
| F512_O0 | 355 | 0 | 0.576087 | 0.692583 | 0.622373 | 12.509 | 15.902 |
| F512_O64 | 398 | 0 | 0.652174 | 0.739776 | 0.688587 | 13.611 | 17.245 |
| F1024_O0 | 184 | 0 | 0.467391 | 0.529755 | 0.490942 | 8.340 | 13.225 |
| F1024_O128 | 204 | 0 | 0.608696 | 0.658897 | 0.630435 | 10.271 | 16.928 |
| S512 | 389 | 0 | 0.619565 | 0.703486 | 0.641304 | 13.383 | 16.803 |
| R512 | 406 | 0 | 0.750000 | 0.775815 | 0.775815 | 13.499 | 17.180 |
| R1024 | 198 | 0 | 0.673913 | 0.691304 | 0.691304 | 9.383 | 15.749 |
| P256_1024 | 853 | 198 | 0.652174 | 0.668478 | 0.668478 | 12.667 | 18.142 |
| P256_1536 | 823 | 135 | 0.554348 | 0.580072 | 0.580072 | 14.518 | 19.030 |

累计5625个不同chunk记录（含333父块），低于180,000工程上限。
这批论文内检索的文档命中没有区分力，因此不把doc hit当作找齐证据。
初筛候选按预先规则选为简单组F512_O64和结构组R512，证据在`BM25_SCREEN_SELECTION.json`。
R512当前BM25上下文整组覆盖较高；这只是dev初筛，不证明生成答案更准确，也不证明同样结果适用于Dense或其他数据。
更大父块和更大固定块没有单调改善；预算内是否能放下完整证据比单个块更长更关键。

## 已执行的上下文扩展对照

在F512_O64、R512和已有最佳父子组P256_1024上比较none、neighbor以及适用时的parent。
父子组之外没有父映射，未编造“parent结果”。已有主矩阵行直接复用，共额外400条真实BM25问题记录。
三个P256_1024策略使用完全相同的子块检索排序，只改变context扩展。
逐条产物为`records/E1_<group>_bm25_<strategy>.jsonl`，每个文件均含不可变`.config.json`及记录哈希。
目前BM25总计1500条，无失败；这不含任何生成调用。

| 固定子块配置 | 扩展 | dev context整组覆盖 | dev context连续覆盖 |
|---|---|---:|---:|
| F512_O64 | none | 0.652174 | 0.739776 |
| F512_O64 | neighbor | 0.576087 | 0.691500 |
| R512 | none | 0.750000 | 0.775815 |
| R512 | neighbor | 0.652174 | 0.670290 |
| P256_1024 | none | 0.586957 | 0.708426 |
| P256_1024 | neighbor | 0.630435 | 0.687568 |
| P256_1024 | parent | 0.652174 | 0.668478 |

在这批dev上，邻块扩展没有改善已选F512_O64/R512的整组覆盖；P256_1024父扩展增加整组覆盖，但连续覆盖下降。这些不同指标不能拼成单一“全面提升”。

## 已完成的test上下文与全库E8

全库索引构建前实际启动检查可用RAM为1,248,690,176字节，高于768MiB门槛；原GPU任务未被修改。
固定同一509篇论文语料，为dev初筛的两个候选分别构建完整BM25索引：F512_O64 6405块，R512 6551块。
两者共12,956块，加dev矩阵保守累计18,581块，低于180,000上限；相同源片段跨索引可能重复计入该保守工程统计。
全库索引IDF在同一509篇上计算，这与dev初筛的36篇IDF范围不同，实验身份已经单列。

| run_id | 任务范围 | N / 证据适用N | context整组覆盖 | context连续覆盖 | 文档Recall@前10检索块 |
|---|---|---:|---:|---:|---:|
| TEST_CONTEXT_F512_O64_bm25_none | 指定论文 | 200 / 182 | 0.631868 | 0.716879 | 1.00 |
| TEST_CONTEXT_R512_bm25_none | 指定论文 | 200 / 182 | 0.703297 | 0.740404 | 1.00 |
| E8_R512_bm25_none | 固定509篇全库 | 100 / 100 | 0.170000 | 0.178333 | 0.37 |

这500条全部实际执行、0失败；目前总计2000条BM25检索结果，不含生成调用。
论文内test的18条不可回答证据指标为N/A；所有失败仍保留在预先冻结的任务分母中。
同一篇重复检索块不重复计算文档命中。E8的0.37即100题中37题在前10个检索块见到其金标准论文。
E8全库只给问题文字和global scope，gold_doc_id仅在结果返回后用于评估；不将论文答案或证据位置传给检索器。
全库结果明显弱于用户指定论文任务，不能将后者接近1的文档命中冒充全库检索能力。
test从未反向改变两个候选选择；当前R512默认仍源自dev初筛。
实际ContextBundle和raw候选可供后续生成复用，但未执行生成，不称“答案质量实验完成”。

`TEST_CONTEXT_E8_SUMMARY.json`明确E8四条计划路径中BM25=`PASS`，Dense/Hybrid/Hybrid+Rerank=`BLOCKED_RESOURCE`，总体`PARTIAL_BLOCKED`。

## Dense/Hybrid历史资源阻断与恢复

`D:/RAGScope_Runtime/runs/20261014_v2/embedding_preflight.jsonl`保存3次启动前预检。
两次Windows可用RAM约1.155GB/1.164GB，低于所需1GiB加512MiB保留量；模型没有加载。
现有WSL替代路径开始前宿主仅约295MB可用；其记录中minimum_host_available字段为512MiB、reason字符串为1GiB，实际余量低于两种阈值，阻断结论一致但阈值记录不一致必须披露。
上述三次是历史启动前失败，不应在环境改变后继续当作永久阻断。后续用户追加授权的原任务安全接管由主执行器单独审计；内存释放后，主执行器已核验独立CPU-only Qwen服务的真实1024维输出。
11组Dense/Hybrid矩阵现已实际开始，固定Qwen模型、量化、pooling、归一化和tokenizer身份；完成前不声称已有完整质量结果。50→10神经重排仍由后续独立运行完成。
缓存实测前184个向量在约333秒内落盘，约0.55片段/秒（以256-token块为主）；这是早期吞吐观察，不作为512/1024块或全程耗时保证。每个完成文本的数组及身份立即落盘，可供恢复。
完整dev编码工作量经原文哈希去重为4,737段、1,840,835个Qwen正文tokens；5,292个子块中有555段可复用，不编码333个父块。
本次启动时的适配器`input_tokens`来自`add_special_tokens=False`，因此仅为正文/查询字符串的token数；后端特殊token总数未保存，不补造历史usage。正文上限1024，适配器的1100检查同样是文本token阈值，不冒充实测后端总数；后续新进程采用主执行器补充的`backend_usage`字段。
首组Dense存在冷查询编码，后续方法可命中同一查询向量缓存。逐题保存cache_hit，不能把缓存带来的时延差异解释为检索算法加速；后续汇总应同时给出检索组件耗时与编码冷热状态。

首个完成的真实神经组为F256_O0，各100题、0失败、92个证据适用样本。Dense整组覆盖0.706522、连续覆盖0.810365；Hybrid整组覆盖0.608696、连续覆盖0.776352；同组BM25整组覆盖0.521739。Hybrid在此组低于Dense，保留该结果，不推断融合必然改善。
对应文件为`summaries/E1_F256_O0_dense_none.json`及`E1_F256_O0_hybrid_none.json`；其余组尚在执行，不能用此局部结果提前冻结最终候选。

首组延迟审计进一步拆分如下，P50为各组件独立中位数，不能相加当作总耗时中位数：

| 方法 | 查询向量cache hit / miss | 查询编码P50 ms | 检索组件P50 ms | context构建P50 ms | 实际总P50 ms |
|---|---:|---:|---:|---:|---:|
| Dense | 4 / 96 | 247.676 | 0.303 | 10.587 | 259.949 |
| Hybrid | 100 / 0 | 15.447 | 2.524 | 10.267 | 28.513 |

因此Hybrid较低的此处总延迟主要来自查询缓存，不能宣称Hybrid检索比Dense更快。早期BM25在原计算任务仍运行时测得，神经矩阵在后续资源释放后执行，背景负载也不同；这些是本机实测工程耗时，不是完全隔离负载下的硬件性能结论。
所有当前质量对比均为dev点估计，问题在36篇论文内相关，选择容差0.01不是显著性阈值。最终置信区间应按论文聚类、同题配对；不能把100个问题当100个独立论文，也不能用已选择的dev结果宣称test泛化提升。

## 有界CPU线程优化与中断记录

原CPU任务已按用户后续授权由主执行器处理后，在宿主可用RAM约3.60GB、启动额外服务前至少2GiB的约束下，执行4/8/12线程小对照。
固定同一CPU二进制、Q8_0权重、tokenizer、pooling、归一化和输入；仅改变`-t`/`-tb`。3个实际256/512/1024正文token案例各重复2次，另检查同一查询向量和三个候选的排序。

| CPU线程 / batch线程 | 正文tokens/s | 相对4线程 | 抽检最大逐元素差 | 排序一致 |
|---|---:|---:|---:|---|
| 4 / 4 | 142.099 | 1.000× | 基线 | 基线 |
| 8 / 8 | 258.052 | 1.816× | 0 | 是 |
| 12 / 12 | 315.255 | 2.219× | 0 | 是 |

12线程超过预声明1.15×材料性阈值，已用于后续编码。结果是3个长度案例的有界实测，不保证所有论文、系统负载和批量大小都取得相同倍数；也不是GPU加速。
两个额外对照服务均在完成后退出，11437端口释放。正式CPU服务仍只监听11436，替换前后均为`-ngl 0 --no-op-offload`，没有修改用户原进程。
完整输入与向量保存在`D:/RAGScope_Runtime/runs/20261014_v2/cpu_embedding_benchmark_20261014_023547`，包括`RESULT.json`、`baseline_vectors.npz`和8/12线程向量、日志。
后端usage直接观测到256/512/1024正文tokens对应257/513/1025总tokens，证明这些实际长输入案例没有被截为512；不将其反填成所有历史调用的实测值。

`THREAD_TRANSITION.json`记录旧/新PID及create_time、命令、性能配置哈希、模型语义身份、缓存兼容依据。旧向量被保留；其与新线程向量的兼容基于上述逐元素相同的抽检，未暗中改变模型语义身份。
`CPU4_CACHE_KEYS_AT_TRANSITION.json`保存已有4线程缓存键，并明确边界最多有1个尚未写回的旧线程向量。后续时延须按此线程阶段解释。
替换时暂停的索引worker有一条HTTP请求收到WinError10054连接重置，旧进程退出；`embedding_runtime_failures.jsonl`保存这次真实失败，恢复次数1/3。
随后执行完全相同的矩阵命令，从已经完成的两个组及逐文本缓存恢复，未删除完成行、未减少任何问题，也未把这次索引请求中断记成“零运行错误”。
F512_O0中断前部分向量的原冷编码耗时未逐条落盘，因此恢复时的cache-hit耗时不能冒充这部分完整冷索引构建成本；`chunk_manifest.build_seconds`及矩阵同名字段仅指切块构建，不是完整embedding索引端到端耗时。
后续全库索引构建已增加逐chunk首个已完成编码观测的不可变记录，批内后续请求失败也在finally中fsync此前完成的测量。该改动不回补旧数据，不影响当前暂停进程已加载的旧代码。模拟真实中断位置的离线测试证明恢复后保留原冷耗时；该数值夹具明确标为非真实模型。
暂停期间另执行`scripts/audit_embedding_indexes_v2.py`，4个完成索引共2235个向量全部通过1024维float32、有限值、L2归一化、身份和文件哈希核验，每块正文重新用已固定真实tokenizer计数均符合预算；每索引seed42抽20个，共80个向量与文本缓存逐元素一致。已索引去重文本中523个存在真实backend usage，1639个历史无usage；可观测样本的后端tokens−正文tokens为1。详细证据为`EMBEDDING_INDEX_AUDIT.json`。向量/缓存对齐是有界抽查，未声称逐个复核全部对应关系；未完成索引不计入完成数。

## 可恢复执行与边界

```powershell
. ./scripts/runtime_env.ps1
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -u scripts/run_retrieval_v2.py --home 'D:/RAGScope_Runtime' --campaign 20261014_v2 --stage bm25
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -u scripts/run_retrieval_v2.py --home 'D:/RAGScope_Runtime' --campaign 20261014_v2 --stage bm25 --groups F512_O64 R512 P256_1024 --expansion-comparison
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -u scripts/run_retrieval_v2.py --home 'D:/RAGScope_Runtime' --campaign 20261014_v2 --test-contexts
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -u scripts/run_retrieval_v2.py --home 'D:/RAGScope_Runtime' --campaign 20261014_v2 --stage dense --embedding-url 'http://127.0.0.1:11436' --embedding-identity 'D:/RAGScope_Runtime/runs/20261014_v2/embedding_identity.json'
```

索引按数据、split、scope、完整切块参数、真实tokenizer及revision签名隔离；内容哈希与条数匹配才复用。
每题保存raw候选50的ID/得分/源坐标，完整原文可从对应索引恢复；保存实际ContextBundle、预算丢弃原因、去重记录、证据分数、错误及实际计时。
RecordStore每条fsync，Campaign持单写锁；中断从完整记录恢复，已完成问题不重复统计。
15个检索离线单元测试通过，覆盖切块源覆盖、父子边界、真实BM25 scope、失败分母、不可回答N/A、dev容差选择、幂等恢复及编码批次中断的原始耗时保留；全库测试额外证明gold_doc_id不进入scope、干扰文档仍被实际搜索。新增数组索引恢复和身份拒绝使用明确标记的离线数值夹具，不计入模型结果。
独立反向核验见`E1_BM25_INDEPENDENT_AUDIT.json`：重读1100条实际context源坐标，使用独立区间合并代码重算33个表格数字全部一致；其中seed42抽10数字保存，未用汇总表核验汇总表。
新增可重复命令`python scripts/audit_retrieval_v2.py --home D:/RAGScope_Runtime --campaign 20261014_v2`最近实际通过：44个已完成运行、4600行、18430个context原文片段、132个汇总数字全部一致，另以seed42抽10项记录。
该脚本独立实现字符区间求并、逐单元覆盖和多合法标注集合的预声明选择，不调用生产评分/汇总代码；同时核验行哈希、配置哈希、完整问题分母、原文坐标及正文片段。证据保存于`RETRIEVAL_INDEPENDENT_AUDIT.json`，每次只核验已完成summary，不将进行中的运行算作完成。
`RETRIEVAL_LATENCY_AUDIT.json`另存编码缓存命中与各实际组件的N/P50/P95；旧行缺失的组件保留不可得，不反填估计值。
最终神经test上下文执行前已冻结`configs/v2/retrieval_statistics.json`：2000次seed42配对论文cluster bootstrap，95% percentile区间；SQuAD使用article_cluster_id。5组对照包括论文内Hybrid−BM25、Hybrid−Dense、结构−简单Hybrid，以及全库Hybrid−BM25、Hybrid−Dense；全部明确为探索性，不将未经多重比较调整的区间解释为确认性显著结论。
执行器`scripts/summarize_retrieval_statistics_v2.py`验证真实行/配置哈希、完整冻结问题集合、失败为零和不可回答N/A，再计算按问题加权、以整篇文档重采样的估计。未完成配置保持UNAVAILABLE；最终选择文件未就绪时保持WAITING_FOR_DEV_SELECTION，不使用初筛选择冒充最终神经方案。统计输入核验的3个离线测试通过。
Dense/Hybrid要求显式CPU后端身份与loopback URL；没有就不将词法检索或Mock包装成Dense实验。
重排、E8其余三种检索方式及最终生成主实验尚未完成，当前报告不声称它们已经完成。

## S4在线检索调用接口

`ragscope/routing/online_retrieval.py`提供`create_live_retriever(home, campaign, embedding_url=..., reranker=...)`；返回对象的`retrieve(question, scope, route, model_budget_checks=checks)`实际搜索已完成索引，支持BM25、Hybrid和HybridRerank。只接受问题文字和明确scope，不接收gold字段；全库与单篇scope不会互换。
工厂要求最终dev候选、匹配候选哈希的context选择、真实完成索引和源文件哈希全部就绪。它不自动加载模型、下载文件或用冻结ContextBundle替代在线检索。多个路线可共享已验证Dense ArrayIndex中相同的BM25语料和统计，以减少内存；BM25调用时仍披露实际所用Dense索引签名，不声称载入了另一个索引。
在线结果保存raw50、pack前真正selected10、重排窗口、query cache状态、编码/检索/融合/重排/context实际耗时；索引加载单列在`load_timings_ms`，不混入逐题热运行时延。少于50个候选时保存真实数量。backend错误保留，不悄悄回退BM25。
调用者提供逐题、全部formal模型的真实prompt+512≤4096共同预算检查。在线路径严格复现freezer.shared_context的两步逻辑：先冻结扩展+2048 reference-only打包，再只对已保留pieces做共同模型预算repack；不会因模型预算拒绝前段而重新加入初次reference预算已排除的候选。初始`retrieval_context_bundle`与最终`context_bundle`及两个实际pack耗时均保留。8项离线接口测试通过，包括实际BM25调用、50→10合约、scope禁止gold、backend失败保留、未就绪不加载、两步共同预算与正式packer一致，以及拒绝模型实际device为GPU但身份声明CPU的错误注入。这些数值夹具不是神经在线实验结果。
实际环境ready检查记录`ONLINE_RETRIEVAL_READINESS.json`为WAITING_FOR_FINAL_DENSE_INDEX：候选和扩展已冻结，但完整Dense manifest尚未就绪，探测没有模型调用或伪造在线结果。后续物化结果也已加入`selected_retrieval`及schema version2；历史不可变行保持原状，缺该字段的错误分析只能标记packing_or_ranking_loss_uncertain，不能据raw50将丢失证据全部归因于打包预算。

路由上下文的独立物化入口为`scripts/materialize_routing_contexts_v2.py`，只在最终完整选中索引就绪后运行train200/dev60的BM25与Hybrid，共520条实际检索。测试集100题只核验身份，在已有TEST_CONTEXT200中取得，不在此新增测试选型。
root冻结的`ROUTING_QUERY_MANIFEST.json`哈希f1b6ae1ef21a44b720b53e1a6fffe6600a12d16d51e4b88aefe26a69c0ff952b已实际通过哈希、SPLIT源、问题文件、freezer.seeded_subset完整顺序和跨split文档不重叠检查。train200来自159篇、dev60来自30篇、test100来自59篇；证据`ROUTING_SOURCE_VALIDATION.json`明确实际物化尚未发生。新增2个离线测试拒绝重签名后的不同抽样顺序、错误外部预期哈希和变动源文件。

## 辅助排序指标与来源映射

`scripts/summarize_retrieval_auxiliary_v2.py`已实际处理44个完成运行、4600条原始结果，输出`RETRIEVAL_AUXILIARY_METRICS.json`及`auxiliary/*.jsonl`。原记录与配置哈希、冻结分母、完整索引manifest均核验，不改写正式实验行。定义固定于`configs/v2/retrieval_auxiliary.json`：MRR@10为首个相关chunk的倒数rank；binary nDCG@10的理想排序分母来自scope内完整索引中的相关chunk数量，不只用已检索50候选。相关性指任何正长度的合法证据单元交集，多个合法标注取union。失败为0、不可回答N/A。R3使用真实selected_retrieval重排顺序，不能用上游Hybrid50冒充。

这是离线辅助描述，不参与选择：少量证据交集未必足够回答，overlap和chunk粒度改变相关item数，因此不同切块间MRR/nDCG可比性有限；完整证据单元与完整参考集合覆盖仍为主指标。两项手算测试验证全索引IDCG、实际rank、失败/N/A与重复ID拒绝。

`scripts/export_retrieval_source_map_v2.py`已导出并核验`SOURCE_MAP.json`：在排序后的dev文档ID上以seed42固定抽3篇，覆盖11组、462个子块、31个父块、153个稳定证据单元，所有坐标片段逐个与canonical原文相同。导出包含canonical文字、Unicode半开区间、chunk参数、父子ID和明确标识为evaluation_only的标注，用于来源映射可视化，不进入在线检索输入。样本按文档ID抽取，未按检索成绩挑选。产物522587bytes，SHA256 `6c9faa968e9a930e614bf00f8ca2e0845022ca7b39c7b1db04ad4b596622c34c`；可重复执行要求相同内容。

训练完整122步并明确资源handover后，CPU服务按精确旧命令恢复为PID36892/create1791909754.8890424，二进制/权重哈希、CPU0层offload、真实1024维向量均通过。暂停worker43932的在途HTTP连接重置已保存，确认退出后以原命令重启为28832，这是有界恢复2/3；1000条前五组Dense/Hybrid结果完成后继续第六组，无已完成query丢失。恢复过程见`TRAINING_RAM_RESUME.json`，当前身份以`worker.json`和主`embedding_service.json`为准。

SQuAD E4将复用已冻结的Qwen查询指令（含scientific paper措辞），不单独为SQuAD调指令或检索参数。它用于同一短文本任务内的模型回归对照；数据领域、标注形式和长度均与QASPER不同，跨数据集差异不能归因为长度，也不能合并排名。

历史元数据限制：独立`CHUNK_PREFLIGHT_SCOPE_AUDIT.json`核验13个chunk manifests，发现早期两个all509 BM25索引的preflight.split误写为dev。它们的不可变identity明确为frozen_all_splits_corpus，document_checks保留真实文档ID，数据/hash/检索结果均正确；审计另列实际预检前3篇及其source split。未改写旧manifest。未来manifest使用真实identity范围、显式sample_document_ids/source_split_counts。该耗时预检不参与质量选择，也不把全库预检冒称dev样本。

## CPU 请求批量的有界实测

现有 llama.cpp b11429 CPU12线程服务支持string-array输入（[固定版本官方API文档](https://github.com/ggml-org/llama.cpp/blob/b11429/tools/server/README.md)）。在正常完成dev11组后，用固定seed42选择的4块512 tokens和4块1024 tokens，对batch1/2/4各测两轮，顺序1→2→4→4→2→1；另一次固定dev query，共29次请求。无共享cache读写、无重启、无额外模型，RAM每次调用前至少1.25GiB。

两轮合计serial37.7094s、batch2 37.7043s、batch4 40.6022s；相对吞吐分别1.00013倍和0.92875倍，均未达到预声明1.15倍门槛，保持request batch1。48个逐文本比较max_abs_diff=0、固定query排名完全一致；结论仅覆盖这八个文本，不能推广为所有输入的数值保证。实际backend usage保留在逐请求层，未将数组aggregate usage虚分到每个文本。

凭据：`D:/RAGScope_Runtime/runs/20261014_v2_retrieval/cpu_batch_benchmark/RESULT.json`，配置hash`0ac6e5562df3d8cf161008a1e09d63b0fa33ddd413554e44da93485c16aad91d`。不将这个有界性能实验当检索质量比较，也不因其结果改动候选或上下文规则。

## 完整真实 Dense/Hybrid dev 矩阵

下表每个方法100题，92题证据覆盖指标适用、8题不可回答为N/A；各组0失败。数值为实际打包后的整组引用证据覆盖/连续覆盖，父子组此表使用预声明parent，其他组使用none。正式上下文扩展选择在下表候选冻结之后进行。

| Group | Dense full / continuous | Hybrid full / continuous | Chunks / parents |
|---|---:|---:|---:|
| F256_O0 | 0.706522 / 0.810365 | 0.608696 / 0.776352 | 696 / 0 |
| F256_O32 | 0.630435 / 0.790381 | 0.619565 / 0.773523 | 786 / 0 |
| F512_O0 | 0.652174 / 0.749798 | 0.663043 / 0.777828 | 355 / 0 |
| F512_O64 | 0.673913 / 0.776798 | 0.619565 / 0.725493 | 398 / 0 |
| F1024_O0 | 0.608696 / 0.666910 | 0.630435 / 0.683214 | 184 / 0 |
| F1024_O128 | 0.554348 / 0.642577 | 0.630435 / 0.679306 | 204 / 0 |
| S512 | 0.663043 / 0.724876 | 0.641304 / 0.699310 | 389 / 0 |
| R512 | 0.739130 / 0.800272 | 0.739130 / 0.779438 | 406 / 0 |
| R1024 | 0.673913 / 0.707609 | 0.706522 / 0.727536 | 198 / 0 |
| P256_1024 | 0.804348 / 0.824275 | 0.728261 / 0.749275 | 853 / 198 |
| P256_1536 | 0.630435 / 0.662319 | 0.586957 / 0.607971 | 823 / 135 |

预声明规则使用Hybrid结果，先整组证据覆盖、再连续覆盖，0.01容差内取小索引/低P50；因此simple=F512_O0、structured=R512。不能从Dense更高的单项数字事后替换该选择规则。

| Candidate | Expansion | Full reference | Continuous |
|---|---|---:|---:|
| F512_O0 | none | 0.663043 | 0.777828 |
| F512_O0 | neighbor | 0.684783 | 0.707951 |
| R512 | none | 0.739130 | 0.779438 |
| R512 | neighbor | 0.750000 | 0.755435 |

两组最终均选择neighbor；所有正式test/global/SQuAD/ROUTE上下文沿用此冻结结果。早期BM25筛选阶段的none test结果保留为探索性历史，不能替代正式neighbor结果。完整证据为`E1_dense_SUMMARY.json`、`CANDIDATE_SELECTION.json`和`CONTEXT_SELECTION.json`。

当前11个已完成神经索引共5292个向量全部通过独立维度、float32、finite/L2及真实正文token检查；每索引seed42抽20个向量与内容寻址缓存逐项对齐，共220次。3098个去重文本具备实际backend usage；1639个历史文本无此观察，继续保留N/A。证据`EMBEDDING_INDEX_AUDIT.json`。
