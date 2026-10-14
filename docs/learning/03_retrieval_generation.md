# 检索、上下文、生成分别负责什么

检索器决定哪些片段可能有关，上下文打包器决定模型真正看到哪些片段，生成器根据这些文字回答。把三个步骤混成一个“准确率”，会掩盖失败发生的位置。

BM25 主要匹配词语；Dense 用向量表示文本；Hybrid 把两类排序融合；Rerank 再对候选逐对评分。主重排对照是召回 50 个、保留 10 个。若只把同样的 10 个重新排一次，可能改变前几名，却不能增加候选集合中原本没有的证据。实现见 `ragscope/retrieval/` 和 `ragscope/indexing/array_index.py`。

这里的50是候选上限，不是造出来的固定条数。R512 dev的100题都限定单篇论文，每篇实际只有5～28块，平均11.57块；已完成的正式重排处理了该论文所有可用候选，并记录不足50的原因。全库检索有更大的候选池；不足时仍不复制片段凑数。实际输入审计位于 `D:/RAGScope_Runtime/runs/20261014_v2_retrieval/R3_DEV_CANDIDATE_VALIDATION.json`。

到2026-10-14 06:06 UTC，R3五组的有效来源已完整签收，共660条：`E1_R512_hybrid_rerank_neighbor`的dev100、`TEST_CONTEXT_R512_hybrid_rerank_neighbor`的test200在CPU完成；global100与ROUTE train200/dev60在用户追加GPU授权后完成，文件名带`_gpu`。GPU360收据为`R3_GPU_COMPLETION_VALIDATION.json`，SHA-256 `b10cb5b52bb54606eada1fccfbedcb0a302c22a4bfdd6acd0c5f77563fbdef89`。原CPU global因RAM保护只完成的29条仍保留，不与新GPU全100条合并为129题。GPU路由训练来源的重排耗时不能冒充CPU E2或后续在线CPU重排的速度。

记录里的`raw_retrieval`是进入重排的最多50条，`selected_retrieval`才是真正保留的最多10条；后续ContextBundle还可能因预算少发一些。只有保留这三个层次，才能分清候选缺证据、排名丢证据和打包丢证据。

返回 10 块不等于输入长度一样：每块可能几十词，也可能上千词。`ragscope/generation/context.py` 把标题、引用标签、段间分隔符一起计入参考 token 预算。父块扩展按最佳子块的首次排名，多个命中同一父块只发送一次；邻块和父块不能越过用户指定论文 scope。被预算拒绝的片段留下原因，不能悄悄消失。

重叠块也不能重复当作新增证据。系统根据精确 canonical 坐标去重，必要时只保留未覆盖的片段，并创建带新坐标的引用 ID。ContextBundle 保存实际片段、证据单位坐标、预算、去重与截断记录以及内容哈希；同一份冻结 bundle 可交给不同生成模型。

参考 tokenizer 用来比较大家收到的文字量，模型自己的 tokenizer 和 chat template 用来保证不会超出上下文窗口。相同字符串在两个 tokenizer 下可能 token 数不同。正式比较必须使同一份上下文同时满足全部主模型限制，还要预留输出 token；不能只给大模型悄悄裁短输入。

E2的四种方法各200条检索来源已齐，`20261014_v2_generation_e2`冻结原定三模型×四方法×200题，共2400个任务；manifest_hash为`6a2862e3fed696fb0224d23abc78437c5eaa318efb190c13b2422724ca3246e5`。该独立分片的来源由主清单引用，主矩阵仍在按固定模型串行推进；本页不读取进行中的test质量均值来改默认模型或方法。E3的1200条Oracle/No Context已经封存，主清单引用这些记录，不能再发同一批请求来凑“主实验完成”。

路由准备也已经成为真实运行：Laya CPU得到1080条原始概率和3条预热，来源`20261014_v2_laya/RELEASE_AFTER_RAW1080.json`，尚未因此获得质量优势。线性、MLP和LSTM在冻结train200标签上训练，七种规定策略已分别保存dev60与test100决策，共1120条。`20261014_v2_routing/ALL_SEVEN_DECISIONS_FROZEN.json`的SHA-256为`f0e0ff0005600e4e9fccd7c814cb8f54a269f06dbac76ab824be03712fd05ddc`。冻结发生在读取test质量之前；三种学习路由最终在dev60题全部选择BM25，这是必须保留的实际输出，不能把“模型训练完成”解释为“学会了更好的方法分配”。在线质量与耗时门槛另需真实配对实验。

4K/8K压力诊断也已真正执行：同8个dev问题每窗口8次正式请求和3次预热，共22次、0失败，实际完整输入为3584/7680 tokens，另留512输出。92个原文不够长的问题有明确排除原因。两个窗口顺序加载，并非两个GPU模型并载；当时CPU侧R3/辅助索引有并行工作，资源采样与墙钟时间保留这一条件，不把它称为独占机器基准。所有生成请求仍计入同一10,000次总预算，详见`docs/execution/GENERATION_HANDOVER.md`。

`ragscope/generation/llm_client.py` 只访问已启动的项目本地服务。模型 revision、权重哈希、量化、后端与模板身份进入记录。默认不启用 thinking 的参数会真实传给后端，但是否生效仍必须靠模型预检核验，不能只靠提示词声称关闭。非流式请求的首 token 时间为 N/A。

`pipeline.py` 接收模型原始输出，要求 JSON 的 answer、citations、abstain 类型正确。解析失败留下原文与 parse_error，答案为空；不会自动把第一段挂为引用。引用只在实际发送的 bundle 中验证；原始检索到但没发出的片段不能作为合法来源。

缓存正式评测关闭。日常 UI 若使用缓存，键需要包含模型、索引、上下文、prompt、scope 和解码配置；错误不缓存，否则一次超时可能被反复展示成“答案”。当前 v2 界面默认关闭答案缓存并明确标记历史回放。

UI资源候选由`scripts/configure_ui_v2.py`生成，区别于直接把`ready`写成true。04:01核验中，全509篇R512/neighbor索引和四模型包已验证，但CPU embedding服务关闭、生成调用锁被E2持有，所以相关交互路径不可用；页面不能自行启动模型或挤进正在执行的实验。候选只写runtime目录，正式GUI/实时生成验收尚未完成；自适应路由也必须有真实门槛和当前运行身份绑定，不能凭离线支持就成为默认。

定位错误时先看：是否找到正确论文；是否找到需要的证据；预算是否丢掉证据；模型是否在证据充分时答错；是否错误拒答；是否生成不可见引用。Oracle Context 只提供标注证据、不提供答案，用来帮助区分检索与生成瓶颈；No Context 是独立对照，不能当作系统自动识别无答案的证明。

离线软件验证入口：`tests/unit/test_generation_v2.py`。真实质量和运行编号以 `reports/v2/FINAL_REPORT.md` 与 `D:/RAGScope_Runtime/runs` 的冻结记录为准；本页不预先宣称任何模型改善。
