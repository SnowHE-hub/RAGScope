# 数据与CPU检索恢复点

此文件由数据/检索执行子任务维护，不替代主STATUS、RECOVERY或实验账本。

已完成真实数据冻结：QASPER509篇，train1000/dev100/test200/global100，技术集32题；官方SQuAD2公开dev全库1204段落及E4固定100题。主数据和SQuAD均已完成独立哈希、证据坐标、scope检查。

已完成BM25：dev11组1100行、额外context400行、两个论文内test各200行、E8全库100行，共2000行，0失败。所有行及配置位于`D:/RAGScope_Runtime/runs/20261014_v2_retrieval/records`。

实际Qwen CPU Dense+Hybrid11组dev矩阵已在训练结束明确handover后恢复。当前终端session13666，实际worker PID28832/create_time1791909810.3224635、launcher18392；worker RUNNING。原session37932/PID43932恢复后因训练RAM释放期间断开的在途HTTP出现WinError10054，exit1已核验；随后从相同缓存和原记录恢复，是保存的有界恢复2/3。原session71547在线程切换时的第一次连接重置也保留；两次均未丢已完成query记录。
已完成前四组共800条真实神经检索记录，第五组F1024_O0尚在编码。训练暂停前2404个完整文本向量落盘，最新5个缓存哈希验证一致。`runs/20261014_v2_retrieval/worker.json`保存当前实际身份，不能因旧进程已结束就同时启动第二个相同campaign。
本项目CPU服务当前PID36892/create_time1791909754.8890424，loopback11436，12线程/12batch线程、-ngl0/--no-op-offload。`TRAINING_RAM_RESUME.json`记录二进制/模型SHA核验、health与1024维实请求PASS；`embedding_service.json`已原子更新。训练释放证据为`runs/20261014_v2_lora/train_2b_1024_seed42/GPU_RELEASE_RECEIPT.json`，仅在明确handover后才恢复。启动后RAM3,069,685,760 bytes，持续编码时16:44:06UTC为2,532,061,184 bytes；其他模型须查实时余量。
旧11224、39808与43932均不得仅凭PID再次清理。历史暂停详情保留在`PLANNED_TRAINING_RAM_PAUSE.json`；恢复脚本`scripts/resume_embedding_after_training.py`拒绝重复启动，不能再次直接执行。当前worker重启已读取新版selected_retrieval记录逻辑；前四组历史不可变行仍可能缺该字段，错误分析应保留ranking/packing无法细分的不确定性。已部分完成但未持久化的历史cold索引耗时不得用cache-hit时间替代。

仅在确认原执行进程已结束且服务READY时，用以下**准确命令**从逐文本向量缓存和逐题结果继续；保持相同模型/数据配置，不重选split：

```powershell
. ./scripts/runtime_env.ps1
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -u scripts/run_retrieval_v2.py --home 'D:/RAGScope_Runtime' --campaign 20261014_v2 --stage dense --embedding-url 'http://127.0.0.1:11436' --embedding-identity 'D:/RAGScope_Runtime/runs/20261014_v2/embedding_identity.json'
```

完整矩阵成功后读取`CANDIDATE_SELECTION.json`，冻结simple/structured两个候选；候选必须来自dev。先对已选方案运行必要的context none/neighbor/parent对照，再按冻结选择准备真实test/E8及E4上下文。主任务书要求2048 reference tokens；生成模型的共同真实token预算检查由生成执行器负责。

```powershell
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -u scripts/run_retrieval_v2.py --home 'D:/RAGScope_Runtime' --campaign 20261014_v2 --selected-expansions --embedding-url 'http://127.0.0.1:11436' --embedding-identity 'D:/RAGScope_Runtime/runs/20261014_v2/embedding_identity.json'
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -u scripts/run_retrieval_v2.py --home 'D:/RAGScope_Runtime' --campaign 20261014_v2 --final-contexts --dataset squad --embedding-url 'http://127.0.0.1:11436' --embedding-identity 'D:/RAGScope_Runtime/runs/20261014_v2/embedding_identity.json'
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -u scripts/run_retrieval_v2.py --home 'D:/RAGScope_Runtime' --campaign 20261014_v2 --final-contexts --dataset qasper --embedding-url 'http://127.0.0.1:11436' --embedding-identity 'D:/RAGScope_Runtime/runs/20261014_v2/embedding_identity.json'
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -u scripts/materialize_routing_contexts_v2.py --home 'D:/RAGScope_Runtime' --campaign 20261014_v2 --embedding-url 'http://127.0.0.1:11436' --embedding-identity 'D:/RAGScope_Runtime/runs/20261014_v2/embedding_identity.json' --routing-manifest-hash f1b6ae1ef21a44b720b53e1a6fffe6600a12d16d51e4b88aefe26a69c0ff952b
```

最终test索引范围：QASPER固定509篇，SQuAD全部1204官方dev段落；全库检索不会收到gold_doc_id。SQuAD默认仅使用dev选定结构方案的Hybrid，恰好100条上下文，不再从SQuAD test搜索更优方法。`--methods`可以按阶段执行bm25/dense/hybrid，结果ID和配置不随执行顺序变化；相同向量可复用，延迟结果必须披露cache-hit与cold编码差异。

待完成的神经重排由chunking_core子任务负责，必须恢复Hybrid原始50候选文字再做50→10，不能用BM25或10→10冒充。生成冻结器由generation_evaluation子任务负责，尚无结果的配置不得生成虚构PASS。
root已明确将R3正式上下文物化交给chunking_core；本子任务负责提供E1选中Hybrid dev100、TEST_CONTEXT Hybrid200、E8 Hybrid100，以及ROUTE_CONTEXT_TRAIN Hybrid200/DEV Hybrid60的源文件。E1每组200是Dense100+Hybrid100两个独立文件，不是一个Hybrid200文件。

`--selected-expansions`仅评估两个已选候选；若结构候选没有父块，再加入预声明的P256_1024作三种扩展诊断。`CONTEXT_SELECTION.json`按dev整组覆盖→连续覆盖、容差0.01、近似时优先none→neighbor→parent冻结；test不可改写它。final-context执行器要求该文件与候选哈希匹配。

检索+SQuAD原16个离线测试及独立审计新增3个手算测试均通过。独立审计命令为`python scripts/audit_retrieval_v2.py --home D:/RAGScope_Runtime --campaign 20261014_v2`，会验证全部已完成运行，保留每项数值与seed42抽查10项。
最近审计PASS：26个完成运行、2800行、12602个原文片段、78个汇总数字。冻结探索性统计协议`configs/v2/retrieval_statistics.json`；待最终test上下文完成后运行`python scripts/summarize_retrieval_statistics_v2.py --home D:/RAGScope_Runtime --campaign 20261014_v2`，按论文或SQuAD原始文章成组抽样2000次，最终数字写入`RETRIEVAL_STATISTICS.json`。
新增编码中断测量保留测试后，检索14+SQuAD3+审计3+统计3=23个相关离线测试通过。向量独立审计命令`python scripts/audit_embedding_indexes_v2.py --home D:/RAGScope_Runtime --campaign 20261014_v2`已实际通过4个完成索引、2235个向量、80个向量缓存对齐抽查；523个去重文本有backend usage，1639个无usage，未补造。
暂停期间已按root要求完成`ragscope/routing/online_retrieval.py`，API为`create_live_retriever(...).retrieve(question,scope,route,model_budget_checks=checks)`，8个接口测试通过，root负责实际S4在线生成执行。在线common budget严格复现先reference-only打包、再仅repack retainedpieces两步逻辑，保留初始retrieval_context_bundle与最终context_bundle，禁止重新加入第一遍已排除候选。当前实际ready probe明确缺最终dev选择，未加载模型；禁止用BM25初筛数据冒充正式在线准备完成。后续新进程物化行保存raw50与selected10，历史行不改写；当前worker28832已重启并加载新版，未来dev行有selected字段；前四组历史行保留，不改写。
root指定的ROUTING_QUERY_MANIFEST哈希为f1b6ae1ef21a44b720b53e1a6fffe6600a12d16d51e4b88aefe26a69c0ff952b，实际源核验通过：train200/159篇、dev60/30篇、test100/59篇，文档跨split无交集。`verified_routing_queries`调用freezer.seeded_subset验证完整顺序，之后只使用清单中的ID；2项测试通过。`scripts/materialize_routing_contexts_v2.py`已实现，尚待最终完整索引才执行；将生成ROUTE_CONTEXT_TRAIN/DEV_{group}_{bm25|hybrid}_{expansion}共520行，不在此执行test100，不构建替代索引，不临时重新抽样。
最终QASPER索引物化按结构候选优先、简单对照随后排序，实验配置与逐题identity不变，实际查询缓存状态继续单列。全部索引就绪后执行上方路由物化命令，随后通知chunking_core源文件完成。
当前恢复后的长进程已经载入新adapter，后续冷编码保存backend usage；旧cache无法回补历史usage。CPU线程对照实测256/512/1024正文tokens对应257/513/1025后端tokens，仅作为有界直接观测，不反填所有旧行。

资源调度边界（root明确指定）：完成dev11、context选择、结构候选all509的TEST200/E8和ROUTE train200/dev60源上下文后，必须等命令正常结束再释放本项目CPUembedding，交给chunking_core顺序跑R3和Laya；不得自动无间隙进入simple大索引。使用`--final-contexts --dataset qasper --family structured`形成独立边界，然后执行路由物化；SQuAD100与simple比较可在资源handback后继续，实验配置/ID不因调度顺序变更。`--family`边界参数相关检索/在线/路由24项测试通过。
辅助排序脚本已实际PASS28运行/3000行，定义与原始source哈希保存；`SOURCE_MAP.json`实际PASS3篇×11组、462child/31parent/153unit，固定seed42与成绩无关。最终实际结果更新后重新运行`summarize_retrieval_auxiliary_v2.py`，R3输出到齐时会读取selected10顺序；统计脚本也将R3论文内/全库描述性CI列入，不增添事后配对对照。

恢复后最新独立核验：30个已完成运行/3200行/13384个原文context片段/90个汇总数字PASS；6个完整神经索引/2623vectors/每索引20共120次缓存对齐PASS；875个去重文本有backend usage、1639历史无usage，未反填。辅助排序也已更新到30运行/3200行。当前S512组编码进行中，未计入完成。
最终全库索引助手新增`build_performance.json`：实际成功尝试的encoding loop、数组构建、落盘、总时长和cache/modelcall计数，并链接逐chunk首次完成观察；不将warm/resumed耗时称作完整历史cold成本。现有14检索测试再次通过；这项测量改变不修改索引语义identity或逐题配置。
