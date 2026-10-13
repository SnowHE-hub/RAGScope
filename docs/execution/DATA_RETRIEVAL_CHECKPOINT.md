# 数据与CPU检索恢复点

本文件是数据/检索子任务的当前准确恢复点；主STATUS/RECOVERY由root维护。不要根据下述旧PID操作其他进程，不自动恢复用户原任务。

## 正在运行（不要启动第二个worker）

正式QASPER结构候选R512全509篇索引正在CPU编码。终端session60616；实际worker PID24020/create_time1791913376.0555313，launcher8052。当前主循环最近输出2168/6551，18:26:42Z逐块凭据已2192/6551；逐块首次观察和文本向量均落D盘，可恢复。实时身份在`D:/RAGScope_Runtime/runs/20261014_v2_retrieval/worker.json`，RUNNING/stage=final_structured_qasper。

当前自有CPU服务PID36892/create_time1791909754.8890424，loopback11436，12线程/12batch线程，CPU二进制、-ngl0/--no-op-offload、pooling last、L2、单slot。准确完整命令/模型和二进制SHA在`runs/20261014_v2/embedding_service.json`。它仍驻留，未向R3/Laya或其他模型宣告权重已释放。

先确认该worker确已退出并读取退出状态。只有确有运行失败、服务仍READY且有界恢复次数允许时，才用**相同准确命令**恢复；不可并行执行两个相同campaign：

```powershell
. ./scripts/runtime_env.ps1
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -u scripts/run_retrieval_v2.py --home 'D:/RAGScope_Runtime' --campaign 20261014_v2 --final-contexts --dataset qasper --family structured --embedding-url 'http://127.0.0.1:11436' --embedding-identity 'D:/RAGScope_Runtime/runs/20261014_v2/embedding_identity.json'
```

完整索引目标6551 children；路径`indexes/20261014_v2/all_chunks/ba23a7f21f5bc3ed9fccb5778163c07a6b2fda530c769ff235e15d44dd5238f6/dense_397d90b3f4fbe03e`。`embedding_first_observations.jsonl`和content-addressed cache均按已完成观察落盘。不能把cache向量数量当作最终manifest已完成。

最近18:26:42Z盘点：2192条durable observations、2601所需chunk已有cache、3950个独立文本/1676459正文tokens尚需编码；最近100次冷调用369.61 tokens/s，剩余约1.3–1.6h只作当时编码估算，不包含查询物化；host可用2365521920 bytes（2.20 GiB）。最新状态文件`STRUCTURED_ENCODING_WORK_REMAINING.json`。

## 已完成并冻结

- 真实QASPER509篇：398train/36dev/75test，train1000/dev100/test200/global100，稳定证据34615单元；技术集32题明确agent_authored_not_human_validated。
- 官方SQuAD2全dev1204段落，E4固定100可答短文本题，source/证据/hash均已实际核验。SQuAD真实上下文尚未物化。
- E1主矩阵BM25/Dense/Hybrid 11组共3300行，0失败；扩展诊断另800行。早期初筛BM25 test400/global100单列探索历史；共44完成运行4600行。
- 最终`CANDIDATE_SELECTION.json`：simple=F512_O0、structured=R512，Hybrid/dev-only，SHA fdb2a034acfb932caec71aff4a67fc02fc9154aa9b35d47e78c18a7184bc8c74。
- `CONTEXT_SELECTION.json`两个候选均neighbor。早期BM25 simple=F512_O64以及none test不能冒充正式结果。
- CPU batch1/2/4基准29次请求完成、向量/排名全部一致；batch2=1.00013倍、batch4=0.92875倍，保持原12thread/requestbatch1。凭据`cpu_batch_benchmark/RESULT.json`，不再增加优化组合。
- 独立审计PASS44runs/4600rows/18430pieces/132汇总数；11神经索引/5292vectors/220缓存对齐PASS。3098去重文本有backend usage、1639历史不可得保留N/A。辅助MRR/nDCG也PASS44runs/4600rows。
- dev图表`reports/v2/retrieval_interim/DEV_RESULTS.md`三张PNG/SVG目视PASS；冻结`analysis_complete_dev/SOURCE_SNAPSHOT.json` SHA eb7de862fe1603e18cbea2a26459560d3aed4e818d7aafa6a34180fe61633429，41dev条件/4100行/16412pieces/123独立数字/seed42抽10/164源hash。新配对表说明R512 neighbor全覆盖8改善7退步净1，连续下降；F512_O0全覆盖16改善14退步净2，连续下降，不作显著性结论。

## 当前命令正常结束后的下一条准确命令

更新worker receipt COMPLETE/exit0并确认PID身份不再运行；然后立即物化已冻结ROUTE train200/dev60的BM25+Hybrid共520条，不另抽样：

```powershell
. ./scripts/runtime_env.ps1
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -u scripts/materialize_routing_contexts_v2.py --home 'D:/RAGScope_Runtime' --campaign 20261014_v2 --embedding-url 'http://127.0.0.1:11436' --embedding-identity 'D:/RAGScope_Runtime/runs/20261014_v2/embedding_identity.json' --routing-manifest-hash f1b6ae1ef21a44b720b53e1a6fffe6600a12d16d51e4b88aefe26a69c0ff952b
```

随后既定**正常退出资源边界**：全部dev11、R512完整509、TEST200/global100与ROUTE260已齐，确认自身worker结束并持有campaign单写锁，记录已完成源hash与准确service PID/create/command，只停止本项目无在途请求的CPU embedding。保留cache与全部记录，保存释放/恢复凭据。不得自动进入simple索引。

R3由chunking_core处理，同retrieval Campaign单写锁。交接给其以下五组Hybrid源（全部neighbor）：

- E1_R512_hybrid_neighbor.jsonl：dev100，已经就绪。
- TEST_CONTEXT_R512_hybrid_neighbor.jsonl：test200，当前待物化。
- E8_R512_hybrid_neighbor.jsonl：global100，当前待物化。
- ROUTE_CONTEXT_TRAIN_R512_hybrid_neighbor.jsonl：train200，待物化。
- ROUTE_CONTEXT_DEV_R512_hybrid_neighbor.jsonl：dev60，待物化。

必须保留raw50与真正selected10；历史缺selected字段的误差类型保持不确定。root最新冻结资源顺序：structured+ROUTE完成/停止embedding → R3五组完成退出 → generation_evaluation的22-call 2B 4K/8K诊断完成退出GPU → Laya正常5GiB门槛或有证据的一次4.5–5GiB预检备选 → 归还本子任务CPU槽。不要自行并载；压力诊断有效guard为3GiB，由root与generation协调。等待实际资源归还后，核验同一CPU12服务可靠恢复，再继续下列两个命令，不改冻结问题/算法/模型/扩展。simple已通过独立SIMPLE_SCOPE_PLAN预检：75篇/777chunks，保留509篇/5710chunks的BM25参考统计，test200的top50排名与分数全相等；省去4933个额外向量行（含cache复用，不是4933次冷调用）。后续simple使用scoped_chunks，明确拒绝global；准确证据位于runs/20261014_v2_simple_scope_preflight/。不再执行simple全509神经编码：

```powershell
. ./scripts/runtime_env.ps1
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -u scripts/run_retrieval_v2.py --home 'D:/RAGScope_Runtime' --campaign 20261014_v2 --final-contexts --dataset squad --family structured --embedding-url 'http://127.0.0.1:11436' --embedding-identity 'D:/RAGScope_Runtime/runs/20261014_v2/embedding_identity.json'
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -u scripts/run_retrieval_v2.py --home 'D:/RAGScope_Runtime' --campaign 20261014_v2 --final-contexts --dataset qasper --family simple --embedding-url 'http://127.0.0.1:11436' --embedding-identity 'D:/RAGScope_Runtime/runs/20261014_v2/embedding_identity.json'
```

## 已实现且待真实就绪调用

`ragscope/routing/online_retrieval.py`工厂从最终选择与已完成all509索引加载真实BM25/Hybrid/HybridRerank，输入只有question+scope；逐题外部传入所有formal模型共同预算checks。严格reference-only初pack→仅retainedpieces共同预算repack，保存raw50/selected10/窗口和实际计时。8测试PASS。实际probe目前WAITING_FOR_FINAL_DENSE_INDEX，无模型调用，不是伪造ready。

`verified_routing_queries`实际校验f1b6...清单与freezer.seeded_subset排序/seed42完全相同：train200/159篇、dev60/30篇、test100/59篇，跨split不重叠。test100从已有TEST_CONTEXT200复用，不在物化器重新采样。

最后重跑独立核验：`scripts/audit_retrieval_v2.py`、`scripts/audit_embedding_indexes_v2.py`、`scripts/summarize_retrieval_auxiliary_v2.py`、`scripts/summarize_retrieval_statistics_v2.py`（均--home D:/RAGScope_Runtime --campaign 20261014_v2）。统计协议2000次seed42成组bootstrap、QASPER论文/SQuADarticle聚类，明确探索性；未完成条件保留UNAVAILABLE。

## 历史恢复与限制（仅证据，勿据旧PID操作）

训练RAM边界曾暂停43932并停止自有39808；训练122步/972例完成后，按正式GPU_RELEASE_RECEIPT核验再恢复为当前36892。旧worker43932在途连接WinError10054后exit1，新worker28832从缓存恢复并已完成dev11/exit0。加上第一次CPU线程切换的连接重置，目前有界恢复已使用2/3；两个事件均存embedding_runtime_failures.jsonl，没有丢完成query。`PLANNED_TRAINING_RAM_PAUSE.json`、`TRAINING_RAM_RESUME.json`保留完整来源。一次性resume_embedding_after_training.py不能再次运行。

旧索引部分首次cold成本没完整保存，必须保留不可得，不能用warm/resumed成功时长冒充全cold。未来全库build_performance.json记录实际成功尝试及first observations。
历史两个all509 BM25 chunkmanifest的preflight.split误写dev，其identity/数据/hash正确；CHUNK_PREFLIGHT_SCOPE_AUDIT明确PASS_WITH_METADATA_LIMITATION，旧manifest未改写，新构块用真实范围/样本文档ID。SQuAD沿用scientific-paper query指令且领域/长度不同，只作独立回归，不作长度因果解释。
