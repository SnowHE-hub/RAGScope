# 数据与CPU检索恢复点

本文件是数据/检索子任务的当前准确恢复点；主STATUS/RECOVERY由root维护。不要根据下述旧PID操作其他进程，不自动恢复用户原任务。

## 当前：simple三方法600条完整PASS，资源已归还root（04:31Z）

root明确授权复用仅3条有固定SHA来源的历史backend_usage=null缓存后，`materialize_simple_cache_only_v2.py --allow-historical-usage`已实际执行完毕，session90177 exit0，worker23220/create1791952156.3107731已不存在。主writer已实际acquire/release证实释放，11436保持关闭；available2575036416B。原BM25200由ledger复用，新增Dense200+Hybrid200，三组全部ID/config/record hashes独立核验、0error。实际406次缓存命中（400正式+6检索预热），网络/HTTP请求0，新推理0，GPU0；旧3/3后端失败journal不变。3个旧vector/manifest字节未动、仍未伪造usage sidecar，后两组相应metrics保留null。4项离线定向行为检查PASS。

记录位于retrieval/records/TEST_CONTEXT_F512_O0_{bm25,dense,hybrid}_neighbor.jsonl。Hybrid SHA8af1439521c2b1e999123dfd91946d88e77cca62e28a0a45711530bb92ba509a；Dense SHA65cbe75f02a773588276b2e04d4d61aa2a2195b19e0b7cd663c6c8c28bfb6805；BM25 SHAe0ec15b9aea2a06ad4c86ff8f798410990dd83274fab68e3b2106fa719a1825c。新receipt在retrieval/SIMPLE_CACHE_ONLY/HISTORICAL_NA_REUSE：RESULT SHA3d98ccdb2b8fee111b56cd97ff985b77615b324351b92c1a8c2b49ffe5f1448f，EXIT_OBSERVATION SHAf3fbdada10e4c5259e279715f002a209fcdd9d5f297844e67116e84728c28ab7；DECISION_BEFORE_RUN在启动前保存原来源及执行源码SHA。严格检查的首次partial记录全部保留，没有覆盖。

当前没有自有服务/worker/锁。SQuAD最终检索仍未运行；下一准确动作是等待root在4B生成正常结束并释放GPU服务后的短CPU窗口通知，再按原固定CPU4、11436、权重身份和启动guard完成SQuAD100真实新query及上下文。不要提前启动服务，不重新执行simple600，不碰其他原任务。用户最新明确任何阶段可用GPU覆盖旧阶段禁令，但本次cache-only真实仍为零GPU；后续单卡调度由root统一，R3将另获GPU窗口。

## 历史：strict cache-only先完成BM25并保留三项usage缺失（04:27Z）

R3在已提交329条后触发原1.25GiB运行RAM保护并退出，root核验其RESOURCE_PAUSE_PID30108.json SHA8b6e9e388cab318355eed66315a3889c91e8ae6cf62b3d6559cd23bbb8b1d4b5后，明确把主writer临时归还给本子任务，授权只读真实query cache、不启动CPU服务的simple最终物化。新入口`scripts/materialize_simple_cache_only_v2.py`用原run_final_contexts及同配置/身份，限定CPU4；网络/HTTP运输层主动拒绝，cache miss直接PENDING而不生成伪失败行，3个定向行为测试PASS。

实际preflight：777块索引及aux handoff全hash有效；200题向量都在，但仅197题usage sidecar完整。3题历史usage缺失，未补造；按授权独立完成`TEST_CONTEXT_F512_O0_bm25_neighbor`200条/0error。Dense/Hybrid未执行，SQuAD未启动。session50812已结束，工具exit1、脚本partial意图exit2均保存在EXIT_OBSERVATION；worker34676/create1791951901.955112实际不存在，11436关闭、主writer能acquire/release，available2235445248B。旧3/3后端失败journal字节未变。RESULT SHA247d772f2a63f5dc1341930e0f026ec0750df5a1d6af9e0fe383595054dba3c1，EXIT_OBSERVATION SHA44f703b9cd53c5ccb88fe639c52935437411b45db3046a6049b9063d7c78e429，位于retrieval/SIMPLE_CACHE_ONLY。

同目录MISSING_USAGE_PROVENANCE.json只读关联3个完整向量/manifest与原structured Dense和Hybrid共6条真实记录：同问题/identity、cache_hit=true、backend_usage=null、正文tokens29/29/30。不可把N/A补写成实测usage。root已获说明；当前没有活跃自有worker/服务，不自动重跑，不重新编码已存在向量。下一准确动作：等待root对这3项历史N/A来源及资源窗口的调度；如果允许复用，保留旧receipt后以明确旁证约束扩展cache-only预检；否则继续PENDING，只在获准CPU窗口执行真正缺失的SQuAD查询。

## 已完成：aux和UI候选核验（04:01Z）

UI候选入口已实际完成，session1127 exit0：`main/ui_candidate/ui.json` SHA73ef6704d26afeba4eb961057ad0a86e2d71d43c0bd21b7e2116746f78e5ff81；`ui.verification.json` SHAeca75ef12397ee7643e8d7ed9574dcaf34ca094e6936196da609461bd2856d9f。全509/6551索引、四模型真实权重/模板、GET metadata通过；11436不可用，E2持生成锁，所以神经检索/交互生成标不可用。默认意图2B+fixed Hybrid/neighbor；非fixed不启用。项目ui.json字节未变；13定向UI检查PASS。详细依据和保留的一次回放文件名失败见UI_CANDIDATE.md。当前没有本子任务worker/模型。下一准确依赖操作仍是等待root明确主writer释放/资源交回，然后启动经核验CPU4服务并执行本文件下方既定SQuAD/simple最终上下文命令，不能自行抢R3锁；aux不能重复执行。

aux session99141已**exit0**，worker30192与service33428的准确身份均已不存在，11436端口关闭。03:50:23Z独立退出观察available4863193088 bytes；`EXIT_INDEPENDENT_OBSERVATION.json`与`SERVICE_LIFETIME.json`（SHA85ee127ae1b75a0f51dea486c353891e49cf1da23e831f6d7049835e1bfd738c）保存证据。没有新增后端失败/重试，原3/3journal字节未变。当前本子任务**没有活跃模型或worker**，不得据下述历史PID或一次性启动脚本再次启动。

SQuAD1212、simple777共1989向量全部完成。实际冷embedding calls分别1212/775，cache hits0/2；encoding分别1350.932秒/2537.883秒，不是独占资源性能基准。03:50:58Z独立readback session75664 **exit0/PASS**：正确expected identity/全部manifest artifacts、1989个float32/1024维/finite/unit norm、1989次逐项cache完全一致、全部正文<=512、完整1989条observations集合/row hashes。`INDEPENDENT_INDEX_READBACK.json`与`AUX_INDEX_HANDOFF.json`在独立aux run。SQuAD manifest SHA45cbd9c90f261841d8255bcfa6441e482e48f6975232d22f3ad8ccff71d4ea25；simple manifest SHAfb12663a59afd35fa1e105f67b45abbea273f2d5dd52d9bd3db777367facc03c。

本次**新增quality experiment行数仍为0**，SQuAD/E6最终上下文尚待主campaign物化；不能把预编码写成实验完成。root确认R3仍持主writer锁（先前dev100/test200已完，继续global100/ROUTE260），必须等root正式锁释放通知才进入主campaign。不要重跑aux或readback。

上述UI独立操作已完成，准确收据见顶部和UI_CANDIDATE.md；不是GUI最终验收。必要的后续QA由root统一安排。

## 历史：8小时截止下的独立CPU4预编码启动（2026-10-14 02:43Z）

用户已resume并新增10:29:06Z硬截止。root明确授权在R3持有主retrieval单写锁期间并行预编码SQuAD R512和scoped-simple777：只写各自索引及`20261014_v2_aux_preencode`独立campaign，不写主实验records/state，不改选择/语料/向量语义，不增加组合。下文旧的“等待全部R3→stress→Laya后才能开始索引”顺序已由本次授权取代；正式逐题物化仍必须等主锁释放。

实际命令`. ./scripts/runtime_env.ps1; & 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -u scripts/preencode_aux_v2.py`已运行，exec session **99141**。worker **30192/create1791945828.5850294**，自有CPU4 service **33428/create1791945830.6247053**。不要重复启动；下一准确工具操作是`write_stdin(session_id=99141,chars='',yield_time_ms=1000)`观察该已有句柄。`runs/20261014_v2_aux_preencode/SERVICE_LIFETIME.json`保存启动源码SHA、准确argv、固定权重/二进制、主锁实际被持有、独立账本、deadline、旧3/3失败journal SHA。启动前available3642847232 bytes，真实1024维控制请求通过后2734419968 bytes；CUDA空串/-ngl0/--no-op-offload/4线程。

三个离线定向行为检查PASS：主锁持有时只写独立账本/目标锁；目标身份与最终消费者公式一致；低RAM/截止阻止下一请求且只保留已完成metrics。实际请求前1GiB guard，不称在途自动暂停；不足则在请求边界退出并由controller释放准确自有service，无自动后端重试。历史网络恢复仍3/3，不重置。

SQuAD实际1204段→**1212个R512 chunks、0 parent、204332 Qwen body tokens、冷cache1212/1212**；完整结构坐标已核。WORKLOAD.json与精确目标在`squad_chunks/6ed89a898b20e5d929221fbed76465c8461405c406d6363fc9049540bdaa3f2b`，其自身writer.lock与R3的主锁/structured索引互不相同。已开始真实编码；完成后同一入口继续冻结simple777。aux仅索引，quality experiment新增行数为0，不冒称SQuAD/E6已完成。两个索引完成会正常退出/释放服务，随后由root安排主campaign最终物化。

更新：SQuAD1212/1212已实际完成，1212冷模型调用/0cache hits，encoding1350.932秒；dense manifest SHA45cbd9c90f261841d8255bcfa6441e482e48f6975232d22f3ad8ccff71d4ea25，index signature5bdf788ee95cf8e78f4ac8b4a51db38f768ca716657caa6f17694ffb178f4075。已顺序进入simple F512_O0，777块/377551正文Qwen tokens，开始时2条cache命中、775条未命中；精确目标`scoped_chunks/da7d14af299a8b4ed9e25f662e7e00b6a4edb4a03897edc1fb84fe6b684aad08`。仍同一session99141/worker30192/service33428，勿重复启动。

两索引全部完成且controller exit0/自有service退出后，下一只读核验命令为`. ./scripts/runtime_env.ps1; & 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' scripts/verify_aux_preencode_v2.py`。该命令逐一按最终消费者身份从磁盘load，核manifest/artifacts/全部float32维度1024/finite/unit norm/逐向量cache完全一致/正文<=512/observations集合和row hashes；无模型调用或主campaign写入。结果写aux/INDEPENDENT_INDEX_READBACK.json。新建索引save分支没有执行磁盘reload，aux task字段index_loaded_and_validated仅覆盖构造对象；以此单独readback收据确认真实落盘复载，不能提前声称已做独立读回。

root后续新增独立交付：aux/readback/正常释放完成后、等待R3主锁时，实现生产UI配置候选入口（拟`scripts/configure_ui_v2.py`）与实际只读核验收据。先CodeGraph核`ui/chat_app.py`、model_identity和routing_options；全509 R512/neighbor、已完成dense真实index+hash/embedding身份、原四模型UI_MODEL_PACKAGE_PINS，默认只按dev/可运行最小2B，不读test选默认。非fixed路由必须真实门槛binding；missing服务明确不可就绪。先写runtime candidate，不改项目configs/v2/ui.json，待root最终UI QA再发布。0.34.4专用11435只允许metadata-only probe，先与root/gen协调，不抢其请求/不startup模型。无GPU/模型调用、不占主writer、不新增实验组合；GUI真实QA留root窗口。此UI源码尚未开始，不能套用root正在运行的旧全套core测试收据，必要时做对应定向验收。

## 已完成structured与ROUTE及之前资源交接

正式R512全509篇索引6551/6551已完成，manifest/count/artifact SHA实际验证PASS；structured TEST200/global100 × BM25/Dense/Hybrid共900行，ROUTE train200/dev60 × BM25/Hybrid共520行均正常exit0，全部row/config hashes、冻结query ID、0error已核。主机重启前5976条及所有旧结果保留；网络恢复历史仍3/3，没有伪造第四次错误或重置预算。

structured实际worker3312/create1791942111.7390096/session37283和ROUTE worker10584/create1791944091.5318315/session25598均已退出。ROUTE身份在运行时工具实际观测，写RUNNING receipt前已正常退出，因此只记录真实观察和退出凭据，不伪造启动时落盘。STRUCTURED_WORKER_EXIT.json与ROUTING_WORKER_EXIT.json为准。

单slot真实1024维drain barrier后，仅停止准确自有CPU4 service6016/create1791942102.6736171；确认PID不存在、11436关闭。当时主service receipt为EXITED_PLANNED_R3_HANDOFF。释放后available6705647616 bytes（约6.25GiB）。STRUCTURED_ROUTING_RESOURCE_HANDOFF.json SHA d9dca64ec3f03cbcad80f7e9d9dadf623fc4bdad2fac4ebb83ff479adf2cbe08保存全部退出、来源、sidecar、选择/index及恢复条件。随后授权的aux活跃状态见本文件顶部。

已把R3五组660条R512 neighbor Hybrid来源交root与现有chunking_core；root已统一发加载指令。R3→压力诊断→Laya的其他agent资源窗口仍由root协调。新增aux仅按顶部明确授权隔离并行，不抢campaign单写锁，不重复全回归。

aux结束后的依赖性操作：主锁实际可用且root归还资源时，按相同固定权重/binary、CPU4恢复独立服务，确认host≥3GiB、11436free/无旧PID，验证health和1024维并记账，再执行下文既定SQuAD和scoped-simple两条最终物化命令。不得复用一次性HOST_RESTART_REENTRY或aux服务启动脚本。

## 已完成并冻结

- 真实QASPER509篇：398train/36dev/75test，train1000/dev100/test200/global100，稳定证据34615单元；技术集32题明确agent_authored_not_human_validated。
- 官方SQuAD2全dev1204段落，E4固定100可答短文本题，source/证据/hash均已实际核验。SQuAD真实上下文尚未物化。
- E1主矩阵BM25/Dense/Hybrid 11组共3300行，0失败；扩展诊断另800行。早期初筛BM25 test400/global100单列探索历史；共44完成运行4600行。
- 最终`CANDIDATE_SELECTION.json`：simple=F512_O0、structured=R512，Hybrid/dev-only，SHA fdb2a034acfb932caec71aff4a67fc02fc9154aa9b35d47e78c18a7184bc8c74。
- `CONTEXT_SELECTION.json`两个候选均neighbor。早期BM25 simple=F512_O64以及none test不能冒充正式结果。
- CPU batch1/2/4基准29次请求完成、向量/排名全部一致；batch2=1.00013倍、batch4=0.92875倍，当时保持12thread/requestbatch1；最新AGENTS后续启动改为CPU4，批量仍为1。凭据`cpu_batch_benchmark/RESULT.json`，不再增加优化组合。
- 独立审计PASS44runs/4600rows/18430pieces/132汇总数；11神经索引/5292vectors/220缓存对齐PASS。3098去重文本有backend usage、1639历史不可得保留N/A。辅助MRR/nDCG也PASS44runs/4600rows。
- dev图表`reports/v2/retrieval_interim/DEV_RESULTS.md`三张PNG/SVG目视PASS；冻结`analysis_complete_dev/SOURCE_SNAPSHOT.json` SHA eb7de862fe1603e18cbea2a26459560d3aed4e818d7aafa6a34180fe61633429，41dev条件/4100行/16412pieces/123独立数字/seed42抽10/164源hash。新配对表说明R512 neighbor全覆盖8改善7退步净1，连续下降；F512_O0全覆盖16改善14退步净2，连续下降，不作显著性结论。

## 已完成的ROUTE命令（历史，不再执行）

以下命令已实际exit0，固定ROUTE train200/dev60的BM25+Hybrid共520条已全部完成，不再执行。当前准确下一操作为等待root归还资源，再按后文CPU4服务条件与SQuAD/scoped-simple两条命令继续：

```powershell
. ./scripts/runtime_env.ps1
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -u scripts/materialize_routing_contexts_v2.py --home 'D:/RAGScope_Runtime' --campaign 20261014_v2 --embedding-url 'http://127.0.0.1:11436' --embedding-identity 'D:/RAGScope_Runtime/runs/20261014_v2/embedding_identity.json' --routing-manifest-hash f1b6ae1ef21a44b720b53e1a6fffe6600a12d16d51e4b88aefe26a69c0ff952b
```

本轮已完成的**正常退出资源边界**（详见STRUCTURED_ROUTING_RESOURCE_HANDOFF.json）：全部dev11、R512完整509、TEST200/global100与ROUTE260已齐，确认自身worker结束并持有campaign单写锁，记录已完成源hash与准确service PID/create/command，只停止本项目无在途请求的CPU embedding。保留cache与全部记录，保存释放/恢复凭据。不得自动进入simple索引。

R3由chunking_core处理，同retrieval Campaign单写锁。交接给其以下五组Hybrid源（全部neighbor）：

- E1_R512_hybrid_neighbor.jsonl：dev100，已经就绪。
- TEST_CONTEXT_R512_hybrid_neighbor.jsonl：test200，已物化并验hash。
- E8_R512_hybrid_neighbor.jsonl：global100，已物化并验hash。
- ROUTE_CONTEXT_TRAIN_R512_hybrid_neighbor.jsonl：train200，已物化并验hash。
- ROUTE_CONTEXT_DEV_R512_hybrid_neighbor.jsonl：dev60，已物化并验hash。

必须保留raw50与真正selected10；历史缺selected字段的误差类型保持不确定。root最新冻结资源顺序：structured+ROUTE完成/停止embedding → R3五组完成退出 → generation_evaluation的22-call 2B 4K/8K诊断完成退出GPU → Laya正常5GiB门槛或有证据的一次4.5–5GiB预检备选 → 归还本子任务CPU槽。不要自行并载；压力诊断有效guard为3GiB，由root与generation协调。等待实际资源归还后，按最新AGENTS核验同一模型CPU4服务可靠恢复，再继续下列两个命令，不改冻结问题/算法/模型/扩展。simple已通过独立SIMPLE_SCOPE_PLAN预检：75篇/777chunks，保留509篇/5710chunks的BM25参考统计，test200的top50排名与分数全相等；省去4933个额外向量行（含cache复用，不是4933次冷调用）。后续simple使用scoped_chunks，明确拒绝global；准确证据位于runs/20261014_v2_simple_scope_preflight/。不再执行simple全509神经编码：

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

训练RAM边界曾暂停43932并停止自有39808；训练122步/972例完成后，按正式GPU_RELEASE_RECEIPT核验再恢复为当前36892。旧worker43932在途连接WinError10054后exit1，新worker28832从缓存恢复并已完成dev11/exit0。加上第一次CPU线程切换的连接重置，当时有界恢复已使用2/3；两个事件均存embedding_runtime_failures.jsonl，没有丢完成query。`PLANNED_TRAINING_RAM_PAUSE.json`、`TRAINING_RAM_RESUME.json`保留完整来源。一次性resume_embedding_after_training.py不能再次运行。

旧索引部分首次cold成本没完整保存，必须保留不可得，不能用warm/resumed成功时长冒充全cold。未来全库build_performance.json记录实际成功尝试及first observations。
历史两个all509 BM25 chunkmanifest的preflight.split误写dev，其identity/数据/hash正确；CHUNK_PREFLIGHT_SCOPE_AUDIT明确PASS_WITH_METADATA_LIMITATION，旧manifest未改写，新构块用真实范围/样本文档ID。SQuAD沿用scientific-paper query指令且领域/长度不同，只作独立回归，不作长度因果解释。

当前structured进程的源码来源：ACTIVE_WORKER_CODE_PROVENANCE.json明确未记录启动时source SHA、没有热重载或因scoped补丁重启。旧实现git16d49f8归档SHA0565384e214b563c27782af0cea862e186ee4ec2d767a511be40d5022f19b691为事后重建，当前磁盘新版SHA只适用于后续启动；不得混称。

运行时内存边界：当前补丁前worker只有开始时768MiB guard，没有每请求自动内存检查或自动暂停机制；运行中由外部持续观察，不得称自动保护。19:02:03Z available942284800 bytes（约0.88GiB），worker RSS325033984、CPU service RSS1387638784；前一分钟也曾回升，实际波动保存structured_resource_observations.jsonl。保持现有进程且不增加负载。若持续极低RAM或请求错误需要干预，先保存原因、已落盘进度、准确自身PID/create/argv，再按有界恢复处理；不操作其他进程，不重启用户原服务、WSL或安全/维护进程。

19:26附近低RAM事件：先保存身份/缓存/5088条observations，暂停自身worker，单slot真实drain barrier通过后停止仅PID36892 CPU服务。STRUCTURED_LOW_RAM_PAUSE.json、STRUCTURED_PAUSED_BATCH_CACHE.json保留完整证据。未计划连接恢复仍2/3，尚未恢复所以不能预记消耗第三次。最近暂停worker RSS164515840 bytes、host可用2772496384 bytes（2.58GiB），仍低于3GiB恢复guard。恢复script经AST语法检查，尚无模型加载调用；原training恢复脚本的一次性main不可重跑。

恢复脚本现已修正resume前/后的异常所有权边界并通过2个离线行为fixture（无模型）：resume前失败仅清理身份匹配自有service，resume后写receipt失败保留service和worker_resumed=True。最新源码SHA f856b5d4b7fc126bcaae0f786abb3be7798d59f42928c36fd6a0a0c6b14d0039，以pause receipt的artifact为准。只读确认已运行Ubuntu内部MemAvailable约10.05GiB，但不能据此断言Windows物理可用或降低3GiB guard；原始数字/命令在WSL_READONLY_MEMORY_EVIDENCE.json。未改WSL、内存回收或其他程序。

2026-10-13 19:45Z：已核验 guard 来源，3 GiB 是此前 agent 恢复脚本的保守门槛，不是任务书固定数值；原服务 requested 1 GiB + reserve .5 GiB。历史加载峰值 N/A，稳态 RSS 1.28–1.31 GiB。root 明确授权一次受监督备选：连续两次相隔至少30秒的2.5–3 GiB观察，加载期间每.25秒检查、低于1 GiB只停止精确自有stateless服务，worker保持暂停；健康60秒/真实1024维验证30秒；监控必须停止并join，再核身份及1 GiB后才resume。恢复后异常保留service。>=3 GiB优先原路径，不消费备选。5个离线行为测试PASS，无模型加载。此次脚本SHA 8f99d1a35eda13703fe4d3e5574aeec174779a7754fe20f2d1b32e2a3a53610b；下一准确命令为原resume命令附 --supervised-fallback，实际路径由fresh RAM决定；最多一次备选凭据以x独占创建，不能反复尝试或再降门槛。

19:46Z实际受监督恢复PASS：唯一备选已消费，加载前available2716409856 bytes；9个.25秒采样最低1744457728 bytes，采样最高service RSS/Windows peak_wset948436992 bytes（只涵盖本次加载到控制请求结束，不能冒充后续运行峰值）。真实1024维控制请求通过，监测线程已join后resume旧worker。CPU新PID19112/create1791920756.4124439。旧worker24020在历史socket收到WinError10054并exit1，finally保留5088→5090条first observations；主服务没有再重启。第三/最后一次缓存重放已登记，new worker20816/create1791920803.708412/session22849真实运行并逐条读取既有缓存。重放控制台位置从8重新计数仅表示缓存读取位置，不代表丢失5090条已完成编码。启动源码SHA现已真实测量，见STRUCTURED_THIRD_REPLAY_STARTUP.json。恢复预算3/3已消费，不能继续无限失败重试。下一条命令仍为待本structured进程正常完成后运行materialize_routing_contexts_v2.py，不能并写。
