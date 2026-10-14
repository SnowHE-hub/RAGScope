# R3正式重排执行记录

2026-10-14，当前状态：**dev100与TEST200真实重排已完成并通过独立验收，TEST200已交接E2；E8第29条提交后触发既定RAM保护。用户随后授权所有阶段使用GPU，固定剩余条件的GPU360已真实完成并通过独立验收**。有效五组660条现由CPU dev100/TEST200及GPU global100/ROUTE200/60构成。CPU329条均原样保留；global使用独立完整GPU100，ROUTE使用GPU200/60，额外29条CPU历史部分执行明确披露，不混合设备或删样本。11组完整dev矩阵冻结structured=R512、simple=F512_O0，两者扩展均为neighbor。Laya尚未加载。

入口`python scripts/run_rerank_v2.py`只检查冻结输入；主执行器交回RAM后在models CPU环境加`--run`。可选`--prefix`仅指定冻结任务，不改变人群。全量五组为selected dev100、test200、E8 global100、ROUTE train200/dev60。train/dev路由ID直接来自主执行器的`ROUTING_QUERY_MANIFEST.json`，检查其manifest_hash和corpus来源，不重新采样。

`ragscope/experiments/rerank_campaign.py`只接收真实、无失败、哈希完整的Hybrid原始记录与对应index identity。恢复每一候选的完整canonical文本，逐项比较doc_id、坐标和unit_spans。必须有50条真实候选；如果指定论文中总块数少于50，使用全部可用块并显式记录shortfall，不重复填充或将旧top10冒充50。按已预检的BGE windowed_max对完整query+片段评分，返回前10，沿用最终dev冻结扩展方式和2048参考token上下文预算。

输出保留`raw_retrieval`（Hybrid原始50）、`selected_retrieval`（重排10）、每个候选所有窗口的字符区间/query-inclusive token数/实际分数、输入与配置hash、上下文与证据指标、失败、实际RSS与耗时。指标以重排选中的前10和实际打包上下文计算。源检索与新重排分时执行，合成耗时明确不能冒称在线端到端延迟。每组另存3个真实预热，排除在正式耗时摘要外。

正式进程另在`R3_RUNTIME/worker.json`写入实际PID/创建时间/argv/cwd，在加载模型前启动1秒周期RSS及主机可用RAM采样；完成或异常退出时停止采样并标记EXITING。外部验收必须再次确认实际进程已退出，不能将该标记单独当作资源已释放。

`R3_RUNTIME/LOAD_STATE.json`单独保存加载尝试及后端失败，最多3次；RAM暂停单列PENDING_RESOURCE，不消耗后端失败次数。该守卫与观察器退出回归共16项PASS（3.13秒），日志`rerank_load_retry_bound_tests.log`。2026-10-13 18:01:28 UTC（Sydney次日）上游实测仍有5,240个未缓存文本、2,223,347正文tokens，主机可用RAM约1.97GB；仅按最近吞吐估算约1.69小时编码，随后还需查询物化，不据此承诺资源交接时间。

公共加载器`ragscope/retrieval/verified_reranker.py`被R3和在线路由共同使用：仅固定本地BGE revision `2cfc18c9415c912f9d8155881c133215df768a70`，逐文件SHA校验；CPU-only Torch、四线程、max_length512、overlap32、batch2。没有自动下载、启动服务、GPU切换或原用户任务控制。加载前至少3GiB可用RAM，逐题落盘后留1.25GiB；资源暂停保留记录且不消耗后端失败重试预算。

正式输出命名与generation freezer一致，位于retrieval campaign的records/：`TEST_CONTEXT_<group>_hybrid_rerank_<expansion>.jsonl`、`ROUTE_CONTEXT_{TRAIN|DEV}_...`、`E1_...`与`E8_...`。运行持有同campaign单写锁，防止与数据agent并发改账本。

初轮与E0合计17项离线测试通过，3.40秒；追加冻结ID回归后R3专项12项通过，3.66秒，日志`D:/RAGScope_Runtime/logs/rerank_materialization_tests_final.log`。夹具均标offline_fixture/real_model=false，正式RecordStore拒绝它们。覆盖50候选完整性、短scope、坐标/ID/排序/NaN拒绝、global不使用gold过滤、候选第50移入前10后指标正确、失败保留、source hash与配置绑定、manifest修改拒绝、直接复用主执行器ID顺序。真实质量仍待后续实际结果。

追加本地模型manifest路径及禁止设备的前置测试后，与统一runner共37项通过，9.03秒，日志`D:/RAGScope_Runtime/logs/rerank_runner_integration_tests.log`。无模型的readiness CLI实际返回非零并明确指出最终CANDIDATE_SELECTION缺失。正式任一失败不会被摘要丢弃，连续3次失败停止并保留剩余待执行人群，避免在明显失效的后端上无限继续。

最终输入已冻结后的首轮无模型检查：`E1_R512_hybrid_neighbor.jsonl`100题来源与406块dev索引的完整canonical候选恢复全部PASS。单论文scope的真实候选数为5–28，平均11.57，100题均少于50；这不是丢候选，全部可用scoped块均经ID/坐标验证。证据`R3_DEV_CANDIDATE_VALIDATION.json`逐题记录可用块和实际候选数，不能将此范围实验描述为每题严格50→10。进程观察器加入后R3专项15 PASS（3.53秒），日志`rerank_worker_observation_tests.log`。

2026-10-14 02:21 UTC，全量交接独立验收`R3_INPUT_VALIDATION.json`为PASS：660个冻结ID、row/config/source/index哈希及canonical候选全部匹配。dev实际5–28（均值11.57），test3–50（11.69），global全部50，ROUTE train3–50（13.43），ROUTE dev5–21（11.4833）；分别有100/199/0/198/60题少于50。上游3312/10584/6016已退出、11436关闭，未在验收时导入Torch。

2026-10-14 02:57 UTC，`R3_GROUP_E1_VALIDATION.json`签收100/100正式dev记录、0失败、3条独立排除预热、1,826个完整候选窗口，实际query-inclusive窗口最大512。全部windowed_max排序、来源坐标、2048预算与证据指标从落盘记录重算一致。92题证据指标适用，完整证据参考集覆盖率为0.652174，连续覆盖率0.675725；这些是实际描述性结果，不能仅据它们宣称相对Hybrid提高。平均上下文1,906.2参考tokens；摘要p50合成离线延迟17.462秒、p95 30.386秒，不是在线端到端延迟。日志`rerank_dev100_integrity_acceptance.log`。

独立轻量验收入口`core/Scripts/python.exe scripts/validate_r3_v2.py`默认要求模型worker已退出且五组660及15条排除预热完整；`--prefix TEST_CONTEXT`只签收完整不可变test200组，明确`whole_campaign_complete=false`，允许原worker继续其他组。该分组签收用于用户八小时截止下的E2核心先行调度，不增加实验。九项窗口人口、坐标覆盖、排序、预算、源身份破坏测试实际通过，日志`rerank_validation_tests_final.log`；真实dev100全组验收亦已运行通过。硬截止为2026-10-14 10:29:06 UTC（Sydney21:29:06），配置、失败预算和CPU4/batch2保持不变。

2026-10-14 03:47:53 UTC，`R3_GROUP_TEST_CONTEXT_VALIDATION.json`签收200/200正式test记录、0失败、3条独立排除预热、3,711个完整候选窗口，实际query-inclusive最大512。200个固定测试ID、row/config/source/model哈希、完整候选覆盖、windowed_max顺序、上下文预算及证据重算全部一致。真实候选范围3–50，199题少于50；保持完整scope，不补齐虚构候选。182题证据指标适用，完整参考集覆盖率0.659341、连续覆盖率0.697802、平均上下文1,913.395参考tokens。这些是冻结策略的测试描述结果，不能用于重新选型。合成离线延迟p50 12.472秒、p95 31.695秒，不能当作在线延迟。记录SHA为`a4366ef11e137a239da889eb0ef8528ec5990976d8f4dbfe8efdc1377e6cbfbc`，签收receipt SHA为`4d0ee2ba01788b0778ba182f67440ee25f24adf4685ea70995c0042071afdb7d`。验收exit0，日志`rerank_test200_integrity_acceptance_20261014.log`，已立即通知主执行器和generation执行器；`whole_campaign_complete=false`，原CPU worker继续后三组。

运行条件：硬截止下主执行器允许独立CPU4辅助索引预编码与R3并行；2026-10-14 03:21 UTC起又明确授权GPU串行执行既定22调用压力诊断，随后test200签收后GPU串行执行E2核心。generation执行器已报告22调用全部完成、0失败且2B正常卸载；R3没有参与这些GPU调用。R3始终是同一CPU4/batch2进程，资源guard未降低。各阶段存在CPU/GPU及主机资源竞争，因此这些真实耗时不是完全隔离环境下的硬件基准；不能将离线合成延迟当成受控在线实验。Laya仍需后续明确资源交接，未在此并行窗口加载。

首次进程32476在用户pause/resume与助手turn中断期间消失，session55504不可用；资源曲线最后停在加载阶段、RSS约248MiB，正式0条、空日志。代理没有执行停止动作，不能称其正常合作退出，也不能据此断言退出原因。原worker、LOAD_STATE、资源曲线、日志与暂停观察完整保存在`R3_RUNTIME/history/interrupted_pid32476_attempt1/`。恢复前再次确认没有R3进程且FileLock实际可获得/释放。按用户resume用同一入口接续为session83335、PID30108/create1791944791.4540648；attempts从1继续到2，backend_failures仍0，没有清零预算、删除数据或更换配置。准确后续状态以运行账本与日志为准。

2026-10-14 04:15:58 UTC，PID30108在E8第29条完整提交后触发`R3 RAM reserve after committed query`并exit1；最后主机可用1,222,057,984 bytes，低于既定1.25GiB。当前329条均已复核完整换行、真实记录与config hash及唯一ID，0条模型失败；这是资源等待，不是完整实验完成或模型后端失败。精确PID/创建时间已不存在，主writer.lock实际可获得和释放，模型进程退出。`LOAD_STATE`原样为attempts2/backend_failures0，已连同worker、资源曲线、原日志归档到`R3_RUNTIME/history/resource_pause_pid30108_attempt2/`。receipt `R3_RUNTIME/RESOURCE_PAUSE_PID30108.json` SHA为`8b6e9e388cab318355eed66315a3889c91e8ae6cf62b3d6559cd23bbb8b1d4b5`。未降低guard或反复重载；恢复需主执行器交回资源及fresh可用RAM至少3GiB，使用原命令从E8剩余71条接续，再执行ROUTE200/60。已封存TEST200来源和验收不变。

最新设备变更覆盖上述CPU接续安排：用户明确授权任何阶段使用GPU，仍单卡串行。`R3_GPU_MIGRATION.json`（SHA `697ce7ac3e0c213072359f03232a3b81d5ca4d8e33f0b4907f7ce58af77bbc51`）冻结并保护原CPU300和global29、原加载计数、完整源/选择/代码哈希及GPU100+200+60题目。新三组run_id和路径均追加`_gpu`；不改CPU TEST200及E2所引SHA。最终已执行总数将为CPU329+GPU360=689（另计排除预热），对应660个计划条件及明示额外29个历史CPU partial调用。只有完整GPU100用于global统计，不能将CPU29接到GPU71形成混合条件。

GPU使用现有train环境Torch2.9.1+cu128、Transformers5.19.0；必要的应用导入仅新增rank-bm250.2.2，原46包版本全部不变，pip check通过，旧LoRA环境manifest/lock原样保留，另存r3_gpu环境清单；固定BGE权重与revision、windowed_max、max_pair512、overlap32、batch2及CPU4预处理完全相同。该环境没有sentence-transformers，因此本地加载Transformers单logit分类器，显式sigmoid匹配已审计CPU CrossEncoder默认激活；FP32、eager、TF32关闭、autocast关闭。GPU与CPU后端版本差异逐项入身份，不声称数值或时间完全等价。复用同一窗口与排序/打包/证据计算逻辑，保留候选不足50的真实N。

运行入口为`train/Scripts/python.exe -u scripts/run_rerank_gpu_v2.py --home D:/RAGScope_Runtime --campaign 20261014_v2 --run`，先执行`runtime_env.ps1 -Gpu`。current-boot租约、全局`gpu.lock`和retrieval主写锁必须同时满足；原generation精确服务进程实际退出、无驻留compute PID、fresh主机和显存各至少3GiB才可加载，逐题主机1.25GiB保护保留。只操作自有模型资源，逐题记录实际CUDA分配/保留/峰值/空闲/总显存；退出释放模型并由外部再次核验PID与锁。GPU加载状态从CPU attempts2/backend_failures0继承但单独落盘，不改原收据；真正限制为最多3次backend_failures，成功加载attempts不是失败上限，MemoryError资源等待不计后端失败。

48项离线契约测试PASS（日志`rerank_gpu_contract_tests_verified_20261014.log`），涵盖CPU不可变产物/源变化拒绝、GPU完整固定人群及独立路径、设备服务互斥、batch2/sigmoid/no-truncation、显存证据和继承失败预算。无Torch的实际prepare返回`READY_NOT_EXECUTED`、model_loaded=false、torch_imported=false。真实GPU结果仍待资源交接，不能将这些工程测试写成GPU实验成功。最终`core/Scripts/python.exe scripts/validate_r3_v2.py --gpu`将独立验收360条完整窗口、模型/迁移/来源哈希、上下文与指标；生成`R3_GPU_COMPLETION_VALIDATION.json`，保留CPU验收独立语义。

WDDM实际列出的Codex PID11276是on-device utility而非已证实显示进程。仅其精确PID/create1791922058.5791745、已观察WindowsApps exe及utility/subtype/sandbox参数匹配，且fresh设备总used≤64MiB、free≥3GiB时允许；其余compute PID一律拒绝，包括自有llama-server。不会停止桌面应用。来源观察保存`R3_GPU_RUNTIME/CODEX_UTILITY_SOURCE_OBSERVATION.json`；原READY收据按SHA归档在同目录history/下，新READY重新绑定源码hash。实际tokenizer两种pair输入的IDs一致；TF5即使USE_TORCH=0仍会导入Torch，因此原“无Torch”断言失败日志保留，不能把tokenizer探针称为无Torch。该探针CUDA_VISIBLE_DEVICES为空、无权重加载；无Torch的READY prepare单独实际通过。

2026-10-14 04:55:53 UTC，GPU360独立验收PASS/0失败：global100有7,850窗口，ROUTE train200有4,301窗口，dev60有1,071窗口；全部13,222窗口完整覆盖原候选且query-inclusive≤512，9条排除预热完整。复算身份、源/迁移hash、排序、2048预算与指标一致。`R3_GPU_COMPLETION_VALIDATION.json` SHA `b10cb5b52bb54606eada1fccfbedcb0a302c22a4bfdd6acd0c5f77563fbdef89`；所有CPU300+partial29保护hash仍一致。session15599 exit0，精确PID22180/create1791953187.8157816已不存在，三锁重取释放，GPU回到12MiB，release receipt SHA `2249a3f97fc28f1bd09af797279d5887ea8c66ce1aa97bfc43d88d05d2e573c1`。加载attempts3/backend_failures0，实际CUDA参数全部FP32，最大观测allocated1,190,839,296B、reserved1,275,068,416B。

GPU global100的完整参考集覆盖率0.14、连续覆盖率0.148333、doc_recall@10=0.42；ROUTE train在171适用题中完整参考集覆盖率0.567251，dev在56适用题中0.625。这些是已冻结策略的描述性检索结果，不能声称生成质量提升或GPU/CPU等价。重排部分实际中位耗时分别为global2.558s、train0.672s、dev0.597s，尚不是在线端到端延迟。主执行器负责把完整GPU来源接入main freezer，不能引用旧CPU global29作为完整100。
