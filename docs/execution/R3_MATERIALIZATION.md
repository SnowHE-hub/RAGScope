# R3正式重排执行记录

2026-10-14。当前状态：实现与离线功能测试完成，**正式真实模型materialization尚未运行**。11组完整dev矩阵已冻结structured=R512、simple=F512_O0，两者扩展均为neighbor。selected dev100已就绪；all509索引与test/global/ROUTE来源仍在上游CPU物化，等待其正常退出、释放服务后交接。没有加载BGE-base/Laya权重。

入口`python scripts/run_rerank_v2.py`只检查冻结输入；主执行器交回RAM后在models CPU环境加`--run`。可选`--prefix`仅指定冻结任务，不改变人群。全量五组为selected dev100、test200、E8 global100、ROUTE train200/dev60。train/dev路由ID直接来自主执行器的`ROUTING_QUERY_MANIFEST.json`，检查其manifest_hash和corpus来源，不重新采样。

`ragscope/experiments/rerank_campaign.py`只接收真实、无失败、哈希完整的Hybrid原始记录与对应index identity。恢复每一候选的完整canonical文本，逐项比较doc_id、坐标和unit_spans。必须有50条真实候选；如果指定论文中总块数少于50，使用全部可用块并显式记录shortfall，不重复填充或将旧top10冒充50。按已预检的BGE windowed_max对完整query+片段评分，返回前10，沿用最终dev冻结扩展方式和2048参考token上下文预算。

输出保留`raw_retrieval`（Hybrid原始50）、`selected_retrieval`（重排10）、每个候选所有窗口的字符区间/query-inclusive token数/实际分数、输入与配置hash、上下文与证据指标、失败、实际RSS与耗时。指标以重排选中的前10和实际打包上下文计算。源检索与新重排分时执行，合成耗时明确不能冒称在线端到端延迟。每组另存3个真实预热，排除在正式耗时摘要外。

正式进程另在`R3_RUNTIME/worker.json`写入实际PID/创建时间/argv/cwd，在加载模型前启动1秒周期RSS及主机可用RAM采样；完成或异常退出时停止采样并标记EXITING。外部验收必须再次确认实际进程已退出，不能将该标记单独当作资源已释放。

`R3_RUNTIME/LOAD_STATE.json`单独保存加载尝试及后端失败，最多3次；RAM暂停单列PENDING_RESOURCE，不消耗后端失败次数。该守卫与观察器退出回归共16项PASS（3.13秒），日志`rerank_load_retry_bound_tests.log`。2026-10-13 18:01:28 UTC（Sydney次日）上游实测仍有5,240个未缓存文本、2,223,347正文tokens，主机可用RAM约1.97GB；仅按最近吞吐估算约1.69小时编码，随后还需查询物化，不据此承诺资源交接时间。

公共加载器`ragscope/retrieval/verified_reranker.py`被R3和在线路由共同使用：仅固定本地BGE revision `2cfc18c9415c912f9d8155881c133215df768a70`，逐文件SHA校验；CPU-only Torch、四线程、max_length512、overlap32、batch2。没有自动下载、启动服务、GPU切换或原用户任务控制。加载前至少3GiB可用RAM，逐题落盘后留1.25GiB；资源暂停保留记录且不消耗后端失败重试预算。

正式输出命名与generation freezer一致，位于retrieval campaign的records/：`TEST_CONTEXT_<group>_hybrid_rerank_<expansion>.jsonl`、`ROUTE_CONTEXT_{TRAIN|DEV}_...`、`E1_...`与`E8_...`。运行持有同campaign单写锁，防止与数据agent并发改账本。

初轮与E0合计17项离线测试通过，3.40秒；追加冻结ID回归后R3专项12项通过，3.66秒，日志`D:/RAGScope_Runtime/logs/rerank_materialization_tests_final.log`。夹具均标offline_fixture/real_model=false，正式RecordStore拒绝它们。覆盖50候选完整性、短scope、坐标/ID/排序/NaN拒绝、global不使用gold过滤、候选第50移入前10后指标正确、失败保留、source hash与配置绑定、manifest修改拒绝、直接复用主执行器ID顺序。真实质量仍待后续实际结果。

追加本地模型manifest路径及禁止设备的前置测试后，与统一runner共37项通过，9.03秒，日志`D:/RAGScope_Runtime/logs/rerank_runner_integration_tests.log`。无模型的readiness CLI实际返回非零并明确指出最终CANDIDATE_SELECTION缺失。正式任一失败不会被摘要丢弃，连续3次失败停止并保留剩余待执行人群，避免在明显失效的后端上无限继续。

最终输入已冻结后的无模型检查：`E1_R512_hybrid_neighbor.jsonl`100题来源与406块dev索引的完整canonical候选恢复全部PASS。单论文scope的真实候选数为5–28，平均11.57，100题均少于50；这不是丢候选，全部可用scoped块均经ID/坐标验证。证据`R3_DEV_CANDIDATE_VALIDATION.json`逐题记录可用块和实际候选数，不能将此范围实验描述为每题严格50→10。其余test/global/ROUTE来源尚待上游完整交接。进程观察器加入后R3专项15 PASS（3.53秒），日志`rerank_worker_observation_tests.log`。
