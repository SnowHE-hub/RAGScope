# RAGScope v2 执行状态

更新：2026-10-14，用户在短暂停后明确执行`/goal resume`，Goal工具已确认 **ACTIVE，尚未最终验收**。campaign_id：`20261014_v2`。R3执行器先核验暂停/进程及已落盘断点，再接续原固定实验，禁止重复启动。历史过程保存在运行记录和同目录恢复文档。

用户新增8小时硬截止：**2026-10-14 21:29:06 Australia/Sydney / 10:29:06UTC**，主run/EXECUTION_DEADLINE.json。原实验范围、CPU4/CPU-only与真实结果标准不缩减。R3当前恢复session83335、worker30108/create1791944791.4540648，已真实加载、前25正式题0error，均值17.18秒；旧32476中断归档且attempts保留为2。独立aux索引预编码与E2先行分片工程正在落实，以减少串行等待；执行身份与实际启动须看各模块凭据，不把工程准备当已运行。

## 当前正在运行

- **6551/6551**全库编码及完整index manifest已完成；structured TEST/global共900条、ROUTE train200/dev60的BM25/Hybrid共520条，均完整ID/hash核验、0error、两个命令真实exit0。
- data精确排空并退出CPU embedding服务6016；3312/10584/6016均不存在，11436关闭。root独立核验释放收据SHA及进程/端口，host可用6.20GiB后，已正式交接chunking_core执行固定R3五组660条、CPU4。其新运行身份以模块receipt为准，不能沿用旧PID。收据为retrieval/STRUCTURED_ROUTING_RESOURCE_HANDOFF.json，SHA d9dca64ec3f03cbcad80f7e9d9dadf623fc4bdad2fac4ebb83ff479adf2cbe08。
- 当前并行依赖：R3按原顺序执行；独立aux索引-only worker30192/session99141与CPU4服务33428在SQuAD/scoped-simple目标锁内预编码，不写主retrieval campaign。SQuAD实际1212chunks/204332正文tokens，已有328+真实向量；完成后做磁盘readback并释放服务。R3 test200封存后冻结原2400 E2分片，在RAM/租约核验后单卡串行stress22→E2，与余下CPU R3重叠。R3释放主writer锁后物化aux最终检索，再冻结main并引用E2+E3；Laya、路由/LoRA对照和终验继续按实际资源窗口执行。
- E2分片工程提交bb2c874，54项CPU契约PASS，R0/R1/R2各200来源READY，R3等待中；本轮尚无E2生成，统一预算仍1421/10000。源码入口scripts/freeze_e2_v2.py，命令与条件见E2_CORE_SHARD.md。工程支持不代表实际2400条已完成。
- 03:13UTC最新进度：R3 dev100全量独立验收通过，test54/200、累计154/660；SQuAD预编码1212/1212完成，simple03:10已72/777。两个原执行器继续运行，RAM约3.22GiB；实际E4/E6检索/生成仍待物化，不能从向量数量推导QA实验完成。
- 03:21:05UTC独立即时核验：R3/aux worker/aux服务的PID、创建时间及完整argv均匹配，host可用4,142,514,176B（约3.86GiB），GPU12MiB/0%，WSL运行列表为空，11435/18501无监听，当前boot租约CLI通过。证据主run/PARALLEL_GPU_PREFLIGHT_20261014T032105Z.json。root已授权gen在重新核资源后先执行既定stress22，与CPU任务并行；实际启动、驻留和完成以gen新生命周期凭据为准，不把交接授权当完成。
- gen已实测系统Ollama漂移至0.40.3，空服务0请求即正常退出。官方专用portable0.34.4下载/82文件hash及原exe SHA一致性通过，服务3644/create1791948671.6012218，原11434未动。4K压力诊断实际8/8正式+3预热、0失败，卸载API ps空/GPU12MiB后RAM4.042GB再次guard通过，8K正在执行；最终22条全部完成仍待凭据。
- stress22现已全部封存：两组各8正式+3预热，tickets1422–1443连续唯一，全部real/cacheoff/token一致、0失败。4K/8K正式均值1.366/1.920s，P95 1.648/2.319s；设备总显存采样max1,517,289,472/1,628,438,528B。COMPLETION_RECEIPT.json绑定全来源。预算1443/10000、核心1200；空服务3644继续等待E2，不驻留模型。用户已允许必要时用新版且模型复用；现继续可用旧版，不再次切换。
- root新增scripts/build_final_report_v2.py，仅消费真实hash来源、保留missing/失败及作用范围，拒绝部分清单写整项目FINAL_REPORT。已从原E3封存1200行实际生成两个不可变预览，0模型调用；预览位于主run/recovery/report_builder_preview_20261014，**不是最终报告或最终验收**。
- 新全套离线回归core584 PASS/2明确skip、models routing24 PASS，去重585不同测试PASS（23重叠），唯一剩余WinError1314测试软链接能力。194源hash前后吻合，FULL_OFFLINE_REGRESSION_20261014_1444.json。后续新增或改变源码要另核，不把这项功能测试算真实质量。

## 已完成并有真实证据

- E1：11组切块×3种检索×100dev，共3300条；另有8组上下文扩展×100dev。只按dev选择F512_O0与R512，两者neighbor。结果、图和负向指标见 `reports/v2/retrieval_interim/DEV_RESULTS.md`。
- E3：3模型×200题×Oracle/No Context，**1200/1200，0请求错误**，另有9次预热。全部实际token计数一致，30个报告数字独立复算通过；原分片封存，主矩阵以后引用而不重跑。Oracle可答F1为2B 0.5785604、4B 0.5455298、9B 0.4704581，没有规模单调收益。No Context的9B全部拒答。
- E0：固定官方版本CPU重建22254个向量，Dense/Hybrid共400条；原8索引哈希不变。旧历史权重UNKNOWN，不能声称精确复现旧模型。独立10数核验和文章聚类区间已完成。
- LoRA：2B BF16正式seed42、972训练例、122步/1epoch、59dev；5→10→122完整状态恢复已实测，best122 dev loss 0.1356631。**100×2问答对照尚未运行**，不把loss下降写成质量改善。
- 已有完整离线回归去重后528个不同测试通过，Windows symlink权限一项skip；后续simple、CPU恢复与GPU租约有各自定向验证，数字不能相加冒充去重总量，最终回归仍须重跑。
- UI已做实际BM25查询、证据展开与E3回放；最终神经索引、真实实时生成及路由默认部署尚待验证。
- E3错误子报告、资源报告、dev检索图、E0及LoRA图均已保存；人工审核字段仍是NOT_REVIEWED。代码支持与局部报告不代替终验。

## 恢复与资源政策

电脑07:03启动过，原因UNKNOWN，本任务没有发出重启命令。旧20816/19112已不存在；5976条完整观察保留，当前worker已继续新增。重启后按最新AGENTS用CPU4线程，旧CPU12性能记录保留；三个历史网络失败和上限未清零。入口和证据见 `HOST_RESTART_REENTRY.json`、`DATA_RETRIEVAL_CHECKPOINT.md`。

GPU租约跨启动周期的缺口已修复，50项CPU定向测试通过；旧真实租约经CLI/PowerShell拒绝。原40961字节租约原样归档，新凭据绑定当前win32启动时间并实际校验通过，收据为主run/`GPU_LEASE_REVALIDATION_20261014.json`。没有因此加载模型或修改旧实验。后续加载仍须资源交接与检查。

用户最新要求 **不改WSL内存上限**。配置未修改，memory=12GB/swap=8GB/processors=12，原SHA保持一致。容量上限不等于当前常驻占用。原训练恢复政策为 **USER_MANUAL_RESUME**：最终只释放本项目资源并通知用户；不自动续原作业，不操作日更。

S1/S2实际计算保持CPU-only，但S3/S4与尚未结束的S2曾有时间重叠，最终criterion14必须披露，不能称为严格阶段串行。所有新模型/数据/环境/缓存/日志在D:/RAGScope_Runtime；最终磁盘与原件核验尚待执行。

## 仍须实际完成

E2 **0/2400**；E4 300、E5 200、E6最多400、E7最多240；E8全库4种检索；ROUTE780真实生成标签、七策略决策/重放、在线30×2；Laya实际能力；LoRA200条同BF16对照；支持度proxy最多200、最终UI；主报告/图表/错误分析、学习材料最终更新、15条ACCEPTANCE、从FINAL_REPORT随机至少10数回查、轻量结果包与最终资源释放。gen已经获得E2→judge→BF16连续资源执行授权，每段核原guard/身份，阶段结束不等待。

统一生成预算目前 **1443/10000**，其中正式核心1200；E2的2400次仍受保护。03:46UTC实际reconcile-only返回2，25项任务账本9 PASS/1 RUNNING/15 PENDING，**不是第11节15条最终验收**。未执行任务命令/未请求模型。此前8–16小时粗估仅为历史；当前按用户21:29硬截止执行。

## 项目与下一条命令

分支：`codex/ragscope-v2-autonomous-20261014`。主任务书副本SHA-256：`157040e71eab86b44bb3007bf43c1ba25240403b892a7cd5f93c1fb37e280117`。原tracked源码、diff、HEAD及用户未跟踪清单保存在 `D:/RAGScope_Runtime/backups/20261014_baseline`。只stage本任务文件，不push。

最新root调度提交8f84bd6；历史里程碑见 `docs/learning/CHANGELOG_FOR_LEARNING.md`。详细恢复点见 `ROOT_CHECKPOINT.md`、`DATA_RETRIEVAL_CHECKPOINT.md`、`GENERATION_HANDOVER.md`、`RECOVERY.md`。本次整理前STATUS原文逐字节归档至主run/`status_history/STATUS_before_compact_20261014T020704934468Z.md`，SHA ebfa1ea96b1d66d168e1966682f6f8c421aec862c37120679513847ac38eff81。

下一条准确命令只读；不要重复启动data拥有的worker：

```powershell
Get-Content -LiteralPath D:/RAGScope_Runtime/runs/20261014_v2_retrieval/R3_RUNTIME/worker.json -Raw
```
