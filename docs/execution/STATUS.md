# RAGScope v2 执行状态

更新：2026-10-14 13:07 Australia/Sydney。campaign_id：`20261014_v2`；主目标 **ACTIVE，尚未最终验收**。下面是当前快照，历史过程保存在运行记录和同目录恢复文档。

## 当前正在运行

- data_preparation独占CPU4 embedding服务6016/create1791942102.6736171、worker3312/create1791942111.7390096/session37283。最近02:05:45UTC观测为 **6416/6551** 条持久编码观察，完整index manifest尚未发布；同模型/二进制哈希不变。
- 随后完成structured的TEST/global及ROUTE结果，正常退出并核验释放自身embedding，再交接R3五组660条。chunking_core已做轻量入口核验，没有提前加载模型。
- 后续顺序：R3退出 → 2B的22调用4K/8K诊断 → GPU服务退出 → Laya有界尝试 → SQuAD与simple索引 → 主生成矩阵、路由、LoRA同精度对照及最终交付。
- 目前没有本项目GPU模型驻留。02:04:42UTC资源核验时WSL运行列表为空，11435/18501关闭，GPU compute-apps仅列Codex桌面；没有操作原任务。内存有波动，02:05:45UTC可用约2.00GiB；不能把这个快照当永久值。

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

E2 **0/2400**；E4 300、E5 200、E6最多400、E7最多240；E8全库4种检索；ROUTE780真实生成标签、七策略决策/重放、在线30×2；Laya实际能力；LoRA200条同BF16对照；支持度proxy最多200、最终UI；主报告/图表/错误分析、学习材料最终更新、15条ACCEPTANCE、从FINAL_REPORT随机至少10数回查、轻量结果包与最终资源释放。

统一生成预算目前 **1421/10000**，其中正式核心1200；E2的2400次仍受保护。25项任务账本当前9 PASS/16 PENDING，**不是第11节15条最终验收**。此前给用户的剩余8–16小时仅为粗估；资源反复中断可能超过24小时，不是完成承诺。

## 项目与下一条命令

分支：`codex/ragscope-v2-autonomous-20261014`。主任务书副本SHA-256：`157040e71eab86b44bb3007bf43c1ba25240403b892a7cd5f93c1fb37e280117`。原tracked源码、diff、HEAD及用户未跟踪清单保存在 `D:/RAGScope_Runtime/backups/20261014_baseline`。只stage本任务文件，不push。

最新root文档提交9325431；历史里程碑见 `docs/learning/CHANGELOG_FOR_LEARNING.md`。详细恢复点见 `ROOT_CHECKPOINT.md`、`DATA_RETRIEVAL_CHECKPOINT.md`、`GENERATION_HANDOVER.md`、`RECOVERY.md`。本次整理前STATUS原文逐字节归档至主run/`status_history/STATUS_before_compact_20261014T020704934468Z.md`，SHA ebfa1ea96b1d66d168e1966682f6f8c421aec862c37120679513847ac38eff81。

下一条准确命令只读；不要重复启动data拥有的worker：

```powershell
Get-Content -LiteralPath D:/RAGScope_Runtime/runs/20261014_v2_retrieval/worker.json -Raw
```
