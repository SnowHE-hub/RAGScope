# RAGScope v2 执行状态

更新：2026-10-14 04:36 UTC / 15:36 Sydney。Goal **ACTIVE，未最终验收**；campaign `20261014_v2`。硬截止 **10:29:06 UTC / 21:29:06 Sydney**，主run/EXECUTION_DEADLINE.json。

## 最新授权和资源

用户明确“现在可以在任何阶段使用GPU，我没有任何进程跟你抢夺GPU资源”，覆盖任务书/AGENTS旧S1/S2 CPU-only限制。旧CPU结果保留原身份，新GPU执行明确设备/后端与可比性，仍单卡串行加载本项目模型。不操作原11434、日更、WSL、驱动和安全工具。

WSL内存上限不改，配置SHA不变，当前WSL未运行。原训练恢复 **USER_MANUAL_RESUME**：结束释放本项目资源并通知用户，不自动续原作业。当前boot租约SHA 884715ec4556624cd069b461b1b9cfc3483f0bab32f6f77c6cf2165f6d6b0cac；每次加载另核实际PID/create/端口/锁/RAM/GPU。用户允许必要时新版后端；当前继续可用专用Ollama0.34.4复用已有模型。

## 当前执行和队列

- E2冻结2400任务，manifest 6a2862e3fed696fb0224d23abc78437c5eaa318efb190c13b2422724ca3246e5。2B已800正式+3warmup，803 token一致/0error、AUDIT_qwen_2b.json。4B session36734继续完整800，自有服务3644/create1791948671.6012218归gen。9B尚0；4B完整审计后正常退出11435/释放gpu.lock，暂不加载9B。不要重复启动或poll其他agent句柄。
- R3 CPU worker30108在04:15:58UTC写完global29后真实RAM guard/exit1：dev100/test200/global29共329行0error完整保留，CPU TEST200/3711窗口PASS并已用于E2，绝不改写。RESOURCE_PAUSE_PID30108.json SHA 8b6e9e388cab318355eed66315a3889c91e8ae6cf62b3d6559cd23bbb8b1d4b5，PID absent/主writer释放。LOAD_STATE attempts2/backend_failures0，不能把成功加载次数当backend失败上限。
- GPU R3工程40 focused tests PASS，仅prepare无模型，READY_NOT_EXECUTED。同固定BGE-base权重/512窗口/overlap32/batch2/FP32/TF32关闭，直接TF CUDA后端，后端差异明示。独立E8完整100与ROUTE train200/dev60 GPU路径，旧300+partial29原样保留。R3_GPU_MIGRATION.json SHA db93e97ad60cfa30926a73615c19026befa8eec315e505b8abf3a416e07fa841。待4B服务和SQuAD窗口退出后执行，不能称已跑。
- simple实际全部完成：BM25/Dense/Hybrid各200，600行0error。新增Dense/Hybrid400正式+6预热真实缓存命中，HTTP/socket尝试0。三条历史缺usage只允许精确三cache key，原样null、不伪造token；原manifest/vector字节不变。worker23220/session90177 exit0/身份absent/锁释放/11436关闭。HISTORICAL_NA_REUSE/RESULT SHA 3d98ccdb2b8fee111b56cd97ff985b77615b324351b92c1a8c2b49ffe5f1448f；来源在主records，可供E6。
- SQuAD1212chunks真实预编码/readback PASS，最终Hybrid100查询尚未运行。4B退出后CPU4短窗口完成100及正常退出，再释放主writer给GPU R3。
- 当前单卡队列：4B退出 → SQuAD100 CPU短窗口 → GPU R3完整360与验收释放 → gen9B E2 → E2scope验收 → judge200 → 同BF16 LoRA paired200。main冻结接入完整GPU ROUTE来源并引用E2/E3，不重复核心；其他E4/E5/E6/E7、路由标签/七策略/在线仍须真实完成。

## 已完成且可追溯

- structured全6551向量/509论文index、TEST/global900、BM25/Hybrid ROUTE520全部0error。SQuAD1212+simple777共1989aux向量磁盘readback PASS，AUX_INDEX_HANDOFF SHA de498e5a3980bc42ad56c3369a786b8b625a209c2fc9c0c1309c1cbbc2239113。
- E1 dev：11切块×3检索×100=3300，另8扩展×100=800；按dev选择F512_O0/R512，均neighbor。DEV_RESULTS真实41条件4100/0error，负向指标保留，不声称全部提升。
- E3：三模型Oracle/NoCtx×200=1200+9warmup，0error/token mismatch，30报告数字独立复算PASS。Oracle可答F1 2B .5785604、4B .5455298、9B .4704581，不声称规模单调收益。原不可变分片以后引用。
- E0：固定官方CPU重建22254向量、400检索/文章聚类CI/10数核验；旧8索引不变、旧历史权重UNKNOWN，不称精确旧模型复现。
- LoRA：2B BF16/seed42/972train/122步/1epoch/59dev、best122 loss .1356631；5→10→122 optimizer/scheduler/RNG恢复已实测。paired100×2未运行，loss下降不代表QA收益。
- 压力4K/8K各8正式+3预热，22真实/0error/tickets1422–1443，独立raw审核PASS；资源PNG/SVG/data与目视QA完成。共享CPU背景、设备总VRAM/TTFT N/A如实写。
- 离线core584PASS/2skip、models24PASS，去重585不同PASS/23重叠，唯一remaining WinError1314 symlink能力skip。194源hash前后一致；新GPU/UI/路由代码须最终回归，旧receipt不覆盖新源。
- UI candidate四models/full509 index metadata核验及13tests PASS，候选hash73ef6704d26afeba4eb961057ad0a86e2d71d43c0bd21b7e2116746f78e5ff81，ready=false真实保留。最终发布/live generation/UI QA尚待。
- train/dev loader只打开ROUTE_TRAIN/DEV，不打开test journals；50定向tests PASS。历史操作者工具曾显示摘要质量字段，未据此调参，不声称从未见test字段。
- final report builder/资源图/错误子报告/学习文档已有局部产物；E3预览不是最终报告，部分manifest写FINAL_REPORT的guard已实测拒绝。

## 尚须完成与验收

E2剩余4B/9B；E4 300/E5 200/E6 400/E7最多240；E8 GPU R3完整100；ROUTE780真实标签、七策略冻结/重放/在线30×2/Laya能力；judge200/LoRA200；最终UI、主报告/聚类CI/图表/错误分析、最终学习说明、15条ACCEPTANCE、FINAL_REPORT随机至少10数反查、轻量包、存储和原件保护复核、全部项目GPU释放。最近runner reconcile9PASS/1RUNNING/15PENDING/exit2，不是最终15条验收；预算读当前budget文件，完成数按逐行数据核。核心范围和失败分母不缩减，阶段不等待。

## 持久恢复

原目录/分支 `codex/ragscope-v2-autonomous-20261014`；MASTER_PLAN SHA157040e71eab86b44bb3007bf43c1ba25240403b892a7cd5f93c1fb37e280117，不改原书掩盖用户覆盖。整理前STATUS逐字节归档到主run/status_history/STATUS_before_compact_20261014T0434206158306Z.md，SHA b8bf76830fe9a3cec611c064514a4545aa1916fab5a040219773374fd83b5536。所有大件D:/RAGScope_Runtime，只stage本任务文件，不push。root无常驻exec句柄。

下一条准确只读命令：

```powershell
Get-Content -LiteralPath D:/RAGScope_Runtime/runs/20261014_v2_generation_e2/EXECUTION_CONDITIONS.json -Raw
Get-Content -LiteralPath D:/RAGScope_Runtime/runs/20261014_v2_retrieval/R3_GPU_MIGRATION.json -Raw
```

详细身份/命令/收据见ROOT_CHECKPOINT、GENERATION_HANDOVER、R3_MATERIALIZATION、DATA_RETRIEVAL_CHECKPOINT、RECOVERY。
