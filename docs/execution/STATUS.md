# RAGScope v2 执行状态

更新：2026-10-14 05:48 UTC / 16:48 Sydney。Goal **ACTIVE，未最终验收**；campaign `20261014_v2`。硬截止 **10:29:06 UTC / 21:29:06 Sydney**，主run/EXECUTION_DEADLINE.json。

**05:58UTC资源/修复覆盖：** root实际浏览器BM25四片/证据展开/资源提示与截图已保存，BROWSER_PHASE1_QA SHA8b332...16855；自身18501已正常请求停止/Windows实际exit1保留、26204gone/port可bind，UI不再常驻。chunk修复replay最后来源凭据误找本地E2路径的问题（原失败完整留档），25定向PASS，216旧源215同SHA/仅campaign.py变更/新增7覆盖测试，labels和七冻结决策均不变。chunk等fresh≥3GiB再replay，不降资源门槛；root不叠加重分析。data正在独立验收实证与学习材料核验，尚不写最终ACCEPTANCE/四正式报告。gen4B336/420（两条输出长度失败保留）后原模型队列连续执行。

**05:48UTC最新覆盖：** main2B1200+3warmups完整（1条length/parse_error保留），4B177/420正在运行（1条同类失败保留），gen继续原LFM/9B/judge/BF16队列。三种CPU路由已真实训练，七mandatory每种dev60/test100决策全部冻结（SHAf0e0ff...），chunk正在七策略replay/Laya报告，无模型。root唯一live session99187是CPU4本地Streamlit18501，IAB实际浏览器验收BM25/证据/资源状态，不请求正在被gen占锁的模型；最终在线60/真实UI生成/正式发布仍待。主分析等chunk轻量报告释放后运行，完整5520正式生成尚未全部完成。详细身份、失败分母、准确下一动作见ROOT_CHECKPOINT顶部，覆盖下方旧状态。

**05:29UTC最新：** main FROZEN5520/20READY，gen own11435 PID31596/session55731与main2B36036/session29105正在1200 local-owned，最近563/1200零error，约11–13min是当时速率外推；后续4B/LFM/E2 9B/main9B/judge/BF16已有连续授权。Laya1080+3warmups完整PASS/0error/646confidencefallback/142顺序变化，仅描述不当质量；11880实际exit0/锁释放，raw292a.../release0b3...。chunk idle等ROUTE780完整后同agentfollowup训练/七冻结/重放/Laya校准。root全离线core678PASS/2skip+models24PASS，去重679不同PASS/唯一symlink能力skip，216源hash前后同，FULL_OFFLINE_REGRESSION_20261014_0527.json；源稳定不重复全套。ROOT没有常驻exec，剩余期限约5h；详细livePID/create/准确下一动作见ROOT_CHECKPOINT顶部，覆盖下方旧“Laya在跑/主freeze中/585测试”快照。

**当前覆盖下方早期快照：** GPU360已独立PASS/0error/13222窗/max512/实际worker退出，CPU300+partial29原SHA不变；validation b10cb5b52bb54606eada1fccfbedcb0a302c22a4bfdd6acd0c5f77563fbdef89。Laya CPU4真实load75.774s/3devwarmups PASS，session76916/PID11880，1080正式现300，尚未全完。root main spec907282...9f66已配置，main freeze session73845/PID24476正在真实算统一预算，不重复启动；gen无服务/HOLD，FROZEN后fresh≥3GiB再执行主2B优先队列。S3.1接口55相关JUnit/8当前generation源核验和S2.4有效660真实R3组合已各签收，仍不是最终15验收。详细当前活进程/下一准确工具命令以ROOT_CHECKPOINT顶部为准；不要据下方历史“GPU正在运行”或“Laya等待”段重启模型。

**05:10UTC已正式FROZEN覆盖上句freeze进程：** root73845 exit0，main manifest ac56e65060b49d8d536fdaf1584727a1f0e736b4ab1266c1845c0c1d6383f07c，20 READY/5520tasks、无blocked/pending；E2/E3外部3600不重跑。ROOT_FROZEN_HANDOFF SHA00f60dffbfcbfa3f5d0e03951813df343b0f1bf4e4337d2442982cc94882a02f。gen已获即时正式GPU队列授权，freshguard通过后先main2B local1200产ROUTE780，其后4B420/LFM200/E2 9B800/main9B100/judge/BF16。实际新service/worker及质量结果以新receipt为准，FROZEN不是5520生成完成。Laya继续CPU4，最新540/1080；root无liveexec。后续仍须所有真实运行/15终验，不以输入冻结结束。

## 最新授权和资源

用户明确“现在可以在任何阶段使用GPU，我没有任何进程跟你抢夺GPU资源”，覆盖任务书/AGENTS旧S1/S2 CPU-only限制。旧CPU结果保留原身份，新GPU执行明确设备/后端与可比性，仍单卡串行加载本项目模型。不操作原11434、日更、WSL、驱动和安全工具。

WSL内存上限不改，配置SHA不变，当前WSL未运行。原训练恢复 **USER_MANUAL_RESUME**：结束释放本项目资源并通知用户，不自动续原作业。当前boot租约SHA 884715ec4556624cd069b461b1b9cfc3483f0bab32f6f77c6cf2165f6d6b0cac；每次加载另核实际PID/create/端口/锁/RAM/GPU。用户允许必要时新版后端；当前继续可用专用Ollama0.34.4复用已有模型。

## 当前执行和队列

- E2冻结2400任务，manifest 6a2862e3fed696fb0224d23abc78437c5eaa318efb190c13b2422724ca3246e5。2B/4B各800正式+3warmup，均803 token一致/0error、两份AUDIT PASS；累计1600/2400。4B已正常卸载及请求停止服务3644/create1791948671.6012218/API11435，Windows supervisor实际exit1保留，卸载/stop_reason=request证明意图释放，不能冒称0。RELEASE_4B_SERVICE_FOR_GPU_R3 SHA8aa39f2d3b707f85cd3cc0a4d7fec204c68c78d972f9776047615f20d67752f3。9B尚0/HOLD，预算3049/core2800；不要重复启动或poll其他agent句柄。
- R3 CPU worker30108在04:15:58UTC写完global29后真实RAM guard/exit1：dev100/test200/global29共329行0error完整保留，CPU TEST200/3711窗口PASS并已用于E2，绝不改写。RESOURCE_PAUSE_PID30108.json SHA 8b6e9e388cab318355eed66315a3889c91e8ae6cf62b3d6559cd23bbb8b1d4b5，PID absent/主writer释放。LOAD_STATE attempts2/backend_failures0，不能把成功加载次数当backend失败上限。
- GPU R3已真实加载运行：48 focused tests PASS，prepare旧receipt db93...归档，现迁移SHA697ce7ac3e0c213072359f03232a3b81d5ca4d8e33f0b4907f7ce58af77bbc51。session15599/PID22180/create1791953187.8157816、GPU加载33.73s、attempts3/backend_failures0，前global24/100零error。同BGE权重/512/overlap32/batch2/FP32/eager/TF32off，实际allocated1112257024B；CPU旧300+29不改。独立GPU E8全100+ROUTE200/60共360仍未全完。启动收据R3_GPU_RUNTIME/RESOURCE_HANDOFF_BEFORE_START SHAa91224c35ae1c54a3d30c5671b176fd1d7a3b58c3ecea6866268d95f4594a3de，host6045691904B/GPU12MiB/7888free。空GPU表仍列精确Codex on-device utility11276，不是display；仅PID/create/exe/type/subtype一致且used≤64MiB时允许，其他计算PID拒绝，不停止桌面。
- simple实际全部完成：BM25/Dense/Hybrid各200，600行0error。新增Dense/Hybrid400正式+6预热真实缓存命中，HTTP/socket尝试0。三条历史缺usage只允许精确三cache key，原样null、不伪造token；原manifest/vector字节不变。worker23220/session90177 exit0/身份absent/锁释放/11436关闭。HISTORICAL_NA_REUSE/RESULT SHA 3d98ccdb2b8fee111b56cd97ff985b77615b324351b92c1a8c2b49ffe5f1448f；来源在主records，可供E6。
- SQuAD Hybrid100已实际PASS/0error/session48604exit0；100冷query推理+3预热cache hit，全backend usage存在，1212正文chunks未重编码。33020/18392准确absent、11435/11436可bind、三锁复取，EXIT_INDEPENDENT_VALIDATION SHAe148257beb752393803918c23fcbd55b3c85947c0e5884949afcfc8c514d2c48。源records SHA b84f5dbec226213cdc84c7dbdc79796f70301fd7a193e7bebc4efc68293041b7，可供E4；生成仍未完成。
- 当前后续单卡队列：GPU360验收/释放 → Laya无其他项目模型的CPU加载窗口（normal≥5GiB/原唯一4.5–5监督备选；3dev预热后核是否可与GPUgen并行）→ main冻结GPU ROUTE source并引用已有E2/E3 → gen优先main2B local-owned产真实ROUTE标签，其后main4B/LFM及9B E2+main → E2/main验收/judge/BF16。主模型单卡串行，不重复外部E2/E3任务；七路由train/dev训练和决策可与后续生成并行，test回放在七策略冻结之后。

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
- GPU来源接入脚本/统计接线21项定向PASS4.51s；必须360独立完整验收/worker退出/CPU保护hash不变才写显式MAIN_SPEC，真实未完成来源已实际拒绝exit1且无spec。第一次测试命令误写不存在的文件名exit4/no tests，日志保留后正确路径通过，不能把首轮计PASS。S2.5 dev选型正式receipt签收4100行/41条件/123独立数字/164来源hash，SHA0eac2014ab2c41cf938768f026bdd2635e0f29f2b806cb5bb355e89440c7197e，不授最终验收。
- UI离线候选真实Streamlit AppTest启动/全509BM25 scope/neighbor4片2022参考tokens/dev历史回放PASS，零HTTP/模型/Torch；OFFLINE_STARTUP_QA SHA19cb41475bd19e2643bfa1ba94b4a4d3413c1a863819711bb95a869f498630d2。首次过宽socket隔离误拦Windows asyncio self-pipe失败保留；缩为禁止应用HTTP后一次PASS，不计模型失败。root最终浏览器/live GPU仍待执行。
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
