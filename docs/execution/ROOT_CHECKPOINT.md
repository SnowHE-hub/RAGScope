# 主执行器恢复点

**04:36UTC最新覆盖：** 用户明确任何阶段可用GPU/无原GPU竞争，覆盖旧CPU-only阶段规则，原书不改。4B session36734完整800后gen审计并正常退出服务3644/create1791948671.6012218/API11435/gpu.lock，不加载9B；下一窗口data CPU4完成SQuAD Hybrid100并退出，然后chunk执行GPU R3 E8全100+ROUTE200/60共360。同BGE固定weights/FP32/TF32off/512/overlap32/batch2，train TF5.19 CUDA后端与CPU ST后端差异明示，新路径不改CPU E2 test。GPU工程40tests PASS/prepare READY_NOT_EXECUTED，迁移receipt db93e97ad60cfa30926a73615c19026befa8eec315e505b8abf3a416e07fa841。CPU R3 30108已04:15:58 guard exit1，329成功0error/global29历史保留，PID absent/锁释放，attempts2/backend_failures0不能按成功加载次数误封禁。simple600全PASS，新增400正式+6预热真实缓存/0HTTP，三个历史usage N/A具精确旁证仍null，session90177 exit0/23220absent/主锁释放；RESULT SHA3d98ccdb2b8fee111b56cd97ff985b77615b324351b92c1a8c2b49ffe5f1448f。data无活进程等待正式短窗口，提交33b575a。GPU R3后交回gen9B→E2scope→judge200→BF16paired200，main须显式接完整GPU ROUTE映射/CPU E2不变。root当前无常驻exec，硬截止10:29:06UTC，尚约5h53。最新STATUS已整理，旧11166字节备份SHA b8bf76830fe9a3cec611c064514a4545aa1916fab5a040219773374fd83b5536。

下一准确只读命令：`Get-Content -LiteralPath D:/RAGScope_Runtime/runs/20261014_v2_retrieval/R3_GPU_MIGRATION.json -Raw`；查看gen进度通过文件/消息，不poll其工具句柄。GPU运行准确命令：`. ./scripts/runtime_env.ps1 -Gpu; & D:/RAGScope_Runtime/envs/train/Scripts/python.exe -u scripts/run_rerank_gpu_v2.py --home D:/RAGScope_Runtime --campaign 20261014_v2 --run`，只由chunk原owner在实际交接后执行；验收core `scripts/validate_r3_v2.py --home D:/RAGScope_Runtime --campaign 20261014_v2 --gpu`。无fresh交接不能重复启动。

**04:12UTC最新：** E2 2B已800正式+3预热、803实际token parity/0error、全部store/task/input/票据签收，AUDIT_qwen_2b；卸载后GPU12MiB/API空/RAM4.687GB guard，4B session36734已接续，首50generation2.447s、记录间隔2.457s，64时0error/mismatch。交接时预算2246/core2000；E2尚未2400全完。gen操作者一次input_hash自包含审计错误保留OPERATOR_AUDIT_ATTEMPT1后修核规则，记录未改；MODEL摘要曾工具输出质量字段，未据此调参/默认，不能声称从未接触test字段。data候选main/ui_candidate/ui.json hash73ef6704...ff81真实核四models/fullsparse+dense索引，关闭11436/生成锁occupied明示未就绪，13 UI测试；提交916de34，指定五学习文档b93484e；data现在idle无模型，后续用同原followup。root修analysis_report.validated_rows可选experiments边界，使train/dev loader只打开ROUTE_TRAIN/DEV而不触test journal；完整分析默认全量不变，负向test journal夹具+相关50 tests实际PASS38.20s（session18758 exit0，routing_train_read_boundary_tests.log/XML），0生成。ROOT/STATUS/DECISIONS已更新；这些源/新UI不属于之前194快照，需最后统一测试或必要增量签收。report builder已把三张真实dev图以同字节复制到预览figures并验证链接/24主表cells，预览v5非终验。下一准确只读仍E2 EXECUTION_CONDITIONS与R3_RUNTIME/worker真实psutil核身份。无root常驻exec句柄，CPU R3继续global/ROUTE，截止10:29:06UTC（约6h17）。

**03:55UTC当前运行与任务交接：** R3原30108/session83335已经TEST200独立PASS，records SHA a4366ef11e137a239da889eb0ef8528ec5990976d8f4dbfe8efdc1377e6cbfbc/config3d49a5a4ef3a5b5e259643eff5a8d04d5b6352b82798ce935ac948e1d7580631/3711完整窗口/max512/0error；继续原global100→ROUTE train200/dev60，第一global实际78窗口/0error。aux session99141 exit0、30192/33428准确不存在/11436关闭，1989vector全磁盘/cache readbackPASS，AUX_INDEX_HANDOFF SHA de498e5a3980bc42ad56c3369a786b8b625a209c2fc9c0c1309c1cbbc2239113。data等待主writer释放前，只做UIcandidate新入口及chat_app最小tokenizer字段bug修复/定向测试，不发布configs、不请求模型。gen E2实际freeze成功hash6a2862e3fed696fb0224d23abc78437c5eaa318efb190c13b2422724ca3246e5，2400/每模型800；2B worker session80850，service3644/session26765归gen，root不要poll。起跑guardhost5.002GB、GPU12MiB/API空、leasecurrentboot，pytest与aux均已退出/R3身份MATCH。首50平均1.256s、78时0error/token mismatch，后235/800；预计该2B余15–22min是当时真实速率外推。E2冻结曾短时host1.29GB，随后辅助服务退出/冻结完成回升4.73GB，两个CPU原guard保留、没有因此重试或停止。gen已有连续E2→judge200→11435退出→同E2-source pairedBF16授权。最新root22dc2f6、gen661a521/32fe2f4/b8a35b5；后端当前精确命令带--binary D:/RAGScope_Runtime/tools/ollama-0.34.4/ollama.exe。截止10:29:06UTC，约6h34剩余，原恢复USER_MANUAL_RESUME/WSL未改。

下一条准确只读命令：
```powershell
Get-Content -LiteralPath D:/RAGScope_Runtime/runs/20261014_v2_generation_e2/EXECUTION_CONDITIONS.json -Raw
Get-Content -LiteralPath D:/RAGScope_Runtime/runs/20261014_v2_retrieval/R3_RUNTIME/worker.json -Raw
```
之后核psutil PID/create/fullargv；进程结束/某模型summary不能推断整矩阵完成。TEST质量均值不用于改协议/路由超参，root最终test报告待mandatory路由决策全部冻结。root当前无exec常驻句柄。

**03:46UTCroot报表/资源图独立交付与实际账本：** 中文报表入口已从封存E3实际1200行生成v3预览（scope明确不是最终交付），24个现有表格数字与新从raw/source独立复算的30指标匹配；PREVIEW_REPORT_NUMBERS_QA.json，0新模型调用。部分E3清单试写FINAL_REPORT真实被guard拒绝exit1且无文件，不计第15最终验收。4K/8K真实资源图已生成PNG/SVG/data到reports/v2/figures/context_stress_resource.*并view_image目视核验，CONTEXT_STRESS_FIGURE_VISUAL_QA.json；首轮因reports env无filelock失败日志保留，修为只读JSON/hash后exit0，没有新增依赖或模型。此后build_final_report/plot_results两脚本source已改，194快照不能称覆盖更新后的两脚本；真实入口smoke与图像QA作为增量检查。rootreconcile-only session7675真实exit2，9PASS/1RUNNING/15PENDING，不执行实验命令。下一条准确只读命令仍为R3_RUNTIME/worker.json与aux/SERVICE_LIFETIME.json加psutil真实身份；root所有测试/对账句柄已结束，gen服务与两个CPU执行器归原owner。

**03:42UTC新增真实离线回归与GPU连续授权：** root core session1293 exit0，584 PASS/2明确skip，144.44s；models CPU session74918 exit0，routing24 PASS/20.73s。按JUnit具体ID去重585 PASS、23重叠，唯一remaining skip WinError1314软链接能力。194个Python源hash在开始/结束一致，收据FULL_OFFLINE_REGRESSION_20261014_1444.json；后续新增/改源码须自己的必要检查，不能套该旧快照。stress22完成/0error/tickets1422–1443，root另外从22原始行独立核record/config/hash/token/cache/tickets与算术，ROOT_INDEPENDENT_STRESS_AUDIT_20261014.json PASS。预算1443/core1200。gen拥有空11435/PID3644/session26765，等R3 TEST200签收；现已获得连续执行E2三模型→固定E2-source judge200/4B→卸载/服务退出→E2/R2绑定LoRA check-only→GPU独占同BF16 baseline/adapted100各自执行授权，无需阶段确认，所有原guard/失败/全局计数保留。BF16完释放交回root，不重训。用户允许必要时后端新版；现在旧版可用不再切换。data aux完成后独立准备runtime UI candidate（不直接发布configs），root最终GUI/GPU实测。两CPU任务继续，最后checkpoints准确命令仍为R3/aux receipts加真实psutil身份，不抢代理session。

**03:27UTCGPU后端漂移已发现，0调用，gen在有界恢复原固定版本：** root03:21创建PARALLEL_GPU_PREFLIGHT_20261014T032105Z.json：R3/aux三进程PID/create/fullargv MATCH，host4,142,514,176B约3.86GiB，GPU12MiB/0%，WSL空列表，11435/18501无监听；当前boot lease CLI PASS。同一gen原代理启动自有空服务后发现系统Ollama0.40.3，冻结身份要求0.34.4；未请求模型/预算未增。已授权其正常停止该空服务、核已有旧binary或有界下载官方Windows portable v0.34.4至D专用目录，不能改系统/原11434/冻结身份/混用新版GPU库。恢复后重新核有效3GiB及服务身份执行原22 stress，与CPU4 R3/aux并行；TEST200封存后直接E2。E2后judge200也可绑定E2来源但需root另交接；BF16仍另等GPU窗口。原任务USER_MANUAL_RESUME、WSL上限未改，截止10:29:06UTC。

**03:13UTC最新实测与下一条命令：** R3 E1 dev100全量独立验收PASS（1826完整窗口/3预热/源hash/排序/预算/证据），TEST54/200、累计154/660、未观察错误，30108/session83335同身份在跑，host3.22GiB。SQuAD独立预编码1212/1212完成，冷模型1212/0cache hit，编码1350.93s；index sig5bdf788ee95cf8e78f4ac8b4a51db38f768ca716657caa6f17694ffb178f4075、manifest SHA45cbd9c90f261841d8255bcfa6441e482e48f6975232d22f3ad8ccff71d4ea25。simple777/377551正文tokens、起始cache2/冷775，03:10已72/777，同aux session99141继续。实际检索质量结果尚未物化，不能把预编码完成当E4/E6完成。最新root提交8f84bd6。

root已直接审阅evaluate_lora_v2.py：只依赖训练best122与冻结E2/R2全200inputs，不依赖main其他组或任何QA输出。因此E2冻结后，可在核心GPU完成而main仍等CPU的空窗，用原CLI的`--campaign-dir D:/RAGScope_Runtime/runs/20261014_v2_generation_e2`运行原100×2同BF16对照；输入sort+seed42规则与未来main严格相同。冻结后的paired目录将永久绑定E2 manifest，不要后续再传main而触不可变输入hash冲突；这不是新实验组合。

下一条准确只读命令（receipt后仍要psutil核PID/create/argv，禁止凭旧RUNNING重启）：
```powershell
Get-Content -LiteralPath D:/RAGScope_Runtime/runs/20261014_v2_retrieval/R3_RUNTIME/worker.json -Raw
Get-Content -LiteralPath D:/RAGScope_Runtime/runs/20261014_v2_aux_preencode/SERVICE_LIFETIME.json -Raw
```

**当前实际双CPU运行与E2入口：** R3 session83335/PID30108已dev74/100、累计74/660、0error，CPU4/batch2，owner chunking_core，不能root轮询/抢重启。data已启动独立aux session99141/PID30192、CPU4 llama33428，身份/源hash在aux/SERVICE_LIFETIME.json；SQuAD1212chunks/204332 Qwen正文tokens，328+向量逐条落盘，之后simple777。aux从不打开主retrieval Campaign，使用独立ledger和squad_chunks/scoped_chunks锁；主writer锁仍由R3持有。host约2.04–2.25GiB，data有请求边界1GiB guard、R3保持1.25GiB，不能另加载模型。aux完成后data将独立磁盘load/readback留证并正常释放。

gen已提交bb2c874：freeze_e2_v2.py、E2生产验收、main自动引用E2+E3（包括旧显式spec）及54项CPU契约PASS；R0/R1/R2各200 READY，R3 TEST200未齐，0新生成。R3 test200封存通知后，先corepython scripts/freeze_e2_v2.py --campaign 20261014_v2 --check-only，再去check-only完整freeze，输出20261014_v2_generation_e2；实际GPU窗口由root核RAM≥3GiB/当前boot lease/原程序/端口后交接gen：先原stress22串行并卸载，再E2 qwen_2b→4b→9b各800+3warmups。写E2/EXECUTION_CONDITIONS.json记录CPU并行身份/线程及资源；不等global/ROUTE660全完成，不重复E3/E2。精确命令E2_CORE_SHARD.md。Gen当前completed需followup唤醒原代理，无需新代理。

**最新硬截止与并行调度：** 用户要求从现在起8h完成，已记录deadline2026-10-14T10:29:06.606796Z（悉尼21:29），EXECUTION_DEADLINE.json。Goal仍ACTIVE，无token预算。R3旧32476确实不存在、0正式行，未执行合作暂停，原因未知；旧证据在R3_RUNTIME/history/interrupted_pid32476_attempt1/。新session83335、worker30108/create1791944791.4540648、CPU4/batch2，LOAD_STATE attempts2/backend_failures0，真实加载68.24秒；前25题0error、473真实query-inclusive窗口，平均17.18秒/题，source/window验收PASS。不要重复启动或清零attempts。

为满足截止，不等所有CPU工作才启动生成：gen正在实现freeze_e2_v2.py（独立2400E2、同未来main task/input/model/prompt/预算，main引用E2+E3严格ownership，不重跑），等待四个TEST_CONTEXT200完整来源；mainrunner只在需要时读SQuAD。data正在实现/核验独立aux_preencode索引-only任务（SQuAD/scoped-simple777），不写R3持有锁的主campaign，保证原index身份/路径/缓存模型语义完全一致。R3先按原dev100→test200顺序，TEST全量就绪即发通知，余下global/ROUTE可与单卡串行E2重叠；所有RAMguard/CPU4保留。若锁隔离无法确认不双写。Laya保持未加载，待资源归还；旧串行交接顺序由本次硬截止调度覆盖，实验语义未改。

**用户已经恢复，覆盖下方PAUSED指令：** 用户随后明确`/goal resume`，Goal工具确认ACTIVE。root已通知R3原owner核验session55504/PID32476真实状态后继续同660条；若进程仍活跃则不得重复启动，若已安全退出则核已保存记录/锁后续原断点，暂停历史与失败预算保留。data仍等待R3→stress→Laya后归还CPU。目标继续，不创建新goal/代理，不改WSL。

**用户明确暂停，覆盖下方ACTIVE与继续执行指令：** `/goal pause`已应用，Goal工具返回PAUSED。root仅落实安全收尾，不继续实验；已通知chunking_core停止新增R3请求、当前记录落盘后释放模型/锁并保存恢复命令，通知data保持无worker/model状态。R3暂停前实际已启动session55504、worker32476/create1791944469.3008657（CPU4），新退出证据尚待owner报告，不凭暂停Goal推断后台进程已退出。不得启动stress、Laya、SQuAD/simple或主矩阵；恢复需用户`/goal resume`，届时先核当前进程及收据，禁止重复启动。

**当前最新交接，覆盖下方CPU编码运行快照：** structured900和ROUTE520均真实exit0/完整行ID与hash/0error；6551全库索引已发布。data精确drain后退出服务6016，worker3312/10584也不存在，11436关闭；正式收据retrieval/STRUCTURED_ROUTING_RESOURCE_HANDOFF.json SHA d9dca64ec3f03cbcad80f7e9d9dadf623fc4bdad2fac4ebb83ff479adf2cbe08。root已独立复核hash、PID/端口、RAM6.20GiB，并通过send_message正式授权已有chunking_core执行固定R3五组660（CPU4、BGE，原资源guard/失败预算）；尚待其报告新PID与session。root/data不要再启动embedding、simple/SQuAD。R3退出→root唤醒gen22 stress→退出→Laya→data恢复CPU4的SQuAD/simple。用户只咨询模型切换，未请求暂停Goal，目标仍ACTIVE。

**13:12模型切换前持久恢复点：** 用户咨询能否在本对话暂停后切换GPT-6.1 Sol，尚未明确发出Goal暂停指令，root没有自行暂停/完成目标。02:12:12UTC独立只读确认structured观察 **6551/6551，index manifest已经发布**，host1.13GiB；data仍拥有worker3312/session37283，须查实际状态后继续TEST/global/ROUTE，不能重复启动。13:10曾观察可用RAM0.30GiB，后回升；当时Windows ollama app.exe PID20072/create1791922078.7872417 RSS3.823GiB，是非本项目原桌面应用，未操作。data短间隔身份与资源观察已落盘，当前没有第四次网络重试。两个其他原代理已完成轻量预检/提交，R3与stress均未启动。当前重要提交9325431、eedb5d9（boot租约50测试）、4fe1c07（精简STATUS/真实新lease）。

**13:04新增签收：** 新boot租约guard50项CPU定向测试通过，旧真实lease拒绝exit2。root已在GPU锁内重新只读核WSL/端口/GPU并归档40961B旧lease（SHA003f52ad…77eb30），新lease SHA884715ec4556624cd069b461b1b9cfc3483f0bab32f6f77c6cf2165f6d6b0cac，validator实际PASS；收据GPU_LEASE_REVALIDATION_20261014.json。USER_MANUAL_RESUME继续，原作业零动作，未启动GPU模型。generation代理负责提交其修复与测试。chunking_core原代理也已followup恢复，仅轻量核对R3入口，等待data正常释放；四个原任务已经恢复，无新建代理。02:02 durable6336/6551，host3.45GiB。

**12:59最新：** durable6272/6551，同worker3312/service6016正常，host3.48GiB。generation_evaluation已发现并正在最小修复GPU租约跨主机重启仍被接受的缺口；涉及共享Python校验与runtime_env.ps1 -Gpu，禁止改冻结stress语义。root持有真实lease，待明确新schema后归档旧文件并按当前WSL/GPU/服务证据重新签发，再在实际GPU交接前复查。不要直接运行旧ACQUIRED租约；12:58只读设备12MiB/0%、11435关闭、WSL空列表证据在PRE_GPU_REVALIDATION_20261014_1258.json。CPU执行不受影响。最新root文档提交9325431。

**12:50更新：** 同一CPU4 worker已经实际新增104条，durable6080/6551，manifest未发布；RAM1.47GiB并已通知data观察。generation_evaluation原代理已followup恢复，仅允许轻量离线执行预检、不加载模型，继续等R3释放后22调用stress。给用户的全项目8–16小时为粗略估计；资源反复中断可能24小时以上。WSL配置继续原样。

**12:45新增实际恢复结果：** data_preparation已启动CPU4服务6016/create1791942102.6736171、worker3312/create1791942111.7390096/session37283。root另以psutil重核服务完整argv/create吻合、health200。5976完整观察与末16条cache/vector hash、固定权重/二进制、锁/端口经data核验；新脚本reenter_structured_after_host_restart.py，源SHA及perf hash a316fc68…在HOST_RESTART_REENTRY.json。旧3/3失败journal字节/SHA未改。12:45:42缓存读取5608/6551，durable5976未丢，host3.23GiB；不要把重放控制台较小数字当成果丢失。仍由data独占session，不重复启动。

用户询问WSL内存，随后明确“先不动那个wsl的内存上限”。**没有修改WSL配置，没有待恢复的WSL配置变更。** 当前WSL未运行，C:/Users/lenovo/.wslconfig仍memory=12GB/swap=8GB/processors=12，原SHA d995d866826cbfd36c261689ba27c1673108016d38fe8be20af067234bb3e686保持一致。只读Windows进程工作集统计存MEMORY_OCCUPANCY_20261014_1244.json；ChatGPT/Edge/Codex工具后台为主要应用占用，不能把WSL容量上限当物理常驻占用。不要停浏览器/安全工具/工具服务；用户未请求这项操作。

**2026-10-14 12:39恢复轮次的权威状态，覆盖下方旧进程快照：** 主机Win32 LastBootUpTime为07:03:31.5+11:00，晚于旧worker启动；20816与19112均经psutil确认不存在。原因UNKNOWN，本任务没有执行电脑/WSL重启。5976/6551观察最后写于06:59:17；当前host约4.42GiB，`wsl --list --running --quiet` exit0且结果为空，WSL未运行。证据主run/HOST_RESTART_OBSERVATION_20261014_1239.json。data_preparation已被同一子任务followup唤醒，负责校验缓存尾行/锁/端口/模型后单次正常checkpoint重入；这次是主机重启恢复，不把历史3/3网络失败上限清零或冒称第4次网络失败。新AGENTS已替换旧指令：后续S1/S2进程CPU-only显式4线程；此前12线程保留历史记录，不修改旧结果哈希。root不与data竞争启动资源；等其报告新PID/句柄后更新。其他两个已有子任务待正式交接时再followup，不创建新goal/代理/聊天。

2026-10-14 06:47 Australia/Sydney。Goal ACTIVE、无预算、未终验。用户禁止阶段等待；还有可执行核心任务，不能提前完成或标整体BLOCKED。不要创建新goal/线程/代理。原训练由用户手动续跑，gpu_lease=USER_MANUAL_RESUME；日更不动，结束只释放自己的资源并通知。

**当前恢复已发生，覆盖下面暂停快照：一次受监督备选真实成功，CPU服务19112/create1791920756.4124439。加载9次采样最低host1.62GiB，8-token控制健康请求1024维PASS；这是加载/小健康请求阶段，不是所有正文请求峰值。monitor已join后才resume旧worker，旧socket WinError10054随后真实exit1/session60616已结束。已记第三/最后一次有界缓存重放：新worker20816/create1791920803.708412/session22849，仍归data_preparation所有。5088→5090 durable observations；新控制台的较小计数是重新读取缓存的位置，不是丢失结果。STRUCTURED_THIRD_REPLAY_STARTUP.json实测了此次启动源码hash。host约1.58GiB、服务不重启，按structured→ROUTE→R3交接继续。不要再执行一次性resume脚本/消费第二次备选。**

**当前状态覆盖较早快照：19:26:06Z因持续RAM降至0.59–0.76GiB，data_preparation已暂停精确核验的worker24020，排空自身CPU服务后使36892退出，11436关闭。5088/6551 durable observations，manifest未完成。worker仍SUSPENDED_LOW_RAM并持有campaign锁；Python内存中可能有1在途请求/最多7条未批次记账缓存，不冒称持久进程断点。缓存与旧结果不变，可用RAM回升1.91GiB。STRUCTURED_LOW_RAM_PAUSE.json有完整前后哈希与身份；恢复guard保持≥3GiB，仍由data代理持有和处理，root不重启/解锁/重复编码。现有非计划连接恢复计数2/3，本次恢复尚未发生。其他进程未动。**

学习文档补充已本地提交d95387d：近期里程碑说明、历史BM25状态纠正、真实dev论文相邻块坐标示例。主run/LEARNING_CHUNK_BOUNDARY_EXAMPLE.json SHA88e21a955858d051b9b621d21718516c08ffec0a2b8ca0105fc390f43b7a2afc；三项实际原文坐标检查PASS，token数来自原构建记录，不称新tokenizer实测。此后没有root常驻进程。

最新只读观察06:14：同一R512 worker已写4528/6551，manifest尚未发布，可用RAM1.55GiB，正在正常推进。当前等待期间不增加模型、测试或性能实验；data_preparation拥有worker和session。现有代码/source provenance已保存，当前旧worker不冒称执行了提交后新增的simple分支。

最新补记06:06：root只读观察当前R512 4168/6551已写入，manifest未发布，RAM约1.48GiB。19:00–19:03Z一度.92/.87GiB后回升，进程仍正常推进；没有停止任何进程、重启服务或新增模型。务必准确：当前旧worker仅有开始时768MiB guard，**没有运行中每请求自动RAM暂停**，由外部观察并在持续极低RAM/请求错误时按已有有界恢复策略保存证据处理。topWorkingSet只读观察未发现闲置自有模型；WSL、浏览器、系统维护/安全进程均不动。最新提交还包括16d49f8（E3资源）和0be4b56（simple限定范围），后者36项额外契约PASS。

## 正在运行与下一次交接

- data_preparation拥有CPU embedding PID36892/create1791909754.8890424、11436、12线程、-ngl0/no-op-offload；全509 R512 worker PID24020/create1791913376.0555313/session60616仍运行。不要root轮询其session或重启。05:26快照2192/6551 chunk observations，2601所需vectors已cache，剩3950 unique/1676459tokens，encoding-only预计1.3–1.6小时、RAM2.20GiB。此ETA不是承诺，后续test/global900行、ROUTE520行另计。
- 正常边界：structured全509 test/global和ROUTE260全部完成→worker退出/无在途请求→精确核验后停自有embedding→R3五组660条→R3模型/进程退出→22调用4K/8K诊断→自有GPU服务退出→Laya→恢复embedding处理SQuAD与simple。
- chunking_core现在idle，data已知需用collaboration.followup_task唤醒原代理执行R3。R3实际单论文候选5–28、均11.57，不冒称实际每题50；global才可完整50。R3源码/16测试与CPU采样、有界加载失败已就绪。
- generation_evaluation现在idle；R3退出后root用followup_task唤醒它执行既定22调用stress。E3服务已退出，原11434未动；无root常驻模型或测试进程。
- data正在落实计划379允许的simple范围优化：E6仅scoped test200覆盖75篇，E8/ROUTE/UI均structured。直接缩BM25语料会改变IDF/平均长度，禁止；候选实现只缩simple神经编码至75篇，仍用全509同配置chunk sparse统计，manifest记录scoped_only、75doc hash、all509 sparse reference hash且global拒绝。需逐题实测200个BM25排序/分数等价，再采用；若不等价保留原方案。不修改正在运行的structured/原ArrayIndex。
- 05:46该优化真实200题BM25 top50完整ID/顺序/分数全部相等，计划SHA2f69eebd56caaccdcf027ff41dfbb6a89bcdd6de42f1ca26130020b45d90b26d，在独立20261014_v2_simple_scope_preflight。5710→777神经向量行；4933gross避免，其中4566文本/2230720tokens在预检时尚未缓存。21定向测试+root36冻结/loader/aux契约PASS。主worker同时3168/6551，3582vectors缓存，剩2969unique/1257911tokens，encoding ETA .95–1.16h，RAM1.89GiB。新simple分支尚未实跑。

## 最新代码与验证

- 分支codex/ragscope-v2-autonomous-20261014。最新提交7fef669（UI/dev图表/stress）、9c058fd（FINAL_REPORT正文随机数字反查）、093c18a（Laya受监督加载和完整回归）；更早9c43163/496354d/af94414/3b49ac4/fd836f4/3f67392/50513b4。
- 全core tests/unit 527 PASS/2skip、138.68秒；models CPU routing24 PASS、58.21秒补Torch。去重528PASS、仅Windows symlink权限1skip。176份源hash+JUnit/log在主run/FULL_OFFLINE_REGRESSION_20261014_0533.json，SHA484ef64abb2ab555df1bff443446ff36fb5cdc5fe2ec06162b441d00d44459ab。后续simple新代码须定向验证，最后统一签收；不能把功能夹具当实际模型成绩。
- Laya监督已提交且未执行。固定SDK无官方low-memory选项；默认≥5GiB原路径。仅交接后4.5–5GiB且无本项目模型可一次supervised同SDK/FP32加载+原3dev warmups；低于4.5不尝试。精确PID/create/argv/cwd/ancestor/token握手，首次WARMUP前可停止自己无结果加载子进程，此后只合作落盘退出。加载180秒、采样.25秒、reserve1.25GiB。Windows状态原子写最多5次/.30秒。决策签收routing/LAYA_SUPERVISED_FALLBACK_DECISION.json。没有消费唯一备选。
- final audit脚本读取FINAL_REPORT已有9列表，先按seed42从正文抽10再独立重算raw/source；9项契约PASS。实际FINAL_REPORT尚无，criteria15仍PENDING。命令见ACCEPTANCE_GAP_REVIEW。
- UI冷加载profile与在线常驻profile不等价，默认保持fixed；实验路由仅ready+完整绑定后可手选。实时生成锁/租约/模型实际blob与tag/全局budget验证，raw在锁内落盘、附加timings独立supplement；未实际最终实时UIQA。4模型CPU pins/clientidentity已PASS，ready仍false。之前BM25+E3回放IAB/Streamlit进程已结束。

## 已完成真实实验与证据

- E1全11×3dev100=3300，扩展8×100=800；选F512_O0和R512、neighbor。44runs共4600/18430pieces/132数独立PASS（含其他组）。冻结dev报告reports/v2/retrieval_interim/，41dev条件4100行、16412pieces、123数/164hash；SOURCE_SNAPSHOT SHAeb7de862fe1603e18cbea2a26459560d3aed4e818d7aafa6a34180fe61633429。R512 full净+1/92但continuous下降，F512 full净+2/92但continuous下降；不宣称全指标改善。
- E3 generation_e3全1200/0errors/1200token parity/9warmups完成且封存释放。Oracle可答F1 2B .5785604023、4B .5455298179、9B .4704580676；NoContext .0414930593/.0044469761/0。analysis_complete全30数独立PASS，早期analysis快照保留。
- E3新error_cases子报告：1200raw+1200coverage+1200引用+897片段核验PASS；8例agent inspection，1200人工行全NOT_REVIEWED。53缺引用候选有49空上下文N/A，524适用输出缺4；Oracle226非拒答低F1不等于已确认事实错。总200题75论文、可答182题72论文（旧报告误写72全篇已修，旧报告/receipt在history）。E3_ERROR_REPORT新SHA6931daab8f21e46ddd9248679498622a90dd07a2bfbf476e9325d1c73de7d37b。一个真实Oracle标注证据集缺参考答案相邻原文事实，coverage1不代表所有答案事实已提供；学习02有例子。源/旧ANALYSIS未改，0新调用。
- 05:43 root生成reports/v2/resource_interim/：E3的1200正式行+9独立预热资源报告，P50和显存/RSS逐项与analysis复算匹配，0模型调用。来源快照72de64e2acb7c50154bf63da8f2c3aa1692429dbc3e8217e9fb795d8c237ffe2。build_e3_resource_report.py一次运行即保留结果，拒绝覆盖；第一次warmup字段KeyError日志留在logs/e3_resource_report_build_attempt1.log，已修按config取model_id后exit0。不要作为全项目资源验收。
- LoRA train_2b_1024_seed42真实122steps/972train/59dev、完整5→10→122恢复、best122 devloss .1356631101；BF16同精度100×2问答仍待最终R2inputs。旧unseeded10步只作可行性。adapter SHAa532dc22942d84e4dceabacb64b936c191b37ebdfa9fda092f3f87464d0bbae4。
- E0 fixed revision CPU22254vectors400rows完成，原8索引SHA不变。Dense Recall5/.10 .870/.905、Hybrid .915/.950；MRR .7615595/.7970119，nDCG .7966956/.8347330。146原文章2000bootstrap与独立10数PASS。历史weight UNKNOWN；标legacy_reconstructed_fixed_revision。E0与LoRA图已实际目视，主run/FIGURES_E0_LORA_VISUAL_QA.json。
- stress CPU冻结8个源可行dev，92too-short保留，4K3584input/8K7680input+512out，16formal+6warmup=22未运行；仅2B，manifest e87d506a9eab594eac0f088317d094df315c82097bc3eaa3ce0a56cbb04589c9。

## 仍必须完成

E2 2400、E4 300、E5 200、E6最多400、E7最多240、E8global4methods；ROUTE780真实标签→fixed3/rules/linear/MLP/LSTM与Laya、dev60/test100全部决策先冻结、online30×2；LoRA200；support judge≤200；真实最终UI；报告图/误差/人类NOT_REVIEWED；十五项ACCEPTANCE与从FINAL_REPORT随机≥10数字raw反查；轻包、最终磁盘/原件核验、释放GPU并通知手动续原任务。当前globalbudget1421reserved/1200core，max10000/core3600保留。任务账本9PASS16PENDING不是最终验收。

S1/S2计算CPU-only，但S3/S4与S2未结束有时间重叠，criterion14必须披露限制，不能伪称严格阶段串行。所有大文件D runtime；中期hardlinkdedup27.32GB，final重查。不要把整盘C/D变化全部归因项目。原Ollama11434/日更/原训练断点不动。

只显式stage任务文件；保留用户untracked plan/md/zip/startprompt、architecture.md/html、rerun/、svg/；不add-A、不push、不删除。Git用-c core.autocrlf=false -c core.whitespace=cr-at-eol。已冻结Matplotlib SVG trailing spaces例外只排除其文件，不改已签名图片。

下一条准确命令（只读；data继续拥有真实worker）：
```powershell
Get-Content -LiteralPath D:/RAGScope_Runtime/runs/20261014_v2_retrieval/worker.json -Raw
```

历史恢复入口`scripts/resume_structured_after_low_ram.py`已实际执行，**不可再运行**。root审阅发现的“resume成功后写盘失败误清理服务”已修：成功resume立即记录本地状态，之后异常保留服务；resume前异常才按PID/create/argv清理。后续加入一次受监督备选，5项小型行为fixturePASS；实际启动源SHA8f99d1a35eda13703fe4d3e5574aeec174779a7754fe20f2d1b32e2a3a53610b。WSL已运行的Ubuntu meminfo只读检查有大量guest可用内存，但不能当host可用，未修改配置、清缓存、启动或重启WSL。
