# 主执行器恢复点

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
