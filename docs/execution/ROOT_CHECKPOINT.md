# 主执行器恢复点

2026-10-14 05:14 Australia/Sydney。Goal ACTIVE，无预算，未终验；不要新建goal/线程或提前完成。

用户最终恢复政策：原训练/定时检查已停，日更不动；结束释放自己的资源后通知，由用户手动续跑。gpu_lease=USER_MANUAL_RESUME，禁止自动恢复。

## 当前资源与调度

- data_preparation：dev11×3全部完成，最终simple F512_O0 / structured R512，扩展均neighbor。实际44runs/4600rows/18430pieces/132数字独立PASS。batch29请求基准完成，batch2=1.00013×、batch4=.92875×，不达1.15门槛，保持CPU12线程/request batch1。
- CPU embedding PID36892/create1791909754.8890424，11436，-ngl0/no-op-offload；full509 R512 worker PID24020/create1791913376.0555313/session60616仍运行，最近1368/6551已持久化，估算尚约1.5小时（不是保证）。不要中途重启。旧28832/session13666正常结束。
- 独立命令边界：structured all509 TEST200/E8global100完成后ROUTE260；然后正常退出worker、核验无在途请求并停自己的embedding。接续顺序为R3五组→模型退出→22调用4K/8K诊断→GPU服务退出→Laya→恢复embedding继续SQuAD/simple。代理均已知，不需用户阶段确认。
- chunking_core：R3真实输入dev100已核验，单论文实际5–28候选、平均11.57，不能声称每题实际50；其余4组仍等sources。CPU sampler/有界load失败守卫已16测试PASS，未实际载R3。正在准备Laya唯一条件备选：仍优先5GiB；正常资源交接后若4.5–5GiB可用且无其他项目模型，可同固定SDK一次受监督加载+既定3dev warmups；低于4.5不试、失败不再降阈值。实际421293827F16params+3F32buffers，FP32模型1.685GB+mappedweights.843GB=2.354GiB张量地址空间估计，非RSS下界。SDK没有官方dtype/assign/low_cpu_memory接口。源码审计报告LAYA_LOAD_MEMORY_AUDIT.md，当前实现中，不要碰其文件。
- generation_evaluation：E3已1200/1200完成且自身GPU服务退出。UI默认门槛修复完成40项案例，代理已结束；root统一稳定模块回归482 PASS/1 skip（125.65秒），仅排除Laya正在开发的三份测试，不能称最终全套完成。下一次真实GPU工作等R3退出后的压力窗口，可followup原代理。
- root：UI/模型身份/锁内原始落盘、E0/LoRA图表/文档更新完成。无root常驻exec或GPU进程。原Ollama11434未动，自有11435已退出。available近期~2GiB，低于stress有效3GiB，不启动GPU。

## 已完成证据

E3 run D:/RAGScope_Runtime/runs/20261014_v2_generation_e3。1200rows/0errors/1200实际token匹配，9warmups单列。COMPLETION_RECEIPT和GPU_RELEASE_RECEIPT封存；analysis_complete/ANALYSIS.json COMPLETE；INDEPENDENT_NUMBERS_ALL.json 30数PASS，E3_REPORT与PNG/SVG已目视。原2B analysis/快照不要覆盖。Oracle F1 2B .5785604023、4B .5455298179、9B .4704580676；NoContext .0414930593/.0044469761/0，9B全部200拒答，不是后端错误。E2仍0/2400。

Globalbudget1421reserved/1200core，上限10000、核心保留3600。UI调用现在也计入同账本，尚0次。实际LoRA122steps/972train/59dev、5→10→122真恢复及3个完整checkpoint，best122 adapter a532dc22942d84e4dceabacb64b936c191b37ebdfa9fda092f3f87464d0bbae4。同BF16 100×2 QA仍待最终R2inputs，不能以loss替代问答收益。

E0重建400rows已独立10数PASS，原8index SHA不变；额外按146原始SQuAD文章聚类2000bootstrap的10指标CI完成。新plot_results_v2.py --legacy-analysis实际生成PNG/SVG，图与更新ylabel的LoRA图均重新实际目视，主run/FIGURES_E0_LORA_VISUAL_QA.json记录。

dev图表在reports/v2/retrieval_interim/DEV_RESULTS.md，三图PNG/SVG与FIGURE_MANIFEST PASS_VISUAL_QA。冻结runtime retrieval/analysis_complete_dev/SOURCE_SNAPSHOT.json SHA eb7de862fe1603e18cbea2a26459560d3aed4e818d7aafa6a34180fe61633429：41dev条件/4100行/16412pieces/123数字/seed42抽10/164源hash。neighbor有明确取舍，R512完整覆盖净+1/92、连续-2.40pp，F512完整净+2/92、连续-6.99pp；1/92=.01087略大于预设.01工程容差。不能写所有指标改善。

压力诊断已CPU冻结8/100dev源可行，92too-short证据保留；4K实际输入3584，8K7680，各512输出；16正式+6warmup=22预算。run 20261014_v2_context_stress manifest e87d506a9eab594eac0f088317d094df315c82097bc3eaa3ce0a56cbb04589c9。仅最小2B，未生成；guard require_ram(2GiB)+默认reserve1GiB=3GiB，未改代码不降阈值。冻结实现context_stress.py/run_context_stress_v2.py勿改。

production_acceptance固定生产人口/方法/ownership验证已完成，E3scope COMPLETE1200，mainscope PENDING无manifest。任务catalog S3.5接入。主账本已9PASS16PENDING，S2.3完整33×100实际核验PASS；历史S1.3回执因ui config更新过期，已保存history并重新验证134PASS，不要反复因root新功能改动刷新，最后统一签收。

## UI新实现与审阅

ui/routing_options.py：可见request只来自question/用户scope/规范docs metadata。与冻结decisions/router identity/index/expansion绑定；未就绪不出现可选项。generation代理正加严格deployment_readiness(policy, *, model_identity, index_manifest, runtime_config) 与deployable_default同参bool。已定义RESIDENT_PROFILE='resident_index_router_reranker_and_generator_after_separate_warmups'，其他profile直接False。当前UI传cold_application_loads，所以固定默认，实验可手选，不能借预热延迟收益上线。真实通过分支必须核dev60/线上30×2实际记录/票据/完整identity并复算；代理正在测试。

ui/model_identity.py：模型同名不够，probe同时核tag digest与冻结manifest；调用锁内核服务实际models目录/对应manifest/真实weight blob。四模型实际包CPU校验PASS，主run/UI_MODEL_PACKAGE_PINS.json保存ready=false（不是服务运行证据）。build_ui_model_pins.py可复现。另UI_CLIENT_IDENTITY_CHECK.json实际四客户端与冻结pilot身份完全一致，真实本地模板计数，0HTTP/GPU。最初临时脚本两个工程错误保留文档：LFM未注入既定runtimecache policy；Qwen默认counter_kind省略，正式脚本已修，不改旧实验。

ui/live_generation.py：只接受real_model，复用既有GuardedClient租约/PID核验，持generation-client.lock；CountedClient记UI用途，MeasuredClient采样。answer_live必须model_entry，基础原始输出在释放锁前save_live_record；页面routing/timing单独supplement=True，不覆盖原始。UI R3改用load_verified_reranker及完整identity（正式batch2/CPU4），不再原始默认batch4。13项app/live/modelidentity测试PASS，先前UI全22PASS，代理route新增完成后统一测试。

实际浏览器QA仍仅BM25+E3历史回放，原临时UI22472/27044已退出、IAB页关闭。最终R3/实时生成UI尚未验证。cua后续须rewriteDocumentation，再建临时IAB页。模型/检索最终config发布与dev选择待完成。

## 下一步与禁止提前终验

稳定模块482 PASS/1 skip已完成，root将显式提交本次UI与dev报告里程碑；Laya备选完成后全core/models最终复验。持续完成CPU全索引/R3/Laya/GPU主E2及E4-7/ROUTE780、七路由train/dev/test、30dev×2在线、LoRA200、support judge≤200、最终报告图/错误/人工未审标记、15项ACCEPTANCE、从FINAL_REPORT随机≥10数反查、轻包、清理并通知手动续原任务。

主项目路径D:/MCS_26S1课程资料/个人项目/RAGScope。runtime全D。中期hardlinkdedup27.32GB，final重查；不可将整盘C/D变化独占归因。S1/S2运算CPU-only，但S3/S4与未完成S2有时间重叠，criterion14如实限定。

Git branch codex/ragscope-v2-autonomous-20261014，提交9c43163/496354d/af94414/3b49ac4/fd836f4/3f67392/50513b4。当前UI/图表/stress/dev报告等未提交，agent Laya备选仍开发，不混入其半成品。用户原untracked plan/zip/startprompt、architecture.md/html、rerun/、svg/不得add-A/push/delete。diff用-c core.autocrlf=false -c core.whitespace=cr-at-eol。

补记：UI/dev报告/stress里程碑已本地提交`7fef669`，47个文件；仅已冻结Matplotlib SVG自带的trailing whitespace从源码空白检查中排除，以保留图表manifest SHA，其他检查通过。Laya代理仍修复Windows child.json原子替换WinError5竞态，勿将其文件纳入本次提交或完整测试。generation代理在独立E3错误子目录整理实际1200记录案例，没有新增模型请求。最终报告数字反查新入口及9个离线契约测试已通过，正式反查仍待FINAL_REPORT。

下一条准确命令（只读）：
```powershell
Get-Content -LiteralPath D:/RAGScope_Runtime/runs/20261014_v2/UI_CLIENT_IDENTITY_CHECK.json -Raw
```
