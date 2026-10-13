# 主执行器恢复点

2026-10-14 04:45 Australia/Sydney。Goal ACTIVE，无预算；尚未终验，不要新建goal/线程或提前完成。

用户最新恢复政策：原训练和定时检查已停，日更不动；结束释放后通知，由用户手动续跑。gpu_lease=USER_MANUAL_RESUME，不自动恢复。

## 当前资源/代理

- data_preparation：CPU Dense/Hybrid 11/11 dev已完成2200条、0失败；worker28832/session13666正常exit0。服务36892/create1791909754.8890424仍CPU12/11436。正式dev冻结simple F512_O0、structured R512。当前一次有界batch1/2/4 benchmark session33240，固定8dev文本、最多29请求；≥1.15×且向量/排名等价/资源保护通过才采用，否则保留串行。下一步选定扩展→structured all509 TEST200/E8/ROUTE260→正常边界释放embedding给R3/Laya→再simple/SQuAD。不得中途切断HTTP；此前2/3恢复保留。
- generation_evaluation：完整E3 1200/1200、0失败，实际token匹配1200；9warmp单列，global1421total/1200core。自有11435/PID14564已卸载退出、端口关闭、gpu.lock重新获取PASS，当前GPU模型空闲。正在生产acceptance和S3.5上下文压力诊断：实际8dev源足够长，其余92too-short有证据；4K输入3584、8K7680、统一512输出，16冻结正式输入+6warmups=22预算调用，独立run_context_stress、尚未生成。manifest e87d506a9eab594eac0f088317d094df315c82097bc3eaa3ce0a56cbb04589c9。等data benchmark与RAM交接再启动2B，做完释放。
- chunking_core：E0全部完成，额外200题严格回映原SQuAD146articles、2000cluster CI已真实计算，8测试PASS。R3真实50→10五个population和Laya1080 raw decisions等待structured/full/ROUTE源和RAM。当前可加Streamlit1.65.0到models隔离env用于最终R3 UI；必须保持原45包版本，旧manifest/lock归档、新锁/pipcheck，若冲突不改模型栈。root已授权该依赖增补。
- root临时UI服务22472+venv launcher27044、session41625已退出，两PID不存在，浏览器临时页关闭。UI_INTERIM_BROWSER_QA.json记录BM25实际查询及4B400行浏览器回放。Chrome工具不可用，使用既有in-app browser，没有安装Chrome。cua_repl ragTab已close，若后续浏览器任务需重取新tab/文档。

## 已完成证据

E3 run：D:/RAGScope_Runtime/runs/20261014_v2_generation_e3，manifest 8c6a71031d25f0a0218a85a3abbd9d8dc35d3c0f56dafa046e8249beeffb5599。COMPLETION_RECEIPT和GPU_RELEASE_RECEIPT已封存。analysis_complete/ANALYSIS.json COMPLETE1200，INDEPENDENT_NUMBERS_ALL.json实际30数字独立复算PASS。原2B analysis/快照不要覆盖。Oracle F1分别2B .5785604023、4B .5455298179、9B .4704580676；NoContext .0414930593/.0044469761/0，9B全部200拒答，不是请求失败。图PNG/SVG和E3_REPORT.md已实际生成/目视，当前完整主E2仍0。

新production_acceptance.py / scripts/accept_generation_v2.py：固定主3模型/200题及扩展预声明集合、blocked证据、跨shard逐行核验。E3独立scope COMPLETE1200不等于main；主scope PENDING无manifest。完整性错误状态FAILED_VALIDATION，审计本身audit_generation_calls0。10测试通过；catalog S3.5接入该实际校验。

路由decision guard已修：完整dev/test唯一ID集合、策略名、合法route/时间、实参与已冻结文件一致，7种策略都先验证再读test质量。campaign+online19测试PASS。

正式seed42 LoRA122步/972train/59dev已完成，5→10→122真实恢复、3init一致、3完整断点、best122 adapter a532dc22942d84e4dceabacb64b936c191b37ebdfa9fda092f3f87464d0bbae4。TRAINING_AUDIT PASS，模型/GPU已退出；BF16base/adapter固定100test仍缺最终R2inputs，不能宣称QA质量提升。新的training loss PNG/SVG实际从122条hash记录生成；ylabel已修为样本loss均值，最后图需在最终汇总中复看。

11个CPUdev矩阵完成；旧all509BM25两索引preflight.split历史错误为dev，但实际identity/scope/doc_checks是全库且有效，CHUNK_PREFLIGHT_SCOPE_AUDIT已明示，不改旧manifest。未来buildmetadata已修。

core全套先前414PASS2skip仍是最近完整套；本次变化相关95PASS/32.01s，日志D:/RAGScope_Runtime/logs/milestone_e3_regressions.log和主run/milestone_e3_regressions.xml。之后UI新增RAMguard、startup Environment auto未再全套；最终统一复验。UI caption不再把仅检索写成模型实际接收，启动器scripts/run_ui_v2.ps1+项目.streamlit/config.toml强制loopback、telemetryoff、端口占用拒绝，18501空闲check0/11436占用check1已实测。

## 下一步与不可提前完成项

CPU最终上下文/structured全索引→R3/Laya→simple/SQuAD；main生成manifest严格外部引用E3不复制改写，运行E2和扩展/ROUTE780；七路由真实训练、test100冻结决策、Laya校准和30dev×2在线；BF16pair100×2；support judge≤200；最终分析图/错误/中文FINAL_REPORT/15条ACCEPTANCE/随机≥10从最终报告反查/轻包/真实清理。原任务最终只通知手动续跑。

D:/Runtime中期hardlink去重27.32GB，final需重查；C例外为既有可执行/字体读取，整盘差异不独占归因。阶段S1/S2运算CPU-only但后续已按追加资源优先授权与S3/S4重叠，criterion14必须如实标注，不能虚构顺序。

Git分支codex/ragscope-v2-autonomous-20261014，已提交9c43163/496354d/af94414/3b49ac4/fd836f4/3f67392。root将提交E3/守卫/UI里程碑，压力诊断仍在开发不要混stage。用户原untracked plan/zip/startprompt、architecture.md/html、rerun/、svg/保留，不add-A、不push。git diff检查用-c core.autocrlf=false -c core.whitespace=cr-at-eol避开CRLF误报。

下一条准确只读命令：
```powershell
Get-Content -LiteralPath D:/RAGScope_Runtime/runs/20261014_v2_retrieval/CANDIDATE_SELECTION.json -Raw
```
