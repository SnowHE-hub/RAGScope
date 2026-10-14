# 决策记录

2026-10-14 07:33UTC：实际源码修复后的完整回归已封存，core700PASS/2skip及models24PASS/0skip去重701PASS，唯一剩Windows符号链接权限skip；不通过改安全权限补测。219源前后逐字SHA相等、两suite实际exit0，receipt FULL_OFFLINE_REGRESSION_20261014_0658.json SHA55d9637ef34c9797ea20bccb7beef8c76e12e258fb6b58586aa22759ee232c0e。models CPU4测试与BF16前几题有短重叠，实测条件单列。主5520生成已完整，32解析失败不重试/不丢弃；支持性judge200保留1UNKNOWN，人审仍NOT_REVIEWED。同BF16固定配对基础100已完成且1解析错误保留，adapter100进行中；prompt_only/applied=false实际协议优先于通用interpretation字符串，不把Q4 schema保证套用。根大对象统计明确在此GPU进程退出后、CPU embedding启动前执行，保护当前约3.6GB可用RAM且保持原预载安全下限。


2026-10-14 06:42UTC：ONLINE在七策略冻结后仅使用完整60dev选择fixed_hybrid/rules各30相同ID。发现原实现只有BGE预载effective3GiB检查，正式请求/warmup没有RAM边界保护；在0次真实online请求前作最小安全修复2b2ba18，逐warmup/新任务开始前1.25GiB不足正常INCOMPLETE_RESOURCE/剩余pending，不耗预算、不伪造失败行。旧manifest/prepare/source逐字按SHA归档，fresh freeze独立比较仅implementation与online_hash改变，策略/30IDs/模型/prompt/配置/时序不变。当前online_hash16a0c607082e95cd54336c86889cf7f887f8bbdaf32f4b6419b7339e0043a3f4，16定向测试PASS（原9+新增7）。正式在线仍未实跑，等原BF16释放后同2B驻留窗口。

同日固定模型train200各一次共600分类诊断真实完成，不重训、不新增种子或LLM请求。线性train accuracy .675/dev .516667；MLP/LSTM .64/.516667，后两者train也全BM25，gap恰等于多数类比例差，不能单凭gap宣称过拟合。labels接线仅把task_catalog的6处proxy_labels.jsonl改为实际producer的labels.jsonl；旧catalog/签收留档。S4.2修订签收2235e1a51518a908060037d662cd8e4bc2ba17d04d87918a8bf38b3ad3917201。

全新D盘core_acceptance_20261014按74锁包offline require-hashes安装/pipcheck/importsmoke全部actualexit0，补足干净环境可重建实证；不改当前core，不加载Torch/模型。检索12完整组及5预声明cluster对照actualPASS，所有失败和不可答分母/NA保持。S2.6以PASS_FALLBACK明示官方旧模型重建的UNKNOWN历史权重、CPU partial29与完整GPU100来源差异，以及用户授权后的S2/S3墙钟交叠；不声称全过程原任务未动或C盘零写入。

2026-10-14 05:58UTC：实际浏览器验收在GPU生成持锁期间只调用已就绪BM25，返回四个真实上下文片段并保存DOM与截图。就绪模型0/生成N/A原样显示，不用Mock填补；结束正常退出自身UI服务释放RAM。最终真实生成/UI配置发布将在在线GPU窗口核验后执行。路由replay发现外部E2来源凭据路径误找本地副本，只最小修复来源绑定并25定向测试；原失败留档、冻结labels和1120决策不变。最新679全套是旧216源签名；该修复另有215不变/1变化及新增7测试的差异签收，不把旧套件冒充新文件已全套通过。三学习路由dev全选BM25是实际类偏斜结果，不能据test或“智能”目标重训/调参。

### 2026-10-14 04:52UTC：真实资源释放与路由标签优先

4B800+3预热真实完整后正常卸载/请求退出自己的服务，Windows supervise实际exit1保留；unloaded=true/stop_reason=request/精确PID及端口/两锁释放才构成安全交接，不把非零意图停止改成0。SQuAD100已复用1212索引正式CPU4执行，100冷query+3cache预热、0error、实际exit0、三锁释放，全部原usage存在；不是重新编码正文。freshRAM约5.5GiB、GPU12MiB，GPU R3360按新用户授权真实启动，48功能测试不计真实360，已成功load/首24零error。

R3_GPU_COMPLETION_VALIDATION完整360/退出/配置及保护CPU artifacts全部hash绑定后，主spec只显式替换ROUTE_TRAIN/DEV hybrid_rerank，E2仍原CPU test来源；全球E8只取uniform GPU100，CPU partial29保留历史不拼合。21定向测试及真实未完成来源拒绝已执行，0新增生成。第一次测试命令拼错已有测试文件名exit4/no tests，正确路径第二轮21PASS，两日志保留。新增CPU Laya原加载窗口若≥5GiB可先加载后稳态guard允许时与GPU生成并行，不降低原阈值或改变SDK算法。main2B优先local-owned1200最多，使780 ROUTE标签尽早就绪/七策略先冻结；后续模型仍固定完整分母，9B核心由原E2 owner执行，main不会补跑外部缺行。该优先级不使用test质量选模型或策略。

### 2026-10-14 04:36UTC：用户任意阶段GPU授权及R3迁移

人类最新指示“现在可以在任何阶段使用GPU，我没有任何进程跟你抢夺GPU资源”，覆盖旧主书/AGENTS阶段CPU-only约束，不修改原任务书以掩盖覆盖。单卡串行、安全保护、原任务USER_MANUAL_RESUME/WSL不改仍有效。R3 CPU已04:15:58在global第29条写后触RAM guard真实exit1，329成功0error/完整CPU E2测试来源保留。使用现有CUDA train env TF5.19直接固定BGE-base分类器，FP32/eager/TF32off/sigmoid、同512窗/overlap32/batch2；CPU旧ST后端与新TF版本差异必须披露。同模型不代表设备数值与延迟完全相同。独立GPU E8全100+ROUTE200/60，不把CPU partial29拼成uniform GPU组，不改CPU test200用于E2。R3_GPU_MIGRATION及独立validator绑定保护哈希/全部IDs/新来源/资源；prepare和40功能测试不是360真实完成。4B先完整结束退出，再SQuAD100短CPU窗口，再GPU R3，释放后9B/judge/BF16；不重下模型、不结束未知应用或安全工具。LOAD_STATE backend_failures仍0，attempts2含成功加载，不能按成功次数误说已耗尽三backend失败重试。

simple三种检索现600完整0error。cache-only曾因三历史key缺backend_usage而退出，这三vector均真实且全identity/hash/shape/finite符合，原structured相同六行本就cache_hit/usage=null。明确只允许这三key历史N/A复用，正式400+6预热全部真实缓存命中/HTTP socket尝试0，记录null不编usage、不改原vector/manifest，不把cache-only当新编码实验。四行为测试与完整RESULT/EXIT证据保留，未知其他缺usage仍拒绝。

## 2026-10-14 初始化
- 以当前已打开且含 ragscope/、scripts/build_indexes.py 的目录为原项目，不搬迁。
- 原 Git tracked 内容无修改，用户未跟踪文件按清单保留；只提交本轮明确文件。
- CodeGraph 无索引。按本次自主执行授权初始化本项目索引，不修改全局设置。
- Runtime 原不存在，创建 `D:/RAGScope_Runtime`；缓存只在本项目子进程设置，不使用 setx。
- 32逻辑核按任务书限制为4线程；初始空闲RAM较低，CPU模型需串行、低内存实现与加载前预算核验。
- 采用 Windows Python 3.12 隔离环境；现有 uv 可执行文件可用，但尚未发现受管已安装 Python。必要时由 uv 从官方 Python standalone 发行资产安装到 D 盘专用位置。
- 保留现有模块增量修改，不新建平行项目。Mock仅离线功能验证，正式结果必须有真实模型身份。

## CPU检索、生成与资源交接
- Qwen3-Embedding-0.6B采用官方Q8_0 GGUF+llama.cpp CPU构建，真实tokenizer计数；三次内存预检拒绝后，在阶段三获追加CPU/GPU终止授权并释放已核验研究任务，重新启动成功。S2计算本身始终CPU。
- BM25矩阵与E8已实际运行；Dense/Hybrid仍在逐条可恢复运行，不能依据BM25初筛冒充最终dev选择。
- 主生成模型采用官方Ollama明确2b/4b/9b-q4_K_M标签，避免默认2B Q8_0与其他Q4_K_M混比。自建环回11435服务和独立D盘模型仓库，不重启原服务。
- Qwen3.5提示词使用固定v0.34.4原生renderer源码审计，完整chat token与真实后端逐条比较。首例1981=1981；冷启动83s、首轮prefill76s单列，不当作稳态延迟。
- 预检发现模型可能把引用ID外方括号包含进JSON；原始无效引用保留，prompt v2.1明确去掉外括号，后续所有模型同prompt重新预检，不伪造或自动补引用。
- 原研究队列曾被另一个Cursor会话重启；两轮中断/断点核验历史保留。最新用户确认已停止训练及定时检查，且用户将自行续跑。最终只释放本项目资源并通知，不自动恢复原队列，不触碰日更进程。

### 2026-10-14：定位并关闭本项目后端的历史 prompt RAM 缓存
2B/4B 连续真实预检中，独立服务多次因可用 RAM 连续低于 512 MiB 退出。4B 日志每次换题保存约121.803 MiB prompt state；这不是权重大小变化。安装的 llama-server --help 实证默认 --cache-ram=8192 MiB，并支持 LLAMA_ARG_CACHE_RAM；Ollama v0.34.4 llm/llama_server.go:417 使用 os.Environ() 传递环境。因此只给本项目服务增加 LLAMA_ARG_CACHE_RAM=0，不改原服务/全局环境，不降低保护线。重启后需由实际日志确认禁用。已失败行保留，重试仅补失败/缺失题；过去的pilot延迟不能与正式禁缓存结果混合。
官方来源：https://raw.githubusercontent.com/ollama/ollama/v0.34.4/llm/llama_server.go 与 https://raw.githubusercontent.com/ggml-org/llama.cpp/b11429/common/arg.cpp 。上游问题 #18264 作为线索，实际决策以已安装二进制帮助和版本源码为证据。

### 2026-10-14：正式冻结前最后一次引用输出约束修正
对相同20条dev预检独立汇总发现：2B/4B/9B/LFM分别有10/11/17/20条回答包含不可见引用ID（常见为漏掉qasper前缀），虽然JSON和token预算均有效。保留这些负结果与原始输出，不能称为引用质量通过。
在正式test生成尚未开始的前提下，统一为全部本地生成模型使用ContextBundle可见chunk_id的JSON schema enum；无上下文只允许空引用数组。该约束只限制合法引用标识，不选择金标准证据、不修改答案事实、不在生成后修补引用。schema/hash/配置写入结果，prompt版本升级rag-json-v2.2并做一次20条×4模型的实际兼容预检；这之后冻结，不继续增加prompt搜索组合。
引用ID可见性因此成为受输出语法保证的工程性质，正式报告必须明确这一点，不能将100%可见性解释成模型事实支持能力或与旧自由ID条件直接比较。最终引用内容是否覆盖证据、回答是否受所引内容支持仍需真实评测。

- 最终LFM预检20次真实完成且token一致，1次done_reason=length在统一512输出上限截断JSON。兼容状态单列PASS_WITH_OUTPUT_FAILURES，质量失败19/20有效输出如实保留，不因输出质量更换模型、提高上限或重试挑答案。冻结器仅接受有确切length证据的此类输出失败，传输/OOM/token或schema差异仍阻止冻结。

- 路由正式接线固定读取生成manifest内ROUTE_TRAIN200/ROUTE_DEV60；test100在看路径结果前seed42冻结，避免旧辅助hash采样选成另一批题。可答质量仍为token F1；不可答的F1保留N/A、标签使用明确correct_abstention代理，单列拒答率，不把两者混称平均F1。所有回放成本为实际检索与生成计时之和，尚不是在线耗时。26项路由测试PASS，1项Torch专属测试另环境。

- 报告独立reports环境使用Matplotlib3.11.2与11项hash锁依赖，未改变core/train/models环境；实际缓存D:/RAGScope_Runtime/cache/matplotlib，PNG/SVG落盘D:/RAGScope_Runtime/runs/20261014_v2/figures。官方savefig接口核验：https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.savefig.html 。最终预检显存图已实际导出并目视核验；主矩阵图在3×4完整200test结果之前拒绝生成。

- 调度偏离显式记录：所有S1/S2计算与HTTP embedding始终CPU-only；为并行推进，S3真实模型预检与S4训练准备在S2完整Dense矩阵尚未结束时执行。不能将这描述为严格完成全部阶段二后才进入阶段三。资源控制采用用户随后“可随时终止CPU/GPU，本任务优先”和最新“训练/定时检查已停、结束后用户手动续跑”的直接授权；原断点校验/精确PID控制证据保留。阶段二最终非干扰报告应按这条实际时间线写明，而不是虚构当时原训练仍在运行。

### 2026-10-14：先执行与检索选择独立的E3核心对照
CPU全矩阵仍在运行时，先冻结同200test、相同最终提示/模型/共同预算的Oracle与NoContext独立shard，共1200次，不新增组合。最终主manifest引用其不可变manifest；任务ownership覆盖尚未完成的外部分组，主runner禁止重复跑，分析直接读取原记录而不改写哈希/调用票据。独立数字auditor另实现同等来源验证。已真实完成2B400条并从raw输出及source坐标核验10数字，尚不能称完整核心完成。并行CPU任务可能影响墙钟延迟，故本轮延迟为所记录资源条件下的描述性测量，不承诺独占机器的理想性能。

### 2026-10-14：启动与本地交付核验
已有core环境的bootstrap改为只读dry-run与pip check，不默认覆盖或同步；新环境固定Python3.12.13并require-hashes。实际74包Would make no changes。Windows显式GPU/资源交接参数已接线；WSL只做bash语法和缺环境失败路径验证，环境未安装、退出码2，不声明Linux端到端通过。本地轻量打包只取明确报告/公开benchmark结果和协议，十五项验收仍PENDING时拒绝导出，不上传，不含权重或原任务快照；portable路径修改与原哈希同时明确记录。

### 2026-10-14：完整E3与资源诊断边界
E3三模型1200条全部完成、0请求失败，完整分析写入新的analysis_complete，保持早期2B analysis快照不变；30数字从raw/source独立复算PASS。观察到Oracle F1为2B .5785604、4B .5455298、9B .4704581，保留非单调结果，不据此换模型或调prompt。NoContext9B全部拒答的0 F1不冒称后端故障。
按任务书S3.5落实尚缺4K/8K资源诊断：仅2B、最多20个按源可行性预选dev问题，两条件与每条件3warmups，独立冻结/记账，不与主质量排名混算；先等CPU有界batch验证与资源交接，未运行之前不声称8K能力。

### 2026-10-14：Laya一次受监督加载备选

固定SDK实际421,293,827个F16参数与3个F32 buffer；默认FP32目标加mmap权重的2.354GiB只是张量地址空间估计，不是RSS下界。未发现官方减少加载副本参数，不改dtype或SDK算法。正常释放embedding/R3和压力诊断模型后，仍优先可用RAM≥5GiB原路径；若4.5–5GiB且无其他本项目模型，允许一次受监督原SDK加载与原3题dev预热，低于4.5不试。该工程备选来自本任务自主处理授权，不是已验证可行性。

模型子进程必须经过PID/create/argv/cwd/祖先/token核验与两次许可；只有未授权预热的加载阶段可停止精确自有无结果进程，此后仅合作保存退出。180秒加载上限、.25秒采样、1.25GiB余量；失败保留证据且不再降低门槛或消耗第二次备选。原任务和日更不参与此控制。源码审阅、静态统计与签收在`LAYA_LOAD_MEMORY_AUDIT.md`；真实执行仍PENDING，不能用528个离线测试通过替代Laya实测。

### 2026-10-14：E3资源与错误解释保持独立范围

`reports/v2/resource_interim/RESOURCE_REPORT.md`从1200封存行重算P50与显存/RSS峰值，另列9次预热。设备总显存采样不当作模型独占或硬件最低需求；非流式首token保持N/A。首次报表脚本假定warmup行有model_id，实际该字段在其config中，工程错误日志保留后已修并完成；不修改任何实验行或统计协议。

E3错误报告明确全部200题75论文、可答182题72论文，修正副报告的原措辞并保留旧报告/收据。53条缺引用候选中49条上下文为空，仅此指标N/A且不删行；Oracle226条非拒答低F1不等于226条已确认事实错误。代理核对不是人类审核，人工字段始终NOT_REVIEWED。

### 2026-10-14：simple正式索引缩减编码范围，保留全库稀疏统计

按任务书379行，E6 simple仅需冻结test200所选75论文；E8/ROUTE/UI继续用structured全509论文。直接按75篇重建BM25会改变IDF及平均长度，不采用。新显式scoped-only索引仅编码该75篇全部777个相同chunk，BM25仍按509篇5710chunks计算，固定外部参考的文件/内容/配置hash，拒绝global和未知paper。实际200题top50 ID、顺序、分数与全509 sparse参考完全一致，无神经调用；Dense/Hybrid同向量和scope的等价性另用功能夹具核验，不冒称已实测完整simple神经索引。

避免4933个多余向量行，但其中有已有缓存；预检时实际尚未缓存4566个文本、2,230,720正文tokens，不把gross数量全部当冷编码，也不承诺节省固定时长。神经实跑尚待交接。冻结源位于`runs/20261014_v2_simple_scope_preflight`。root额外36项冻结/读取/统计接口测试PASS，日志`simple_scope_integration_contracts.log`。不更改当前structured进程、问题数、切块选型、扩展、模型或评价规则。

### 2026-10-14：结构化CPU编码低内存暂停与一次受监督恢复备选

19:22–19:26Z主机可用RAM持续降到0.59–0.76GiB。精确核验后仅暂停自身worker24020，排空单slot CPU embedding服务并退出36892，保留5088条已写观察及全部缓存。worker仍驻留内存且持有锁，可能有一条尚未消费的HTTP响应；这不是完整持久进程断点。`STRUCTURED_LOW_RAM_PAUSE.json`保存动作、SHA和限制，历史连接恢复仍2/3；后续若reset才消耗剩余一次，不凭暂停动作假定已失败。

`RESUME_RAM_GUARD_PROVENANCE.json`证实3GiB源于此前恢复脚本的保守工程条件，不是任务书、模型或线程性能identity。初始服务有效guard为1GiB请求量加0.5GiB余量；原receipt继承这两个字段，不能冒充后续脚本没有检查3GiB。历史稳态RSS约1.28–1.31GiB，但没有可靠加载峰值，host前后差值不当峰值或独占模型占用。

优先≥3GiB原恢复路径。若释放本项目模型后仍在2.5–3GiB，允许一次受监督加载备选：模型/量化/CPU12/微批/上下文参数全部相同；只在worker保持暂停时加载和健康检查，持续采样host RAM，跌破1GiB或达到原加载/请求时限时仅退出精确自有stateless服务。任何worker.resume前先结束并join监测器、复核RAM≥1GiB与worker身份；恢复成功之后的异常必须保留服务，不误走加载失败清理。备选启动前独占保存尝试、配置与源码哈希，不再降低门槛或无限重试。这个方案改变的是资源启动控制，不改变实验输入或质量标准；真实加载与续跑成功仍需单独记录。

19:45:58Z该唯一备选实际成功：服务19112，9个加载采样最低host1,744,457,728字节，真实8-token健康请求得到1024维；监测线程已join后才恢复旧worker。此后旧TCP连接WinError10054使worker24020退出1，日志保留；服务保持健康，按第三/最后一次有界缓存重放启动20816/session22849，启动源码哈希实测保存。19:50:21Z已写5120/6551，超过旧5090条观察。新控制台从前面重新读取缓存不是旧成果丢失。加载阶段峰值不当作后续全部正文请求峰值，运行期间仍为外部观察。代码、5项行为测试与模块记录提交b71a51e。

### 2026-10-14：用户8小时硬截止下的依赖并行

03:21UTC实际资源核验host3.86GiB、GPU12MiB/0%、原R3/aux进程全部身份匹配后，允许原22次4K/8K诊断先与两项CPU4任务重叠；仍由gen保持单卡串行及有效3GiB启动guard。服务启动发现系统Ollama已经0.40.3，与原0.34.4冻结身份不符，0请求即正常退出。使用官方v0.34.4 Windows portable ZIP到D专用目录，SHA/大小和完整82文件校验通过；exe SHA与原pilot/E3实际0.34.4二进制完全相同，不修改系统Ollama、原11434或冻结身份，不混新版库。实际恢复后的生命周期、资源与每次请求另外留证，下载/安装不计作模型实验。4K诊断已真实8/8加3预热、0失败，显式卸载后才执行8K。最终报告必须保留并行资源背景。

用户随后明确“不一定非得一致”，允许使用新版后端，覆盖任务书中由执行器推导的严格后端一致性要求。已有模型blob可复用，不重新下载模型；上述下载仅固定版运行程序。收到该授权时旧版隔离程序与全部22压力请求已经真实完成，继续已有可用旧版最少新增工作；若后续旧版出现真实阻断可迁移新版并保留后端差异、旧原始结果与迁移预检证据，不能将新版冒称原冻结版本。该放宽不要求现在再切换一次，不改变问题、失败分母或质量口径。

路由训练的旧读取入口先对整份生成清单做验证，再取train/dev，因而也会打开E2测试输出文件，虽然其质量从未参与训练标签或超参。现在只对请求的实验文件读取/验证，train/dev入口仅打开ROUTE_TRAIN/ROUTE_DEV；完整报告保持原完整扫描。用存在但不可读取的test journal做负向夹具：训练边界仍拒绝不完整train结果，且从不读取test，完整报告路径仍读test而触该夹具。50项分析/分片/路由/在线相关契约PASS，0模型调用；不改变任何标签公式、协议、已冻结记录或模型选择。生产完整性审计曾输出模型摘要内质量字段（gen已记录），没有据此调参；最终不能写成所有操作者从未见过任何test字段。

用户在暂停/恢复并切换会话模型后新增“只有8h，在这之内必须完成”。记录截止为10:29:06UTC/悉尼21:29，EXECUTION_DEADLINE.json；不改变任务范围、冻结问题、模型、prompt、失败分母或CPU4要求。原CPU全库6551向量、900结构化检索、520路由上下文已完整完成，R3前25正式题平均17.18秒，表明继续按全部CPU→全部GPU串行等待会浪费可用时间。

使用已有严格外部分片来源机制实现E2先行：只要R3 test200及其他三路径完整，冻结原2400核心E2，输入/任务/身份与未来main逐项一致；最终main同时引用不可变E2与E3，不复制改写、不重跑。该工程准备尚不代表E2运行。不同GPU模型仍串行；与余下CPU R3重叠时记录资源条件，不能声称独占机器性能。

SQuAD与scoped-simple仅先做独立索引预编码，使用原模型和最终index身份/路径；只有已核验独立索引锁与独立记账、没有主campaign双写时才执行。R3释放主writer锁后再物化最终检索结果。Laya移到实际可用资源窗口，不为了截止降低其已签收内存底线或改变SDK算法。所有新增入口需要必要契约检查及真实运行凭据；工程支持不能代替最终完成。


2026-10-14 最终交付覆盖：用户允许任意阶段GPU，单卡串行；原任务USER_MANUAL_RESUME。在线设备错误通过逐请求核验当前boot lease/私有服务PID/create/身份的CPU→remote GPU通道修复；CPU空CUDA与默认禁止GPU不变，3次旧拒绝预热/原源码/manifest精确保留。真实online60+6完整0error，提速5.0446%不达10%门槛，固定Hybrid/neighbor/2B默认不改。LoRA同BF16配对100+100完整：F1下降、P50升至3.62倍；不重训不启用adapter，原模型OS exit UNKNOWN而独立audit actual0。UI两条真实问答后发布配置，所有本项目CPU/GPU/UI进程和锁已释放，GPU12MiB是既有Codex。最终full711不同PASS/1Windows symlink权限skip，220源稳定。原断点仅验证历史租约哈希链，WSL kept off；没有自动起原训练/定时/日更，没有改.wslconfig，不能宣称原日志继续前进。普通应用重启及正常关闭命令见根LOCAL_RUN_V2.md；历史once-shot online wrapper不作为通用重启入口。完整失败账本保留，轻量包采用有原hash的全任务投影，不上传/push。
