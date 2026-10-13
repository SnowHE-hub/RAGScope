# 决策记录

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
