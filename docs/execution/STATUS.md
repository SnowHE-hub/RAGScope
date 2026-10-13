# RAGScope v2 执行状态

更新时间：2026-10-14（Australia/Sydney）。campaign_id：`20261014_v2`。

当前：真实CPU Dense/Hybrid矩阵运行中；四模型最终生成预检完成并保留LFM输出截断失败；训练准备进行中。尚未最终验收，不得宣称核心E2/E3完成。

- 已完整读取 MASTER_PLAN.md；原件与副本 SHA-256：`157040e71eab86b44bb3007bf43c1ba25240403b892a7cd5f93c1fb37e280117`。
- 项目身份已核验；分支 `codex/ragscope-v2-autonomous-20261014`。
- 原 main 源码、HEAD、状态和 tracked diff 在 `D:/RAGScope_Runtime/backups/20261014_baseline`。
- 原始用户未跟踪文件保留，不纳入本轮提交。CodeGraph 已初始化。
- 阶段一/二严格CPU-only，原GPU程序在这两个阶段未被停止、暂停或修改。阶段三按用户追加授权开始安全交接（见RECOVERY与运行目录gpu_lease.json）。
- 机器：Windows 11、32逻辑核、约15.3 GiB RAM、RTX 5060 Laptop 8151 MiB。
- 初始剩余磁盘：D 232345247744 bytes，C 60073107456 bytes。空闲 RAM 初测约2.4 GiB，加载前重检。
- 并行实现：清洗/切块与结构、QASPER/技术数据、生成/评测正确性；主执行器负责隔离环境和可恢复账本。

新增已验证：
- Python3.12.13解释器和core环境实际base_prefix均在D；74包哈希锁安装成功。
- QASPER509论文，train1000/dev100/test200、global100；技术3页32例；数据复验与153MiB峰值证据见DATA_REPORT。
- 清洗/切块127测试PASS，确定性核心覆盖98%；生成/评测49测试PASS，覆盖97%。
- runner真实子进程中断→半写入隔离→恢复→重复执行测试PASS；ArrayIndex/caches+runner 7测试PASS。
- Qwen3 embedding Q8_0（639150592 bytes）与官方CPU llama.cpp b11429已下载校验；CPU服务11436实际运行，Dense缓存逐条落盘。
- Windows RAM仍紧张，大型安装/下载与GPU预检已改为串行。独立生成服务触发低于512MiB保护时卸载并退出，失败请求保留。
- 新账本主campaign已实际运行S1.1/S1.5；E0旧BM25子campaign完成。

已补充验证：
- 最新统一测试285 PASS、1项Torch环境测试单列；独立models环境路由24项PASS（含真实小型LSTM训练/恢复），覆盖92%。旧BM25实际200例，Recall@5=.845、Recall@10=.905，保留与旧CSV差异。
- 11组dev切块+上下文策略1500条真实BM25；后续test上下文400条与E8全库100条，合计2000条，失败0。
- E1独立重算33个数字全部一致；逐行context真实规范文本坐标核对1100条通过。
- Dense三次初始CPU内存预检拒绝加载；资源交接后恢复。首组F256_O0 Dense/Hybrid各100条PASS，完整证据覆盖分别65/92与56/92；其余10组仍在运行。后续全库索引仍待完成。
- 官方Qwen3.5 2B/4B/9B Q4_K_M权重已校验；纯文本别名只省略未使用视觉投影器，语言权重不变。2B旧缓存设置20/20成功；4B最终禁prompt RAM缓存设置20/20成功，token均与后端一致。2B须按最终缓存身份重做，9B待预检。过去内存保护失败保留。
- 已定位默认8192MiB后端prompt RAM缓存增长，独立服务设置LLAMA_ARG_CACHE_RAM=0；实际日志确认禁用，4B20连续调用不再触发低内存保护。仅修改本项目服务。
- CPU-only BGE固定版本权重已就绪，尚未加载；BF16 Qwen3.5-2B官方12文件约4.57GB已校验，独立train环境38包安装/锁定/pip check完成；尚未GPU训练。
- LFM2.5-1.2B官方Q4_K_M权重730895168字节已下载，SHA与官方HF一致，真实预检待运行。
- 新增逐行生成分析模块与预声明对照configs/v2/analysis.json；8项完整性/失败分母/论文聚类测试PASS。无正式生成结果时不输出最终估计。
- 最新用户明确：训练与定时检查已停，日更未动；结束后用户手动续跑。原任务不再自动恢复，脚本restore已保护。
- 本地里程碑提交：9c43163（基础/数据）、496354d（证据检索/生成正确性）。不push。

当前恢复点（2026-10-14 02:57，本地时区）：
- 最终提示rag-json-v2.2、可见引用enum统一约束已冻结；四模型各20次实际dev请求，输入token80/80一致，逐请求资源80/80实测。Qwen三个规格均20条解析成功；LFM19/20，1条512输出token截断错误保留、不挑选重试。所有历史212次调用、5条错误均保留，正式核心调用仍0。
- 自有Ollama PID17200已按停止标记正常卸载退出；11435关闭、显存0MiB。GPU已交训练代理，正在准备真实BF16训练输入；root不得同时启动GPU服务。
- CPU embedding PID39808、Dense worker PID43932（creation1791906175.5201824）继续12线程、GPU offload0运行。进度见DATA_RETRIEVAL_CHECKPOINT.md；4/11 dev切块组Dense/Hybrid已完成，后续全库索引仍待计算。
- CPU BGE reranker真实50→10预检通过，278044417参数、FP32 CPU，见preflight_bge_cpu.json；不等同已完成R3语料实验。
- 独立数据审计24个已完成运行、2600条、11821片段、72个汇总数字均通过。
- runner工程账本8 PASS、S2.3 RUNNING、16 PENDING；本地离线CI等价执行346 PASS、1项Torch环境测试单列。CI只提供手动工作流，不上传。
- 本地新增提交af94414（真实逐行分析与失败分母/聚类统计）；尚未最终验收。

下一条准确命令：`. ./scripts/runtime_env.ps1; & D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/audit_model_pilots.py`（只读实际结果汇总，无GPU调用）。训练代理持有当前session33605，完成prep后自行顺序执行2B BF1610步/断点恢复，详见TRAINING_REPORT.md。检索worker37932继续运行；禁止重复启动。

2026-10-14 03:12恢复点追加：CPU Dense worker43932已SUSPENDED，CPU服务39808已安全退出，2404缓存向量保留；勿沿用上文RUNNING描述。旧2B LoRA10步真实成功且5→10精确恢复，因发现PEFT初始化前seed未设而保留为可行性证据，正式同路径seed42修复新run正在训练代理手中重做预检并续122步（不增加模型/超参选择）。原任务手动恢复约束不变。本地提交3b49ac4。下一准确命令：`Get-Content -LiteralPath D:/RAGScope_Runtime/runs/20261014_v2_retrieval/PLANNED_TRAINING_RAM_PAUSE.json -Raw`；恢复CPU服务前必须与训练RAM状态协调。

2026-10-14 03:24追加：正式seed42 LoRA运行中（session72907，最近确认step44/122、全部finite）；E0固定revision重建CPU进程42092（session54394）已实际编码2304/22254条，RSS约658MiB，hostfree约2.56GiB，4threads/1.25GiB安全暂停阈值。大CPUembedding仍停、worker43932暂停，不可重复启动。最新core统一372 PASS、1项Torch隔离测试skip；后续反向数字审计2项fixture测试PASS。ROUTING_QUERY_MANIFEST.json冻结train200/dev60/test100，SHA身份f1b6ae1ef21a44b720b53e1a6fffe6600a12d16d51e4b88aefe26a69c0ff952b；dataagent补BM25/Hybrid源，chunkagent补R3源。
