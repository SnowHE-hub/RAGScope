# E2核心分片先行

为满足 `EXECUTION_DEADLINE.json` 的2026-10-14T10:29:06.606796Z硬截止，E2允许在辅助实验来源尚未齐全时独立冻结和执行。范围仍是原计划的三Qwen×四方法×200题=2400次正式请求，没有新增模型、题目或实验组合。E3已完成1200次，不重跑。

新增 `scripts/freeze_e2_v2.py`，输出目录为 `D:/RAGScope_Runtime/runs/20261014_v2_generation_e2`。复用主矩阵的 `verified_context_rows`、`shared_context`、`freeze_campaign`，保持相同query、task/input ID、所有原始来源字段、prompt v2.2、schema引用约束、greedy512及三个真实聊天模板共同上下文预算。分片不嵌套其他分片。

## 唯一待齐的核心来源

冻结QASPER全200题及三主模型最终pilot/模型包/分词器仍须通过原有校验。dev选择固定 `R512` / `neighbor`。四个来源必须恰好同一200个test ID，无重复、无错误行，RecordStore config/record哈希有效，question/scope/split与冻结语料一致。

| E2方法 | `20261014_v2_retrieval/records/`下来源 |
|---|---|
| R0 | `TEST_CONTEXT_R512_bm25_neighbor.jsonl` |
| R1 | `TEST_CONTEXT_R512_dense_neighbor.jsonl` |
| R2 | `TEST_CONTEXT_R512_hybrid_neighbor.jsonl` |
| R3 | `TEST_CONTEXT_R512_hybrid_rerank_neighbor.jsonl` |

每个文件须附同名去掉`.jsonl`后的`.config.json`。R3代理将TEST200组完成、summary落盘和单组验收通过后主动通知；不等待其余global/ROUTE完成，不反复轮询其工作句柄。首次真实检查确认R0/R1/R2各200已就绪，R3待完成；保存在分片 `SOURCE_READINESS.json`。此检查零生成、零模型/分词器加载。

```powershell
. ./scripts/runtime_env.ps1
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/freeze_e2_v2.py --campaign 20261014_v2 --check-only
# 四组来源全部通过后，执行完整模型包/pilot/tokenizer及来源冻结验证：
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/freeze_e2_v2.py --campaign 20261014_v2
```

check-only为`SOURCE_RECORDS_READY_MODELS_NOT_CHECKED`仅表示来源记录就绪。完整freeze必须返回`FROZEN`、tasks2400、per_model800。缺组、少题、多题、模型缺失、来源错误不会缩小分母，也不会创建可执行分片manifest。代码不会读取SQuAD/simple/E7/ROUTE来源。

## GPU窗口与实际执行

当前E2已冻结，manifest_hash为`6a2862e3fed696fb0224d23abc78437c5eaa318efb190c13b2422724ca3246e5`，2B正式执行中。实际当前状态以运行目录的`OPERATOR_CHECKPOINT.json`为准；不要对正在运行的服务重复启动。系统Ollama已更新，冻结实验使用的官方0.34.4便携版在D盘。仅在自有旧服务已退出、11435可用、旧stop标记已归档及资源/当前启动周期租约校验通过后，使用以下精确启动命令：

```powershell
. ./scripts/runtime_env.ps1 -Gpu
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/serve_generation.py --binary D:/RAGScope_Runtime/tools/ollama-0.34.4/ollama.exe
```

等待主代理明确资源交接。当前启动周期lease校验、有效3GiB RAM门槛和自有单模型11435服务规则继续适用。22调用stress可在同一次GPU窗口中先串行完成，然后正常卸载其模型，再开始E2；stress的冻结源码、输入和语义均未修改。

R3其余组若继续CPU4运行，主代理在E2分片 `EXECUTION_CONDITIONS.json` 记录实际并行CPU进程身份/线程及启动RAM快照。逐请求仍实测host available、项目服务RSS及设备总显存；这些是采样极值，不是进程显存峰值。延迟解释须注明与E3运行时的并行负载差异，不能把不同主机负载算作模型质量或检索改进。

自有服务READY后，按2B→4B→9B顺序执行，每个模型完成后显式卸载并核验API无resident model、GPU/RAM再继续。以下命令每次仅执行一条，不并发：

```powershell
. ./scripts/runtime_env.ps1 -Gpu
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/run_generation_campaign.py --campaign-dir D:/RAGScope_Runtime/runs/20261014_v2_generation_e2 --model-id qwen_2b
# 2B卸载及资源核验后：
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/run_generation_campaign.py --campaign-dir D:/RAGScope_Runtime/runs/20261014_v2_generation_e2 --model-id qwen_4b
# 4B卸载及资源核验后：
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/run_generation_campaign.py --campaign-dir D:/RAGScope_Runtime/runs/20261014_v2_generation_e2 --model-id qwen_9b
```

每模型800正式请求及首个执行会话3次warmup，共2409次；恢复会话的额外warmup和失败也全部计入原10000总预算。核心E2的2400预算已经保护。复用RecordStore恢复，不能删失败、重复已落盘题目或在新目录重新调用。`run_generation_campaign.py`现在仅当该模型实际有E4任务才读取SQuAD语料，因此E2执行也独立于辅助数据。

```powershell
. ./scripts/runtime_env.ps1
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/accept_generation_v2.py --campaign-dir D:/RAGScope_Runtime/runs/20261014_v2_generation_e2 --scope e2
```

生产验收固定12组/2400题；失败仍计observed和失败分母，缺行显示PENDING/RUNNING。E2分片完成不等于主矩阵或项目完成。完成后释放自有GPU服务，等待主代理后续调度与deadline收尾。

## 主矩阵导入与防重复

`default_spec`和主freeze都会自动发现已存在的E3、E2 sibling manifest，按不可变artifact写入主manifest的`provenance.external_campaigns`，因此早于分片保存的显式spec也不会漏导。已有引用原样保留，不能自动更新哈希来接受被改写的分片。主freeze在写盘前验证分片task、完整input、model、generation/prompt/metric/seed语义一致，禁止重复、循环和嵌套。主runner跳过外部分片拥有的全部任务，即使分片暂未完成；缺行保持missing，不能补跑到主目录。两个分片都齐全后核心ownership是E3=1200、E2=2400、主目录=0。

主矩阵仍按原freezer等待并显式声明E4/E5/E6/E7/ROUTE等全部原定组。导入原记录与原hash，不复制重写旧实验记录或receipt。
