# 主生成与 LoRA 的重启后恢复点

> 后续进展（03:34 UTC）：旧租约已由root重新签发，固定stress22已经真实COMPLETE并卸载。恢复了官方portable0.34.4到D盘，服务当前READY空载；不要重跑stress。下一执行边界已改为R3 TEST200封存后执行独立E2，不再等待其余R3全660退出。最新条件与精确命令以GENERATION_HANDOVER.md和E2_CORE_SHARD.md为准。本文下方1421预算与旧资源状态是重启恢复时的历史审计快照。


2026-10-14 的轻量预检没有启动模型、服务或联网。主代理报告主机于当日07:03 Sydney重启；所有旧进程与资源收据只保留作历史证据。原任务维持 `USER_MANUAL_RESUME`，不恢复原作业或日更，不改变 WSL 配置。

证据目录：`D:/RAGScope_Runtime/runs/20261014_v2/recovery/generation_post_reboot_20261014`。`audit.py` 是可重复的只读实验预检，输出仅在该目录；`PREFLIGHT.json` 保存来源哈希、分片导入演练、调用账本、时间计算及限制。`GPU_LEASE_IMPLEMENTATION_AUDIT.json` 绑定50项CPU测试和真实旧租约拒绝收据。旧handover及旧pilot源码保存在其 `provenance` 中。

## 已核实的边界

- E3封存18个artifact、1200条原始记录哈希和对应调用票据全部匹配。没有请求失败，全部实际token一致。E3已完成，不能重复执行。
- 默认主freezer会引用原E3 manifest artifact，跨分片保留原config/hash，外部分片拥有的任务不再在主目录运行。使用内存中的E3父清单验证了完整导入接口；真正主manifest尚不存在，最终全矩阵导入仍须在freeze时验证。
- dev选择已经固定为 `F512_O0` / `R512`，两者均为 `neighbor`。等待的是全库检索来源与R3、SQuAD等产物，不能退用旧 `_none` 记录。
- LoRA正式训练已完成122步。best/latest均为step122，完整checkpoint校验通过，adapter SHA为 `a532dc22942d84e4dceabacb64b936c191b37ebdfa9fda092f3f87464d0bbae4`。无需再次训练。CPU check-only当前返回2：`WAITING_FOR_FROZEN_GENERATION`。
- 4K/8K压力实验的两个实现文件、冻结manifest和全部来源哈希未变。仍为8题×2窗口及6个warmup，共22次，尚未请求模型。

## 当前启动周期租约

新增统一入口 `ragscope.experiments.runtime.validate_gpu_lease(path)`；仅校验操作者签发的凭据，不替代原作业安全核验或获取租约。

在原有 `status=ACQUIRED, owner=ragscope, safety_verified=true, gpu_device=0` 上新增：

| 字段 | 当前值生成方式 | 校验 |
|---|---|---|
| `host_platform` | `sys.platform`，Windows为`win32` | 必须与当前进程平台一致 |
| `host_boot_time_unix` | `psutil.boot_time()` | 有限int/float，距当前启动时间≤2秒 |
| `validated_at` | `time.time()` | 有限int/float，不早于当前/凭据启动时间，不晚于当前时间 |

拒绝bool、字符串、NaN、Inf、缺失字段与旧启动周期；不会回填旧租约。训练、项目服务、正式生成/UI共用服务入口、pilot、PowerShell `-Gpu` 均采用同一Python校验器。CPU路径保持4线程，不新增模型依赖。冻结stress实现及旧实验收据不变。

主代理归档旧租约、只读核验并重新签发后，可执行下面的纯CPU验证，不启动CUDA或模型：

```powershell
. ./scripts/runtime_env.ps1
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/validate_gpu_lease.py --lease D:/RAGScope_Runtime/runs/20261014_v2/gpu_lease.json
```

成功为0和`VALID_CURRENT_BOOT`；拒绝为2。旧真实lease已实测被拒绝，原文件哈希未变。PowerShell拒绝后不会设置GPU设备；Windows PS5把被清空的环境变量显示为null，父CPU Python观察值为空串。

主代理随后已归档旧租约并签发当前启动周期租约，SHA为 `884715ec4556624cd069b461b1b9cfc3483f0bab32f6f77c6cf2165f6d6b0cac`，真实validator成功记录位于主run的 `GPU_LEASE_REVALIDATION_20261014.json`。新凭据记为 `NOT_RUNNING_OBSERVED_WSL_OFF`，恢复仍由用户手动完成，通过previous_lease artifact保留原历史。实际stress仍等待单独资源交接。

## 接下来的精确命令与前置条件

先等待主代理明确的 **R3进程退出后资源交接**。收到交接后重新核验当前租约、端口、进程、有效3GiB可用RAM。只在原项目服务已退出且11435可用时，归档项目自己的 `generation.stop` 到唯一历史路径，再启动 `scripts/serve_generation.py`；不可操作旧PID或其他服务。具体生命周期由主代理当前资源证据驱动。

```powershell
. ./scripts/runtime_env.ps1 -Gpu
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/serve_generation.py --binary D:/RAGScope_Runtime/tools/ollama-0.34.4/ollama.exe
```

项目11435服务READY后，在另一执行句柄串行运行以下两条件。两条之间显式卸载2B、核验API无resident model与当前RAM；最后正常卸载、停止自有服务，核验进程退出及GPU锁释放，交回Laya窗口。

```powershell
. ./scripts/runtime_env.ps1 -Gpu
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/run_context_stress_v2.py --mode run --window 4096
# 卸载并核验后：
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/run_context_stress_v2.py --mode run --window 8192
```

待全部最终来源完成，CPU冻结主矩阵。该命令会校验真实模型包与pilot/tokenizer来源，耗时可能高于本次轻量预检；仍不请求模型。

```powershell
. ./scripts/runtime_env.ps1
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/freeze_generation_v2.py --campaign 20261014_v2
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/evaluate_lora_v2.py --training-run D:/RAGScope_Runtime/runs/20261014_v2_lora/train_2b_1024_seed42 --campaign-dir D:/RAGScope_Runtime/runs/20261014_v2_generation --model-dir D:/RAGScope_Runtime/models/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc --check-only
```

只有freeze成功且新项目服务READY、当前租约有效时，正式主生成才运行。单卡串行，以下命令每次只选一个model-id；每个模型完成后先卸载并核验资源，再运行下一个，不并发。

```powershell
. ./scripts/runtime_env.ps1 -Gpu
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/run_generation_campaign.py --campaign-dir D:/RAGScope_Runtime/runs/20261014_v2_generation --model-id qwen_2b
# 后续分别使用 qwen_4b、qwen_9b、lfm；不得重跑E3分片命令。
```

BF16评估须在项目Ollama退出、GPU锁可用且另获资源交接后，执行同一已训练adapter的100题成对比较：

```powershell
. ./scripts/runtime_env.ps1 -Gpu
& D:/RAGScope_Runtime/envs/train/Scripts/python.exe scripts/evaluate_lora_v2.py --training-run D:/RAGScope_Runtime/runs/20261014_v2_lora/train_2b_1024_seed42 --campaign-dir D:/RAGScope_Runtime/runs/20261014_v2_generation --model-dir D:/RAGScope_Runtime/models/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc
```

两BF16条件共200正式题及至少6warmup，失败保留。BF16均用prompt-only JSON，与Q4主矩阵的schema约束不同，不能将跨协议差异当作纯精度效果。后续online和support judge仍要求冻结输入、明确服务就绪及同一全局预算，不能用已实现CLI冒充真实完成。

## 剩余调用量与实测耗时外推

本次账本为1421/10000，核心已用1200，余8579；其中2400仍保护核心E2。所有1421票据连续唯一，E3票据逐项相符。

| 剩余主组 | 正式请求上限 |
|---|---:|
| E2（3模型×4方法×200） | 2400 |
| E4 | 300 |
| E5 LFM | 200 |
| E6 | 400 |
| ROUTE train/dev标签 | 780 |
| E7（最多40题） | 240 |
| 合计 | 4320 |

加主模型首次4×3warmup、stress22、BF16评估206、online66、judge最多200，尚留3753调用给有界恢复等；这不是增加组合的许可。E7实际源可行题数在最终freeze决定，可能少于40。每次服务/客户端会话产生的新warmup和失败均继续计数。

E3 oracle每模型200题的实测均值/P95如下（秒，包含管线遥测收尾）。平均实际输入约355token，明显短于正式检索上下文：

| 模型 | 平均单次 | P95单次 | 主矩阵剩余次数上限 |
|---|---:|---:|---:|
| Qwen 2B | 0.989 | 1.683 | 2000 |
| Qwen 4B | 1.414 | 2.741 | 1220 |
| Qwen 9B | 2.179 | 4.705 | 900 |

4120次Qwen部分的情景外推约 **1.7～6.0小时**。低端使用1024实际输入token及E3中位数prefill秒/token与其余请求耗时；高端使用3584实际输入token及两项各自P95，按模型剩余次数加总。计算明细保存在PREFLIGHT，不是置信区间或承诺完成时间；prefill随长度线性变化是显式假设，输出长度也可能改变。

此范围不含LFM200、BF16评估206、judge200、online66、stress22，也不含冷启动、重试、检索/R3/Laya准备、核验落盘和资源排队。主机重启后的负载、共享显存与offload可能改变速度；没有用训练步速估计BF16推理，也没有给这些尚未真实运行的项目虚构耗时。
