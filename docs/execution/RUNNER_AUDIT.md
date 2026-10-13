# 统一执行器与任务证据账本

日期：2026-10-14，Australia/Sydney。任务书实际含25项：S1六项、S2六项、S3七项（含S3.7）、S4六项。

`configs/v2/tasks.json`覆盖全部25项，区分实际任务命令、离线功能验证和仍需真实实验的验收条件。`full --resume`只有全部请求任务完成才返回0；有PENDING/RUNNING/FAILED/BLOCKED时返回2。`--reconcile-only`校验并记录证据，不启动实验。Windows帮助入口已实际运行。

主campaign原有`state.tasks`执行身份不被对账改写；新增`plan_tasks`、`plan_status.json`和追加`plan_events.jsonl`。每份子实验凭据先验证config签名及输出文件大小/SHA，再以内容寻址文件保存在`reconciled_evidence/`，保留原始观测指标、失败数和分母。旧子campaign的汇总PARTIAL_BLOCKED不会把仍待运行的依赖永久标成BLOCKED。

正在运行的任务同时核验writer锁和`worker.json`：PID、创建时间、完整argv、cwd须与当前进程一致，避免误认复用PID。2026-10-14本机核验CPU矩阵worker43932对应真实创建时间1791906175.5201824，故S2.3为RUNNING，即使writer锁处于未持有间隙。未来进程退出后，状态会按剩余证据回到PENDING而不虚构完成。

新执行使用`PLAN_<task>_<config-hash>`身份，绑定源文件、声明输入、前置文件和任务规格。完成后的源输入变更产生新身份；旧记录继续保留。失败最多3次；同一配置的未就绪检查也有界，前置文件变化后可重新核验，不把等待当成永远失败。功能辅助程序退出0不能自动完成真实实验。

命令显示明确的依赖、缺失文件、解释器、GPU启用、外部资源交接和活跃子campaign前置条件。GPU命令只调用现有租约验证，不停止、启动或恢复原用户进程。恢复策略固定`manual_user_only`。调用者已完成资源交接后可传`--coordinated-task Sx.y`；GPU还要求`--allow-gpu-tasks`和有效租约。默认始终CPU-only。

实验/多项交付的完成凭据位于`runs/<campaign>/completion/Sx.y.json`。必须绑定task_spec_hash、全部requirements_met及项目/runtime内可验证artifacts。真实实验还必须有real_experiment=true。BLOCKED/FAILED凭据必须有具体原因。它是对实际证据的签收，不允许用接口存在代替运行结果。

2026-10-14工程签收已逐条核对MASTER：S1.2、S1.3、S1.4、S1.6、S2.1、S2.2；审查明细为`D:/RAGScope_Runtime/runs/20261014_v2/completion/ENGINEERING_REVIEW.json`。加原S1.1/S1.5，主账本8 PASS、1 RUNNING、16 PENDING。未把S2.3矩阵、S2.4重排或后续正式生成/训练标成完成。S2.2证明固定CPU权重与实际256/512/1024正文输入对应后端257/513/1025 tokens；早期逐题记录只有body token计数的事实明确保留。

本机验证：初始26项runner/catalog测试通过；随后加入live PID凭据和惰性retriever导入后，runner/catalog/indexing共48项通过，11.81秒。新增CI配置为手动触发；本地等价`compileall`退出0，离线单元346 passed、1项隔离Torch测试skip，52.44秒；该Torch项另在models环境实际通过，未推送或触发远程工作流。日志为`D:/RAGScope_Runtime/logs/ci_equivalent_cpu_tests.log`和`runner_lazy_import_validation.log`。

WSL限制：`scripts/run_v2.sh`指向单独`/mnt/d/RAGScope_Runtime/envs/core-linux/bin/python`，该环境目前不存在，尚不能声明WSL完整入口已验收。不能把Windows的Scripts/python.exe当成Linux虚拟环境；未为此修改或重启WSL。

只读/对账命令：

```powershell
. ./scripts/runtime_env.ps1
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -m ragscope.experiments.status --campaign 20261014_v2 --details
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -m ragscope.experiments.run --profile full --campaign 20261014_v2 --resume --cpu-only --reconcile-only
```

下一项独立工作：隔离的E0 Dense/Hybrid真实复跑。原BM25已完成200题；原Chroma缺历史模型revision，必须保留未知身份事实。不得运行原`rerun/rerun.py`覆盖旧报告；若无法确认历史权重，重建相同legacy chunks的固定官方权重对照并标`legacy_reconstructed_fixed_revision`。

后续资源交接：矩阵worker明确标SUSPENDED且身份仍匹配时，逻辑任务为PENDING并带resource_wait=true，仍阻止重复启动；即使原writer锁仍由暂停进程持有，也不误标RUNNING。该分支实际离线测试通过。最新对账8 PASS、17 PENDING，原计划及失败记录仍保留。工程凭据因新增核验/文本换行变化需重新签收时，将旧凭据存入completion/history内容寻址文件，不删除旧版本。

2026-10-14包装入口实测：PowerShell新增-Gpu、-ReconcileOnly和-CoordinatedTask，默认CPU mask；-Gpu仍要求已核验租约，不自动改动服务或原任务。实际Windows只对账返回native exit2，25任务中8 PASS、其余进行/等待，未假装验收完成。WSL实际bash语法检查exit0；执行CPU入口因core-linux缺失明确exit2，此为失败路径验证，不是Linux端到端运行通过。两种入口均使用D盘缓存路径；Linux新增MPLCONFIGDIR。完整跨平台模型运行仍以Windows实证为准。
