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

E0隔离重建已真实完成：Dense/Hybrid各200条、均0失败，全部22,254条CPU编码及原8个索引哈希通过独立复算核验。历史权重仍UNKNOWN，标签为`legacy_reconstructed_fixed_revision`；没有运行原`rerun/rerun.py`或覆盖旧报告。详情见`E0_RECONSTRUCTION.md`及`D:/RAGScope_Runtime/runs/20261014_v2_legacy_reconstructed/INDEPENDENT_VALIDATION.json`。E0完成不会替代S2.6其他必需实验。

后续资源交接：矩阵worker明确标SUSPENDED且身份仍匹配时，逻辑任务为PENDING并带resource_wait=true，仍阻止重复启动；即使原writer锁仍由暂停进程持有，也不误标RUNNING。该分支实际离线测试通过。最新对账8 PASS、17 PENDING，原计划及失败记录仍保留。工程凭据因新增核验/文本换行变化需重新签收时，将旧凭据存入completion/history内容寻址文件，不删除旧版本。

主执行器随后核验包装器：PowerShell增加`-Gpu`、`-ReconcileOnly`、`-CoordinatedTask`并拒绝CPU/GPU冲突；GPU分支检查现有租约。Windows实际reconcile返回2（8 PASS，其余PENDING/RUNNING）。WSL包装器`bash -n`返回0，实际入口因专用core-linux环境不存在而明确返回2；没有复用Windows venv或安装Linux环境。最新主执行器离线检查414 PASS、2 skip，跳过项分别为隔离Torch与Windows symlink权限，不能宣称这两项在本轮core环境已运行。

终验门槛复审又补两项：旧Sx.y成功执行不直接完成experiment/delivery；S2.3自动PASS前核每个正式run的100条唯一dev ID、逐行hash/config/index/CPU和真实神经模型身份、失败分母。BM25不要求神经模型标记。新36项runner测试PASS（7.48秒），已完成29个真实run/2900行额外审查通过；33组尚未全齐时不PASS。完整15条终验差距记录在`ACCEPTANCE_GAP_REVIEW.md`，不把25任务账本状态当最终验收。

2026-10-14包装入口实测：PowerShell新增-Gpu、-ReconcileOnly和-CoordinatedTask，默认CPU mask；-Gpu仍要求已核验租约，不自动改动服务或原任务。实际Windows只对账返回native exit2，25任务中8 PASS、其余进行/等待，未假装验收完成。WSL实际bash语法检查exit0；执行CPU入口因core-linux缺失明确exit2，此为失败路径验证，不是Linux端到端运行通过。两种入口均使用D盘缓存路径；Linux新增MPLCONFIGDIR。完整跨平台模型运行仍以Windows实证为准。

完整dev矩阵到齐后，S2.3已实际通过33组×100条逐行核验（BM25/Dense/Hybrid各11组，0失败），主账本更新为9 PASS、16 PENDING。S1.3的旧凭据因已提交UI新增E3真实结果路径而失效；重新运行chunking/UI共134项PASS（8.59秒），将旧凭据及旧工程审查存入内容寻址history后签收新配置。日志`runner_full_dev_matrix_reconcile_revalidated.log`和`engineering_s13_ui_update_revalidation.log`保存实际输出。后续UI尚在计划内更新，待稳定后统一签收；矩阵完成不等于test/global/ROUTE上下文及R3已经完成。
