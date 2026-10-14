# 恢复记录

campaign：`20261014_v2`；runtime：`D:/RAGScope_Runtime`。

**最新用户指示：原训练和定时检查均已停止，日更进程未动。RAGScope结束后释放自己的资源并通知用户，由用户手动续跑。禁止自动执行恢复命令。** 用户确认XGB63已完成6/9折，LGBM63已完成3折；原结果保留。租约`restoration_policy=USER_MANUAL_RESUME`，脚本restore已加拒绝保护。下面的历史恢复命令只作审计记录。

阶段三原任务交接：用户追加明确授权可终止CPU/GPU进程并希望恢复。实际操作状态唯一来源是 `D:/RAGScope_Runtime/runs/20261014_v2/gpu_lease.json`；不要只依据本文推断进程是否仍运行。

已识别Ubuntu中QuantMind的两个T-030研究队列：XGBoost 63d子PID114865（start_ticks10435121），以及LightGBM 63d子PID114866。两者完整父进程身份、cwd、argv、日志、恢复命令和检查点哈希已写入租约。没有修改原项目代码或断点。

检查点语义是已完成的配置/种子/折单元可恢复，未完成单元重算，不能宣称可恢复到中断瞬间。停止前核验83个XGB单元、24个LGBM单元；所有JSON可解析，NPY头与数据长度匹配，UBJSON和Parquet边界完整，文件逐个SHA256记录。停止后再次验证所有已完成文件未变。

实际安全停止：身份逐一匹配后先向队列父进程发SIGTERM防止推进下一任务，再对训练子进程发SIGINT。XGB以KeyboardInterrupt退出；LGBM的`/proc/status`显示SIGINT被忽略（SigIgn末位7），55秒后仍运行。依据用户追加的CPU/GPU终止授权，再次复验断点后仅对PID114866发送SIGTERM，未使用SIGKILL。未完成单元重算；不能宣称这个进程进行了中途保存。两任务已退出，全部先前完成文件哈希不变，租约现ACQUIRED。

历史自动恢复入口仅供审计留存，已经被最新USER_MANUAL_RESUME政策禁用，不执行：`wsl.exe -d Ubuntu --exec python3 /mnt/d/MCS_26S1课程资料/个人项目/RAGScope/scripts/manage_previous_jobs.py restore`。本轮终验应确认RAGScope资源释放并通知用户，状态记录为等待用户手动续跑，不能声称原日志已经推进或原训练已经恢复。

源码基线：`D:/RAGScope_Runtime/backups/20261014_baseline/tracked-source.zip`；HEAD、tracked.patch、未跟踪状态在同目录。不要用 reset/clean 覆盖用户文件。

本项目当前恢复入口：在原项目目录运行 `. ./scripts/runtime_env.ps1`，然后按ROOT_CHECKPOINT及各模块检查点继续准确的独立命令。不要假定其他聊天状态，也不要重复启动尚活跃的worker。

运行入口已实现并实际使用：`./scripts/run_v2.ps1 -Profile full -Resume -CpuOnly`。查询状态：`. ./scripts/runtime_env.ps1; & D:/RAGScope_Runtime/envs/core/Scripts/python.exe -m ragscope.experiments.status`。

中断恢复性质测试：`tests/unit/test_runner_v2.py`通过，专用人工fixture以exit99退出，恢复隔离半写入尾行并保留全部5条唯一记录，再次执行文件字节不变。它不是模型实验。

独立子campaign：`20261014_v2_legacy`、`20261014_v2_retrieval`；分别持有自己的writer.lock。不得同时启动同一campaign两个写入者。

下一条准确命令：`Get-Content D:/RAGScope_Runtime/runs/20261014_v2/gpu_lease.json -Raw`

当前内部资源交接顺序：structured全库和ROUTE正常完成，核验退出后释放本项目embedding；R3五组完成退出；4K/8K共22请求压力诊断完成退出GPU服务；Laya有界运行；再归还CPU给SQuAD/simple索引。每次按实际PID、创建时间、命令及来源核验，仅操作本项目进程，不重启原11434、WSL或日更程序。

2026-10-14 12:39只读重查确认电脑最近启动为07:03:31.5+11:00；此前20816/19112均不存在，索引最后写06:59:17，5976/6551观察保留。启动原因未知，本任务未发出重启命令，不能推断是系统更新或用户操作。WSL当前没有运行的分发，Windows可用约4.42GiB；不为“释放缓存”启动或重启WSL。主run/HOST_RESTART_OBSERVATION_20261014_1239.json保存完整观测。原任务手动续跑政策仍生效，也不能把重启前的日更状态当成重启后的已验证状态。

本次按外部主机重启处理正常checkpoint重入：先核验缓存完整行、内容哈希、锁与端口，再启动同模型CPU服务和剩余工作。历史3次连接重置及上限不改写、不清零。新AGENTS要求后续CPU进程4线程，必须记录该性能配置变化，不能把历史12线程墙钟结果改名为4线程。恢复是否成功以新PID/create/argv、真实健康检查及新增落盘为准，不依据一次旧状态文件。

12:41实际启动CPU4服务6016和worker3312/session37283，原模型/二进制hash不变，真实1024维控制检查通过；root另重核服务PID/create/argv及health200。缓存重放保留5976条既有观察，HOST_RESTART_REENTRY.json与启动脚本保存新来源/性能配置。12:45重放仍在原断点之前，尚不能以恢复控制成功替代最终索引完成。

用户曾提出暂时减少WSL的10GB额度，核对后又明确“先不动那个wsl的内存上限”；**未写入任何WSL设置，也无需恢复配置**。原.wslconfig仍memory=12GB、swap=8GB、processors=12，SHA d995d866826cbfd36c261689ba27c1673108016d38fe8be20af067234bb3e686。WSL未运行；官方说明该配置是VM容量设置且在WSL启动时读取（https://learn.microsoft.com/en-us/windows/wsl/wsl-config），不声称12GB在当前主机上被预留占住。没有因诊断而启动/关闭WSL、结束其他进程或删除任何缓存文件。

12:58 GPU前置只读核验在`PRE_GPU_REVALIDATION_20261014_1258.json`：设备12MiB/0%，compute-apps仅列Codex桌面进程（WDDM内存N/A）；这不涵盖所有图形消费者。原项目GPU服务11435和UI18501均未监听，CPU11436仍在；WSL运行列表为空。没有重新读取WSL原断点或改变原作业，最终仍由用户手动续跑。

生成执行器预检发现旧`gpu_lease.json`虽为ACQUIRED，但creation_time早于当前主机启动时间且缺少boot绑定。下一次GPU加载前必须归档原凭据、重新核验并签发当前启动周期的凭据，不能沿用旧状态。generation_evaluation正在给共享Python/PowerShell入口增加fail-closed启动周期检查与小型CPU测试；root负责真实租约归档/签发。当前CPU编码不受影响，尚未因这次核验启动任何GPU模型。原实验记录和旧租约历史不回写。

13:04该修复与真实签发已经完成：50项CPU定向测试通过；旧租约经CLI实际拒绝exit2，PowerShell拒绝后CUDA保持关闭。02:04:42UTC在gpu.lock内重新确认WSL空列表、本项目11435/18501关闭、compute-apps仅Codex桌面；原租约40961字节逐字节归档，SHA `003f52ad1159316e9a5e82ca4e8b830af77d238798531604e02a75658a77eb30`。新租约SHA `884715ec4556624cd069b461b1b9cfc3483f0bab32f6f77c6cf2165f6d6b0cac`绑定win32、host_boot_time_unix及validated_at，共享validator实际通过。收据`GPU_LEASE_REVALIDATION_20261014.json`。

新租约用previous_lease artifact追溯原断点/历史PID，不复制历史PID为活进程。当前状态仅为`NOT_RUNNING_OBSERVED_WSL_OFF`，不冒称重新读取了原断点或检查日更存活；`USER_MANUAL_RESUME`保留。全过程空CUDA mask、没有加载模型、没有停止或恢复原任务。实际GPU模型交接仍需复查当时资源，当前凭据不代表已经执行stress或主实验。
