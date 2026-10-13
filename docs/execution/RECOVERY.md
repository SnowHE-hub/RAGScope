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
