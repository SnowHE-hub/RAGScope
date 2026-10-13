# 恢复记录

campaign：`20261014_v2`；runtime：`D:/RAGScope_Runtime`。

阶段三原任务交接：用户追加明确授权可终止CPU/GPU进程并希望恢复。实际操作状态唯一来源是 `D:/RAGScope_Runtime/runs/20261014_v2/gpu_lease.json`；不要只依据本文推断进程是否仍运行。

已识别Ubuntu中QuantMind的两个T-030研究队列：XGBoost 63d子PID114865（start_ticks10435121），以及LightGBM 63d子PID114866。两者完整父进程身份、cwd、argv、日志、恢复命令和检查点哈希已写入租约。没有修改原项目代码或断点。

检查点语义是已完成的配置/种子/折单元可恢复，未完成单元重算，不能宣称可恢复到中断瞬间。停止前核验83个XGB单元、24个LGBM单元；所有JSON可解析，NPY头与数据长度匹配，UBJSON和Parquet边界完整，文件逐个SHA256记录。停止后再次验证所有已完成文件未变。

实际安全停止：身份逐一匹配后先向队列父进程发SIGTERM防止推进下一任务，再对训练子进程发SIGINT。XGB以KeyboardInterrupt退出；LGBM的`/proc/status`显示SIGINT被忽略（SigIgn末位7），55秒后仍运行。依据用户追加的CPU/GPU终止授权，再次复验断点后仅对PID114866发送SIGTERM，未使用SIGKILL。未完成单元重算；不能宣称这个进程进行了中途保存。两任务已退出，全部先前完成文件哈希不变，租约现ACQUIRED。

恢复入口（仅本项目资源全部释放、租约ACQUIRED且原队列未运行时）：`wsl.exe -d Ubuntu --exec python3 /mnt/d/MCS_26S1课程资料/个人项目/RAGScope/scripts/manage_previous_jobs.py restore`。恢复后必须检查原日志继续增长/新训练单元完成，才能标RESTORED；不能重复启动。

源码基线：`D:/RAGScope_Runtime/backups/20261014_baseline/tracked-source.zip`；HEAD、tracked.patch、未跟踪状态在同目录。不要用 reset/clean 覆盖用户文件。

当前恢复入口：在原项目目录运行 `. ./scripts/runtime_env.ps1`。统一 runner 建成后补充命令和中断测试证据。不要假定其他聊天状态。

运行入口已实现并实际使用：`./scripts/run_v2.ps1 -Profile full -Resume -CpuOnly`。查询状态：`. ./scripts/runtime_env.ps1; & D:/RAGScope_Runtime/envs/core/Scripts/python.exe -m ragscope.experiments.status`。

中断恢复性质测试：`tests/unit/test_runner_v2.py`通过，专用人工fixture以exit99退出，恢复隔离半写入尾行并保留全部5条唯一记录，再次执行文件字节不变。它不是模型实验。

独立子campaign：`20261014_v2_legacy`、`20261014_v2_retrieval`；分别持有自己的writer.lock。不得同时启动同一campaign两个写入者。

下一条准确命令：`Get-Content D:/RAGScope_Runtime/runs/20261014_v2/gpu_lease.json -Raw`
