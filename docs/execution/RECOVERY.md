# 恢复记录

campaign：`20261014_v2`；runtime：`D:/RAGScope_Runtime`。

原 GPU 任务：当前仅只读 nvidia-smi 体检。未停止/暂停/改写任何进程；未取得 GPU 租约，因此无需恢复、不得另起同任务。

源码基线：`D:/RAGScope_Runtime/backups/20261014_baseline/tracked-source.zip`；HEAD、tracked.patch、未跟踪状态在同目录。不要用 reset/clean 覆盖用户文件。

当前恢复入口：在原项目目录运行 `. ./scripts/runtime_env.ps1`。统一 runner 建成后补充命令和中断测试证据。不要假定其他聊天状态。

运行入口已实现并实际使用：`./scripts/run_v2.ps1 -Profile full -Resume -CpuOnly`。查询状态：`. ./scripts/runtime_env.ps1; & D:/RAGScope_Runtime/envs/core/Scripts/python.exe -m ragscope.experiments.status`。

中断恢复性质测试：`tests/unit/test_runner_v2.py`通过，专用人工fixture以exit99退出，恢复隔离半写入尾行并保留全部5条唯一记录，再次执行文件字节不变。它不是模型实验。

独立子campaign：`20261014_v2_legacy`、`20261014_v2_retrieval`；分别持有自己的writer.lock。不得同时启动同一campaign两个写入者。

下一条准确命令：`Get-Content D:/RAGScope_Runtime/logs/legacy_bm25.log -Tail 20`
