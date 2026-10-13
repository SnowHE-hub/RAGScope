# RAGScope v2 本轮执行范围

完整任务书：`docs/execution/MASTER_PLAN.md`；当前进度：`docs/execution/STATUS.md`；决策和恢复：同目录 `DECISIONS.md`、`RECOVERY.md`。

保持原目录，增量修改现有包。保护用户文件、旧报告、原断点。只stage本任务文件，不push、不上传、不付款、不搜索密钥、不修改驱动或安全工具。

阶段一/二严格CPU-only：启动前 CUDA_VISIBLE_DEVICES 空串、显式 device=cpu、4线程。不调用原 Ollama 服务，不停止/暂停/修改原GPU工作负载，不重启WSL或电脑。阶段三才核验GPU租约；没有准确身份、完整checkpoint、安全退出及可靠恢复命令则保留原任务，记录阻断。

模型/数据/缓存/环境/日志/断点放 `D:/RAGScope_Runtime`。使用 `scripts/runtime_env.ps1` 设置子进程环境，不改全局环境。所有下载有界且校验，所有真实实验逐条落盘保留失败分母。Mock不能计入真实实验。阶段完成继续执行，不等待用户确认。

## CodeGraph

结构问题优先 CodeGraph（本项目已初始化）。context取得任务上下文，一次explore获取相关源码；查符号用search，调用关系用callers/callees，影响用impact，目录用files。信任AST结果，不用grep重复验证。字面文本和已打开文件可读/rg。不要委托纯架构探索；独立实现/测试任务可并行。索引写后有约500ms延迟。
