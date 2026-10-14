# 在线与最终UI的CPU embedding窗口

当前只完成准备，未启动服务、未发HTTP、未加载模型。已有SQuAD/aux一次性启动收据不复用；新入口是`scripts/serve_online_embedding_v2.py`，只为在线/UI查询提供原服务，不编码正文。

准备已实际运行，HTTP运输层与`subprocess.Popen`被临时guard禁止，exit0。固定GGUF、CPU程序、tokenizer及其manifest、全部6551块索引文件和三个已有cache数组样本通过哈希检查。所有新日志、临时文件与缓存均在`D:/RAGScope_Runtime`。main/online_embedding下：

- `PREPARATION.json` SHA `c64ab31a1d2fe9aac15282483834c0e9cbe4b002f7e9645f5b405b605fe4c6b7`。
- `MODEL_INPUT_MANIFEST.json` SHA `42679bd2218198917f5dd8989378fac1e7251a864c4ded28c645ff95aca64c26`。这是本地已有资产绑定，非新的上游模型版本。
- `ENGINEERING_VALIDATION.json` SHA `0d3b62d966ac697f3f72d67a016d18fa72fa6331ea66725fabda4e79facb141d`。8项定向行为测试PASS，原679项全回归不自动覆盖新增两个Python文件。

实际准备观察：11436可bind，11435仍被现有生成任务占用，GPU/client锁被持有，主retrieval锁可用。此观察不构成启动许可。最新调度先等待BF16与11435完整退出，再由root完成主矩阵的大对象验收/分析/图表并释放内存；root此后提供fresh release receipt及SHA，才运行：

```powershell
. ./scripts/runtime_env.ps1
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe -u scripts/serve_online_embedding_v2.py --run --handoff-receipt '<实际释放收据路径>' --handoff-sha256 '<实际SHA256>'
```

两个占位值必须由真实交接替换，不能直接执行示例。入口重核绑定的源码/数据文件哈希、旧CPU服务PID/create不存在、11435/11436空闲及GPU/client/retrieval锁，fresh available至少3GiB。加载和控制检查至少保留1GiB；60秒health时限、30秒1024维控制请求，不降阈值、不自动重启。历史3/3后端失败journal只读验证，原值不改。

原实际成功命令保持：`llama-server.exe -m Qwen3-Embedding-0.6B-Q8_0.gguf --embedding --pooling last --embd-normalize 2 --host 127.0.0.1 --port 11436 -ngl 0 --no-op-offload -t 4 -tb 4 -c 2048 -b 1100 -ub 1100 -np 1 --threads-http 2`。程序SHA `7e236243c9b670c6a49410c03186fd575c57a0efe3fd8a21d8727503d571d5b6`，GGUF SHA `06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439`；完整绝对路径见准备收据。CUDA为空串、4线程，不使用GPU、不接触原11434或WSL/原任务。

只有真实health、监听端口所属PID、1024维有限非零向量和RAM后检通过，才发布新的`embedding_service.json`，状态`RUNNING_ONLINE_UI`。原`embedding_identity.json`保持原389字节与SHA `5bde6b6123200f22497c9eee79dec2c4dbfa3d8f3f66d60cb46ab641e441502c`；新进程收据绑定其SHA，不伪造新的语义模型身份。此后CPU服务仅持有自身service.lock，共享GPU/client锁释放给2B在线及UI任务。

运行时观察RAM，低于1GiB会落盘并通知协调器，**不会声称自动暂停请求，也不会在客户端正在使用时自行杀服务**。READY交接后的控制器异常保留服务和准确身份，状态`RUNNING_WITHOUT_SUPERVISOR_REQUIRES_COORDINATOR`，由协调器接手；加载交接前失败才清理准确自有子进程。

在线/UI所有客户端已经结束、GPU生成服务已卸载退出后，由协调器请求正常停止：

```powershell
. ./scripts/runtime_env.ps1
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/serve_online_embedding_v2.py --request-stop --reason 'Online and final UI clients have completed and exited'
```

此命令只写与PID/create/start时间绑定的stop请求。仍活跃的控制器取得共享锁、发单slot drain控制请求、再次核精确PID/create/argv后停止自己的stateless服务，保存实际Windows退出码（不能改写为0）、端口和失败journal核验。最后还需协调器确认控制器退出、11436已关闭、进程消失，才算资源归还。原任务始终由用户手动续跑。

正式四份局部报告继续等待完整主ANALYSIS及协调器的来源spec；最新固定train600路由诊断必须加入最终报告绑定，旧“训练差距不可观测”说明仅保留为历史，不能覆盖新证据。
