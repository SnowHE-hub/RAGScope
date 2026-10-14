# RAGScope v2 本地启动与退出

从本项目目录 `D:/MCS_26S1课程资料/个人项目/RAGScope` 执行。环境、模型、索引、缓存和日志使用 `D:/RAGScope_Runtime`；现有环境已安装，不需要重建或下载模型。最终交付时本项目CPU/GPU/UI服务全部退出，原用户计算任务由用户手动恢复（`USER_MANUAL_RESUME`）。

## 先查看现有结果

```powershell
. ./scripts/runtime_env.ps1
./scripts/run_v2.ps1 -Profile full -CpuOnly -Resume -ReconcileOnly
./scripts/run_ui_v2.ps1 -Environment models -Port 8501 -CheckOnly
./scripts/run_ui_v2.ps1 -Environment models -Port 8501
```

UI只监听 `http://127.0.0.1:8501`。没有服务时可查看已保存结果及BM25检索；界面可用性配置是历史快照，实际请求还会检查当前进程、身份、锁和资源。不要把历史 `ready=true` 当作服务仍在运行。默认保持固定Hybrid/2B；规则路由在线P50未达到预设10%收益门槛，LoRA同精度F1下降，两者都不默认启用。已有模型请求全部关闭应用缓存，UI交互调用另记账，不追加到冻结test成绩。

## 需要真实Hybrid问答时

使用三个终端分别运行CPU embedding、私有GPU生成服务和UI。先确认没有本项目同端口服务或其它GPU计算；不停止其他进程来抢资源。新模型加载前至少保留3GiB可用系统RAM，逐题保护线仍有效。现有原始服务11434不使用。

终端一启动普通长期CPU服务，**不要使用历史一次性online交接wrapper**：

```powershell
. ./scripts/runtime_env.ps1
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/serve_embedding.py --port 11436
```

该入口显式CPU4线程、GPU层数0、空CUDA mask，读取既有Qwen3-Embedding-0.6B Q8_0和完整索引；只有实际请求时编码问题，不重新编码语料。

终端二先验证当前boot GPU租约：

```powershell
. ./scripts/runtime_env.ps1
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/validate_gpu_lease.py --lease D:/RAGScope_Runtime/runs/20261014_v2/gpu_lease.json
if ($LASTEXITCODE -ne 0) { throw 'GPU租约不属于当前启动周期；保留拒绝状态，不能绕过校验。' }
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe -c "from ragscope.experiments.runtime import require_ram; print(require_ram(2*1024**3))"
if ($LASTEXITCODE -ne 0) { throw '当前RAM不足；不启动模型服务。' }
```

主机重启后旧租约会被拒绝；必须重新核验实际资源并生成当前boot凭据，不能只把旧JSON状态改成ACQUIRED。校验成功也不代表GPU当前没有其他消费者，启动前仍要查看实际设备/进程。

先归档本项目上次退出标记和服务凭据，保留原文件字节；只移动精确的停止标记，不删除缓存、原断点或任何用户任务文件：

```powershell
$runRoot = 'D:/RAGScope_Runtime/runs/20261014_v2'
$historyDir = Join-Path $runRoot ('service_history/manual_start_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Path $historyDir | Out-Null
foreach ($name in @('generation_service.json','embedding_service.json')) {
    $sourcePath = Join-Path $runRoot $name
    if (Test-Path -LiteralPath $sourcePath) { Copy-Item -LiteralPath $sourcePath -Destination (Join-Path $historyDir $name) }
}
$stopPath = Join-Path $runRoot 'generation.stop'
if (Test-Path -LiteralPath $stopPath) { Move-Item -LiteralPath $stopPath -Destination (Join-Path $historyDir 'generation.stop') }
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/serve_generation.py --binary D:/RAGScope_Runtime/tools/ollama-0.34.4/ollama.exe
```

私有服务只使用11435、D盘模型目录、一个GPU锁、单模型和单并发。启动器核验租约后只给它拥有的服务子进程启用CUDA设备0；终端三再次执行普通 `runtime_env.ps1`，保持UI及本地BGE为CPU4/空CUDA。底层CPU客户端只有在验证自有服务身份与当前boot lease后才允许远程GPU请求，每次请求重新核验。

终端三：

```powershell
. ./scripts/runtime_env.ps1
./scripts/run_ui_v2.ps1 -Environment models -Port 8501
```

只在CPU服务健康、私有生成服务READY、索引和模型哈希匹配时运行。界面拒绝资源不足或服务身份不匹配时保留错误，不改guard，也不启动第二套同模型服务。UI重排使用隔离models环境的CPU BGE；加载成本与正式resident在线计时不同。

## 正常退出与复核

UI终端按Ctrl+C；CPU embedding终端按Ctrl+C，入口只退出自己的无状态子进程。私有GPU服务可在另一CPU终端写本项目停止标记：

```powershell
Set-Content -LiteralPath 'D:/RAGScope_Runtime/runs/20261014_v2/generation.stop' -Value 'Own service shutdown requested' -Encoding utf8
```

等待生成服务卸载模型并退出，检查自己的精确PID/创建时间、11435/11436/8501端口及gpu/client锁实际释放。Windows合作终止的exit1不自动等同实验失败，需保留实际退出码和释放证据，不能批量强杀。结束仅释放本项目资源；不恢复原训练、定时器或日更，不启动/重启WSL，不修改`.wslconfig`。

离线验证用core环境；Torch专属路由测试用models环境，均CPU-only。Linux/WSL需要独立Linux环境，本轮未安装，不能直接使用Windows虚拟环境或承诺Linux入口已通过实跑。
