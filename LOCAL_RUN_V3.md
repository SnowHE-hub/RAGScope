# RAGScope v3 本地启动

在 `D:/MCS_26S1课程资料/个人项目/RAGScope` 执行。使用已安装的 D 盘 Python 环境、模型与完整索引，不需要重新下载。最终配置由 `configs/v3/default.json` 指向；逐题实验与应用记录分别保存。

## 启动问答服务

终端一：

```powershell
. ./scripts/runtime_env.ps1
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/v3/serve_full_workbench.py --check-only
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/v3/serve_full_workbench.py
```

检查模式核验全库索引和现有模型，不加载GPU模型。真实服务监听 `127.0.0.1:11437`，私有模型后端监听 `127.0.0.1:11435`。启动核对 GPU 消费者和本项目锁；发现未知计算任务或端口冲突会拒绝，不会结束其他进程。原11434服务不使用。

启动需要至少1.5GiB可用系统内存，驻留编码保留512MiB底线；批量编码会受控卸载自己的编码模型回收内存。只保留一个编码器和一个生成模型，请求串行。不修改WSL内存上限或系统环境。

## 打开工作台

终端二：

```powershell
. ./scripts/runtime_env.ps1
./scripts/run_ui_v3.ps1 -CheckOnly
./scripts/run_ui_v3.ps1
```

打开 `http://127.0.0.1:8502`。论文范围留空搜索全库；选一篇可指定论文提问，选多篇可比较。回答方式支持自动、事实、解释、比较和总结。页面展示答案及引用原文，也可展开全部实际输入证据。

答案缓存默认关闭；查询向量可复用同身份缓存。应用交互记录独立于冻结实验，不修改正式成绩。问答服务未启动时页面明确提示，实验记录仍可查看，不回退到Mock。

生成模型统一保持8K后端窗口，避免普通问答与总结切换时重载。普通问答仍只使用最多2048参考token证据、512输出token；总结和比较最多6144参考token证据、768输出token，并显示实际原文覆盖。多篇回答限制引用数量以避免JSON被截断。首次请求有模型冷加载，正式热请求耗时不等于冷启动耗时。

## 正常退出

两个终端分别按Ctrl+C。也可只给当前问答服务写精确停止标记：

```powershell
$ragConfig = Get-Content -LiteralPath configs/v3/default.json -Raw | ConvertFrom-Json
$ragReceipt = Join-Path $ragConfig.run 'FULL_WORKBENCH_SERVICE.json'
$ragService = Get-Content -LiteralPath $ragReceipt -Raw | ConvertFrom-Json
if ($ragService.status -eq 'READY') {
    Set-Content -LiteralPath $ragService.stop_file -Value 'Own service exit requested' -Encoding utf8
}
```

退出会卸载模型并释放自己持有的GPU锁和进程，不批量强杀。原训练保持 `USER_MANUAL_RESUME`，不自动恢复训练、定时器、日更或WSL。

## 查看本轮结果

最终结论见 `reports/v3/FINISH_REPORT_ZH.md`。运行目录包含完整索引回执、开发选型、H_QA200与H_tasks20的配对生成、逐题评分、全库检索与真实在线结果。失败样本保留；语义相似度和规则检查都是代理，不是人工事实正确率。

正式部署版本使用文件名含 `RESIDENT8K` 的结果；原4K/8K交替版本保留诊断。真实API9场景及Streamlit组件与真实API集成已通过。浏览器视觉验证被自动审批拒绝，未通过其他浏览器或CDP绕过，因此没有声称浏览器截图验收通过。

本轮用户明确解除旧10.1入围门槛，要求直接执行正式实验。旧运行与报告保留，旧“候选未入围所以不执行”的状态不代表本轮状态。旧启动说明另存 `docs/execution/v3/LOCAL_RUN_BEFORE_FINISH.md`。

v2入口仍是 `scripts/run_ui_v2.ps1`，其配置和历史实验保留。
