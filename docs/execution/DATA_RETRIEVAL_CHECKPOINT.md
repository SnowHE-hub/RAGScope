# 数据与CPU检索恢复点

此文件由数据/检索执行子任务维护，不替代主STATUS、RECOVERY或实验账本。

已完成真实数据冻结：QASPER509篇，train1000/dev100/test200/global100，技术集32题；官方SQuAD2公开dev全库1204段落及E4固定100题。主数据和SQuAD均已完成独立哈希、证据坐标、scope检查。

已完成BM25：dev11组1100行、额外context400行、两个论文内test各200行、E8全库100行，共2000行，0失败。所有行及配置位于`D:/RAGScope_Runtime/runs/20261014_v2_retrieval/records`。

实际Qwen CPU Dense+Hybrid11组dev矩阵暂时暂停给真实LoRA训练让出RAM。终端session37932，实际worker PID43932/create_time1791906175.5201824、launcher38136；worker仍存活但SUSPENDED，不能同时启动第二个相同campaign。原session71547在受控CPU线程切换时有一条索引HTTP连接重置，已保存错误并从缓存恢复；已完成记录未丢失。
已完成前四组共800条真实神经检索记录，第五组F1024_O0尚未完成。此次暂停前2404个完整文本向量落盘，最新5个缓存哈希与manifest一致；暂停后不得假称全矩阵完成。`runs/20261014_v2_retrieval/worker.json`保存当前worker身份与SUSPENDED状态。
本项目CPU-only服务PID39808已因主执行器要求释放训练内存而停止，当前11436不可用。其原12线程/12batch线程命令和精确进程身份在`PLANNED_TRAINING_RAM_PAUSE.json`；冻结模型身份仍在`D:/RAGScope_Runtime/runs/20261014_v2/embedding_identity.json`。线程改变有独立性能身份与逐元素相同的有界对照证据，见`cpu_embedding_benchmark_20261014_023547/THREAD_TRANSITION.json`；旧11224及39808都不得仅凭PID再次清理。
停止服务前成功执行drain barrier，1024维输出、8个后端tokens；随后只停止精确PID39808/create_time1791906104.7015493。`embedding_service.json`为EXITED_PLANNED_RAM_RELEASE。原监督器写回旧PID的问题已修复并存档。主执行器结束训练内存占用后才恢复同一CPU服务，再恢复43932；暂停时可能有一个未落盘HTTP调用，若出现连接重置，仅在旧worker确实退出后执行下方准确命令，从缓存继续。不可将这条可能的在途请求当成已完成断点。

仅在确认原执行进程已结束且服务READY时，用以下**准确命令**从逐文本向量缓存和逐题结果继续；保持相同模型/数据配置，不重选split：

```powershell
. ./scripts/runtime_env.ps1
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -u scripts/run_retrieval_v2.py --home 'D:/RAGScope_Runtime' --campaign 20261014_v2 --stage dense --embedding-url 'http://127.0.0.1:11436' --embedding-identity 'D:/RAGScope_Runtime/runs/20261014_v2/embedding_identity.json'
```

完整矩阵成功后读取`CANDIDATE_SELECTION.json`，冻结simple/structured两个候选；候选必须来自dev。先对已选方案运行必要的context none/neighbor/parent对照，再按冻结选择准备真实test/E8及E4上下文。主任务书要求2048 reference tokens；生成模型的共同真实token预算检查由生成执行器负责。

```powershell
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -u scripts/run_retrieval_v2.py --home 'D:/RAGScope_Runtime' --campaign 20261014_v2 --selected-expansions --embedding-url 'http://127.0.0.1:11436' --embedding-identity 'D:/RAGScope_Runtime/runs/20261014_v2/embedding_identity.json'
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -u scripts/run_retrieval_v2.py --home 'D:/RAGScope_Runtime' --campaign 20261014_v2 --final-contexts --dataset squad --embedding-url 'http://127.0.0.1:11436' --embedding-identity 'D:/RAGScope_Runtime/runs/20261014_v2/embedding_identity.json'
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -u scripts/run_retrieval_v2.py --home 'D:/RAGScope_Runtime' --campaign 20261014_v2 --final-contexts --dataset qasper --embedding-url 'http://127.0.0.1:11436' --embedding-identity 'D:/RAGScope_Runtime/runs/20261014_v2/embedding_identity.json'
```

最终test索引范围：QASPER固定509篇，SQuAD全部1204官方dev段落；全库检索不会收到gold_doc_id。SQuAD默认仅使用dev选定结构方案的Hybrid，恰好100条上下文，不再从SQuAD test搜索更优方法。`--methods`可以按阶段执行bm25/dense/hybrid，结果ID和配置不随执行顺序变化；相同向量可复用，延迟结果必须披露cache-hit与cold编码差异。

待完成的神经重排由chunking_core子任务负责，必须恢复Hybrid原始50候选文字再做50→10，不能用BM25或10→10冒充。生成冻结器由generation_evaluation子任务负责，尚无结果的配置不得生成虚构PASS。

`--selected-expansions`仅评估两个已选候选；若结构候选没有父块，再加入预声明的P256_1024作三种扩展诊断。`CONTEXT_SELECTION.json`按dev整组覆盖→连续覆盖、容差0.01、近似时优先none→neighbor→parent冻结；test不可改写它。final-context执行器要求该文件与候选哈希匹配。

检索+SQuAD原16个离线测试及独立审计新增3个手算测试均通过。独立审计命令为`python scripts/audit_retrieval_v2.py --home D:/RAGScope_Runtime --campaign 20261014_v2`，会验证全部已完成运行，保留每项数值与seed42抽查10项。
最近审计PASS：24个完成运行、2600行、11821个原文片段、72个汇总数字。另有统计输入核验3个测试通过（合计22个检索/SQuAD/审计/统计离线测试）。冻结探索性统计协议`configs/v2/retrieval_statistics.json`；待最终test上下文完成后运行`python scripts/summarize_retrieval_statistics_v2.py --home D:/RAGScope_Runtime --campaign 20261014_v2`，按论文或SQuAD原始文章成组抽样2000次，最终数字写入`RETRIEVAL_STATISTICS.json`。
当前恢复后的长进程已经载入新adapter，后续冷编码保存backend usage；旧cache无法回补历史usage。CPU线程对照实测256/512/1024正文tokens对应257/513/1025后端tokens，仅作为有界直接观测，不反填所有旧行。
