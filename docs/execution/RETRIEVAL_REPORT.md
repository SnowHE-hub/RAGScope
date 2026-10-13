# S2 CPU 检索实验报告

执行日期：2026-10-14（Australia/Sydney）。子campaign：`20261014_v2_retrieval`。
当前已完成BM25全11组及有界上下文扩展对照；Dense/Hybrid因CPU后端启动前RAM预检不足而`BLOCKED_RESOURCE`，不能把BM25结果写成已验证Dense。固定语料上的test上下文与E8 BM25独立执行，不用test重选参数。

## 冻结输入与公平性

使用S1.5冻结的36篇dev论文、100个问题，92个可回答、8个不可回答。test未用于选型。
原始数据及切分身份见`D:/RAGScope_Runtime/corpora/20261014_v2/SPLIT_MANIFEST.json`。

切块使用实际Qwen3-Embedding-0.6B tokenizer，revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`，tokenizer文件SHA-256 `def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a`。
所有块都再次按真实tokenizer计数，并验证canonical源文本全覆盖、源片段与输出相等、父子包含关系。没有静默截断。
`cl100k_base`仅用于统一context 2048-token预算；标题和引用标签计入预算。

BM25使用`rank_bm25.BM25Okapi`，k1=1.5、b=0.75、epsilon=0.25；词法分词为alnum-cjk-v1。
IDF在相同冻结dev全文库上建立，查询限制为用户指定论文。先取最多50候选、再前10个按序打包。父子组先按子块检索再扩父、去重和预算打包。
查询顺序seed42打乱，预热3次。时延为真实检索加context构建端到端，不含数据下载和索引构建。
全部CPU-only；本执行器未启动或请求GPU模型，也未修改原GPU任务。

## BM25实测矩阵

每行100个问题、0失败。证据指标分母都是92，8个不可回答的证据召回为N/A。
“整组”表示至少一个合法参考证据集合全部覆盖；“连续”是参考单元字符范围的真实覆盖。
所有值来自`D:/RAGScope_Runtime/runs/20261014_v2_retrieval/E1_bm25_SUMMARY.json`及对应`records`逐行JSONL。

| group | 子块数 | 父块数 | context整组覆盖 | context连续覆盖 | 完整单元召回 | P50 ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| F256_O0 | 696 | 0 | 0.521739 | 0.721533 | 0.583514 | 13.278 | 16.402 |
| F256_O32 | 786 | 0 | 0.586957 | 0.736286 | 0.646014 | 14.616 | 21.837 |
| F512_O0 | 355 | 0 | 0.576087 | 0.692583 | 0.622373 | 12.509 | 15.902 |
| F512_O64 | 398 | 0 | 0.652174 | 0.739776 | 0.688587 | 13.611 | 17.245 |
| F1024_O0 | 184 | 0 | 0.467391 | 0.529755 | 0.490942 | 8.340 | 13.225 |
| F1024_O128 | 204 | 0 | 0.608696 | 0.658897 | 0.630435 | 10.271 | 16.928 |
| S512 | 389 | 0 | 0.619565 | 0.703486 | 0.641304 | 13.383 | 16.803 |
| R512 | 406 | 0 | 0.750000 | 0.775815 | 0.775815 | 13.499 | 17.180 |
| R1024 | 198 | 0 | 0.673913 | 0.691304 | 0.691304 | 9.383 | 15.749 |
| P256_1024 | 853 | 198 | 0.652174 | 0.668478 | 0.668478 | 12.667 | 18.142 |
| P256_1536 | 823 | 135 | 0.554348 | 0.580072 | 0.580072 | 14.518 | 19.030 |

累计5625个不同chunk记录（含333父块），低于180,000工程上限。
这批论文内检索的文档命中没有区分力，因此不把doc hit当作找齐证据。
初筛候选按预先规则选为简单组F512_O64和结构组R512，证据在`BM25_SCREEN_SELECTION.json`。
R512当前BM25上下文整组覆盖较高；这只是dev初筛，不证明生成答案更准确，也不证明同样结果适用于Dense或其他数据。
更大父块和更大固定块没有单调改善；预算内是否能放下完整证据比单个块更长更关键。

## 已执行的上下文扩展对照

在F512_O64、R512和已有最佳父子组P256_1024上比较none、neighbor以及适用时的parent。
父子组之外没有父映射，未编造“parent结果”。已有主矩阵行直接复用，共额外400条真实BM25问题记录。
三个P256_1024策略使用完全相同的子块检索排序，只改变context扩展。
逐条产物为`records/E1_<group>_bm25_<strategy>.jsonl`，每个文件均含不可变`.config.json`及记录哈希。
目前BM25总计1500条，无失败；这不含任何生成调用。

| 固定子块配置 | 扩展 | dev context整组覆盖 | dev context连续覆盖 |
|---|---|---:|---:|
| F512_O64 | none | 0.652174 | 0.739776 |
| F512_O64 | neighbor | 0.576087 | 0.691500 |
| R512 | none | 0.750000 | 0.775815 |
| R512 | neighbor | 0.652174 | 0.670290 |
| P256_1024 | none | 0.586957 | 0.708426 |
| P256_1024 | neighbor | 0.630435 | 0.687568 |
| P256_1024 | parent | 0.652174 | 0.668478 |

在这批dev上，邻块扩展没有改善已选F512_O64/R512的整组覆盖；P256_1024父扩展增加整组覆盖，但连续覆盖下降。这些不同指标不能拼成单一“全面提升”。

## 已完成的test上下文与全库E8

全库索引构建前实际启动检查可用RAM为1,248,690,176字节，高于768MiB门槛；原GPU任务未被修改。
固定同一509篇论文语料，为dev初筛的两个候选分别构建完整BM25索引：F512_O64 6405块，R512 6551块。
两者共12,956块，加dev矩阵保守累计18,581块，低于180,000上限；相同源片段跨索引可能重复计入该保守工程统计。
全库索引IDF在同一509篇上计算，这与dev初筛的36篇IDF范围不同，实验身份已经单列。

| run_id | 任务范围 | N / 证据适用N | context整组覆盖 | context连续覆盖 | 文档Recall@前10检索块 |
|---|---|---:|---:|---:|---:|
| TEST_CONTEXT_F512_O64_bm25_none | 指定论文 | 200 / 182 | 0.631868 | 0.716879 | 1.00 |
| TEST_CONTEXT_R512_bm25_none | 指定论文 | 200 / 182 | 0.703297 | 0.740404 | 1.00 |
| E8_R512_bm25_none | 固定509篇全库 | 100 / 100 | 0.170000 | 0.178333 | 0.37 |

这500条全部实际执行、0失败；目前总计2000条BM25检索结果，不含生成调用。
论文内test的18条不可回答证据指标为N/A；所有失败仍保留在预先冻结的任务分母中。
同一篇重复检索块不重复计算文档命中。E8的0.37即100题中37题在前10个检索块见到其金标准论文。
E8全库只给问题文字和global scope，gold_doc_id仅在结果返回后用于评估；不将论文答案或证据位置传给检索器。
全库结果明显弱于用户指定论文任务，不能将后者接近1的文档命中冒充全库检索能力。
test从未反向改变两个候选选择；当前R512默认仍源自dev初筛。
实际ContextBundle和raw候选可供后续生成复用，但未执行生成，不称“答案质量实验完成”。

`TEST_CONTEXT_E8_SUMMARY.json`明确E8四条计划路径中BM25=`PASS`，Dense/Hybrid/Hybrid+Rerank=`BLOCKED_RESOURCE`，总体`PARTIAL_BLOCKED`。

## Dense/Hybrid历史资源阻断与恢复

`D:/RAGScope_Runtime/runs/20261014_v2/embedding_preflight.jsonl`保存3次启动前预检。
两次Windows可用RAM约1.155GB/1.164GB，低于所需1GiB加512MiB保留量；模型没有加载。
现有WSL替代路径开始前宿主仅约295MB可用；其记录中minimum_host_available字段为512MiB、reason字符串为1GiB，实际余量低于两种阈值，阻断结论一致但阈值记录不一致必须披露。
上述三次是历史启动前失败，不应在环境改变后继续当作永久阻断。后续用户追加授权的原任务安全接管由主执行器单独审计；内存释放后，主执行器已核验独立CPU-only Qwen服务的真实1024维输出。
11组Dense/Hybrid矩阵现已实际开始，固定Qwen模型、量化、pooling、归一化和tokenizer身份；完成前不声称已有完整质量结果。50→10神经重排仍由后续独立运行完成。
缓存实测前184个向量在约333秒内落盘，约0.55片段/秒（以256-token块为主）；这是早期吞吐观察，不作为512/1024块或全程耗时保证。每个完成文本的数组及身份立即落盘，可供恢复。

## 可恢复执行与边界

```powershell
. ./scripts/runtime_env.ps1
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -u scripts/run_retrieval_v2.py --home 'D:/RAGScope_Runtime' --campaign 20261014_v2 --stage bm25
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -u scripts/run_retrieval_v2.py --home 'D:/RAGScope_Runtime' --campaign 20261014_v2 --stage bm25 --groups F512_O64 R512 P256_1024 --expansion-comparison
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -u scripts/run_retrieval_v2.py --home 'D:/RAGScope_Runtime' --campaign 20261014_v2 --test-contexts
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -u scripts/run_retrieval_v2.py --home 'D:/RAGScope_Runtime' --campaign 20261014_v2 --stage dense --embedding-url 'http://127.0.0.1:11436' --embedding-identity 'D:/RAGScope_Runtime/runs/20261014_v2/embedding_identity.json'
```

索引按数据、split、scope、完整切块参数、真实tokenizer及revision签名隔离；内容哈希与条数匹配才复用。
每题保存raw候选50的ID/得分/源坐标，完整原文可从对应索引恢复；保存实际ContextBundle、预算丢弃原因、去重记录、证据分数、错误及实际计时。
RecordStore每条fsync，Campaign持单写锁；中断从完整记录恢复，已完成问题不重复统计。
12个检索离线单元测试通过，覆盖切块源覆盖、父子边界、真实BM25 scope、失败分母、不可回答N/A、dev容差选择及幂等恢复；全库测试额外证明gold_doc_id不进入scope、干扰文档仍被实际搜索。新增数组索引恢复和身份拒绝使用明确标记的离线数值夹具，不计入模型结果。
独立反向核验见`E1_BM25_INDEPENDENT_AUDIT.json`：重读1100条实际context源坐标，使用独立区间合并代码重算33个表格数字全部一致；其中seed42抽10数字保存，未用汇总表核验汇总表。
Dense/Hybrid要求显式CPU后端身份与loopback URL；没有就不将词法检索或Mock包装成Dense实验。
重排、E8其余三种检索方式及最终生成主实验尚未完成，当前报告不声称它们已经完成。
