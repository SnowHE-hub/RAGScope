# S1.3 / S1.4 清洗、数据契约与切块核验

记录日期：2026-10-14（客户端 Australia/Sydney）。本子任务全程 CPU-only，未导入 torch、下载模型或操作既有 GPU 进程。测试环境为 `D:/RAGScope_Runtime/envs/core/Scripts/python.exe`。

## 本机事实、修改与验证

| 任务书问题 | 本机事实 | 当前处理与证据 |
|---|---|---|
| 长文档被长度过滤 | 旧 `clean_documents` 限定 50–5000 字符，确实会删除全文 | 新默认仅拒绝空白文档，长度上下限显式配置，拒绝原因日志；>5000 字符 Markdown 实例通过 |
| 清洗破坏结构 | 原 `\\s+` 压平换行，`<[^>]+>` 可删代码尖括号 | `clean_text` 区分文本/Markdown/代码/显式 HTML；仅统一换行、移除 NUL，不改代码 Unicode 拼写；HTML 使用标准解析器 |
| fixed 超预算 | 原单词拆分允许超大单词独立形成超限块 | 以完整候选文本重新计 token，超大单词/中文按 Unicode 码点边界回退；强制前进，无隐式 tokenizer 解码损伤 |
| sentence 超预算 | 原实现超长句不拆，带 overlap 后也可能超限 | 超长句固定窗口回退，最多一整句 overlap；与硬预算冲突时缩减 overlap 并保存原因 |
| semantic 名称误导 | 原实现是段落/句子/词的结构递归 | 正式名称 `StructureRecursiveChunker`；`SemanticChunker` 为同类兼容别名，没有声称 embedding 语义分割 |
| tokenizer 不匹配 | 原构造固定 cl100k_base | `TokenizerAdapter` 接受已加载的 tokenizer，签名含 ID/revision。无参数默认明确为 unicode_codepoint/v1，仅离线工程/参考用途，不能当成目标 embedding 的 token |
| ID 跨配置冲突 | 原 ID 仅 doc_id + 块序号 | ID 包含文档文本/revision、切块签名、canonical 起止坐标；稳定证据单位独立于切块配置 |
| NLTK 导入下载 | 原模块导入执行 `_ensure_nltk` 并可能联网 | 新模块无 NLTK/tiktoken import，无模型/下载副作用；句边界规则确定、离线 |

旧清洗可通过 `clean_documents(..., legacy=True)` 或 `clean_documents_legacy` 复现。原源文件备份由主任务保存于 `D:/RAGScope_Runtime/backups/20261014_baseline`。新版切块不宣称逐字符重现旧有错误行为。

## 共享接口与坐标约定

`ragscope/schema.py` 定义 Document、EvidenceUnit、Chunk、Query、ContextBundle、GenerationRecord，均有 `to_dict/from_dict`。旧 Chunk 的前八个构造字段和 `ragscope.ingestion.chunkers.Chunk` 导入路径保持兼容。

所有源位置是 **规范化文档字符串的 Unicode 码点半开区间** `[start,end)`，绝不冒充原始 HTML/CRLF 字节位置。每个块严格满足 `chunk.text == document.text[canonical_start:canonical_end]`，不拼接空白、不用 find 回搜重复文字。`unit_spans` 保存 unit_id、doc_id、全局 start/end 及块内 chunk_start/chunk_end。文本重复时仍能区分不同来源段落。

`evidence_units(doc)` 优先验证数据适配器提供的结构单位；没有单位时，在切块前派生段落单位。给定单位的内容、范围、doc_id、唯一性必须一致。清洗会改变文字时拒绝已赋予的旧偏移，要求先清洗后标注，避免静默错位。

`ParentChildChunker.chunk_with_parents(doc)` 返回 `(children, parents)`；`.chunk(doc)` 保持旧列表接口，`.parents` 提供本次文档父块映射。子块不会越过所属父块，所有父块依旧有预算和坐标。父块扩展的 scope 验证、全局预算、去重由 ContextBundle 层负责，不能把本模块的父子关系测试当成整个生成流程已经验证。

`dedup_documents(..., alias_map=...)` 保留去重别名及别名证据单位，`remap_query_aliases` 同时重写 query scope/gold document/evidence 引用。相同文字但证据边界不同会报错，要求保留两份而不是丢失标签。跨 split 相同文本保留并标记 duplicate_group/cross_split_duplicate，数据准备阶段必须按组处理后才能冻结。近重复检测不在本函数冒充已经完成。

## 真实执行验证

测试命令（PowerShell）：

```powershell
. ./scripts/runtime_env.ps1
$env:COVERAGE_FILE='D:/RAGScope_Runtime/runs/chunking_unit.coverage'
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -m coverage run --source=ragscope.schema,ragscope.ingestion.cleaner,ragscope.ingestion.chunkers -m pytest tests/unit/test_chunking_v2.py -q -p no:cacheprovider
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -m coverage report -m
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -m coverage json -o 'D:/RAGScope_Runtime/runs/chunking_coverage.json'
```

实际结果：**127 passed in 3.23s，退出码 0**。覆盖率：chunkers 97%（247 statements / 8 missed）、cleaner 100%（140/0）、schema 100%（97/0），合计 98%（484/8）。完整机器报告：`D:/RAGScope_Runtime/runs/chunking_coverage.json`。

`tests/unit/test_chunking_v2.py` 含 26 条中文/代码/长度/重叠/空白人工夹具，每条运行 fixed、sentence、structure、parent_child；另以 seed=42 生成 60 条 Unicode 文本，对三种策略检查全覆盖、严格前进、坐标准确、真实计数预算。另验证非加性 tokenizer、无法容纳单个 Unicode 码点的显式失败、配置签名、稳定证据、同文不同位置、父子关系、JSON 往返、清洗拒绝、别名重映射。

本结果仅为离线功能与性质验证，**不是中文问答成绩，不是实际模型实验**。目标 HF tokenizer 的实际计数与完整 embedding 包装开销仍由 S2 的固定模型预检验证。句子分割是确定性标点规则，不宣称英语缩写/语言学句边界完全正确。Unicode 码点边界能防止非法字符，但不保证每个视觉字素（如组合 emoji）永不跨块。BPE 搜索保证实际输出在预算内，不宣称找到全局最大 token 填充。
