# S1.2 本机事实—测试—修改清单

基线：原项目 main 的源码在 `D:/RAGScope_Runtime/backups/20261014_baseline/tracked-source.zip`；当前分支 `codex/ragscope-v2-autonomous-20261014`。阅读入口、配置、所有功能层、旧测试、README与architecture.md；旧文档数字不是本次实验。

| # | 本机核验与修改状态 | 代码/验证 |
|---|---|---|
|1|已确认旧50～5000字符过滤；本机已修复，保留legacy函数|`ragscope/ingestion/cleaner.py`；`tests/unit/test_chunking_v2.py`|
|2|已确认旧空白压缩和HTML正则破坏结构；按格式清洗已修复|同上；长文本、Markdown、HTML、代码性质测试|
|3|已确认原fixed有overlap；新实现严格预算与Unicode坐标覆盖|`ragscope/ingestion/chunkers.py`，127项核心测试|
|4|已确认原sentence超长句不拆；fallback与overlap收缩已修复|同上；非加性tokenizer/长句测试|
|5|已确认semantic实际结构切分；重命名structure_recursive，保留别名|同上；文档说明不声称embedding语义切分|
|6|已确认原cl100k不等于embedding tokenizer；新接口显式身份并注入目标Qwen tokenizer|`indexing/local_embedding.py`；正式矩阵仍待执行|
|7|已确认工厂硬编码、固定名、存在即复用；修复中|`scripts/build_indexes.py`、`indexing/identity.py`、`array_index.py`；完整manifest绑定待整体验证|
|8|已确认旧chunk ID无配置版本/坐标；稳定证据先于切块，坐标标canonical已修复|`schema.py`、`ingestion/qasper.py`、`chunkers.py`；34615证据单元坐标复验|
|9|已确认旧生成缓存键缺身份且缓存错误；新键包含context/model/prompt/config并不缓存错误|`generation/pipeline.py`、`cache.py`；离线测试|
|10|已确认旧JSON失败伪造首块引用；现为明确parse_error并只接受实际可见引用|`generation/pipeline.py`、`context.py`；引用/恶意ID/预算测试|
|11|已确认benchmark错误后continue成功分母；现保留失败行及总体/条件均值|`evaluation/benchmark.py`、`v2.py`；失败分母测试|
|12|已确认单相关文档Recall=HitRate；原CSV列不一致且原件保留|`evaluation/retrieval_metrics.py`；E0新目录复跑中，未手工修CSV|
|13|已确认旧重排10→10；新50→10、query计入滑窗预算实现中|`retrieval/reranker.py`、`hybrid_retriever.py`；独立单元测试随后归档|
|14|已确认旧评测只有文档命中；新增连续证据/完整单元/完整集合及生成评测|`evaluation/evidence.py`、`v2.py`；49项生成评测测试，真实生成尚未运行|
|15|已确认导入NLTK会下载、依赖不锁、旧pickle兼容风险；已取消导入下载并锁依赖，新embedding只用校验npy|`indexing/sparse_index.py`、`array_index.py`、`requirements-v2.lock`；旧pickle仅显式可信项目迁移|

阶段不是验收等待点。上述“已修复”指代码与对应离线测试，真实模型结果按独立run保存，未运行项不标PASS。
