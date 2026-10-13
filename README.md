# RAGScope

## RAGScope v2 本地执行

v2 使用免费本地模型、可追溯证据坐标和逐条实验账本。当前执行状态以
[`docs/execution/STATUS.md`](docs/execution/STATUS.md) 为准；预检通过、功能测试通过与正式问答实验完成分别记录。
原始任务书保存在 [`MASTER_PLAN.md`](docs/execution/MASTER_PLAN.md)，决策和资源恢复记录分别在
[`DECISIONS.md`](docs/execution/DECISIONS.md) 与 [`RECOVERY.md`](docs/execution/RECOVERY.md)。

Windows 主路径使用项目外的 `D:/RAGScope_Runtime`，模型、环境、数据、索引、日志和断点均放在那里。
从项目目录执行：

```powershell
# 新机器首次安装隔离的 core 环境，按哈希锁文件验证依赖
./scripts/bootstrap_v2.ps1

# 查看/推进 CPU 任务；检测正在运行或暂停的 worker，避免重复启动
./scripts/run_v2.ps1 -Profile full -CpuOnly -Resume

# 单元测试不下载模型，不触发 GPU 推理
. ./scripts/runtime_env.ps1
& "$env:RAGSCOPE_HOME/envs/core/Scripts/python.exe" -m pytest tests/unit -q

# 打开本地 UI；需要已完成并校验的 v2 索引
& "$env:RAGSCOPE_HOME/envs/core/Scripts/python.exe" -m streamlit run ui/chat_app.py
```

runner 在任务未齐备时返回退出码 `2` 并留下具体原因；这不表示整套实验已经完成。
GPU 服务、真实训练和正式生成另有资源交接及冻结清单校验，不会因为打开 UI 而自动下载或训练模型。
Linux 启动脚本要求单独的 Linux 环境，不能复用 Windows 虚拟环境；本轮未安装 Linux 模型环境。

阅读实现与结果解释可从 [`docs/learning/00_project_map.md`](docs/learning/00_project_map.md) 开始。
v2 不需要配置云 API 密钥，Mock 仅用于离线功能测试。

## 历史版本说明与结果

以下保留原版本的架构、启动方式和报告数字，**不是本轮 v2 的实测结果**。
历史 Dense/Hybrid 的原始 embedding revision 不完整；本轮另以固定版本重建并独立记录，不将两者冒充同一个实验。

A modular **Retrieval-Augmented Generation** benchmarking system that lets you compare chunking strategies, retrieval methods, and generation quality side-by-side — with a transparent Streamlit UI.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                      RAGPipeline                        │
│                                                         │
│  ┌─────────────┐    ┌──────────────────────────────┐   │
│  │  Retriever  │    │       LLM Client             │   │
│  │             │    │  (OpenAI gpt-4o-mini /        │   │
│  │  ┌────────┐ │    │   MockLLMClient)              │   │
│  │  │ Dense  │ │    └──────────────────────────────┘   │
│  │  │(Chroma)│ │              │                         │
│  │  └────┬───┘ │    ┌─────────▼──────────┐             │
│  │       │ RRF │    │   format_context    │             │
│  │  ┌────▼───┐ │    │   (tiktoken budget) │             │
│  │  │  BM25  │ │    └─────────┬──────────┘             │
│  │  └────────┘ │              │                         │
│  │     │       │    ┌─────────▼──────────┐             │
│  │  Reranker   │    │   RAGResult        │             │
│  │ (optional)  │    │  answer/citations  │             │
│  └─────────────┘    │  confidence/cost   │             │
│                     └────────────────────┘             │
│                              │                          │
│                     ┌────────▼────────┐                │
│                     │   RAGCache      │                │
│                     │  (diskcache)    │                │
│                     └─────────────────┘                │
└─────────────────────────────────────────────────────────┘

Data flow:
  Raw docs → Cleaner → Chunker → EmbeddingService → DenseIndex
                                                  └→ BM25Index
```

---

## Directory Structure

```
RAGScope/
├── ragscope/
│   ├── ingestion/        # Loaders (SQuAD, arXiv) + cleaner + chunkers
│   ├── indexing/         # EmbeddingService, DenseIndex (ChromaDB), BM25Index
│   ├── retrieval/        # Dense / BM25 / Hybrid (RRF) / Reranker
│   ├── generation/       # LLMClient, MockLLMClient, RAGPipeline, RAGCache
│   └── evaluation/       # Retrieval metrics, generation metrics, benchmark
├── scripts/
│   ├── ingest.py                  # Step 1 — download & clean data
│   ├── build_indexes.py           # Step 2 — chunk & index
│   ├── run_retrieval_eval.py      # Step 3 — benchmark retrieval
│   ├── generate_eval_report.py    # Step 4 — render HTML report
│   └── run_all.py                 # Orchestrator (--from N to skip steps)
├── ui/
│   └── chat_app.py       # Streamlit demo (transparent RAG UI)
├── data/
│   ├── raw/              # Downloaded datasets (git-ignored)
│   ├── processed/        # Cleaned JSONL + eval_queries.jsonl
│   ├── chunks/           # Parquet chunk files
│   └── indexes/          # ChromaDB + BM25 pickle indexes
├── reports/
│   ├── retrieval_results.csv
│   └── eval_report.html
├── configs/config.yaml
└── requirements.txt
```

---

## Quick Start (6 Steps)

```bash
# 1. Clone and enter
git clone <repo-url> && cd RAGScope

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Add OpenAI key for real LLM calls
cp .env.example .env
# Edit .env → set OPENAI_API_KEY=sk-...
# Skip this step to use MockLLMClient (no API key needed)

# 5. Run the full pipeline
python scripts/run_all.py
# Or resume from a specific step:
# python scripts/run_all.py --from 3

# 6. Launch the interactive UI
streamlit run ui/chat_app.py
```

---

## Retrieval Evaluation Results

Evaluated on 200 sampled SQuAD + arXiv queries, fixed-512 chunking strategy.

| Config         | Recall@5 | Recall@10 | MRR   | NDCG@10 | Latency  |
|----------------|----------|-----------|-------|---------|----------|
| Dense          | 0.865    | 0.900     | 0.757 | 0.792   | 35 ms    |
| BM25           | 0.845    | 0.905     | 0.726 | 0.770   | 40 ms    |
| Hybrid (RRF)   | 0.915    | 0.950     | 0.788 | 0.828   | 50 ms    |
| Hybrid+Rerank  | 0.885    | 0.950     | 0.678 | 0.745   | 1 736 ms |

> Full interactive report: `reports/eval_report.html`
> (run `scripts/generate_eval_report.py` to regenerate)

---

## Key Findings

**Hybrid RRF leads on recall and NDCG**
Combining dense and sparse signals via Reciprocal Rank Fusion (α = 0.5) achieves the best Recall@5 (0.915) and NDCG@10 (0.828), outperforming either retriever alone by 5–7%.

**Reranker raises Recall@10 but hurts MRR**
Adding a cross-encoder reranker (BAAI/bge-reranker-base) matches Hybrid's Recall@10 (0.950) but drops MRR from 0.788 to 0.678, suggesting the model still struggles with exact top-1 precision. Latency jumps to 1 736 ms — suitable for offline batch evaluation, not real-time serving.

**BM25 is a strong baseline for rare-term queries**
BM25 latency (∼40 ms) is comparable to Dense retrieval and excels on queries with specific technical terms or proper nouns. In hybrid configurations it consistently adds signal not captured by embedding similarity alone.

---

## Module Overview

| Module | Key Classes | Description |
|--------|-------------|-------------|
| `ingestion` | `SQuADLoader`, `ArXivLoader`, `DocumentCleaner` | Download, deduplicate, NFKC-clean |
| `ingestion` | `FixedSizeChunker`, `SentenceAwareChunker`, `SemanticChunker` | Three chunking strategies |
| `indexing` | `EmbeddingService` | BGE-small embeddings, diskcache-backed |
| `indexing` | `DenseIndex`, `BM25Index` | ChromaDB (cosine) + rank_bm25 |
| `retrieval` | `DenseRetriever`, `BM25Retriever` | Single-modal retrievers |
| `retrieval` | `HybridRetriever`, `Reranker` | RRF fusion + CrossEncoder reranking |
| `generation` | `RAGPipeline`, `RAGCache` | End-to-end pipeline with diskcache |
| `generation` | `LLMClient`, `MockLLMClient` | OpenAI wrapper + offline mock |
| `evaluation` | `run_retrieval_benchmark` | Multi-config benchmark runner |
| `evaluation` | `recall_at_k`, `ndcg_at_k`, `mrr`, … | Retrieval metrics |
| `evaluation` | `exact_match`, `token_f1`, `rouge_l` | Generation metrics |

---

## Configuration

All key parameters live in [`configs/config.yaml`](configs/config.yaml).
API keys and paths go in `.env` (copy from `.env.example`).

```yaml
# configs/config.yaml (excerpt)
chunking:
  fixed:
    chunk_size: 512
    overlap: 64
embedding:
  model: BAAI/bge-small-en-v1.5
retrieval:
  hybrid_alpha: 0.5   # 1.0 = pure dense, 0.0 = pure BM25
```
