"""
RAGScope Chat Demo — Streamlit single-page application.

Run from the project root:
    streamlit run ui/chat_app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as go
import streamlit as st

# ── Path bootstrap ────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Page config (MUST be the first Streamlit call) ────────────────────────
st.set_page_config(
    page_title="RAGScope",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Late imports (after path is set) ─────────────────────────────────────
from ragscope.generation.llm_client import MockLLMClient, OllamaLLMClient  # noqa: E402
from ragscope.generation.pipeline import RAGPipeline, RAGResult   # noqa: E402
from ragscope.indexing.embedding import EmbeddingService           # noqa: E402
from ragscope.retrieval.bm25_retriever import BM25Retriever        # noqa: E402
from ragscope.retrieval.dense_retriever import DenseRetriever      # noqa: E402
from ragscope.retrieval.hybrid_retriever import HybridRetriever    # noqa: E402
from ragscope.retrieval.reranker import Reranker                   # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRESET_QUESTIONS: List[str] = [
    "What is machine learning?",
    "How does attention mechanism work in transformers?",
    "What are the applications of reinforcement learning?",
]

RETRIEVER_OPTIONS = ["Dense", "BM25", "Hybrid", "Hybrid+Rerank"]

_COMPARISON_RETRIEVERS = ["Dense", "BM25", "Hybrid"]

# ---------------------------------------------------------------------------
# Wrapper: adds reranking transparently inside retrieve()
# ---------------------------------------------------------------------------

class _RerankedRetriever:
    """Wraps a base retriever with cross-encoder reranking."""

    def __init__(self, retriever, reranker: Reranker) -> None:
        self._retriever = retriever
        self._reranker  = reranker

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        candidates = self._retriever.retrieve(query, k=k * 2)
        return self._reranker.rerank(query, candidates, top_k=k)


# ---------------------------------------------------------------------------
# Ollama availability probe (cached 30 s to avoid hammering on every render)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30, show_spinner=False)
def _ollama_available(base_url: str = "http://localhost:11434") -> bool:
    import requests
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Cached resource loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_all_resources() -> Dict:
    """Load every model/index once; cache for the session lifetime."""
    resources: Dict = {}

    with st.spinner("⏳ 加载 embedding 模型…"):
        emb = EmbeddingService()

    with st.spinner("⏳ 加载 Dense 索引…"):
        dense = DenseRetriever(chunking_strategy="fixed", embedding_service=emb)

    with st.spinner("⏳ 加载 BM25 索引…"):
        bm25 = BM25Retriever(chunking_strategy="fixed")

    hybrid = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25, alpha=0.5)

    with st.spinner("⏳ 加载 Reranker 模型…"):
        reranker = Reranker()

    reranked_hybrid = _RerankedRetriever(hybrid, reranker)
    llm = MockLLMClient()

    retriever_map = {
        "Dense":         dense,
        "BM25":          bm25,
        "Hybrid":        hybrid,
        "Hybrid+Rerank": reranked_hybrid,
    }
    pipeline_map = {
        name: RAGPipeline(ret, llm, use_cache=True)
        for name, ret in retriever_map.items()
    }

    resources["retrievers"] = retriever_map
    resources["pipelines"]  = pipeline_map
    return resources


# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------

def _confidence_gauge(confidence: float) -> go.Figure:
    color = (
        "#4CAF50" if confidence >= 0.7
        else "#FF9800" if confidence >= 0.4
        else "#F44336"
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(confidence * 100, 1),
        number={"suffix": "%", "font": {"size": 28}},
        gauge={
            "axis":  {"range": [0, 100], "tickwidth": 1},
            "bar":   {"color": color, "thickness": 0.3},
            "steps": [
                {"range": [0,  40], "color": "#ffcdd2"},
                {"range": [40, 70], "color": "#fff9c4"},
                {"range": [70, 100], "color": "#c8e6c9"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 2},
                "thickness": 0.75,
                "value": confidence * 100,
            },
        },
        title={"text": "Confidence", "font": {"size": 14}},
    ))
    fig.update_layout(height=180, margin=dict(t=40, b=5, l=15, r=15))
    return fig


def _latency_chart(retrieve_ms: float, llm_ms: float) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=[retrieve_ms, max(llm_ms, 0)],
        y=["Retrieve", "LLM"],
        orientation="h",
        marker_color=["#4C78A8", "#F58518"],
        text=[f"{retrieve_ms:.0f} ms", f"{max(llm_ms,0):.0f} ms"],
        textposition="auto",
    ))
    fig.update_layout(
        height=130,
        margin=dict(t=5, b=5, l=5, r=5),
        xaxis_title="ms",
        showlegend=False,
    )
    return fig


def _token_chart(prompt_tokens: int, completion_tokens: int) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=["Prompt", "Completion"],
        y=[prompt_tokens, completion_tokens],
        marker_color=["#4C78A8", "#54A24B"],
        text=[f"{prompt_tokens:,}", f"{completion_tokens:,}"],
        textposition="outside",
    ))
    fig.update_layout(height=180, margin=dict(t=10, b=5, l=5, r=5), showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "history":        [],     # List[Tuple[str, RAGResult, float, float]]
        "question_input": "",
        "last_result":    None,
        "last_retrieve_ms": 0.0,
        "last_llm_ms":    0.0,
        "last_question":  "",
        "last_llm_name":  "Mock",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    _init_state()

    # ── Load resources ────────────────────────────────────────────────────
    try:
        resources = _load_all_resources()
    except Exception as exc:
        st.error(
            f"**索引加载失败** — {exc}\n\n"
            "请先运行：\n```bash\npython scripts/ingest.py\n"
            "python scripts/build_indexes.py\n```"
        )
        st.stop()

    pipelines  = resources["pipelines"]
    retrievers = resources["retrievers"]

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🔍 RAGScope")
        st.caption("透明 RAG 问答演示系统")

        retriever_name: str = st.radio(
            "检索器",
            RETRIEVER_OPTIONS,
            index=0,
            help="选择底层检索策略",
        )

        _OLLAMA_LABEL = "Ollama (qwen2.5:7b)"
        llm_name: str = st.radio(
            "LLM",
            ["Mock", _OLLAMA_LABEL],
            index=0,
            help="Mock 无需 API key；Ollama 需要在 WSL 中运行 `ollama serve`",
        )
        if llm_name == _OLLAMA_LABEL and not _ollama_available():
            st.warning(
                "⚠️ Ollama 不可用（localhost:11434），已自动降级到 Mock。\n\n"
                "WSL 中运行：`ollama serve`",
                icon="🦙",
            )
            llm_name = "Mock (fallback)"

        k: int = st.slider("Top-k", min_value=1, max_value=10, value=5,
                           help="检索返回的 chunk 数量")

        show_chunks: bool = st.toggle("显示检索结果", value=True)
        show_cost:   bool = st.toggle("显示成本", value=True)

        st.divider()
        st.markdown("**💡 预设问题**")
        for pq in PRESET_QUESTIONS:
            if st.button(pq, use_container_width=True, key=f"preset_{pq}"):
                st.session_state["question_input"] = pq
                st.rerun()

        st.divider()
        if st.session_state["history"]:
            if st.button("🗑 清空历史", use_container_width=True):
                st.session_state["history"] = []
                st.session_state["last_result"] = None
                st.rerun()

    # ── Top-level columns ─────────────────────────────────────────────────
    col_qa, col_trans = st.columns([3, 2], gap="large")

    # ── Input area (col_qa top) ───────────────────────────────────────────
    with col_qa:
        st.subheader("💬 问答")
        question: str = st.text_input(
            "输入问题",
            key="question_input",
            placeholder="Ask anything about ML / AI…",
            label_visibility="collapsed",
        )
        submitted: bool = st.button(
            "Submit", type="primary", use_container_width=True
        )

    # ── Run pipeline on submission ────────────────────────────────────────
    error_msg: Optional[str] = None

    if submitted and question.strip():
        try:
            with st.spinner("🔍 检索中 + 🤖 生成中…"):
                retriever_obj = retrievers[retriever_name]

                # Select LLM client based on sidebar choice.
                # Ollama: disable cache (avoid key collisions with Mock results).
                # Mock / fallback: use the pre-built cached pipeline.
                use_ollama = llm_name.startswith("Ollama")
                if use_ollama:
                    active_llm      = OllamaLLMClient()
                    active_pipeline = RAGPipeline(retriever_obj, active_llm, use_cache=False)
                else:
                    active_pipeline = pipelines[retriever_name]  # Mock, cached

                # Time retrieve separately for the latency breakdown
                t0 = time.perf_counter()
                retriever_obj.retrieve(question, k=k)
                retrieve_ms = (time.perf_counter() - t0) * 1000

                result: RAGResult = active_pipeline.answer(question, k=k)
                llm_ms = max(0.0, result.latency_ms - retrieve_ms)

            # Push to history (keep last 5)
            entry = (question, result, retrieve_ms, llm_ms)
            st.session_state["history"] = (
                [entry] + st.session_state["history"]
            )[:5]
            st.session_state["last_result"]      = result
            st.session_state["last_retrieve_ms"] = retrieve_ms
            st.session_state["last_llm_ms"]      = llm_ms
            st.session_state["last_question"]    = question
            st.session_state["last_llm_name"]    = llm_name

        except Exception as exc:
            error_msg = str(exc)

    # ── Resolve what to display ───────────────────────────────────────────
    display_result: Optional[RAGResult] = st.session_state["last_result"]
    display_q:      str  = st.session_state["last_question"]
    display_r_ms:   float = st.session_state["last_retrieve_ms"]
    display_l_ms:   float = st.session_state["last_llm_ms"]

    # ── QA column ─────────────────────────────────────────────────────────
    with col_qa:
        if error_msg:
            st.error(f"⚠️ 出错了：{error_msg}")

        elif display_result is not None:
            res = display_result

            # Answer
            st.markdown("#### 📝 答案")
            st.markdown(
                f"<div style='font-size:1.1rem; line-height:1.7; "
                f"padding:12px; background:#f8f9fa; border-radius:8px; "
                f"border-left:4px solid #4C78A8'>{res.answer}</div>",
                unsafe_allow_html=True,
            )

            st.write("")

            # Citations
            if res.citations:
                st.markdown("**🔖 引用来源**")
                cols_cit = st.columns(min(len(res.citations), 3))
                for i, cid in enumerate(res.citations):
                    cols_cit[i % 3].code(cid, language=None)

            st.write("")

            # Confidence gauge
            st.plotly_chart(
                _confidence_gauge(res.confidence),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        else:
            st.info("在上方输入问题，点击 **Submit** 开始。")

        # ── Conversation history ──────────────────────────────────────────
        history = st.session_state["history"]
        if len(history) > 1:
            with st.expander(f"🕑 最近对话历史（{len(history) - 1} 条）", expanded=False):
                for q_h, r_h, _, _ in history[1:]:
                    st.markdown(f"**Q:** {q_h}")
                    preview = r_h.answer
                    if len(preview) > 200:
                        preview = preview[:200] + "…"
                    st.markdown(f"**A:** {preview}")
                    st.caption(
                        f"Confidence: {r_h.confidence:.0%} | "
                        f"Cache: {'✅' if r_h.cache_hit else '❌'} | "
                        f"Latency: {r_h.latency_ms:.0f} ms"
                    )
                    st.divider()

    # ── Transparency column ───────────────────────────────────────────────
    with col_trans:
        st.subheader("🔬 透明面板")

        if display_result is None:
            st.info("提交问题后，这里会显示检索细节、延迟和成本。")
        else:
            res = display_result

            # Cache status
            cache_label = "✅ Cache 命中" if res.cache_hit else "❌ Cache 未命中"
            cache_color = "#c8e6c9" if res.cache_hit else "#ffcdd2"
            st.markdown(
                f"<div style='padding:6px 12px; background:{cache_color}; "
                f"border-radius:6px; margin-bottom:8px'>{cache_label}</div>",
                unsafe_allow_html=True,
            )

            # ── Retrieved chunks ──────────────────────────────────────────
            if show_chunks and res.retrieved_chunks:
                st.markdown("##### 📄 检索到的 Chunks")
                for i, chunk in enumerate(res.retrieved_chunks):
                    score_val = chunk.get("rerank_score", chunk.get("score", 0.0))
                    score_label = (
                        f"rerank_score: {score_val:.4f}"
                        if "rerank_score" in chunk
                        else f"score: {score_val:.4f}"
                    )
                    with st.expander(
                        f"#{i+1}  {chunk['chunk_id']}  —  {score_label}",
                        expanded=(i == 0),
                    ):
                        st.caption(f"**doc_id**: {chunk.get('doc_id', '—')}")
                        st.text(chunk["text"][:500] + ("…" if len(chunk["text"]) > 500 else ""))

            # ── Latency breakdown ─────────────────────────────────────────
            st.markdown("##### ⏱ 延迟分解")
            st.plotly_chart(
                _latency_chart(display_r_ms, display_l_ms),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.caption(f"Total: **{res.latency_ms:.0f} ms**")

            # ── Token usage & cost ────────────────────────────────────────
            if show_cost:
                st.markdown("##### 💰 Token 用量与成本")
                st.plotly_chart(
                    _token_chart(res.prompt_tokens, res.completion_tokens),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
                total_tokens = res.prompt_tokens + res.completion_tokens
                c1, c2, c3 = st.columns(3)
                c1.metric("Prompt",     f"{res.prompt_tokens:,}")
                c2.metric("Completion", f"{res.completion_tokens:,}")
                c3.metric("Total",      f"{total_tokens:,}")
                if res.cost_usd > 0:
                    st.metric("预估费用", f"${res.cost_usd:.6f} USD")
                elif st.session_state.get("last_llm_name", "Mock").startswith("Ollama"):
                    st.caption("🦙 Ollama 本地模型 — 免费")
                else:
                    st.caption("🤖 MockLLMClient — 无实际费用")

    # ── Bottom: retriever comparison ──────────────────────────────────────
    if display_result is not None and display_q:
        st.divider()
        with st.expander("🆚 检索器横向对比（Dense / BM25 / Hybrid）", expanded=False):
            comp_cols = st.columns(3)
            for col, name in zip(comp_cols, _COMPARISON_RETRIEVERS):
                with col:
                    st.markdown(f"#### {name}")
                    try:
                        with st.spinner(f"{name} 检索中…"):
                            comp_result: RAGResult = pipelines[name].answer(
                                display_q, k=5
                            )
                        st.markdown(
                            f"<div style='font-size:0.9rem; padding:8px; "
                            f"background:#f8f9fa; border-radius:6px; "
                            f"min-height:80px'>{comp_result.answer}</div>",
                            unsafe_allow_html=True,
                        )
                        st.caption(
                            f"Confidence: {comp_result.confidence:.0%} | "
                            f"Latency: {comp_result.latency_ms:.0f} ms"
                        )
                        st.markdown("**Top-3 Chunks**")
                        for chunk in comp_result.retrieved_chunks[:3]:
                            score = chunk.get("score", 0.0)
                            st.markdown(
                                f"<div style='background:#eef2ff; border-radius:4px; "
                                f"padding:6px; margin-bottom:4px; font-size:0.82rem'>"
                                f"<code>{chunk['chunk_id']}</code> &nbsp; "
                                f"<span style='color:#666'>score: {score:.4f}</span><br>"
                                f"{chunk['text'][:140]}…</div>",
                                unsafe_allow_html=True,
                            )
                    except Exception as exc:
                        st.error(f"检索失败：{exc}")


if __name__ == "__main__":
    main()
