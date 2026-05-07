"""
Generate reports/eval_report.html from reports/retrieval_results.csv.

Embeds three Plotly charts:
  1. Grouped bar  — Recall@5 / NDCG@10 / MRR per config
  2. Scatter      — Latency vs Recall@5 (labelled)
  3. Radar        — all metrics across all configs

Run:
    python scripts/generate_eval_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

INPUT_CSV   = _ROOT / "reports" / "retrieval_results.csv"
OUTPUT_HTML = _ROOT / "reports" / "eval_report.html"

# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

_COLORS = ["#4C78A8", "#54A24B", "#F58518", "#E45756"]

_RADAR_METRICS = ["Recall@5", "Recall@10", "HitRate@5", "MRR", "NDCG@10"]


def _bar_chart(df: pd.DataFrame) -> str:
    metrics = ["Recall@5", "NDCG@10", "MRR"]
    fig = go.Figure()
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=metric,
            x=df["Config"],
            y=df[metric],
            marker_color=_COLORS[i],
            text=[f"{v:.3f}" for v in df[metric]],
            textposition="outside",
        ))
    fig.update_layout(
        title="检索器性能对比 — Recall@5 / NDCG@10 / MRR",
        barmode="group",
        yaxis=dict(range=[0, 1.05], title="Score"),
        xaxis_title="配置",
        legend=dict(orientation="h", y=1.12),
        height=420,
        margin=dict(t=80, b=40),
        template="plotly_white",
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _scatter_chart(df: pd.DataFrame) -> str:
    fig = go.Figure()
    for i, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["Latency(ms)"]],
            y=[row["Recall@5"]],
            mode="markers+text",
            name=row["Config"],
            marker=dict(size=16, color=_COLORS[i % len(_COLORS)]),
            text=[row["Config"]],
            textposition="top center",
            showlegend=True,
        ))
    fig.update_layout(
        title="延迟 vs Recall@5（越靠左上越好）",
        xaxis=dict(title="Latency (ms)", type="log"),
        yaxis=dict(title="Recall@5", range=[0.7, 1.0]),
        height=420,
        template="plotly_white",
        legend=dict(orientation="h", y=1.12),
        margin=dict(t=80, b=40),
    )
    fig.add_annotation(
        x=0.98, y=0.02,
        xref="paper", yref="paper",
        text="← 快  /  好 ↑  (左上角最优)",
        showarrow=False, font=dict(color="grey", size=11),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _radar_chart(df: pd.DataFrame) -> str:
    fig = go.Figure()
    categories = _RADAR_METRICS + [_RADAR_METRICS[0]]  # close the polygon

    for i, (_, row) in enumerate(df.iterrows()):
        values = [row[m] for m in _RADAR_METRICS] + [row[_RADAR_METRICS[0]]]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name=row["Config"],
            line_color=_COLORS[i % len(_COLORS)],
            opacity=0.7,
        ))
    fig.update_layout(
        title="各检索器综合指标雷达图",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        legend=dict(orientation="h", y=-0.15),
        height=480,
        template="plotly_white",
        margin=dict(t=80, b=80),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>RAGScope — Retrieval Evaluation Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    max-width: 1100px;
    margin: 40px auto;
    padding: 0 24px;
    color: #1a1a2e;
    background: #fafafa;
  }}
  h1 {{ font-size: 2rem; border-bottom: 3px solid #4C78A8; padding-bottom: 10px; }}
  h2 {{ font-size: 1.3rem; color: #4C78A8; margin-top: 40px; }}
  .chart-wrapper {{
    background: #fff;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,.08);
    padding: 20px;
    margin: 24px 0;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 0.9rem;
    margin: 16px 0;
  }}
  th {{
    background: #4C78A8;
    color: #fff;
    padding: 10px 14px;
    text-align: left;
  }}
  td {{ padding: 8px 14px; border-bottom: 1px solid #e0e0e0; }}
  tr:nth-child(even) {{ background: #f5f8ff; }}
  .analysis {{
    background: #eef4ff;
    border-left: 4px solid #4C78A8;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin: 20px 0;
    line-height: 1.8;
  }}
  .best {{ color: #2e7d32; font-weight: 600; }}
  .warn {{ color: #b71c1c; font-weight: 600; }}
  footer {{
    margin-top: 60px;
    font-size: 0.8rem;
    color: #aaa;
    text-align: center;
    border-top: 1px solid #ddd;
    padding-top: 16px;
  }}
</style>
</head>
<body>

<h1>🔍 RAGScope — 检索评估报告</h1>
<p>数据集：SQuAD / arXiv &nbsp;|&nbsp; 分块策略：Fixed-512 &nbsp;|&nbsp; 样本数：200 queries</p>

<h2>📊 评估结果汇总</h2>
{table_html}

<h2>图表一：核心指标对比</h2>
<div class="chart-wrapper">{bar_html}</div>

<h2>图表二：延迟 vs Recall@5 权衡</h2>
<div class="chart-wrapper">{scatter_html}</div>

<h2>图表三：综合指标雷达图</h2>
<div class="chart-wrapper">{radar_html}</div>

<h2>📝 分析结论</h2>
<div class="analysis">
  <p><strong>① Hybrid RRF 召回率最优：</strong>
  Hybrid 检索器（Reciprocal Rank Fusion, α=0.5）在 Recall@5 和 NDCG@10 上均优于
  单一的 Dense 或 BM25，验证了稠密与稀疏信号互补的价值。
  Recall@5 达到 <span class="best">0.915</span>，较 Dense 提升约 5.8%。</p>

  <p><strong>② Reranker 延迟过高，不适合实时场景：</strong>
  Hybrid+Rerank 配置的端到端延迟高达 <span class="warn">1 735 ms</span>，
  是纯 Hybrid 方案的 30 倍以上。尽管 Recall@10 与 Hybrid 持平（0.950），
  但 MRR 从 0.788 下降至 0.678，说明 cross-encoder 重排序在精确位次上
  仍存在挑战。建议仅在离线批量场景或对延迟不敏感的业务中使用。</p>

  <p><strong>③ BM25 稀有词优势：</strong>
  BM25 在 Recall@10（0.905）上与 Dense 相近，且延迟（∼40 ms）与 Dense 几乎持平。
  对于含专有名词或罕见术语的查询，BM25 的精确词匹配更具优势，
  可作为 Hybrid 的重要补充信号。</p>
</div>

<footer>Generated by RAGScope &nbsp;·&nbsp; scripts/generate_eval_report.py</footer>
</body>
</html>
"""


def _df_to_html_table(df: pd.DataFrame) -> str:
    metric_cols = ["Recall@5", "Recall@10", "HitRate@5", "MRR", "NDCG@10", "Latency(ms)"]
    rows = ""
    for _, row in df.iterrows():
        cells = f"<td><strong>{row['Config']}</strong></td>"
        for col in metric_cols:
            v = row[col]
            if col == "Latency(ms)":
                cells += f"<td>{v:.0f} ms</td>"
            else:
                cells += f"<td>{v:.3f}</td>"
        rows += f"<tr>{cells}</tr>\n"

    header = "".join(
        f"<th>{'Config' if c == 'Config' else c}</th>"
        for c in ["Config"] + metric_cols
    )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not INPUT_CSV.exists():
        print(f"[ERROR] {INPUT_CSV} not found. Run scripts/run_retrieval_eval.py first.")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")

    bar_html     = _bar_chart(df)
    scatter_html = _scatter_chart(df)
    radar_html   = _radar_chart(df)
    table_html   = _df_to_html_table(df)

    html = _HTML_TEMPLATE.format(
        bar_html=bar_html,
        scatter_html=scatter_html,
        radar_html=radar_html,
        table_html=table_html,
    )

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"Report written → {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
