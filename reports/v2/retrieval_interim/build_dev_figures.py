"""Freeze source-verified dev-only retrieval evidence and export PNG/SVG figures."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

from ragscope.indexing.identity import artifact, atomic_json, file_hash, signature, verify_artifacts
from scripts.audit_retrieval_v2 import independently_score, rows, METRICS


def frozen_json(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("Refusing to overwrite a changed frozen dev snapshot: " + str(path))
    else:
        atomic_json(path, value)


def prepare(home, campaign, output):
    run = home / "runs" / (campaign + "_retrieval")
    destination = run / "analysis_complete_dev"
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_path = destination / "SOURCE_SNAPSHOT.json"
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if not verify_artifacts(snapshot["sources"]):
            raise ValueError("Frozen dev figure sources changed")
        return snapshot, destination
    corpus = home / "corpora" / campaign
    split_path = corpus / "SPLIT_MANIFEST.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    for name in ("documents.jsonl", "queries_dev.jsonl"):
        if file_hash(corpus/name) != split["files"][name]["sha256"]:
            raise ValueError("Frozen dev corpus changed")
    queries = {q["query_id"]: q for q in rows(corpus / "queries_dev.jsonl")}
    assert len(queries) == 100 and all(q["split"] == "dev" for q in queries.values())
    doc_ids = {q["doc_id"] for q in queries.values()}
    documents = {d["doc_id"]: d for d in rows(corpus / "documents.jsonl") if d["doc_id"] in doc_ids}
    units = {u["unit_id"]: u for d in documents.values() for u in d["evidence_units"]}
    sources = [artifact(corpus / name) for name in ("SPLIT_MANIFEST.json", "documents.jsonl", "queries_dev.jsonl")]
    candidates_path, contexts_path = run / "CANDIDATE_SELECTION.json", run / "CONTEXT_SELECTION.json"
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    contexts = json.loads(contexts_path.read_text(encoding="utf-8"))
    assert candidates["split"] == contexts["split"] == "dev" and candidates["method"] == "hybrid"
    assert contexts["candidate_selection_sha256"] == file_hash(candidates_path)
    sources += [artifact(candidates_path), artifact(contexts_path), artifact(output / "CHART_CONTRACT.json")]
    primary = []
    for stage in ("bm25", "dense"):
        path = run / f"E1_{stage}_SUMMARY.json"
        summaries = json.loads(path.read_text(encoding="utf-8"))
        primary.extend(summaries["groups"])
        sources.append(artifact(path))
    assert len(primary) == 33 and len({r["run_id"] for r in primary}) == 33
    primary_ids = {r["run_id"] for r in primary}
    all_runs, cells, source_pieces = [], [], 0
    for summary_path in sorted((run / "summaries").glob("E1_*.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        record_path = Path(summary["records"])
        config_path = record_path.with_suffix(".config.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert summary["split"] == "dev" and set(config["query_ids"]) == set(queries)
        assert config["index_identity"]["split"] == "dev"
        config_hash = signature(config)
        sums, denominators = {k: 0.0 for k in METRICS}, {k: 0 for k in METRICS}
        ids, failures = set(), 0
        for row in rows(record_path):
            expected_hash = row.pop("record_hash")
            assert signature(row) == expected_hash and row["config_hash"] == config_hash
            assert row["query_id"] not in ids and row["split"] == "dev"
            ids.add(row["query_id"])
            failed = bool(row.get("error"))
            failures += failed
            bundle = row["context_bundle"]
            assert bundle.get("unit_spans", []) == [s for p in bundle.get("pieces", []) for s in p["unit_spans"]]
            for piece in bundle.get("pieces", []):
                start, end = piece["canonical_start"], piece["canonical_end"]
                assert documents[piece["doc_id"]]["text"][start:end] == piece["text"]
                for span in piece["unit_spans"]:
                    unit = units[span["unit_id"]]
                    assert span["doc_id"] == piece["doc_id"] == unit["doc_id"]
                    assert start <= span["start"] < span["end"] <= end
                    assert unit["canonical_start"] <= span["start"] < span["end"] <= unit["canonical_end"]
                source_pieces += 1
            expected = independently_score(queries[row["query_id"]], bundle.get("unit_spans", []), units, failed)
            for metric, value in expected.items():
                observed = row["scores"][metric]
                assert observed is None if value is None else abs(observed - value) <= 1e-12
                if value is not None:
                    sums[metric] += value
                    denominators[metric] += 1
        assert ids == set(queries) and len(ids) == summary["n_total"] == 100
        assert failures == summary["n_failed"]
        measured = {}
        for metric in METRICS:
            mean = sums[metric] / denominators[metric]
            assert abs(mean - summary[metric]["overall"]) <= 1e-12
            assert denominators[metric] == summary[metric]["n_applicable"] == 92
            measured[metric] = mean
            cells.append({"run_id": summary["run_id"], "metric": metric, "independent_sum": sums[metric],
                          "n_applicable": denominators[metric], "recomputed": mean,
                          "reported": summary[metric]["overall"], "match": True})
        all_runs.append({"run_id": summary["run_id"], "group": summary["group"], "method": summary["method"],
                         "expansion": summary["expansion"], "n_total": 100, "n_applicable": 92,
                         "n_na": 8, "n_failed": failures, "is_primary_matrix": summary["run_id"] in primary_ids,
                         "metrics": measured, "record_artifact": artifact(record_path), "config_hash": config_hash,
                         "mean_context_reference_tokens": summary["mean_context_reference_tokens"],
                         "mean_context_budget_use": summary["mean_context_budget_use"],
                         "mean_context_duplicate_ratio": summary["mean_context_duplicate_ratio"],
                         "mean_context_pieces": summary["mean_context_pieces"]})
        sources += [artifact(summary_path), artifact(record_path), artifact(config_path)]
    assert len(all_runs) == 41 and len(primary_ids) == 33
    chunk_statistics = []
    order = [r["group"]["id"] for r in primary if r["method"] == "bm25"]
    for group in order:
        matches = []
        for path in (home / "indexes" / campaign / "dev_chunks").glob("*/chunk_manifest.json"):
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest["identity"]["group"]["id"] == group:
                matches.append((path, manifest))
        assert len(matches) == 1
        path, manifest = matches[0]
        assert verify_artifacts(manifest["artifacts"])
        child_count = sum(1 for _ in rows(path.parent / "chunks.jsonl"))
        parent_count = sum(1 for _ in rows(path.parent / "parents.jsonl"))
        assert child_count == manifest["count"] and parent_count == manifest["parent_count"]
        chunk_statistics.append({"group": group, "children": child_count, "parents": parent_count,
                                 "body_token_max": manifest["body_token_max"],
                                 "body_token_mean": manifest["body_token_mean"],
                                 "chunk_build_seconds": manifest["build_seconds"],
                                 "build_time_semantics": "Chunk construction only, excluding model encoding",
                                 "group_parameters": manifest["identity"]["group"]})
        sources += [artifact(path)] + manifest["artifacts"]
    snapshot = {"status": "COMPLETE_DEV_ONLY", "campaign": campaign, "data_split": "dev",
                "question_count": 100, "document_count": len(documents), "n_answerable": 92, "n_na": 8,
                "records_verified": len(all_runs)*100, "source_pieces_verified": source_pieces,
                "primary_matrix": primary, "runs": all_runs, "chunk_statistics": chunk_statistics,
                "group_order": order, "candidates": candidates, "contexts": contexts,
                "sources": sources, "aggregate_checks": cells,
                "seed42_sample_10_numbers": random.Random(42).sample(cells, 10),
                "metric_policy": "Best complete legal evidence reference set; failed answerable requests zero; unanswerable N/A",
                "independent_implementation": artifact(PROJECT / "scripts/audit_retrieval_v2.py"),
                "limit": "Descriptive dev selection evidence; no test rows, no unfinished full-corpus indexes, no significance claims"}
    frozen_json(snapshot_path, snapshot)
    frozen_json(destination / "INDEPENDENT_SAMPLE_10.json", {"status": "PASS", "seed": 42,
        "source_snapshot": artifact(snapshot_path), "sample": snapshot["seed42_sample_10_numbers"],
        "note": "All123 dev summary numbers independently rebuilt from canonical intervals; ten numbers selected for explicit inspection."})
    return snapshot, destination


def draw(snapshot, output, destination):
    existing_manifest = output / "FIGURE_MANIFEST.json"
    if existing_manifest.exists():
        existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if existing.get("status") == "PASS_VISUAL_QA":
            if not verify_artifacts(existing["files"] + [existing["plotter_artifact"]]):
                raise ValueError("Frozen validated figure or plotter changed")
            print(json.dumps({"status": "REUSED_VERIFIED_FIGURES", "figures": len(existing["files"])}))
            return
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt, font_manager
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.is_file():
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    blue, gold, ink, grid = "#3878A8", "#C99A38", "#22313B", "#E4E8EB"
    plt.rcParams.update({"font.size": 10, "text.color": ink, "axes.labelcolor": ink,
        "xtick.color": ink, "ytick.color": ink, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": ink, "grid.color": grid, "axes.axisbelow": True, "svg.fonttype": "path"})
    figures = []

    def save(fig, name, source_rows):
        for suffix in ("png", "svg"):
            path = output / (name + "." + suffix)
            fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
            figures.append(artifact(path))
        frozen_json(output / (name + ".data.json"), {"source_snapshot": artifact(destination / "SOURCE_SNAPSHOT.json"),
                   "plotted_data": source_rows, "scope": "dev only"})
        plt.close(fig)

    group_order, methods = snapshot["group_order"], ("bm25", "dense", "hybrid")
    primary = [r for r in snapshot["runs"] if r["is_primary_matrix"]]
    lookup = {(r["group"]["id"], r["method"]): r for r in primary}
    labels = [group + (" *" if group in snapshot["candidates"]["selected"].values() else "") for group in group_order]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 7.5), layout="constrained")
    cmap = LinearSegmentedColormap.from_list("declared_blue", ["#F4F8FB", blue])
    for ax, metric, title in zip(axes, ("full_reference_coverage", "continuous_coverage"), ("完整参考证据集合覆盖", "连续证据覆盖")):
        values = np.asarray([[lookup[(group, method)]["metrics"][metric] for method in methods] for group in group_order])
        im = ax.imshow(values, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax.set_xticks(range(3), ("BM25", "Dense", "Hybrid"))
        ax.set_yticks(range(11), labels)
        ax.set_title(title, fontsize=12, pad=12)
        for y in range(11):
            for x in range(3):
                ax.text(x, y, f"{values[y,x]:.3f}", ha="center", va="center", color=ink)
        ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 11, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)
    fig.colorbar(im, ax=axes, fraction=.025, pad=.025, label="覆盖比例（固定 0–1 色阶）")
    fig.suptitle("E1：切块方式与检索方法的 dev 证据覆盖", x=.01, ha="left", fontsize=16, fontweight="bold")
    fig.supxlabel("36 篇论文 · 每格 100 题（92 可答 / 8 N/A），0 失败 · * 按预声明 Hybrid 规则冻结的候选\n父子组此表用 parent 扩展，其余用 none；描述性 dev 选型结果，不代表 test 泛化或显著差异。", fontsize=9)
    save(fig, "dev_evidence_matrix", primary)

    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    groups = [snapshot["candidates"]["selected"][family] for family in ("simple", "structured")]
    context_lookup = {(r["group"]["id"], r["expansion"]): r for r in snapshot["runs"] if r["method"] == "hybrid"}
    deltas = []
    for i, group in enumerate(groups):
        for offset, metric, name, color, hatch in ((-.18, "full_reference_coverage", "完整集合覆盖", blue, None),
                                                  (.18, "continuous_coverage", "连续覆盖", gold, "///")):
            baseline = context_lookup[(group, "none")]["metrics"][metric]
            neighbor = context_lookup[(group, "neighbor")]["metrics"][metric]
            value = 100*(neighbor-baseline)
            ax.barh(i+offset, value, height=.3, color=color if hatch is None else "white", edgecolor=color,
                    hatch=hatch, linewidth=1.1, label=name if i == 0 else None)
            ax.text(value + (.25 if value >= 0 else -.25), i+offset, f"{value:+.2f} pp",
                    ha="left" if value >= 0 else "right", va="center", fontsize=10)
            deltas.append({"group": group, "metric": metric, "none": baseline, "neighbor": neighbor, "delta_percentage_points": value})
    ax.set_yticks(range(2), groups); ax.invert_yaxis(); ax.set_xlim(-8.7, 4.5)
    ax.axvline(0, color=ink, linewidth=.9); ax.grid(axis="x")
    ax.set_xlabel("neighbor − none（百分点，pp）")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.13), ncol=2, frameon=False)
    fig.suptitle("已选候选：邻块扩展的 dev 指标变化", x=.01, ha="left", fontsize=16, fontweight="bold")
    fig.supxlabel("相同 dev 100 题 / 条件（92 可答，8 N/A）· 2048 reference-token 预算含标题与引用标签\n完整集合覆盖上升，但连续覆盖下降；预声明规则优先完整集合覆盖，因此两组冻结为 neighbor。", fontsize=9)
    save(fig, "dev_expansion_delta", deltas)

    counts = snapshot["chunk_statistics"]
    fig, ax = plt.subplots(figsize=(10, 6.5), layout="constrained")
    y = np.arange(11)
    children, parents = [r["children"] for r in counts], [r["parents"] for r in counts]
    ax.barh(y, children, color=blue, edgecolor=ink, height=.62, linewidth=.6, label="子块（参与编码）")
    parent_positions = [i for i, count in enumerate(parents) if count]
    ax.barh(parent_positions, [parents[i] for i in parent_positions], left=[children[i] for i in parent_positions], color="white", edgecolor=gold, hatch="///", height=.62,
            linewidth=1, label="父块（扩展映射）")
    for pos, child, parent in zip(y, children, parents):
        label = str(child) if not parent else f"{child} + {parent} = {child+parent}"
        ax.text(child+parent+16, pos, label, va="center", fontsize=9)
    ax.set_yticks(y, group_order); ax.invert_yaxis(); ax.set_xlim(0, 1250); ax.grid(axis="x")
    ax.set_xlabel("记录数（子块 + 父块）")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.09), frameon=False, ncol=2)
    fig.suptitle("Dev 切块与父块记录数量", x=.01, ha="left", fontsize=16, fontweight="bold")
    fig.supxlabel("相同 36 篇 dev 论文 · 总计 5292 子块 + 333 父块 · 实际 Qwen tokenizer\n父块用于扩展映射，并非额外独立编码；这里的记录数不是正式 509 篇全库索引大小。", fontsize=9)
    save(fig, "dev_chunk_counts", counts)
    atomic_json(output / "FIGURE_MANIFEST.json", {"status": "RENDERED_AWAITING_VISUAL_QA", "files": figures,
                "source_snapshot": artifact(destination / "SOURCE_SNAPSHOT.json"), "matplotlib": matplotlib.__version__,
                "font": str(font), "matplotlib_cache": matplotlib.get_cachedir(), "scope": "dev only"})
    print(json.dumps({"figures": len(figures), "snapshot": str(destination / "SOURCE_SNAPSHOT.json"),
                      "dev_runs": len(snapshot["runs"]), "actual_rows": snapshot["records_verified"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path("D:/RAGScope_Runtime"))
    parser.add_argument("--campaign", default="20261014_v2")
    args = parser.parse_args()
    output = Path(__file__).resolve().parent
    snapshot, destination = prepare(args.home, args.campaign, output)
    draw(snapshot, output, destination)
