"""Assemble the Chinese final report from verified, actual local artifacts.

This is a read-only consumer of experiments. It never calls a model, chooses a
configuration, supplies missing measurements, or declares final acceptance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragscope.experiments.analysis_report import markdown_table, validated_rows
from ragscope.experiments.generation_campaign import load_campaign
from ragscope.experiments.runtime import configure, runtime_home
from ragscope.indexing.identity import artifact, atomic_json, verify_artifacts


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def fmt(value, digits=4):
    return "N/A" if value is None else f"{value:.{digits}f}"


def observed(path, title, keys, sources):
    """Missing outputs stay missing; the presence of a CLI is not evidence."""
    if not path.is_file():
        return f"### {title}\n\n状态：**UNAVAILABLE**。尚无 `{path}` 的实际结果，不能据接口支持判断完成。\n"
    data = read(path)
    sources.append(artifact(path))
    selected = {key: data[key] for key in keys if key in data}
    return f"### {title}\n\n来源：`{path}`。\n\n```json\n{json.dumps(selected, ensure_ascii=False, indent=2)}\n```\n"


def build(campaign, analysis_path, output, *, allow_incomplete=False):
    campaign, analysis_path, output = map(Path, (campaign, analysis_path, output))
    project = Path(__file__).resolve().parents[1]
    home = runtime_home()
    manifest, inputs = load_campaign(campaign)
    experiments = {group["experiment_id"] for group in manifest["groups"]}
    if output.name == "FINAL_REPORT.md" and not {
        "E2", "E3", "E4", "E5", "E6", "E7", "ROUTE_TRAIN", "ROUTE_DEV"
    } <= experiments:
        raise ValueError("A partial shard cannot become the whole-project FINAL_REPORT.md")
    report = read(analysis_path)
    if report["manifest_hash"] != manifest["manifest_hash"]:
        raise ValueError("Analysis belongs to another immutable manifest")
    if not verify_artifacts(report["sources"]):
        raise ValueError("Analysis source artifacts changed; rebuild it before reporting")
    rows, _ = validated_rows(campaign, manifest, inputs)
    if report["n_observed"] != len(rows) or report["n_expected_runnable"] != len(manifest["tasks"]):
        raise ValueError("Analysis denominator differs from actual verified records")
    if not allow_incomplete and report["status"] != "COMPLETE":
        raise ValueError("Incomplete experiments require explicit --allow-incomplete and truthful reporting")
    if output.exists() or output.with_suffix(".build.json").exists():
        raise FileExistsError("Existing report/receipt is preserved; choose a new output path")
    base = home / "runs" / manifest["provenance"]["spec"]["campaign"]
    retrieval = home / "runs" / (base.name + "_retrieval")
    routing = home / "runs" / (base.name + "_routing")
    training = home / "runs" / (base.name + "_lora") / "train_2b_1024_seed42"
    sources = [artifact(analysis_path), artifact(campaign / "EXPERIMENT_MANIFEST.json"),
               artifact(campaign / "FROZEN_INPUTS.json")]
    environment = read(base / "environment.json")
    sources.append(artifact(base / "environment.json"))
    spec = manifest["provenance"]["spec"]
    # Only a referenced and already-validated shard can supply a sidecar proxy.
    judge_paths = [campaign / "support_judge/SUMMARY.json"]
    for reference in manifest.get("provenance", {}).get("external_campaigns", []):
        shard = Path(reference["manifest"]["path"]).parent
        judge_paths.append(shard / "support_judge/SUMMARY.json")
    actual_judges = [path for path in judge_paths if path.is_file()]
    if len(actual_judges) > 1:
        raise ValueError("Multiple support-proxy summaries require an explicit source decision")
    judge_path = actual_judges[0] if actual_judges else judge_paths[0]
    splits = []
    for label, directory in (("QASPER", Path(spec["corpus_dir"])),
                             ("SQuAD E4", Path(spec["corpus_dir"]) / "squad_e4")):
        path = directory / "SPLIT_MANIFEST.json"
        if not path.is_file():
            splits.append(f"{label}：UNAVAILABLE，未以预计样本代替实测。")
            continue
        frozen = read(path)
        files = [{"path": str(directory / name), **entry} for name, entry in frozen["files"].items()]
        if not verify_artifacts(files):
            raise ValueError(f"Frozen {label} corpus changed")
        sources.append(artifact(path))
        counts = {}
        for split in ("train", "dev", "test"):
            query_path = directory / f"queries_{split}.jsonl"
            if query_path.is_file():
                population = [json.loads(line) for line in query_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                counts[split] = {"n": len(population), "answerable": sum(q["answerability"] == "answerable" for q in population),
                                 "unanswerable": sum(q["answerability"] == "unanswerable" for q in population)}
        splits.append(f"{label} 的实际冻结人群：`{json.dumps(counts, ensure_ascii=False)}`；来源 `{path}`。")
    title = "最终实验报告" if output.name == "FINAL_REPORT.md" else "报告构建预览（不是最终交付）"
    parts = [f"# RAGScope v2 {title}", "",
        f"生成于 {datetime.now(timezone.utc).isoformat()}。本次清单包含 {', '.join(sorted(experiments))}，"
        f"该清单的真实实验汇总状态：**{report['status']}**。",
        "本报告由逐行记录及不可变来源重建，验收结论另见 ACCEPTANCE.json；本报告生成成功不代表十五项验收全部通过。",
        "", "## 任务与实际环境", "",
        "目标是在英文公开资料上比较切块、检索、上下文、模型规模、路由及有限LoRA。界面和讲解使用中文；中文边界夹具只验证文本处理，不是中文问答质量基准。",
        f"环境体检记录：{environment['platform']}；Python {environment['python'].split()[0]}；"
        f"{environment['logical_cpus']} 个逻辑CPU；物理内存 {environment['memory']['total']/2**30:.2f} GiB。",
        "GPU体检原文：", "```text", environment["nvidia_smi"]["stdout"].strip(), "```", "",
        "隔离环境、模型、语料、缓存、索引、日志和断点使用 D:/RAGScope_Runtime。后续CPU计算显式4线程与空CUDA掩码；较早12线程的历史耗时不能改写为4线程实测。",
        "", "## 数据与工程修复", "", *splits, "",
        "保留规范文本坐标、证据参考集合及真实不可回答标签。多标注冲突隔离并保留来源，论文在train/dev/test之间互斥；SQuAD按原始文章聚类，不把同文多题当成独立来源。",
        "索引与缓存绑定数据、切块、tokenizer、embedding、query指令和配置身份。完整文本用包含query的512-token窗口进行BGE重排；论文块不足50时显式保存短缺，不复制候选补数。",
        "simple只为固定test论文编码，仍保留全509篇的BM25统计。长块、overlap、父子关系及证据映射有性质测试；原旧索引和报告保留独立hash核验。",
        "", "## 实验协议与完整结果", "",
        "E1用同一CPU embedding进行dev选型；E2为三主模型×四检索方法×200test；E3为相同test的Oracle和No Context；E4为独立SQuAD100；E5为LFM两路径100；E6为两个模型的简单/结构切块100；E7为生成前固定的证据位置诊断；E8为100题全库检索，不额外调用生成器。ROUTE的200train/60dev三路径用于真实代理标签。实际N与阻断以冻结清单和下表为准。",
        "主生成统一greedy、512输出上限、关闭thinking、Q4_K_M及各自核验过的聊天模板。所有主模型收到同一重打包上下文，同时检查2048参考预算和各真实模型窗口；实际prompt token数与参考token数单列。",
        "引用ID受可见上下文的JSON schema enum约束，因此合法ID率是工程性质，不能视为事实支持性改善。",
        "", markdown_table(report), "", "## 统计、长度与资源", "",
        "论文/原始文章配对cluster bootstrap固定2000次、95%区间。多重对照全部标为探索性，不据未校正区间宣称显著或因果。完整区间、预声明对照及失败分母见 ANALYSIS.json。",
        "质量采用文字重合F1及实际上下文证据覆盖；不可答F1和证据指标为N/A，拒答另算。请求失败留在适用质量分母并记零，同时提供成功条件统计。",
        "|实验|模型|方法|实际输入token P50|decode token/s P50|采样设备显存GiB|服务树RSS GiB|",
        "|---|---|---|---:|---:|---:|---:|" ]
    identity_lines = ["### 冻结模型身份", "", "|模型ID|实际本地模型tag|量化|后端版本|",
                      "|---|---|---|---|"]
    for model_id, identity in manifest["models"].items():
        identity_lines.append(f"|{model_id}|{identity['model']}|{identity.get('quantization', 'UNKNOWN')}|"
                              f"{identity.get('backend_version', 'UNKNOWN')}|")
    identity_lines += ["", "权重、revision/本地manifest和模板身份：", ""]
    for model_id, identity in manifest["models"].items():
        identity_lines.append(f"- `{model_id}`：weight SHA `{identity.get('weight_sha256', 'UNKNOWN')}`；"
            f"revision `{identity.get('revision', 'UNKNOWN')}`；template SHA `{identity.get('chat_template_hash', 'UNKNOWN')}`。")
    identity_lines += ["", "Qwen与LFM的模型结构、模板和tokenizer不同，异构对照单独报告。LFM许可限制见固定模型包来源/模型卡，本轮是本地个人研究，不声称商业部署许可。", ""]
    # Keep the audit's exact nine-column generation table contiguous and unique.
    table_index = next(i for i, text in enumerate(parts) if text.startswith("# RAGScope v2 逐行生成结果重建"))
    parts[table_index:table_index] = identity_lines
    dev_path = project / "reports/v2/retrieval_interim/DEV_RESULTS.md"
    if dev_path.is_file():
        sources.append(artifact(dev_path))
        dev_text = dev_path.read_text(encoding="utf-8")
        dev_text = dev_text.replace("# 已完成的 dev 检索结果", "### E1：已完成的dev切块与上下文结果", 1)
        for filename in ("dev_evidence_matrix.png", "dev_expansion_delta.png", "dev_r512_neighbor_changes.png"):
            dev_text = dev_text.replace(f"]({filename})", f"](retrieval_interim/{filename})")
            figure = dev_path.parent / filename
            if figure.is_file():
                sources.append(artifact(figure))
        parts[table_index:table_index] = [dev_text, ""]
    for group in report["groups"]:
        p = group["performance"]
        memory, rss = p.get("peak_sampled_device_total_used_bytes"), p.get("peak_sampled_service_tree_rss_bytes")
        parts.append(f"|{group['experiment_id']}|{group['model_id']}|{group['method']}|"
            f"{fmt(p.get('backend_input_tokens_p50'), 1)}|{fmt(p.get('decode_tokens_per_second_p50'), 2)}|"
            f"{fmt(memory/2**30 if memory is not None else None, 3)}|{fmt(rss/2**30 if rss is not None else None, 3)}|")
    parts += ["", "### 按规范文档长度分组", "",
        "短文≤512、中等513–4096、长文>4096参考tokens，按切块前文档定义；与收到的上下文长度分开。各小组为描述性统计，失败保留，未知长度显示unknown。", "",
        "|实验|模型|方法|文档长度组|已记录N|F1适用N|失败|文字F1|生成P50 ms|",
        "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for group in report["groups"]:
        for length in group["document_length_groups"]:
            metric = length["token_f1"]
            parts.append(f"|{group['experiment_id']}|{group['model_id']}|{group['method']}|"
                f"{length['canonical_document_length_group']}|{length['n_total']}|{metric['n_applicable']}|"
                f"{length['n_failed']}|{fmt(metric['overall'])}|{fmt(length['latency_ms']['p50'])}|")
    parts += ["", "### 失败分母、成功条件与不可答拒答", "",
        "|实验|模型|方法|已记录/计划|成功N|可答适用N|总体F1|成功条件F1|拒答precision|拒答recall|拒答F1|",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group in report["groups"]:
        s = group["observed_summary"]; m = s["token_f1"]; abstain = s["abstention"]
        parts.append(f"|{group['experiment_id']}|{group['model_id']}|{group['method']}|"
            f"{group['n_observed']}/{group['n_expected']}|{s['n_success']}|{m['n_applicable']}|"
            f"{fmt(m['overall'])}|{fmt(m['success_conditional'])}|{fmt(abstain['precision'])}|"
            f"{fmt(abstain['recall'])}|{fmt(abstain['f1'])}|")
    parts += ["", "生成P50/P95从冻结上下文开始计时，不含检索；路由组件重放之和与真实在线总时间分开。非流式调用无实测首token，保持N/A。显存是整个设备的采样量，包含其他进程；采样可能漏瞬时峰值，不能写成模型最低显存。CPU并行任务和历史负载条件不同，耗时是各运行条件下的描述性观测。", "",
        "## 路由与微调是否值得启用", "",
        "路由只读取问题和用户选择的范围等可见元数据。训练标签来自同一最小模型的三条真实路径，F1差≤0.01时优先较低实测成本，其后才是证据覆盖。它是文字重合与成本的代理最优标签，不能称为真实任务最优。所有策略的test决策在访问test路径分数之前固定。",
        "UI改变默认路径需要完整60dev质量下降≤0.01及至少30条真实在线P50改善≥10%，且实际部署条件吻合。未达门槛继续固定路径，不用test挑默认。",
        observed(training / "training_result.json", "BF16 LoRA训练", ("status", "actual_optimizer_steps", "best_dev_loss", "best_checkpoint_available", "seed"), sources),
        observed(training / "paired_test100/SUMMARY.json", "相同BF16基础模型与adapter的100题对照", ("status", "expected_n_per_condition", "baseline", "adapted", "completed", "pending", "error", "identity", "adapter"), sources),
        "两个BF16条件共用后端、模板、上下文和精度；prompt-only JSON与Q4主矩阵的schema约束不同，跨协议差异不能全部归因于精度或微调。训练loss下降本身不证明QA改善。"]
    for name in ("fixed_bm25", "fixed_hybrid", "fixed_hybrid_rerank", "rules", "linear", "mlp", "lstm", "laya"):
        path = routing / f"REPLAY_{name}.json"
        if path.is_file():
            data = read(path); sources.append(artifact(path))
            parts += [f"### 路由 {name}", "", "组件计时重放：", "```json",
                json.dumps({split: g["summary"] for split, g in data["groups"].items()}, ensure_ascii=False, indent=2), "```", ""]
        else:
            parts += [f"路由 `{name}`：实际回放结果UNAVAILABLE；入口实现不计为实验完成。", ""]
    parts += ["## 错误分析与支持性代理", "",
        "逐条错误、timeout、长度截断JSON及拒答均保留；没检到文档、漏证据、预算丢证据、排序噪音、证据齐但文字F1低、缺引用、无效JSON和资源错误分开。无法确认的语义原因标uncertain，不把低F1直接当作幻觉。",
        "Oracle只提供已标注证据，证据覆盖1并不保证参考答案所有事实已在提示中。No Context作为独立诊断，其空上下文引用覆盖不适用；不能移除这类行来抬高总体成绩。",
        "详细样例见 ERROR_CASES.md/JSON；导出的人工审核字段保持NOT_REVIEWED。自动支持性裁判是proxy，同模型自评偏差保留，不能称为人工真值或已核验幻觉率。",
        observed(judge_path, "本地支持性proxy", ("status", "n_expected", "n_completed", "n_pending", "verdict_counts", "n_actual_judge_calls", "n_same_model_bias", "n_failed_or_unassessable", "interpretation"), sources),
        "## 检索、资源诊断与部署证据", "",
        observed(retrieval / "RETRIEVAL_STATISTICS.json", "检索统计", ("status", "test_used_for_selection", "comparisons"), sources),
        observed(home / "runs" / (base.name + "_context_stress") / "SUMMARY_4096.json", "4K dev压力诊断", ("status", "window", "n_expected", "n_observed", "reason", "quality_ranking_eligible"), sources),
        observed(home / "runs" / (base.name + "_context_stress") / "SUMMARY_8192.json", "8K dev压力诊断", ("status", "window", "n_expected", "n_observed", "reason", "quality_ranking_eligible"), sources),
        "压力诊断只使用固定dev人群，不混入主质量排名。UI仅127.0.0.1，缺安装/评测项显示真实状态，不在打开页面时下载或默认Mock。最终浏览器/HTTP实测范围见终验凭据。",
        "## 执行、恢复与已知限制", "",
        "S1/S2实际计算保持CPU-only，但S3/S4曾与未结束的S2重叠。严格阶段串行不满足，必须在criterion14披露。用户后续明确本任务优先，并已停止原训练及定时检查；日更未操作。",
        "原任务恢复遵循最新USER_MANUAL_RESUME：释放本项目模型、服务和锁后通知用户自行续跑，不自动重启原训练。WSL上限未改；外部主机重启原因UNKNOWN，本项目没有执行电脑或WSL重启。",
        "后端系统更新若与冻结版本不符则拒绝运行，使用D盘项目专用固定版，不改旧实验身份。运行中断从RecordStore恢复，保留失败与预算票据，不重复已落盘题。所有真实调用总上限10000，warmup、失败、UI、路由、LoRA与裁判均计入。",
        "本报告不声称中文QA、人工幻觉审核、独占设备最低硬件要求或test选型收益。原旧embedding权重未知的E0只算固定官方版本重建，不是精确历史复现。",
        "", "实际复算命令：", "```powershell",
        f"& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/analyze_generation_v2.py --campaign {campaign.as_posix()} --output reports/v2",
        f"& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/audit_final_report_v2.py --report {output.as_posix()} --campaign {campaign.as_posix()} --analysis {analysis_path.as_posix()} --corpus {Path(spec['corpus_dir']).as_posix()} --output D:/RAGScope_Runtime/runs/{base.name}/FINAL_REPORT_REVERSE_AUDIT.json",
        "```", "", "逐项命令、exit code、证据和限制见 ACCEPTANCE.json；从本报告表格抽取至少10数字独立回到原始结果，不拿报告验证报告。", "",
        "## 本次报告来源", ""]
    figures = []
    for name, label in (("main_quality", "模型与方法F1"), ("quality_latency", "质量与生成耗时"),
                        ("evidence_answer", "实际证据与答案F1"), ("context_stress_resource", "4K/8K实际资源诊断")):
        figure = output.parent / "figures" / (name + ".png")
        if figure.is_file():
            sources.append(artifact(figure))
            figures.append(f"![{label}](figures/{name}.png)")
        else:
            figures.append(f"{label}图：UNAVAILABLE，尚无实际图文件。")
    parts[-2:-2] = ["## 结果图", "", *figures, ""]
    parts.extend(f"- `{s['path']}` · SHA-256 `{s['sha256']}`" for s in sources)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")
    receipt = {"status": "REPORT_WRITTEN_ACCEPTANCE_NOT_IMPLIED", "analysis_status": report["status"],
        "n_observed": len(rows), "n_expected": len(manifest["tasks"]), "manifest_hash": manifest["manifest_hash"],
        "report": artifact(output), "sources": sources, "generation_calls": 0, "source": artifact(Path(__file__))}
    atomic_json(output.with_suffix(".build.json"), receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, default=Path("reports/v2/ANALYSIS.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/v2/FINAL_REPORT.md"))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    configure(cpu_only=True)
    result = build(args.campaign, args.analysis, args.output, allow_incomplete=args.allow_incomplete)
    print(json.dumps({k: result[k] for k in ("status", "analysis_status", "n_observed", "n_expected", "generation_calls")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
