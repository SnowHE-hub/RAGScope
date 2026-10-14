"""Reproducible local report figures; do not turn partial groups into conclusions."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json

from ragscope.experiments.runtime import configure
configure(cpu_only=True)
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np

from ragscope.indexing.identity import artifact, atomic_json, signature, verify_artifacts

BLUE, INK, GRID = '#3878A8', '#22313B', '#E4E8EB'
FONT = Path('C:/Windows/Fonts/msyh.ttc')
if FONT.is_file():
    font_manager.fontManager.addfont(FONT)
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({'font.size': 10, 'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': INK, 'ytick.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': INK, 'grid.color': GRID, 'axes.axisbelow': True, 'svg.fonttype': 'path'})


def save(fig, output, name, data, source):
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ('png', 'svg'):
        path = output/f'{name}.{suffix}'
        fig.savefig(path, dpi=160, facecolor='white', bbox_inches='tight')
        paths.append(artifact(path))
    atomic_json(output/f'{name}.data.json', {'source': artifact(source), 'plotted_data': data,
        'files': paths, 'matplotlib': matplotlib.__version__, 'cache': matplotlib.get_cachedir()})
    plt.close(fig)


def pilot(source, output):
    report = json.loads(source.read_text(encoding='utf-8'))
    if report['status'] != 'PASS_COMPATIBILITY_NOT_QUALITY':
        raise ValueError('Completed compatibility audit required')
    groups = report['groups']
    if len(groups) != 4 or any(g['n_dev'] != 20 or g['resource_measured_rows'] != 20 for g in groups):
        raise ValueError('Expected all four measured final-setting pilots')
    labels = ['Qwen3.5 2B', 'Qwen3.5 4B', 'Qwen3.5 9B', 'LFM2.5 1.2B']
    values = [g['peak_sampled_device_total_used_bytes']/2**30 for g in groups]
    fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
    y = np.arange(4)
    ax.barh(y, values, color=BLUE, edgecolor=INK, linewidth=.7, height=.55)
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlim(0, 8)
    ax.set_xlabel('采样到的全设备显存峰值（GiB）'); ax.grid(axis='x')
    for i, (v, group) in enumerate(zip(values, groups)):
        ax.text(v+.12, i, f"{v:.2f} GiB · 解析成功 {20-group['parse_or_transport_errors']}/20", va='center', fontsize=9)
    fig.suptitle('最终模型兼容预检：显存观测', x=.01, ha='left', fontsize=15, fontweight='bold')
    ax.set_title('相同 20 个 dev 问题 / 模型 · Q4_K_M · 4096 上下文 · 512 输出上限\n仅兼容性预检；设备总量含其他进程，采样可能漏掉瞬时峰值', loc='left', fontsize=10, pad=15)
    save(fig, output, 'pilot_memory', groups, source)


def main_figures(source, output):
    report = json.loads(source.read_text(encoding='utf-8'))
    groups = [g for g in report['groups'] if g['experiment_id'] == 'E2']
    models, methods = ('qwen_2b', 'qwen_4b', 'qwen_9b'), ('R0', 'R1', 'R2', 'R3')
    lookup = {(g['model_id'], g['method']): g for g in groups}
    if set(lookup) != {(model, method) for model in models for method in methods} or any(g['status'] != 'COMPLETE' for g in groups):
        raise ValueError('Full3x4 E2matrix must be complete before final figures')
    if any(g['n_expected'] != 200 for g in groups):
        raise ValueError('Main figure population differs from frozen200test design')
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 5), layout='constrained', sharex=True, sharey=True)
    for ax, model in zip(axes, models):
        for i, method in enumerate(methods):
            group = lookup[model, method]; stats = group['estimates']['token_f1']; mean = stats['mean']
            ax.errorbar(mean, i, xerr=[[mean-stats['low']], [stats['high']-mean]], fmt='o', color=BLUE, capsize=4)
            ax.text(mean+.015, i-.17, f"{mean:.3f} (N={stats['n_rows']})", fontsize=9)
        ax.set_title(model.replace('qwen_', 'Qwen3.5 ').upper()); ax.set_xlim(0, 1)
        ax.set_yticks(range(4), ['BM25', 'Dense', 'Hybrid', 'Hybrid+重排']); ax.set_ylim(3.5, -.6); ax.grid(axis='x')
        ax.set_xlabel('答案 token F1（文字重合代理）')
    fig.suptitle('模型规模与检索方法：答案 F1', x=.01, ha='left', fontsize=15, fontweight='bold')
    fig.supxlabel('每条件 200 test，F1仅对可答题适用、失败记零；论文聚类 2,000 次重采样的探索性 95% 区间', fontsize=9)
    save(fig, output, 'main_quality', groups, source)
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    markers = dict(zip(methods, ['o', 's', '^', 'D']))
    colors = dict(zip(models, [BLUE, '#B06C36', '#54806A']))
    plotted = []
    for mi, model in enumerate(models):
        for method in methods:
            g = lookup[model, method]; latency = g['observed_summary']['latency_ms']['p50']/1000
            f1 = g['estimates']['token_f1']['mean']
            memory = g['performance'].get('peak_sampled_device_total_used_bytes')
            ax.scatter(latency, f1, marker=markers[method], edgecolor=colors[model], facecolor=colors[model], s=50+mi*22, alpha=.8)
            plotted.append({'model': model, 'method': method, 'generation_p50_s': latency, 'token_f1': f1, 'sampled_device_peak_bytes': memory})
    ax.set_xlim(left=0); ax.set_ylim(0, 1); ax.grid()
    model_legend = ax.legend(handles=[Line2D([0], [0], marker='o', color='none', markerfacecolor=colors[model],
        label='Qwen3.5 '+model.removeprefix('qwen_').upper()) for model in models], title='模型颜色', loc='upper right')
    ax.add_artist(model_legend)
    ax.legend(handles=[Line2D([0], [0], marker=markers[method], color='none', markeredgecolor=INK,
        markerfacecolor='white', label=label) for method, label in zip(methods, ['BM25', 'Dense', 'Hybrid', 'Hybrid+重排'])],
        title='检索标记', loc='upper left')
    ax.set_xlabel('冻结上下文后的生成 P50（秒；不含检索）'); ax.set_ylabel('答案 token F1')
    fig.suptitle('答案质量、生成耗时与显存', x=.01, ha='left', fontsize=15, fontweight='bold')
    ax.set_title('12 个模型×检索条件；每条件 200 test · 显存设备总量另见资源表\n模型驻留、统一预算；不推断首 token 延迟或纯硬件速度', loc='left', fontsize=10, pad=14)
    save(fig, output, 'quality_latency', plotted, source)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), layout='constrained', sharey=True)
    plotted = []
    for ax, model in zip(axes, models):
        for i, method in enumerate(methods):
            g = lookup[model, method]
            for j, label in enumerate(('fully_covered', 'not_fully_covered')):
                conditional = next(r for r in g['evidence_answer_relationship'] if r['context_group'] == label)
                metric = conditional['token_f1']; value, n = metric['overall'], metric['n_applicable']
                x = i+(j-.5)*.36
                if value is not None:
                    ax.bar(x, value, width=.34, color=BLUE if j == 0 else 'white', edgecolor=BLUE,
                           hatch=None if j == 0 else '//', linewidth=1,
                           label=('完整证据', '证据不完整')[j] if i == 0 else None)
                    ax.text(x, value+.018, f'n={n}', ha='center', fontsize=7)
                else:
                    ax.text(x, .02, 'N/A', rotation=90, ha='center', fontsize=7)
                plotted.append({'model': model, 'method': method, 'context_group': label, 'f1': value, 'n_applicable': n})
        ax.set_title(model.replace('qwen_', 'Qwen3.5 ').upper()); ax.set_ylim(0, 1)
        ax.set_xticks(range(4), methods); ax.grid(axis='y'); ax.set_xlabel('检索条件')
    axes[0].set_ylabel('答案 token F1'); axes[-1].legend(loc='upper right', fontsize=9)
    fig.suptitle('实际上下文证据覆盖与答案 F1', x=.01, ha='left', fontsize=15, fontweight='bold')
    fig.supxlabel('每条件 200 test，按实际证据覆盖分组、可答失败记零；条件样本数标在柱上。关联不等于因果或完整答案正确性。', fontsize=9)
    save(fig, output, 'evidence_answer', plotted, source)


def e3_figures(source, output):
    report = json.loads(source.read_text(encoding='utf-8'))
    groups = [g for g in report['groups'] if g['experiment_id'] == 'E3']
    models = ('qwen_2b', 'qwen_4b', 'qwen_9b')
    methods = ('no_context', 'oracle')
    lookup = {(g['model_id'], g['method']): g for g in groups}
    if (len(groups) != 6 or set(lookup) != {(m, r) for m in models for r in methods}
            or any(g['status'] != 'COMPLETE' or g['n_expected'] != 200 or g['n_observed'] != 200 for g in groups)):
        raise ValueError('E3 requires the complete three-model paired200 population')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
    colors = {'no_context': '#788690', 'oracle': BLUE}
    names = {'no_context': '无上下文', 'oracle': 'Oracle 证据上下文'}
    for method, offset in zip(methods, [-.14, .14]):
        for i, model in enumerate(models):
            group = lookup[model, method]
            stats = group['estimates']['token_f1']
            y = i + offset
            mean = stats['mean']
            axes[0].errorbar(mean, y, xerr=[[mean-stats['low']], [stats['high']-mean]],
                fmt='o', color=colors[method], capsize=4, label=names[method] if i == 0 else None)
            axes[0].text(mean+.018, y-.07, f'{mean:.3f}', fontsize=9)
            elapsed = group['observed_summary']['latency_ms']['p50']/1000
            axes[1].scatter(elapsed, y, color=colors[method], s=40)
            axes[1].text(elapsed+.03, y-.06, f'{elapsed:.2f} s', fontsize=9)
    for ax in axes:
        ax.set_yticks(range(3), ['Qwen3.5 2B', 'Qwen3.5 4B', 'Qwen3.5 9B'])
        ax.set_ylim(2.45, -.5); ax.grid(axis='x')
    axes[0].set_xlim(-.02, 1); axes[0].set_xlabel('答案 token F1（文字重合代理）')
    axes[0].set_title('可回答题；论文聚类 95% 区间', loc='left', fontsize=11)
    axes[0].legend(loc='lower right', fontsize=9)
    axes[1].set_xlim(left=0, right=axes[1].get_xlim()[1]*1.3)
    axes[1].set_xlabel('生成包装器 P50（秒；含遥测收尾）')
    axes[1].set_title('全部 200 题；不含检索', loc='left', fontsize=11)
    fig.suptitle('E3：有无已知证据时的真实模型表现', x=.01, ha='left', fontsize=15, fontweight='bold')
    fig.supxlabel('每条件 200 test；不可答题另列拒答指标。2,000 次论文聚类重采样。\n单卡串行生成，CPU 同时索引；延迟为该运行环境下的观测值。Oracle 不保证生成正确。', fontsize=9)
    save(fig, output, 'e3_oracle_no_context', groups, source)


def training_figures(source, output):
    audit = json.loads(source.read_text(encoding='utf-8'))
    if audit['status'] != 'PASS_REAL_TRAINING_AUDIT' or not verify_artifacts(audit['artifacts']):
        raise ValueError('Completed, unchanged real training audit required')
    rows = [json.loads(line) for line in (source.parent/'steps.jsonl').read_text(encoding='utf-8').splitlines() if line]
    if len(rows) != audit['actual_optimizer_steps'] or [r['step'] for r in rows] != list(range(1, len(rows)+1)):
        raise ValueError('Optimizer step population differs')
    for row in rows:
        if signature({k: v for k, v in row.items() if k != 'record_hash'}) != row['record_hash']:
            raise ValueError('Training step checksum changed')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
    axes[0].plot([r['step'] for r in rows], [r['loss'] for r in rows], color=BLUE, linewidth=1)
    axes[0].set_title('逐步训练 loss（原始值，不平滑）', loc='left', fontsize=11)
    for step in audit['restoration_steps']:
        axes[0].axvline(step, color='#9BA5AD', linestyle='--', linewidth=.8)
    dev = audit['dev_losses']
    axes[1].plot([r['step'] for r in dev], [r['dev_loss'] for r in dev], color=BLUE, marker='o')
    for row in dev:
        axes[1].annotate(f"{row['dev_loss']:.4f}", (row['step'], row['dev_loss']),
                         xytext=(0, 9), textcoords='offset points', ha='center', fontsize=9)
    axes[1].margins(y=.3)
    axes[1].set_title('冻结 dev 集的验证 loss', loc='left', fontsize=11)
    for ax in axes:
        ax.set_xlabel('Optimizer step'); ax.set_ylabel('样本 loss 均值（交叉熵）'); ax.grid(axis='y')
    fig.suptitle('2B BF16 LoRA：实际训练与恢复', x=.01, ha='left', fontsize=15, fontweight='bold')
    fig.supxlabel(f"seed42 · {audit['train_examples']} train · 1 epoch / {len(rows)} 步 · dev 59 条\n虚线为保存后恢复的第 5 / 10 步；loss 下降不能替代同精度问答对照。", fontsize=9)
    save(fig, output, 'lora_training_loss', {'steps': rows, 'audit': audit}, source)


def legacy_figures(source, output):
    report = json.loads(source.read_text(encoding='utf-8'))
    body = {k: v for k, v in report.items() if k != 'analysis_hash'}
    if (report.get('status') != 'COMPLETE' or signature(body) != report['analysis_hash']
            or not verify_artifacts(report['sources'])
            or report.get('reconstruction_label') != 'legacy_reconstructed_fixed_revision'):
        raise ValueError('Completed, unchanged legacy reconstruction analysis required')
    metrics = [('recall5', 'Recall@5 / Hit@5'), ('recall10', 'Recall@10'),
               ('mrr', 'MRR'), ('ndcg10', 'nDCG@10')]
    fig, ax = plt.subplots(figsize=(10, 5.2), layout='constrained')
    for method, offset, color in [('dense', -.13, BLUE), ('hybrid', .13, '#B57635')]:
        group = report['groups'][method]
        if group['n_total'] != report['query_count']:
            raise ValueError('Legacy method population differs')
        for i, (metric, _) in enumerate(metrics):
            stats = group['descriptive_ci95'][metric]
            mean = stats['mean']
            ax.errorbar(mean, i+offset,
                xerr=[[mean-stats['low']], [stats['high']-mean]],
                fmt='o', capsize=4, color=color, label=method.capitalize() if i == 0 else None)
            ax.annotate(f'{mean:.3f}', (mean, i+offset), xytext=(0, -14 if offset > 0 else 8),
                        textcoords='offset points', ha='center', color=color, fontsize=9)
    ax.set_yticks(range(4), [label for _, label in metrics])
    ax.set_ylim(3.65, -.55); ax.set_xlim(0, 1.03); ax.grid(axis='x')
    ax.set_xlabel('旧版段落文档相关性指标（均值与文章聚类 95% 区间）')
    ax.legend(loc='lower left')
    fig.suptitle('E0 固定版本重建：文档命中与排序', x=.01, ha='left', fontsize=15, fontweight='bold')
    fig.supxlabel(f"每方法 {report['query_count']} 题 · {report['n_article_clusters']} 篇原始文章 · 2,000 次聚类重采样\n历史模型版本未知；固定官方版本与精确余弦重建，不是 v2 证据覆盖或因果改善。", fontsize=9)
    save(fig, output, 'e0_reconstructed_retrieval', report['groups'], source)


def stress_figures(source, output):
    """Plot completed dev resource diagnosis, never main quality evidence."""
    audit = json.loads(source.read_text(encoding='utf-8'))
    if (audit.get('status') != 'PASS' or audit.get('formal_n') != 16
            or audit.get('warmup_n') != 6 or not verify_artifacts(audit['artifacts'])):
        raise ValueError('Complete independently checked stress22 source required')
    records = {}
    for window in (4096, 8192):
        paths = [Path(a['path']) for a in audit['artifacts'] if Path(a['path']).name == f'window_{window}.jsonl']
        if len(paths) != 1:
            raise ValueError('Stress formal source is ambiguous')
        rows = [json.loads(line) for line in paths[0].read_text(encoding='utf-8').splitlines() if line.strip()]
        if any(r['record_hash'] != signature({k: v for k, v in r.items() if k != 'record_hash'}) for r in rows):
            raise ValueError('Stress raw record hash differs')
        if len(rows) != 8 or any(r.get('error') or r.get('real_model') is not True for r in rows):
            raise ValueError('Stress formal population differs')
        records[window] = {r['query_id']: r for r in rows}
    ids = sorted(records[4096])
    if set(ids) != set(records[8192]):
        raise ValueError('Paired stress query populations differ')
    data = []
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.7), layout='constrained')
    for i, qid in enumerate(ids):
        latency = [records[w][qid]['timings']['generation_ms']/1000 for w in (4096, 8192)]
        axes[0].plot([0, 1], latency, marker='o', linewidth=.7, alpha=.7, color=BLUE)
    for window in (4096, 8192):
        rows = list(records[window].values())
        peak = max(r['resources']['peak_sampled_device_total_used_bytes'] for r in rows)/2**30
        minimum = min(r['resources']['minimum_sampled_host_available_bytes'] for r in rows)/2**30
        data.append({'window': window, 'actual_input_tokens': rows[0]['expected_actual_input_tokens'],
            'formal_n': len(rows), 'mean_generation_s': float(np.mean([r['timings']['generation_ms']/1000 for r in rows])),
            'sampled_device_max_gib': peak, 'minimum_host_available_gib': minimum,
            'record_hashes': [r['record_hash'] for r in rows]})
    for ax, field, label in ((axes[1], 'sampled_device_max_gib', '采样全设备显存最大值（GiB）'),
                            (axes[2], 'minimum_host_available_gib', '采样主机可用RAM最小值（GiB）')):
        values = [r[field] for r in data]
        ax.bar([0, 1], values, color=BLUE, width=.5)
        ax.set_ylim(0, max(values)*1.3); ax.set_ylabel(label)
        for i, value in enumerate(values):
            ax.text(i, value+.06, f'{value:.3f}', ha='center')
    axes[0].set_ylabel('正式请求生成时间（秒）'); axes[0].set_ylim(bottom=0)
    axes[0].set_title('同8题配对的实际观测', loc='left', fontsize=11)
    for ax in axes:
        ax.set_xticks([0, 1], ['4K / 输入3584', '8K / 输入7680']); ax.grid(axis='y')
    fig.suptitle('2B固定dev上下文诊断：时间与资源', x=.01, ha='left', fontsize=15, fontweight='bold')
    fig.supxlabel('每窗口8正式请求；排除3次预热 · R3与辅助CPU4工作同时运行\n设备量含其他进程、采样可能漏峰值；这是资源诊断，不是质量排名或最低硬件要求。', fontsize=9)
    save(fig, output, 'context_stress_resource', data, source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path); parser.add_argument('--analysis', type=Path)
    parser.add_argument('--e3-analysis', type=Path)
    parser.add_argument('--training-audit', type=Path)
    parser.add_argument('--legacy-analysis', type=Path)
    parser.add_argument('--context-stress', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if not any((args.pilot, args.analysis, args.e3_analysis, args.training_audit, args.legacy_analysis, args.context_stress)): parser.error('A verified source is required')
    if args.pilot: pilot(args.pilot, args.output)
    if args.analysis: main_figures(args.analysis, args.output)
    if args.e3_analysis: e3_figures(args.e3_analysis, args.output)
    if args.training_audit: training_figures(args.training_audit, args.output)
    if args.legacy_analysis: legacy_figures(args.legacy_analysis, args.output)
    if args.context_stress: stress_figures(args.context_stress, args.output)


if __name__ == '__main__':
    main()
