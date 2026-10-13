"""Sample actual FINAL_REPORT table cells, then recompute from immutable raw rows.

The accepted table is the existing markdown_table output. This audits numerical
cells in that table, not prose, confidence intervals, or every report claim.
No saved independent-audit receipt can substitute for a fresh raw-data audit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_generation_numbers_v2 import audit, sha

HEADER = ['实验', '模型', '方法', '状态', '已记录/计划', '失败', '文字 F1',
          '完整证据覆盖', '生成 P50 ms']
METRICS = {5: 'failed_n', 6: 'token_f1', 7: 'context_full_reference_coverage',
           8: 'generation_p50_ms'}


def report_cells(text):
    """Read displayed numbers and exact line locations before consulting results."""
    active, found, groups, cells = False, False, set(), []
    for number, line in enumerate(text.splitlines(), 1):
        values = [value.strip() for value in line.strip().strip('|').split('|')]
        if values == HEADER:
            if found:
                raise ValueError('Duplicate report generation table')
            active, found = True, True
            continue
        if not active:
            continue
        if not line.strip().startswith('|'):
            active = False
            continue
        if all(re.fullmatch(r':?-+:?', value) for value in values):
            continue
        if len(values) != len(HEADER):
            raise ValueError(f'Malformed report row at line {number}')
        group = tuple(values[:3])
        if group in groups:
            raise ValueError('Duplicate report group')
        groups.add(group)
        if values[3] != 'COMPLETE':
            continue
        for column, metric in METRICS.items():
            displayed = values[column]
            if displayed == 'N/A':
                continue
            if not re.fullmatch(r'\d+(?:\.\d+)?', displayed):
                raise ValueError('Unsupported numerical cell: '+displayed)
            cells.append({'group': list(group), 'metric': metric, 'displayed': displayed,
                          'report_line': number, 'report_column': column+1,
                          'report_row': line})
    if not found or len(cells) < 10:
        raise ValueError('At least10 final-report numerical cells are required')
    return cells


def compare_cells(cells, independent):
    checks = {(tuple(row['group']), row['metric']): row for row in independent['all_checks']}
    compared = []
    for cell in cells:
        key = (tuple(cell['group']), cell['metric'])
        if key not in checks:
            raise ValueError('Report group/metric has no independently recomputed raw result')
        check = checks[key]
        digits = len(cell['displayed'].split('.')[1]) if '.' in cell['displayed'] else 0
        actual = check['recomputed']
        rendered = None if actual is None else format(actual, f'.{digits}f')
        compared.append({**cell, 'independently_recomputed': actual,
                         'recomputed_at_display_precision': rendered,
                         'match': rendered == cell['displayed'],
                         'n_raw_rows': check['n_rows'],
                         'raw_record_hashes': check['raw_record_hashes']})
    return compared


def audit_report(report_path, campaign, analysis_path, corpus, output):
    if report_path.name != 'FINAL_REPORT.md':
        raise ValueError('This acceptance audit requires the actual FINAL_REPORT.md')
    if output.exists() or output.with_suffix('.raw.json').exists():
        raise FileExistsError('Preserve earlier audit receipts; choose a new output')
    report_hash = sha(report_path)
    cells = report_cells(report_path.read_text(encoding='utf-8-sig'))
    selected_indexes = random.Random(42).sample(range(len(cells)), 10)
    # Selection is fixed from the report before reading analysis or raw outputs.
    independent = audit(campaign, analysis_path, corpus, output.with_suffix('.raw.json'))
    compared = compare_cells(cells, independent)
    if sha(report_path) != report_hash:
        raise ValueError('Final report changed during audit')
    result = {'status': 'PASS' if independent['status'] == 'PASS' and all(
                  cell['match'] for cell in compared) else 'FAILED_VALIDATION',
              'report_source': {'path': str(report_path), 'sha256': report_hash},
              'seed': 42, 'eligible_report_cells': len(cells),
              'random_report_numbers': [compared[i] for i in selected_indexes],
              'all_report_table_checks': compared,
              'fresh_raw_audit': {'path': str(output.with_suffix('.raw.json')),
                                  'sha256': sha(output.with_suffix('.raw.json'))},
              'scope': 'Generation table displayed means, failure counts and median latency; '
                       'random10 selected from report before raw recomputation. '
                       'Not a validation of every prose number or bootstrap interval.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('report', 'campaign', 'analysis', 'corpus', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    result = audit_report(args.report, args.campaign, args.analysis, args.corpus, args.output)
    print(result['status'], result['eligible_report_cells'], 'report cells; random10 retained')
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
