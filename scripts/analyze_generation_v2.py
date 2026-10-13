"""Rebuild failure-aware tables/cluster intervals from an immutable campaign."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
from ragscope.experiments.analysis_report import write_analysis

parser = argparse.ArgumentParser()
parser.add_argument('--campaign', type=Path, required=True)
parser.add_argument('--config', type=Path, default=Path('configs/v2/analysis.json'))
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
report = write_analysis(args.campaign, args.config, args.output)
print(json.dumps({key: report[key] for key in ('status', 'n_observed', 'n_expected_runnable', 'n_expected_blocked')}))
raise SystemExit(0 if report['status'] == 'COMPLETE' else 2)
