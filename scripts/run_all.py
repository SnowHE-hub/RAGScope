"""
RAGScope end-to-end pipeline runner.

Steps:
  1  ingest.py             — download & clean SQuAD / arXiv data
  2  build_indexes.py      — chunk documents, build Dense + BM25 indexes
  3  run_retrieval_eval.py — benchmark retrieval configs, save CSV
  4  generate_eval_report.py — render HTML report from CSV

Usage:
    python scripts/run_all.py              # run all steps
    python scripts/run_all.py --from 3    # skip steps 1-2, start at step 3
    python scripts/run_all.py --from 2 --to 3  # run only steps 2-3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_ROOT    = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"

STEPS = [
    (1, "ingest.py",               "Data ingestion (SQuAD + arXiv)"),
    (2, "build_indexes.py",        "Build Dense + BM25 indexes (fixed chunking)"),
    (3, "run_retrieval_eval.py",   "Retrieval benchmark (200 queries)"),
    (4, "generate_eval_report.py", "Generate HTML evaluation report"),
]


def _run_step(n: int, script: str, description: str) -> bool:
    """Run a single step; return True on success."""
    script_path = _SCRIPTS / script
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  Step {n}: {description}")
    print(f"  Script : scripts/{script}")
    print(sep)

    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(_ROOT),
    )
    elapsed = time.perf_counter() - t0

    if result.returncode == 0:
        print(f"\n  ✅  Step {n} completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n  ❌  Step {n} FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full RAGScope pipeline, optionally skipping early steps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--from", dest="from_step", type=int, default=1, metavar="N",
        help="Start from step N (1-4).  Steps before N are skipped.",
    )
    parser.add_argument(
        "--to", dest="to_step", type=int, default=len(STEPS), metavar="N",
        help="Stop after step N (1-4, inclusive).  Default: last step.",
    )
    args = parser.parse_args()

    from_step = args.from_step
    to_step   = args.to_step

    if not (1 <= from_step <= len(STEPS)):
        parser.error(f"--from must be between 1 and {len(STEPS)}")
    if not (1 <= to_step <= len(STEPS)):
        parser.error(f"--to must be between 1 and {len(STEPS)}")
    if from_step > to_step:
        parser.error("--from must be ≤ --to")

    selected = [(n, s, d) for n, s, d in STEPS if from_step <= n <= to_step]
    skipped  = [(n, s, d) for n, s, d in STEPS if n < from_step]

    print("\n" + "═" * 60)
    print("  RAGScope — Full Pipeline Runner")
    print("═" * 60)
    if skipped:
        print(f"  Skipping steps: {', '.join(str(n) for n,_,_ in skipped)}")
    print(f"  Running steps : {', '.join(str(n) for n,_,_ in selected)}")

    total_t0 = time.perf_counter()
    failed_at: int | None = None

    for n, script, description in selected:
        ok = _run_step(n, script, description)
        if not ok:
            failed_at = n
            break

    total_elapsed = time.perf_counter() - total_t0
    print("\n" + "═" * 60)

    if failed_at is not None:
        print(f"  Pipeline FAILED at step {failed_at}.")
        print(f"  Fix the error and re-run with:  python scripts/run_all.py --from {failed_at}")
        sys.exit(1)
    else:
        print(f"  ✅  All steps completed in {total_elapsed:.1f}s")
        report = _ROOT / "reports" / "eval_report.html"
        if report.exists():
            print(f"  📄  Report: {report}")
        print("═" * 60)


if __name__ == "__main__":
    main()
