"""Prepare immutable GPU migration, or run after explicit single-card handoff."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragscope.experiments.rerank_gpu import main

if __name__ == "__main__":
    raise SystemExit(main())
