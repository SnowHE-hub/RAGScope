"""Freeze readiness, or explicitly execute local CPU R3 after RAM handoff."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ragscope.experiments.rerank_campaign import main

if __name__=="__main__":raise SystemExit(main())
