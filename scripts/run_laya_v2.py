"""Freeze Laya visible inputs; --run requires a coordinated CPU RAM window."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ragscope.routing.laya_campaign import main

if __name__=="__main__":raise SystemExit(main())
