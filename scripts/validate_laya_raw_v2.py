"""Verify the fixed Laya raw experiment after its model process has exited."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragscope.routing.laya_validation import main

if __name__ == "__main__":
    raise SystemExit(main())
