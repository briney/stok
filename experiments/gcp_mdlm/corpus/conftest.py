"""Ensure repo root and src/ are importable when running corpus tests directly."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
