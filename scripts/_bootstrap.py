from __future__ import annotations

import sys
from pathlib import Path


def add_src_to_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    sys.path.insert(0, str(src))
