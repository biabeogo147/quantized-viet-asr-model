from __future__ import annotations

import sys
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
SRC_ROOT = REPO_ROOT / "src"

src_root_str = str(SRC_ROOT)
sys.path = [entry for entry in sys.path if entry != src_root_str]
sys.path.insert(0, src_root_str)
