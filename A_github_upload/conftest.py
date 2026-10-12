"""Ensures `stat_arb` (src/) and `config` (project root) are importable without requiring an
editable pip install -- pytest auto-discovers this file and runs it before collecting tests."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / 'src'

for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
