"""Re-export the CLI module so `separate-vocals` is a real entry point.

The original implementation lives in `vocal_notes.py` at the repository
root for historical reasons; this package re-exports its `main` so
`pyproject.toml`'s `console_scripts` works without code duplication.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the sibling `vocal_notes.py` importable without restructuring it.
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

import vocal_notes as _vocal_notes  # noqa: E402

main = _vocal_notes.main

__all__ = ["main"]
