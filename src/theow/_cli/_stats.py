"""The `theow stats` command."""

from __future__ import annotations

from pathlib import Path

from theow._core._engine import Theow


def stats(theow_dir: str = ".theow") -> None:
    """Print theow stats."""
    engine = Theow(theow_dir=Path(theow_dir))
    engine.meow()
