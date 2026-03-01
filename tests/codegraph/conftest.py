"""Shared fixtures for codegraph tests."""

import tempfile
from pathlib import Path

import pytest

# Theow's own source tree as test corpus
THEOW_SRC = Path(__file__).resolve().parent.parent.parent / "src" / "theow"


@pytest.fixture
def theow_src():
    """Path to theow's own source tree."""
    return THEOW_SRC


@pytest.fixture
def empty_project():
    """Empty directory for edge-case tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
