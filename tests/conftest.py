"""Shared test fixtures."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def theow_dir(temp_dir):
    """Create a .theow directory structure."""
    theow_path = temp_dir / ".theow"
    theow_path.mkdir()
    (theow_path / "rules").mkdir()
    (theow_path / "actions").mkdir()
    (theow_path / "chroma").mkdir()
    return theow_path


@pytest.fixture
def mock_chroma():
    """Pre-configured ChromaStore mock."""
    chroma = MagicMock()
    chroma.get_metadata_keys.return_value = set()
    chroma.list_rules.return_value = []
    chroma.list_actions.return_value = []
    chroma.query_rules.return_value = []
    chroma.query_actions.return_value = []
    return chroma
