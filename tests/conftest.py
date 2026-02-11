"""Shared test fixtures."""

import tempfile
from pathlib import Path

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
def sample_rule_yaml():
    """Sample rule YAML content."""
    return """
name: module_path_rename
description: >
  Module had a URL rename. Build fails because the required
  path doesn't match the module's declared path.
tags: [go, dep_resolution]

when:
  - fact: problem_type
    equals: dep_resolution
  - fact: stderr
    regex: 'declares its path as: (?P<new_path>\\S+)\\s+but was required as: (?P<old_path>\\S+)'
    examples:
      - "module declares its path as: dario.cat/mergo but was required as: github.com/imdario/mergo"

then:
  - action: fix_path_rename
    params:
      new_path: "{new_path}"
      old_path: "{old_path}"
"""
