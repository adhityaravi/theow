"""Tests for rule augmentation tools."""

import pytest

from theow._core._models import Rule
from theow._core._tools import make_augmentation_tools


@pytest.fixture
def _env(tmp_path, mock_chroma):
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    (rules_dir / "my_rule.rule.yaml").write_text(
        "name: my_rule\ndescription: test\n"
        "when:\n  - fact: problem_type\n    equals: build\n"
        "  - fact: stderr\n    regex: 'module (?P<mod>\\S+)'\n"
        "    examples:\n      - 'module foo'\n"
        "then:\n  - action: fix\n"
    )
    ctx = {"problem_type": "build", "stderr": "cannot find module foo"}
    tools = make_augmentation_tools(rules_dir, ctx, mock_chroma)
    return tools[0], tools[1], rules_dir, mock_chroma


def test_add_fact_success(_env):
    add_fact, _, rules_dir, chroma = _env
    result = add_fact("my_rule", "stderr", contains="cannot find module")
    assert result["status"] == "ok"
    assert len(Rule.from_yaml(rules_dir / "my_rule.rule.yaml").when) == 3
    chroma.index_rule.assert_called_once()


def test_add_fact_rejects_nonmatching(_env):
    add_fact, *_ = _env
    assert "does not match" in add_fact("my_rule", "stderr", contains="ZZZZ")["error"]


def test_add_fact_rejects_bad_operator(_env):
    add_fact, *_ = _env
    assert "exactly one" in add_fact("my_rule", "stderr")["error"]
    assert "exactly one" in add_fact("my_rule", "stderr", contains="x", equals="y")["error"]


def test_add_fact_missing_rule(_env):
    add_fact, *_ = _env
    assert "error" in add_fact("nope", "stderr", contains="x")


def test_add_example_success(_env):
    _, add_example, _, chroma = _env
    result = add_example("my_rule", "stderr", "module bar/baz")
    assert result["status"] == "ok"
    chroma.index_rule.assert_called_once()


def test_add_example_wrong_key(_env):
    _, add_example, *_ = _env
    assert "error" in add_example("my_rule", "stdout", "x")


def test_file_mode_preserved(_env):
    add_fact, _, rules_dir, _ = _env
    path = rules_dir / "my_rule.rule.yaml"
    path.chmod(0o444)
    mode = path.stat().st_mode
    add_fact("my_rule", "stderr", contains="cannot find module")
    assert path.stat().st_mode == mode
