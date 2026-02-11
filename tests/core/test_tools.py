"""Tests for tools."""

from theow._core._tools import make_ephemeral_tools


def test_list_ephemeral_rules_empty(tmp_path):
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()

    tools = make_ephemeral_tools(rules_dir)
    list_ephemeral = tools[0]

    assert list_ephemeral() == []


def test_list_ephemeral_rules_finds_rules(tmp_path):
    rules_dir = tmp_path / "rules"
    ephemeral_dir = rules_dir / "ephemeral"
    ephemeral_dir.mkdir(parents=True)

    rule_content = """
name: test_rule
description: A test rule
when:
  - fact: x
    equals: y
tags: [incomplete]
notes: Some progress notes
"""
    (ephemeral_dir / "test_rule.rule.yaml").write_text(rule_content)

    tools = make_ephemeral_tools(rules_dir)
    list_ephemeral = tools[0]

    results = list_ephemeral()
    assert len(results) == 1
    assert results[0]["name"] == "test_rule"
    assert "incomplete" in results[0]["tags"]
    assert results[0]["notes"] == "Some progress notes"


def test_read_ephemeral_rule(tmp_path):
    rules_dir = tmp_path / "rules"
    ephemeral_dir = rules_dir / "ephemeral"
    ephemeral_dir.mkdir(parents=True)

    rule_content = "name: test_rule\ndescription: Test\nwhen: []\n"
    (ephemeral_dir / "test_rule.rule.yaml").write_text(rule_content)

    tools = make_ephemeral_tools(rules_dir)
    read_ephemeral = tools[1]

    content = read_ephemeral("test_rule")
    assert content == rule_content


def test_read_ephemeral_rule_not_found(tmp_path):
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()

    tools = make_ephemeral_tools(rules_dir)
    read_ephemeral = tools[1]

    result = read_ephemeral("nonexistent")
    assert "not found" in result
