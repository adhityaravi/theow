"""Tests for tools."""

import pytest

from theow._core._tools import (
    Done,
    GiveUp,
    RequestTemplates,
    SubmitRule,
    _done,
    _give_up,
    _request_templates,
    _submit_rule,
    make_ephemeral_tools,
    make_search_tools,
    make_validation_tools,
    read_file,
    write_file,
    run_command,
    list_directory,
)


def test_give_up_raises():
    with pytest.raises(GiveUp, match="Gave up"):
        _give_up("nope")


def test_request_templates_raises():
    with pytest.raises(RequestTemplates):
        _request_templates()


def test_submit_rule_raises():
    with pytest.raises(SubmitRule):
        _submit_rule("rule.yaml", "action.py")


def test_done_raises():
    with pytest.raises(Done):
        _done("all good")


def test_make_search_tools_search_rules(mock_chroma):
    mock_chroma.query_rules.return_value = [
        ("my_rule", 0.1, {"type": "deterministic"}),
    ]
    tools = make_search_tools(mock_chroma, "default")
    search_rules = tools[0]
    results = search_rules("fix module")
    assert len(results) == 1
    assert results[0]["name"] == "my_rule"
    assert results[0]["file"] == "my_rule.rule.yaml"


def test_make_search_tools_search_actions(mock_chroma):
    mock_chroma.query_actions.return_value = [{"name": "fix_rename"}]
    tools = make_search_tools(mock_chroma, "default")
    search_actions = tools[1]
    results = search_actions("rename")
    assert len(results) == 1
    assert results[0]["name"] == "fix_rename"


def test_make_search_tools_list_rules(mock_chroma):
    mock_chroma.list_rules.return_value = ["rule_a", "rule_b"]
    tools = make_search_tools(mock_chroma, "default")
    list_rules = tools[2]
    results = list_rules()
    assert len(results) == 2
    assert results[0]["file"] == "rule_a.rule.yaml"


def test_make_search_tools_list_actions(mock_chroma):
    mock_chroma.list_actions.return_value = ["action_a"]
    tools = make_search_tools(mock_chroma, "default")
    list_actions_fn = tools[3]
    results = list_actions_fn()
    assert results == ["action_a"]


def test_validation_tool_match_success(tmp_path):
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    rule_content = """
name: test
description: Test
when:
  - fact: problem_type
    equals: build
"""
    rule_path = rules_dir / "test.rule.yaml"
    rule_path.write_text(rule_content)

    context = {"problem_type": "build"}
    tools = make_validation_tools(rules_dir, context)
    test_match = tools[0]
    result = test_match(str(rule_path))
    assert result["matches"] is True


def test_validation_tool_match_failure(tmp_path):
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    rule_content = """
name: test
description: Test
when:
  - fact: problem_type
    equals: build
"""
    rule_path = rules_dir / "test.rule.yaml"
    rule_path.write_text(rule_content)

    context = {"problem_type": "test"}
    tools = make_validation_tools(rules_dir, context)
    test_match = tools[0]
    result = test_match(str(rule_path))
    assert result["matches"] is False
    assert result["facts"][0]["status"] == "FAIL"


def test_validation_tool_file_not_found(tmp_path):
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    context = {}
    tools = make_validation_tools(rules_dir, context)
    test_match = tools[0]
    result = test_match("/nonexistent/file.yaml")
    assert "error" in result


def test_validation_tool_parse_error(tmp_path):
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    bad_file = rules_dir / "bad.rule.yaml"
    bad_file.write_text("not: valid: yaml: [")

    context = {}
    tools = make_validation_tools(rules_dir, context)
    test_match = tools[0]
    result = test_match(str(bad_file))
    assert "error" in result


def test_validation_tool_missing_fact_value(tmp_path):
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    rule_content = """
name: test
description: Test
when:
  - fact: missing_key
    equals: something
"""
    rule_path = rules_dir / "test.rule.yaml"
    rule_path.write_text(rule_content)

    context = {}
    tools = make_validation_tools(rules_dir, context)
    test_match = tools[0]
    result = test_match(str(rule_path))
    assert result["matches"] is False
    assert "(missing)" in result["facts"][0]["actual"]


def test_validation_tool_captures_shown(tmp_path):
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    rule_content = """
name: test
description: Test
when:
  - fact: stderr
    regex: 'version: (?P<ver>\\d+)'
"""
    rule_path = rules_dir / "test.rule.yaml"
    rule_path.write_text(rule_content)

    context = {"stderr": "version: 42"}
    tools = make_validation_tools(rules_dir, context)
    test_match = tools[0]
    result = test_match(str(rule_path))
    assert result["matches"] is True
    assert result["facts"][0]["captures"] == {"ver": "42"}


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


def test_list_ephemeral_rules_skips_bad_files(tmp_path):
    rules_dir = tmp_path / "rules"
    ephemeral_dir = rules_dir / "ephemeral"
    ephemeral_dir.mkdir(parents=True)
    (ephemeral_dir / "bad.rule.yaml").write_text("not: valid: [yaml")

    tools = make_ephemeral_tools(rules_dir)
    list_ephemeral = tools[0]
    assert list_ephemeral() == []


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


def test_write_rule_creates_file(tmp_path):
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()

    tools = make_ephemeral_tools(rules_dir)
    write_rule = tools[2]

    result = write_rule("my_rule", "name: my_rule\n")
    assert "path" in result
    assert (rules_dir / "ephemeral" / "my_rule.rule.yaml").exists()


def test_write_action_creates_file(tmp_path):
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    actions_dir = tmp_path / "actions"
    # Don't create actions_dir, write_action should create it

    tools = make_ephemeral_tools(rules_dir)
    write_action = tools[3]

    result = write_action("my_action", "def my_action(): pass\n")
    assert "path" in result
    assert (actions_dir / "my_action.py").exists()


def test_read_file(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("hello world")
    assert read_file(str(f)) == "hello world"


def test_write_file(tmp_path):
    path = tmp_path / "sub" / "file.txt"
    result = write_file(str(path), "content")
    assert "Written" in result
    assert path.read_text() == "content"


def test_run_command():
    result = run_command("echo hello")
    assert result["returncode"] == 0
    assert "hello" in result["stdout"]


def test_list_directory(tmp_path):
    (tmp_path / "a.txt").touch()
    (tmp_path / "b.txt").touch()
    items = list_directory(str(tmp_path))
    assert "a.txt" in items
    assert "b.txt" in items
