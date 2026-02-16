"""Tests for core data models."""

import tempfile
from unittest.mock import MagicMock

import pytest

from theow._core._models import Action, Fact, LLMConfig, Rule


def test_fact_equals_match():
    fact = Fact(fact="problem_type", equals="build_failure")
    assert fact.matches("build_failure") == {}
    assert fact.matches("other") is None


def test_fact_contains_match():
    fact = Fact(fact="stderr", contains="module declares")
    assert fact.matches("error: module declares its path") == {}
    assert fact.matches("something else") is None


def test_fact_regex_with_captures():
    fact = Fact(fact="stderr", regex=r"path as: (?P<new>\S+) but was required as: (?P<old>\S+)")
    result = fact.matches("path as: new.io/pkg but was required as: old.io/pkg")
    assert result == {"new": "new.io/pkg", "old": "old.io/pkg"}


def test_fact_from_dict_examples_as_dicts():
    """YAML may parse multi-line example strings with colons as dicts."""
    data = {
        "fact": "stderr",
        "regex": "pattern",
        "examples": [{"key1": "value1", "key2": "value2"}, "normal string"],
    }
    fact = Fact.from_dict(data)
    assert fact.examples[0] == "key1: value1\nkey2: value2"
    assert fact.examples[1] == "normal string"


def test_fact_no_condition_matches_any():
    """Fact with no equals/contains/regex matches any non-None value."""
    fact = Fact(fact="x")
    assert fact.matches("anything") == {}
    assert fact.matches(None) is None


def test_action_resolve_params():
    action = Action(action="fix", params={"path": "{workspace}/file"})
    result = action.resolve_params({}, {"workspace": "/tmp"})
    assert result == {"path": "/tmp/file"}


def test_action_captures_override_context():
    action = Action(action="fix", params={"path": "{path}"})
    result = action.resolve_params({"path": "/capture"}, {"path": "/context"})
    assert result == {"path": "/capture"}


def test_llmconfig_inline_prompt_with_context():
    config = LLMConfig(prompt_template="Fix error in {package_name}@{version}")
    result = config.get_prompt(context={"package_name": "foo/bar", "version": "v1.0"})
    assert result == "Fix error in foo/bar@v1.0"


def test_llmconfig_file_prompt(tmp_path):
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Fix the build error")
    config = LLMConfig(prompt_template=f"file://{prompt_file}")
    assert config.get_prompt() == "Fix the build error"


def test_llmconfig_file_prompt_relative_path(tmp_path):
    prompt_file = tmp_path / "prompts" / "fix.md"
    prompt_file.parent.mkdir()
    prompt_file.write_text("Fix the error: {error}")
    config = LLMConfig(prompt_template="file://prompts/fix.md")
    result = config.get_prompt(base_path=tmp_path, context={"error": "bad stuff"})
    assert result == "Fix the error: bad stuff"


def test_rule_matches_all_facts():
    rule = Rule(
        name="test",
        description="Test",
        when=[
            Fact(fact="problem_type", equals="build"),
            Fact(fact="stderr", contains="error"),
        ],
    )
    context = {"problem_type": "build", "stderr": "build error occurred"}
    assert rule.matches(context) == {}


def test_rule_matches_fails_if_any_fact_fails():
    rule = Rule(
        name="test",
        description="Test",
        when=[
            Fact(fact="problem_type", equals="build"),
            Fact(fact="stderr", contains="specific"),
        ],
    )
    context = {"problem_type": "build", "stderr": "other error"}
    assert rule.matches(context) is None


def test_rule_matches_returns_captures():
    rule = Rule(
        name="test",
        description="Test",
        when=[Fact(fact="stderr", regex=r"version: (?P<ver>\d+)")],
    )
    context = {"stderr": "version: 123"}
    assert rule.matches(context) == {"ver": "123"}


def test_rule_yaml_roundtrip():
    rule = Rule(
        name="test_rule",
        description="A test rule",
        when=[Fact(fact="x", equals="y")],
        then=[Action(action="fix", params={"a": "b"})],
        tags=["test"],
    )

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        rule.to_yaml(f.name)
        loaded = Rule.from_yaml(f.name)

    assert loaded.name == rule.name
    assert loaded.when[0].equals == "y"
    assert loaded.then[0].action == "fix"


def test_rule_get_embedding_text():
    rule = Rule(
        name="test",
        description="Fix module rename",
        when=[Fact(fact="stderr", regex="pattern", examples=["ex1", "ex2"])],
    )
    text = rule.get_embedding_text()
    assert "Fix module rename" in text
    assert "ex1" in text


def test_rule_get_metadata_includes_equals_facts():
    rule = Rule(
        name="test",
        description="Test",
        when=[
            Fact(fact="problem_type", equals="build"),
            Fact(fact="stderr", contains="error"),
        ],
    )
    meta = rule.get_metadata()
    assert meta["problem_type"] == "build"
    assert "stderr" not in meta


def test_rule_content_hash_changes_on_diff():
    rule1 = Rule(name="test", description="Test", when=[])
    rule2 = Rule(name="test", description="Different", when=[])
    assert rule1.content_hash() != rule2.content_hash()


def test_rule_is_ephemeral_by_path(tmp_path):
    """Rule is ephemeral if its source path is in ephemeral/ folder."""
    rule = Rule(name="test", description="Test", when=[])
    rule._source_path = tmp_path / "rules" / "ephemeral" / "test.rule.yaml"
    assert rule.is_ephemeral is True


def test_rule_bind_preserves_runtime_state(tmp_path):
    rule = Rule(name="test", description="Test", when=[], tags=["some_tag"])
    rule._source_path = tmp_path / "rules" / "ephemeral" / "rule.yaml"
    rule._created_files = [tmp_path / "rule.yaml", tmp_path / "action.py"]

    bound = rule.bind({}, {}, None)

    assert bound._source_path == rule._source_path
    assert bound._created_files == rule._created_files
    assert bound.tags == ["some_tag"]


def test_rule_notes_serialization():
    rule = Rule(
        name="test",
        description="Test rule",
        when=[Fact(fact="x", equals="y")],
        notes="Tried approach A, didn't work. Next: try B.",
    )

    yaml_str = rule.to_yaml()
    assert "notes:" in yaml_str
    assert "Tried approach A" in yaml_str

    loaded = Rule.from_yaml_string(yaml_str)
    assert loaded.notes == rule.notes


def test_rule_act_calls_action_registry():
    registry = MagicMock()
    registry.call.return_value = "result"

    rule = Rule(
        name="test",
        description="Test",
        when=[Fact(fact="x", equals="y")],
        then=[Action(action="fix_it", params={"x": "{val}"})],
    )
    bound = rule.bind({"val": "hello"}, {}, registry)
    results = bound.act()
    assert results == ["result"]
    registry.call.assert_called_once()


def test_rule_act_without_registry_raises():
    rule = Rule(name="test", description="Test", when=[], then=[Action(action="fix")])
    with pytest.raises(RuntimeError, match="not bound"):
        rule.act()


def test_rule_yaml_with_collection():
    rule = Rule(name="test", description="Test", when=[], collection="custom")
    yaml_str = rule.to_yaml()
    assert "collection: custom" in yaml_str
    loaded = Rule.from_yaml_string(yaml_str)
    assert loaded.collection == "custom"


def test_rule_yaml_with_llm_config():
    rule = Rule(
        name="test",
        description="Test",
        when=[Fact(fact="x", equals="y")],
        llm_config=LLMConfig(prompt_template="Fix {error}", tools=["read_file"]),
    )
    yaml_str = rule.to_yaml()
    assert "llm_config:" in yaml_str
    loaded = Rule.from_yaml_string(yaml_str)
    assert loaded.llm_config is not None
    assert loaded.llm_config.prompt_template == "Fix {error}"
    assert loaded.llm_config.tools == ["read_file"]
