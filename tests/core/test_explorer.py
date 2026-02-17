"""Tests for explorer."""

import time
from pathlib import Path
from unittest.mock import MagicMock

from theow._core._decorators import ActionRegistry, TracingInfo
from theow._core._explorer import Explorer
from theow._core._models import Action, Fact, Rule
from theow._core._tools import Done, GiveUp, RequestTemplates, SubmitRule


def _make_explorer(
    chroma=None,
    gateway=None,
    action_registry=None,
    rules_dir=None,
    session_limit=20,
):
    return Explorer(
        chroma=chroma or MagicMock(),
        gateway=gateway,
        action_registry=action_registry or ActionRegistry(),
        rules_dir=rules_dir or Path("/tmp/fake/rules"),
        session_limit=session_limit,
    )


def test_find_failed_fact_missing():
    explorer = _make_explorer()
    rule = Rule(name="r", description="R", when=[Fact(fact="missing_key", equals="x")])
    result = explorer._find_failed_fact(rule, {})
    assert "missing from context" in result


def test_find_failed_fact_mismatch():
    explorer = _make_explorer()
    rule = Rule(name="r", description="R", when=[Fact(fact="x", equals="expected")])
    result = explorer._find_failed_fact(rule, {"x": "actual"})
    assert "actual" in result
    assert "expected" in result


def test_build_tools_section():
    explorer = _make_explorer()

    def my_tool(path: str) -> str:
        """Read a file."""
        return ""

    section = explorer._build_tools_section([my_tool])
    assert "my_tool(path)" in section
    assert "Read a file." in section


def test_build_error_prompt_basic():
    explorer = _make_explorer()
    prompt = explorer._build_error_prompt(
        context={"stderr": "error"},
        tracing=None,
        rejected_attempts=None,
    )
    assert "stderr: error" in prompt


def test_build_error_prompt_with_tracing():
    explorer = _make_explorer()
    tracing = TracingInfo(
        traceback="Traceback...",
        exception_type="ValueError",
        exception_message="bad value",
    )
    prompt = explorer._build_error_prompt(
        context={"stderr": "error"},
        tracing=tracing,
        rejected_attempts=None,
    )
    assert "ValueError" in prompt
    assert "bad value" in prompt


def test_build_error_prompt_with_continuation():
    explorer = _make_explorer()
    prompt = explorer._build_error_prompt(
        context={"stderr": "error"},
        tracing=None,
        rejected_attempts=None,
        attempt_number=3,
    )
    assert "attempt 3" in prompt
    assert "list_ephemeral_rules()" in prompt


def test_build_error_prompt_with_rejected_attempts():
    explorer = _make_explorer()
    rejected = [
        {"rule_name": "bad_rule", "description": "A bad rule", "error": "didn't work"},
    ]
    prompt = explorer._build_error_prompt(
        context={"stderr": "error"},
        tracing=None,
        rejected_attempts=rejected,
    )
    assert "bad_rule" in prompt
    assert "didn't work" in prompt
    assert "different approach" in prompt


def test_explore_no_gateway():
    explorer = _make_explorer(gateway=None)
    rule, explored = explorer.explore(context={"x": "y"}, tools=[])
    assert rule is None
    assert explored is False


def test_explore_session_limit():
    explorer = _make_explorer(gateway=MagicMock(), session_limit=0)
    rule, explored = explorer.explore(context={"x": "y"}, tools=[])
    assert rule is None
    assert explored is False


def test_explore_cache_hit():
    gateway = MagicMock()
    explorer = _make_explorer(gateway=gateway)

    # Manually prime the session cache
    from theow._core._session_cache import SessionCache

    cache = SessionCache()
    cached_rule = Rule(name="cached", description="Cached", when=[])
    cache.store({"x": "y"}, cached_rule)
    explorer._session_cache = cache

    rule, explored = explorer.explore(context={"x": "y"}, tools=[])
    assert rule is cached_rule
    assert explored is False


def test_explore_chroma_hit(theow_dir):
    chroma = MagicMock()
    chroma.query_rules.return_value = [("existing_rule", 0.1, {})]

    action_registry = ActionRegistry()
    rules_dir = theow_dir / "rules"

    existing_rule = Rule(
        name="existing_rule",
        description="Existing",
        when=[Fact(fact="x", equals="y")],
    )
    existing_rule.to_yaml(rules_dir / "existing_rule.rule.yaml")

    gateway = MagicMock()
    explorer = _make_explorer(
        chroma=chroma,
        gateway=gateway,
        action_registry=action_registry,
        rules_dir=rules_dir,
    )

    rule, explored = explorer.explore(context={"x": "y"}, tools=[])
    assert rule is not None
    assert rule.name == "existing_rule"
    assert explored is False


def test_explore_chroma_miss_high_distance(theow_dir):
    """Chroma result too distant should not be returned."""
    chroma = MagicMock()
    chroma.query_rules.return_value = [("far_rule", 0.9, {})]

    gateway = MagicMock()
    # conversation raises no signal (budget exhausted)
    gateway.conversation.return_value = MagicMock()

    rules_dir = theow_dir / "rules"
    explorer = _make_explorer(chroma=chroma, gateway=gateway, rules_dir=rules_dir)

    rule, explored = explorer.explore(context={"x": "y"}, tools=[])
    assert explored is True  # Fell through to conversation


def test_converse_signal():
    gateway = MagicMock()
    gateway.conversation.side_effect = GiveUp("nope")
    explorer = _make_explorer(gateway=gateway)
    signal = explorer._converse([], [])
    assert isinstance(signal, GiveUp)


def test_handle_signal_none_with_orphan(theow_dir):
    """Budget exhausted, finds orphaned ephemeral rule."""
    rules_dir = theow_dir / "rules"
    ephemeral_dir = rules_dir / "ephemeral"
    ephemeral_dir.mkdir()

    rule = Rule(name="orphan", description="Orphan", when=[Fact(fact="x", equals="y")])
    path = ephemeral_dir / "orphan.rule.yaml"
    rule.to_yaml(path)

    explorer = _make_explorer(rules_dir=rules_dir)
    result = explorer._handle_signal(None, [], [], {}, "default")
    assert result == (None, True)

    # Verify incomplete tag was added
    loaded = Rule.from_yaml(path)
    assert "incomplete" in loaded.tags


def test_handle_signal_request_templates():
    gateway = MagicMock()
    # After templates, conversation returns no signal (budget exhausted)
    gateway.conversation.return_value = MagicMock()

    explorer = _make_explorer(gateway=gateway)
    signal = RequestTemplates()

    messages = []
    result = explorer._handle_signal(signal, messages, [], {}, "default")
    # Should have appended templates message
    assert len(messages) >= 1
    assert result[1] is True  # explored


def test_handle_signal_submit_rule(theow_dir):
    rules_dir = theow_dir / "rules"
    action_registry = ActionRegistry()

    @action_registry.register("fix")
    def fix():
        return True

    rule = Rule(
        name="submitted",
        description="A submitted rule",
        when=[Fact(fact="x", equals="y")],
        then=[Action(action="fix")],
    )
    rule_path = rules_dir / "ephemeral"
    rule_path.mkdir()
    rule.to_yaml(rule_path / "submitted.rule.yaml")

    chroma = MagicMock()
    chroma.list_rules.return_value = []

    explorer = _make_explorer(chroma=chroma, action_registry=action_registry, rules_dir=rules_dir)

    signal = SubmitRule(str(rule_path / "submitted.rule.yaml"))
    result = explorer._handle_signal(signal, [], [], {"x": "y"}, "default")
    assert result[0] is not None
    assert result[0].name == "submitted"
    assert result[1] is True


def test_validate_rule_no_match(theow_dir):
    rules_dir = theow_dir / "rules"
    rule = Rule(name="r", description="R", when=[Fact(fact="x", equals="y")])
    path = rules_dir / "r.rule.yaml"
    rule.to_yaml(path)
    explorer = _make_explorer(rules_dir=rules_dir)
    result = explorer._validate_rule(str(path), None, {"x": "z"}, "default")
    assert result is None


def test_validate_rule_action_missing(theow_dir):
    rules_dir = theow_dir / "rules"
    action_registry = ActionRegistry()

    rule = Rule(
        name="r",
        description="R",
        when=[Fact(fact="x", equals="y")],
        then=[Action(action="nonexistent_action")],
    )
    path = rules_dir / "r.rule.yaml"
    rule.to_yaml(path)

    chroma = MagicMock()
    chroma.list_rules.return_value = []
    explorer = _make_explorer(chroma=chroma, action_registry=action_registry, rules_dir=rules_dir)
    result = explorer._validate_rule(str(path), None, {"x": "y"}, "default")
    assert result is None


def test_validate_rule_conflict(theow_dir):
    rules_dir = theow_dir / "rules"
    action_registry = ActionRegistry()

    @action_registry.register("fix_a")
    def fix_a():
        return True

    @action_registry.register("fix_b")
    def fix_b():
        return True

    # Existing rule
    existing = Rule(
        name="existing",
        description="Existing",
        when=[Fact(fact="x", equals="y")],
        then=[Action(action="fix_a")],
    )
    existing.to_yaml(rules_dir / "existing.rule.yaml")

    # New rule with same when, different then
    new_rule = Rule(
        name="new",
        description="New",
        when=[Fact(fact="x", equals="y")],
        then=[Action(action="fix_b")],
    )
    path = rules_dir / "new.rule.yaml"
    new_rule.to_yaml(path)

    chroma = MagicMock()
    chroma.list_rules.return_value = ["existing"]

    explorer = _make_explorer(chroma=chroma, action_registry=action_registry, rules_dir=rules_dir)
    result = explorer._validate_rule(str(path), None, {"x": "y"}, "default")
    assert result is None


def test_validate_rule_success(theow_dir):
    rules_dir = theow_dir / "rules"
    action_registry = ActionRegistry()

    @action_registry.register("fix")
    def fix():
        return True

    rule = Rule(
        name="good",
        description="Good rule",
        when=[Fact(fact="x", equals="y")],
        then=[Action(action="fix")],
    )
    path = rules_dir / "ephemeral"
    path.mkdir()
    rule.to_yaml(path / "good.rule.yaml")

    chroma = MagicMock()
    chroma.list_rules.return_value = []

    explorer = _make_explorer(chroma=chroma, action_registry=action_registry, rules_dir=rules_dir)
    result = explorer._validate_rule(str(path / "good.rule.yaml"), None, {"x": "y"}, "default")
    assert result is not None
    assert result.name == "good"
    assert len(result._created_files) == 1


def test_validate_rule_with_action_file(theow_dir):
    from theow._core._decorators import set_standalone_registry

    rules_dir = theow_dir / "rules"
    actions_dir = theow_dir / "actions"
    action_registry = ActionRegistry()

    # Set standalone registry so @action decorator in the action file works
    set_standalone_registry(action_registry)

    rule = Rule(
        name="r",
        description="R",
        when=[Fact(fact="x", equals="y")],
        then=[Action(action="new_action")],
    )
    path = rules_dir / "r.rule.yaml"
    rule.to_yaml(path)

    # Write a real action file that registers via @action
    action_file = actions_dir / "new_action.py"
    action_file.write_text(
        'from theow import action\n\n@action("new_action")\ndef new_action():\n    return True\n'
    )

    chroma = MagicMock()
    chroma.list_rules.return_value = []

    explorer = _make_explorer(chroma=chroma, action_registry=action_registry, rules_dir=rules_dir)
    result = explorer._validate_rule(str(path), str(action_file), {"x": "y"}, "default")
    assert result is not None
    assert len(result._created_files) == 2


def test_run_direct_no_gateway():
    explorer = _make_explorer(gateway=None)
    result = explorer.run_direct("fix this", [], {"max_tool_calls_per_session": 5})
    assert result is False


def test_run_direct_done():
    gateway = MagicMock()
    gateway.conversation.side_effect = Done("fixed")
    explorer = _make_explorer(gateway=gateway)
    result = explorer.run_direct("fix this", [], {"max_tool_calls_per_session": 5})
    assert result is True


def test_run_direct_give_up():
    gateway = MagicMock()
    gateway.conversation.side_effect = GiveUp("nope")
    explorer = _make_explorer(gateway=gateway)
    result = explorer.run_direct("fix this", [], {"max_tool_calls_per_session": 5})
    assert result is False


def test_run_direct_budget_exhausted():
    gateway = MagicMock()
    gateway.conversation.return_value = MagicMock()
    explorer = _make_explorer(gateway=gateway)
    result = explorer.run_direct("fix this", [], {"max_tool_calls_per_session": 5})
    assert result is False


def test_run_direct_unknown_signal():
    gateway = MagicMock()
    gateway.conversation.side_effect = RequestTemplates()
    explorer = _make_explorer(gateway=gateway)
    result = explorer.run_direct("fix this", [], {"max_tool_calls_per_session": 5})
    assert result is False


def test_reset_session():
    gateway = MagicMock()
    explorer = _make_explorer(gateway=gateway)
    explorer._session_count = 5

    from theow._core._session_cache import SessionCache

    explorer._session_cache = SessionCache()
    explorer._session_cache.store({"x": "y"}, Rule(name="r", description="R", when=[]))
    explorer._pending_cleanup = [Path("/tmp/fake")]

    explorer.reset_session()

    assert explorer._session_count == 0
    assert explorer._session_cache.size == 0
    assert explorer._pending_cleanup == []


def test_cleanup_deletes_rejected_files(tmp_path):
    explorer = _make_explorer()

    rule_file = tmp_path / "bad.rule.yaml"
    action_file = tmp_path / "bad.py"
    rule_file.write_text("bad")
    action_file.write_text("bad")

    rejected = [
        {"_files": [rule_file, action_file], "_incomplete": False},
    ]

    explorer.cleanup(rejected)
    assert not rule_file.exists()
    assert not action_file.exists()


def test_cleanup_skips_incomplete(tmp_path):
    explorer = _make_explorer()

    rule_file = tmp_path / "incomplete.rule.yaml"
    rule_file.write_text("keep")

    rejected = [
        {"_files": [rule_file], "_incomplete": True},
    ]

    explorer.cleanup(rejected)
    assert rule_file.exists()


def test_check_conflict_same_name_ignored(theow_dir):
    rules_dir = theow_dir / "rules"
    chroma = MagicMock()
    chroma.list_rules.return_value = ["my_rule"]

    existing = Rule(
        name="my_rule",
        description="Existing",
        when=[Fact(fact="x", equals="y")],
        then=[Action(action="fix")],
    )
    existing.to_yaml(rules_dir / "my_rule.rule.yaml")

    explorer = _make_explorer(chroma=chroma, rules_dir=rules_dir)
    rule = Rule(
        name="my_rule",
        description="Updated",
        when=[Fact(fact="x", equals="y")],
        then=[Action(action="fix")],
    )
    assert explorer._check_conflict(rule) is None


def test_find_orphaned_rules_found(theow_dir):
    rules_dir = theow_dir / "rules"
    ephemeral_dir = rules_dir / "ephemeral"
    ephemeral_dir.mkdir()

    rule = Rule(name="orphan", description="Orphan", when=[Fact(fact="x", equals="y")])
    path = ephemeral_dir / "orphan.rule.yaml"
    rule.to_yaml(path)

    explorer = _make_explorer(rules_dir=rules_dir)
    found = explorer._find_orphaned_rules()
    assert found is not None
    assert found.name == "orphan"


def test_find_orphaned_rules_old_file(theow_dir):
    """Old files (>5 min) should be ignored."""
    rules_dir = theow_dir / "rules"
    ephemeral_dir = rules_dir / "ephemeral"
    ephemeral_dir.mkdir()

    rule = Rule(name="old", description="Old", when=[])
    path = ephemeral_dir / "old.rule.yaml"
    rule.to_yaml(path)

    # Make file appear old
    import os

    old_time = time.time() - 600
    os.utime(path, (old_time, old_time))

    explorer = _make_explorer(rules_dir=rules_dir)
    assert explorer._find_orphaned_rules() is None


def test_ensure_incomplete_tag(theow_dir):
    rules_dir = theow_dir / "rules"
    ephemeral_dir = rules_dir / "ephemeral"
    ephemeral_dir.mkdir()

    rule = Rule(name="r", description="R", when=[Fact(fact="x", equals="y")])
    path = ephemeral_dir / "r.rule.yaml"
    rule.to_yaml(path)
    rule._source_path = path

    explorer = _make_explorer(rules_dir=rules_dir)
    explorer._ensure_incomplete_tag(rule)

    assert "incomplete" in rule.tags
    loaded = Rule.from_yaml(path)
    assert "incomplete" in loaded.tags
