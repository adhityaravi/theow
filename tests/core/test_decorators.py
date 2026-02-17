"""Tests for decorators."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from theow._core._decorators import (
    ActionNotFoundError,
    ActionRegistry,
    MarkConfig,
    MarkDecorator,
    ToolRegistry,
    action,
    set_standalone_registry,
)
from theow._core._models import Action, Fact, LLMConfig, Rule


def test_tool_registry_custom_name():
    registry = ToolRegistry()

    @registry.register("custom_name")
    def my_tool(x: str) -> str:
        return x

    assert registry.get("custom_name") is my_tool
    assert registry.get("my_tool") is None


def test_tool_registry_get_declarations():
    registry = ToolRegistry()

    @registry.register()
    def read_file(path: str) -> str:
        """Read a file."""
        return ""

    declarations = registry.get_declarations()
    assert len(declarations) == 1
    assert declarations[0]["name"] == "read_file"
    assert declarations[0]["description"] == "Read a file."
    assert "path" in declarations[0]["parameters"]["properties"]


def test_action_registry_register_and_call():
    registry = ActionRegistry()

    @registry.register("fix_thing")
    def fix_thing(workspace: str) -> dict:
        return {"fixed": workspace}

    result = registry.call("fix_thing", {"workspace": "/tmp"})
    assert result == {"fixed": "/tmp"}


def test_action_registry_call_not_found():
    registry = ActionRegistry()
    with pytest.raises(ActionNotFoundError, match="not found"):
        registry.call("missing", {})


def test_action_registry_get_metadata():
    registry = ActionRegistry()

    @registry.register("my_action")
    def my_action(x: str) -> str:
        """Do something."""
        return x

    meta = registry.get_metadata("my_action")
    assert meta["docstring"] == "Do something."
    assert "x: str" in meta["signature"]


def test_action_registry_discover(tmp_path):
    actions_dir = tmp_path / "actions"
    actions_dir.mkdir()

    action_code = '''from theow._core._decorators import action
@action("discovered_action")
def discovered_action(x: str) -> str:
    """A discovered action."""
    return x
'''
    (actions_dir / "my_action.py").write_text(action_code)

    registry = ActionRegistry()
    set_standalone_registry(registry)
    registry.discover(actions_dir)

    assert registry.exists("discovered_action")


def test_action_registry_load_action_file_bad_syntax(tmp_path):
    actions_dir = tmp_path / "actions"
    actions_dir.mkdir()
    (actions_dir / "bad.py").write_text("this is not valid python [[[")

    registry = ActionRegistry()
    registry._load_action_file(actions_dir / "bad.py")
    # Should not raise, just log warning


def test_standalone_action_decorator():
    registry = ActionRegistry()
    set_standalone_registry(registry)

    @action("standalone_fix")
    def standalone_fix(x: str) -> str:
        """Fix something."""
        return x

    assert registry.exists("standalone_fix")
    meta = registry.get_metadata("standalone_fix")
    assert meta["docstring"] == "Fix something."


def test_should_explore_enabled():
    resolver = MagicMock()
    explorer = MagicMock()
    tool_registry = ToolRegistry()
    md = MarkDecorator(resolver, explorer, tool_registry)

    config = MarkConfig(
        context_from=lambda *a, **k: {},
        max_retries=3,
        rules=None,
        tags=None,
        fallback=True,
        explorable=True,
        collection="default",
    )

    with patch.dict(os.environ, {"THEOW_EXPLORE": "1"}):
        assert md._should_explore(config) is True


def _make_mark_decorator(resolver=None, explorer=None, tool_registry=None):
    return MarkDecorator(
        resolver=resolver or MagicMock(),
        explorer=explorer or MagicMock(),
        tool_registry=tool_registry or ToolRegistry(),
    )


def test_run_with_recovery_success_first_try():
    md = _make_mark_decorator()
    config = MarkConfig(
        context_from=lambda *a, **k: {},
        max_retries=3,
        rules=None,
        tags=None,
        fallback=True,
        explorable=False,
        collection="default",
    )

    def fn():
        return "ok"

    result = md._run_with_recovery(fn, (), {}, config)
    assert result == "ok"


def test_run_with_recovery_resolve_then_success():
    resolver = MagicMock()
    action_registry = ActionRegistry()

    @action_registry.register("fix")
    def fix():
        return True

    rule = Rule(
        name="test_rule",
        description="Test",
        when=[Fact(fact="x", equals="y")],
        then=[Action(action="fix")],
    )
    bound_rule = rule.bind({}, {}, action_registry)
    resolver.resolve.return_value = bound_rule

    explorer = MagicMock()
    explorer.cleanup.return_value = None
    md = _make_mark_decorator(resolver=resolver, explorer=explorer)

    config = MarkConfig(
        context_from=lambda *a, **k: {"x": "y"},
        max_retries=3,
        rules=None,
        tags=None,
        fallback=True,
        explorable=False,
        collection="default",
    )

    call_count = 0

    def fn():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("first try fails")
        return "success"

    result = md._run_with_recovery(fn, (), {}, config)
    assert result == "success"


def test_run_with_recovery_no_match_reraises():
    resolver = MagicMock()
    resolver.resolve.return_value = None

    explorer = MagicMock()
    explorer.cleanup.return_value = None
    md = _make_mark_decorator(resolver=resolver, explorer=explorer)

    config = MarkConfig(
        context_from=lambda *a, **k: {"x": "y"},
        max_retries=1,
        rules=None,
        tags=None,
        fallback=True,
        explorable=False,
        collection="default",
    )

    def fn():
        raise ValueError("always fails")

    with pytest.raises(ValueError, match="always fails"):
        md._run_with_recovery(fn, (), {}, config)


def test_run_with_recovery_explore_creates_rule(theow_dir):
    resolver = MagicMock()
    resolver.resolve.return_value = None

    action_registry = ActionRegistry()

    @action_registry.register("fix")
    def fix():
        return True

    rule = Rule(
        name="ephemeral_rule",
        description="Ephemeral",
        when=[Fact(fact="x", equals="y")],
        then=[Action(action="fix")],
    )
    eph_path = theow_dir / "rules" / "ephemeral" / "ephemeral_rule.rule.yaml"
    eph_path.parent.mkdir(exist_ok=True)
    rule.to_yaml(eph_path)
    bound_rule = rule.bind({}, {}, action_registry)
    bound_rule._source_path = eph_path

    explorer = MagicMock()
    explorer.explore.return_value = (bound_rule, True)
    explorer.run_direct.return_value = True
    explorer.cleanup.return_value = None
    explorer._rules_dir = theow_dir / "rules"
    explorer._chroma = MagicMock()
    explorer._session_cache = None

    tool_registry = ToolRegistry()
    md = _make_mark_decorator(resolver=resolver, explorer=explorer, tool_registry=tool_registry)

    config = MarkConfig(
        context_from=lambda *a, **k: {"x": "y"},
        max_retries=3,
        rules=None,
        tags=None,
        fallback=True,
        explorable=True,
        collection="default",
    )

    call_count = 0

    def fn():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("first try fails")
        return "success"

    with patch.dict(os.environ, {"THEOW_EXPLORE": "1"}):
        result = md._run_with_recovery(fn, (), {}, config)
    assert result == "success"


def test_run_with_recovery_promotes_ephemeral(theow_dir):
    """After fn() succeeds, ephemeral rule should be promoted."""
    resolver = MagicMock()

    action_registry = ActionRegistry()

    @action_registry.register("fix")
    def fix():
        return True

    rule = Rule(
        name="eph",
        description="Ephemeral",
        when=[Fact(fact="x", equals="y")],
        then=[Action(action="fix")],
    )
    eph_path = theow_dir / "rules" / "ephemeral" / "eph.rule.yaml"
    eph_path.parent.mkdir(exist_ok=True)
    rule.to_yaml(eph_path)
    bound_rule = rule.bind({}, {}, action_registry)
    bound_rule._source_path = eph_path

    resolver.resolve.return_value = bound_rule

    explorer = MagicMock()
    explorer.cleanup.return_value = None
    explorer._rules_dir = theow_dir / "rules"
    explorer._chroma = MagicMock()
    explorer._session_cache = None

    md = _make_mark_decorator(resolver=resolver, explorer=explorer)

    config = MarkConfig(
        context_from=lambda *a, **k: {"x": "y"},
        max_retries=3,
        rules=None,
        tags=None,
        fallback=True,
        explorable=False,
        collection="default",
    )

    call_count = 0

    def fn():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("first try fails")
        return "success"

    result = md._run_with_recovery(fn, (), {}, config)
    assert result == "success"
    # Promotion happens via _promote_rule, which renames the file
    assert not eph_path.exists()  # moved out of ephemeral


def test_run_with_recovery_rejects_ephemeral(theow_dir):
    """If fn() fails after ephemeral rule, it should be rejected."""
    action_registry = ActionRegistry()

    @action_registry.register("fix")
    def fix():
        return True

    rule = Rule(
        name="bad_eph",
        description="Bad ephemeral",
        when=[Fact(fact="x", equals="y")],
        then=[Action(action="fix")],
    )
    eph_path = theow_dir / "rules" / "ephemeral" / "bad_eph.rule.yaml"
    eph_path.parent.mkdir(exist_ok=True)
    rule.to_yaml(eph_path)
    bound_rule = rule.bind({}, {}, action_registry)
    bound_rule._source_path = eph_path
    bound_rule._created_files = [eph_path]

    resolver = MagicMock()
    resolver.resolve.return_value = bound_rule

    explorer = MagicMock()
    explorer.cleanup.return_value = None
    explorer._session_cache = MagicMock()

    md = _make_mark_decorator(resolver=resolver, explorer=explorer)

    config = MarkConfig(
        context_from=lambda *a, **k: {"x": "y"},
        max_retries=1,
        rules=None,
        tags=None,
        fallback=True,
        explorable=False,
        collection="default",
    )

    def fn():
        raise ValueError("always fails")

    with pytest.raises(ValueError, match="always fails"):
        md._run_with_recovery(fn, (), {}, config)

    # Session cache should be invalidated
    explorer._session_cache.invalidate.assert_called_with("bad_eph")


def test_run_with_recovery_theow_error():
    """Internal theow errors should not propagate, original exception re-raised."""
    resolver = MagicMock()
    resolver.resolve.side_effect = RuntimeError("theow internal error")

    explorer = MagicMock()
    explorer.cleanup.return_value = None

    md = _make_mark_decorator(resolver=resolver, explorer=explorer)

    config = MarkConfig(
        context_from=lambda *a, **k: {"x": "y"},
        max_retries=3,
        rules=None,
        tags=None,
        fallback=True,
        explorable=False,
        collection="default",
    )

    def fn():
        raise ValueError("original error")

    with pytest.raises(ValueError, match="original error"):
        md._run_with_recovery(fn, (), {}, config)


def test_attempt_recovery_context_from_fails():
    md = _make_mark_decorator()

    def bad_context_from(*args, **kwargs):
        raise ValueError("context_from broken")

    config = MarkConfig(
        context_from=bad_context_from,
        max_retries=3,
        rules=None,
        tags=None,
        fallback=True,
        explorable=False,
        collection="default",
    )

    success, rule, explored = md._attempt_recovery(lambda: None, (), {}, ValueError("test"), config)
    assert success is False
    assert rule is None
    assert explored is False


def test_execute_rule_deterministic():
    action_registry = ActionRegistry()

    @action_registry.register("fix")
    def fix():
        return True

    rule = Rule(
        name="r",
        description="R",
        when=[],
        then=[Action(action="fix")],
    )
    bound = rule.bind({}, {}, action_registry)

    md = _make_mark_decorator()
    assert md._execute_rule(bound) is True


def test_execute_rule_deterministic_action_fails():
    action_registry = ActionRegistry()

    @action_registry.register("bad")
    def bad():
        raise RuntimeError("boom")

    rule = Rule(
        name="r",
        description="R",
        when=[],
        then=[Action(action="bad")],
    )
    bound = rule.bind({}, {}, action_registry)

    md = _make_mark_decorator()
    assert md._execute_rule(bound) is False


def test_execute_probabilistic_rule():
    explorer = MagicMock()
    explorer._rules_dir = Path("/tmp/rules")
    explorer.run_direct.return_value = True

    tool_registry = ToolRegistry()

    @tool_registry.register()
    def my_tool():
        pass

    md = _make_mark_decorator(explorer=explorer, tool_registry=tool_registry)

    rule = Rule(
        name="prob",
        description="Probabilistic",
        when=[],
        llm_config=LLMConfig(
            prompt_template="Fix {error}",
            tools=["my_tool"],
            constraints={"max_tool_calls": 5, "max_tokens": 2048},
        ),
    )
    bound = rule.bind({}, {"error": "bad stuff"}, None)

    result = md._execute_rule(bound, {"error": "bad stuff"})
    assert result is True
    explorer.run_direct.assert_called_once()


def test_promote_rule(theow_dir):
    explorer = MagicMock()
    explorer._chroma = MagicMock()
    md = _make_mark_decorator(explorer=explorer)

    eph_dir = theow_dir / "rules" / "ephemeral"
    eph_dir.mkdir()
    rule = Rule(name="r", description="R", when=[Fact(fact="x", equals="y")], tags=["incomplete"])
    eph_path = eph_dir / "r.rule.yaml"
    rule.to_yaml(eph_path)

    bound = rule.bind({}, {}, None)
    bound._source_path = eph_path

    md._promote_rule(bound)

    assert not eph_path.exists()
    promoted_path = theow_dir / "rules" / "r.rule.yaml"
    assert promoted_path.exists()
    assert "incomplete" not in bound.tags
    explorer._chroma.index_rule.assert_called_once()


def test_reject_rule():
    explorer = MagicMock()
    explorer._session_cache = MagicMock()
    md = _make_mark_decorator(explorer=explorer)

    rule = Rule(name="bad", description="Bad rule desc", when=[])
    rule._created_files = [Path("/tmp/bad.yaml")]

    rejection = md._reject_rule(rule, ValueError("didn't work"))
    assert rejection["rule_name"] == "bad"
    assert "didn't work" in rejection["error"]
    assert rejection["_files"] == [Path("/tmp/bad.yaml")]
    explorer._session_cache.invalidate.assert_called_with("bad")


def test_recovery_hooks_lifecycle():
    """Hooks wrap recovery attempts, not the initial fn() call."""
    action_registry = ActionRegistry()

    @action_registry.register("fix")
    def fix():
        return True

    rule = Rule(
        name="r",
        description="R",
        when=[Fact(fact="x", equals="y")],
        then=[Action(action="fix")],
    )
    bound = rule.bind({}, {}, action_registry)

    resolver = MagicMock()
    resolver.resolve.return_value = bound
    explorer = MagicMock()
    explorer.cleanup.return_value = None
    md = _make_mark_decorator(resolver=resolver, explorer=explorer)

    log = []

    def my_setup(state, attempt):
        log.append(f"setup:{attempt}:ws={state.get('workspace')}")
        return state

    def my_teardown(state, attempt, success):
        log.append(f"teardown:{attempt}:{success}")

    config = MarkConfig(
        context_from=lambda *a, **k: {"x": "y"},
        max_retries=2,
        rules=None,
        tags=None,
        fallback=True,
        explorable=False,
        collection="default",
        setup=my_setup,
        teardown=my_teardown,
    )

    call_count = 0

    def fn(workspace):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("fails")
        return "ok"

    result = md._run_with_recovery(fn, ("/tmp/ws",), {}, config)
    assert result == "ok"
    # Initial fn() has no hooks. Recovery attempt 1: setup -> recovery -> verify -> teardown
    assert log == [
        "setup:1:ws=/tmp/ws",
        "teardown:1:True",
    ]


def test_setup_failure_aborts_recovery():
    """If setup fails, no more recovery attempts are made."""
    resolver = MagicMock()
    explorer = MagicMock()
    explorer.cleanup.return_value = None
    md = _make_mark_decorator(resolver=resolver, explorer=explorer)

    def bad_setup(state, attempt):
        raise RuntimeError("sandbox failed")

    config = MarkConfig(
        context_from=lambda *a, **k: {},
        max_retries=3,
        rules=None,
        tags=None,
        fallback=True,
        explorable=False,
        collection="default",
        setup=bad_setup,
    )

    call_count = 0

    def fn():
        nonlocal call_count
        call_count += 1
        raise ValueError("fail")

    with pytest.raises(ValueError, match="fail"):
        md._run_with_recovery(fn, (), {}, config)

    # Only the initial attempt, setup fails on first recovery attempt so no retry
    assert call_count == 1


def test_teardown_failure_does_not_propagate():
    """Teardown errors are swallowed, fn() result is returned."""
    action_registry = ActionRegistry()

    @action_registry.register("fix")
    def fix():
        return True

    rule = Rule(
        name="r",
        description="R",
        when=[Fact(fact="x", equals="y")],
        then=[Action(action="fix")],
    )
    bound = rule.bind({}, {}, action_registry)

    resolver = MagicMock()
    resolver.resolve.return_value = bound
    explorer = MagicMock()
    explorer.cleanup.return_value = None
    md = _make_mark_decorator(resolver=resolver, explorer=explorer)

    def bad_teardown(state, attempt, success):
        raise RuntimeError("teardown exploded")

    config = MarkConfig(
        context_from=lambda *a, **k: {"x": "y"},
        max_retries=1,
        rules=None,
        tags=None,
        fallback=True,
        explorable=False,
        collection="default",
        teardown=bad_teardown,
    )

    call_count = 0

    def fn():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("fails")
        return "ok"

    result = md._run_with_recovery(fn, (), {}, config)
    assert result == "ok"
