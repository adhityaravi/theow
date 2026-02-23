"""Tests for pure recovery function."""

from unittest.mock import MagicMock, patch

from theow._core._recover import Attempt, RecoveryConfig, recover, _reject_info, _truncate


def _make_engine(resolve_return=None, execute_return=True):
    engine = MagicMock()
    engine.resolve = MagicMock(return_value=resolve_return)
    engine.execute_rule = MagicMock(return_value=execute_return)
    engine._explorer = MagicMock()
    engine._explorer._session_cache = None
    engine._chroma = MagicMock()
    return engine


def test_recover_success_first_try():
    engine = _make_engine()
    run = MagicMock(return_value=Attempt(success=True, value=42))

    result = recover(run, engine, RecoveryConfig())

    assert result.success
    assert result.value == 42
    run.assert_called_once()
    engine.resolve.assert_not_called()


def test_recover_resolve_then_success():
    rule = MagicMock()
    rule.name = "fix-it"
    rule.is_ephemeral = False
    engine = _make_engine(resolve_return=rule)

    calls = [
        Attempt(success=False, context={"stderr": "error"}),
        Attempt(success=True, value="fixed"),
    ]
    run = MagicMock(side_effect=calls)

    result = recover(run, engine, RecoveryConfig(max_retries=2))

    assert result.success
    assert result.value == "fixed"
    engine.execute_rule.assert_called_once_with(rule, {"stderr": "error"})


def test_recover_no_rule_stops():
    engine = _make_engine(resolve_return=None)
    run = MagicMock(return_value=Attempt(success=False, context={"stderr": "fail"}))

    result = recover(run, engine, RecoveryConfig(max_retries=3))

    assert not result.success
    assert run.call_count == 1


def test_recover_retries_exhausted():
    rule = MagicMock()
    rule.name = "flaky"
    rule.is_ephemeral = False
    engine = _make_engine(resolve_return=rule)

    fail = Attempt(success=False, context={"stderr": "still broken"})
    run = MagicMock(return_value=fail)

    result = recover(run, engine, RecoveryConfig(max_retries=2))

    assert not result.success
    assert run.call_count == 3  # initial + 2 retries (mock ignores exclude_rules)


def test_recover_promotes_ephemeral_on_success():
    rule = MagicMock()
    rule.name = "new-rule"
    rule.is_ephemeral = True
    rule.tags = ["incomplete"]
    rule._source_path = None
    engine = _make_engine(resolve_return=rule)

    calls = [
        Attempt(success=False, context={"stderr": "err"}),
        Attempt(success=True, value="ok"),
    ]
    run = MagicMock(side_effect=calls)

    result = recover(run, engine, RecoveryConfig())

    assert result.success
    engine._chroma.index_rule.assert_called_once_with(rule)
    assert "incomplete" not in rule.tags


def test_recover_rejects_ephemeral_on_failure():
    rule = MagicMock()
    rule.name = "bad-rule"
    rule.is_ephemeral = True
    rule.tags = []
    rule._created_files = []
    engine = _make_engine(resolve_return=rule)
    engine._explorer._session_cache = None

    fail = Attempt(success=False, context={"stderr": "nope"})
    run = MagicMock(return_value=fail)

    result = recover(run, engine, RecoveryConfig(max_retries=1))

    assert not result.success
    engine._explorer.cleanup.assert_called_once()


def test_recover_calls_hooks():
    engine = _make_engine(resolve_return=None)
    run = MagicMock(return_value=Attempt(success=False, context={"stderr": "err"}))

    before = MagicMock(return_value={"setup": True})
    after = MagicMock()

    recover(run, engine, RecoveryConfig(max_retries=1), before_attempt=before, after_attempt=after)

    before.assert_called_once()
    after.assert_called_once()


def test_recover_internal_error_returns_last_attempt():
    engine = _make_engine()
    engine.resolve.side_effect = RuntimeError("internal boom")

    fail = Attempt(success=False, context={"stderr": "err"})
    run = MagicMock(return_value=fail)

    result = recover(run, engine, RecoveryConfig(max_retries=2))

    assert not result.success


def test_reject_info_structure():
    rule = MagicMock()
    rule.name = "test-rule"
    rule.description = "A test rule"
    rule.tags = []
    rule._created_files = []

    info = _reject_info(rule, {"stderr": "something failed"})

    assert info["rule_name"] == "test-rule"
    assert info["error"] == "something failed"
    assert "_files" in info


def test_truncate():
    assert _truncate("short", 10) == "short"
    assert _truncate("a" * 300, 200) == "a" * 200 + "..."


@patch("theow._core._recover._attempt_fix")
def test_recover_progress_detection(mock_attempt_fix):
    """When a rule fixes its target error but reveals a new one, recovery continues."""
    rule_a = MagicMock()
    rule_a.name = "fix-error-a"
    rule_a.is_ephemeral = False
    rule_a.matches = lambda ctx: {} if "error A" in ctx.get("stderr", "") else None

    rule_b = MagicMock()
    rule_b.name = "fix-error-b"
    rule_b.is_ephemeral = False
    rule_b.matches = lambda ctx: {} if "error B" in ctx.get("stderr", "") else None

    engine = _make_engine()

    mock_attempt_fix.side_effect = [
        (rule_a, False),
        (rule_b, False),
    ]

    run = MagicMock(side_effect=[
        Attempt(success=False, context={"stderr": "error A"}),  # initial
        Attempt(success=False, context={"stderr": "error B"}),  # after rule_a → progress
        Attempt(success=True, value="fixed"),                    # after rule_b → success
    ])

    result = recover(run, engine, RecoveryConfig(max_retries=2, max_depth=3))

    assert result.success
    assert result.value == "fixed"
    assert mock_attempt_fix.call_count == 2
    # Both rules recorded as success
    engine._chroma.update_rule_stats.assert_any_call("default", "fix-error-a", True)
    engine._chroma.update_rule_stats.assert_any_call("default", "fix-error-b", True)


@patch("theow._core._recover._attempt_fix")
def test_recover_progress_resets_retries(mock_attempt_fix):
    """On progress, retry counter resets so the new error gets full budget."""
    rule_a = MagicMock()
    rule_a.name = "fix-error-a"
    rule_a.is_ephemeral = False
    rule_a.matches = lambda ctx: {} if "error A" in ctx.get("stderr", "") else None

    rule_b = MagicMock()
    rule_b.name = "fix-error-b"
    rule_b.is_ephemeral = False
    # rule_b matches error B (no progress on failure)
    rule_b.matches = lambda ctx: {} if "error B" in ctx.get("stderr", "") else None

    engine = _make_engine()

    mock_attempt_fix.side_effect = [
        (rule_a, False),  # depth 0
        (rule_b, False),  # depth 1, retry 1
        (rule_b, False),  # depth 1, retry 2 — rule_b retried after reset
    ]

    run = MagicMock(side_effect=[
        Attempt(success=False, context={"stderr": "error A"}),  # initial
        Attempt(success=False, context={"stderr": "error B"}),  # after rule_a → progress
        Attempt(success=False, context={"stderr": "error B"}),  # after rule_b retry 1 → no progress
        Attempt(success=True, value="done"),                     # after rule_b retry 2 → success
    ])

    result = recover(run, engine, RecoveryConfig(max_retries=2, max_depth=3))

    assert result.success
    assert mock_attempt_fix.call_count == 3


@patch("theow._core._recover._attempt_fix")
def test_recover_max_depth_exceeded(mock_attempt_fix):
    """Recovery stops when max_depth is reached even if progress keeps happening."""
    rules = []
    for i in range(4):
        rule = MagicMock()
        rule.name = f"fix-error-{i}"
        rule.is_ephemeral = False
        error_text = f"error {i}"
        rule.matches = lambda ctx, et=error_text: {} if et in ctx.get("stderr", "") else None
        rules.append(rule)

    engine = _make_engine()
    mock_attempt_fix.side_effect = [(r, False) for r in rules]

    run = MagicMock(side_effect=[
        Attempt(success=False, context={"stderr": "error 0"}),  # initial
        Attempt(success=False, context={"stderr": "error 1"}),  # after rule 0 → progress, depth=1
        Attempt(success=False, context={"stderr": "error 2"}),  # after rule 1 → progress, depth=2
    ])

    result = recover(run, engine, RecoveryConfig(max_retries=3, max_depth=2))

    assert not result.success
    # Only 2 rules attempted (depth 0 and depth 1), stopped at depth=2
    assert mock_attempt_fix.call_count == 2
    assert run.call_count == 3  # initial + 2 retries


@patch("theow._core._recover._attempt_fix")
def test_recover_progress_on_changed_captures(mock_attempt_fix):
    """Same rule matches but with different captures = progress (fixed one instance)."""
    rule = MagicMock()
    rule.name = "fix-mismatch"
    rule.is_ephemeral = False
    # Matches both errors but with different captures
    def match_fn(ctx):
        stderr = ctx.get("stderr", "")
        if "module-A" in stderr:
            return {"mod": "module-A"}
        if "module-B" in stderr:
            return {"mod": "module-B"}
        return None
    rule.matches = match_fn

    engine = _make_engine()

    mock_attempt_fix.side_effect = [
        (rule, False),  # depth 0: resolve rule for module-A
        (rule, False),  # depth 1: same rule re-resolved for module-B
    ]

    run = MagicMock(side_effect=[
        Attempt(success=False, context={"stderr": "error: module-A mismatch"}),
        Attempt(success=False, context={"stderr": "error: module-B mismatch"}),
        Attempt(success=True, value="fixed"),
    ])

    result = recover(run, engine, RecoveryConfig(max_retries=3, max_depth=3))

    assert result.success
    assert result.value == "fixed"
    engine._chroma.update_rule_stats.assert_any_call("default", "fix-mismatch", True)


@patch("theow._core._recover._attempt_fix")
def test_recover_breaks_on_give_up(mock_attempt_fix):
    """Recovery loop should stop immediately when LLM explicitly gives up."""
    rule = MagicMock()
    rule.name = "explored-rule"
    rule.is_ephemeral = False

    engine = _make_engine(execute_return=False)

    # Simulate LLM calling _give_up during execute_rule:
    # the reason is set as a side effect of execution, not before.
    def set_give_up_on_execute(*_args, **_kwargs):
        engine._explorer._last_give_up_reason = "This error is unfixable"
        return False

    engine.execute_rule = MagicMock(side_effect=set_give_up_on_execute)

    # _attempt_fix returns an explored rule (explored=True)
    mock_attempt_fix.return_value = (rule, True)

    fail = Attempt(success=False, context={"stderr": "unfixable error"})
    run = MagicMock(return_value=fail)

    result = recover(run, engine, RecoveryConfig(max_retries=3))

    assert not result.success
    # Only the initial run() call — the loop should break after first attempt
    assert run.call_count == 1
    # _attempt_fix called once, then loop breaks (not 3 times)
    assert mock_attempt_fix.call_count == 1
