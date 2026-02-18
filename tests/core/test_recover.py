"""Tests for pure recovery function."""

from unittest.mock import MagicMock

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
    assert run.call_count == 3  # initial + 2 retries


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
