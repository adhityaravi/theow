"""Pure recovery loop shared by @mark decorator and CLI."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

from theow._core._decorators import TracingInfo
from theow._core._logging import get_engine_name, get_logger

if TYPE_CHECKING:
    from theow._core._engine import Theow
    from theow._core._models import Rule

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class Attempt(Generic[T]):
    """Result of running a target (Python function or subprocess)."""

    success: bool
    value: T | None = None
    context: dict[str, Any] = field(default_factory=dict)
    tracing: TracingInfo | None = None


@dataclass
class RecoveryConfig:
    """Configuration for a recovery session."""

    max_retries: int = 3
    rules: list[str] | None = None
    tags: list[str] | None = None
    collection: str = "default"
    fallback: bool = True
    explorable: bool = False


def recover(
    run: Callable[[], Attempt[T]],
    engine: Theow,
    config: RecoveryConfig,
    before_attempt: Callable[[dict[str, Any], int], dict[str, Any] | None] | None = None,
    after_attempt: Callable[[dict[str, Any], int, bool], None] | None = None,
) -> Attempt[T]:
    """Run target with automatic recovery.

    Steps:
        1. Try run(), return if succeeds (no hooks on initial attempt)
        2. For each retry:
           a. before_attempt - lifecycle hook before recovery
           b. _attempt_fix - find rule via resolve or explore
           c. If exploration ran: after_attempt(False) + before_attempt to
              reset the workspace, then execute rule action on clean state
           d. run() again to verify if fix worked
           e. after_attempt - lifecycle hook after recovery
           f. If success: promote ephemeral rule, return
           g. If failure: reject ephemeral rule, update context
        3. _cleanup - clean up ephemeral files after all retries exhausted

    Theow never blocks the consumer pipeline. If recovery fails or theow
    itself errors, the last attempt is returned as-is.

    Args:
        run: Callable that executes the target and returns an Attempt.
        engine: Theow engine instance (provides resolve/explore/execute).
        config: Recovery configuration.
        before_attempt: Optional setup hook(state, attempt) -> state.
        after_attempt: Optional teardown hook(state, attempt, success).

    Returns:
        Final Attempt (success or last failure).
    """
    attempt = run()
    if attempt.success:
        return attempt

    logger.info("Failure captured", error=_truncate(attempt.context.get("stderr", "")))

    hook_state: dict[str, Any] = {}
    failed_rules: list[str] = []
    rejected: list[dict[str, Any]] = []

    try:
        for attempt_num in range(1, config.max_retries + 1):
            if before_attempt:
                try:
                    hook_state = before_attempt(hook_state, attempt_num) or hook_state
                except Exception as hook_err:
                    logger.error("Setup hook failed", error=str(hook_err))
                    break

            rule, explored = _attempt_fix(
                attempt.context,
                engine,
                config,
                failed_rules,
                rejected,
                attempt_num,
                attempt.tracing,
            )

            if rule is None:
                _teardown_failure(engine, hook_state, after_attempt, attempt_num)
                break

            if explored:
                # LLM exploration dirtied the workspace. Reset via the
                # existing hooks so the rule action runs on a clean baseline.
                _teardown_failure(engine, hook_state, after_attempt, attempt_num)
                if before_attempt:
                    try:
                        hook_state = before_attempt(hook_state, attempt_num) or hook_state
                    except Exception as hook_err:
                        logger.error("Setup hook failed after explore reset", error=str(hook_err))
                        break
                if not engine.execute_rule(rule, attempt.context):
                    engine._chroma.update_rule_stats(config.collection, rule.name, False)
                    _teardown_failure(engine, hook_state, after_attempt, attempt_num)
                    failed_rules.append(rule.name)
                    if rule.is_ephemeral:
                        rejected.append(_reject_info(rule, attempt.context))
                        if engine._explorer._session_cache:
                            engine._explorer._session_cache.invalidate(rule.name)
                    continue

            attempt = run()

            if attempt.success:
                done_msg = engine._explorer._last_done_message
                engine._explorer._last_done_message = None  # consume
                if engine._archive_llm_attempt and done_msg:
                    hook_state["_observation"] = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "outcome": "success",
                        "rule": rule.name,
                        "reasoning": done_msg,
                    }
                _call_after(after_attempt, hook_state, attempt_num, True)
                _flush_observation(engine, hook_state)
            else:
                _teardown_failure(engine, hook_state, after_attempt, attempt_num)

            if attempt.success:
                engine._chroma.update_rule_stats(config.collection, rule.name, True)
                if rule.is_ephemeral:
                    _promote_rule(rule, engine)
                return attempt

            # Rule action ran but function still failed; don't retry this rule.
            engine._chroma.update_rule_stats(config.collection, rule.name, False)
            failed_rules.append(rule.name)
            if rule.is_ephemeral:
                rejected.append(_reject_info(rule, attempt.context))
                if engine._explorer._session_cache:
                    engine._explorer._session_cache.invalidate(rule.name)
    except Exception as internal_err:
        logger.error(f"{get_engine_name()} internal error", error=str(internal_err))

    _cleanup(engine, rejected)
    return attempt


def _attempt_fix(
    context: dict[str, Any],
    engine: Theow,
    config: RecoveryConfig,
    failed_rules: list[str],
    rejected: list[dict[str, Any]],
    attempt_num: int,
    tracing: TracingInfo | None,
) -> tuple[Rule | None, bool]:
    """Find and execute a recovery rule.

    Returns:
        (rule, explored) where explored=True if LLM exploration ran.
    """
    rule = engine.resolve(
        context=context,
        collection=config.collection,
        rules=config.rules,
        tags=config.tags,
        fallback=config.fallback,
        n_results=config.max_retries,
        exclude_rules=failed_rules,
    )

    if rule:
        if engine.execute_rule(rule, context):
            return rule, False
        engine._chroma.update_rule_stats(config.collection, rule.name, False)
        failed_rules.append(rule.name)

    if config.explorable and os.environ.get("THEOW_EXPLORE") == "1":
        tools = engine.get_tools()
        explored_rule, explored = engine._explorer.explore(
            context=context,
            tools=tools,
            collection=config.collection,
            tracing=tracing,
            rejected_attempts=rejected,
            attempt_number=attempt_num,
        )
        if explored_rule and explored_rule.name not in failed_rules:
            return explored_rule, True
        if not explored:
            return None, False

    return None, False


def _promote_rule(rule: Rule, engine: Theow) -> None:
    """Promote ephemeral rule to permanent after verification succeeds."""
    logger.debug("Promoting rule", rule=rule.name)

    if "incomplete" in rule.tags:
        rule.tags.remove("incomplete")

    try:
        if rule._source_path and "ephemeral" in str(rule._source_path):
            new_path = rule._source_path.parent.parent / rule._source_path.name
            rule._source_path.rename(new_path)
            rule._source_path = new_path

        if rule._source_path and rule._source_path.exists():
            rule.to_yaml(rule._source_path)

        engine._chroma.index_rule(rule)
    except Exception as e:
        logger.warning("Failed to promote rule", rule=rule.name, error=str(e))


def _reject_info(rule: Rule, context: dict[str, Any]) -> dict[str, Any]:
    """Build rejection info for LLM feedback."""
    logger.warning("Rejecting ephemeral rule", rule=rule.name)
    error_text = context.get("stderr", context.get("error", ""))
    return {
        "rule_name": rule.name,
        "description": rule.description[:200] if rule.description else "",
        "error": str(error_text)[:500],
        "_files": list(rule._created_files),
        "_incomplete": "incomplete" in rule.tags,
    }


def _cleanup(engine: Theow, rejected: list[dict[str, Any]]) -> None:
    """Clean up ephemeral files after recovery exhausted."""
    try:
        engine._explorer.cleanup(rejected)
    except Exception as cleanup_err:
        logger.error("Cleanup failed", error=str(cleanup_err))


def _sync_give_up_reason(engine: Theow, hook_state: dict[str, Any]) -> None:
    """Copy the explorer's give-up reason (if any) into hook_state."""
    reason = engine._explorer._last_give_up_reason
    engine._explorer._last_give_up_reason = None  # consume
    if reason:
        hook_state["_give_up_reason"] = reason
        if engine._archive_llm_attempt:
            hook_state["_observation"] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "outcome": "failure",
                "give_up_reason": reason,
            }


def _flush_observation(engine: Theow, hook_state: dict[str, Any]) -> None:
    """Pop and persist any pending observation from hook_state."""
    if not engine._archive_llm_attempt:
        return
    obs = hook_state.pop("_observation", None)
    if obs:
        engine._flush_observation(obs)


def _teardown_failure(
    engine: Theow,
    hook_state: dict[str, Any],
    after_attempt: Callable[[dict[str, Any], int, bool], None] | None,
    attempt_num: int,
) -> None:
    """Handle failure: sync give-up reason, call teardown, flush observation."""
    _sync_give_up_reason(engine, hook_state)
    _call_after(after_attempt, hook_state, attempt_num, False)
    _flush_observation(engine, hook_state)


def _call_after(
    after_attempt: Callable[[dict[str, Any], int, bool], None] | None,
    state: dict[str, Any],
    attempt: int,
    success: bool,
) -> None:
    """Call teardown hook safely."""
    if not after_attempt:
        return
    try:
        after_attempt(state, attempt, success)
    except Exception as hook_err:
        logger.warning("Teardown hook failed", error=str(hook_err))


def _truncate(text: Any, max_len: int = 200) -> str:
    """Truncate text for log messages."""
    s = str(text)
    return s[:max_len] + "..." if len(s) > max_len else s
