"""Pure recovery loop shared by @mark decorator and CLI."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

from theow._core._decorators import TracingInfo
from theow._core._logging import get_engine_name, get_logger
from theow._core._models import LLMRecord

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
    max_depth: int = 3
    rules: list[str] | None = None
    tags: list[str] | None = None
    collection: str = "default"
    fallback: bool = True
    explorable: bool = False
    hint: str | None = None
    allow_escalation: bool = False


class _RecoveryLoop:
    """Encapsulates mutable state and side effects for the recovery loop."""

    def __init__(
        self,
        engine: Theow,
        config: RecoveryConfig,
        before_attempt: Callable[[dict[str, Any], int], dict[str, Any] | None] | None,
        after_attempt: Callable[[dict[str, Any], int, bool], None] | None,
    ) -> None:
        self._engine = engine
        self._config = config
        self._before = before_attempt
        self._after = after_attempt

        self._hook_state: dict[str, Any] = {}
        self._failed_rules: list[str] = []
        self._rejected: list[dict[str, Any]] = []
        self._depth = 0
        self._retry = 0
        self._attempt_num = 0
        self._explored = False
        self._seen_errors: set[str] = set()
        self.abort = False

    def check_cycle(self, context: dict[str, Any]) -> bool:
        """Check if this error context has been seen before (cycle detection).

        Returns True if context fingerprint was already seen (cycle).
        Otherwise records it and returns False.
        """
        fp = self._error_fingerprint(context)
        if fp in self._seen_errors:
            return True
        self._seen_errors.add(fp)
        return False

    @staticmethod
    def _error_fingerprint(context: dict[str, Any]) -> str:
        """Create a hash fingerprint of error context for cycle detection."""
        raw = str(sorted((k, str(v)[:500]) for k, v in context.items()))
        return hashlib.sha256(raw.encode()).hexdigest()

    @property
    def max_iterations(self) -> int:
        extra = self._config.max_depth if self._config.explorable else 0
        return self._config.max_retries * self._config.max_depth + extra

    def has_budget(self) -> bool:
        # Guarantee at least one explore attempt when explore is enabled.
        if (
            self._config.explorable
            and not self._explored
            and self._retry >= self._config.max_retries
        ):
            return True
        return self._retry < self._config.max_retries

    def setup(self) -> bool:
        """Increment counters and run before_attempt hook. False = abort."""
        self._retry += 1
        self._attempt_num += 1
        if not self._before:
            return True
        try:
            self._hook_state = self._before(self._hook_state, self._attempt_num) or self._hook_state
            return True
        except Exception as err:
            logger.error("Setup hook failed", error=str(err))
            return False

    def find_rule(
        self,
        context: dict[str, Any],
        tracing: TracingInfo | None,
    ) -> tuple[Rule | None, bool]:
        return _attempt_fix(
            context,
            self._engine,
            self._config,
            self._failed_rules,
            self._rejected,
            self._attempt_num,
            tracing,
        )

    def teardown_no_rule(self) -> None:
        _teardown_failure(
            self._engine,
            self._hook_state,
            self._after,
            self._attempt_num,
            rule_name="explore",
        )

    def replay_explored_rule(self, rule: Rule, context: dict[str, Any]) -> bool:
        """Reset workspace after exploration, replay rule on clean state.

        Returns True if rule executed successfully (caller should verify).
        Returns False with self.abort=True to break, or False to continue.
        """
        _teardown_failure(
            self._engine,
            self._hook_state,
            self._after,
            self._attempt_num,
            rule_name=rule.name,
        )
        if self._before:
            try:
                self._hook_state = (
                    self._before(self._hook_state, self._attempt_num) or self._hook_state
                )
            except Exception as err:
                logger.error("Setup hook failed after explore reset", error=str(err))
                self.abort = True
                return False

        if self._engine.execute_rule(rule, context):
            return True

        self._engine._chroma.update_rule_stats(self._config.collection, rule.name, False)
        gave_up = self._engine._explorer._last_give_up_reason
        if not gave_up and self._try_escalate_action_failure(rule, context):
            return True
        _teardown_failure(
            self._engine,
            self._hook_state,
            self._after,
            self._attempt_num,
            rule_name=rule.name,
        )
        self._mark_failed(rule, context)
        if gave_up:
            self.abort = True
        return False

    def on_success(self, rule: Rule) -> None:
        """Handle successful verification: teardown hooks + promote."""
        done_msg = self._engine._explorer._last_done_message
        self._engine._explorer._last_done_message = None  # consume
        if self._engine._archive_llm_attempt and done_msg:
            self._hook_state["_observation"] = LLMRecord(
                outcome="success",
                rule=rule.name,
                reason=done_msg,
            )
        _call_after(self._after, self._hook_state, self._attempt_num, True)
        _flush_observation(self._engine, self._hook_state)
        self._engine._chroma.update_rule_stats(self._config.collection, rule.name, True)
        if rule.is_ephemeral:
            _promote_rule(rule, self._engine)

    def on_progress(self, rule: Rule) -> bool:
        """Rule fixed its issue but a new error appeared.

        Returns True if recovery can continue at the next depth level.
        """
        logger.info("Recovery progress", rule=rule.name, depth=self._depth)
        self._engine._chroma.update_rule_stats(self._config.collection, rule.name, True)
        if rule.is_ephemeral:
            _promote_rule(rule, self._engine)
        _call_after(self._after, self._hook_state, self._attempt_num, True)
        _flush_observation(self._engine, self._hook_state)
        self._failed_rules.clear()
        self._rejected.clear()
        self._hook_state = {}
        self._depth += 1
        self._retry = 0
        self._explored = False
        if self._depth >= self._config.max_depth:
            logger.warning("Max recovery depth reached", depth=self._depth)
            return False
        return True

    def execute_rule(self, rule: Rule, context: dict[str, Any]) -> bool:
        """Execute a resolved rule's action. Returns True if action succeeded."""
        return self._engine.execute_rule(rule, context)

    def on_action_failed(self, rule: Rule, context: dict[str, Any]) -> None:
        """Handle action execution failure: teardown + mark as failed."""
        self._engine._chroma.update_rule_stats(self._config.collection, rule.name, False)
        _teardown_failure(
            self._engine,
            self._hook_state,
            self._after,
            self._attempt_num,
            rule_name=rule.name,
        )
        self._mark_failed(rule, context)

    def on_failure(self, rule: Rule, context: dict[str, Any]) -> None:
        """Handle failed verification: teardown + mark rule as failed."""
        # If LLM claimed success but verification failed, record as unverified
        done_msg = self._engine._explorer._last_done_message
        self._engine._explorer._last_done_message = None  # consume
        if self._engine._archive_llm_attempt and done_msg:
            self._hook_state["_observation"] = LLMRecord(
                outcome="unverified",
                rule=rule.name,
                reason=done_msg,
            )
        _teardown_failure(
            self._engine,
            self._hook_state,
            self._after,
            self._attempt_num,
            rule_name=rule.name,
        )
        self._engine._chroma.update_rule_stats(self._config.collection, rule.name, False)
        self._mark_failed(rule, context)

    def try_escalate_unverified(self, rule: Rule, context: dict[str, Any]) -> bool:
        """Try escalating to secondary after primary claimed done but verification failed.

        Returns True if escalation was attempted (caller should re-run and verify).
        """
        done_msg = self._engine._explorer._last_done_message
        if not done_msg:
            return False
        if not self._config.allow_escalation:
            return False
        if self._engine._explorer._secondary_gateway is None:
            return False

        # Consume the done message so it's not reused
        self._engine._explorer._last_done_message = None

        error_text = context.get("stderr", context.get("error", ""))
        findings = (
            f"The previous model claimed fix: {done_msg}. "
            f"Verification failed with: {str(error_text)[:500]}"
        )
        return self._engine.execute_rule(rule, context, escalation_context=findings)

    def _try_escalate_action_failure(self, rule: Rule, context: dict[str, Any]) -> bool:
        """Flag explorer to use secondary gateway on next exploration.

        When an explored rule's action fails, the recovery loop already retries
        with a new exploration.  This just ensures that next exploration uses
        the secondary (escalated) model instead of the primary.
        """
        if not self._config.allow_escalation:
            return False
        if self._engine._explorer._secondary_gateway is None:
            return False

        logger.info("Escalating after action failure", rule=rule.name)
        self._engine._explorer._escalate_next = True
        return False

    def cleanup(self) -> None:
        _cleanup(self._engine, self._rejected)

    def _mark_failed(self, rule: Rule, context: dict[str, Any]) -> None:
        self._failed_rules.append(rule.name)
        if rule.is_ephemeral:
            self._rejected.append(_reject_info(rule, context))
            if self._engine._explorer._session_cache:
                self._engine._explorer._session_cache.invalidate(rule.name)


def recover(
    run: Callable[[], Attempt[T]],
    engine: Theow,
    config: RecoveryConfig,
    before_attempt: Callable[[dict[str, Any], int], dict[str, Any] | None] | None = None,
    after_attempt: Callable[[dict[str, Any], int, bool], None] | None = None,
) -> Attempt[T]:
    """Run target with automatic recovery.

    The loop has two counters: ``retry`` resets on progress, ``depth``
    increments.  Progress is detected when the applied rule no longer
    matches the new context (its target error is gone, a different one
    appeared).  On progress the workspace changes are kept and recovery
    continues against the new error up to ``max_depth`` times.

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
    loop = _RecoveryLoop(engine, config, before_attempt, after_attempt)
    loop.check_cycle(attempt.context)  # seed with initial error

    try:
        for _ in range(loop.max_iterations):
            if not loop.has_budget():
                break
            if not loop.setup():
                break

            rule, explored = loop.find_rule(attempt.context, attempt.tracing)
            if rule is None:
                loop.teardown_no_rule()
                break

            if explored:
                loop._explored = True
                if not loop.replay_explored_rule(rule, attempt.context):
                    if loop.abort:
                        break
                    continue
            else:
                if not loop.execute_rule(rule, attempt.context):
                    loop.on_action_failed(rule, attempt.context)
                    continue

            prev_captures = rule.matches(attempt.context)
            attempt = run()

            if attempt.success:
                loop.on_success(rule)
                return attempt

            # Progress: rule no longer matches, or matches with different
            # captures (fixed one instance, another instance of same pattern).
            new_captures = rule.matches(attempt.context)
            if new_captures is None or new_captures != prev_captures:
                if loop.check_cycle(attempt.context):
                    logger.warning(
                        "Recovery cycle detected",
                        rule=rule.name,
                        depth=loop._depth,
                    )
                    break
                if not loop.on_progress(rule):
                    break
                continue

            # Try escalating to secondary before giving up on this attempt
            if loop.try_escalate_unverified(rule, attempt.context):
                attempt = run()
                if attempt.success:
                    loop.on_success(rule)
                    return attempt

            loop.on_failure(rule, attempt.context)
    except Exception as internal_err:
        logger.error(f"{get_engine_name()} internal error", error=str(internal_err))

    loop.cleanup()
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
    """Find a recovery rule via resolve or explore.  Does NOT execute.

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
        return rule, False

    if config.explorable and os.environ.get("THEOW_EXPLORE") == "1":
        tools = engine.get_tools()
        explored_rule, explored = engine._explorer.explore(
            context=context,
            tools=tools,
            collection=config.collection,
            tracing=tracing,
            rejected_attempts=rejected,
            attempt_number=attempt_num,
            hint=config.hint,
            exclude_rules=failed_rules,
            allow_escalation=config.allow_escalation,
        )
        if explored_rule:
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


def _sync_give_up_reason(
    engine: Theow, hook_state: dict[str, Any], rule_name: str | None = None
) -> None:
    """Copy the explorer's give-up reason (if any) into hook_state."""
    reason = engine._explorer._last_give_up_reason
    engine._explorer._last_give_up_reason = None  # consume
    if reason:
        hook_state["_give_up_reason"] = reason
        if engine._archive_llm_attempt:
            hook_state["_observation"] = LLMRecord(
                outcome="failure",
                rule=rule_name or "explore",
                reason=reason,
            )


def _flush_observation(engine: Theow, hook_state: dict[str, Any]) -> None:
    """Pop and persist any pending observation from hook_state."""
    if not engine._archive_llm_attempt:
        return
    obs: LLMRecord | None = hook_state.pop("_observation", None)
    if obs:
        engine._flush_observation(obs.to_dict())


def _teardown_failure(
    engine: Theow,
    hook_state: dict[str, Any],
    after_attempt: Callable[[dict[str, Any], int, bool], None] | None,
    attempt_num: int,
    rule_name: str | None = None,
) -> None:
    """Handle failure: sync give-up reason, call teardown, flush observation."""
    _sync_give_up_reason(engine, hook_state, rule_name)
    _call_after(after_attempt, hook_state, attempt_num, False)
    _flush_observation(engine, hook_state)


def _call_after(
    after_attempt: Callable[[dict[str, Any], int, bool], None] | None,
    state: dict[str, Any],
    attempt: int,
    success: bool,
) -> None:
    """Call teardown hook safely.

    By default, exceptions are suppressed with a warning. The teardown
    can set ``state["suppress_exc"] = False`` before raising to make
    the exception propagate to the caller.
    """
    if not after_attempt:
        return
    try:
        after_attempt(state, attempt, success)
    except Exception as hook_err:
        if state.get("suppress_exc", True):
            logger.warning("Teardown hook failed", error=str(hook_err))
        else:
            logger.error("Teardown hook failed", error=str(hook_err))
            raise


def _truncate(text: Any, max_len: int = 200) -> str:
    """Truncate text for log messages."""
    s = str(text)
    return s[:max_len] + "..." if len(s) > max_len else s
