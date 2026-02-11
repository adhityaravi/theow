"""Conversational LLM exploration for novel situations."""

from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from theow._core._logging import get_logger
from theow._core._models import Rule
from theow._core._prompts import ERROR, INTRO, TEMPLATES
from theow._core._session_cache import SessionCache
from theow._core._tools import (
    Done,
    ExplorationSignal,
    GiveUp,
    RequestTemplates,
    SubmitRule,
    make_direct_fix_tools,
    make_ephemeral_tools,
    make_search_tools,
    make_signal_tools,
)

if TYPE_CHECKING:
    from theow._core._chroma_store import ChromaStore
    from theow._core._decorators import ActionRegistry, TracingInfo
    from theow._gateway._base import LLMGateway

logger = get_logger(__name__)


class Explorer:
    """Conversational LLM exploration for novel situations."""

    def __init__(
        self,
        chroma: ChromaStore,
        gateway: LLMGateway | None,
        action_registry: ActionRegistry,
        rules_dir: Path,
        session_limit: int = 20,
        max_tool_calls: int = 30,
        max_tokens: int = 8192,
    ) -> None:
        self._chroma = chroma
        self._gateway = gateway
        self._action_registry = action_registry
        self._rules_dir = rules_dir
        self._session_limit = session_limit
        self._max_tool_calls = max_tool_calls
        self._max_tokens = max_tokens
        self._session_count = 0
        self._session_cache: _SessionCache | None = None

    def set_gateway(self, gateway: LLMGateway) -> None:
        """Set the LLM gateway (for lazy initialization)."""
        self._gateway = gateway

    def explore(
        self,
        context: dict[str, Any],
        tools: list[Callable[..., Any]],
        collection: str = "default",
        tracing: TracingInfo | None = None,
        rejected_attempts: list[dict[str, Any]] | None = None,
        attempt_number: int = 1,
    ) -> tuple[Rule | None, bool]:
        """Explore a novel situation using LLM.

        Returns:
            (rule, explored) tuple where:
            - rule: ephemeral Rule for decorator to validate via actual fn() execution
            - explored: True if exploration was attempted (even if no rule produced)

        Does NOT run action or retry here; that's the decorator's job.
        """
        if self._gateway is None:
            logger.warning("Exploration disabled", reason="no LLM configured")
            return None, False

        if self._session_count >= self._session_limit:
            logger.warning(
                "Session limit reached", count=self._session_count, limit=self._session_limit
            )
            return None, False

        if self._session_cache is None:
            self._session_cache = SessionCache()

        cached = self._session_cache.check(context)
        if cached:
            return cached, False

        chroma_match = self._check_chroma(context, collection)
        if chroma_match:
            return chroma_match, False

        self._session_count += 1
        logger.info("Investigating", session=f"{self._session_count}/{self._session_limit}")

        rule, explored = self._run_conversation(
            context, tools, collection, tracing, rejected_attempts, attempt_number
        )
        if rule:
            self._session_cache.store(context, rule)

        return rule, explored

    def _check_chroma(self, context: dict[str, Any], collection: str) -> Rule | None:
        query_text = self._extract_query_text(context)
        if not query_text:
            return None

        results = self._chroma.query_rules(
            collection=collection, query_text=query_text, n_results=1
        )
        if not results:
            return None

        rule_name, distance, _ = results[0]
        if distance > 0.3:
            return None

        rule_path = self._rules_dir / f"{rule_name}.rule.yaml"
        if not rule_path.exists():
            return None

        rule = Rule.from_yaml(rule_path)
        captures = rule.matches(context)
        if captures is not None:
            logger.info("Chroma match", rule=rule_name)
            return rule.bind(captures, context, self._action_registry)

        return None

    def _run_conversation(
        self,
        context: dict[str, Any],
        tools: list[Callable[..., Any]],
        collection: str,
        tracing: TracingInfo | None,
        rejected_attempts: list[dict[str, Any]] | None,
        attempt_number: int = 1,
    ) -> tuple[Rule | None, bool]:
        """Run multi-phase conversation with LLM.

        Returns:
            (rule, explored) tuple where explored=True indicates exploration was attempted.
        """
        signal_tools = make_signal_tools()
        search_tools = make_search_tools(self._chroma, collection)
        ephemeral_tools = make_ephemeral_tools(self._rules_dir)
        all_tools = signal_tools + search_tools + ephemeral_tools + tools

        tools_section = self._build_tools_section(all_tools)
        intro = INTRO.format(
            tools_section=tools_section,
            rules_dir=self._rules_dir,
            actions_dir=self._rules_dir.parent / "actions",
        )
        error_prompt = self._build_error_prompt(context, tracing, rejected_attempts, attempt_number)
        initial_prompt = intro + "\n" + error_prompt
        messages = [{"role": "user", "content": initial_prompt}]

        logger.info(
            "Theow --> LLM",
            prompt_tokens_est=len(initial_prompt) // 4,
            tools=len(all_tools),
        )

        signal = self._converse(messages, all_tools)

        return self._handle_signal(signal, messages, all_tools, context, collection)

    def _converse(
        self,
        messages: list[dict[str, Any]],
        tools: list[Callable[..., Any]],
        budget: dict[str, Any] | None = None,
    ) -> ExplorationSignal | None:
        """Run conversation until signal or budget exhausted."""
        assert self._gateway is not None
        if budget is None:
            budget = {"max_tool_calls": self._max_tool_calls, "max_tokens": self._max_tokens}
        try:
            self._gateway.conversation(
                messages=messages,
                tools=tools,
                budget=budget,
            )
            return None
        except ExplorationSignal as signal:
            return signal
        except Exception as e:
            logger.error("Conversation failed", error=str(e))
            return None

    def _handle_signal(
        self,
        signal: ExplorationSignal | None,
        messages: list[dict[str, Any]],
        tools: list[Callable[..., Any]],
        context: dict[str, Any],
        collection: str,
    ) -> tuple[Rule | None, bool]:
        """Handle exploration signal with pattern matching.

        Returns:
            (rule, explored) tuple where explored=True indicates exploration was attempted.
        """
        match signal:
            case None:
                # Budget exhausted - tag any orphaned rules as incomplete for next attempt
                orphaned = self._find_orphaned_rules()
                if orphaned:
                    self._ensure_incomplete_tag(orphaned)
                    logger.info("Budget exhausted, marked rule as incomplete", rule=orphaned.name)
                else:
                    logger.warning("Exploration completed without signal")
                return None, True  # explored=True, no rule to validate

            case GiveUp(reason=reason):
                logger.warning("Exploration declined", reason=reason)
                return None, True

            case RequestTemplates():
                logger.info("Writing rule")
                templates = TEMPLATES.format(
                    rules_dir=self._rules_dir,
                    actions_dir=self._rules_dir.parent / "actions",
                )
                messages.append({"role": "user", "content": templates})
                signal = self._converse(messages, tools)
                return self._handle_signal(signal, messages, tools, context, collection)

            case SubmitRule(rule_file=rule_file, action_file=action_file):
                rule = self._validate_rule(rule_file, action_file, context, collection)
                return rule, True

            case _:
                logger.error("Unknown signal", signal=type(signal).__name__)
                return None, True

    def _build_error_prompt(
        self,
        context: dict[str, Any],
        tracing: TracingInfo | None,
        rejected_attempts: list[dict[str, Any]] | None,
        attempt_number: int = 1,
    ) -> str:
        context_lines = "\n".join(f"{k}: {v}" for k, v in context.items())

        tracing_section = ""
        if tracing:
            tracing_section = f"""
## Tracing

{tracing.exception_type}: {tracing.exception_message}

{tracing.traceback}"""

        attempt_section = ""
        if attempt_number > 1:
            attempt_section = f"""
## Continuation

This is attempt {attempt_number}. Previous exploration attempts may have saved
incomplete rules. Use `list_ephemeral_rules()` to check for work to continue from.
"""

        rejected_section = ""
        if rejected_attempts:
            rejected_lines = []
            for attempt in rejected_attempts[-3:]:  # Limit to last 3
                desc = attempt.get("description", "")
                desc_part = f" ({desc})" if desc else ""
                rejected_lines.append(f"- {attempt['rule_name']}{desc_part}: {attempt['error']}")
            rejected_section = f"""
## Previous Failed Attempts

These rules were tried but did not fix the problem (files deleted, don't try to read them):
{chr(10).join(rejected_lines)}

Try a different approach."""

        return (
            ERROR.format(
                context=context_lines,
                tracing=tracing_section,
            )
            + attempt_section
            + rejected_section
        )

    def _validate_rule(
        self,
        rule_file: str,
        action_file: str | None,
        context: dict[str, Any],
        collection: str,
    ) -> Rule | None:
        """Validate rule syntax and structure. Does NOT execute action.

        Actual validation happens when decorator runs fn() after action.
        """
        rule_path = Path(rule_file)

        if not rule_path.exists():
            logger.error("Rule file not found", path=rule_file)
            return None

        try:
            rule = Rule.from_yaml(rule_path)
        except Exception as e:
            logger.error("Failed to parse rule", error=str(e))
            return None

        # Override collection - don't trust LLM's choice
        rule.collection = collection

        captures = rule.matches(context)
        if captures is None:
            logger.error("Rule facts don't match context", rule=rule.name)
            return None

        # Load newly written action file if provided
        if action_file:
            action_path = Path(action_file)
            if action_path.exists():
                self._action_registry._load_action_file(action_path)

        for action in rule.then:
            if not self._action_registry.exists(action.action):
                logger.error("Action not found", action=action.action)
                return None

        conflict = self._check_conflict(rule)
        if conflict:
            logger.error("Rule conflict", conflict=conflict)
            return None

        # Track created files for cleanup on rejection
        # Only includes files created during THIS exploration (not pre-existing actions)
        created_files = [rule_path]
        if action_file:
            action_path = Path(action_file)
            if action_path.exists():
                created_files.append(action_path)

        bound_rule = rule.bind(captures, context, self._action_registry)
        bound_rule._created_files = created_files

        # Don't index ephemeral rules - they get indexed when promoted
        # This avoids ghost entries if validation fails or process crashes
        logger.info("Rule validated", rule=rule.name)

        return bound_rule

    def _find_orphaned_rules(self) -> Rule | None:
        """Find rule files written during session but not submitted.

        Checks for recent .rule.yaml files in ephemeral folder.
        """
        ephemeral_dir = self._rules_dir / "ephemeral"
        if not ephemeral_dir.exists():
            return None

        cutoff = time.time() - 300  # Last 5 minutes

        for path in ephemeral_dir.glob("*.rule.yaml"):
            if path.stat().st_mtime < cutoff:
                continue

            try:
                return Rule.from_yaml(path)
            except Exception:
                continue

        return None

    def _ensure_incomplete_tag(self, rule: Rule) -> None:
        """Add 'incomplete' tag and save to file."""
        if "incomplete" not in rule.tags:
            rule.tags.append("incomplete")

        # Save updated tags to file
        if rule._source_path and rule._source_path.exists():
            rule.to_yaml(rule._source_path)

    def _check_conflict(self, rule: Rule) -> str | None:
        existing_rules = self._chroma.list_rules(rule.collection)

        for name in existing_rules:
            if name == rule.name:
                continue

            path = self._rules_dir / f"{name}.rule.yaml"
            if not path.exists():
                continue

            existing = Rule.from_yaml(path)
            if self._same_when(rule, existing) and not self._same_then(rule, existing):
                return f"Conflicts with {name}: same when, different then"

        return None

    def _build_tools_section(self, tools: list[Callable[..., Any]]) -> str:
        """Build tools documentation from actual registered tools."""
        lines = []
        for fn in tools:
            name = getattr(fn, "__name__", str(id(fn)))
            sig = inspect.signature(fn)
            params = ", ".join(sig.parameters.keys())
            doc = inspect.getdoc(fn) or ""
            first_line = doc.split("\n")[0] if doc else ""
            lines.append(f"- `{name}({params})` - {first_line}")
        return "\n".join(lines)

    def _same_when(self, a: Rule, b: Rule) -> bool:
        return a.when == b.when

    def _same_then(self, a: Rule, b: Rule) -> bool:
        return a.then == b.then

    def _extract_query_text(self, context: dict[str, Any]) -> str:
        longest = ""
        for value in context.values():
            if isinstance(value, str) and len(value) > len(longest):
                longest = value
        return longest

    def run_direct(
        self,
        prompt: str,
        tools: list[Callable[..., Any]],
        budget: dict[str, Any],
    ) -> bool:
        """Run direct fix conversation (for probabilistic rules).

        Returns True if LLM signaled Done, False otherwise.
        """
        if self._gateway is None:
            logger.warning("Cannot run direct fix: no LLM configured")
            return False

        # Add signal tools (give_up, done)
        all_tools = make_direct_fix_tools() + tools

        messages = [{"role": "user", "content": prompt}]
        signal = self._converse(messages, all_tools, budget)

        match signal:
            case Done(message=message):
                logger.info("Direct fix completed", message=message)
                return True
            case GiveUp(reason=reason):
                logger.warning("Direct fix gave up", reason=reason)
                return False
            case None:
                logger.warning("Direct fix exhausted budget without signal")
                return False
            case _:
                logger.warning("Unexpected signal in direct fix", signal=type(signal).__name__)
                return False

    @property
    def session_count(self) -> int:
        return self._session_count

    def reset_session(self) -> None:
        self._session_count = 0
        if self._session_cache:
            self._session_cache.clear()


_SessionCache = Any
