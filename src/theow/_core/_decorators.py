"""Tool, action, and mark decorators."""

from __future__ import annotations

import ast
import functools
import inspect
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ParamSpec, TypeVar

from theow._core._logging import get_logger

if TYPE_CHECKING:
    from theow._core._models import Rule
    from theow._core._resolver import Resolver
    from theow._core._explorer import Explorer

logger = get_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class TracingInfo:
    """Python tracing info for LLM exploration."""

    traceback: str
    exception_type: str
    exception_message: str


class ToolRegistry:
    """Registry of tools available to LLM during exploration."""

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}

    def register(self, name: str | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
        def decorator(fn: Callable[P, R]) -> Callable[P, R]:
            tool_name = name or getattr(fn, "__name__", str(id(fn)))
            self._tools[tool_name] = fn
            return fn

        return decorator

    def get(self, name: str) -> Callable[..., Any] | None:
        return self._tools.get(name)

    def get_all(self) -> dict[str, Callable[..., Any]]:
        return dict(self._tools)

    def get_declarations(self) -> list[dict[str, Any]]:
        """Generate LLM function schemas from type hints."""
        return [_build_function_declaration(name, fn) for name, fn in self._tools.items()]


def _build_function_declaration(name: str, fn: Callable[..., Any]) -> dict[str, Any]:
    sig = inspect.signature(fn)
    doc = inspect.getdoc(fn) or ""

    properties: dict[str, dict[str, str]] = {}
    required: list[str] = []

    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        if param.annotation != inspect.Parameter.empty:
            origin = getattr(param.annotation, "__origin__", param.annotation)
            param_type = type_map.get(origin, "string")
        else:
            param_type = "string"

        properties[param_name] = {"type": param_type}

        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    return {
        "name": name,
        "description": doc,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


class ActionRegistry:
    """Registry of actions that rules can invoke."""

    def __init__(self) -> None:
        self._actions: dict[str, Callable[..., Any]] = {}
        self._metadata: dict[str, dict[str, str]] = {}

    def register(self, name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
        def decorator(fn: Callable[P, R]) -> Callable[P, R]:
            self._actions[name] = fn
            self._metadata[name] = {
                "docstring": inspect.getdoc(fn) or "",
                "signature": str(inspect.signature(fn)),
            }
            return fn

        return decorator

    def get(self, name: str) -> Callable[..., Any] | None:
        return self._actions.get(name)

    def call(self, name: str, params: dict[str, Any]) -> Any:
        action_fn = self._actions.get(name)
        if not action_fn:
            raise ActionNotFoundError(f"Action not found: {name}")
        return action_fn(**params)

    def exists(self, name: str) -> bool:
        return name in self._actions

    def get_all(self) -> dict[str, Callable[..., Any]]:
        return dict(self._actions)

    def get_metadata(self, name: str) -> dict[str, str] | None:
        return self._metadata.get(name)

    def discover(self, path: Path) -> None:
        """Load @action decorated functions from .theow/actions/*.py."""
        if not path.exists():
            return

        for py_file in path.glob("*.py"):
            self._load_action_file(py_file)

    def _load_action_file(self, path: Path) -> None:
        try:
            source = path.read_text()
            tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    for decorator in node.decorator_list:
                        if self._is_action_decorator(decorator):
                            action_name = self._get_action_name(decorator)
                            if action_name:
                                namespace: dict[str, Any] = {"action": action}
                                exec(compile(tree, path, "exec"), namespace)
                                if action_name in self._actions:
                                    logger.debug(
                                        "Discovered action", action=action_name, path=str(path)
                                    )
        except Exception as e:
            logger.warning("Failed to load actions", path=str(path), error=str(e))

    def _is_action_decorator(self, node: ast.expr) -> bool:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return node.func.id == "action"
        return False

    def _get_action_name(self, node: ast.expr) -> str | None:
        if isinstance(node, ast.Call) and node.args:
            if isinstance(node.args[0], ast.Constant):
                return str(node.args[0].value)
        return None


class ActionNotFoundError(Exception):
    """Action referenced by rule not found in registry."""


_standalone_registry: ActionRegistry | None = None


def action(name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Standalone decorator for .theow/actions/ files."""
    global _standalone_registry

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        if _standalone_registry is not None:
            _standalone_registry._actions[name] = fn
            _standalone_registry._metadata[name] = {
                "docstring": inspect.getdoc(fn) or "",
                "signature": str(inspect.signature(fn)),
            }
        return fn

    return decorator


def set_standalone_registry(registry: ActionRegistry) -> None:
    global _standalone_registry
    _standalone_registry = registry


@dataclass
class MarkConfig:
    """Configuration for @mark decorated function."""

    context_from: Callable[..., dict[str, Any]]
    max_retries: int
    rules: list[str] | None
    tags: list[str] | None
    fallback: bool
    explorable: bool
    collection: str


class MarkDecorator:
    """Creates @mark decorators bound to a Theow instance."""

    def __init__(
        self,
        resolver: Resolver,
        explorer: Explorer,
        tool_registry: ToolRegistry,
    ) -> None:
        self._resolver = resolver
        self._explorer = explorer
        self._tool_registry = tool_registry

    def __call__(
        self,
        context_from: Callable[..., dict[str, Any]],
        max_retries: int = 3,
        rules: list[str] | None = None,
        tags: list[str] | None = None,
        fallback: bool = True,
        explorable: bool = False,
        collection: str = "default",
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        config = MarkConfig(
            context_from=context_from,
            max_retries=max_retries,
            rules=rules,
            tags=tags,
            fallback=fallback,
            explorable=explorable,
            collection=collection,
        )

        def decorator(fn: Callable[P, R]) -> Callable[P, R]:
            @functools.wraps(fn)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                return self._run_with_recovery(fn, args, kwargs, config)

            return wrapper

        return decorator

    def _run_with_recovery(
        self,
        fn: Callable[P, R],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        config: MarkConfig,
    ) -> R:
        """Run function with theow recovery.

        Steps:
            1. Try fn(), return if succeeds
            2. On failure, attempt recovery (find/create rule, run action)
            3. If recovery applied a rule, retry fn()
            4. If fn() succeeds after ephemeral rule, promote it
            5. If fn() fails after ephemeral rule, delete it and track rejection
            6. Repeat until max_retries or no recovery possible
            7. If exploration was attempted but failed, continue to next attempt
               (budget resets between attempts)

        Theow never blocks the consumer pipeline. If recovery fails or theow
        itself errors, the original exception is re-raised.
        """
        last_exception: Exception | None = None
        last_applied_rule: Rule | None = None
        rejected_attempts: list[dict[str, Any]] = []

        for attempt in range(config.max_retries + 1):
            try:
                result = fn(*args, **kwargs)

                # Success! If we applied an ephemeral rule, promote it
                if last_applied_rule and last_applied_rule.is_ephemeral:
                    self._promote_rule(last_applied_rule)

                return result

            except Exception as exc:
                last_exception = exc
                logger.debug("Attempt failed", attempt=attempt + 1, error=str(exc))

                # If we applied an ephemeral rule and it failed, reject it
                if last_applied_rule and last_applied_rule.is_ephemeral:
                    rejected_attempts.append(self._reject_rule(last_applied_rule, exc))
                    last_applied_rule = None

            if attempt >= config.max_retries:
                break

            # Theow recovery is best-effort. Internal errors are logged, not propagated.
            try:
                success, rule, explored = self._attempt_recovery(
                    fn,
                    args,
                    kwargs,
                    last_exception,
                    config,
                    rejected_attempts,
                    attempt_number=attempt + 1,
                )
                if not success:
                    # Only break if exploration wasn't attempted
                    # If explored, continue to next attempt (budget resets)
                    if not explored:
                        break
                last_applied_rule = rule
            except Exception as theow_err:
                logger.error("Theow internal error", error=str(theow_err))
                break

        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected state: no exception captured")

    def _attempt_recovery(
        self,
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        exc: Exception,
        config: MarkConfig,
        rejected_attempts: list[dict[str, Any]] | None = None,
        attempt_number: int = 1,
    ) -> tuple[bool, Rule | None, bool]:
        """Attempt recovery after fn() failure.

        Steps:
            1. Build context from exception using context_from
            2. Try resolver to find matching rule
            3. If rule found, execute action and return
            4. If explorable, run LLM exploration to create new rule
            5. If rule created, execute action and return

        Returns:
            (success, rule, explored) where:
            - success: indicates action executed
            - rule: the applied rule (for ephemeral tracking)
            - explored: True if exploration was attempted (even if no rule)
        """
        context = self._build_context(config.context_from, args, kwargs, exc)
        if context is None:
            return False, None, False

        rule = self._try_resolve(context, config)
        if rule:
            success = self._execute_rule(rule, context)
            return success, rule if success else None, False

        if self._should_explore(config):
            tracing = self._capture_tracing(exc)
            rule, explored = self._try_explore(
                context, tracing, config, rejected_attempts, attempt_number
            )
            if rule:
                success = self._execute_rule(rule, context)
                return success, rule if success else None, True
            return False, None, explored

        return False, None, False

    def _build_context(
        self,
        context_from: Callable[..., dict[str, Any]],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        exc: Exception,
    ) -> dict[str, Any] | None:
        try:
            return context_from(*args, exc, **kwargs)
        except Exception as err:
            logger.error("context_from failed", error=str(err))
            return None

    def _capture_tracing(self, exc: Exception) -> TracingInfo:
        return TracingInfo(
            traceback=traceback.format_exc(),
            exception_type=type(exc).__name__,
            exception_message=str(exc),
        )

    def _try_resolve(self, context: dict[str, Any], config: MarkConfig) -> Rule | None:
        rule = self._resolver.resolve(
            context=context,
            collection=config.collection,
            rules=config.rules,
            tags=config.tags,
            fallback=config.fallback,
        )
        if rule:
            logger.info("Resolved rule", rule=rule.name)
        return rule

    def _should_explore(self, config: MarkConfig) -> bool:
        return config.explorable and os.environ.get("THEOW_EXPLORE") == "1"

    def _try_explore(
        self,
        context: dict[str, Any],
        tracing: TracingInfo,
        config: MarkConfig,
        rejected_attempts: list[dict[str, Any]] | None = None,
        attempt_number: int = 1,
    ) -> tuple[Rule | None, bool]:
        """Run LLM exploration to create a new rule.

        Steps:
            1. Gather registered tools for LLM
            2. Call explorer with context and tracing
            3. Pass rejected_attempts so LLM avoids repeating failures
            4. Explorer writes ephemeral rule (no validation yet)
            5. Return rule for decorator to execute and validate

        Returns:
            (rule, explored) tuple where explored=True if exploration was attempted.
        """
        logger.info("Entering exploration mode", collection=config.collection)
        tools = list(self._tool_registry.get_all().values())

        rule, explored = self._explorer.explore(
            context=context,
            tools=tools,
            collection=config.collection,
            tracing=tracing,
            rejected_attempts=rejected_attempts,
            attempt_number=attempt_number,
        )

        if rule:
            logger.info("Explored rule", rule=rule.name, ephemeral=rule.is_ephemeral)
        return rule, explored

    def _execute_rule(self, rule: Rule, context: dict[str, Any] | None = None) -> bool:
        """Execute rule's action (deterministic) or run LLM (probabilistic)."""
        try:
            if rule.type == "probabilistic":
                return self._execute_probabilistic_rule(rule, context or {})
            else:
                rule.act()
                return True
        except Exception as err:
            logger.warning("Action failed", rule=rule.name, error=str(err))
            return False

    def _execute_probabilistic_rule(self, rule: Rule, context: dict[str, Any]) -> bool:
        """Execute a probabilistic rule by running LLM with the configured prompt."""
        if not rule.llm_config:
            logger.error("Probabilistic rule missing llm_config", rule=rule.name)
            return False

        # Get prompt with placeholders replaced
        rules_dir = self._explorer._rules_dir
        prompt = rule.llm_config.get_prompt(
            base_path=rules_dir.parent,
            context=context,
        )

        # Resolve configured tools from registry
        tools = []
        for tool_name in rule.llm_config.tools:
            tool_fn = self._tool_registry.get(tool_name)
            if tool_fn:
                tools.append(tool_fn)
            else:
                logger.warning("Tool not found for probabilistic rule", tool=tool_name)

        budget = {
            "max_tool_calls": rule.llm_config.constraints.get("max_tool_calls", 10),
            "max_tokens": rule.llm_config.constraints.get("max_tokens", 4096),
        }

        logger.info("Executing probabilistic rule", rule=rule.name)
        return self._explorer.run_direct(prompt, tools, budget)

    def _promote_rule(self, rule: Rule) -> None:
        """Promote ephemeral rule to permanent after fn() succeeds."""
        logger.info("Promoting rule", rule=rule.name)

        # Remove incomplete tag if present
        if "incomplete" in rule.tags:
            rule.tags.remove("incomplete")

        # Move from ephemeral/ to main rules folder
        if rule._source_path and "ephemeral" in str(rule._source_path):
            new_path = rule._source_path.parent.parent / rule._source_path.name
            rule._source_path.rename(new_path)
            rule._source_path = new_path

        if rule._source_path and rule._source_path.exists():
            rule.to_yaml(rule._source_path)

        self._explorer._chroma.index_rule(rule)

    def _reject_rule(self, rule: Rule, exc: Exception) -> dict[str, Any]:
        """Reject ephemeral rule after fn() fails. Returns info for LLM feedback."""
        logger.warning("Rejecting ephemeral rule", rule=rule.name, error=str(exc))

        # Include rule details so LLM knows what was tried without needing to read deleted files
        rejection = {
            "rule_name": rule.name,
            "description": rule.description[:200] if rule.description else "",
            "error": str(exc)[:500],
        }

        # Invalidate session cache so we don't return this rule again
        if self._explorer._session_cache:
            self._explorer._session_cache.invalidate(rule.name)

        # Don't delete incomplete rules - keep for next attempt
        if "incomplete" in rule.tags:
            logger.info("Keeping incomplete rule for next attempt", rule=rule.name)
            return rejection

        for path in rule._created_files:
            if path.exists():
                path.unlink()
                logger.debug("Deleted file", path=str(path))

        return rejection
