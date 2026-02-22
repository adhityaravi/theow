"""Tool, action, and mark decorators."""

from __future__ import annotations

import ast
import functools
import inspect
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ParamSpec, TypeVar

from theow._core._logging import get_logger
from theow._gateway._base import build_tool_declaration

if TYPE_CHECKING:
    from theow._core._explorer import Explorer
    from theow._core._models import Rule
    from theow._core._resolver import Resolver

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
        return [
            build_tool_declaration(name, fn, schema_key="parameters")
            for name, fn in self._tools.items()
        ]


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
_standalone_tool_registry: ToolRegistry | None = None


def action(name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Standalone decorator for .theow/actions/ files."""

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        if _standalone_registry is not None:
            _standalone_registry._actions[name] = fn
            _standalone_registry._metadata[name] = {
                "docstring": inspect.getdoc(fn) or "",
                "signature": str(inspect.signature(fn)),
            }
        return fn

    return decorator


def tool(name: str | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Standalone decorator for plugin files."""

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        if _standalone_tool_registry is not None:
            tool_name = name or getattr(fn, "__name__", str(id(fn)))
            _standalone_tool_registry._tools[tool_name] = fn
        return fn

    return decorator


def set_standalone_registry(registry: ActionRegistry) -> None:
    global _standalone_registry
    _standalone_registry = registry


def set_standalone_tool_registry(registry: ToolRegistry | None) -> None:
    global _standalone_tool_registry
    _standalone_tool_registry = registry


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
    hint: str | None = None
    setup: Callable[[dict[str, Any], int], dict[str, Any] | None] | None = None
    teardown: Callable[[dict[str, Any], int, bool], None] | None = None


class MarkDecorator:
    """Creates @mark decorators bound to a Theow instance."""

    def __init__(
        self,
        resolver: Resolver,
        explorer: Explorer,
        tool_registry: ToolRegistry,
        engine: Any = None,
    ) -> None:
        self._resolver = resolver
        self._explorer = explorer
        self._tool_registry = tool_registry
        self._engine = engine

    def __call__(
        self,
        context_from: Callable[..., dict[str, Any]],
        max_retries: int = 3,
        rules: list[str] | None = None,
        tags: list[str] | None = None,
        fallback: bool = True,
        explorable: bool = False,
        collection: str = "default",
        hint: str | None = None,
        setup: Callable[[dict[str, Any], int], dict[str, Any] | None] | None = None,
        teardown: Callable[[dict[str, Any], int, bool], None] | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        config = MarkConfig(
            context_from=context_from,
            max_retries=max_retries,
            rules=rules,
            tags=tags,
            fallback=fallback,
            explorable=explorable,
            collection=collection,
            hint=hint,
            setup=setup,
            teardown=teardown,
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

        Delegates to the shared recover() loop. See recover() for the full
        step-by-step flow. This method adapts the Python function call pattern
        (try/except) into the Attempt-based interface that recover() expects.

        Theow never blocks the consumer pipeline. If recovery fails or theow
        itself errors, the original exception is re-raised.
        """
        from theow._core._recover import Attempt, RecoveryConfig, recover

        last_exc: Exception | None = None

        def run() -> Attempt[R]:
            nonlocal last_exc
            try:
                result = fn(*args, **kwargs)
                return Attempt(success=True, value=result)
            except Exception as e:
                last_exc = e
                ctx = self._build_context(config.context_from, args, kwargs, e)
                tracing = self._capture_tracing(e)
                return Attempt(success=False, context=ctx or {}, tracing=tracing)

        recovery_config = RecoveryConfig(
            max_retries=config.max_retries,
            rules=config.rules,
            tags=config.tags,
            collection=config.collection,
            fallback=config.fallback,
            explorable=config.explorable,
            hint=config.hint,
        )

        # Wrap setup/teardown hooks to maintain hook_state
        hook_state: dict[str, Any] = (
            self._build_hook_state(fn, args, kwargs) if (config.setup or config.teardown) else {}
        )

        def before_attempt(state: dict[str, Any], attempt: int) -> dict[str, Any] | None:
            nonlocal hook_state
            if config.setup:
                hook_state = config.setup(hook_state, attempt) or hook_state
            return state

        def after_attempt(state: dict[str, Any], attempt: int, success: bool) -> None:
            if config.teardown:
                # Flow recovery metadata into the decorator's hook state
                hook_state.update(
                    {k: state[k] for k in ("_give_up_reason", "_observation") if k in state}
                )
                hook_state["_fn_name"] = getattr(fn, "__name__", "")
                config.teardown(hook_state, attempt, success)
                # Copy enriched observation back for recover() to flush
                obs = hook_state.pop("_observation", None)
                if obs:
                    state["_observation"] = obs

        attempt = recover(
            run,
            self._engine,
            recovery_config,
            before_attempt=before_attempt if config.setup else None,
            after_attempt=after_attempt if config.teardown else None,
        )

        if attempt.success:
            return attempt.value  # type: ignore[return-value]
        raise last_exc  # type: ignore[misc]

    def _build_hook_state(
        self,
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Pre-populate hook state with the function's named arguments."""
        sig = inspect.signature(fn)
        state = dict(zip(sig.parameters.keys(), args))
        state.update(kwargs)
        return state

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

    def _execute_rule(self, rule: Rule, context: dict[str, Any] | None = None) -> bool:
        """Execute rule's action (deterministic) or run LLM (probabilistic)."""
        logger.info("Attempting recovery", rule=rule.name)
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
            logger.error("Bad rule config", rule=rule.name, error="missing llm_config")
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
                logger.warning("Tool not found", tool=tool_name)

        budget = {
            "max_tool_calls_per_session": rule.llm_config.constraints.get("max_tool_calls", 10),
            "max_tokens_per_session": rule.llm_config.constraints.get("max_tokens", 4096),
        }

        logger.debug("Executing LLM action", rule=rule.name)
        return self._explorer.run_direct(prompt, tools, budget)
