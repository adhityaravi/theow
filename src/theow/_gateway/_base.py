"""LLM gateway base class."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from theow._core._logging import get_logger
from theow._core._tools import ExplorationSignal

logger = get_logger(__name__)

# Type mapping for JSON schema generation
TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def build_tool_declaration(
    name: str,
    fn: Callable[..., Any],
    schema_key: str = "input_schema",
) -> dict[str, Any]:
    """Convert a Python callable to a JSON schema tool declaration.

    Args:
        name: Tool name to use in the declaration.
        fn: The callable to introspect.
        schema_key: Key for the schema dict ("input_schema" for Anthropic, "parameters" for others).

    Returns:
        Tool declaration dict with name, description, and schema.
    """
    sig = inspect.signature(fn)
    doc = inspect.getdoc(fn) or ""

    properties: dict[str, dict[str, str]] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        if param.annotation != inspect.Parameter.empty:
            origin = getattr(param.annotation, "__origin__", param.annotation)
            param_type = TYPE_MAP.get(origin, "string")
        else:
            param_type = "string"

        properties[param_name] = {"type": param_type}

        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    return {
        "name": name,
        "description": doc,
        schema_key: {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


# Budget configuration
SOFT_LIMIT_RATIO = 0.8  # Warn at 80% of budget


@dataclass
class ConversationResult:
    """Result from a conversation session."""

    messages: list[dict[str, Any]]
    tool_calls: int = 0
    tokens_used: int = 0


@dataclass
class SessionState:
    """Tracks mutable state across the tool-calling loop."""

    tool_calls: int = 0
    tokens_used: int = 0
    warned_about_budget: bool = False


class LLMGateway(ABC):
    """Abstract base for LLM provider implementations."""

    @abstractmethod
    def conversation(
        self,
        messages: list[dict[str, Any]],
        tools: list[Callable[..., Any]],
        budget: dict[str, Any],
    ) -> ConversationResult:
        """Run a conversation with tool use.

        Messages list is modified in-place as conversation progresses.
        Raises ExplorationSignal subclasses when LLM calls signal tools.
        """
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Single generation, optionally with structured output."""
        ...

    def reset(self) -> None:
        """Reset gateway state after conversation ends.

        Override in stateful gateways (e.g., Copilot) to clean up sessions.
        Stateless gateways (Anthropic, Gemini) can use this default noop.
        """
        pass

    def set_gateway_config(self, config: dict[str, Any]) -> None:
        """Set gateway-specific config. Only applies if gateway has _gateway_config."""
        gateway_config = getattr(self, "_gateway_config", None)
        if isinstance(gateway_config, dict):
            gateway_config.update(config)

    def check_budget_warning(
        self,
        state: SessionState,
        max_calls: int,
    ) -> str | None:
        """Check if budget warning should be issued.

        Returns warning message if at soft limit, None otherwise.
        Sets state.warned_about_budget to prevent duplicate warnings.
        """
        if state.warned_about_budget:
            return None

        soft_limit = int(max_calls * SOFT_LIMIT_RATIO)
        if state.tool_calls < soft_limit:
            return None

        remaining = max_calls - state.tool_calls
        state.warned_about_budget = True
        logger.debug("Budget warning triggered", remaining=remaining, max_calls=max_calls)

        return (
            f"NOTE: {remaining} tool calls remaining. Start wrapping up, but don't rush.\n\n"
            f"Quality matters more than speed. Write a GENERIC solution that works for "
            f"similar errors, not just this specific case. Avoid hardcoding package names or paths.\n\n"
            f"If you have a fix: request_templates() → write_rule/action → test_rule_match → submit_rule.\n\n"
            f"If you can't write a proper generic solution: use tags: [incomplete] and add notes. "
            f"The next attempt will continue from there."
        )

    def _build_tool_map(self, tools: list[Callable[..., Any]]) -> dict[str, Callable[..., Any]]:
        """Create name → function mapping from tools list."""
        return {getattr(fn, "__name__", str(id(fn))): fn for fn in tools}

    def _extract_budget(self, budget: dict[str, Any]) -> tuple[int, int]:
        """Extract budget params with defaults.

        Returns:
            (max_tool_calls, max_tokens) tuple.
        """
        max_calls = budget.get("max_tool_calls", 30)
        max_tokens = budget.get("max_tokens", 8192)
        return max_calls, max_tokens

    def _execute_tool(
        self,
        name: str,
        args: dict[str, Any],
        tool_map: dict[str, Callable[..., Any]],
    ) -> tuple[Any, bool]:
        """Execute a single tool by name.

        Args:
            name: Tool function name.
            args: Arguments to pass to the tool.
            tool_map: Mapping of tool names to functions.

        Returns:
            (result, is_error) tuple.

        Raises:
            ExplorationSignal: If the tool raises a signal (Done, GiveUp, etc.).
        """
        fn = tool_map.get(name)
        if not fn:
            return {"error": f"Unknown tool: {name}"}, True

        try:
            result = fn(**args)
            return result, False
        except ExplorationSignal:
            raise
        except Exception as e:
            logger.warning("Tool execution failed", tool=name, error=str(e))
            return {"error": str(e)}, True
