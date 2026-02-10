"""Anthropic LLM gateway implementation."""

from __future__ import annotations

import inspect
import json
import os
from dataclasses import dataclass
from typing import Any, Callable

import anthropic

from theow._core._logging import get_logger
from theow._core._tools import ExplorationSignal
from theow._gateway._base import ConversationResult, LLMGateway

logger = get_logger(__name__)


@dataclass
class _SessionState:
    """Tracks mutable state across the tool-calling loop."""

    tool_calls: int = 0
    tokens_used: int = 0


class AnthropicGateway(LLMGateway):
    """Anthropic implementation using anthropic SDK."""

    def __init__(self, model: str = "claude-sonnet-4-20250514") -> None:
        self._model = model
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self._client = anthropic.Anthropic(api_key=api_key)

    def conversation(
        self,
        messages: list[dict[str, Any]],
        tools: list[Callable[..., Any]],
        budget: dict[str, Any],
    ) -> ConversationResult:
        """Run conversation with tool use until signal or budget exhausted.

        Messages list is modified in-place.
        Raises ExplorationSignal subclasses when LLM calls signal tools.
        """
        max_calls = budget.get("max_tool_calls", 30)
        max_tokens = budget.get("max_tokens", 8192)
        tool_map = {getattr(fn, "__name__", str(id(fn))): fn for fn in tools}
        declarations = self._build_declarations(tools)

        state = _SessionState()

        while state.tool_calls < max_calls:
            response = self._call_model(messages, declarations, max_tokens, state)
            if response is None:
                break

            tool_uses = self._extract_tool_uses(response, messages)
            if not tool_uses:
                break

            # Execute tools - may raise ExplorationSignal
            self._execute_tools(tool_uses, tool_map, messages, state)

        return ConversationResult(
            messages=messages,
            tool_calls=state.tool_calls,
            tokens_used=state.tokens_used,
        )

    def _call_model(
        self,
        messages: list[dict[str, Any]],
        declarations: list[dict[str, Any]],
        max_tokens: int,
        state: _SessionState,
    ) -> anthropic.types.Message | None:
        """Send conversation to Claude, update token count."""
        logger.debug("Theow --> LLM", turn=state.tool_calls, model=self._model)
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=messages,  # type: ignore[arg-type]
                tools=declarations,  # type: ignore[arg-type]
            )
        except Exception as e:
            logger.error("Anthropic API error", error=str(e))
            raise

        state.tokens_used += response.usage.input_tokens + response.usage.output_tokens
        tool_names = [b.name for b in response.content if hasattr(b, "name")]
        logger.debug(
            "Theow <-- LLM", tools=tool_names or ["text"], tokens=response.usage.output_tokens
        )
        return response

    def _extract_tool_uses(
        self,
        response: anthropic.types.Message,
        messages: list[dict[str, Any]],
    ) -> list[Any]:
        """Extract tool_use blocks from response, append to history."""
        messages.append({"role": "assistant", "content": response.content})
        return [b for b in response.content if b.type == "tool_use"]

    def _execute_tools(
        self,
        tool_uses: list[Any],
        tool_map: dict[str, Callable[..., Any]],
        messages: list[dict[str, Any]],
        state: _SessionState,
    ) -> None:
        """Execute tools, append all results in single user message."""
        tool_results: list[dict[str, Any]] = []
        signal_to_raise: ExplorationSignal | None = None

        for block in tool_uses:
            state.tool_calls += 1
            if signal_to_raise:
                # Already got a signal, add placeholder for remaining tools
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Skipped - exploration signal received",
                    }
                )
                continue

            try:
                result = self._execute_single(block.name, block.input, block.id, tool_map)
                tool_results.append(result)
            except ExplorationSignal as sig:
                # Capture signal, add result for this tool, continue to collect all
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Signal: {type(sig).__name__}",
                    }
                )
                signal_to_raise = sig

        messages.append({"role": "user", "content": tool_results})

        if signal_to_raise:
            raise signal_to_raise

    def _execute_single(
        self,
        name: str,
        input_args: Any,
        tool_use_id: str,
        tool_map: dict[str, Callable[..., Any]],
    ) -> dict[str, Any]:
        """Execute one tool. ExplorationSignal propagates up."""
        fn = tool_map.get(name)
        if not fn:
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": json.dumps({"error": f"Unknown tool: {name}"}),
                "is_error": True,
            }

        args = input_args if isinstance(input_args, dict) else {}

        try:
            result = fn(**args)
            content = json.dumps(result) if not isinstance(result, str) else result
            return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
        except ExplorationSignal:
            # Let signals propagate - explorer handles them
            raise
        except Exception as e:
            logger.warning("Tool failed", tool=name, error=str(e))
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": str(e),
                "is_error": True,
            }

    def generate(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Single generation, optionally with structured output via tool forcing."""
        messages = [{"role": "user", "content": prompt}]

        if schema:
            tool = {
                "name": "structured_output",
                "description": "Return structured data",
                "input_schema": schema,
            }
            response = self._client.messages.create(
                model=self._model,
                max_tokens=4096,
                messages=messages,  # type: ignore[arg-type]
                tools=[tool],  # type: ignore[arg-type]
                tool_choice={"type": "tool", "name": "structured_output"},
            )
            for block in response.content:
                if block.type == "tool_use":
                    return block.input if isinstance(block.input, dict) else {}
            return {}

        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            messages=messages,  # type: ignore[arg-type]
        )

        for block in response.content:
            if hasattr(block, "text"):
                try:
                    return json.loads(block.text)
                except json.JSONDecodeError:
                    return {"text": block.text}

        return {}

    def _build_declarations(self, tools: list[Callable[..., Any]]) -> list[dict[str, Any]]:
        """Convert Python callables to Anthropic tool declarations."""
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        declarations = []
        for fn in tools:
            sig = inspect.signature(fn)
            doc = inspect.getdoc(fn) or ""

            properties: dict[str, Any] = {}
            required: list[str] = []

            for name, param in sig.parameters.items():
                if name in ("self", "cls"):
                    continue

                if param.annotation != inspect.Parameter.empty:
                    origin = getattr(param.annotation, "__origin__", param.annotation)
                    param_type = type_map.get(origin, "string")
                else:
                    param_type = "string"

                properties[name] = {"type": param_type}

                if param.default == inspect.Parameter.empty:
                    required.append(name)

            declarations.append(
                {
                    "name": getattr(fn, "__name__", str(id(fn))),
                    "description": doc,
                    "input_schema": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                }
            )

        return declarations
