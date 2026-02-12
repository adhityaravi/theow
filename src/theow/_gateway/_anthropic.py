"""Anthropic LLM gateway implementation.

Anthropic gateway currently follows a Theow native approach for conversation.
Anthropic's models act as pure brains, with Theow being the hands managing the conversation flow and tool execution.
This lets Theow have a leash on the model's behavior and ensures consistent handling of tool calls and budget across providers.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

import anthropic

from theow._core._logging import get_engine_name, get_logger
from theow._core._tools import ExplorationSignal
from theow._gateway._base import (
    ConversationResult,
    LLMGateway,
    SessionState,
    build_tool_declaration,
)

logger = get_logger(__name__)


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
        max_calls, max_tokens = self._extract_budget(budget)
        tool_map = self._build_tool_map(tools)
        declarations = self._build_tool_declarations(tools)

        state = SessionState()

        while state.tool_calls < max_calls:
            response = self._call_model(messages, declarations, max_tokens, state)
            if response is None:
                break

            tool_calls = self._extract_tool_calls(response, messages)
            if not tool_calls:
                break

            # Execute tools - may raise ExplorationSignal
            self._execute_tool_calls(tool_calls, tool_map, messages, state)

            # Check for budget warning after tool execution
            warning = self.check_budget_warning(state, max_calls)
            if warning:
                messages.append({"role": "user", "content": warning})

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
        state: SessionState,
    ) -> anthropic.types.Message | None:
        """Send conversation to Claude, update token count."""
        logger.debug(
            f"{get_engine_name()} --> LLM",
            provider="anthropic",
            turn=state.tool_calls,
            model=self._model,
        )
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
            f"{get_engine_name()} <-- LLM",
            tools=tool_names or ["text"],
            tokens=response.usage.output_tokens,
        )
        return response

    def _extract_tool_calls(
        self,
        response: anthropic.types.Message,
        messages: list[dict[str, Any]],
    ) -> list[Any]:
        """Extract tool_use blocks from response, append to history."""
        messages.append({"role": "assistant", "content": response.content})
        return [b for b in response.content if b.type == "tool_use"]

    def _execute_tool_calls(
        self,
        tool_calls: list[Any],
        tool_map: dict[str, Callable[..., Any]],
        messages: list[dict[str, Any]],
        state: SessionState,
    ) -> None:
        """Execute tools, append all results in single user message."""
        tool_results: list[dict[str, Any]] = []
        signal_to_raise: ExplorationSignal | None = None

        for block in tool_calls:
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
                result = self._format_tool_result(block.name, block.input, block.id, tool_map)
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

    def _format_tool_result(
        self,
        name: str,
        input_args: Any,
        tool_use_id: str,
        tool_map: dict[str, Callable[..., Any]],
    ) -> dict[str, Any]:
        """Execute tool and format result for Anthropic API."""
        args = input_args if isinstance(input_args, dict) else {}
        result, is_error = self._execute_tool(name, args, tool_map)

        if is_error:
            content = json.dumps(result) if isinstance(result, dict) else str(result)
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": content,
                "is_error": True,
            }

        content = json.dumps(result) if not isinstance(result, str) else result
        return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}

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

    def _build_tool_declarations(self, tools: list[Callable[..., Any]]) -> list[dict[str, Any]]:
        """Convert Python callables to Anthropic tool declarations."""
        return [
            build_tool_declaration(
                name=getattr(fn, "__name__", str(id(fn))),
                fn=fn,
                schema_key="input_schema",
            )
            for fn in tools
        ]
