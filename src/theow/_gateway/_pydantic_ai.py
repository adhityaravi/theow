"""PydanticAI LLM gateway implementation.

Uses pydantic_ai.direct.model_request_sync() for unified access to 15+ providers.
Theow manages the conversation loop (signals, budget warnings, nudging) while
PydanticAI handles provider-specific API translation.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from pydantic_ai import ModelRequest, ToolDefinition
from pydantic_ai.direct import model_request_sync
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings

from theow._core._logging import get_engine_name, get_logger
from theow._core._tools import ExplorationSignal, GiveUp
from theow._gateway._base import (
    MAX_TEXT_NUDGES,
    TEXT_REPLY_NUDGE,
    ConversationResult,
    LLMGateway,
    build_tool_declaration,
)
from theow._gateway._observability import SessionState, instrumented, span_tool_call

logger = get_logger(__name__)

# Per-response output cap, matching Anthropic gateway's default.
_PER_RESPONSE_TOKENS = 8192


class PydanticAIGateway(LLMGateway):
    """Unified gateway using pydantic_ai.direct for any supported provider.

    The model spec uses PydanticAI format: ``"provider:model"``
    (e.g. ``"anthropic:claude-opus-4-6"``, ``"openai:gpt-5"``).

    No API key handling — PydanticAI reads env vars natively per provider.
    """

    def __init__(self, model: str = "anthropic:claude-sonnet-4-20250514") -> None:
        self._model = model
        self._state: SessionState | None = None

    @property
    def provider_name(self) -> str:
        """Extract actual provider from PydanticAI model spec (e.g. 'anthropic:model' → 'anthropic')."""
        return self._model.split(":", 1)[0] if ":" in self._model else self.gateway_name

    @instrumented
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
        allow_escalation = budget.get("allow_escalation", False)
        tool_map = self._build_tool_map(tools)
        tool_definitions = self._build_tool_definitions(tools)

        request_params = ModelRequestParameters(
            function_tools=tool_definitions,
            allow_text_output=True,
        )
        model_settings = ModelSettings(max_tokens=_PER_RESPONSE_TOKENS)

        self._state = state = SessionState()
        text_nudges = 0

        # Build PydanticAI message history from theow messages
        pai_messages: list[ModelMessage] = self._to_pai_messages(messages)

        while state.tool_calls < max_calls and not (
            max_tokens > 0 and state.tokens_used > max_tokens
        ):
            response = self._call_model(pai_messages, model_settings, request_params, state)
            if response is None:
                break

            # Hard cutoff right after model response updates token count
            if self.check_budget_exceeded(state, max_calls, max_tokens):
                pai_messages.append(response)
                break

            # Append model response to PydanticAI history
            pai_messages.append(response)

            # Extract tool calls
            tool_calls = response.tool_calls
            if not tool_calls:
                if text_nudges < MAX_TEXT_NUDGES:
                    text_nudges += 1
                    logger.debug("Text-only reply, nudging LLM", nudge=text_nudges)
                    pai_messages.append(
                        ModelRequest(parts=[UserPromptPart(content=TEXT_REPLY_NUDGE)])
                    )
                    continue
                break

            # Execute tools — may raise ExplorationSignal
            tool_return_msg = self._execute_tool_calls(tool_calls, tool_map, state)
            pai_messages.append(tool_return_msg)

            # Check for budget warning after tool execution
            warning = self.check_budget_warning(state, max_calls, max_tokens, allow_escalation)
            if warning:
                pai_messages.append(ModelRequest(parts=[UserPromptPart(content=warning)]))

        # Sync back to theow message format
        self._sync_messages(messages, pai_messages)

        # Budget exceeded → raise signal so @instrumented records the exception
        # and the explorer gets an explicit GiveUp instead of None
        if state.tool_calls > max_calls or (max_tokens > 0 and state.tokens_used > max_tokens):
            raise GiveUp("Session budget exceeded")

        return ConversationResult(
            messages=messages,
            tool_calls=state.tool_calls,
            tokens_used=state.tokens_used,
        )

    def _call_model(
        self,
        pai_messages: list[ModelMessage],
        model_settings: ModelSettings,
        request_params: ModelRequestParameters,
        state: SessionState,
    ) -> ModelResponse | None:
        """Send conversation to model via PydanticAI, update token count."""
        logger.debug(
            f"{get_engine_name()} --> LLM",
            provider=self._model.split(":")[0],
            turn=state.tool_calls,
            model=self._model,
        )
        try:
            response = model_request_sync(
                self._model,
                pai_messages,
                model_settings=model_settings,
                model_request_parameters=request_params,
            )
        except Exception as e:
            logger.error("PydanticAI API error", error=str(e))
            raise

        # Track token usage
        if response.usage:
            inp = response.usage.request_tokens or 0
            out = response.usage.response_tokens or 0
            state.input_tokens += inp
            state.output_tokens += out
            state.tokens_used = state.input_tokens + state.output_tokens

        tool_names = [tc.tool_name for tc in response.tool_calls]
        output_tokens = response.usage.response_tokens if response.usage else 0
        logger.debug(
            f"{get_engine_name()} <-- LLM",
            tools=tool_names or ["text"],
            tokens=output_tokens,
        )
        return response

    def _execute_tool_calls(
        self,
        tool_calls: list[ToolCallPart],
        tool_map: dict[str, Callable[..., Any]],
        state: SessionState,
    ) -> ModelRequest:
        """Execute tools and return a ModelRequest with ToolReturnPart entries."""
        tool_returns: list[ToolReturnPart] = []
        signal_to_raise: ExplorationSignal | None = None

        for tc in tool_calls:
            state.tool_calls += 1

            if signal_to_raise:
                tool_returns.append(
                    ToolReturnPart(
                        tool_name=tc.tool_name,
                        content="Skipped - exploration signal received",
                        tool_call_id=tc.tool_call_id,
                    )
                )
                continue

            args = tc.args if isinstance(tc.args, dict) else {}
            with span_tool_call(tc.tool_name, state):
                try:
                    result, is_error = self._execute_tool(tc.tool_name, args, tool_map)
                    content = json.dumps(result) if not isinstance(result, str) else result
                    if is_error:
                        content = json.dumps(result) if isinstance(result, dict) else str(result)
                    tool_returns.append(
                        ToolReturnPart(
                            tool_name=tc.tool_name,
                            content=content,
                            tool_call_id=tc.tool_call_id,
                        )
                    )
                except ExplorationSignal as sig:
                    tool_returns.append(
                        ToolReturnPart(
                            tool_name=tc.tool_name,
                            content=f"Signal: {type(sig).__name__}",
                            tool_call_id=tc.tool_call_id,
                        )
                    )
                    signal_to_raise = sig

        tool_return_msg = ModelRequest(parts=tool_returns)

        if signal_to_raise:
            raise signal_to_raise

        return tool_return_msg

    @instrumented
    def generate(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Single generation, optionally with structured output via tool forcing."""
        pai_messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content=prompt)])]

        if schema:
            tool_def = ToolDefinition(
                name="structured_output",
                description="Return structured data",
                parameters_json_schema=schema,
            )
            request_params = ModelRequestParameters(
                function_tools=[tool_def],
                allow_text_output=False,
            )
            response = model_request_sync(
                self._model,
                pai_messages,
                model_settings=ModelSettings(max_tokens=4096),
                model_request_parameters=request_params,
            )
            for part in response.parts:
                if isinstance(part, ToolCallPart):
                    args = part.args
                    return args if isinstance(args, dict) else {}
            return {}

        response = model_request_sync(
            self._model,
            pai_messages,
            model_settings=ModelSettings(max_tokens=4096),
        )

        for part in response.parts:
            if isinstance(part, TextPart):
                try:
                    return json.loads(part.content)
                except json.JSONDecodeError:
                    return {"text": part.content}

        return {}

    def _build_tool_definitions(self, tools: list[Callable[..., Any]]) -> list[ToolDefinition]:
        """Convert Python callables to PydanticAI ToolDefinition objects."""
        definitions = []
        for fn in tools:
            name = getattr(fn, "__name__", str(id(fn)))
            decl = build_tool_declaration(name, fn, schema_key="parameters_json_schema")
            definitions.append(
                ToolDefinition(
                    name=decl["name"],
                    description=decl["description"],
                    parameters_json_schema=decl["parameters_json_schema"],
                )
            )
        return definitions

    def _to_pai_messages(self, messages: list[dict[str, Any]]) -> list[ModelMessage]:
        """Convert theow dict messages to PydanticAI message types."""
        pai_messages: list[ModelMessage] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "user" and isinstance(content, str):
                pai_messages.append(ModelRequest(parts=[UserPromptPart(content=content)]))
        return pai_messages

    def _sync_messages(
        self,
        messages: list[dict[str, Any]],
        pai_messages: list[ModelMessage],
    ) -> None:
        """Sync PydanticAI messages back to theow dict format."""
        messages.clear()
        for msg in pai_messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, UserPromptPart):
                        messages.append({"role": "user", "content": part.content})
            elif isinstance(msg, ModelResponse):
                text_parts = [p.content for p in msg.parts if isinstance(p, TextPart)]
                if text_parts:
                    messages.append({"role": "assistant", "content": " ".join(text_parts)})
