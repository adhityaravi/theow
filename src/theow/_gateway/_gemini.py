"""Gemini LLM gateway with manual history management."""

from __future__ import annotations

import json
import os
from typing import Any, Callable

from google import genai
from google.genai import types

from theow._core._logging import get_logger
from theow._core._tools import ExplorationSignal
from theow._gateway._base import ConversationResult, LLMGateway, SessionState

logger = get_logger(__name__)


class GeminiGateway(LLMGateway):
    """Gemini implementation with manual history management.

    Uses generate_content() directly instead of Chat Sessions to properly
    handle function responses with role='tool'.
    """

    def __init__(self, model: str = "gemini-2.0-flash") -> None:
        self._model = model
        self._is_gemini3 = "gemini-3" in model
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        self._client = genai.Client(api_key=api_key)
        # Manual history management
        self._history: list[types.Content] = []

    def conversation(
        self,
        messages: list[dict[str, Any]],
        tools: list[Callable[..., Any]],
        budget: dict[str, Any],
    ) -> ConversationResult:
        """Run conversation with tool use until signal or budget exhausted.

        Uses manual history management with proper role='tool' for function responses.
        Messages list is modified in-place.
        Raises ExplorationSignal subclasses when LLM calls signal tools.
        """
        max_calls = budget.get("max_tool_calls", 30)
        tool_map = {getattr(fn, "__name__", str(id(fn))): fn for fn in tools}

        # Build config
        config_kwargs: dict[str, Any] = {
            "tools": tools,
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        if self._is_gemini3:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level="high"  # type: ignore[arg-type]
            )

        config = types.GenerateContentConfig(**config_kwargs)
        state = SessionState()

        # Initialize or continue history
        self._init_history(messages)

        # Get pending user message and add to history
        user_content = self._get_pending_user_content(messages)
        if not user_content:
            return ConversationResult(messages=messages, tool_calls=0, tokens_used=0)

        self._history.append(user_content)

        # Main loop
        while state.tool_calls < max_calls:
            response = self._call_model(config, state)
            if response is None:
                break

            # Add model response to history
            model_content = self._extract_model_content(response)
            if model_content:
                self._history.append(model_content)

            # Check for function calls
            function_calls = self._extract_function_calls(response)
            if not function_calls:
                break

            # Execute functions and add tool response to history
            tool_content = self._execute_functions(function_calls, tool_map, state)
            self._history.append(tool_content)

            # Check for budget warning after tool execution
            warning = self.check_budget_warning(state, max_calls)
            if warning:
                self._history.append(
                    types.Content(role="user", parts=[types.Part.from_text(text=warning)])
                )

        # Sync messages with history
        self._sync_messages(messages)

        return ConversationResult(
            messages=messages,
            tool_calls=state.tool_calls,
            tokens_used=state.tokens_used,
        )

    def _init_history(self, messages: list[dict[str, Any]]) -> None:
        """Initialize history from messages if starting fresh."""
        # If history exists and has content, we're continuing (after signal)
        if self._history:
            return

        # Build initial history from text messages (exclude last, we'll add it separately)
        for msg in messages[:-1]:
            content = msg.get("content")
            if isinstance(content, str):
                role = "user" if msg["role"] == "user" else "model"
                self._history.append(
                    types.Content(role=role, parts=[types.Part.from_text(text=content)])
                )

    def _get_pending_user_content(self, messages: list[dict[str, Any]]) -> types.Content | None:
        """Get the last user message as Content."""
        if not messages:
            return None

        last_msg = messages[-1]
        if last_msg.get("role") != "user":
            return None

        content = last_msg.get("content")
        if isinstance(content, str):
            return types.Content(role="user", parts=[types.Part.from_text(text=content)])

        return None

    def _call_model(
        self,
        config: types.GenerateContentConfig,
        state: SessionState,
    ) -> types.GenerateContentResponse | None:
        """Call model with current history."""
        logger.debug("Theow --> LLM", turn=state.tool_calls, model=self._model)
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=self._history,
                config=config,
            )
        except Exception as e:
            logger.error("Gemini API error", error=str(e))
            raise

        if response.usage_metadata:
            state.tokens_used += response.usage_metadata.total_token_count or 0

        # Log response
        tool_names = []
        if response.candidates and response.candidates[0].content:
            for p in response.candidates[0].content.parts or []:
                if p.function_call and p.function_call.name:
                    tool_names.append(p.function_call.name)

        # Log output tokens only (like Anthropic) for consistency
        output_tokens = (
            response.usage_metadata.candidates_token_count if response.usage_metadata else 0
        )
        logger.debug("Theow <-- LLM", tools=tool_names or ["text"], tokens=output_tokens)

        return response

    def _extract_model_content(
        self, response: types.GenerateContentResponse
    ) -> types.Content | None:
        """Extract model content from response, preserving thought signatures."""
        if not response.candidates:
            return None

        candidate = response.candidates[0]
        if not candidate.content:
            return None

        # Return the full content to preserve thought signatures
        return candidate.content

    def _extract_function_calls(self, response: types.GenerateContentResponse) -> list[types.Part]:
        """Extract function call parts from response."""
        if not response.candidates:
            return []

        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            return []

        return [p for p in candidate.content.parts if p.function_call]

    def _execute_functions(
        self,
        function_calls: list[types.Part],
        tool_map: dict[str, Callable[..., Any]],
        state: SessionState,
    ) -> types.Content:
        """Execute functions and return tool response Content with role='tool'."""
        tool_results: list[types.Part] = []
        signal_to_raise: ExplorationSignal | None = None

        for part in function_calls:
            call = part.function_call
            if call is None or call.name is None:
                continue

            state.tool_calls += 1
            name = call.name
            args = dict(call.args) if call.args else {}

            if signal_to_raise:
                tool_results.append(
                    types.Part.from_function_response(
                        name=name, response={"skipped": "signal received"}
                    )
                )
                continue

            try:
                result = self._execute_single(name, args, tool_map)
                tool_results.append(types.Part.from_function_response(name=name, response=result))
            except ExplorationSignal as sig:
                tool_results.append(
                    types.Part.from_function_response(
                        name=name, response={"signal": type(sig).__name__}
                    )
                )
                signal_to_raise = sig

        # Create tool response with role='tool' (critical for Gemini)
        tool_content = types.Content(role="tool", parts=tool_results)

        if signal_to_raise:
            # Add to history before raising so context is preserved
            self._history.append(tool_content)
            raise signal_to_raise

        return tool_content

    def _execute_single(
        self,
        name: str,
        args: dict[str, Any],
        tool_map: dict[str, Callable[..., Any]],
    ) -> dict[str, Any]:
        """Execute one function. Returns result dict."""
        fn = tool_map.get(name)
        if not fn:
            return {"error": f"Unknown tool: {name}"}

        try:
            result = fn(**args)
            return {"result": result}
        except ExplorationSignal:
            raise
        except Exception as e:
            logger.warning("Tool failed", tool=name, error=str(e))
            return {"error": str(e)}

    def _sync_messages(self, messages: list[dict[str, Any]]) -> None:
        """Sync messages list with history for text content."""
        messages.clear()
        for content in self._history:
            role = "user" if content.role == "user" else "assistant"
            if content.role == "tool":
                role = "assistant"  # Map tool responses to assistant for simple view
            if content.parts:
                text_parts = [p.text for p in content.parts if hasattr(p, "text") and p.text]
                if text_parts:
                    messages.append({"role": role, "content": " ".join(text_parts)})

    def reset_history(self) -> None:
        """Reset conversation history."""
        self._history = []

    def generate(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Single generation, optionally with JSON schema."""
        config = types.GenerateContentConfig()

        if schema:
            config.response_mime_type = "application/json"
            config.response_schema = schema

        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )

        if response.text:
            return json.loads(response.text)

        return {}
