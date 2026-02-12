"""GitHub Copilot LLM gateway implementation.

Copilot's SDK is async-first and manages tool execution via handlers itself.
This doesnt allow for a Theow native logic to control the tool calls. This also causes Theow to lose some visibility into the conversation state (e.g., tool calls, tokens used) and a clean leash on the model since the SDK abstracts away the agentic loop.
Currently the gateway is bridged to be run as sync calls.
Basically Theow triggers a SDK loop and waits for it to finish, letting it call the tools automatically. The SDK only raises termination signals.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
from pathlib import Path
from typing import Any, Callable

import copilot
from copilot import CopilotClient, Tool
from copilot.types import ToolInvocation, ToolResult

from theow._core._logging import get_engine_name, get_logger
from theow._core._prompts import TEMPLATES
from theow._core._tools import ExplorationSignal, RequestTemplates
from theow._gateway._base import ConversationResult, LLMGateway, SessionState

logger = get_logger(__name__)

# Workaround for github-copilot-sdk packaging bug on Linux
# The bundled binary lacks execute permission. LOL
_copilot_bin = Path(copilot.__file__).parent / "bin" / "copilot"
if _copilot_bin.exists() and not os.access(_copilot_bin, os.X_OK):
    _copilot_bin.chmod(_copilot_bin.stat().st_mode | stat.S_IXUSR)


class CopilotGateway(LLMGateway):
    """Copilot implementation using github-copilot-sdk.

    The SDK is async-first and manages tool execution via handlers.
    We bridge to sync context using asyncio.run().
    """

    def __init__(self, model: str = "claude-sonnet-4") -> None:
        self._model = model
        self._client: CopilotClient | None = None
        self._session: Any | None = None  # Copilot session, kept alive across turns
        self._state: SessionState | None = None
        self._tool_map: dict[str, Callable[..., Any]] = {}
        self._signal: ExplorationSignal | None = None
        self._gateway_config: dict[str, Any] = {}
        self._max_calls: int = 30
        # We use a persistent event loop instead of asyncio.run() because
        # multi-turn conversations (e.g., after RequestTemplates signal) require
        # multiple async calls. asyncio.run() closes the loop after each call,
        # causing loop closed errors on subsequent calls.
        self._loop: asyncio.AbstractEventLoop | None = None

    def conversation(
        self,
        messages: list[dict[str, Any]],
        tools: list[Callable[..., Any]],
        budget: dict[str, Any],
    ) -> ConversationResult:
        """Run conversation with tool use until signal or budget exhausted."""
        self._max_calls, _ = self._extract_budget(budget)
        self._tool_map = self._build_tool_map(tools)
        self._state = SessionState()
        self._signal = None

        user_prompt = self._get_user_prompt(messages)
        if not user_prompt:
            return ConversationResult(messages=messages, tool_calls=0, tokens_used=0)

        logger.debug(f"{get_engine_name()} --> LLM", provider="copilot", model=self._model)

        try:
            # Reuse event loop across conversation turns
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
            response = self._loop.run_until_complete(self._run_conversation(user_prompt))
        except Exception as e:
            logger.error("Copilot API error", error=str(e))
            raise

        if response:
            messages.append({"role": "assistant", "content": response})

        logger.debug(
            f"{get_engine_name()} <-- LLM",
            tools=["signal"] if self._signal else ["text"],
            tool_calls=self._state.tool_calls if self._state else 0,
        )

        if self._signal:
            raise self._signal

        return ConversationResult(
            messages=messages,
            tool_calls=self._state.tool_calls if self._state else 0,
            tokens_used=self._state.tokens_used if self._state else 0,
        )

    async def _run_conversation(self, prompt: str) -> str | None:
        """Run async conversation with Copilot.

        Session is kept alive across turns to maintain context (e.g., after
        RequestTemplates signal). Session is destroyed in reset().
        """
        if self._client is None:
            self._client = CopilotClient()

        # Create session only if we don't have one
        if self._session is None:
            copilot_tools = self._build_copilot_tools()
            self._session = await self._client.create_session(
                {
                    "model": self._model,
                    "tools": copilot_tools,
                }
            )

        response = await self._session.send_and_wait({"prompt": prompt}, timeout=900)
        if response and hasattr(response, "data") and hasattr(response.data, "content"):
            return response.data.content
        return None

    def _build_copilot_tools(self) -> list[Tool]:
        """Convert tool map to Copilot Tool objects with handlers."""
        copilot_tools = []

        for name, fn in self._tool_map.items():
            doc = fn.__doc__ or ""

            def make_handler(tool_fn: Callable[..., Any]) -> Callable[[ToolInvocation], ToolResult]:
                def handler(invocation: ToolInvocation) -> ToolResult:
                    return self._handle_tool_call(tool_fn, invocation)

                return handler

            copilot_tools.append(
                Tool(
                    name=name,
                    description=doc,
                    handler=make_handler(fn),
                )
            )

        return copilot_tools

    def _handle_tool_call(self, fn: Callable[..., Any], invocation: ToolInvocation) -> ToolResult:
        """Execute tool and return result."""
        if self._state:
            self._state.tool_calls += 1

        raw_args = invocation.get("arguments")
        args = raw_args if isinstance(raw_args, dict) else {}
        tool_name = invocation.get("tool_name", "unknown")

        logger.debug(f"{get_engine_name()} <-- LLM", tools=[tool_name])

        # SDK controls agentic loop, so we enforce budget here
        if self._state and self._state.tool_calls > self._max_calls:
            return ToolResult(
                textResultForLlm="BUDGET EXCEEDED. Stop and submit your rule now with submit_rule().",
                resultType="failure",
            )
        warning = self.check_budget_warning(self._state, self._max_calls) if self._state else None

        try:
            result = fn(**args)
            text_result = json.dumps(result) if not isinstance(result, str) else result
            if warning:
                text_result = f"{text_result}\n\n{warning}"
            return ToolResult(textResultForLlm=text_result)
        except RequestTemplates:
            # SDK agentic loop doesn't support mid-turn interrupts, return templates inline
            rules_dir = self._gateway_config.get("rules_dir")
            if rules_dir:
                actions_dir = rules_dir.parent / "actions"
                templates = TEMPLATES.format(rules_dir=rules_dir, actions_dir=actions_dir)
                # Reinforce guidelines since SDK context can drift
                reminder = (
                    "\n\n---\n"
                    "REMINDER - Follow guidelines from the initial prompt:\n"
                    "- Verify patterns against ACTUAL field values from the error context\n"
                    "- Least invasive fix - prefer config over code changes\n"
                    "- NEVER hardcode package names, paths, or versions\n"
                    "- Write GENERIC solutions that handle all similar cases"
                )
                return ToolResult(textResultForLlm=templates + reminder)
            # Fallback if config not set
            self._signal = RequestTemplates()
            return ToolResult(
                textResultForLlm="Signal: RequestTemplates",
                resultType="failure",
            )
        except ExplorationSignal as sig:
            self._signal = sig
            return ToolResult(
                textResultForLlm=f"Signal: {type(sig).__name__}",
                resultType="failure",
            )
        except Exception as e:
            logger.warning("Tool execution failed", tool=tool_name, error=str(e))
            return ToolResult(textResultForLlm=str(e), resultType="failure")

    def _get_user_prompt(self, messages: list[dict[str, Any]]) -> str | None:
        """Extract last user message content."""
        if not messages:
            return None
        last = messages[-1]
        if last.get("role") != "user":
            return None
        content = last.get("content")
        return content if isinstance(content, str) else None

    def reset(self) -> None:
        """Clean up client resources."""
        if self._session and self._loop and not self._loop.is_closed():
            try:
                self._loop.run_until_complete(self._session.destroy())
            except Exception as e:
                logger.warning("Failed to destroy session", gateway="copilot", error=str(e))
            self._session = None

        if self._client and self._loop and not self._loop.is_closed():
            try:
                self._loop.run_until_complete(self._client.stop())
            except Exception as e:
                logger.warning("Failed to stop gateway client", gateway="copilot", error=str(e))
            self._client = None

        if self._loop and not self._loop.is_closed():
            self._loop.close()
        self._loop = None
        self._state = None
        self._signal = None
        self._tool_map = {}

    def generate(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Single generation, optionally with structured output."""
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop.run_until_complete(self._generate_async(prompt, schema))

    async def _generate_async(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Async single generation."""
        client = CopilotClient()
        session = await client.create_session({"model": self._model})

        try:
            if schema:
                prompt = (
                    f"{prompt}\n\nRespond with JSON matching this schema:\n{json.dumps(schema)}"
                )

            response = await session.send_and_wait({"prompt": prompt})
            if response and hasattr(response, "data") and hasattr(response.data, "content"):
                content = response.data.content
                if content:
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        return {"text": content}
            return {}
        finally:
            await session.destroy()
            client.stop()
