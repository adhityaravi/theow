"""Claude Agent SDK LLM gateway implementation.

Like Copilot, the Claude Agent SDK manages its own agentic loop. Theow's tools
are exposed as MCP tools via an in-process MCP server. The SDK calls tools
automatically and Theow captures signals from tool handlers.
Basically Theow triggers a SDK loop and waits for it to finish, letting it call
the tools automatically. The SDK only raises termination signals.

The SDK spawns the Claude Code CLI as a subprocess, which injects its own
lengthy system prompt that conflicts with theow's explorer instructions.
Unlike Copilot (which gives full control over the prompt and tool loop), the
Claude Agent SDK is a much cruder integration — we can only prepend a
system_prompt and hope the CLI's internal prompt doesn't override our intent.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import termios
from contextlib import aclosing
from pathlib import Path
from typing import Any, Callable

from opentelemetry import context as otel_context

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    create_sdk_mcp_server,
    query,
    tool,
)
from claude_agent_sdk._internal.message_parser import parse_message
from claude_agent_sdk.types import StreamEvent

from theow._core._logging import get_engine_name, get_logger
from theow._core._prompts import TEMPLATES
from theow._core._tools import ExplorationSignal, GiveUp, RequestTemplates
from theow._gateway._base import (
    MAX_TEXT_NUDGES,
    TEXT_REPLY_NUDGE,
    ConversationResult,
    LLMGateway,
    build_tool_declaration,
)
from theow._gateway._observability import (
    ClaudeAgentUsageTracker,
    SessionState,
    instrumented,
    span_tool_call,
)

logger = get_logger(__name__)

# JSON schema type → Python type for the @tool decorator
_JSON_TYPE_TO_PYTHON: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}

# Tools that have superior Claude Code built-in equivalents (Read, Write, Bash, Glob).
# These are filtered out of the MCP server so the model uses its native tools instead.
_NATIVE_EQUIVALENT_TOOLS: frozenset[str] = frozenset(
    {
        "read_file",
        "write_file",
        "run_command",
        "list_directory",
    }
)


def _mcp_result(text: str) -> dict[str, Any]:
    """Wrap text in MCP tool result format."""
    return {"content": [{"type": "text", "text": text}]}


class ClaudeAgentGateway(LLMGateway):
    """Claude Agent SDK implementation.

    The SDK is async-first and manages tool execution via MCP tool handlers.
    We bridge to sync context using a persistent asyncio event loop.
    """

    # Keys from _gateway_options forwarded to ClaudeAgentOptions.
    _SDK_OPTION_KEYS: frozenset[str] = frozenset(
        {
            "effort",
            "thinking",
            "max_thinking_tokens",
        }
    )

    def __init__(
        self,
        model: str = "claude-sonnet-4",
        cli_path: str | Path | None = None,
        _gateway_options: dict[str, Any] | None = None,
    ) -> None:
        self._model = model
        self._sdk_options: dict[str, Any] = {
            k: v for k, v in (_gateway_options or {}).items() if k in self._SDK_OPTION_KEYS
        }
        # Resolve CLI path: env var > explicit param > auto-detect system install.
        # This lets users authenticate via their existing Claude subscription
        # instead of requiring an API key.
        self._cli_path: str | Path | None = (
            os.environ.get("THEOW_CLAUDE_CLI_PATH") or cli_path or shutil.which("claude")
        )
        self._client: ClaudeSDKClient | None = None
        self._client_connected: bool = False
        self._state: SessionState | None = None
        self._tool_map: dict[str, Callable[..., Any]] = {}
        self._signal: ExplorationSignal | None = None
        self._system_prompt: str | None = None
        self._gateway_config: dict[str, Any] = {}
        self._max_calls: int = 30
        self._max_tokens: int = 0
        self._allow_escalation: bool = False
        self._otel_ctx: Any = None
        self._usage = ClaudeAgentUsageTracker()
        # Persistent event loop — same pattern as Copilot.
        # Multi-turn conversations (e.g., after RequestTemplates signal) require
        # the loop to stay open across conversation() calls.
        self._loop: asyncio.AbstractEventLoop | None = None
        # Saved terminal settings — Claude Code CLI puts the terminal in raw mode
        # and doesn't restore it when the subprocess is killed without cleanup.
        self._saved_termios: list[Any] | None = None

    @instrumented
    def conversation(
        self,
        messages: list[dict[str, Any]],
        tools: list[Callable[..., Any]],
        budget: dict[str, Any],
    ) -> ConversationResult:
        """Run conversation with tool use until signal or budget exhausted."""
        self._max_calls, self._max_tokens = self._extract_budget(budget)
        self._allow_escalation = budget.get("allow_escalation", False)
        self._tool_map = self._build_tool_map(tools)
        self._state = SessionState()
        self._signal = None
        self._otel_ctx = otel_context.get_current()
        self._usage._state = self._state

        if self._saved_termios is None:
            try:
                self._saved_termios = termios.tcgetattr(sys.stdin)
            except (termios.error, OSError):
                pass

        user_prompt = self._get_user_prompt(messages)
        if not user_prompt:
            return ConversationResult(messages=messages, tool_calls=0, tokens_used=0)

        # Split system instructions from user content so the SDK gets a
        # proper system_prompt instead of Claude Code CLI's default.
        self._system_prompt, user_prompt = self._split_system_prompt(user_prompt)

        logger.debug(f"{get_engine_name()} --> LLM", provider="claude_agent", model=self._model)

        text_nudges = 0
        try:
            # Reuse event loop across conversation turns
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)

            response = self._loop.run_until_complete(self._run_conversation(user_prompt))

            # Nudge loop: if LLM replied with text and no signal, re-send with nudge
            while response and not self._signal and text_nudges < MAX_TEXT_NUDGES:
                text_nudges += 1
                logger.debug("Text-only reply, nudging LLM", nudge=text_nudges)
                response = self._loop.run_until_complete(self._run_conversation(TEXT_REPLY_NUDGE))
        except Exception as e:
            logger.error("Claude Agent SDK error", error=str(e))
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

    def _enrich_span(self, span: Any) -> None:
        """Called by @instrumented to set Claude Agent SDK-specific span attributes."""
        self._usage.enrich_span(span, self.model_name, self.gateway_name)

    async def _run_conversation(self, prompt: str) -> str | None:
        """Run async conversation with Claude Agent SDK.

        Client is kept alive across turns to maintain context (e.g., after
        RequestTemplates signal). Client is destroyed in reset().
        """
        if self._client is None:
            mcp_server = self._build_mcp_server()

            options = ClaudeAgentOptions(
                model=self._model,
                permission_mode="bypassPermissions",
                # Override Claude Code CLI's default system prompt with theow's
                # instructions so the model knows it's a Theow exploration agent.
                system_prompt=self._system_prompt,
                # tools=None (default) keeps all Claude Code built-in tools
                # (Read, Write, Bash, Glob, Grep, etc.). Theow-specific tools
                # (signals, rule search, etc.) are added via MCP.
                mcp_servers={"theow": mcp_server},
                max_turns=self._max_calls,
                cli_path=self._cli_path,
                debug_stderr=None,
                # Stream events carry per-turn token counts — the only
                # reliable cumulative source for input/output tokens.
                include_partial_messages=True,
                **self._sdk_options,
            )
            self._client = ClaudeSDKClient(options=options)

        if not self._client_connected:
            await self._client.connect()
            self._client_connected = True

        await self._client.query(prompt)

        query_obj = self._client._query
        assert query_obj is not None, "Client query not initialized after connect()"

        text_parts: list[str] = []
        async with aclosing(query_obj.receive_messages()) as messages:
            async for data in messages:
                message = parse_message(data)
                if message is None:
                    continue

                if isinstance(message, StreamEvent):
                    self._usage.update_from_stream_event(message.event)
                    continue
                elif isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            text_parts.append(block.text)
                elif isinstance(message, ResultMessage):
                    self._usage.update_from_result(message)
                    break  # ResultMessage = end of response

        return "\n".join(text_parts) if text_parts else None

    def _build_mcp_server(self) -> Any:
        """Convert tool map to an in-process MCP server with tool handlers."""
        sdk_tools = []

        for name, fn in self._tool_map.items():
            if name in _NATIVE_EQUIVALENT_TOOLS:
                continue
            decl = build_tool_declaration(name, fn, schema_key="parameters")

            # Convert JSON schema types to Python types for @tool decorator
            params: dict[str, type] = {}
            props = decl["parameters"].get("properties", {})
            for param_name, param_schema in props.items():
                json_type = param_schema.get("type", "string")
                params[param_name] = _JSON_TYPE_TO_PYTHON.get(json_type, str)

            def make_handler(tool_fn: Callable[..., Any], tool_name: str) -> Callable[..., Any]:
                async def handler(args: dict[str, Any]) -> dict[str, Any]:
                    result = self._handle_tool_result(tool_fn, args, tool_name)
                    # If a signal was raised (budget exceeded, GiveUp, etc.),
                    # interrupt the client immediately. Must be awaited here
                    # because _handle_tool_result is sync.
                    if self._signal and self._client:
                        await self._client.interrupt()
                    return result

                return handler

            decorated = tool(name, decl["description"], params)(make_handler(fn, name))
            sdk_tools.append(decorated)

        return create_sdk_mcp_server(name="theow", tools=sdk_tools)

    def _handle_tool_result(
        self, fn: Callable[..., Any], args: dict[str, Any], tool_name: str
    ) -> dict[str, Any]:
        """Execute tool and return MCP-format result."""
        # Restore OTEL context captured in conversation() so tool call spans
        # are children of the @instrumented conversation span.
        token = otel_context.attach(self._otel_ctx) if self._otel_ctx else None
        try:
            return self._handle_tool_result_inner(fn, args, tool_name)
        finally:
            if token is not None:
                otel_context.detach(token)

    def _handle_tool_result_inner(
        self, fn: Callable[..., Any], args: dict[str, Any], tool_name: str
    ) -> dict[str, Any]:
        """Execute tool within restored OTEL context."""
        if self._state:
            self._state.tool_calls += 1

        logger.debug(f"{get_engine_name()} <-- LLM", tools=[tool_name])

        # Hard kill: if LLM ignores the 80% warning and exceeds 100%.
        # Sets _signal so the async handler (make_handler) awaits client.interrupt().
        if self._state and self.check_budget_exceeded(
            self._state, self._max_calls, self._max_tokens
        ):
            self._signal = GiveUp("Session budget exceeded")
            return _mcp_result("Session budget exceeded")

        warning = (
            self.check_budget_warning(
                self._state, self._max_calls, self._max_tokens, self._allow_escalation
            )
            if self._state
            else None
        )

        with span_tool_call(tool_name, self._state):
            try:
                result = fn(**args)
                text_result = json.dumps(result) if not isinstance(result, str) else result
                if warning:
                    text_result = f"{text_result}\n\n{warning}"
                return _mcp_result(text_result)
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
                        "- NEVER hardcode case-specific values\n"
                        "- Write GENERIC solutions that handle all similar cases"
                    )
                    return _mcp_result(templates + reminder)
                # Fallback if config not set
                self._signal = RequestTemplates()
                return _mcp_result("Signal: RequestTemplates")
            except ExplorationSignal as sig:
                self._signal = sig
                return _mcp_result(f"Signal: {type(sig).__name__}")
            except Exception as e:
                logger.warning("Tool execution failed", tool=tool_name, error=str(e))
                return _mcp_result(str(e))

    _FALLBACK_SYSTEM_PROMPT = (
        "Complete the task described below using your built-in tools and the "
        "MCP signal tools provided. When finished, you MUST call mcp__theow___done(message) "
        "if you solved the problem, or mcp__theow___give_up(reason) if you cannot."
    )

    @staticmethod
    def _split_system_prompt(prompt: str) -> tuple[str, str]:
        """Split theow's combined prompt into system instructions and user content.

        The explorer builds a single string: INTRO (system instructions) +
        ERROR (user-facing error context).  The boundary is ``## Error Context``
        which is the first line of the ERROR template.  When found, INTRO goes
        to ``system_prompt`` in ClaudeAgentOptions so the SDK uses theow's
        identity instead of Claude Code CLI's default.

        For run_direct (LLM actions), there's no INTRO — the prompt is a
        self-contained task template.  A minimal fallback system prompt is
        used to override Claude Code CLI's default identity.
        """
        marker = "\n## Error Context\n"
        idx = prompt.find(marker)
        if idx == -1:
            return ClaudeAgentGateway._FALLBACK_SYSTEM_PROMPT, prompt
        return prompt[:idx].strip(), prompt[idx:].strip()

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
        self._usage.reset()

        # The SDK's subprocess transport uses anyio cancel scopes that are
        # bound to the task they were created in.  Calling disconnect() from
        # run_until_complete() creates a *new* task, triggering
        # "Attempted to exit cancel scope in a different task".
        # Instead, just drop the references and let the subprocess terminate
        # naturally (the GC / __del__ path closes stdin which signals EOF).
        self._client = None
        self._client_connected = False

        if self._loop and not self._loop.is_closed():
            self._loop.close()
        self._loop = None
        self._state = None
        self._signal = None
        self._tool_map = {}

        # Restore terminal settings — Claude Code CLI puts the terminal in raw
        # mode and the subprocess may not clean up when killed via GC.
        if self._saved_termios is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._saved_termios)
            except (termios.error, OSError):
                pass
            self._saved_termios = None

    @instrumented
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
        """Async single generation using the standalone query() function."""
        if schema:
            prompt = f"{prompt}\n\nRespond with JSON matching this schema:\n{json.dumps(schema)}"

        options = ClaudeAgentOptions(
            model=self._model,
            permission_mode="bypassPermissions",
            max_turns=1,
            cli_path=self._cli_path,
            debug_stderr=None,
        )

        text_parts: list[str] = []
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)

        content = "\n".join(text_parts)
        if content:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"text": content}
        return {}

    @property
    def provider_name(self) -> str:
        """Provider name for OTEL gen_ai.system attribute."""
        return "anthropic"
