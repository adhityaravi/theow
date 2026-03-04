"""Tests for PydanticAI gateway."""

from unittest.mock import MagicMock, patch

import pytest

from theow._core._tools import GiveUp


def _make_response(parts=None, request_tokens=10, response_tokens=5):
    """Build a mock ModelResponse."""
    resp = MagicMock()
    resp.parts = parts or []
    resp.usage.request_tokens = request_tokens
    resp.usage.response_tokens = response_tokens
    resp.tool_calls = [p for p in resp.parts if hasattr(p, "tool_name") and hasattr(p, "args")]
    resp.text = ""
    return resp


def _make_text_part(content="Hello"):
    part = MagicMock()
    part.content = content
    part.tool_name = None  # not a tool call
    part.args = None
    # Make isinstance checks work for _sync_messages
    part.__class__ = MagicMock
    return part


def _make_tool_call_part(tool_name="read_file", args=None, tool_call_id="call-1"):
    part = MagicMock()
    part.tool_name = tool_name
    part.args = args or {}
    part.tool_call_id = tool_call_id
    return part


def test_pydantic_ai_gateway_conversation_no_tool_use():
    """Text-only replies trigger nudges before giving up."""
    with patch("theow._gateway._pydantic_ai.model_request_sync") as mock_request:
        from theow._gateway._pydantic_ai import PydanticAIGateway

        text_part = _make_text_part("Hello")
        response = _make_response(parts=[text_part])
        response.tool_calls = []
        mock_request.return_value = response

        gateway = PydanticAIGateway(model="anthropic:claude-sonnet-4-20250514")
        result = gateway.conversation(
            messages=[{"role": "user", "content": "test"}],
            tools=[],
            budget={"max_tool_calls_per_session": 5},
        )

        # 1 original + 2 nudge retries = 3 calls, each 15 tokens
        assert result.tokens_used == 45
        assert result.tool_calls == 0
        assert mock_request.call_count == 3


def test_pydantic_ai_conversation_with_tool_use():
    with patch("theow._gateway._pydantic_ai.model_request_sync") as mock_request:
        from theow._gateway._pydantic_ai import PydanticAIGateway

        tool_part = _make_tool_call_part(
            tool_name="read_file", args={"path": "/tmp/test.txt"}, tool_call_id="call-1"
        )
        resp1 = _make_response(parts=[tool_part], request_tokens=20, response_tokens=10)
        resp1.tool_calls = [tool_part]

        text_part = _make_text_part("Done")
        resp2 = _make_response(parts=[text_part], request_tokens=30, response_tokens=5)
        resp2.tool_calls = []

        # tool call, then text reply + 2 nudge retries (all text)
        mock_request.side_effect = [resp1, resp2, resp2, resp2]

        def read_file(path: str) -> str:
            """Read a file."""
            return "file contents"

        gateway = PydanticAIGateway(model="anthropic:claude-sonnet-4-20250514")
        messages = [{"role": "user", "content": "read the file"}]
        result = gateway.conversation(
            messages=messages,
            tools=[read_file],
            budget={"max_tool_calls_per_session": 10},
        )

        assert result.tool_calls == 1
        # 1 tool resp (30 tokens) + 3 text responses (35 tokens each)
        assert result.tokens_used == 30 + 35 * 3


def test_pydantic_ai_conversation_signal_during_tool():
    with patch("theow._gateway._pydantic_ai.model_request_sync") as mock_request:
        from theow._gateway._pydantic_ai import PydanticAIGateway

        tool_part = _make_tool_call_part(
            tool_name="give_up", args={"reason": "impossible"}, tool_call_id="call-1"
        )
        resp = _make_response(parts=[tool_part])
        resp.tool_calls = [tool_part]
        mock_request.return_value = resp

        def give_up(reason: str) -> None:
            raise GiveUp(reason)

        gateway = PydanticAIGateway(model="anthropic:claude-sonnet-4-20250514")
        with pytest.raises(GiveUp):
            gateway.conversation(
                messages=[{"role": "user", "content": "test"}],
                tools=[give_up],
                budget={"max_tool_calls_per_session": 10},
            )


def test_pydantic_ai_conversation_budget_warning():
    with patch("theow._gateway._pydantic_ai.model_request_sync") as mock_request:
        from theow._gateway._pydantic_ai import PydanticAIGateway

        call_counter = 0

        def make_tool_response():
            nonlocal call_counter
            call_counter += 1
            part = _make_tool_call_part(
                tool_name="noop", args={}, tool_call_id=f"call-{call_counter}"
            )
            resp = _make_response(parts=[part], request_tokens=5, response_tokens=5)
            resp.tool_calls = [part]
            return resp

        text_part = _make_text_part("done")
        text_resp = _make_response(parts=[text_part], request_tokens=5, response_tokens=5)
        text_resp.tool_calls = []

        # 4 tool calls, then text reply + 2 nudge retries (all text)
        responses = [make_tool_response() for _ in range(4)] + [text_resp] * 3
        mock_request.side_effect = responses

        def noop() -> str:
            return "ok"

        gateway = PydanticAIGateway(model="anthropic:claude-sonnet-4-20250514")
        messages = [{"role": "user", "content": "test"}]
        gateway.conversation(
            messages=messages,
            tools=[noop],
            budget={"max_tool_calls_per_session": 5},
        )

        user_messages = [m for m in messages if m.get("role") == "user"]
        warning_msgs = [
            m for m in user_messages if "tool calls remaining" in str(m.get("content", ""))
        ]
        assert len(warning_msgs) >= 1


def test_pydantic_ai_generate_with_schema():
    with patch("theow._gateway._pydantic_ai.model_request_sync") as mock_request:
        from theow._gateway._pydantic_ai import PydanticAIGateway

        tool_part = _make_tool_call_part(
            tool_name="structured_output", args={"key": "value"}, tool_call_id="call-1"
        )
        # Make isinstance(part, ToolCallPart) work in generate()
        from pydantic_ai.messages import ToolCallPart

        tool_part.__class__ = ToolCallPart
        resp = _make_response(parts=[tool_part])
        mock_request.return_value = resp

        gateway = PydanticAIGateway(model="anthropic:claude-sonnet-4-20250514")
        result = gateway.generate(
            prompt="generate data",
            schema={"type": "object", "properties": {"key": {"type": "string"}}},
        )
        assert result == {"key": "value"}


def test_pydantic_ai_generate_fallback_text():
    with patch("theow._gateway._pydantic_ai.model_request_sync") as mock_request:
        from theow._gateway._pydantic_ai import PydanticAIGateway

        from pydantic_ai.messages import TextPart

        text_part = _make_text_part("not json")
        text_part.__class__ = TextPart
        resp = _make_response(parts=[text_part])
        mock_request.return_value = resp

        gateway = PydanticAIGateway(model="anthropic:claude-sonnet-4-20250514")
        result = gateway.generate(prompt="test")
        assert result == {"text": "not json"}
