"""Tests for Anthropic gateway."""

from unittest.mock import MagicMock, patch

import pytest

from theow._core._tools import GiveUp


def test_anthropic_gateway_requires_api_key():
    with patch.dict("os.environ", {}, clear=True):
        from theow._gateway._anthropic import AnthropicGateway

        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            AnthropicGateway()


def test_anthropic_gateway_conversation_no_tool_use():
    """Text-only replies trigger nudges before giving up."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        with patch("theow._gateway._anthropic.anthropic") as mock_anthropic:
            from theow._gateway._anthropic import AnthropicGateway

            mock_client = MagicMock()
            mock_anthropic.Anthropic.return_value = mock_client

            mock_response = MagicMock()
            mock_response.content = [MagicMock(type="text", text="Hello")]
            mock_response.usage.input_tokens = 10
            mock_response.usage.output_tokens = 5
            mock_client.messages.create.return_value = mock_response

            gateway = AnthropicGateway(model="claude-sonnet-4-20250514")
            result = gateway.conversation(
                messages=[{"role": "user", "content": "test"}],
                tools=[],
                budget={"max_tool_calls_per_session": 5},
            )

            # 1 original + 2 nudge retries = 3 calls, each 15 tokens
            assert result.tokens_used == 45
            assert result.tool_calls == 0
            assert mock_client.messages.create.call_count == 3


def test_anthropic_conversation_with_tool_use():
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        with patch("theow._gateway._anthropic.anthropic") as mock_anthropic:
            from theow._gateway._anthropic import AnthropicGateway

            mock_client = MagicMock()
            mock_anthropic.Anthropic.return_value = mock_client

            tool_block = MagicMock()
            tool_block.type = "tool_use"
            tool_block.name = "read_file"
            tool_block.input = {"path": "/tmp/test.txt"}
            tool_block.id = "tool-123"

            mock_resp1 = MagicMock()
            mock_resp1.content = [tool_block]
            mock_resp1.usage.input_tokens = 20
            mock_resp1.usage.output_tokens = 10

            text_block = MagicMock()
            text_block.type = "text"
            text_block.text = "Done"
            mock_resp2 = MagicMock()
            mock_resp2.content = [text_block]
            mock_resp2.usage.input_tokens = 30
            mock_resp2.usage.output_tokens = 5

            # tool call, then text reply + 2 nudge retries (all text)
            mock_client.messages.create.side_effect = [
                mock_resp1,
                mock_resp2,
                mock_resp2,
                mock_resp2,
            ]

            def read_file(path: str) -> str:
                """Read a file."""
                return "file contents"

            gateway = AnthropicGateway(model="claude-sonnet-4-20250514")
            messages = [{"role": "user", "content": "read the file"}]
            result = gateway.conversation(
                messages=messages,
                tools=[read_file],
                budget={"max_tool_calls_per_session": 10},
            )

            assert result.tool_calls == 1
            # 1 tool resp (30 tokens) + 3 text responses (35 tokens each)
            assert result.tokens_used == 30 + 35 * 3


def test_anthropic_conversation_signal_during_tool():
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        with patch("theow._gateway._anthropic.anthropic") as mock_anthropic:
            from theow._gateway._anthropic import AnthropicGateway

            mock_client = MagicMock()
            mock_anthropic.Anthropic.return_value = mock_client

            tool_block = MagicMock()
            tool_block.type = "tool_use"
            tool_block.name = "give_up"
            tool_block.input = {"reason": "impossible"}
            tool_block.id = "tool-456"

            mock_resp = MagicMock()
            mock_resp.content = [tool_block]
            mock_resp.usage.input_tokens = 10
            mock_resp.usage.output_tokens = 5
            mock_client.messages.create.return_value = mock_resp

            def give_up(reason: str) -> None:
                raise GiveUp(reason)

            gateway = AnthropicGateway(model="claude-sonnet-4-20250514")
            with pytest.raises(GiveUp):
                gateway.conversation(
                    messages=[{"role": "user", "content": "test"}],
                    tools=[give_up],
                    budget={"max_tool_calls_per_session": 10},
                )


def test_anthropic_conversation_budget_warning():
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        with patch("theow._gateway._anthropic.anthropic") as mock_anthropic:
            from theow._gateway._anthropic import AnthropicGateway

            mock_client = MagicMock()
            mock_anthropic.Anthropic.return_value = mock_client

            def make_tool_response():
                block = MagicMock()
                block.type = "tool_use"
                block.name = "noop"
                block.input = {}
                block.id = f"tool-{id(block)}"
                resp = MagicMock()
                resp.content = [block]
                resp.usage.input_tokens = 5
                resp.usage.output_tokens = 5
                return resp

            text_resp = MagicMock()
            text_resp.content = [MagicMock(type="text", text="done")]
            text_resp.usage.input_tokens = 5
            text_resp.usage.output_tokens = 5

            # 4 tool calls, then text reply + 2 nudge retries (all text)
            responses = [make_tool_response() for _ in range(4)] + [text_resp] * 3
            mock_client.messages.create.side_effect = responses

            def noop() -> str:
                return "ok"

            gateway = AnthropicGateway(model="claude-sonnet-4-20250514")
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


def test_anthropic_generate_with_schema():
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        with patch("theow._gateway._anthropic.anthropic") as mock_anthropic:
            from theow._gateway._anthropic import AnthropicGateway

            mock_client = MagicMock()
            mock_anthropic.Anthropic.return_value = mock_client

            tool_block = MagicMock()
            tool_block.type = "tool_use"
            tool_block.input = {"key": "value"}

            mock_resp = MagicMock()
            mock_resp.content = [tool_block]
            mock_client.messages.create.return_value = mock_resp

            gateway = AnthropicGateway(model="claude-sonnet-4-20250514")
            result = gateway.generate(
                prompt="generate data",
                schema={"type": "object", "properties": {"key": {"type": "string"}}},
            )
            assert result == {"key": "value"}


def test_anthropic_generate_fallback_text():
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        with patch("theow._gateway._anthropic.anthropic") as mock_anthropic:
            from theow._gateway._anthropic import AnthropicGateway

            mock_client = MagicMock()
            mock_anthropic.Anthropic.return_value = mock_client

            text_block = MagicMock()
            text_block.text = "not json"

            mock_resp = MagicMock()
            mock_resp.content = [text_block]
            mock_client.messages.create.return_value = mock_resp

            gateway = AnthropicGateway(model="claude-sonnet-4-20250514")
            result = gateway.generate(prompt="test")
            assert result == {"text": "not json"}
