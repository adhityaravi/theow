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
                budget={"max_tool_calls": 5},
            )

            assert result.tokens_used == 15
            assert result.tool_calls == 0


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

            mock_client.messages.create.side_effect = [mock_resp1, mock_resp2]

            def read_file(path: str) -> str:
                """Read a file."""
                return "file contents"

            gateway = AnthropicGateway(model="claude-sonnet-4-20250514")
            messages = [{"role": "user", "content": "read the file"}]
            result = gateway.conversation(
                messages=messages,
                tools=[read_file],
                budget={"max_tool_calls": 10},
            )

            assert result.tool_calls == 1
            assert result.tokens_used == 65


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
                    budget={"max_tool_calls": 10},
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

            responses = [make_tool_response() for _ in range(4)] + [text_resp]
            mock_client.messages.create.side_effect = responses

            def noop() -> str:
                return "ok"

            gateway = AnthropicGateway(model="claude-sonnet-4-20250514")
            messages = [{"role": "user", "content": "test"}]
            gateway.conversation(
                messages=messages,
                tools=[noop],
                budget={"max_tool_calls": 5},
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
