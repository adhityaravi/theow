"""Tests for Anthropic gateway."""

from unittest.mock import MagicMock, patch

import pytest


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


def test_anthropic_gateway_build_declarations():
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        with patch("theow._gateway._anthropic.anthropic"):
            from theow._gateway._anthropic import AnthropicGateway

            gateway = AnthropicGateway(model="claude-sonnet-4-20250514")

            def my_tool(path: str, count: int = 5) -> str:
                """Read a file."""
                return ""

            declarations = gateway._build_tool_declarations([my_tool])

            assert len(declarations) == 1
            assert declarations[0]["name"] == "my_tool"
            assert declarations[0]["description"] == "Read a file."
            assert declarations[0]["input_schema"]["properties"]["path"]["type"] == "string"
            assert declarations[0]["input_schema"]["properties"]["count"]["type"] == "integer"
            assert "path" in declarations[0]["input_schema"]["required"]
            assert "count" not in declarations[0]["input_schema"]["required"]
