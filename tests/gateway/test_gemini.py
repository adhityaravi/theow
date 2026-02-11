"""Tests for Gemini gateway."""

from unittest.mock import MagicMock, patch

import pytest


def test_gemini_gateway_requires_api_key():
    with patch.dict("os.environ", {}, clear=True):
        from theow._gateway._gemini import GeminiGateway

        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            GeminiGateway()


def test_gemini_gateway_conversation_no_candidates():
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        with patch("theow._gateway._gemini.genai") as mock_genai:
            from theow._gateway._gemini import GeminiGateway

            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client

            mock_response = MagicMock()
            mock_response.candidates = []
            mock_response.usage_metadata = None
            mock_client.models.generate_content.return_value = mock_response

            gateway = GeminiGateway(model="gemini-2.0-flash")
            result = gateway.conversation(
                messages=[{"role": "user", "content": "test"}],
                tools=[],
                budget={"max_tool_calls": 5},
            )

            assert result.tool_calls == 0


def test_gemini_gateway_generate():
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        with patch("theow._gateway._gemini.genai") as mock_genai:
            from theow._gateway._gemini import GeminiGateway

            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client

            mock_response = MagicMock()
            mock_response.text = '{"result": "ok"}'
            mock_client.models.generate_content.return_value = mock_response

            gateway = GeminiGateway(model="gemini-2.0-flash")
            result = gateway.generate(prompt="test")

            assert result == {"result": "ok"}
