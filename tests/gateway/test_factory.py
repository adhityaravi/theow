"""Tests for gateway factory."""

from unittest.mock import patch

import pytest


def test_create_gateway_anthropic():
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        from theow._gateway import create_gateway
        from theow._gateway._anthropic import AnthropicGateway

        gateway = create_gateway("anthropic/claude-sonnet-4")
        assert isinstance(gateway, AnthropicGateway)
        assert gateway._model == "claude-sonnet-4"


def test_create_gateway_gemini():
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        from theow._gateway import create_gateway
        from theow._gateway._gemini import GeminiGateway

        gateway = create_gateway("gemini/gemini-2.0-flash")
        assert isinstance(gateway, GeminiGateway)
        assert gateway._model == "gemini-2.0-flash"


def test_create_gateway_copilot():
    with patch.dict("os.environ", {"GITHUB_TOKEN": "test-token"}):
        from theow._gateway import create_gateway
        from theow._gateway._copilot import CopilotGateway

        gateway = create_gateway("copilot/gpt-4")
        assert isinstance(gateway, CopilotGateway)
        assert gateway._model == "gpt-4"


def test_create_gateway_unknown_provider():
    from theow._gateway import create_gateway

    with pytest.raises(ValueError, match="Unknown LLM provider"):
        create_gateway("unknown/model")
