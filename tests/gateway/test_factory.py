"""Tests for gateway factory."""

import sys
import warnings
from unittest.mock import MagicMock, patch

import pytest

from theow._gateway import create_gateway
from theow._gateway._base import GatewayConfig, GatewayProvider
from theow._gateway._pydantic_ai import PydanticAIGateway


# --- Default (PYDANTIC) routing ---


def test_create_gateway_pydantic_anthropic():
    gateway = create_gateway("anthropic/claude-sonnet-4")
    assert isinstance(gateway, PydanticAIGateway)
    assert gateway._model == "anthropic:claude-sonnet-4"


def test_create_gateway_pydantic_gemini_alias():
    """Gemini prefix maps to google-gla: in PydanticAI."""
    gateway = create_gateway("gemini/gemini-2.0-flash")
    assert isinstance(gateway, PydanticAIGateway)
    assert gateway._model == "google-gla:gemini-2.0-flash"


def test_create_gateway_pydantic_openai():
    """Any PydanticAI-supported provider works out of the box."""
    gateway = create_gateway("openai/gpt-5")
    assert isinstance(gateway, PydanticAIGateway)
    assert gateway._model == "openai:gpt-5"


def test_create_gateway_copilot_always_native():
    """Copilot always uses native SDK regardless of provider setting."""
    mock_copilot = MagicMock()
    mock_copilot.__file__ = "/fake/copilot/__init__.py"
    mock_copilot.CopilotClient = MagicMock()
    mock_copilot.PermissionHandler = MagicMock()
    mock_copilot.Tool = MagicMock()
    mock_types = MagicMock()

    with patch.dict(
        "sys.modules",
        {"copilot": mock_copilot, "copilot.types": mock_types},
    ):
        with patch.dict("os.environ", {"GITHUB_TOKEN": "test-token"}):
            # Force re-import so the gateway module picks up the mock
            sys.modules.pop("theow._gateway._copilot", None)

            from theow._gateway._copilot import CopilotGateway

            gateway = create_gateway("copilot/gpt-4")
            assert isinstance(gateway, CopilotGateway)
            assert gateway._model == "gpt-4"

            # Explicitly passing PYDANTIC shouldn't change copilot routing
            gateway2 = create_gateway(
                "copilot/gpt-4", config=GatewayConfig(provider=GatewayProvider.PYDANTIC)
            )
            assert isinstance(gateway2, CopilotGateway)


def test_create_gateway_claude_agent_always_native():
    """Claude Agent SDK always uses native SDK regardless of provider setting."""
    mock_sdk = MagicMock()
    mock_sdk.AssistantMessage = type("AssistantMessage", (), {})
    mock_sdk.ClaudeAgentOptions = MagicMock()
    mock_sdk.ClaudeSDKClient = MagicMock()
    mock_sdk.ResultMessage = type("ResultMessage", (), {})
    mock_sdk.TextBlock = type("TextBlock", (), {})
    mock_sdk.create_sdk_mcp_server = MagicMock()
    mock_sdk.query = MagicMock()
    mock_sdk.tool = MagicMock(side_effect=lambda *a: lambda fn: fn)

    mock_types = MagicMock()
    mock_types.StreamEvent = type("StreamEvent", (), {})
    mock_parser = MagicMock()
    mock_internal = MagicMock()
    mock_internal.message_parser = mock_parser

    with patch.dict(
        "sys.modules",
        {
            "claude_agent_sdk": mock_sdk,
            "claude_agent_sdk.types": mock_types,
            "claude_agent_sdk._internal": mock_internal,
            "claude_agent_sdk._internal.message_parser": mock_parser,
        },
    ):
        # Force re-import so the gateway module picks up the mock
        sys.modules.pop("theow._gateway._claude_agent", None)

        from theow._gateway._claude_agent import ClaudeAgentGateway

        gateway = create_gateway("claude-agent/claude-sonnet-4")
        assert isinstance(gateway, ClaudeAgentGateway)
        assert gateway._model == "claude-sonnet-4"

        # Explicitly passing PYDANTIC shouldn't change routing
        gateway2 = create_gateway(
            "claude-agent/claude-sonnet-4",
            config=GatewayConfig(provider=GatewayProvider.PYDANTIC),
        )
        assert isinstance(gateway2, ClaudeAgentGateway)

        # Underscore variant also works
        gateway3 = create_gateway("claude_agent/claude-sonnet-4")
        assert isinstance(gateway3, ClaudeAgentGateway)


# --- NATIVE routing (deprecated) ---


def test_create_gateway_native_anthropic():
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        from theow._gateway._anthropic import AnthropicGateway

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gateway = create_gateway(
                "anthropic/claude-sonnet-4",
                config=GatewayConfig(provider=GatewayProvider.NATIVE),
            )

        assert isinstance(gateway, AnthropicGateway)
        assert gateway._model == "claude-sonnet-4"
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated" in str(w[0].message).lower()


def test_create_gateway_native_gemini():
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        from theow._gateway._gemini import GeminiGateway

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gateway = create_gateway(
                "gemini/gemini-2.0-flash",
                config=GatewayConfig(provider=GatewayProvider.NATIVE),
            )

        assert isinstance(gateway, GeminiGateway)
        assert gateway._model == "gemini-2.0-flash"
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)


def test_create_gateway_native_unknown_raises():
    with pytest.raises(ValueError, match="No native gateway for"):
        create_gateway("openai/gpt-5", config=GatewayConfig(provider=GatewayProvider.NATIVE))
