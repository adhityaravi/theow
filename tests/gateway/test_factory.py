"""Tests for gateway factory."""

import warnings
from unittest.mock import patch

import pytest

from theow._gateway import create_gateway
from theow._gateway._base import GatewayProvider
from theow._gateway._copilot import CopilotGateway
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
    with patch.dict("os.environ", {"GITHUB_TOKEN": "test-token"}):
        gateway = create_gateway("copilot/gpt-4")
        assert isinstance(gateway, CopilotGateway)
        assert gateway._model == "gpt-4"

        # Explicitly passing PYDANTIC shouldn't change copilot routing
        gateway2 = create_gateway("copilot/gpt-4", provider=GatewayProvider.PYDANTIC)
        assert isinstance(gateway2, CopilotGateway)


# --- NATIVE routing (deprecated) ---


def test_create_gateway_native_anthropic():
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        from theow._gateway._anthropic import AnthropicGateway

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gateway = create_gateway("anthropic/claude-sonnet-4", provider=GatewayProvider.NATIVE)

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
            gateway = create_gateway("gemini/gemini-2.0-flash", provider=GatewayProvider.NATIVE)

        assert isinstance(gateway, GeminiGateway)
        assert gateway._model == "gemini-2.0-flash"
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)


def test_create_gateway_native_unknown_raises():
    with pytest.raises(ValueError, match="No native gateway for"):
        create_gateway("openai/gpt-5", provider=GatewayProvider.NATIVE)
