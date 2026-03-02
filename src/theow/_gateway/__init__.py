"""LLM gateway implementations."""

from __future__ import annotations

import warnings

from theow._gateway._anthropic import AnthropicGateway
from theow._gateway._base import GatewayProvider, LLMGateway
from theow._gateway._copilot import CopilotGateway
from theow._gateway._gemini import GeminiGateway
from theow._gateway._pydantic_ai import PydanticAIGateway

# Native gateway class registry
NATIVE_GATEWAYS: dict[str, type[LLMGateway]] = {
    "anthropic": AnthropicGateway,
    "gemini": GeminiGateway,
}

# Known aliases: theow provider prefix -> PydanticAI provider prefix
_PROVIDER_ALIASES: dict[str, str] = {
    "gemini": "google-gla",
}


def create_gateway(
    llm_spec: str,
    provider: GatewayProvider = GatewayProvider.PYDANTIC,
) -> LLMGateway:
    """Create gateway from provider/model spec (e.g., 'anthropic/claude-sonnet-4').

    Args:
        llm_spec: Provider/model string like ``"anthropic/claude-opus-4-6"``.
        provider: Which backend to use.  ``PYDANTIC`` (default) routes through
            :class:`PydanticAIGateway`; ``NATIVE`` routes to the original
            per-provider gateway classes.

    Any PydanticAI-supported provider works out of the box when using PYDANTIC:
    ``"openai/gpt-5"``, ``"github/openai/gpt-5"``, ``"bedrock/..."``, etc.
    """
    gateway_provider, model = llm_spec.split("/", 1)

    # Copilot always uses native SDK — PydanticAI doesn't support the Copilot protocol
    if gateway_provider == "copilot":
        return CopilotGateway(model=model)

    if provider == GatewayProvider.NATIVE:
        native_cls = NATIVE_GATEWAYS.get(gateway_provider)
        if native_cls is None:
            raise ValueError(f"No native gateway for: {gateway_provider}")
        warnings.warn(
            f"Native {gateway_provider} gateway is deprecated. "
            "Use GatewayProvider.PYDANTIC instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return native_cls(model=model)  # type: ignore[call-arg]

    # PydanticAI: translate "anthropic/model" -> "anthropic:model"
    # Apply known aliases (e.g. "gemini" -> "google-gla")
    pai_provider = _PROVIDER_ALIASES.get(gateway_provider, gateway_provider)
    pydantic_spec = f"{pai_provider}:{model}"
    return PydanticAIGateway(model=pydantic_spec)
