"""LLM gateway implementations."""

from theow._gateway._anthropic import AnthropicGateway
from theow._gateway._base import LLMGateway
from theow._gateway._copilot import CopilotGateway
from theow._gateway._gemini import GeminiGateway

GATEWAYS: dict[str, type[LLMGateway]] = {
    "anthropic": AnthropicGateway,
    "copilot": CopilotGateway,
    "gemini": GeminiGateway,
}


def create_gateway(llm_spec: str) -> LLMGateway:
    """Create gateway from provider/model spec (e.g., 'anthropic/claude-sonnet-4')."""
    provider, model = llm_spec.split("/", 1)

    gateway_cls = GATEWAYS.get(provider)
    if gateway_cls is None:
        raise ValueError(f"Unknown LLM provider: {provider}")

    return gateway_cls(model=model)  # type: ignore[call-arg]
