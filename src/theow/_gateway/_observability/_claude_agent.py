"""Claude Agent SDK observability — usage tracking and span enrichment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from theow._gateway._observability._spans import SessionState


@dataclass
class ClaudeAgentUsageTracker:
    """Tracks Claude Agent SDK usage — tokens and cost.

    Token counts come from stream events (``message_start`` for input,
    ``message_delta`` for output) — the only per-turn source that
    accumulates correctly across the agentic loop.  Cost comes from
    ``ResultMessage.total_cost_usd`` which is already cumulative.
    """

    total_cost_usd: float = 0.0
    _state: SessionState | None = field(default=None, repr=False)

    def update_from_stream_event(self, event: dict[str, Any]) -> None:
        """Accumulate token counts from a raw Anthropic stream event.

        ``message_start`` carries ``message.usage.input_tokens`` and
        optionally ``cache_creation_input_tokens``.
        ``message_delta`` carries ``usage.output_tokens``.
        """
        if not self._state:
            return

        event_type = event.get("type")
        if event_type == "message_start":
            usage = (event.get("message") or {}).get("usage") or {}
            self._state.input_tokens += usage.get("input_tokens", 0)
            self._state.input_tokens += usage.get("cache_creation_input_tokens", 0)
            self._state.tokens_used = self._state.input_tokens + self._state.output_tokens
        elif event_type == "message_delta":
            usage = event.get("usage") or {}
            self._state.output_tokens += usage.get("output_tokens", 0)
            self._state.tokens_used = self._state.input_tokens + self._state.output_tokens

    def update_from_result(self, result_message: Any) -> None:
        """Extract cumulative cost from SDK ResultMessage (cost only, no tokens)."""
        cost = getattr(result_message, "total_cost_usd", None)
        if cost is not None:
            self.total_cost_usd += cost

    def enrich_span(self, span: Any, model_name: str, gateway_name: str) -> None:
        """Set Claude Agent SDK-specific attributes on a logfire span."""
        span.set_attribute("operation.cost", self.total_cost_usd)

    def reset(self) -> None:
        """Reset per-session counters."""
        self._state = None
        self.total_cost_usd = 0.0
