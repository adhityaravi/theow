"""Copilot-specific observability — quota tracking, cost, and span enrichment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from theow._gateway._observability._spans import SessionState

_OVERAGE_RATE_USD = 0.04


@dataclass
class CopilotUsageTracker:
    """Tracks Copilot-specific quota, premium requests, and cost.

    Subscribes to SDK ASSISTANT_USAGE events and updates both itself
    (quota/cost) and the shared SessionState (token counts for budget).

    Cost is computed from the ``premium_interactions`` quota delta — the
    difference between used_requests at the start and end of the session.
    GitHub applies model multipliers server-side, so the delta already
    reflects the actual premium requests consumed.
    """

    quota_remaining_pct: float | None = None
    quota_used_requests: float | None = None
    _initial_premium_used: float | None = field(default=None, repr=False)
    _state: SessionState | None = field(default=None, repr=False)

    @property
    def session_premium_requests(self) -> float:
        """Premium requests consumed in this session (delta from quota snapshots)."""
        if self._initial_premium_used is None or self.quota_used_requests is None:
            return 0.0
        return max(0.0, self.quota_used_requests - self._initial_premium_used)

    @property
    def cost_usd(self) -> float:
        return self.session_premium_requests * _OVERAGE_RATE_USD

    def handle_event(self, event: Any) -> None:
        """Process an SDK session event."""
        from copilot.generated.session_events import SessionEventType

        if event.type != SessionEventType.ASSISTANT_USAGE:
            return
        if not event.data:
            return

        # Token counts → SessionState (for budget checks + gen_ai.* attrs)
        if self._state:
            inp = getattr(event.data, "input_tokens", None) or 0
            out = getattr(event.data, "output_tokens", None) or 0
            self._state.input_tokens += int(inp)
            self._state.output_tokens += int(out)
            self._state.tokens_used = self._state.input_tokens + self._state.output_tokens

        # Quota snapshots — only track premium_interactions (the billing-relevant model)
        qs = getattr(event.data, "quota_snapshots", None)
        if qs:
            premium = qs.get("premium_interactions")
            if premium:
                if self._initial_premium_used is None:
                    self._initial_premium_used = premium.used_requests
                self.quota_remaining_pct = premium.remaining_percentage
                self.quota_used_requests = premium.used_requests

    def enrich_span(self, span: Any, model_name: str, gateway_name: str) -> None:
        """Set Copilot-specific attributes on a logfire span."""
        span.set_attribute("operation.cost", self.cost_usd)
        span.set_attribute("copilot.premium_requests", self.session_premium_requests)
        if self.quota_remaining_pct is not None:
            span.set_attribute("copilot.quota.remaining_pct", self.quota_remaining_pct)
        if self.quota_used_requests is not None:
            span.set_attribute("copilot.quota.used_requests", self.quota_used_requests)

    def reset(self) -> None:
        """Reset per-session counters."""
        self._initial_premium_used = None
        self.quota_remaining_pct = None
        self.quota_used_requests = None
        self._state = None
