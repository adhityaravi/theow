"""Logfire span helpers and session state for token tracking."""

from __future__ import annotations

import functools
import json
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, cast

import logfire

from theow._core._tools import ExplorationSignal


@dataclass
class SessionState:
    """Tracks mutable state across the tool-calling loop."""

    tool_calls: int = 0
    tokens_used: int = 0  # input + output, used for budget checks
    input_tokens: int = 0
    output_tokens: int = 0
    warned_about_budget: bool = False


def instrumented[F: Callable[..., Any]](fn: F) -> F:
    """Decorator that wraps a gateway method in a logfire span.

    Reads gateway_name and model_name from self to build the span name.
    Sets gen_ai.* OTEL semantic convention attributes so Logfire renders
    token usage badges the same way pydantic-ai auto-instrumented spans do.
    """

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        op = "chat" if fn.__name__ == "conversation" else fn.__name__
        name = f"LLM {fn.__name__}"
        signal: ExplorationSignal | None = None
        with logfire.span("LLM {op}", _span_name=name, op=op) as span:
            span.set_attribute("gen_ai.operation.name", op)
            span.set_attribute("gen_ai.request.model", self.model_name)
            span.set_attribute("gen_ai.system", self.provider_name)
            try:
                return fn(self, *args, **kwargs)
            except ExplorationSignal as exc:
                # Done / GiveUp are control flow, not errors — capture and
                # re-raise after the span closes so logfire sees OK status.
                signal = exc
            finally:
                state = getattr(self, "_state", None)
                if state:
                    span.set_attribute("gen_ai.usage.input_tokens", state.input_tokens)
                    span.set_attribute("gen_ai.usage.output_tokens", state.output_tokens)
                    span.set_attribute("gen_ai.response.model", self.model_name)
                # Let gateways set provider-specific attrs (e.g. quota, cost)
                enrich = getattr(self, "_enrich_span", None)
                if enrich:
                    enrich(span)
        # Re-raise outside the span context so logfire doesn't see it
        if signal is not None:
            raise signal

    return cast(F, wrapper)


@contextmanager
def span_tool_call(
    tool_name: str,
    state: SessionState | None = None,
    args: dict[str, Any] | None = None,
):
    """Logfire span for an individual tool call."""
    with logfire.span(
        "tool call: {tool_name}", _span_name="tool call", tool_name=tool_name
    ) as span:
        if args:
            span.set_attribute("tool.arguments", json.dumps(args))
        yield
        if state:
            span.set_attribute("gen_ai.usage.input_tokens", state.input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", state.output_tokens)
