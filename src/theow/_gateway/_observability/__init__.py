"""LLM gateway observability — logfire spans, OTEL attributes, and provider-specific enrichments."""

from theow._gateway._observability._claude_agent import ClaudeAgentUsageTracker
from theow._gateway._observability._config import LogfireConfig, configure_logfire
from theow._gateway._observability._copilot import CopilotUsageTracker
from theow._gateway._observability._spans import (
    SessionState,
    instrumented,
    span_tool_call,
)

__all__ = [
    "LogfireConfig",
    "configure_logfire",
    "SessionState",
    "instrumented",
    "span_tool_call",
    "CopilotUsageTracker",
    "ClaudeAgentUsageTracker",
]
