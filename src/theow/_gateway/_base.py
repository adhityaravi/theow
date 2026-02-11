"""LLM gateway base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from theow._core._logging import get_logger

logger = get_logger(__name__)

# Budget configuration
SOFT_LIMIT_RATIO = 0.8  # Warn at 80% of budget


@dataclass
class ConversationResult:
    """Result from a conversation session."""

    messages: list[dict[str, Any]]
    tool_calls: int = 0
    tokens_used: int = 0


@dataclass
class SessionState:
    """Tracks mutable state across the tool-calling loop."""

    tool_calls: int = 0
    tokens_used: int = 0
    warned_about_budget: bool = False


class LLMGateway(ABC):
    """Abstract base for LLM provider implementations."""

    @abstractmethod
    def conversation(
        self,
        messages: list[dict[str, Any]],
        tools: list[Callable[..., Any]],
        budget: dict[str, Any],
    ) -> ConversationResult:
        """Run a conversation with tool use.

        Messages list is modified in-place as conversation progresses.
        Raises ExplorationSignal subclasses when LLM calls signal tools.
        """
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Single generation, optionally with structured output."""
        ...

    def check_budget_warning(
        self,
        state: SessionState,
        max_calls: int,
    ) -> str | None:
        """Check if budget warning should be issued.

        Returns warning message if at soft limit, None otherwise.
        Sets state.warned_about_budget to prevent duplicate warnings.
        """
        if state.warned_about_budget:
            return None

        soft_limit = int(max_calls * SOFT_LIMIT_RATIO)
        if state.tool_calls < soft_limit:
            return None

        remaining = max_calls - state.tool_calls
        state.warned_about_budget = True
        logger.info("Budget warning triggered", remaining=remaining, max_calls=max_calls)

        return (
            f"⚠️ BUDGET WARNING: You have {remaining} tool calls remaining "
            f"(out of {max_calls}). Wrap up now.\n\n"
            f"If you have a fix: call request_templates() then submit_rule().\n\n"
            f"If you can't finish: submit with tags: [incomplete] and add a 'notes' field. "
            f"The next attempt can continue from there."
        )
