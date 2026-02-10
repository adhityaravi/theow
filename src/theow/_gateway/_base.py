"""LLM gateway base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ConversationResult:
    """Result from a conversation session."""

    messages: list[dict[str, Any]]
    tool_calls: int = 0
    tokens_used: int = 0


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
