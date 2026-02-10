"""Main Theow engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, ParamSpec, TypeVar

from theow._core._chroma_store import ChromaStore
from theow._core._decorators import (
    ActionRegistry,
    MarkDecorator,
    ToolRegistry,
    TracingInfo,
    set_standalone_registry,
)
from theow._core._explorer import Explorer
from theow._core._logging import get_logger
from theow._core._models import Rule
from theow._core._resolver import Resolver
from theow._core._stats import meow as _meow
from theow._gateway._anthropic import AnthropicGateway
from theow._gateway._base import LLMGateway
from theow._gateway._gemini import GeminiGateway

logger = get_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


class Theow:
    """Theow inference engine."""

    def __init__(
        self,
        theow_dir: str = "./.theow",
        llm: str | None = None,
        llm_secondary: str | None = None,
        session_limit: int = 20,
        max_tool_calls: int = 30,
        max_tokens: int = 8192,
    ) -> None:
        self._theow_dir = Path(theow_dir)
        self._llm = llm
        self._llm_secondary = llm_secondary
        self._session_limit = session_limit
        self._max_tool_calls = max_tool_calls
        self._max_tokens = max_tokens

        self._gateway: LLMGateway | None = None
        self._secondary_gateway: LLMGateway | None = None

        self._setup_directories()
        self._setup_components()

    def _setup_directories(self) -> None:
        """Create .theow directory structure."""
        self._theow_dir.mkdir(parents=True, exist_ok=True)
        (self._theow_dir / "rules").mkdir(exist_ok=True)
        (self._theow_dir / "actions").mkdir(exist_ok=True)
        (self._theow_dir / "prompts").mkdir(exist_ok=True)
        (self._theow_dir / "chroma").mkdir(exist_ok=True)

    def _setup_components(self) -> None:
        """Initialize all internal components."""
        self._tool_registry = ToolRegistry()
        self._action_registry = ActionRegistry()

        set_standalone_registry(self._action_registry)

        self._chroma = ChromaStore(path=self._theow_dir / "chroma")

        self._resolver = Resolver(
            chroma=self._chroma,
            action_registry=self._action_registry,
            rules_dir=self._theow_dir / "rules",
        )

        # Explorer is created with lazy gateway
        self._explorer = Explorer(
            chroma=self._chroma,
            gateway=None,  # Lazy - set via property when needed
            action_registry=self._action_registry,
            rules_dir=self._theow_dir / "rules",
            session_limit=self._session_limit,
            max_tool_calls=self._max_tool_calls,
            max_tokens=self._max_tokens,
        )

        self._mark_decorator = MarkDecorator(
            resolver=self._resolver,
            explorer=self._explorer,
            tool_registry=self._tool_registry,
        )

        self._sync_on_startup()

    def _create_gateway(self, llm_spec: str) -> LLMGateway:
        """Create gateway from provider/model spec."""
        provider, model = llm_spec.split("/", 1)

        if provider == "gemini":
            return GeminiGateway(model=model)
        elif provider == "anthropic":
            return AnthropicGateway(model=model)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    def _sync_on_startup(self) -> None:
        """Sync rules and actions with Chroma on startup."""
        self._chroma.sync_rules(self._theow_dir / "rules")
        self._action_registry.discover(self._theow_dir / "actions")

        for name, meta in self._action_registry._metadata.items():
            self._chroma.index_action(name, meta["docstring"], meta["signature"])

        # Initialize gateway if LLM is configured
        self._ensure_gateway()

    def tool(self, name: str | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Register a tool for LLM exploration."""
        return self._tool_registry.register(name)

    def action(self, name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Register an action for rule execution."""
        return self._action_registry.register(name)

    def mark(
        self,
        context_from: Callable[..., dict[str, Any]],
        max_retries: int = 3,
        rules: list[str] | None = None,
        tags: list[str] | None = None,
        fallback: bool = True,
        explorable: bool = False,
        collection: str = "default",
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Mark a function as Theow-managed with automatic recovery."""
        return self._mark_decorator(
            context_from=context_from,
            max_retries=max_retries,
            rules=rules,
            tags=tags,
            fallback=fallback,
            explorable=explorable,
            collection=collection,
        )

    def resolve(
        self,
        context: dict[str, Any],
        collection: str = "default",
        rules: list[str] | None = None,
        tags: list[str] | None = None,
        fallback: bool = True,
    ) -> Rule | None:
        """Match context against rules directly."""
        return self._resolver.resolve(
            context=context,
            collection=collection,
            rules=rules,
            tags=tags,
            fallback=fallback,
        )

    def explore(
        self,
        context: dict[str, Any],
        tools: list[Callable[..., Any]],
        collection: str = "default",
        tracing: TracingInfo | None = None,
    ) -> Rule | None:
        """Explore a novel situation using LLM."""
        self._ensure_gateway()
        return self._explorer.explore(
            context=context,
            tools=tools,
            collection=collection,
            tracing=tracing,
            rejected_attempts=None,
        )

    def _ensure_gateway(self) -> None:
        """Lazily create LLM gateway when needed for exploration."""
        if self._gateway is not None:
            return

        if self._llm is None:
            logger.warning("Exploration disabled", reason="no LLM configured")
            return

        self._gateway = self._create_gateway(self._llm)
        self._explorer.set_gateway(self._gateway)

        if self._llm_secondary:
            self._secondary_gateway = self._create_gateway(self._llm_secondary)

    def meow(self) -> None:
        """Print stats. 🐱"""
        _meow(self._chroma)

    @property
    def theow_dir(self) -> Path:
        return self._theow_dir

    @property
    def session_count(self) -> int:
        return self._explorer.session_count

    def reset_session(self) -> None:
        """Reset session counter and cache."""
        self._explorer.reset_session()
