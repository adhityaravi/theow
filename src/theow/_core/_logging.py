"""Structured logging for Theow."""

import logging
import os
from typing import Any

import structlog

_CONFIGURED = False
_ENGINE_NAME = "Theow"

COMPONENT_MAP = {
    "theow._gateway._anthropic": "llm-gateway",
    "theow._gateway._gemini": "llm-gateway",
    "theow._gateway._copilot": "llm-gateway",
    "theow._core._explorer": "explorer",
    "theow._core._resolver": "resolver",
    "theow._core._decorators": "recovery",
    "theow._core._engine": "engine",
    "theow._core._chroma_store": "chroma",
}


def set_engine_name(name: str) -> None:
    """Set the engine name for log messages."""
    global _ENGINE_NAME
    _ENGINE_NAME = name


def get_engine_name() -> str:
    """Get the engine name for log messages."""
    return _ENGINE_NAME


def _ensure_configured() -> None:
    """Ensure minimal logging config exists if host app hasn't configured it."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    for noisy in ("httpx", "httpcore", "anthropic", "chromadb", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    # Only configure if host app hasn't set up structlog
    if not structlog.is_configured():
        level = os.getenv("THEOW_LOG_LEVEL", os.getenv("LOG_LEVEL", "INFO"))
        logging.basicConfig(
            format="%(message)s",
            level=getattr(logging, level.upper(), logging.INFO),
        )
        structlog.configure(
            processors=[
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.dev.ConsoleRenderer(colors=True),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=False,
        )


class ComponentLogger:
    """Logger wrapper that uses engine:component as logger name.

    The logger name is resolved dynamically at log time so that the engine name
    is always current (even if set_engine_name is called after logger creation).
    """

    def __init__(self, component: str) -> None:
        self._component = component

    def _get_logger(self) -> structlog.stdlib.BoundLogger:
        """Get logger with current engine name."""
        return structlog.get_logger(f"{_ENGINE_NAME}:{self._component}")

    def debug(self, event: str, **kw: Any) -> None:
        self._get_logger().debug(event, **kw)

    def info(self, event: str, **kw: Any) -> None:
        self._get_logger().info(event, **kw)

    def warning(self, event: str, **kw: Any) -> None:
        self._get_logger().warning(event, **kw)

    def error(self, event: str, **kw: Any) -> None:
        self._get_logger().error(event, **kw)


def get_logger(name: str | None = None) -> ComponentLogger:
    """Get a structured logger instance with component name.

    The logger name is formatted as 'engine:component' (e.g., 'SD-Agent:resolver')
    and resolved dynamically at log time.
    """
    _ensure_configured()
    logger_name = name or "theow"
    component = COMPONENT_MAP.get(logger_name, logger_name.split(".")[-1].strip("_"))
    return ComponentLogger(component)
