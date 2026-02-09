"""Structured logging for Theow."""

import logging
import os

import structlog

# Library-friendly default: respect host app's config, but ensure basic setup
_CONFIGURED = False


def _ensure_configured() -> None:
    """Ensure minimal logging config exists if host app hasn't configured it."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    # Only configure if structlog hasn't been configured yet
    if not structlog.is_configured():
        level = os.getenv("THEOW_LOG_LEVEL", os.getenv("LOG_LEVEL", "INFO"))
        logging.basicConfig(
            format="%(message)s",
            level=getattr(logging, level.upper(), logging.INFO),
        )

    # Suppress noisy SDK loggers (they dump full request payloads at DEBUG)
    for noisy in ("httpx", "httpcore", "anthropic", "onnxruntime"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
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


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        A structlog BoundLogger instance
    """
    _ensure_configured()
    return structlog.get_logger(name or "theow")
