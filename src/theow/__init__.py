"""Theow: LLM-in-the-loop rule based expert system."""

import os

# Suppress logfire "not configured" warning for consumers that don't use logfire.
# Theow uses logfire.span() for optional observability but doesn't require the
# consumer to call logfire.configure().
os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")

from theow._core._engine import Theow
from theow._core._models import Action, Fact, Rule
from theow._core._decorators import action, tool
from theow._gateway._base import GatewayConfig, GatewayProvider, MiddlewareConfig
from theow._gateway._observability import LogfireConfig

__all__ = [
    "Theow",
    "Rule",
    "Fact",
    "Action",
    "action",
    "tool",
    "GatewayConfig",
    "GatewayProvider",
    "LogfireConfig",
    "MiddlewareConfig",
]
