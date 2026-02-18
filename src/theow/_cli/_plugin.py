"""Plugin loader for CLI recovery hooks and tools."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from theow._core._decorators import ToolRegistry, set_standalone_tool_registry
from theow._core._logging import get_logger

logger = get_logger(__name__)


@dataclass
class PluginResult:
    """Extracted plugin components."""

    tools: list[Callable[..., Any]] = field(default_factory=list)
    setup: Callable[[dict[str, Any], int], dict[str, Any] | None] | None = None
    teardown: Callable[[dict[str, Any], int, bool], None] | None = None


def load_plugin(path: Path, tool_registry: ToolRegistry) -> PluginResult:
    """Load a plugin file, extracting tools, setup, and teardown.

    Tools are registered via the standalone @tool decorator.
    setup/teardown are extracted by name from the module.
    """
    if not path.exists():
        raise FileNotFoundError(f"Plugin not found: {path}")

    before_tools = set(tool_registry._tools.keys())

    set_standalone_tool_registry(tool_registry)
    try:
        spec = importlib.util.spec_from_file_location("theow_plugin", path)
        if not spec or not spec.loader:
            raise ImportError(f"Cannot load plugin: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        set_standalone_tool_registry(None)

    new_tools = [fn for name, fn in tool_registry._tools.items() if name not in before_tools]

    return PluginResult(
        tools=new_tools,
        setup=getattr(module, "setup", None),
        teardown=getattr(module, "teardown", None),
    )
