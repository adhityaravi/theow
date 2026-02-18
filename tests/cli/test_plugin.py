"""Tests for plugin loading."""

import textwrap

import pytest

from theow._cli._plugin import load_plugin
from theow._core._decorators import ToolRegistry


def test_load_plugin_registers_tools(tmp_path):
    plugin = tmp_path / "plugin.py"
    plugin.write_text(
        textwrap.dedent("""\
        from theow import tool

        @tool("greet")
        def greet(name: str) -> str:
            return f"hello {name}"
    """)
    )

    registry = ToolRegistry()
    result = load_plugin(plugin, registry)

    assert "greet" in registry._tools
    assert registry._tools["greet"]("world") == "hello world"
    assert len(result.tools) == 1


def test_load_plugin_extracts_setup_teardown(tmp_path):
    plugin = tmp_path / "plugin.py"
    plugin.write_text(
        textwrap.dedent("""\
        def setup(state, attempt):
            state["ran"] = True
            return state

        def teardown(state, attempt, success):
            pass
    """)
    )

    registry = ToolRegistry()
    result = load_plugin(plugin, registry)

    assert result.setup is not None
    assert result.teardown is not None
    assert result.setup({}, 1) == {"ran": True}


def test_load_plugin_no_hooks(tmp_path):
    plugin = tmp_path / "plugin.py"
    plugin.write_text("x = 1\n")

    registry = ToolRegistry()
    result = load_plugin(plugin, registry)

    assert result.setup is None
    assert result.teardown is None
    assert result.tools == []


def test_load_plugin_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_plugin(tmp_path / "nope.py", ToolRegistry())


def test_load_plugin_cleans_up_registry_on_error(tmp_path):
    plugin = tmp_path / "bad.py"
    plugin.write_text("raise RuntimeError('boom')\n")

    registry = ToolRegistry()
    with pytest.raises(RuntimeError, match="boom"):
        load_plugin(plugin, registry)

    from theow._core._decorators import _standalone_tool_registry

    assert _standalone_tool_registry is None
