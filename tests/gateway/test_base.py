"""Tests for gateway base class."""

import pytest

from theow._core._tools import GiveUp
from theow._gateway._base import (
    LLMGateway,
    SessionState,
    build_tool_declaration,
)


class _TestGateway(LLMGateway):
    def conversation(self, messages, tools, budget):
        pass

    def generate(self, prompt, schema=None):
        return {}


def test_build_tool_declaration_basic():
    def my_tool(path: str, count: int) -> str:
        """Read a file."""
        return ""

    decl = build_tool_declaration("my_tool", my_tool)
    assert decl["name"] == "my_tool"
    assert decl["description"] == "Read a file."
    assert decl["input_schema"]["properties"]["path"]["type"] == "string"
    assert decl["input_schema"]["properties"]["count"]["type"] == "integer"
    assert "path" in decl["input_schema"]["required"]
    assert "count" in decl["input_schema"]["required"]


def test_build_tool_declaration_optional_param():
    def tool(name: str, verbose: bool = False) -> None:
        """Do something."""

    decl = build_tool_declaration("tool", tool)
    assert "name" in decl["input_schema"]["required"]
    assert "verbose" not in decl["input_schema"]["required"]


def test_build_tool_declaration_custom_schema_key():
    def tool(x: str) -> None:
        """desc"""

    decl = build_tool_declaration("tool", tool, schema_key="parameters")
    assert "parameters" in decl
    assert "input_schema" not in decl


def test_check_budget_warning_at_soft_limit():
    gw = _TestGateway()
    state = SessionState(tool_calls=24)
    result = gw.check_budget_warning(state, max_calls=30)
    assert result is not None
    assert "6 tool calls remaining" in result
    assert state.warned_about_budget is True


def test_check_budget_warning_only_once():
    gw = _TestGateway()
    state = SessionState(tool_calls=25)
    first = gw.check_budget_warning(state, max_calls=30)
    assert first is not None
    second = gw.check_budget_warning(state, max_calls=30)
    assert second is None


def test_check_budget_warning_token_limit():
    gw = _TestGateway()
    state = SessionState(tool_calls=5, tokens_used=6800)
    result = gw.check_budget_warning(state, max_calls=30, max_tokens=8192)
    assert result is not None
    assert "tokens remaining" in result
    assert state.warned_about_budget is True


def test_execute_tool_success():
    gw = _TestGateway()
    tool_map = {"greet": lambda name: f"Hello {name}"}
    result, is_error = gw._execute_tool("greet", {"name": "World"}, tool_map)
    assert result == "Hello World"
    assert is_error is False


def test_execute_tool_exception():
    gw = _TestGateway()

    def bad():
        raise ValueError("boom")

    result, is_error = gw._execute_tool("bad", {}, {"bad": bad})
    assert is_error is True
    assert "boom" in result["error"]


def test_execute_tool_raises_signal():
    gw = _TestGateway()

    def signal_tool():
        raise GiveUp("nope")

    with pytest.raises(GiveUp):
        gw._execute_tool("signal_tool", {}, {"signal_tool": signal_tool})
