"""Tests for decorators."""

from theow._core._decorators import ActionRegistry, ToolRegistry


def test_tool_registry_register():
    registry = ToolRegistry()

    @registry.register()
    def my_tool(x: str) -> str:
        return x

    assert registry.get("my_tool") is my_tool


def test_tool_registry_custom_name():
    registry = ToolRegistry()

    @registry.register("custom_name")
    def my_tool(x: str) -> str:
        return x

    assert registry.get("custom_name") is my_tool
    assert registry.get("my_tool") is None


def test_tool_registry_get_declarations():
    registry = ToolRegistry()

    @registry.register()
    def read_file(path: str) -> str:
        """Read a file."""
        return ""

    declarations = registry.get_declarations()
    assert len(declarations) == 1
    assert declarations[0]["name"] == "read_file"
    assert declarations[0]["description"] == "Read a file."
    assert "path" in declarations[0]["parameters"]["properties"]


def test_action_registry_register_and_call():
    registry = ActionRegistry()

    @registry.register("fix_thing")
    def fix_thing(workspace: str) -> dict:
        return {"fixed": workspace}

    result = registry.call("fix_thing", {"workspace": "/tmp"})
    assert result == {"fixed": "/tmp"}


def test_action_registry_exists():
    registry = ActionRegistry()

    @registry.register("my_action")
    def my_action():
        pass

    assert registry.exists("my_action")
    assert not registry.exists("other")


def test_action_registry_get_metadata():
    registry = ActionRegistry()

    @registry.register("my_action")
    def my_action(x: str) -> str:
        """Do something."""
        return x

    meta = registry.get_metadata("my_action")
    assert meta["docstring"] == "Do something."
    assert "x: str" in meta["signature"]
