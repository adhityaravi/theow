"""Tests for Claude Agent SDK gateway."""

from unittest.mock import MagicMock, patch

from theow._core._tools import GiveUp, RequestTemplates


def _mock_claude_agent_sdk():
    """Patch claude_agent_sdk imports so tests run without the SDK installed."""
    mock_sdk = MagicMock()
    mock_sdk.AssistantMessage = type("AssistantMessage", (), {})
    mock_sdk.ClaudeAgentOptions = MagicMock()
    mock_sdk.ClaudeSDKClient = MagicMock()
    mock_sdk.ResultMessage = type("ResultMessage", (), {})
    mock_sdk.TextBlock = type("TextBlock", (), {})
    mock_sdk.create_sdk_mcp_server = MagicMock()
    mock_sdk.query = MagicMock()
    mock_sdk.tool = MagicMock(side_effect=lambda *a: lambda fn: fn)

    # StreamEvent is in claude_agent_sdk.types (not re-exported from __init__)
    mock_types = MagicMock()
    mock_types.StreamEvent = type("StreamEvent", (), {})

    # parse_message is in claude_agent_sdk._internal.message_parser
    mock_parser = MagicMock()
    mock_internal = MagicMock()
    mock_internal.message_parser = mock_parser

    return {
        "claude_agent_sdk": mock_sdk,
        "claude_agent_sdk.types": mock_types,
        "claude_agent_sdk._internal": mock_internal,
        "claude_agent_sdk._internal.message_parser": mock_parser,
    }


def test_claude_agent_gateway_get_user_prompt():
    with patch.dict("sys.modules", _mock_claude_agent_sdk()):
        from theow._gateway._claude_agent import ClaudeAgentGateway

        gateway = ClaudeAgentGateway()

        assert gateway._get_user_prompt([{"role": "user", "content": "Hello"}]) == "Hello"
        assert gateway._get_user_prompt([]) is None
        assert gateway._get_user_prompt([{"role": "assistant", "content": "Hi"}]) is None
        assert gateway._get_user_prompt([{"role": "user", "content": ["array"]}]) is None


def test_claude_agent_split_system_prompt():
    with patch.dict("sys.modules", _mock_claude_agent_sdk()):
        from theow._gateway._claude_agent import ClaudeAgentGateway

        # Normal case: INTRO + Error Context
        prompt = "You are a Theow agent.\n\n## Error Context\n\nSomething broke"
        system, user = ClaudeAgentGateway._split_system_prompt(prompt)
        assert system == "You are a Theow agent."
        assert user == "## Error Context\n\nSomething broke"

        # No marker (run_direct): fallback system prompt, user prompt unchanged
        system, user = ClaudeAgentGateway._split_system_prompt("Just a plain prompt")
        assert "mcp__theow___done" in system
        assert user == "Just a plain prompt"


def test_claude_agent_reset():
    with patch.dict("sys.modules", _mock_claude_agent_sdk()):
        from theow._gateway._claude_agent import ClaudeAgentGateway

        gateway = ClaudeAgentGateway()
        gateway._tool_map = {"test": lambda: None}
        gateway._state = object()
        gateway._signal = object()

        gateway.reset()

        assert gateway._tool_map == {}
        assert gateway._state is None
        assert gateway._signal is None
        assert gateway._client is None


def test_claude_agent_handle_tool_result_success():
    with patch.dict("sys.modules", _mock_claude_agent_sdk()):
        from theow._gateway._claude_agent import ClaudeAgentGateway
        from theow._gateway._observability import SessionState

        gateway = ClaudeAgentGateway()
        gateway._state = SessionState()
        gateway._max_calls = 30
        gateway._max_tokens = 0

        def greet(name: str) -> str:
            return f"Hello {name}"

        result = gateway._handle_tool_result(greet, {"name": "World"}, "greet")
        assert "Hello World" in result["content"][0]["text"]
        assert gateway._state.tool_calls == 1


def test_claude_agent_handle_tool_result_budget_exceeded():
    with patch.dict("sys.modules", _mock_claude_agent_sdk()):
        from theow._gateway._claude_agent import ClaudeAgentGateway
        from theow._gateway._observability import SessionState

        gateway = ClaudeAgentGateway()
        gateway._state = SessionState(tool_calls=31)
        gateway._max_calls = 30
        gateway._max_tokens = 0

        result = gateway._handle_tool_result(lambda: "ok", {}, "noop")
        assert "Session budget exceeded" in result["content"][0]["text"]


def test_claude_agent_handle_tool_result_token_budget_exceeded():
    with patch.dict("sys.modules", _mock_claude_agent_sdk()):
        from theow._gateway._claude_agent import ClaudeAgentGateway
        from theow._gateway._observability import SessionState

        gateway = ClaudeAgentGateway()
        gateway._state = SessionState(tool_calls=1, tokens_used=9000)
        gateway._max_calls = 30
        gateway._max_tokens = 8192

        result = gateway._handle_tool_result(lambda: "ok", {}, "noop")
        assert "Session budget exceeded" in result["content"][0]["text"]


def test_claude_agent_handle_tool_result_request_templates_with_config(tmp_path):
    with patch.dict("sys.modules", _mock_claude_agent_sdk()):
        from theow._gateway._claude_agent import ClaudeAgentGateway
        from theow._gateway._observability import SessionState

        gateway = ClaudeAgentGateway()
        gateway._state = SessionState()
        gateway._max_calls = 30
        gateway._max_tokens = 0
        gateway._gateway_config = {"rules_dir": tmp_path / "rules"}
        (tmp_path / "actions").mkdir()

        def request_templates():
            raise RequestTemplates()

        result = gateway._handle_tool_result(request_templates, {}, "request_templates")
        assert "Rule & Action Templates" in result["content"][0]["text"]
        assert gateway._signal is None


def test_claude_agent_handle_tool_result_request_templates_no_config():
    with patch.dict("sys.modules", _mock_claude_agent_sdk()):
        from theow._gateway._claude_agent import ClaudeAgentGateway
        from theow._gateway._observability import SessionState

        gateway = ClaudeAgentGateway()
        gateway._state = SessionState()
        gateway._max_calls = 30
        gateway._max_tokens = 0
        gateway._gateway_config = {}

        def request_templates():
            raise RequestTemplates()

        gateway._handle_tool_result(request_templates, {}, "request_templates")
        assert isinstance(gateway._signal, RequestTemplates)


def test_claude_agent_handle_tool_result_other_signal():
    with patch.dict("sys.modules", _mock_claude_agent_sdk()):
        from theow._gateway._claude_agent import ClaudeAgentGateway
        from theow._gateway._observability import SessionState

        gateway = ClaudeAgentGateway()
        gateway._state = SessionState()
        gateway._max_calls = 30
        gateway._max_tokens = 0

        def give_up(reason: str) -> None:
            raise GiveUp(reason)

        result = gateway._handle_tool_result(give_up, {"reason": "impossible"}, "give_up")
        assert isinstance(gateway._signal, GiveUp)
        assert "Signal:" in result["content"][0]["text"]


def test_claude_agent_handle_tool_result_exception():
    with patch.dict("sys.modules", _mock_claude_agent_sdk()):
        from theow._gateway._claude_agent import ClaudeAgentGateway
        from theow._gateway._observability import SessionState

        gateway = ClaudeAgentGateway()
        gateway._state = SessionState()
        gateway._max_calls = 30
        gateway._max_tokens = 0

        def bad_tool():
            raise ValueError("boom")

        result = gateway._handle_tool_result(bad_tool, {}, "bad_tool")
        assert "boom" in result["content"][0]["text"]


def test_claude_agent_handle_tool_result_budget_warning():
    with patch.dict("sys.modules", _mock_claude_agent_sdk()):
        from theow._gateway._claude_agent import ClaudeAgentGateway
        from theow._gateway._observability import SessionState

        gateway = ClaudeAgentGateway()
        gateway._state = SessionState(tool_calls=23)
        gateway._max_calls = 30
        gateway._max_tokens = 0

        result = gateway._handle_tool_result(lambda: "ok", {}, "noop")
        assert "tool calls remaining" in result["content"][0]["text"]


def test_claude_agent_reset_with_client_and_loop():
    with patch.dict("sys.modules", _mock_claude_agent_sdk()):
        from theow._gateway._claude_agent import ClaudeAgentGateway

        gateway = ClaudeAgentGateway()

        mock_loop = MagicMock()
        mock_loop.is_closed.return_value = False

        gateway._client = MagicMock()
        gateway._client_connected = True
        gateway._loop = mock_loop
        gateway._state = MagicMock()

        gateway.reset()

        mock_loop.close.assert_called()
        assert gateway._client is None
        assert gateway._client_connected is False
        assert gateway._loop is None
        assert gateway._state is None


def test_native_equivalent_tools_not_in_mcp_server():
    """Tools with native Claude Code equivalents are filtered out of MCP registration."""
    mocks = _mock_claude_agent_sdk()
    with patch.dict("sys.modules", mocks):
        from theow._gateway._claude_agent import ClaudeAgentGateway, _NATIVE_EQUIVALENT_TOOLS

        gateway = ClaudeAgentGateway()

        # Simulate a tool map with both native-equivalent and theow-specific tools
        gateway._tool_map = {
            "read_file": MagicMock(__doc__="Read a file", __annotations__={"path": str}),
            "write_file": MagicMock(__doc__="Write a file", __annotations__={"path": str}),
            "run_command": MagicMock(__doc__="Run a command", __annotations__={"cmd": str}),
            "list_directory": MagicMock(__doc__="List dir", __annotations__={"path": str}),
            "search_rules": MagicMock(__doc__="Search rules", __annotations__={"query": str}),
            "done": MagicMock(__doc__="Signal done", __annotations__={"message": str}),
        }

        gateway._build_mcp_server()

        # Check which tool names were passed to the @tool decorator
        registered = {call.args[0] for call in mocks["claude_agent_sdk"].tool.call_args_list}

        # Native equivalents should NOT be registered
        for native_tool in _NATIVE_EQUIVALENT_TOOLS:
            assert native_tool not in registered, f"{native_tool} should be filtered out"

        # Theow-specific tools SHOULD be registered
        assert "search_rules" in registered
        assert "done" in registered


# --- Usage tracker tests ---


def test_usage_tracker_stream_event_message_start():
    """message_start accumulates input_tokens + cache_creation_input_tokens."""
    from theow._gateway._observability import SessionState
    from theow._gateway._observability._claude_agent import ClaudeAgentUsageTracker

    tracker = ClaudeAgentUsageTracker()
    state = SessionState()
    tracker._state = state

    event = {
        "type": "message_start",
        "message": {"usage": {"input_tokens": 500, "cache_creation_input_tokens": 100}},
    }
    tracker.update_from_stream_event(event)

    assert state.input_tokens == 600
    assert state.output_tokens == 0
    assert state.tokens_used == 600


def test_usage_tracker_stream_event_message_delta():
    """message_delta accumulates output_tokens."""
    from theow._gateway._observability import SessionState
    from theow._gateway._observability._claude_agent import ClaudeAgentUsageTracker

    tracker = ClaudeAgentUsageTracker()
    state = SessionState()
    tracker._state = state

    event = {"type": "message_delta", "usage": {"output_tokens": 200}}
    tracker.update_from_stream_event(event)

    assert state.output_tokens == 200
    assert state.input_tokens == 0
    assert state.tokens_used == 200


def test_usage_tracker_stream_event_cross_turn_accumulation():
    """Tokens accumulate across multiple stream events (multiple turns)."""
    from theow._gateway._observability import SessionState
    from theow._gateway._observability._claude_agent import ClaudeAgentUsageTracker

    tracker = ClaudeAgentUsageTracker()
    state = SessionState()
    tracker._state = state

    # Turn 1
    tracker.update_from_stream_event(
        {"type": "message_start", "message": {"usage": {"input_tokens": 500}}}
    )
    tracker.update_from_stream_event({"type": "message_delta", "usage": {"output_tokens": 100}})

    # Turn 2
    tracker.update_from_stream_event(
        {"type": "message_start", "message": {"usage": {"input_tokens": 700}}}
    )
    tracker.update_from_stream_event({"type": "message_delta", "usage": {"output_tokens": 150}})

    assert state.input_tokens == 1200
    assert state.output_tokens == 250
    assert state.tokens_used == 1450


def test_usage_tracker_stream_event_no_state():
    """Gracefully handles missing state on stream events."""
    from theow._gateway._observability._claude_agent import ClaudeAgentUsageTracker

    tracker = ClaudeAgentUsageTracker()
    # No state set — should not raise
    tracker.update_from_stream_event(
        {"type": "message_start", "message": {"usage": {"input_tokens": 500}}}
    )


def test_usage_tracker_result_cost_only():
    """ResultMessage.total_cost_usd updates cost but not tokens."""
    from theow._gateway._observability import SessionState
    from theow._gateway._observability._claude_agent import ClaudeAgentUsageTracker

    tracker = ClaudeAgentUsageTracker()
    state = SessionState(input_tokens=500, output_tokens=200, tokens_used=700)
    tracker._state = state

    result = MagicMock()
    result.total_cost_usd = 0.05

    tracker.update_from_result(result)

    assert tracker.total_cost_usd == 0.05
    # Tokens unchanged — result doesn't touch them
    assert state.input_tokens == 500
    assert state.output_tokens == 200
    assert state.tokens_used == 700


def test_usage_tracker_result_no_state():
    """Gracefully handles missing state on result."""
    from theow._gateway._observability._claude_agent import ClaudeAgentUsageTracker

    tracker = ClaudeAgentUsageTracker()

    result = MagicMock()
    result.total_cost_usd = 0.03

    # Should not raise
    tracker.update_from_result(result)
    assert tracker.total_cost_usd == 0.03


def test_usage_tracker_result_no_cost():
    """Handles ResultMessage without total_cost_usd field."""
    from theow._gateway._observability._claude_agent import ClaudeAgentUsageTracker

    tracker = ClaudeAgentUsageTracker()

    result = MagicMock(spec=[])  # no attributes
    tracker.update_from_result(result)

    assert tracker.total_cost_usd == 0.0
