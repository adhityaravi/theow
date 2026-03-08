"""Tests for Copilot gateway."""

from unittest.mock import MagicMock, patch

from theow._core._tools import GiveUp, RequestTemplates


def _mock_copilot_sdk():
    """Patch copilot SDK imports so tests run without the SDK installed."""
    mock_copilot = MagicMock()
    mock_copilot.__file__ = "/fake/copilot/__init__.py"
    mock_copilot.CopilotClient = MagicMock()
    mock_copilot.PermissionHandler = MagicMock()
    mock_copilot.Tool = MagicMock()
    mock_types = MagicMock()
    mock_types.ToolInvocation = dict
    mock_types.ToolResult = dict
    return {"copilot": mock_copilot, "copilot.types": mock_types}


def test_copilot_gateway_get_user_prompt():
    with patch.dict("os.environ", {"GITHUB_TOKEN": "test-token"}):
        with patch.dict("sys.modules", _mock_copilot_sdk()):
            from theow._gateway._copilot import CopilotGateway

            gateway = CopilotGateway()

            assert gateway._get_user_prompt([{"role": "user", "content": "Hello"}]) == "Hello"
            assert gateway._get_user_prompt([]) is None
            assert gateway._get_user_prompt([{"role": "assistant", "content": "Hi"}]) is None
            assert gateway._get_user_prompt([{"role": "user", "content": ["array"]}]) is None


def test_copilot_gateway_reset():
    with patch.dict("os.environ", {"GITHUB_TOKEN": "test-token"}):
        with patch.dict("sys.modules", _mock_copilot_sdk()):
            from theow._gateway._copilot import CopilotGateway

            gateway = CopilotGateway()
            gateway._tool_map = {"test": lambda: None}
            gateway._state = object()
            gateway._signal = object()

            gateway.reset()

            assert gateway._tool_map == {}
            assert gateway._state is None
            assert gateway._signal is None
            assert gateway._client is None


def test_copilot_handle_tool_call_success():
    with patch.dict("os.environ", {"GITHUB_TOKEN": "test-token"}):
        with patch.dict("sys.modules", _mock_copilot_sdk()):
            from theow._gateway._copilot import CopilotGateway
            from theow._gateway._observability import SessionState

            gateway = CopilotGateway()
            gateway._state = SessionState()
            gateway._max_calls = 30
            gateway._max_tokens = 0

            def greet(name: str) -> str:
                return f"Hello {name}"

            invocation = {"arguments": {"name": "World"}, "tool_name": "greet"}
            result = gateway._handle_tool_call(greet, invocation)
            assert "Hello World" in result["textResultForLlm"]
            assert gateway._state.tool_calls == 1


def test_copilot_handle_tool_call_budget_exceeded():
    with patch.dict("os.environ", {"GITHUB_TOKEN": "test-token"}):
        with patch.dict("sys.modules", _mock_copilot_sdk()):
            from theow._gateway._copilot import CopilotGateway
            from theow._gateway._observability import SessionState

            gateway = CopilotGateway()
            gateway._state = SessionState(tool_calls=31)
            gateway._max_calls = 30
            gateway._max_tokens = 0

            invocation = {"arguments": {}, "tool_name": "noop"}
            result = gateway._handle_tool_call(lambda: "ok", invocation)
            assert "Session budget exceeded" in result["textResultForLlm"]


def test_copilot_handle_tool_call_token_budget_exceeded():
    with patch.dict("os.environ", {"GITHUB_TOKEN": "test-token"}):
        with patch.dict("sys.modules", _mock_copilot_sdk()):
            from theow._gateway._copilot import CopilotGateway
            from theow._gateway._observability import SessionState

            gateway = CopilotGateway()
            gateway._state = SessionState(tool_calls=1, tokens_used=9000)
            gateway._max_calls = 30
            gateway._max_tokens = 8192

            invocation = {"arguments": {}, "tool_name": "noop"}
            result = gateway._handle_tool_call(lambda: "ok", invocation)
            assert "Session budget exceeded" in result["textResultForLlm"]


def test_copilot_handle_tool_call_does_not_estimate_tokens():
    """Token tracking comes from SDK ASSISTANT_USAGE events, not from handle_tool_call."""
    with patch.dict("os.environ", {"GITHUB_TOKEN": "test-token"}):
        with patch.dict("sys.modules", _mock_copilot_sdk()):
            from theow._gateway._copilot import CopilotGateway
            from theow._gateway._observability import SessionState

            gateway = CopilotGateway()
            gateway._state = SessionState()
            gateway._max_calls = 30
            gateway._max_tokens = 100000

            invocation = {"arguments": {"data": "x" * 400}, "tool_name": "echo"}
            gateway._handle_tool_call(lambda data: data, invocation)
            assert gateway._state.tokens_used == 0
            assert gateway._state.tool_calls == 1


def test_copilot_handle_tool_call_request_templates_with_config(tmp_path):
    with patch.dict("os.environ", {"GITHUB_TOKEN": "test-token"}):
        with patch.dict("sys.modules", _mock_copilot_sdk()):
            from theow._gateway._copilot import CopilotGateway
            from theow._gateway._observability import SessionState

            gateway = CopilotGateway()
            gateway._state = SessionState()
            gateway._max_calls = 30
            gateway._max_tokens = 0
            gateway._gateway_config = {"rules_dir": tmp_path / "rules"}
            (tmp_path / "actions").mkdir()

            def request_templates():
                raise RequestTemplates()

            invocation = {"arguments": {}, "tool_name": "request_templates"}
            result = gateway._handle_tool_call(request_templates, invocation)
            assert "Rule & Action Templates" in result["textResultForLlm"]
            assert gateway._signal is None


def test_copilot_handle_tool_call_request_templates_no_config():
    with patch.dict("os.environ", {"GITHUB_TOKEN": "test-token"}):
        with patch.dict("sys.modules", _mock_copilot_sdk()):
            from theow._gateway._copilot import CopilotGateway
            from theow._gateway._observability import SessionState

            gateway = CopilotGateway()
            gateway._state = SessionState()
            gateway._max_calls = 30
            gateway._max_tokens = 0
            gateway._gateway_config = {}

            def request_templates():
                raise RequestTemplates()

            invocation = {"arguments": {}, "tool_name": "request_templates"}
            gateway._handle_tool_call(request_templates, invocation)
            assert isinstance(gateway._signal, RequestTemplates)


def test_copilot_handle_tool_call_other_signal():
    with patch.dict("os.environ", {"GITHUB_TOKEN": "test-token"}):
        with patch.dict("sys.modules", _mock_copilot_sdk()):
            from theow._gateway._copilot import CopilotGateway
            from theow._gateway._observability import SessionState

            gateway = CopilotGateway()
            gateway._state = SessionState()
            gateway._max_calls = 30
            gateway._max_tokens = 0

            def give_up(reason: str) -> None:
                raise GiveUp(reason)

            invocation = {"arguments": {"reason": "impossible"}, "tool_name": "give_up"}
            result = gateway._handle_tool_call(give_up, invocation)
            assert isinstance(gateway._signal, GiveUp)
            assert "Signal:" in result["textResultForLlm"]


def test_copilot_handle_tool_call_exception():
    with patch.dict("os.environ", {"GITHUB_TOKEN": "test-token"}):
        with patch.dict("sys.modules", _mock_copilot_sdk()):
            from theow._gateway._copilot import CopilotGateway
            from theow._gateway._observability import SessionState

            gateway = CopilotGateway()
            gateway._state = SessionState()
            gateway._max_calls = 30
            gateway._max_tokens = 0

            def bad_tool():
                raise ValueError("boom")

            invocation = {"arguments": {}, "tool_name": "bad_tool"}
            result = gateway._handle_tool_call(bad_tool, invocation)
            assert "boom" in result["textResultForLlm"]
            assert result["resultType"] == "failure"


def test_copilot_handle_tool_call_budget_warning():
    with patch.dict("os.environ", {"GITHUB_TOKEN": "test-token"}):
        with patch.dict("sys.modules", _mock_copilot_sdk()):
            from theow._gateway._copilot import CopilotGateway
            from theow._gateway._observability import SessionState

            gateway = CopilotGateway()
            gateway._state = SessionState(tool_calls=23)
            gateway._max_calls = 30
            gateway._max_tokens = 0

            invocation = {"arguments": {}, "tool_name": "noop"}
            result = gateway._handle_tool_call(lambda: "ok", invocation)
            assert "tool calls remaining" in result["textResultForLlm"]


def test_copilot_reset_with_session_and_loop():
    with patch.dict("os.environ", {"GITHUB_TOKEN": "test-token"}):
        with patch.dict("sys.modules", _mock_copilot_sdk()):
            from theow._gateway._copilot import CopilotGateway

            gateway = CopilotGateway()

            mock_session = MagicMock()
            mock_loop = MagicMock()
            mock_loop.is_closed.return_value = False
            mock_loop.run_until_complete.return_value = None

            gateway._session = mock_session
            gateway._client = MagicMock()
            gateway._loop = mock_loop
            gateway._state = MagicMock()

            gateway.reset()

            mock_loop.run_until_complete.assert_called()
            assert gateway._session is None
            assert gateway._client is None
            assert gateway._loop is None
            assert gateway._state is None


# --- CopilotUsageTracker tests ---


def test_copilot_usage_tracker_session_premium_requests():
    """Premium requests delta is computed from initial and current used_requests."""
    from theow._gateway._observability._copilot import CopilotUsageTracker

    tracker = CopilotUsageTracker()
    assert tracker.session_premium_requests == 0.0

    tracker._initial_premium_used = 10.0
    tracker.quota_used_requests = 13.0
    assert tracker.session_premium_requests == 3.0


def test_copilot_usage_tracker_cost_usd():
    """Cost is premium requests * overage rate."""
    from theow._gateway._observability._copilot import CopilotUsageTracker

    tracker = CopilotUsageTracker()
    tracker._initial_premium_used = 10.0
    tracker.quota_used_requests = 12.0
    assert tracker.cost_usd == 2.0 * 0.04


def test_copilot_usage_tracker_handle_event_tokens():
    """ASSISTANT_USAGE events accumulate token counts on state."""
    from theow._gateway._observability import SessionState
    from theow._gateway._observability._copilot import CopilotUsageTracker

    # Mock the SessionEventType enum
    mock_event_type = MagicMock()
    mock_event_type.ASSISTANT_USAGE = "ASSISTANT_USAGE"
    mock_session_events = MagicMock()
    mock_session_events.SessionEventType = mock_event_type

    with patch.dict(
        "sys.modules",
        {"copilot.generated.session_events": mock_session_events},
    ):
        tracker = CopilotUsageTracker()
        state = SessionState()
        tracker._state = state

        event = MagicMock()
        event.type = "ASSISTANT_USAGE"
        event.data = MagicMock()
        event.data.input_tokens = 500
        event.data.output_tokens = 200
        event.data.quota_snapshots = None

        tracker.handle_event(event)

        assert state.input_tokens == 500
        assert state.output_tokens == 200
        assert state.tokens_used == 700


def test_copilot_usage_tracker_handle_event_quota():
    """ASSISTANT_USAGE events track quota snapshots."""
    from theow._gateway._observability._copilot import CopilotUsageTracker

    mock_event_type = MagicMock()
    mock_event_type.ASSISTANT_USAGE = "ASSISTANT_USAGE"
    mock_session_events = MagicMock()
    mock_session_events.SessionEventType = mock_event_type

    with patch.dict(
        "sys.modules",
        {"copilot.generated.session_events": mock_session_events},
    ):
        tracker = CopilotUsageTracker()

        premium = MagicMock()
        premium.used_requests = 10.0
        premium.remaining_percentage = 90.0

        event = MagicMock()
        event.type = "ASSISTANT_USAGE"
        event.data = MagicMock()
        event.data.input_tokens = 0
        event.data.output_tokens = 0
        event.data.quota_snapshots = {"premium_interactions": premium}

        tracker.handle_event(event)

        assert tracker._initial_premium_used == 10.0
        assert tracker.quota_remaining_pct == 90.0
        assert tracker.quota_used_requests == 10.0


def test_copilot_usage_tracker_handle_event_wrong_type():
    """Non-ASSISTANT_USAGE events are ignored."""
    from theow._gateway._observability._copilot import CopilotUsageTracker

    mock_event_type = MagicMock()
    mock_event_type.ASSISTANT_USAGE = "ASSISTANT_USAGE"
    mock_session_events = MagicMock()
    mock_session_events.SessionEventType = mock_event_type

    with patch.dict(
        "sys.modules",
        {"copilot.generated.session_events": mock_session_events},
    ):
        tracker = CopilotUsageTracker()

        event = MagicMock()
        event.type = "OTHER_EVENT"
        tracker.handle_event(event)

        assert tracker.quota_used_requests is None


def test_copilot_usage_tracker_handle_event_no_data():
    """Events with no data are ignored."""
    from theow._gateway._observability._copilot import CopilotUsageTracker

    mock_event_type = MagicMock()
    mock_event_type.ASSISTANT_USAGE = "ASSISTANT_USAGE"
    mock_session_events = MagicMock()
    mock_session_events.SessionEventType = mock_event_type

    with patch.dict(
        "sys.modules",
        {"copilot.generated.session_events": mock_session_events},
    ):
        tracker = CopilotUsageTracker()

        event = MagicMock()
        event.type = "ASSISTANT_USAGE"
        event.data = None
        tracker.handle_event(event)

        assert tracker.quota_used_requests is None


def test_copilot_usage_tracker_enrich_span():
    """enrich_span sets expected attributes."""
    from theow._gateway._observability._copilot import CopilotUsageTracker

    tracker = CopilotUsageTracker()
    tracker._initial_premium_used = 5.0
    tracker.quota_used_requests = 8.0
    tracker.quota_remaining_pct = 85.0

    span = MagicMock()
    tracker.enrich_span(span, "gpt-4", "copilot")

    span.set_attribute.assert_any_call("operation.cost", 3.0 * 0.04)
    span.set_attribute.assert_any_call("copilot.premium_requests", 3.0)
    span.set_attribute.assert_any_call("copilot.quota.remaining_pct", 85.0)
    span.set_attribute.assert_any_call("copilot.quota.used_requests", 8.0)


def test_copilot_usage_tracker_reset():
    """reset clears all per-session state."""
    from theow._gateway._observability import SessionState
    from theow._gateway._observability._copilot import CopilotUsageTracker

    tracker = CopilotUsageTracker()
    tracker._state = SessionState()
    tracker._initial_premium_used = 5.0
    tracker.quota_used_requests = 8.0
    tracker.quota_remaining_pct = 85.0

    tracker.reset()

    assert tracker._initial_premium_used is None
    assert tracker.quota_remaining_pct is None
    assert tracker.quota_used_requests is None
    assert tracker._state is None
