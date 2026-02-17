"""Tests for Copilot gateway."""

from unittest.mock import MagicMock, patch

from theow._core._tools import GiveUp, RequestTemplates


def test_copilot_gateway_get_user_prompt():
    with patch.dict("os.environ", {"GITHUB_TOKEN": "test-token"}):
        from theow._gateway._copilot import CopilotGateway

        gateway = CopilotGateway()

        assert gateway._get_user_prompt([{"role": "user", "content": "Hello"}]) == "Hello"
        assert gateway._get_user_prompt([]) is None
        assert gateway._get_user_prompt([{"role": "assistant", "content": "Hi"}]) is None
        assert gateway._get_user_prompt([{"role": "user", "content": ["array"]}]) is None


def test_copilot_gateway_reset():
    with patch.dict("os.environ", {"GITHUB_TOKEN": "test-token"}):
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
        from theow._gateway._copilot import CopilotGateway
        from theow._gateway._base import SessionState

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
        from theow._gateway._copilot import CopilotGateway
        from theow._gateway._base import SessionState

        gateway = CopilotGateway()
        gateway._state = SessionState(tool_calls=31)
        gateway._max_calls = 30
        gateway._max_tokens = 0

        invocation = {"arguments": {}, "tool_name": "noop"}
        result = gateway._handle_tool_call(lambda: "ok", invocation)
        assert "Session budget exceeded" in result["textResultForLlm"]


def test_copilot_handle_tool_call_token_budget_exceeded():
    with patch.dict("os.environ", {"GITHUB_TOKEN": "test-token"}):
        from theow._gateway._copilot import CopilotGateway
        from theow._gateway._base import SessionState

        gateway = CopilotGateway()
        gateway._state = SessionState(tool_calls=1, tokens_used=9000)
        gateway._max_calls = 30
        gateway._max_tokens = 8192

        invocation = {"arguments": {}, "tool_name": "noop"}
        result = gateway._handle_tool_call(lambda: "ok", invocation)
        assert "Session budget exceeded" in result["textResultForLlm"]


def test_copilot_handle_tool_call_estimates_tokens():
    with patch.dict("os.environ", {"GITHUB_TOKEN": "test-token"}):
        from theow._gateway._copilot import CopilotGateway
        from theow._gateway._base import SessionState

        gateway = CopilotGateway()
        gateway._state = SessionState()
        gateway._max_calls = 30
        gateway._max_tokens = 100000

        invocation = {"arguments": {"data": "x" * 400}, "tool_name": "echo"}
        gateway._handle_tool_call(lambda data: data, invocation)
        assert gateway._state.tokens_used > 0


def test_copilot_handle_tool_call_request_templates_with_config(tmp_path):
    with patch.dict("os.environ", {"GITHUB_TOKEN": "test-token"}):
        from theow._gateway._copilot import CopilotGateway
        from theow._gateway._base import SessionState

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
        from theow._gateway._copilot import CopilotGateway
        from theow._gateway._base import SessionState

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
        from theow._gateway._copilot import CopilotGateway
        from theow._gateway._base import SessionState

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
        from theow._gateway._copilot import CopilotGateway
        from theow._gateway._base import SessionState

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
        from theow._gateway._copilot import CopilotGateway
        from theow._gateway._base import SessionState

        gateway = CopilotGateway()
        gateway._state = SessionState(tool_calls=23)
        gateway._max_calls = 30
        gateway._max_tokens = 0

        invocation = {"arguments": {}, "tool_name": "noop"}
        result = gateway._handle_tool_call(lambda: "ok", invocation)
        assert "tool calls remaining" in result["textResultForLlm"]


def test_copilot_reset_with_session_and_loop():
    with patch.dict("os.environ", {"GITHUB_TOKEN": "test-token"}):
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
