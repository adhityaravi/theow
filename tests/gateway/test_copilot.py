"""Tests for Copilot gateway."""

from unittest.mock import patch


def test_copilot_gateway_requires_token():
    with patch.dict("os.environ", {}, clear=True):
        from theow._gateway._copilot import CopilotGateway

        # No exception on init - token check happens later
        gateway = CopilotGateway()
        assert gateway._model == "claude-sonnet-4"


def test_copilot_gateway_custom_model():
    with patch.dict("os.environ", {"GITHUB_TOKEN": "test-token"}):
        from theow._gateway._copilot import CopilotGateway

        gateway = CopilotGateway(model="gpt-4")
        assert gateway._model == "gpt-4"


def test_copilot_gateway_get_user_prompt():
    with patch.dict("os.environ", {"GITHUB_TOKEN": "test-token"}):
        from theow._gateway._copilot import CopilotGateway

        gateway = CopilotGateway()

        # Valid user message
        messages = [{"role": "user", "content": "Hello"}]
        assert gateway._get_user_prompt(messages) == "Hello"

        # Empty messages
        assert gateway._get_user_prompt([]) is None

        # Last message not user
        messages = [{"role": "assistant", "content": "Hi"}]
        assert gateway._get_user_prompt(messages) is None

        # Non-string content
        messages = [{"role": "user", "content": ["array"]}]
        assert gateway._get_user_prompt(messages) is None


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
