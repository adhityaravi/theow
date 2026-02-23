"""Tests for Gemini gateway."""

from unittest.mock import MagicMock, patch

import pytest

from theow._core._tools import GiveUp


def test_gemini_gateway_requires_api_key():
    with patch.dict("os.environ", {}, clear=True):
        from theow._gateway._gemini import GeminiGateway

        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            GeminiGateway()


def test_gemini_gateway_conversation_no_candidates():
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        with patch("theow._gateway._gemini.genai") as mock_genai:
            from theow._gateway._gemini import GeminiGateway

            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client

            mock_response = MagicMock()
            mock_response.candidates = []
            mock_response.usage_metadata = None
            mock_client.models.generate_content.return_value = mock_response

            gateway = GeminiGateway(model="gemini-2.0-flash")
            result = gateway.conversation(
                messages=[{"role": "user", "content": "test"}],
                tools=[],
                budget={"max_tool_calls_per_session": 5},
            )

            assert result.tool_calls == 0


def test_gemini_gateway_generate():
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        with patch("theow._gateway._gemini.genai") as mock_genai:
            from theow._gateway._gemini import GeminiGateway

            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client

            mock_response = MagicMock()
            mock_response.text = '{"result": "ok"}'
            mock_client.models.generate_content.return_value = mock_response

            gateway = GeminiGateway(model="gemini-2.0-flash")
            result = gateway.generate(prompt="test")

            assert result == {"result": "ok"}


def test_gemini_conversation_with_function_call():
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        with patch("theow._gateway._gemini.genai") as mock_genai:
            from theow._gateway._gemini import GeminiGateway

            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client

            fc_part = MagicMock()
            fc_part.function_call = MagicMock()
            fc_part.function_call.name = "greet"
            fc_part.function_call.args = {"name": "World"}
            fc_part.text = None

            candidate1 = MagicMock()
            candidate1.content = MagicMock()
            candidate1.content.parts = [fc_part]

            resp1 = MagicMock()
            resp1.candidates = [candidate1]
            resp1.usage_metadata = MagicMock()
            resp1.usage_metadata.total_token_count = 50
            resp1.usage_metadata.candidates_token_count = 20

            text_part = MagicMock()
            text_part.function_call = None
            text_part.text = "Done"

            candidate2 = MagicMock()
            candidate2.content = MagicMock()
            candidate2.content.parts = [text_part]

            resp2 = MagicMock()
            resp2.candidates = [candidate2]
            resp2.usage_metadata = MagicMock()
            resp2.usage_metadata.total_token_count = 30
            resp2.usage_metadata.candidates_token_count = 10

            # tool call, then text reply + 2 nudge retries (all text)
            mock_client.models.generate_content.side_effect = [resp1, resp2, resp2, resp2]

            def greet(name: str) -> str:
                """Greet someone."""
                return f"Hello {name}"

            gateway = GeminiGateway(model="gemini-2.0-flash")
            messages = [{"role": "user", "content": "greet the world"}]
            result = gateway.conversation(
                messages=messages,
                tools=[greet],
                budget={"max_tool_calls_per_session": 10},
            )

            assert result.tool_calls == 1
            # 1 tool resp (50 tokens) + 3 text responses (30 tokens each)
            assert result.tokens_used == 50 + 30 * 3


def test_gemini_conversation_signal():
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        with patch("theow._gateway._gemini.genai") as mock_genai:
            from theow._gateway._gemini import GeminiGateway

            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client

            fc_part = MagicMock()
            fc_part.function_call = MagicMock()
            fc_part.function_call.name = "give_up"
            fc_part.function_call.args = {"reason": "impossible"}

            candidate = MagicMock()
            candidate.content = MagicMock()
            candidate.content.parts = [fc_part]

            resp = MagicMock()
            resp.candidates = [candidate]
            resp.usage_metadata = MagicMock()
            resp.usage_metadata.total_token_count = 20
            resp.usage_metadata.candidates_token_count = 10

            mock_client.models.generate_content.return_value = resp

            def give_up(reason: str) -> None:
                raise GiveUp(reason)

            gateway = GeminiGateway(model="gemini-2.0-flash")
            with pytest.raises(GiveUp):
                gateway.conversation(
                    messages=[{"role": "user", "content": "test"}],
                    tools=[give_up],
                    budget={"max_tool_calls_per_session": 10},
                )


def test_gemini_init_history_empty():
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        with patch("theow._gateway._gemini.genai") as mock_genai:
            from theow._gateway._gemini import GeminiGateway

            mock_genai.Client.return_value = MagicMock()
            gateway = GeminiGateway(model="gemini-2.0-flash")

            messages = [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "response"},
                {"role": "user", "content": "second"},
            ]
            gateway._init_history(messages)
            assert len(gateway._history) == 2


def test_gemini_generate_with_schema():
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        with patch("theow._gateway._gemini.genai") as mock_genai:
            from theow._gateway._gemini import GeminiGateway

            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client

            mock_response = MagicMock()
            mock_response.text = '{"key": "value"}'
            mock_client.models.generate_content.return_value = mock_response

            gateway = GeminiGateway(model="gemini-2.0-flash")
            result = gateway.generate(
                prompt="test",
                schema={"type": "object", "properties": {"key": {"type": "string"}}},
            )
            assert result == {"key": "value"}


def test_gemini_sync_messages():
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        with patch("theow._gateway._gemini.genai") as mock_genai:
            from theow._gateway._gemini import GeminiGateway

            mock_genai.Client.return_value = MagicMock()
            gateway = GeminiGateway(model="gemini-2.0-flash")

            user_part = MagicMock()
            user_part.text = "hello"
            user_content = MagicMock()
            user_content.role = "user"
            user_content.parts = [user_part]

            model_part = MagicMock()
            model_part.text = "response"
            model_content = MagicMock()
            model_content.role = "model"
            model_content.parts = [model_part]

            gateway._history = [user_content, model_content]

            messages = [{"old": "data"}]
            gateway._sync_messages(messages)

            assert len(messages) == 2
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "hello"
            assert messages[1]["role"] == "assistant"


def test_gemini_conversation_no_user_content():
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        with patch("theow._gateway._gemini.genai") as mock_genai:
            from theow._gateway._gemini import GeminiGateway

            mock_genai.Client.return_value = MagicMock()
            gateway = GeminiGateway(model="gemini-2.0-flash")

            result = gateway.conversation(
                messages=[{"role": "assistant", "content": "test"}],
                tools=[],
                budget={"max_tool_calls_per_session": 5},
            )
            assert result.tool_calls == 0
