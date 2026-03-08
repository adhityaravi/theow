"""Tests for engine."""

from unittest.mock import MagicMock, patch

from theow._core._engine import Theow
from theow._gateway._base import GatewayConfig


def _make_theow(tmp_path, llm=None):
    """Create Theow with patched ChromaStore to avoid real chroma in tmp."""
    with patch("theow._core._engine.ChromaStore") as MockChroma:
        mock_chroma = MagicMock()
        MockChroma.return_value = mock_chroma
        mock_chroma.sync_rules.return_value = None
        mock_chroma.get_all_rules_with_stats.return_value = []

        theow = Theow(
            theow_dir=str(tmp_path / ".theow"),
            name="TestEngine",
            llm=llm,
        )
    return theow


def test_engine_init_creates_directories(tmp_path):
    _make_theow(tmp_path)
    theow_dir = tmp_path / ".theow"
    assert theow_dir.exists()
    assert (theow_dir / "rules").exists()
    assert (theow_dir / "actions").exists()
    assert (theow_dir / "prompts").exists()
    assert (theow_dir / "chroma").exists()


def test_engine_reset_session(tmp_path):
    theow = _make_theow(tmp_path)
    theow._explorer._session_count = 5
    theow.reset_session()
    assert theow.session_count == 0


def test_engine_tool_registration(tmp_path):
    theow = _make_theow(tmp_path)

    @theow.tool()
    def my_tool(x: str) -> str:
        """A tool."""
        return x

    assert theow._tool_registry.get("my_tool") is my_tool


def test_engine_action_registration(tmp_path):
    theow = _make_theow(tmp_path)

    @theow.action("my_action")
    def my_action(x: str) -> str:
        """An action."""
        return x

    assert theow._action_registry.exists("my_action")


def test_engine_mark_creates_decorator(tmp_path):
    theow = _make_theow(tmp_path)

    @theow.mark(context_from=lambda *a, **k: {})
    def my_fn():
        return "ok"

    assert my_fn() == "ok"


def test_engine_ensure_gateway_with_llm(tmp_path):
    with patch("theow._core._engine.ChromaStore") as MockChroma:
        mock_chroma = MagicMock()
        MockChroma.return_value = mock_chroma

        with patch("theow._core._engine.create_gateway") as mock_create:
            mock_gw = MagicMock()
            mock_create.return_value = mock_gw

            theow = Theow(
                theow_dir=str(tmp_path / ".theow"),
                llm="anthropic/claude-sonnet-4",
            )

    assert theow._gateway is mock_gw
    mock_create.assert_called_with(
        "anthropic/claude-sonnet-4",
        config=GatewayConfig(),
    )


def test_engine_meow(tmp_path):
    theow = _make_theow(tmp_path)
    with patch("theow._core._engine._meow") as mock_meow:
        theow.meow()
        mock_meow.assert_called_once()
