"""Tests for logging module."""

from unittest.mock import MagicMock

from theow._core._logging import (
    COMPONENT_MAP,
    ComponentLogger,
    get_engine_name,
    get_logger,
    set_engine_name,
)


def test_engine_name_default():
    set_engine_name("Theow")
    assert get_engine_name() == "Theow"


def test_engine_name_custom():
    set_engine_name("SD-Agent")
    assert get_engine_name() == "SD-Agent"
    set_engine_name("Theow")


def test_component_logger_uses_engine_name():
    """Logger name includes engine:component."""
    from unittest.mock import patch

    comp_logger = ComponentLogger("gateway")

    with patch("theow._core._logging.structlog.get_logger") as mock_get:
        mock_logger = MagicMock()
        mock_get.return_value = mock_logger

        comp_logger.info("Test message", foo="bar")

        mock_get.assert_called_with("Theow:gateway")
        mock_logger.info.assert_called_once_with("Test message", foo="bar")


def test_component_logger_all_levels():
    from unittest.mock import patch

    comp_logger = ComponentLogger("resolver")

    with patch("theow._core._logging.structlog.get_logger") as mock_get:
        mock_logger = MagicMock()
        mock_get.return_value = mock_logger

        comp_logger.debug("debug msg")
        comp_logger.info("info msg")
        comp_logger.warning("warning msg")
        comp_logger.error("error msg")

        mock_logger.debug.assert_called_with("debug msg")
        mock_logger.info.assert_called_with("info msg")
        mock_logger.warning.assert_called_with("warning msg")
        mock_logger.error.assert_called_with("error msg")


def test_get_logger_returns_component_logger():
    logger = get_logger("theow._core._resolver")
    assert isinstance(logger, ComponentLogger)
    assert logger._component == "resolver"


def test_get_logger_fallback_component():
    logger = get_logger("theow._core._unknown_module")
    assert logger._component == "unknown_module"


def test_component_map_coverage():
    expected_components = {"llm-gateway", "explorer", "resolver", "recovery", "engine", "chroma"}
    actual_components = set(COMPONENT_MAP.values())
    assert actual_components == expected_components
