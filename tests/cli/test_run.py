"""Tests for CLI run command logic."""

from unittest.mock import patch

import pytest

from theow._cli._run import _decode_output, _parse_context, _tail, run


def test_parse_context_valid():
    assert _parse_context(["key=value", "a=b=c"]) == {"key": "value", "a": "b=c"}


def test_parse_context_invalid():
    with pytest.raises(ValueError, match="Invalid context format"):
        _parse_context(["no_equals"])


def test_decode_output_utf8():
    assert _decode_output(b"hello") == "hello"


def test_decode_output_invalid_bytes():
    result = _decode_output(b"\xff\xfe")
    assert isinstance(result, str)


def test_tail_truncates():
    text = "a\nb\nc\nd\ne"
    assert _tail(text, 2) == "d\ne"


def test_tail_short_text_unchanged():
    assert _tail("a\nb", 5) == "a\nb"


@patch("theow._cli._run.recover")
@patch("theow._cli._run.Theow")
@patch("theow._cli._run.load_config")
@patch("theow._cli._run.subprocess.run")
def test_run_success_returns_zero(mock_subprocess, mock_load_config, mock_theow_cls, mock_recover):
    from theow._cli._config import TheowConfig
    from theow._core._recover import Attempt

    mock_load_config.return_value = TheowConfig()
    mock_recover.return_value = Attempt(success=True, value=0, context={})

    exit_code = run(command=["echo", "hello"], quiet=True)

    assert exit_code == 0
    mock_recover.assert_called_once()


@patch("theow._cli._run.recover")
@patch("theow._cli._run.Theow")
@patch("theow._cli._run.load_config")
@patch("theow._cli._run.subprocess.run")
def test_run_failure_returns_exit_code(
    mock_subprocess, mock_load_config, mock_theow_cls, mock_recover
):
    from theow._cli._config import TheowConfig
    from theow._core._recover import Attempt

    mock_load_config.return_value = TheowConfig()
    mock_recover.return_value = Attempt(success=False, value=1, context={"exit_code": 1})

    exit_code = run(command=["false"], quiet=True)

    assert exit_code == 1
