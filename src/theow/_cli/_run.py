"""The `theow run` command."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from theow._cli._config import load_config, resolve_profile
from theow._cli._plugin import load_plugin
from theow._core._engine import Theow
from theow._core._logging import get_logger
from theow._core._recover import Attempt, RecoveryConfig, recover
from theow._core._tools import read_file, write_file, run_command, list_directory

logger = get_logger(__name__)


def run(
    command: list[str],
    theow_dir: str = ".theow",
    profile: str | None = None,
    explore: bool = False,
    context: list[str] | None = None,
    tail: int | None = None,
    plugin: str | None = None,
    hint: str | None = None,
    quiet: bool = False,
) -> int:
    """Execute a command with theow recovery. Returns exit code."""
    theow_path = Path(theow_dir)
    config = load_config(theow_path)

    cli_overrides: dict[str, Any] = {}
    if explore:
        cli_overrides["explore"] = True
    if plugin:
        cli_overrides["plugin"] = plugin
    if hint:
        cli_overrides["hint"] = hint

    prof = resolve_profile(config, profile, cli_overrides)

    engine = Theow(
        theow_dir=theow_path,
        llm=config.engine.llm,
        llm_secondary=config.engine.llm_secondary,
        session_limit=config.engine.session_limit,
        max_tool_calls_per_session=config.engine.max_tool_calls_per_session,
        max_tokens_per_session=config.engine.max_tokens_per_session,
        archive_llm_attempt=config.engine.archive_llm_attempt,
    )

    _register_builtin_tools(engine)

    setup_fn = None
    teardown_fn = None
    plugin_path = prof.plugin
    if plugin_path:
        result = load_plugin(Path(plugin_path), engine._tool_registry)
        setup_fn = result.setup
        teardown_fn = result.teardown

    extra_context = _parse_context(context or [])

    def make_run(cmd: list[str]) -> Attempt[int]:
        proc = subprocess.run(cmd, capture_output=True)
        stdout = _decode_output(proc.stdout)
        stderr = _decode_output(proc.stderr)

        if tail:
            stdout = _tail(stdout, tail)
            stderr = _tail(stderr, tail)

        ctx: dict[str, Any] = {
            "command": " ".join(cmd),
            "exit_code": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            **extra_context,
        }
        return Attempt(
            success=proc.returncode == 0,
            value=proc.returncode,
            context=ctx,
        )

    if not quiet:
        logger.info("Running", command=" ".join(command))

    recovery_config = RecoveryConfig(
        max_retries=prof.max_retries,
        max_depth=prof.max_depth,
        rules=prof.rules,
        tags=prof.tags,
        collection=prof.collection,
        fallback=prof.fallback,
        explorable=prof.explore,
        hint=prof.hint,
    )

    attempt = recover(
        run=lambda: make_run(command),
        engine=engine,
        config=recovery_config,
        before_attempt=setup_fn,
        after_attempt=teardown_fn,
    )

    if attempt.success and not quiet:
        logger.info("Success")
    elif not attempt.success and not quiet:
        logger.warning("Failed after recovery", exit_code=attempt.context.get("exit_code"))

    return attempt.value if attempt.value is not None else 1


def _register_builtin_tools(engine: Theow) -> None:
    """Register built-in filesystem and command tools."""
    engine._tool_registry._tools["read_file"] = read_file
    engine._tool_registry._tools["write_file"] = write_file
    engine._tool_registry._tools["run_command"] = run_command
    engine._tool_registry._tools["list_directory"] = list_directory


def _parse_context(pairs: list[str]) -> dict[str, str]:
    """Parse key=value context pairs."""
    ctx: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid context format (expected key=value): {pair}")
        key, value = pair.split("=", 1)
        ctx[key] = value
    return ctx


def _decode_output(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


def _tail(text: str, n: int) -> str:
    """Keep last n lines."""
    lines = text.splitlines()
    return "\n".join(lines[-n:]) if len(lines) > n else text
