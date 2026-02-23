"""Theow CLI."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from theow._version import __version__

app = typer.Typer(name="theow", no_args_is_help=True, add_completion=False)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"theow {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option("--version", "-V", callback=_version_callback, is_eager=True),
    ] = None,
) -> None:
    """Theow: LLM-in-the-loop recovery engine."""


@app.command()
def run(
    command: Annotated[list[str], typer.Argument(help="Command to run")],
    theow_dir: Annotated[str, typer.Option("--theow-dir", "-d", envvar="THEOW_DIR")] = ".theow",
    profile: Annotated[Optional[str], typer.Option("--profile", "-p")] = None,
    explore: Annotated[bool, typer.Option("--explore")] = False,
    context: Annotated[Optional[list[str]], typer.Option("--context", "-C")] = None,
    tail: Annotated[Optional[int], typer.Option("--tail")] = None,
    plugin: Annotated[Optional[str], typer.Option("--plugin")] = None,
    hint: Annotated[Optional[str], typer.Option("--hint")] = None,
    escalate: Annotated[bool, typer.Option("--allow-escalation")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
) -> None:
    """Run a command with theow recovery."""
    from theow._cli._run import run as run_cmd

    exit_code = run_cmd(
        command=command,
        theow_dir=theow_dir,
        profile=profile,
        explore=explore,
        context=context,
        tail=tail,
        plugin=plugin,
        hint=hint,
        escalate=escalate,
        quiet=quiet,
    )
    raise typer.Exit(code=exit_code)


@app.command()
def stats(
    theow_dir: Annotated[str, typer.Option("--theow-dir", "-d", envvar="THEOW_DIR")] = ".theow",
) -> None:
    """Print theow stats."""
    from theow._cli._stats import stats as stats_cmd

    stats_cmd(theow_dir=theow_dir)
