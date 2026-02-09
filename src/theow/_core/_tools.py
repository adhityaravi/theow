"""Built-in tools and signal exceptions for exploration."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from theow._core._chroma_store import ChromaStore


class ExplorationSignal(Exception):
    """Base for all exploration signals."""

    pass


class GiveUp(ExplorationSignal):
    """LLM determined the problem can't/shouldn't be automated."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Gave up: {reason}")


class RequestTemplates(ExplorationSignal):
    """LLM is ready to write a rule, needs the syntax."""

    pass


class SubmitRule(ExplorationSignal):
    """LLM finished writing the rule."""

    def __init__(self, rule_file: str, action_file: str | None = None) -> None:
        self.rule_file = rule_file
        self.action_file = action_file
        super().__init__(f"Submitted: {rule_file}")


# Theow internal tools for explorer. Auto-registered.


def make_signal_tools() -> list[Callable[..., Any]]:
    """Create signal tools for conversation flow control."""

    def give_up(reason: str) -> None:
        """Signal that this problem cannot or should not be automated.

        Use when:
        - The problem is structural (architecture issues, design flaws)
        - It requires human intervention (permissions, credentials)
        - The error is too specific with no generalizable pattern
        - The fix would be too risky to automate

        Args:
            reason: Clear explanation of why automation was declined.
        """
        raise GiveUp(reason)

    def request_templates() -> None:
        """Signal that you understand the problem and are ready to write a rule.

        Call this AFTER you have:
        1. Investigated the error thoroughly
        2. Found a fix that works
        3. Determined the pattern is generalizable

        The system will provide rule and action template syntax.
        """
        raise RequestTemplates()

    def submit_rule(rule_file: str, action_file: str | None = None) -> None:
        """Submit your completed rule (and optionally action) for validation.

        Args:
            rule_file: Path to the rule YAML file you created.
            action_file: Path to the action Python file (if you created one).
        """
        raise SubmitRule(rule_file, action_file)

    return [give_up, request_templates, submit_rule]


def make_search_tools(chroma: ChromaStore, collection: str) -> list[Callable[..., Any]]:
    """Create search tools bound to chroma store."""

    def search_rules(query: str) -> list[dict[str, Any]]:
        """Search existing rules by semantic similarity."""
        results = chroma.query_rules(collection=collection, query_text=query, n_results=5)
        return [{"name": name, "distance": dist, **meta} for name, dist, meta in results]

    def search_actions(query: str) -> list[dict[str, Any]]:
        """Search existing actions by semantic similarity."""
        return chroma.query_actions(query_text=query, n_results=5)

    def list_rules() -> list[str]:
        """List all rule names in the current collection."""
        return chroma.list_rules(collection)

    def list_actions() -> list[str]:
        """List all action names."""
        return chroma.list_actions()

    return [search_rules, search_actions, list_rules, list_actions]


# Theow external tools for consumers. Not registered by default, consumers can choose to register them.


def read_file(path: str) -> str:
    """Read file contents."""
    return Path(path).read_text()


def write_file(path: str, content: str) -> str:
    """Write content to file, creating directories if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Written {len(content)} bytes to {path}"


def run_command(cmd: str, cwd: str | None = None) -> dict[str, Any]:
    """Run shell command and return output."""
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def list_directory(path: str) -> list[str]:
    """List files and directories at path."""
    return [p.name for p in Path(path).iterdir()]
