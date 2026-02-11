"""Built-in tools and signal exceptions for exploration."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import yaml

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


class Done(ExplorationSignal):
    """LLM finished direct fix, ready for retry."""

    def __init__(self, message: str = "") -> None:
        self.message = message
        super().__init__(f"Done: {message}")


# Signal tool functions (module-level for reuse)


def _give_up(reason: str) -> None:
    """Signal that this problem cannot or should not be automated.

    Args:
        reason: Clear explanation of why automation was declined.
    """
    raise GiveUp(reason)


def _request_templates() -> None:
    """Signal that you understand the problem and are ready to write a rule.

    Call this AFTER you have:
    1. Investigated the error thoroughly
    2. Found a fix that works
    3. Determined the pattern is generalizable

    The system will provide rule and action template syntax.
    """
    raise RequestTemplates()


def _submit_rule(rule_file: str, action_file: str | None = None) -> None:
    """Submit your completed rule (and optionally action) for validation.

    Args:
        rule_file: Path to the rule YAML file you created.
        action_file: Path to the action Python file (if you created one).
    """
    raise SubmitRule(rule_file, action_file)


def _done(message: str = "") -> None:
    """Signal that you have completed the task.

    The system will retry the original operation to validate your fix.

    Args:
        message: Brief description of what was done.
    """
    raise Done(message)


# Factory functions for tool sets


def make_signal_tools() -> list[Callable[..., Any]]:
    """Signal tools for explorer mode (rule creation)."""
    return [_give_up, _request_templates, _submit_rule]


def make_direct_fix_tools() -> list[Callable[..., Any]]:
    """Signal tools for direct fix mode (probabilistic rules)."""
    return [_give_up, _done]


def make_search_tools(chroma: ChromaStore, collection: str) -> list[Callable[..., Any]]:
    """Create search tools bound to chroma store."""

    def search_rules(query: str) -> list[dict[str, Any]]:
        """Search existing rules by semantic similarity.

        Returns list of matching rules with their file names (e.g., 'rule_name.rule.yaml').
        """
        results = chroma.query_rules(collection=collection, query_text=query, n_results=5)
        return [
            {"name": name, "file": f"{name}.rule.yaml", "distance": dist, **meta}
            for name, dist, meta in results
        ]

    def search_actions(query: str) -> list[dict[str, Any]]:
        """Search existing actions by semantic similarity."""
        return chroma.query_actions(query_text=query, n_results=5)

    def list_rules() -> list[dict[str, str]]:
        """List all rules in the current collection.

        Returns list of dicts with 'name' and 'file' (e.g., 'rule_name.rule.yaml').
        """
        names = chroma.list_rules(collection)
        return [{"name": name, "file": f"{name}.rule.yaml"} for name in names]

    def list_actions() -> list[str]:
        """List all action names."""
        return chroma.list_actions()

    return [search_rules, search_actions, list_rules, list_actions]


def make_ephemeral_tools(rules_dir: Path) -> list[Callable[..., Any]]:
    """Create tools for accessing ephemeral rules from current/previous attempts."""
    ephemeral_dir = rules_dir / "ephemeral"

    def list_ephemeral_rules() -> list[dict[str, Any]]:
        """List ephemeral rules from current or previous exploration attempts.

        Returns list of ephemeral rules with name, file, tags, description, and notes.
        Check this at the start of exploration to see prior work you can continue from.
        """
        if not ephemeral_dir.exists():
            return []

        results = []
        for path in ephemeral_dir.glob("*.rule.yaml"):
            try:
                content = path.read_text()
                data = yaml.safe_load(content)
                results.append(
                    {
                        "name": data.get("name", path.stem),
                        "file": str(path),
                        "tags": data.get("tags", []),
                        "description": data.get("description", "")[:200],
                        "notes": data.get("notes", "")[:500] if data.get("notes") else None,
                    }
                )
            except Exception:
                continue
        return results

    def read_ephemeral_rule(name: str) -> str:
        """Read the full content of an ephemeral rule to continue from.

        Args:
            name: Rule name (without .rule.yaml extension).

        Returns the full YAML content of the rule file.
        """
        path = ephemeral_dir / f"{name}.rule.yaml"
        if not path.exists():
            return f"Error: Ephemeral rule '{name}' not found"
        return path.read_text()

    return [list_ephemeral_rules, read_ephemeral_rule]


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
