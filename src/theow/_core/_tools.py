"""Built-in tools and signal exceptions for exploration."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import yaml

if TYPE_CHECKING:
    from theow._core._chroma_store import ChromaStore


# Signals that a LLM can raise for theow to intercept and guide.
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


class Escalate(ExplorationSignal):
    """LLM requests escalation to a more capable model."""

    def __init__(self, findings: str) -> None:
        self.findings = findings
        super().__init__(f"Escalate: {findings}")


class RuleResolved(ExplorationSignal):
    """LLM augmented an existing rule so it now handles this case."""

    def __init__(self, summary: str = "") -> None:
        self.summary = summary
        super().__init__(f"Rule resolved: {summary}")


class Done(ExplorationSignal):
    """LLM finished direct fix, ready for retry."""

    def __init__(self, message: str = "") -> None:
        self.message = message
        super().__init__(f"Done: {message}")


# Signal tool functions (module-level for reuse)
def _give_up(reason: str = "") -> None:
    """Signal that this problem is fundamentally unautomatable.

    Only use this when you have a clear, specific reason why NO agent could
    solve this — e.g. requires human judgment, missing credentials, or the
    problem is outside the scope of automation entirely.

    If you're unsure or simply stuck, use `_escalate()` instead to hand off
    your findings to a senior model that may succeed where you couldn't.

    Args:
        reason: Clear explanation of why automation was declined. Omit when
                giving up based on a prior observation to avoid duplicate logs.
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


def _rule_resolved(summary: str = "") -> None:
    """Signal that an existing rule was augmented to handle this error.

    IMPORTANT: Only call this AFTER _test_rule_match() confirms all facts pass.
    The system re-checks rules using fact matching — if facts don't match, the
    rule won't execute. _add_example_to_rule() only improves search recall, it
    does NOT change fact matching. Use _add_fact_to_rule() to extend matchers.

    Args:
        summary: Brief description of what was augmented.
    """
    raise RuleResolved(summary)


def _done(message: str = "") -> None:
    """Signal that you have completed the task.

    The system will retry the original operation to validate your fix.

    Args:
        message: Brief description of what was done.
    """
    raise Done(message)


def _escalate(findings: str) -> None:
    """Escalate to a senior model when you are stuck but the problem looks solvable.

    Use this when you've investigated the problem, have partial findings or
    hypotheses, but cannot complete the fix yourself. A more capable model
    will receive your findings and continue from where you left off.

    Prefer `_done()` if you've already applied a fix.
    Prefer `_give_up()` only if the problem is fundamentally unautomatable
    (e.g. requires human judgment, missing credentials).

    Args:
        findings: Your analysis so far — what you tried, what you found, and
                  any hypotheses about the root cause or fix.
    """
    raise Escalate(findings)


# Factory functions for internal tool sets
def make_signal_tools() -> list[Callable[..., Any]]:
    """Signal tools for explorer mode (rule creation)."""
    return [_give_up, _request_templates, _submit_rule, _rule_resolved]


def make_direct_fix_tools(
    rules_dir: Path | None = None,
    allow_escalation: bool = False,
) -> list[Callable[..., Any]]:
    """Signal tools for direct fix mode (probabilistic rules)."""
    tools: list[Callable[..., Any]] = [_give_up, _done]
    if allow_escalation:
        tools.append(_escalate)
    if rules_dir:
        tools.append(_make_failed_rules_tool(rules_dir))
        tools.append(_make_observations_tool(rules_dir))
    return tools


def make_search_tools(
    chroma: ChromaStore, collection: str, rules_dir: Path | None = None
) -> list[Callable[..., Any]]:
    """Create search tools bound to chroma store."""

    def _search_rules(query: str) -> list[dict[str, Any]]:
        """Search existing rules by semantic similarity.

        Returns list of matching rules with their file names (e.g., 'rule_name.rule.yaml').
        """
        results = chroma.query_rules(collection=collection, query_text=query, n_results=5)
        return [
            {"name": name, "file": f"{name}.rule.yaml", "distance": dist, **meta}
            for name, dist, meta in results
        ]

    def _search_actions(query: str) -> list[dict[str, Any]]:
        """Search existing actions by semantic similarity."""
        return chroma.query_actions(query_text=query, n_results=5)

    def _list_rules() -> list[dict[str, str]]:
        """List all rules in the current collection.

        Returns list of dicts with 'name' and 'file' (e.g., 'rule_name.rule.yaml').
        """
        names = chroma.list_rules(collection)
        return [{"name": name, "file": f"{name}.rule.yaml"} for name in names]

    def _list_actions() -> list[str]:
        """List all action names."""
        return chroma.list_actions()

    tools: list[Callable[..., Any]] = [_search_rules, _search_actions, _list_rules, _list_actions]
    if rules_dir:
        tools.append(_make_failed_rules_tool(rules_dir))
        tools.append(_make_observations_tool(rules_dir))
    return tools


def _make_failed_rules_tool(rules_dir: Path) -> Callable[..., Any]:
    """Create a tool that lists archived failed rules."""
    failed_dir = rules_dir / "failed"

    def _list_failed_rules() -> list[dict[str, Any]]:
        """List previously failed rules and their error context.

        Use this to learn what approaches were already tried and avoid repeating them.
        Returns rule names, descriptions, and the errors that caused rejection.
        """
        if not failed_dir.exists():
            return []

        results = []
        for path in sorted(failed_dir.glob("*.rule.yaml")):
            entry: dict[str, Any] = {"name": path.stem, "file": path.name}
            sidecar = path.with_suffix(path.suffix + ".json")
            if sidecar.exists():
                try:
                    entry["metadata"] = json.loads(sidecar.read_text())
                except Exception:
                    pass
            results.append(entry)
        return results

    return _list_failed_rules


def _make_observations_tool(rules_dir: Path) -> Callable[..., Any]:
    """Create a tool that searches past LLM attempt observations."""
    observations_path = rules_dir.parent / "observations.jsonl"

    def _search_observations(query: str = "") -> list[dict[str, Any]]:
        """Search past LLM attempt observations for patterns and lessons learned.

        Returns entries where the query substring matches any value.
        Each entry has an "outcome" field ("success" or "failure").
        Use this to learn from previous attempts and avoid repeating mistakes.

        Args:
            query: Substring to search for across all observation fields. Empty returns all.
        """
        if not observations_path.exists():
            return []

        results = []
        for line in observations_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            if not query or any(query.lower() in str(v).lower() for v in entry.values()):
                results.append(entry)
        return results

    return _search_observations


def make_validation_tools(
    rules_dir: Path, context: dict[str, Any], action_registry: Any = None
) -> list[Callable[..., Any]]:
    """Create tools for validating rules against current context."""
    from theow._core._models import Rule

    actions_dir = rules_dir.parent / "actions"

    def _test_rule_match(rule_file: str) -> dict[str, Any]:
        """Test if a rule's when-facts match the current error context.

        Call this BEFORE submit_rule() to verify your patterns are correct.
        Validates both fact matching AND action file correctness.

        Args:
            rule_file: Path to the rule file to test.
        """
        path = Path(rule_file)
        if not path.exists():
            return {"error": f"Rule file not found: {rule_file}"}

        try:
            rule = Rule.from_yaml(path)
        except Exception as e:
            return {"error": f"Failed to parse rule: {e}"}

        results = []
        all_match = True

        for fact in rule.when:
            value = context.get(fact.fact)
            match_result = fact.matches(value)

            if match_result is None:
                all_match = False
                condition = fact.equals or fact.contains or fact.regex
                actual = str(value)[:200] if value else "(missing)"
                results.append(
                    {
                        "fact": fact.fact,
                        "status": "FAIL",
                        "expected": condition,
                        "actual": actual + "..." if value and len(str(value)) > 200 else actual,
                    }
                )
            else:
                results.append(
                    {
                        "fact": fact.fact,
                        "status": "OK",
                        "captures": match_result if match_result else None,
                    }
                )

        # Validate actions can be loaded
        action_errors = []
        if action_registry:
            for rule_action in rule.then:
                action_path = actions_dir / f"{rule_action.action}.py"
                if not action_path.exists():
                    action_errors.append(f"Action file not found: {action_path}")
                    continue
                action_registry._load_action_file(action_path)
                if not action_registry.exists(rule_action.action):
                    action_errors.append(
                        f'Action "{rule_action.action}" not registered after loading '
                        f'{action_path.name} - check @action("{rule_action.action}") decorator'
                    )

        ok = all_match and not action_errors
        result: dict[str, Any] = {"matches": ok, "facts": results}
        if action_errors:
            result["action_errors"] = action_errors
        if not ok:
            result["hint"] = "Fix errors before calling submit_rule()"
        return result

    return [_test_rule_match]


def make_ephemeral_tools(rules_dir: Path) -> list[Callable[..., Any]]:
    """Create tools for accessing ephemeral rules from current/previous attempts."""
    ephemeral_dir = rules_dir / "ephemeral"

    def _list_ephemeral_rules() -> list[dict[str, Any]]:
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

    def _read_ephemeral_rule(name: str) -> str:
        """Read the full content of an ephemeral rule to continue from.

        Args:
            name: Rule name (without .rule.yaml extension).

        Returns the full YAML content of the rule file.
        """
        path = ephemeral_dir / f"{name}.rule.yaml"
        if not path.exists():
            return f"Error: Ephemeral rule '{name}' not found"
        return path.read_text()

    def _write_rule(name: str, content: str) -> dict[str, str]:
        """Write a rule file to the ephemeral folder.

        Args:
            name: Rule name (without extension). Will be saved as <name>.rule.yaml
            content: YAML content of the rule.

        Returns dict with 'path' for use with test_rule_match() and submit_rule().
        """
        ephemeral_dir.mkdir(parents=True, exist_ok=True)
        path = ephemeral_dir / f"{name}.rule.yaml"
        path.write_text(content)
        return {"path": str(path), "message": f"Rule written to {path}"}

    def _write_action(name: str, content: str) -> dict[str, str]:
        """Write an action file to the actions folder.

        Args:
            name: Action name (without extension). Will be saved as <name>.py
            content: Python content of the action.

        Returns dict with 'path' for use with submit_rule().
        """
        actions_dir = rules_dir.parent / "actions"
        actions_dir.mkdir(parents=True, exist_ok=True)
        path = actions_dir / f"{name}.py"
        path.write_text(content)
        return {"path": str(path), "message": f"Action written to {path}"}

    return [_list_ephemeral_rules, _read_ephemeral_rule, _write_rule, _write_action]


def make_augmentation_tools(
    rules_dir: Path,
    context: dict[str, Any],
    chroma: ChromaStore,
) -> list[Callable[..., Any]]:
    """Create tools for augmenting existing rules with new facts/examples."""
    from theow._core._models import Fact, Rule

    def _add_fact_to_rule(
        rule_name: str,
        fact: str,
        equals: str = "",
        contains: str = "",
        regex: str = "",
        examples: list[str] | None = None,
    ) -> dict[str, Any]:
        """Add a new when-fact to an existing permanent rule.

        Use this to extend a rule to match new error patterns instead of creating
        a duplicate rule. The new fact must match the current error context.
        Facts are ANDed, so adding a fact makes the rule more specific.

        Args:
            rule_name: Name of the rule to augment (without .rule.yaml).
            fact: The context key to match (e.g. 'stderr', 'error').
            equals: Exact match value. Mutually exclusive with contains/regex.
            contains: Substring match. Mutually exclusive with equals/regex.
            regex: Regex pattern with optional named captures. Mutually exclusive with equals/contains.
            examples: Example strings that improve vector search recall.
        """
        rule_path = rules_dir / f"{rule_name}.rule.yaml"
        if not rule_path.exists():
            return {"error": f"Rule not found: {rule_name}"}

        operators = [v for v in (equals, contains, regex) if v]
        if len(operators) != 1:
            return {"error": "Provide exactly one of: equals, contains, regex"}

        new_fact = Fact(
            fact=fact,
            equals=equals or None,
            contains=contains or None,
            regex=regex or None,
            examples=examples or [],
        )

        # Validate the new fact matches the current context
        value = context.get(fact)
        if new_fact.matches(value) is None:
            return {
                "error": "New fact does not match current context",
                "fact_key": fact,
                "actual_value": str(value)[:200] if value else "(missing)",
            }

        # Temporarily make writable, add fact, re-lock
        original_mode = rule_path.stat().st_mode
        rule_path.chmod(0o644)
        try:
            rule = Rule.from_yaml(rule_path)
            rule.add_fact(new_fact)
            chroma.index_rule(rule)
        finally:
            rule_path.chmod(original_mode)

        return {
            "status": "ok",
            "rule": rule_name,
            "added_fact": new_fact.to_dict(),
        }

    def _add_example_to_rule(
        rule_name: str,
        fact_key: str,
        example: str,
    ) -> dict[str, Any]:
        """Add an example to an existing fact in a rule to improve search recall.

        Examples don't change matching logic — they only help the vector search
        find this rule for similar error patterns. Always safe to add.

        Args:
            rule_name: Name of the rule (without .rule.yaml).
            fact_key: The fact's context key (e.g. 'stderr') to add the example to.
            example: An example error string that this rule should match.
        """
        rule_path = rules_dir / f"{rule_name}.rule.yaml"
        if not rule_path.exists():
            return {"error": f"Rule not found: {rule_name}"}

        original_mode = rule_path.stat().st_mode
        rule_path.chmod(0o644)
        try:
            rule = Rule.from_yaml(rule_path)
            if not rule.add_example_to_fact(fact_key, example):
                return {"error": f"No regex fact with key '{fact_key}' found in rule"}
            chroma.index_rule(rule)
        finally:
            rule_path.chmod(original_mode)

        return {
            "status": "ok",
            "rule": rule_name,
            "fact_key": fact_key,
            "example_added": example,
        }

    return [_add_fact_to_rule, _add_example_to_rule]


# Theow external tools for consumers. Not registered by default, consumers can choose to register them.
def read_file(path: str) -> str:
    """Read file contents."""
    return Path(path).read_text()


def write_file(path: str, content: str) -> str:
    """Write content to file, creating directories if needed.

    Do NOT use this to create rule or action files.
    Use _write_rule for rules and _write_action for actions instead.
    """
    p = Path(path)
    if p.suffix == ".py" and p.parent.name == "actions":
        return (
            "ERROR: Cannot write action files with write_file. Use the _write_action tool instead."
        )
    if ".rule.yaml" in p.name:
        return "ERROR: Cannot write rule files with write_file. Use the _write_rule tool instead."
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
