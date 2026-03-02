# Rules and Actions

Rules are YAML files that pattern-match error contexts. Actions are Python functions that fix them. Together they form theow's memory: once the LLM solves a problem, the rule-action pair fires instantly next time.

## Rule Format

```yaml
name: fix-missing-config
description: Config file missing from expected path
when:
  - fact: stderr
    contains: "FileNotFoundError"
  - fact: stderr
    regex: "No such file.*?(?P<filepath>/[\w/.-]+\.ya?ml)"
then:
  - action: restore_config
    params:
      path: "{filepath}"
tags: [config, filesystem]
collection: default
```

### `when` — Fact Matching

Facts are conditions checked against the error context dict. All facts in a rule are ANDed.

Each fact targets a context key (`fact`) and uses one operator:

| Operator | Description |
|---|---|
| `equals` | Exact string match |
| `contains` | Substring match |
| `regex` | Regex with optional named capture groups (`(?P<name>...)`) |

Captures from regex facts are passed to action params as `{name}` placeholders.

```yaml
when:
  # Exact match
  - fact: exit_code
    equals: "1"

  # Substring
  - fact: stderr
    contains: "connection refused"

  # Regex with captures
  - fact: stderr
    regex: "ModuleNotFoundError: No module named '(?P<module>[^']+)'"
```

#### Examples

Facts can include `examples` to improve vector search recall without changing the matching logic:

```yaml
when:
  - fact: stderr
    regex: "ImportError.*(?P<module>\\w+)"
    examples:
      - "ImportError: cannot import name 'foo' from 'bar'"
      - "ModuleNotFoundError: No module named 'baz'"
```

Examples are embedded alongside the rule description in ChromaDB. They help the resolver find rules for similar but not identical error messages.

### `then` — Actions

Actions execute sequentially. Each references a registered action name and optional params:

```yaml
then:
  - action: install_package
    params:
      package: "{module}"
  - action: restart_service
    params:
      name: "app"
```

Params support `{placeholder}` syntax. Placeholders are resolved from regex captures first, then from the full context dict.

### Tags and Collections

`tags` are string labels for filtering. The resolver can narrow candidates by tag before falling back to semantic search.

`collection` controls which ChromaDB collection the rule belongs to. Defaults to `"default"`. Use collections to isolate rule sets (e.g. `ops`, `tests`, `deploy`).

### Notes

Free-text `notes` field for the LLM to leave context about incomplete rules:

```yaml
notes: "Handles the basic case. Needs extension for nested configs."
```

## Action Files

Actions live in `.theow/actions/` as Python files with the `@action` decorator:

```python
# .theow/actions/install_package.py
from theow import action

@action("install_package")
def install_package(package: str) -> dict:
    """Install a Python package."""
    import subprocess
    result = subprocess.run(
        ["pip", "install", package],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return {"status": "error", "stderr": result.stderr}
    return {"status": "ok", "installed": package}
```

Actions are discovered on startup from `.theow/actions/*.py`. The decorator name (`"install_package"`) is what rules reference in `then.action`.

Actions should do one atomic fix and return. Do not run verification or rebuild commands inside the action — the recovery loop handles that.

## Programmatic Actions

Register actions directly on the engine instead of using files:

```python
@agent.action("install_package")
def install_package(package: str) -> dict:
    """Install a Python package."""
    ...
```

## Deterministic vs Probabilistic Rules

Rules without `llm_config` are **deterministic**: they run Python actions directly.

Rules with `llm_config` are **probabilistic**: they trigger an LLM conversation with a stored prompt instead of running code.

```yaml
name: fix-complex-config
description: Configuration error requiring analysis
when:
  - fact: stderr
    contains: "ConfigurationError"
llm_config:
  prompt_template: "file://prompts/fix_config.md"
  tools: [read_file, write_file]
  constraints:
    max_tool_calls: 10
    max_tokens: 4096
    allow_escalation: true
  use_secondary: false
```

`prompt_template` can be an inline string or `file://` reference relative to the `.theow/` directory. Placeholders like `{stderr}` are resolved from context.

`tools` lists registered tool names the LLM can use. Internal signal tools (`_done`, `_give_up`) are added automatically.

`use_secondary` routes the conversation to `llm_secondary` instead of the primary model.

## Rule Lifecycle

1. **Exploration creates an ephemeral rule** in `rules/ephemeral/`. The LLM writes the YAML and action file, calls `_test_rule_match()` to verify facts, then `_submit_rule()`.

2. **Validation** checks that all facts match the current context, the action file loads, and no conflicting rule exists.

3. **The recovery loop executes the action** and re-runs the original function. If it succeeds, the rule is **promoted** from `rules/ephemeral/` to `rules/`.

4. **On promotion**, the rule is indexed in ChromaDB. Next time the same error occurs, the resolver finds it instantly.

5. **If the rule fails**, it's rejected. With `archive_llm_attempt=True`, failed rules are moved to `rules/failed/` with JSON metadata for debugging. Otherwise they're deleted.

6. **Incomplete rules** (budget exhausted before `_submit_rule()`) are tagged `[incomplete]` and kept in ephemeral. The next exploration attempt can continue from them via `_list_ephemeral_rules()`.
