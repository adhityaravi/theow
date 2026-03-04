# CLI

Theow ships a CLI for running commands with automatic recovery. It uses the same engine, resolver, and explorer as the Python API.

## Usage

```bash
theow run [OPTIONS] -- COMMAND...
```

The `--` separates theow flags from the command to run. Everything after `--` is executed as a subprocess.

## Commands

### `theow run`

Run a command with theow recovery. If the command fails (non-zero exit code), theow tries to fix it using rules, and optionally LLM exploration.

```bash
# Basic usage
theow run -- pytest tests/

# With exploration enabled
theow run --explore -- make deploy

# With a named profile
theow run --profile deploy -- ./deploy.sh

# With extra context
theow run -C environment=prod -C region=us-east-1 -- make deploy

# Truncate output (last 50 lines of stdout/stderr)
theow run --tail 50 -- pytest tests/

# With a plugin
theow run --plugin hooks/deploy_plugin.py -- make deploy

# Quiet mode (suppress theow's own log messages)
theow run -q -- pytest tests/
```

#### Flags

| Flag | Short | Env Var | Description |
|---|---|---|---|
| `--theow-dir` | `-d` | `THEOW_DIR` | Path to `.theow` directory (default: `.theow`) |
| `--profile` | `-p` | | Named profile from `config.yaml` |
| `--explore` | | | Enable LLM exploration (still requires `THEOW_EXPLORE=1`) |
| `--context` | `-C` | | Extra `key=value` pairs added to the error context |
| `--tail` | | | Keep only last N lines of stdout/stderr |
| `--plugin` | | | Path to a plugin Python file |
| `--hint` | | | Free-text hint injected into the exploration prompt |
| `--allow-escalation` | | | Allow escalation to secondary model |
| `--quiet` | `-q` | | Suppress theow log output |

CLI flags override profile values when both are set.

### `theow stats`

Print rule and exploration statistics from the ChromaDB store.

```bash
theow stats
theow stats --theow-dir /path/to/.theow
```

### `theow --version`

```bash
theow -V
```

## Config File

The CLI reads `.theow/config.yaml` for engine settings and named profiles:

```yaml
engine:
  llm: anthropic/claude-sonnet-4-20250514
  llm_secondary: anthropic/claude-opus-4-6
  session_limit: 20
  max_tool_calls_per_session: 30
  max_tokens_per_session: 8192
  archive_llm_attempt: false

profiles:
  deploy:
    tags: [deploy, infra]
    collection: ops
    max_retries: 5
    max_depth: 2
    explore: true
    allow_escalation: true
    hint: "This is a deployment pipeline failure"

  tests:
    tags: [test]
    rules: [fix-import-error, fix-missing-fixture]
    max_retries: 3
    explore: false
    fallback: true
```

### Engine Settings

| Key | Default | Description |
|---|---|---|
| `llm` | `None` | Primary model spec (`provider/model`) |
| `llm_secondary` | `None` | Secondary model for escalation |
| `session_limit` | `20` | Max exploration sessions |
| `max_tool_calls_per_session` | `30` | Tool call budget per session |
| `max_tokens_per_session` | `8192` | Token budget per session |
| `archive_llm_attempt` | `false` | Log LLM outcomes to `observations.jsonl` |

### Profile Settings

| Key | Default | Description |
|---|---|---|
| `tags` | `None` | Filter rules by tag |
| `rules` | `None` | Explicit rule names to try first |
| `collection` | `"default"` | ChromaDB collection |
| `max_retries` | `3` | Rules to try per error |
| `max_depth` | `3` | Cascading errors to chase |
| `explore` | `false` | Enable LLM exploration |
| `fallback` | `true` | Fall back to semantic search |
| `plugin` | `None` | Default plugin file path |
| `hint` | `None` | Default exploration hint |
| `allow_escalation` | `false` | Allow model escalation |

## Plugins

Plugins are Python files that register tools and lifecycle hooks for CLI recovery. Load them with `--plugin` or set a default in the profile.

```python
# my_plugin.py
from theow import tool

@tool()
def check_health(url: str) -> dict:
    """Check service health endpoint."""
    import requests
    r = requests.get(url)
    return {"status": r.status_code}

def setup(state: dict, attempt: int) -> dict:
    """Called before each recovery attempt."""
    state["started_at"] = time.time()
    return state

def teardown(state: dict, attempt: int, success: bool) -> None:
    """Called after each recovery attempt."""
    elapsed = time.time() - state.get("started_at", 0)
    print(f"Attempt {attempt}: {'ok' if success else 'fail'} ({elapsed:.1f}s)")
```

- `@tool()` decorated functions are registered as LLM tools.
- `setup(state, attempt)` and `teardown(state, attempt, success)` are extracted by name. Both are optional.

See [hooks](hooks.md) for details on the hook lifecycle.

## Built-in Tools

The CLI automatically registers `read_file`, `write_file`, `run_command`, and `list_directory`. Plugin tools are added on top of these.
