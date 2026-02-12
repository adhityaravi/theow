# Theow

LLM-in-the-loop [rule-based system](https://en.wikipedia.org/wiki/Production_system_(computer_science)) for failure recovery.

```bash
pip install theow
```

## Working modes

1. **Resolver** - Match failures against known rules, execute fixes deterministically
2. **Explorer** - When no rule matches, LLM investigates and creates new rules

## Initialization

```python
from theow import Theow

my_agent = Theow(
    theow_dir="./.theow",                            # Where rules/actions/chroma live
    name="my_agent",                                  # Name for logging
    llm="gemini/gemini-2.0-flash",                   # Primary LLM (provider/model)
    llm_secondary="anthropic/claude-sonnet-4-20250514",  # Optional fallback
    session_limit=20,                                 # Max LLM explores per session
    max_tool_calls=30,                               # Max tool calls per conversation
    max_tokens=8192,                                 # Max ouput tokens per conversation
)
```

**Supported Providers:** `gemini/`, `anthropic/`, `copilot/`

Each provider uses its own API key from environment:
- `gemini/*`: `GEMINI_API_KEY`
- `anthropic/*`: `ANTHROPIC_API_KEY`
- `copilot/*`: `GITHUB_TOKEN`

## Tools

Tools are what the LLM can do during exploration. Theow provides common tools - import and register what you need:

```python
from theow.tools import read_file, write_file, run_command, list_directory

# Register the ones you want (loose leash)
my_agent.tool()(read_file)
my_agent.tool()(write_file)
my_agent.tool()(run_command)
my_agent.tool()(list_directory)
```

Or write one with tighter constraints and register:

```python
@my_agent.tool()
def read_workspace_file(workspace: str, relative_path: str) -> str:
    """Read file within workspace only."""
    path = Path(workspace) / relative_path
    if not path.is_relative_to(workspace):
        raise ValueError("Path escapes workspace")
    return path.read_text()

@my_agent.tool()
def run_go_mod(workspace: str, args: str) -> dict:
    """Run go mod commands only."""
    result = subprocess.run(f"go mod {args}", shell=True, cwd=workspace, capture_output=True, text=True)
    return {"returncode": result.returncode, "stderr": result.stderr}
```

## The `@mark` Decorator

Marks a function for automatic recovery. When the function raises an exception:

1. Calls `context_from` with the function's args and the exception to build context
2. Tries to find a matching rule (by name > by tags > by vector search)
3. If rule found: executes its action, retries the function
4. If no rule and `explorable=True` and `THEOW_EXPLORE=1`: runs LLM exploration
5. If LLM creates a rule: executes action, retries function
6. Repeats up to `max_retries` times with top rules or `max_tries` number of explorations

```python
@my_agent.mark(
    context_from=lambda x, exc: {
        "error_type": "build_failure",
        "stderr": exc.stderr,
        "workspace": str(x.path),
    },
    max_retries=3,
    explorable=True,
)
def build(x):
    result = run_build(x)
    if result.returncode != 0:
        raise BuildFailed(result.stderr)
```

Theow adds a traceback automatically. That said for better results, the function must raise a an exception with relevent information. If its a custom excpetion with minimal information, there might not be enough context to match rules or explore effectively.

**Parameters with their defaults:**

```python
@my_agent.mark(
    context_from=...,     # Callable: (args, kwargs, exception) -> dict. Can include any number of keys.
    max_retries=3,        # How many rules to try or explorations to do
    rules=["rule1"],      # Try these rules first (by name)
    tags=["go"],          # Then try rules with these tags
    fallback=True,        # Fall back to vector search
    explorable=False,     # Allow LLM exploration (also requires THEOW_EXPLORE=1 and an API key set)
    collection="default", # Chroma collection for rules. Recommended one per tagged function or module.
)
```

Run with exploration enabled:
```bash
THEOW_EXPLORE=1 python my_script.py
```

## Actions

Actions are what rules execute.:

```python
@my_agent.action("patch_config")
def patch_config(workspace: str, key: str, value: str):
    """Patch a config file."""
    # ...
```

Actions live in `.theow/actions/*.py` (auto-discovered):

```python
from theow import action

@action("fix_module_rename")
def fix_module_rename(workspace: str, old_path: str, new_path: str):
    """Add replace directive for renamed module."""
    subprocess.run(
        f"go mod edit -replace {old_path}={new_path}@latest",
        shell=True, cwd=workspace
    )
```

## Rules

Rules live in `.theow/rules/*.rule.yaml`:

```yaml
name: module_rename
description: Module path was renamed upstream
tags: [go, dep]

when:
  - fact: error_type
    equals: dep_resolution
  - fact: stderr
    regex: 'declares its path as: (?P<new_path>\S+).*required as: (?P<old_path>\S+)'
    examples:
      - "module declares its path as: dario.cat/mergo but was required as: github.com/imdario/mergo"

then:
  - action: fix_module_rename
    params:
      old_path: "{old_path}"
      new_path: "{new_path}"
```

**Matching:**
- `equals`: Exact match
- `contains`: Substring match
- `regex`: Regex with named captures (available as `{name}` in params)

**Examples** in facts improve vector search recall.

## Directory Structure

```
.theow/
├── rules/          # Rule YAML files
├── actions/        # Action Python files
├── prompts/        # Prompt templates for probabilistic rules
└── chroma/         # Vector DB (auto-managed)
```

## Probabilistic Rules

Rules can also run a LLM call on match:

```yaml
name: investigate_unknown
description: Unknown failure, use LLM to investigate

when:
  - fact: error_type
    equals: unknown

llm_config:
  prompt_template: file://prompts/investigate.md  # can be a file or a string
  tools: [read_file, run_command]
  constraints:
    max_tool_calls: 20
```

This would also need an `API` key set for the LLM provider. This LLM call will not create a new rule or action but directly act on the failure.
