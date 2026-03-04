# Tools

Tools are Python callables that the LLM can invoke during exploration. Theow provides built-in tools for filesystem access and rule management, plus signal tools that control the conversation flow. For structural code awareness, you can plug in [CodeGraph](codegraph.md) to give the LLM a `search_code` tool that queries symbols, call chains, and class hierarchies instead of reading entire files.

## Registering Tools

```python
from theow import Theow
from theow.tools import read_file, write_file, run_command, list_directory

agent = Theow(llm="anthropic/claude-sonnet-4-20250514")
agent.tool()(read_file)
agent.tool()(write_file)
agent.tool()(run_command)
```

Custom tools are regular functions. Theow generates JSON schema from the signature and docstring:

```python
@agent.tool()
def restart_service(name: str) -> str:
    """Restart a systemd service by name."""
    subprocess.run(["systemctl", "restart", name], check=True)
    return f"Restarted {name}"
```

The function name becomes the tool name. Override with `@agent.tool("custom_name")`.

### CodeGraph

For codebase-aware exploration, register the [CodeGraph](codegraph.md) `search_code` tool:

```python
from theow.codegraph import CodeGraph

graph = CodeGraph(root="./src")
agent.tool()(graph.search_code)
```

This gives the LLM a single tool that covers symbol lookup, caller/callee traversal, file listing, and path finding across your codebase (~260 tokens per query vs ~4000+ for reading whole files). Requires `pip install theow[codegraph]`.

## Built-in Tools

### Filesystem

| Tool | Signature | Description |
|---|---|---|
| `read_file` | `(path: str) -> str` | Read file contents |
| `write_file` | `(path: str, content: str) -> str` | Write content to file. Blocked for `.rule.yaml` and `actions/*.py` (use `_write_rule`/`_write_action` instead) |
| `run_command` | `(cmd: str, cwd: str \| None) -> dict` | Run shell command, returns `{returncode, stdout, stderr}` |
| `list_directory` | `(path: str) -> list[str]` | List files and directories at path |

These are not registered by default. Register the ones you need. The CLI registers all four automatically.

### Signal Tools

Signal tools control the exploration conversation flow. They raise `ExplorationSignal` exceptions that the explorer catches and dispatches.

| Tool | When to Use |
|---|---|
| `_done(message)` | You applied a fix. System retries the original operation to validate. |
| `_give_up(reason)` | Problem is fundamentally unautomatable (missing creds, needs human judgment). |
| `_request_templates()` | You found a fix and want to write a rule. System injects rule/action syntax. |
| `_submit_rule(rule_file, action_file)` | You wrote a rule. System validates facts against current context. |
| `_rule_resolved(summary)` | You augmented an existing rule to handle this case. |
| `_escalate(findings)` | You're stuck but believe the problem is solvable. Hands off to secondary model. |

Signal tools are added automatically during exploration. `_escalate` only appears when `allow_escalation=True` and a secondary gateway is configured.

### Search Tools

Created per-session, bound to the ChromaDB store:

| Tool | Description |
|---|---|
| `_search_rules(query)` | Semantic search over existing rules |
| `_search_actions(query)` | Semantic search over registered actions |
| `_list_rules()` | List all rules in the current collection |
| `_list_actions()` | List all action names |
| `_list_failed_rules()` | List previously failed rules with error context |
| `_search_observations(query)` | Search past LLM attempt outcomes |

### Ephemeral Tools

For reading and writing rule/action files during exploration:

| Tool | Description |
|---|---|
| `_list_ephemeral_rules()` | List rules from current/previous attempts (check for work to continue) |
| `_read_ephemeral_rule(name)` | Read full YAML content of an ephemeral rule |
| `_write_rule(name, content)` | Write a rule YAML to `rules/ephemeral/` |
| `_write_action(name, content)` | Write a Python action to `actions/` |

### Validation Tools

| Tool | Description |
|---|---|
| `_test_rule_match(rule_file)` | Test if a rule's facts match current context. Call before `_submit_rule()`. |

### Augmentation Tools

For extending existing permanent rules instead of creating duplicates:

| Tool | Description |
|---|---|
| `_add_fact_to_rule(rule_name, fact, ...)` | Add a new when-fact to a permanent rule. Makes the rule more specific. |
| `_add_example_to_rule(rule_name, fact_key, example)` | Add an example to improve vector search recall. Doesn't change matching. |

## Tool Sets by Mode

During **exploration** (rule creation), the LLM gets all tool categories: signal + search + ephemeral + validation + augmentation + caller-provided.

During **direct fix** (probabilistic rules via `run_direct()`), the LLM gets a simpler set: `_done`, `_give_up`, optionally `_escalate`, plus caller-provided tools.
