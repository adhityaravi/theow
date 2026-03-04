# Configuration

## Engine Parameters

The `Theow` constructor accepts all engine-level settings:

```python
from theow import Theow
from theow import GatewayProvider, LogfireConfig, MiddlewareConfig

agent = Theow(
    theow_dir=".theow",                     # working directory
    name="Theow",                            # engine name (used in logs)
    llm="anthropic/claude-sonnet-4-20250514",# primary model
    llm_secondary="anthropic/claude-opus-4-6",# escalation model (optional)
    session_limit=20,                        # max exploration sessions
    max_tool_calls_per_session=30,           # tool call budget per session
    max_tokens_per_session=8192,             # token budget per session
    archive_llm_attempt=False,               # log LLM outcomes to observations.jsonl
    gateway_provider=GatewayProvider.PYDANTIC,# routing backend
    logfire=False,                           # OpenTelemetry instrumentation
    middleware=False,                        # input/output guardrails
)
```

### Model Spec

The `llm` parameter uses `provider/model` format. The factory splits on the first `/` and routes accordingly:

| Spec | Gateway | Notes |
|---|---|---|
| `anthropic/claude-sonnet-4-20250514` | PydanticAIGateway | Translates to `anthropic:claude-sonnet-4-20250514` |
| `gemini/gemini-2.5-pro` | PydanticAIGateway | Translates to `google-gla:gemini-2.5-pro` |
| `openai/gpt-5` | PydanticAIGateway | Translates to `openai:gpt-5` |
| `github/openai/gpt-5` | PydanticAIGateway | Translates to `github:openai/gpt-5` |
| `bedrock/anthropic.claude-v3` | PydanticAIGateway | Translates to `bedrock:anthropic.claude-v3` |
| `copilot/gpt-5.3-codex` | CopilotGateway | Always uses native Copilot SDK |

Any provider PydanticAI supports works with no extra configuration. Gemini has an alias (`gemini` -> `google-gla`) since theow uses `gemini/` while PydanticAI uses `google-gla:`.

### API Keys

Each provider reads its own environment variable natively through PydanticAI:

```bash
ANTHROPIC_API_KEY=sk-...
GOOGLE_API_KEY=...
OPENAI_API_KEY=sk-...
GITHUB_TOKEN=ghp_...
```

No key configuration in code. Set the env var and go.

### Gateway Provider

```python
from theow import GatewayProvider
```

| Value | Behaviour |
|---|---|
| `GatewayProvider.PYDANTIC` | Default. Routes through PydanticAI's `model_request_sync()`. Supports 15+ providers. |
| `GatewayProvider.NATIVE` | Uses the original per-provider gateway classes (Anthropic, Gemini). Emits `DeprecationWarning`. |

Copilot always uses its native SDK regardless of this setting.

### Model Escalation

Set `llm_secondary` to enable escalation. When the primary model gets stuck during exploration, it can call `_escalate(findings)` to hand off to the secondary:

```python
agent = Theow(
    llm="anthropic/claude-sonnet-4-20250514",      # cheap, fast
    llm_secondary="anthropic/claude-opus-4-6",  # strong fallback
)
```

Escalation also triggers automatically when a rule's action fails and `allow_escalation=True` on the mark decorator.

### Logfire / OpenTelemetry

```python
# Quick enable (sends to Logfire dashboard)
agent = Theow(llm="...", logfire=True)

# OTel-only (reads OTEL_* env vars, no Logfire dashboard)
agent = Theow(llm="...", logfire=LogfireConfig(enabled=True, send_to_logfire=False))
```

When enabled, every LLM call gets a logfire span with `gen_ai.*` semantic convention attributes, per-tool-call child spans, token usage, and cost tracking. See [observability](observability.md) for the full span structure and cost computation details.

### Middleware

```python
# Sensible defaults: prompt injection detection + secret redaction
agent = Theow(llm="...", middleware=True)

# Custom guardrails
from pydantic_ai_guardrails.guardrails.input import prompt_injection
from pydantic_ai_guardrails.guardrails.output import secret_redaction

agent = Theow(
    llm="...",
    middleware=MiddlewareConfig(
        input_guardrails=[prompt_injection()],
        output_guardrails=[secret_redaction()],
    ),
)
```

Input guardrails run before each exploration conversation. Output guardrails run on LLM responses before archiving. See [middleware](middleware.md) for details.

## Recovery Parameters

The `@agent.mark()` decorator controls per-function recovery behaviour:

```python
@agent.mark(
    context_from=lambda task, exc: {"error": str(exc)},
    max_retries=3,          # rules to try per error
    max_depth=3,            # chase cascading errors
    rules=None,             # explicit rule names to try first
    tags=None,              # filter rules by tag
    fallback=True,          # fall back to semantic search if name/tag miss
    explorable=False,       # allow LLM exploration on novel errors
    collection="default",   # ChromaDB collection for rule matching
    hint=None,              # extra context injected into exploration prompt
    allow_escalation=False, # allow escalation to secondary model
    setup=None,             # before-attempt hook
    teardown=None,          # after-attempt hook
)
def process(task):
    ...
```

| Parameter | Default | Description |
|---|---|---|
| `context_from` | required | Callable that builds the error context dict from the function's arguments and the exception |
| `max_retries` | `3` | How many different rules to try for the same error before giving up |
| `max_depth` | `3` | How many cascading errors to chase (when a fix reveals a new error) |
| `rules` | `None` | Try these rule names first, in order |
| `tags` | `None` | Filter candidate rules to those with matching tags |
| `fallback` | `True` | Fall back to ChromaDB semantic search when name/tag matching finds nothing |
| `explorable` | `False` | Allow LLM exploration when no rule matches. Requires `THEOW_EXPLORE=1` env var |
| `collection` | `"default"` | ChromaDB collection name for rule storage and lookup |
| `hint` | `None` | Free-text hint injected into the exploration prompt for additional context |
| `allow_escalation` | `False` | Allow escalation to `llm_secondary` when primary gets stuck |
| `setup` | `None` | Hook called before each recovery attempt: `(state, attempt_num) -> state` |
| `teardown` | `None` | Hook called after each attempt: `(state, attempt_num, success) -> None` |

## Environment Variables

| Variable | Description |
|---|---|
| `THEOW_EXPLORE` | Set to `"1"` to enable LLM exploration. Required even when `explorable=True`. |
| `THEOW_DIR` | Default `.theow` directory path (CLI only, overridden by `--theow-dir`). |
| `ANTHROPIC_API_KEY` | API key for Anthropic models. |
| `GOOGLE_API_KEY` | API key for Google/Gemini models. |
| `OPENAI_API_KEY` | API key for OpenAI models. |
| `GITHUB_TOKEN` | Token for GitHub Copilot SDK. |

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
```

Use profiles from the CLI:

```bash
theow run --profile deploy -- make deploy
theow run --profile tests -- pytest tests/
```

CLI flags override profile values. See [cli](cli.md) for the full flag reference.

## Directory Structure

Theow creates and manages `.theow/` with this layout:

```
.theow/
  rules/           # permanent rule YAML files
    ephemeral/     # rules created by LLM, not yet verified
  actions/         # Python action scripts
  prompts/         # prompt templates for LLM actions
  chroma/          # ChromaDB vector store
  config.yaml      # engine + profile configuration (CLI)
  observations.jsonl  # LLM outcome log (when archive_llm_attempt=True)
```

Rules start in `rules/ephemeral/` when created by the explorer. After verification succeeds, they're promoted to `rules/`. See [rules and actions](rules-and-actions.md) for the file formats.

## API Reference

### `Theow.run(prompt, tools, max_tool_calls, max_tokens) -> bool`

Run a one-shot LLM conversation with tools. Give it a prompt and tools, theow handles the conversation loop.

```python
agent.run("Fix the broken config file in ./config/", tools=agent.get_tools())
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompt` | `str` | required | The task prompt for the LLM |
| `tools` | `list[Callable]` | `None` | Tools the LLM can call. Defaults to all registered tools. |
| `max_tool_calls` | `int` | `30` | Tool call budget |
| `max_tokens` | `int` | `8192` | Token budget |

Returns `True` if the LLM signaled `Done`, `False` otherwise.

### `Theow.explore(context, tools, collection, tracing) -> Rule | None`

Explore a novel error using LLM. The explorer diagnoses the problem, writes a rule-action pair, and returns a validated rule. Does not execute the rule.

```python
context = {
    "error": "FileNotFoundError: config.yaml not found",
    "stderr": "Traceback ...\nFileNotFoundError: config.yaml not found",
}
rule = agent.explore(context, tools=agent.get_tools())
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `context` | `dict[str, Any]` | required | Error context dict (the same format `context_from` produces) |
| `tools` | `list[Callable]` | required | Tools the LLM can use during exploration |
| `collection` | `str` | `"default"` | ChromaDB collection for rule storage |
| `tracing` | `TracingInfo` | `None` | Python traceback and exception info |

Returns a `Rule` if exploration produced one, `None` otherwise. The returned rule is ephemeral and unverified. Execute it and re-run the failing operation to validate.

### `Theow.resolve(context, ...) -> Rule | None`

Match a context dict against existing rules. No LLM call, no tokens spent.

```python
rule = agent.resolve(context)
```

Resolution order:
1. Explicit rules by name (if `rules` is set)
2. Rules matching tags (if `tags` is set)
3. ChromaDB vector search fallback (if `fallback=True`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `context` | `dict[str, Any]` | required | Error context dict to match against |
| `collection` | `str` | `"default"` | ChromaDB collection to search |
| `rules` | `list[str]` | `None` | Explicit rule names to try first, in order |
| `tags` | `list[str]` | `None` | Filter rules by tag |
| `fallback` | `bool` | `True` | Fall back to semantic search if name/tag matching finds nothing |
| `n_results` | `int` | `10` | Max candidates to retrieve from vector search |
| `exclude_rules` | `list[str]` | `None` | Rule names to skip (already tried and failed) |

Returns a bound `Rule` ready for execution, or `None` if no match.

### `Theow.execute_rule(rule, context, escalation_context) -> bool`

Execute a resolved rule's action.

```python
rule = agent.resolve(context)
if rule:
    success = agent.execute_rule(rule, context)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `rule` | `Rule` | required | A bound rule from `resolve()` or `explore()` |
| `context` | `dict[str, Any]` | `None` | Error context (used for placeholder resolution in probabilistic rules) |
| `escalation_context` | `str` | `None` | If set, skip primary model and run directly on secondary with this context |

Returns `True` if the action succeeded, `False` otherwise. For deterministic rules, runs the Python action. For probabilistic rules, starts an LLM conversation with the configured prompt.
