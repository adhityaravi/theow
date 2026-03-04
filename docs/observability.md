# Observability

Theow instruments every LLM interaction with [Logfire](https://logfire.pydantic.dev/) / OpenTelemetry spans following the [`gen_ai` semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/). This gives you token usage, cost, tool call traces, and budget tracking across all gateway providers in a single dashboard.

## Setup

```python
from theow import Theow, LogfireConfig

# Quick enable — sends traces to Logfire dashboard
agent = Theow(llm="anthropic/claude-sonnet-4-20250514", logfire=True)

# OTel-only — reads OTEL_EXPORTER_* env vars, no Logfire dashboard
agent = Theow(
    llm="...",
    logfire=LogfireConfig(enabled=True, send_to_logfire=False),
)

# Custom service name
agent = Theow(
    llm="...",
    logfire=LogfireConfig(enabled=True, service_name="my-app"),
)
```

`LogfireConfig` fields:

| Field | Default | Description |
|---|---|---|
| `enabled` | `False` | Enable instrumentation |
| `send_to_logfire` | `True` | Send to Logfire dashboard. Set `False` for OTel-only (Jaeger, Grafana, etc.) |
| `service_name` | `None` | Defaults to the Theow engine name |

## Span Structure

Every `conversation()` and `generate()` call produces a root span with child spans for each tool call:

```
LLM conversation                          # root span (@instrumented)
├── tool call: search_files                # child span (span_tool_call)
├── tool call: read_file                   # child span
├── tool call: request_templates           # child span
└── tool call: submit_rule                 # child span — raises SubmitRule signal
```

### Root Span Attributes

Set by the `@instrumented` decorator on every gateway:

| Attribute | Example | Source |
|---|---|---|
| `gen_ai.operation.name` | `chat` | Method name (`conversation` → `chat`) |
| `gen_ai.request.model` | `claude-sonnet-4-20250514` | Gateway's model spec |
| `gen_ai.response.model` | `claude-sonnet-4-20250514` | Same, set after response |
| `gen_ai.system` | `anthropic`, `copilot` | `provider_name` property |
| `gen_ai.usage.input_tokens` | `41858` | Accumulated from SessionState |
| `gen_ai.usage.output_tokens` | `438` | Accumulated from SessionState |

Logfire uses `gen_ai.system` + model + token counts to auto-compute `operation_cost` for providers it knows (Anthropic, OpenAI, Google). This works because `provider_name` returns the actual provider, not the gateway name.

### Tool Call Span Attributes

Each tool call gets its own child span with cumulative token counts at the time of execution.

### Provider-Specific Attributes

Gateways can enrich the root span with provider-specific data via `_enrich_span()`. Currently only the Copilot gateway uses this:

| Attribute | Example | Description |
|---|---|---|
| `operation.cost` | `0.12` | Cost in USD (delta-based from premium request quota) |
| `copilot.premium_requests` | `3` | Premium requests consumed this session |
| `copilot.quota.remaining_pct` | `90.4` | Remaining quota percentage |
| `copilot.quota.used_requests` | `29` | Total used requests on the account |

## Cost Tracking

Cost computation varies by gateway:

**PydanticAI gateway** — Logfire auto-computes cost from the `gen_ai.*` attributes. This works for any provider Logfire has pricing data for (Anthropic, OpenAI, Google, etc.). No custom logic needed.

**Copilot gateway** — GitHub bills by premium requests, not tokens. The `CopilotUsageTracker` captures `premium_interactions` quota snapshots from SDK events and computes cost as:

```
session_cost = (used_requests_end - used_requests_start) × $0.04
```

The delta already reflects model multipliers applied server-side (e.g., Opus 4.6 = 3× = $0.12/interaction, GPT-5.3-Codex = 1× = $0.04).

## Budget Tracking

Budget state is tracked in `SessionState` and checked after every tool execution round:

- **Soft limit (80%)** — Injects a warning message into the conversation telling the LLM to wrap up
- **Hard limit (100%)** — Terminates the conversation loop and raises `GiveUp`

Both tool call count and token count are tracked. The `SessionState` feeds both budget decisions and span attributes.

## Exception Traces

When the LLM exhausts its budget, the gateway raises `GiveUp("Session budget exceeded")`. Since this happens inside the `@instrumented` span, Logfire records it as an exception trace — making budget exhaustion visible in the dashboard as a failed span rather than a silent timeout.

Signal tools (`_done`, `_give_up`, `_escalate`, etc.) also propagate as exceptions through `ExplorationSignal` subclasses, giving you a clear trace of how each exploration session ended.

## Architecture

Observability code lives in `src/theow/_gateway/_observability/`:

```
_observability/
├── __init__.py       # re-exports public API
├── _config.py        # LogfireConfig, configure_logfire()
├── _spans.py         # SessionState, @instrumented, span_tool_call()
└── _copilot.py       # CopilotUsageTracker (premium request cost)
```

The `@instrumented` decorator and `span_tool_call()` context manager are shared across all gateways. PydanticAI's auto-instrumentation (`logfire.instrument_pydantic_ai()`) is intentionally **not** enabled to avoid double-counted token usage — Theow's decorator handles this uniformly.
