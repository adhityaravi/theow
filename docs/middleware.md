# Middleware

Middleware provides input and output guardrails for LLM conversations. Input guardrails check prompts before they're sent to the model. Output guardrails sanitize responses before they leave the explorer.

Theow's guardrails are powered by [pydantic-ai-guardrails](https://github.com/pydantic/pydantic-ai-guardrails), a companion package to PydanticAI that provides pluggable validation for LLM inputs and outputs.

## Quick Start

```python
from theow import Theow

# Sensible defaults: prompt injection detection + secret redaction
agent = Theow(llm="anthropic/claude-sonnet-4-20250514", middleware=True)
```

Passing `middleware=True` enables:
- **Input**: `prompt_injection()` from `pydantic-ai-guardrails` — detects adversarial content in the exploration prompt (which includes error context and file contents that could be attacker-controlled).
- **Output**: `secret_redaction()` — scrubs secrets from LLM responses before they're archived to `observations.jsonl` or surfaced in logs.

## Configuration

```python
from theow import MiddlewareConfig
from pydantic_ai_guardrails.guardrails.input import prompt_injection
from pydantic_ai_guardrails.guardrails.output import secret_redaction

agent = Theow(
    llm="anthropic/claude-sonnet-4-20250514",
    middleware=MiddlewareConfig(
        input_guardrails=[prompt_injection()],
        output_guardrails=[secret_redaction()],
    ),
)
```

Both lists accept any guardrail from the `pydantic-ai-guardrails` package. Pass `None` or `[]` to disable a category.

## Custom Guardrails

Write your own guardrails using `InputGuardrail` and `OutputGuardrail`:

```python
from pydantic_ai_guardrails import GuardrailResult, InputGuardrail, OutputGuardrail

async def block_production_paths(prompt: str) -> GuardrailResult:
    if "/prod/" in prompt or "production" in prompt.lower():
        return {
            "tripwire_triggered": True,
            "message": "Blocked: prompt references production environment",
        }
    return {"tripwire_triggered": False}

async def redact_internal_urls(text: str) -> GuardrailResult:
    if "internal.corp.com" in text:
        return {
            "tripwire_triggered": True,
            "sanitized": text.replace("internal.corp.com", "[REDACTED]"),
        }
    return {"tripwire_triggered": False}

agent = Theow(
    llm="anthropic/claude-sonnet-4-20250514",
    middleware=MiddlewareConfig(
        input_guardrails=[prompt_injection(), InputGuardrail(block_production_paths)],
        output_guardrails=[secret_redaction(), OutputGuardrail(redact_internal_urls)],
    ),
)
```

Custom guardrails mix freely with the built-in ones. See the [pydantic-ai-guardrails documentation](https://jagreehal.github.io/pydantic-ai-guardrails/) for the full API.

## How It Works

### Input Guardrails

Run in `Explorer._run_input_guardrails()` before the exploration prompt is sent to the gateway.

```
prompt built → input guardrails → gateway.conversation()
```

Each guardrail's `.validate(prompt, context)` is called. If any returns `tripwire_triggered: True`, exploration aborts and returns `(None, True)` — explored but no rule produced. The trigger is logged as a warning.

The exploration prompt contains user-supplied error context, file contents read by tools, and stack traces. Any of these could contain injected instructions if the error source is attacker-controlled.

### Output Guardrails

Run in `Explorer._run_output_guardrails()` on all assistant messages after the conversation ends, before the messages leave the explorer.

```
gateway.conversation() → signal handling → output guardrails → return
```

Each guardrail's `.validate(text, context)` is called. If triggered, the response text is replaced with the `sanitized` version from the guardrail result. This prevents secrets or sensitive data from leaking into archived observations or downstream consumers.

## Future: Governance Middleware

The `MiddlewareConfig` is designed to extend beyond guardrails. When PydanticAI adds native support for governance middleware (policy enforcement, audit logging, rate limiting, content filtering at the framework level), theow will integrate it into the same config surface. The current `input_guardrails`/`output_guardrails` lists will coexist alongside governance hooks once available upstream.
