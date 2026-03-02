# Middleware

Middleware provides input and output guardrails for LLM conversations. Input guardrails check prompts before they're sent to the model. Output guardrails sanitize responses before they leave the explorer.

## Quick Start

```python
from theow import Theow

# Sensible defaults: prompt injection detection + secret redaction
agent = Theow(llm="anthropic/claude-sonnet-4-20250514", middleware=True)
```

Passing `middleware=True` enables:
- **Input**: `prompt_injection()` from `pydantic-ai-guardrails` — detects adversarial content in the exploration prompt (which includes error context and file contents that could be attacker-controlled).
- **Output**: `secret_redaction()` — scrubs secrets from LLM responses before they're archived to `observations.jsonl` or surfaced in logs.

## Custom Guardrails

```python
from theow._gateway._base import MiddlewareConfig
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

## How It Works

### Input Guardrails

Run in `Explorer._run_input_guardrails()` before the exploration prompt is sent to the gateway.

```
prompt built → input guardrails → gateway.conversation()
```

Each guardrail's `.validate(prompt, context)` is called. If any returns `tripwire_triggered: True`, exploration aborts and returns `(None, True)` — explored but no rule produced. The trigger is logged as a warning.

This matters because the exploration prompt contains user-supplied error context, file contents read by tools, and stack traces. Any of these could contain injected instructions if the error source is attacker-controlled.

### Output Guardrails

Run in `Explorer._run_output_guardrails()` on all assistant messages after the conversation ends, before the messages leave the explorer.

```
gateway.conversation() → signal handling → output guardrails → return
```

Each guardrail's `.validate(text, context)` is called. If triggered, the response text is replaced with the `sanitized` version from the guardrail result. This prevents secrets or sensitive data from leaking into archived observations or downstream consumers.
