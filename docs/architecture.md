# Theow Architecture

![Theow Workflow](../assets/theow.excalidraw.svg)

Theow is organized in three layers: the **engine** (`Theow`), the **explorer** (`Explorer`), and the **gateway** (`LLMGateway` implementations). Each layer has a distinct responsibility, and they compose together through dependency injection.

## The Conversation Loop

Every LLM interaction follows a conversation loop inside the gateway's `conversation()` method:

1. **Initialize state.** Extract per-session budget limits from the budget dict and create a `SessionState` to track `tool_calls`, `tokens_used`, and `warned_about_budget`.

2. **Call the model.** Send the message history and tool declarations to the LLM provider. Each provider uses its own wire format (Anthropic content blocks, PydanticAI `ModelMessage` types, Gemini `Content` objects, Copilot session prompts), but the loop structure is the same.

3. **Check the response.** If the LLM responded with tool calls, execute them. If it responded with text only, apply the [text nudge mechanism](#text-nudging). If no response, break.

4. **Execute tools.** Resolve tool names against a `tool_map` and call each function. If a tool raises an `ExplorationSignal`, the gateway captures it, fills placeholder results for remaining tool calls in the batch, appends all results to the message history, and re-raises the signal. This keeps the message history well-formed even when a signal interrupts mid-batch.

5. **Check budget.** After each tool execution round, `check_budget_warning()` evaluates whether the session has hit the soft limit. If so, a warning message is injected into the conversation.

6. **Repeat** until a signal is raised, the budget is exhausted, or the LLM stops producing actionable responses.

The loop returns a `ConversationResult` containing the final message list, total tool calls, and total tokens used. Signals propagate as exceptions caught by the explorer with `try/except ExplorationSignal`.

## Signals

Signals are Python exceptions inheriting from `ExplorationSignal`. The LLM invokes them by calling special signal tools (e.g., `_done()`, `_give_up()`). The explorer pattern-matches on the signal type to decide what to do next.

| Signal | Carries | Meaning |
|---|---|---|
| `Done(message)` | Summary string | LLM completed a direct fix. System retries the original operation to validate. |
| `GiveUp(reason)` | Reason string | Problem cannot or should not be automated. Exploration stops. |
| `RequestTemplates()` | Nothing | LLM investigated, found a fix, and is ready to write a rule. Explorer injects rule/action template syntax and resumes the loop. |
| `SubmitRule(rule_file, action_file)` | File paths | LLM wrote a rule. Explorer validates facts against current context, checks for conflicts, returns rule for execution. |
| `Escalate(findings)` | Analysis text | LLM is stuck but believes problem is solvable. Findings forwarded to secondary gateway. `_escalate` tool stripped from escalated conversation to prevent chains. |
| `RuleResolved(summary)` | Summary string | LLM augmented an existing rule to handle the current case. Explorer re-checks Chroma to confirm. |

The explorer's `_handle_signal()` uses `match`/`case` to dispatch. Some signals are recursive: `RequestTemplates` injects templates and re-enters the conversation loop, `Escalate` starts a fresh conversation on the secondary gateway. `SubmitRule` with failed validation can auto-escalate if `allow_escalation` is enabled.

When no signal is received (budget ran out), the explorer tags orphaned rule files in the ephemeral directory as `[incomplete]` so subsequent attempts can continue.

## Budget Tracking

Budget parameters are passed as a dict to the gateway:

```python
{
    "max_tool_calls_per_session": 30,   # default
    "max_tokens_per_session": 8192,     # default
}
```

**Soft limit at 80%.** `check_budget_warning()` triggers when either resource hits 80% consumption. The warning is injected as a user message instructing the LLM to wrap up, suggesting the `request_templates -> write_rule -> test_rule_match -> submit_rule` workflow. If escalation is allowed, it mentions `_escalate()`. Fires at most once per session.

**Hard limit.** The loop's `while` condition checks both counters on every iteration. Once exceeded, the loop terminates and the explorer interprets this as "budget exhausted." The Copilot gateway is a special case: it calls `session.abort()` to forcibly terminate the SDK's own agentic loop.

**Session limit.** Beyond per-session budgets, the explorer tracks `session_count` against `session_limit` (default 20). Once reached, exploration is refused entirely.

## Text Nudging

When the LLM responds with text instead of calling a tool, theow nudges it back on track:

1. If the response contains only text, increment a counter and append a nudge message telling the LLM to call `_done()` or `_give_up()`.
2. Repeat up to `MAX_TEXT_NUDGES = 2` times.
3. If the LLM still responds with text after 2 nudges, the loop breaks.

All gateway implementations share the same `MAX_TEXT_NUDGES` constant and `TEXT_REPLY_NUDGE` message from `_base.py`.

## The Gateway Abstraction

```python
class LLMGateway(ABC):
    @abstractmethod
    def conversation(self, messages, tools, budget) -> ConversationResult: ...

    @abstractmethod
    def generate(self, prompt, schema=None) -> dict[str, Any]: ...

    def reset(self) -> None: ...
```

The base class provides shared helpers: `check_budget_warning()`, `_build_tool_map()`, `_extract_budget()`, and `_execute_tool()`.

**PydanticAIGateway** (default). Uses `pydantic_ai.direct.model_request_sync()` for unified access to 15+ providers. Converts theow's dict messages to PydanticAI's `ModelMessage` types at the boundary and syncs back after the conversation. Tool declarations use `ToolDefinition` with `schema_key="parameters_json_schema"`.

**Native gateways** (Anthropic, Gemini). Use provider SDKs directly. Deprecated in favor of PydanticAIGateway, emit `DeprecationWarning`. AnthropicGateway uses `tool_use`/`tool_result` content blocks. GeminiGateway uses `types.Content` objects with `role="tool"`.

**CopilotGateway**. The Copilot SDK is async-first and manages its own agentic loop, so theow registers tool handlers with the SDK and triggers via `session.send_and_wait()`. Signal tools raise exceptions inside handlers; the gateway stores the signal and calls `session.abort()`. Maintains a persistent `asyncio` event loop across turns. `reset()` destroys the session and closes the loop.

**Gateway factory.** `create_gateway("provider/model")` parses the spec and routes: `copilot/*` always uses `CopilotGateway`, `GatewayProvider.NATIVE` routes to native gateways, `GatewayProvider.PYDANTIC` (default) translates `"provider/model"` to `"provider:model"` with alias mapping (e.g. `gemini` -> `google-gla`).

## How the Three Layers Connect

**Engine (Theow).** Entry point. Owns the `.theow/` directory structure, ChromaStore, ActionRegistry, Resolver, and Explorer. Lazily creates the gateway via `_ensure_gateway()`. Passes configuration (session limits, budgets, middleware) down to the Explorer at construction time.

**Explorer.** Orchestrates multi-session LLM exploration: checks session cache, queries Chroma, assembles tool sets (signal + search + ephemeral + validation + augmentation + caller-provided), builds the initial prompt, runs guardrails, calls `_converse()`, dispatches signals via `_handle_signal()`, and resets gateways. Also supports `run_direct()` mode for probabilistic rules (simpler signal set, returns boolean).

**Gateway.** Lowest layer. Responsible for a single conversation session: provider API interaction, tool execution loop, text nudging, budget warnings, signal propagation. Stateless in most implementations (state lives in the message list). Gemini and Copilot are exceptions with internal state cleaned up in `reset()`.
