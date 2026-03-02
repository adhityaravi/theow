# Exploration

Exploration is the process by which theow uses an LLM to diagnose a novel error and produce a [rule-action pair](rules-and-actions.md). It runs when the resolver finds no matching rule and `explorable=True`.

## Flow

1. **Gate checks.** Gateway must be configured, session count must be under `session_limit`, and `THEOW_EXPLORE=1` must be set.

2. **Session cache.** If this exact context was already explored in the current session, return the cached rule.

3. **ChromaDB check.** Semantic search for a close match (distance < 0.3). If found and facts match, return it without spending an LLM call.

4. **Lock permanent files.** All rules in `rules/` (excluding `ephemeral/`) and all actions in `actions/` are made read-only. This prevents the LLM from modifying proven rules.

5. **Build tool set.** Six categories assembled in order: signal tools, search tools, ephemeral tools, validation tools, augmentation tools, caller-provided tools. If `allow_escalation=True` with a secondary gateway, `_escalate` is added.

6. **Build prompt.** The intro prompt describes available tools, the rules directory, and the exploration workflow. The error prompt contains the context dict, traceback, attempt continuation info, and any rejected attempts from previous tries. If `hint` is set, it's appended as a caller constraints section.

7. **Input guardrails.** If [middleware](middleware.md) is configured, input guardrails run on the prompt. If any trigger, exploration aborts.

8. **Conversation.** The prompt is sent to the gateway. The gateway runs its conversation loop (tool calls, budget warnings, nudging) until the LLM raises a signal or budget runs out. If the primary gateway fails, theow cascades to the secondary.

9. **Signal dispatch.** The explorer pattern-matches on the signal type. See [signals](#signal-handling) below.

10. **Output guardrails.** If middleware is configured, output guardrails run on all assistant messages before they leave the explorer.

11. **Gateway reset.** All gateways are reset to clean up any internal state.

12. **Cache result.** If a rule was produced, it's stored in the session cache.

## Signal Handling

The explorer uses `match`/`case` to dispatch signals from `_handle_signal()`:

**`None` (budget exhausted)** — Look for orphaned rule files written during the session but never submitted. Tag them `[incomplete]` so the next attempt can continue.

**`GiveUp(reason)`** — Store the reason. Exploration stops. The recovery loop can surface this to teardown hooks.

**`RuleResolved(summary)`** — The LLM augmented an existing rule. Re-check ChromaDB to confirm the rule now matches.

**`Escalate(findings)`** — Start a fresh conversation on the secondary gateway with the original prompt plus the findings. The `_escalate` tool is stripped from the escalated conversation to prevent chains. The resulting signal is dispatched recursively.

**`RequestTemplates()`** — The LLM investigated and found a fix. Inject rule/action YAML syntax templates plus action design constraints into the conversation, then resume.

**`SubmitRule(rule_file, action_file)`** — Validate the rule: check it's in `ephemeral/`, parse YAML, verify all facts match current context, load the action file, check for conflicts. If validation fails and `allow_escalation=True`, auto-escalate. On success, return the bound rule to the recovery loop for execution and verification.

## Session Management

Each `explore()` call increments `session_count`. When it reaches `session_limit` (default 20), exploration is refused entirely.

`reset_session()` resets the counter and clears the session cache. Called explicitly when needed.

## Escalation

Escalation happens in three scenarios:

1. **LLM calls `_escalate(findings)`** during exploration. Findings are forwarded to the secondary gateway.
2. **Rule validation fails** after `_submit_rule()`. If `allow_escalation=True`, the explorer auto-escalates with context about what was attempted.
3. **Action execution fails** in the recovery loop. The loop flags `_escalate_next=True` so the next exploration goes straight to the secondary model.

The escalated conversation receives the full original prompt plus an escalation context section. The `_escalate` tool is removed to prevent recursive escalation.

## Direct Fix Mode

`run_direct()` is a simpler conversation mode used by probabilistic rules and `Theow.run()`. Instead of the full exploration tool set, the LLM gets `_done`, `_give_up`, optionally `_escalate`, plus caller-provided tools. No rule creation, no templates. Returns `True` if the LLM signaled `Done`, `False` otherwise.
