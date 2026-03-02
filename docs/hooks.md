# Hooks

Hooks are callbacks that run before and after each recovery attempt. Use them to manage workspace state, log outcomes, or integrate with external systems.

## Setup and Teardown

```python
def my_setup(state: dict, attempt: int) -> dict | None:
    """Called before each recovery attempt.

    Args:
        state: Mutable dict persisted across attempts. First call gets
               the function's named arguments (from @mark) or empty dict (CLI).
        attempt: Attempt number (1-indexed).

    Returns:
        Updated state dict, or None to keep current state.
    """
    state["workspace"] = create_temp_workspace()
    return state


def my_teardown(state: dict, attempt: int, success: bool) -> None:
    """Called after each recovery attempt.

    Args:
        state: Same dict from setup.
        attempt: Attempt number.
        success: Whether this attempt fixed the problem.
    """
    cleanup_workspace(state.get("workspace"))
    if not success and "_give_up_reason" in state:
        log_failure(state["_give_up_reason"])


@agent.mark(
    context_from=lambda task, exc: {"error": str(exc)},
    explorable=True,
    setup=my_setup,
    teardown=my_teardown,
)
def process(task):
    ...
```

## Hook State

The `state` dict flows through the entire recovery session:

1. For `@mark`, it's pre-populated with the decorated function's named arguments.
2. `setup` can modify and return it.
3. `teardown` receives the same dict with recovery metadata injected.

### Injected Keys

The recovery loop injects these keys into state before calling teardown:

| Key | Type | Description |
|---|---|---|
| `_give_up_reason` | `str` | Reason from `_give_up()` signal, if the LLM gave up |
| `_observation` | `LLMRecord` | Structured record of the LLM attempt (when `archive_llm_attempt=True`) |
| `_fn_name` | `str` | Name of the decorated function (from `@mark`) |

### Enriching Observations

When `archive_llm_attempt=True`, teardown can enrich the `LLMRecord` before it's flushed to `observations.jsonl`:

```python
def my_teardown(state: dict, attempt: int, success: bool) -> None:
    obs = state.get("_observation")
    if obs:
        obs.enrich(task_id=state.get("task_id"), environment="prod")
```

The enriched fields are included in the JSON output.

### Suppressing Exceptions

By default, teardown exceptions are caught and logged as warnings. To let them propagate:

```python
def my_teardown(state: dict, attempt: int, success: bool) -> None:
    state["suppress_exc"] = False
    raise CriticalError("Must stop")
```

## Deep Recovery

Deep recovery is theow's handling of cascading errors. When a fix resolves the original error but reveals a new one underneath, the recovery loop detects this as **progress** and continues.

Progress detection: after applying a rule and re-running the function, the loop checks whether the rule still matches the new context. If it doesn't match (or matches with different captures), the original error is gone and a new error appeared. The loop:

1. Promotes the successful rule (moves from ephemeral to permanent, indexes in ChromaDB).
2. Calls teardown for the current attempt.
3. Resets retry counter to 0, increments depth counter.
4. Clears failed rules list and rejected attempts.
5. Continues recovery against the new error, up to `max_depth` times.

Cycle detection prevents infinite loops: the loop hashes each error context and aborts if it sees the same fingerprint twice.

```python
@agent.mark(
    context_from=lambda task, exc: {"error": str(exc)},
    max_retries=3,   # rules to try per error
    max_depth=3,     # cascading errors to chase
    explorable=True,
)
def deploy(task):
    ...
```

With `max_retries=3` and `max_depth=3`, theow can try up to 3 rules for the first error, then if a fix reveals error #2, try 3 more rules for that, and so on up to 3 levels deep.

## CLI Hooks

The CLI uses the same hook mechanism through [plugins](cli.md#plugins). A plugin file exports `setup` and/or `teardown` functions:

```python
# my_plugin.py
from theow import tool

@tool()
def check_health(url: str) -> dict:
    """Check service health endpoint."""
    ...

def setup(state: dict, attempt: int) -> dict:
    state["started_at"] = time.time()
    return state

def teardown(state: dict, attempt: int, success: bool) -> None:
    elapsed = time.time() - state.get("started_at", 0)
    print(f"Attempt {attempt}: {'ok' if success else 'fail'} ({elapsed:.1f}s)")
```

```bash
theow run --plugin my_plugin.py -- make deploy
```
