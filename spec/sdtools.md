# Theow × sd-tools — Integration Design

> How sd-tools integrates Theow for automated failure resolution.

---

## 1. Context

sd-tools onboards ~8500 Go packages into Superdistro (more packages and languages coming). Two phases break most often:

**Prepare phase** (dependency tree generation) where `go mod tidy -e` fails or git operations fail. Causes:
- Tags deleted from upstream repos
- Repos renamed or deleted
- Module path renames (declares path X but required as Y)
- Commit hashes rewritten
- Monorepo tagging schemes not handled

**Onboard phase** (11-step pipeline) where tasks fail. Top failures:
- `sourcecraft-pack` — build failures
- `sourcecraft-test` — test failures
- `early-test` — pre-onboarding test failures

Currently handled with hardcoded heuristics:
- `_try_resolve_module_renames()` in `tree_resolver.py`
- `patch_failed_tests()` in `spread_yaml_gen.py` (go version, vet failure)

Theow replaces hardcoded heuristics with a learning rule engine. Novel failures trigger exploration, producing deterministic rules for future use.

---

## 2. Boundary

| | sd-tools | Theow |
|---|---|---|
| **Owns** | Rules, actions, tools, prompts, LLM config, exploration targets, pipeline, database, git, PRs | Matching engine, Chroma indexing, LLM gateway orchestration, learning loop mechanics |
| **Does** | Defines domain-specific rules/actions/tools, calls Theow at failure points, reviews proposed rules | Searches rules, executes tool loops, proposes new rules as artifacts |

**Theow provides:** The framework. Chroma wrapper, LLM gateway, `@theow.mark()` decorator, rule/action loading.

**sd-tools provides:** Everything domain-specific. Rules, actions, tools, prompts, configuration.

---

## 3. Integration Points

### 3.1 Overview

| Location | Failure Type | Pattern | Exploration |
|----------|--------------|---------|-------------|
| `RepoCache.get()` | Git/tag failures | `@theow.mark` on `_get_impl` | Yes |
| `_get_deps_for_single_package()` | go mod tidy failures | `@theow.mark` on `_get_deps_impl` | Yes |
| `run_tests` | Test failures | `@theow.mark` on `_run_tests_impl` | Yes |
| `early_test` | Early test failures | `@theow.mark` on `_early_test_impl` | Yes |
| `pack` | Build failures | `@theow.mark` on `_pack_impl` | Yes |

### 3.2 Abstraction Pattern

sd-tools code stays clean. Extract failure-prone logic into private methods, mark those with Theow:

```python
# Outer method: sd-tools interface unchanged
@task("sourcecraft-test")
def run_tests(tree: DependencyTree, ob: Onboarder):
    try:
        _run_tests_impl(tree, ob)
        return TaskOutcome.PASS
    except SourcecraftTestFailed:
        return TaskOutcome.FAIL

# Inner method: Theow-managed
@theow.mark(
    context_from=lambda tree, ob, exc: {
        "problem_type": "test_failure",
        "package": tree.package.name,
        "tag": tree.package.tag,
        "stderr": exc.stderr,  # Primary: longest string, used for semantic search
        "go_mod": (tree.package.package_path / "go.mod").read_text(),
        "workspace": str(tree.package.cache_path),
    },
    max_retries=5,
    tags=["go", "test"],
    explorable=True,
)
def _run_tests_impl(tree: DependencyTree, ob: Onboarder):
    out = ob.run(tree, "sourcecraft test", timeout=900)
    if out.returncode != 0:
        raise SourcecraftTestFailed(out.stderr)
```

**Context design:** `stderr` is the verbose field for semantic matching. Other fields (`go_mod`, `workspace`) are for param extraction and action context, not for similarity search.

The `@task` decorator is oblivious to Theow. It just sees success (no exception) or failure (exception caught).

### 3.3 Repo Acquisition

```python
# In repo_cache.py
def get(self, source: Package, files=None, fetch=True) -> bool:
    try:
        return self._get_impl(source, files, fetch)
    except RepoAcquisitionFailed:
        return False

@theow.mark(
    context_from=lambda self, source, files, fetch, exc: {
        "problem_type": "repo_acquisition",
        "package": source.name,
        "tag": source.tag,
        "upstream_source": source.upstream_source,
        "error": str(exc),  # Primary for semantic search
        "workspace": str(self.pkg_cache_path(source.upstream_source)),
    },
    max_retries=3,
    tags=["go", "git", "repo"],
    explorable=True,
)
def _get_impl(self, source: Package, files, fetch) -> bool:
    if self._get_from_git(source, files, fetch):
        return True
    raise RepoAcquisitionFailed(f"failed to acquire {source.name}@{source.tag}")
```

### 3.4 Dependency Resolution

```python
# In tree_resolver.py
@theow.mark(
    context_from=lambda self, package, exc: {
        "problem_type": "dep_resolution",
        "package": package.name,
        "tag": package.tag,
        "stderr": exc.stderr if hasattr(exc, 'stderr') else str(exc),
        "go_mod": (package.package_path / "go.mod").read_text(),
        "workspace": str(package.package_path),
    },
    max_retries=5,
    tags=["go", "dep_resolution"],
    explorable=True,
)
def _get_deps_impl(self, package: Package):
    # ... existing logic but throws on failure
    proc = run_cmd(cmd="go mod tidy -e", cwd=pkg_path)
    if proc.returncode != 0 or self._has_resolution_errors(proc.stderr):
        raise DepResolutionFailed(proc.stderr)
    # ...
```

### 3.5 Build (Pack)

```python
@task("sourcecraft-pack")
def pack(tree: DependencyTree, ob: Onboarder):
    try:
        _pack_impl(tree, ob)
        return TaskOutcome.PASS
    except SourcecraftPackFailed:
        return TaskOutcome.FAIL

@theow.mark(
    context_from=lambda tree, ob, exc: {
        "problem_type": "build_failure",
        "package": tree.package.name,
        "stderr": exc.stderr,
        "workspace": str(tree.package.cache_path),
    },
    max_retries=3,
    tags=["go", "build"],
    explorable=True,
)
def _pack_impl(tree: DependencyTree, ob: Onboarder):
    git = Git(tree.package.cache_path)
    git.sparse_checkout_disable()
    out = ob.run(tree, "sourcecraft pack", timeout=600)
    if out.returncode != 0:
        raise SourcecraftPackFailed(out.stderr)
```

---

## 4. Tools

Four tools cover all exploration needs:

```python
# .theow/tools.py (or wherever sd-tools registers them)

@theow.tool()
def read_file(path: str) -> str:
    """Read file contents from workspace."""
    return Path(path).read_text()

@theow.tool()
def write_file(path: str, content: str) -> dict:
    """Write content to file in workspace."""
    Path(path).write_text(content)
    return {"status": "ok", "path": path}

@theow.tool()
def run_command(cmd: str, cwd: str, timeout: int = 300) -> dict:
    """Run shell command in workspace."""
    result = subprocess.run(
        cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }

@theow.tool()
def create_sandbox(package_name: str, upstream_source: str, tag: str) -> str:
    """Clone package to isolated workspace for safe experimentation."""
    workspace = Path(f"/tmp/theow_sandbox/{package_name.replace('/', '_')}_{tag}")
    if workspace.exists():
        shutil.rmtree(workspace)
    subprocess.run(["git", "clone", upstream_source, str(workspace)], check=True)
    subprocess.run(["git", "checkout", tag], cwd=workspace, check=True)
    return str(workspace)
```

The LLM figures out the rest:
- `go list -m -versions module` for available versions
- `go mod edit -replace old=new@version` for replace directives
- Whatever else the situation requires

---

## 5. Actions

All actions live in `.theow/actions/`, including migrated heuristics:

### 5.1 Migrated from `patch_failed_tests()`

```python
# .theow/actions/patch_spread_go_version.py
from theow import action

@action("patch_spread_go_version")
def patch_spread_go_version(workspace: str, go_version: str):
    """Patch spread task.yaml to install correct Go version."""
    task_yaml_path = Path(workspace) / "spread/general/test/task.yaml"
    content = yaml.safe_load(task_yaml_path.read_text())
    content["prepare"] = f"snap install go --channel {go_version}/stable --classic"
    task_yaml_path.write_text(yaml.safe_dump(content))
```

```python
# .theow/actions/patch_spread_vet_off.py
from theow import action

@action("patch_spread_vet_off")
def patch_spread_vet_off(workspace: str):
    """Add -vet=off flag to spread test execution."""
    task_yaml_path = Path(workspace) / "spread/general/test/task.yaml"
    content = yaml.safe_load(task_yaml_path.read_text())
    content["execute"] = "cd ../../.. && go test ./... -vet=off"
    task_yaml_path.write_text(yaml.safe_dump(content))
```

### 5.2 Migrated from `_try_resolve_module_renames()`

```python
# .theow/actions/pin_to_compatible_version.py
from theow import action

@action("pin_to_compatible_version")
def pin_to_compatible_version(workspace: str, module: str):
    """Find and pin module to last compatible version before rename."""
    result = subprocess.run(
        f"go list -m -versions {module}",
        shell=True, cwd=workspace, capture_output=True, text=True
    )
    versions = result.stdout.split()[1:]
    if versions:
        compatible = _find_last_compatible_version(module, versions)
        if compatible:
            subprocess.run(
                f"go mod edit -require={module}@{compatible}",
                shell=True, cwd=workspace, check=True
            )

def _find_last_compatible_version(module: str, versions: list[str]) -> str | None:
    """Binary search for last version where module path hasn't changed."""
    # ... existing logic from tree_resolver.py
```

### 5.3 New Actions (Discovered by Explorer)

Explorer will propose new actions like:

```python
# .theow/actions/find_alternative_version.py
@action("find_alternative_version")
def find_alternative_version(workspace: str, module: str, missing_tag: str):
    """Find alternative version when tag is missing from remote."""
    # Query go proxy, find closest semver, update requirement
    ...
```

---

## 6. Rules

Rules live in `.theow/rules/`.

### 6.1 Deterministic Rules (Discovered by Explorer)

These are NOT hardcoded upfront. The Explorer discovers patterns and proposes rules. Examples of rules that would emerge:

```yaml
# .theow/rules/go_version_mismatch.rule.yaml
name: go_version_mismatch
description: >
  Test fails because module requires a different Go version.
  Patch spread task.yaml to install correct version.
tags: [go, test, version]

when:
  - fact: problem_type
    equals: test_failure
  - fact: stderr
    regex: 'go: go\.mod requires go >= (?P<go_version>\d+\.\d+)'
    examples:
      - "go: go.mod requires go >= 1.21 (running go 1.25)"
      - "go: go.mod requires go >= 1.18.0"

then:
  - action: patch_spread_go_version
    params:
      go_version: "{go_version}"
```

```yaml
# .theow/rules/go_vet_failure.rule.yaml
name: go_vet_failure
description: >
  go vet fails, blocking test execution. Add -vet=off flag.
tags: [go, test, vet]

when:
  - fact: problem_type
    equals: test_failure
  - fact: stderr
    contains: "vet:"
    examples:
      - "# pkg\nvet: exiting due to errors"

then:
  - action: patch_spread_vet_off
```

```yaml
# .theow/rules/module_path_rename.rule.yaml
name: module_path_rename
description: >
  Module was renamed. go mod tidy fails because declared path
  doesn't match required path.
tags: [go, dep_resolution, rename]

when:
  - fact: problem_type
    equals: dep_resolution
  - fact: stderr
    regex: 'module declares its path as: (?P<new_path>\S+)\s+but was required as: (?P<old_path>\S+)'
    examples:
      - |
        github.com/imdario/mergo@v1.0.2: parsing go.mod:
        module declares its path as: dario.cat/mergo
        but was required as: github.com/imdario/mergo

then:
  - action: pin_to_compatible_version
    params:
      module: "{old_path}"
```

### 6.2 Probabilistic Rule (Test Generation)

Reserved for genuinely creative work — generating tests when coverage is inadequate:

```yaml
# .theow/rules/generate_tests.rule.yaml
name: generate_tests
description: >
  Tests pass but coverage is inadequate.
  Generate meaningful tests for the package.
tags: [go, test, generation]

when:
  - fact: problem_type
    equals: test_coverage_low

llm_config:
  prompt_template: file://prompts/generate_tests.md
  tools: [read_file, write_file, run_command, create_sandbox]
  constraints:
    max_tool_calls: 50
    max_tokens: 32768
  use_secondary: true
```

This is expensive. Regular test failures go through deterministic rules discovered by exploration.

---

## 7. Exploration Mode

### 7.1 Safety

Exploration runs with **no side effects to the store**:
- No commits to remote
- No pushes to Launchpad
- No registration with store
- No publishing

Use existing CLI flags to ensure safety:

```bash
# Exploration run: local only, no store interaction
THEOW_EXPLORE=1 just onboard go noctua-bot \
    --single-pkg="github.com/some/pkg@v1.0.0" \
    --only-step=run_tests \
    --no-register  # or other safety flags as needed
```

Most Theow-marked steps are inherently local (test execution, pack, dep resolution). The dangerous steps (commit, sync-fork, register) are NOT marked with Theow.

### 7.2 Exploration Script

```python
#!/usr/bin/env python3
"""Run exploration on failed packages to discover new rules."""

import os
import subprocess
from automation.orm_sqlite import SQLiteDepTreeORM
from src.base.onboarder import Language

orm = SQLiteDepTreeORM(Language.go)

# Get failed packages, ordered by rank (dependencies first)
failures = list(orm.iter_packages(statuses=["failed"]))

for pkg in failures:
    # Skip structural failures that Theow can't fix
    if pkg.message in ("monorepo packages not supported", "cycle-causing packages not supported"):
        continue

    # Skip external failures
    if pkg.message in ("unregistered", "needs collaboration", "track missing"):
        continue

    # Run the failing step with exploration enabled
    subprocess.run(
        [
            "just", "onboard", "go", "noctua-bot",
            f"--single-pkg={pkg.name}@{pkg.tag}",
            f"--only-step={pkg.message}",  # message contains task name
        ],
        env={**os.environ, "THEOW_EXPLORE": "1"},
    )
```

### 7.3 What Exploration Produces

1. **Rule file**: `.theow/rules/<name>.rule.yaml`
2. **Action file** (if new action needed): `.theow/actions/<name>.py`
3. **Session cache hit** for similar failures in same run

The team reviews proposed rules:
- Good rule? Commit it. Next run, Resolver uses it deterministically.
- Bad rule? Delete it. Explorer tries something different next time.
- Close but not right? Edit it.

---

## 8. Setup

### 8.1 Environment

```bash
# LLM API keys
export GEMINI_API_KEY="..."
export ANTHROPIC_API_KEY="..."  # For secondary model
```

### 8.2 Theow Initialization

```python
# In sd-tools, early in startup
from theow import Theow

theow = Theow(
    theow_dir="./.theow",
    llm="gemini/gemini-2.0-flash",
    llm_secondary="anthropic/claude-sonnet-4-20250514",
    session_limit=50,  # Max LLM explores per run
)
```

Tools and actions are auto-discovered from `.theow/tools.py` and `.theow/actions/*.py`.

### 8.3 Directory Structure

```
sd-tools/
├── .theow/
│   ├── config.yaml              # LLM config, budget settings
│   ├── rules/                   # All rules (deterministic + probabilistic)
│   │   ├── go_version_mismatch.rule.yaml
│   │   ├── go_vet_failure.rule.yaml
│   │   ├── module_path_rename.rule.yaml
│   │   └── generate_tests.rule.yaml
│   ├── actions/                 # All actions (migrated + discovered)
│   │   ├── patch_spread_go_version.py
│   │   ├── patch_spread_vet_off.py
│   │   ├── pin_to_compatible_version.py
│   │   └── ...
│   ├── tools.py                 # Tool registrations
│   ├── prompts/                 # Prompt templates for probabilistic rules
│   │   └── generate_tests.md
│   └── chroma/                  # Vector DB (auto-populated by Theow)
├── src/
│   └── ...
└── automation/
    └── ...
```

---

## 9. What Changes in sd-tools

### 9.1 Changes

1. Add Theow dependency
2. Create `.theow/` directory with migrated actions and initial rules
3. Define custom exceptions (`SourcecraftTestFailed`, `RepoAcquisitionFailed`, etc.)
4. Extract `_impl` methods from integration points
5. Add `@theow.mark()` decorators to `_impl` methods
6. Set LLM API keys in environment
7. Write exploration script

### 9.2 Unchanged

- Task decorator (`@task`)
- Pipeline structure (11 steps)
- Database schema
- CLI interface
- Git workflow
- PR process

---

## 10. Cost Model

| Scenario | LLM Calls | Cost |
|----------|-----------|------|
| Deterministic rule matches | 0 | Free |
| Session cache hit (similar failure) | 0 | Free |
| Exploration (novel failure) | 1 session | Variable |
| Probabilistic rule (test generation) | 1 session | Higher |

**Expected trajectory:**
- Early: Many exploration calls as novel patterns are discovered
- Over time: Exploration calls decrease as deterministic rules accumulate
- Steady state: Most failures resolved deterministically, rare exploration

---

## 11. Open Questions

1. **Sandbox environment**: Does `sourcecraft test` run in a container/LXC? Tool implementations need to match the actual test environment.

2. **Parallel workers**: CI uses multiple workers. With `THEOW_EXPLORE=1`, LLM blocks for seconds. Run exploration single-threaded or implement async?

3. **Coverage detection**: For the probabilistic test generation rule, how do we detect "tests pass but coverage is low"? Need to define trigger criteria.

4. **Prepare phase failures**: Repo acquisition failures happen during `prepare.py` (dependency tree generation), not during `onboard.py`. Need to integrate Theow at that phase too.
