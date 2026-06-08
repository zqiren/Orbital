# CONTEXT.md Cold-Start Workspace Scan — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a user imports an existing folder as a project and opens it for the first time, offer (via a button, no tokens spent until clicked) to scan the workspace, then have the agent read selectively and propose the project's shape (State + draft Goals) for conversational confirmation before any file is written.

**Architecture:** A new **cold-start mode** threaded through `start_agent → PromptContext → prompt_builder` (distinct from the existing reactive onboarding prompt). Stage 1 is a **deterministic, gitignore-respecting, self-bounding skeleton walker** (`workspace_scan.py`, reusing vendored ripgrep via `grep_tool._find_ripgrep`) whose inventory is injected as a system message. Stage 2 is LLM selective reading bounded by the existing coarse ~70% usage nudge (no precise token tool exists). Stage 3 proposes State + draft Goals; on confirmation the agent writes `project_goals.md` via its `write` tool (the onboarding off-switch) and calls the existing `checkpoint_state` tool to persist `PROJECT_STATE.md` + `CONTEXT.md` through `run_session_end_routine` (`_occ_write_metadata` + `_cap_context_tokens`) — **no new write plumbing**. A separate cleanup guards the `instructions → project_goals.md` sync so a later Settings save cannot clobber scan output.

**Tech Stack:** Python 3 / FastAPI / asyncio (backend); React + TypeScript + Vite + Tailwind (frontend); pytest + FastAPI `TestClient` (backend tests); Playwright @ 375×667 + Vitest (frontend tests); vendored ripgrep.

---

## Vocabulary (the codebase names collide with the design — see INVESTIGATION findings)

| Design term | Disk artifact | Owner | Write path at confirm |
|---|---|---|---|
| **State** (descriptive) | `orbital/PROJECT_STATE.md` | agent | `checkpoint_state` → `run_session_end_routine` (`_occ_write_metadata`) |
| **Goals** (suggested) | `orbital/instructions/project_goals.md` | user (agent drafts) | agent `write` tool (= onboarding off-switch) |
| **Instructions** (prescriptive) | `orbital/instructions/user_directives.md` | user only | **agent NEVER writes this** |

Two gates (do not conflate): the **prompt** exits onboarding when `project_goals.md` exists (`prompt_builder.py:436-438`); the **dispatcher gate** `is_onboarding_complete` checks `PROJECT_STATE.md` (`agent_manager.py:2669-2682`). Confirmation writes both, so both flip.

---

## File Structure

**Create:**
- `agent_os/agent/workspace_scan.py` — deterministic skeleton walker (Stage 1). One responsibility: produce a bounded, gitignore-respecting, sized inventory string.
- `tests/regression/test_workspace_scan.py` — walker unit tests.
- `tests/regression/test_cold_start_prompt.py` — prompt-mode selection tests.
- `tests/regression/test_goals_sync_guard.py` — de-entanglement guard test.
- `tests/regression/test_project_workspace_empty_flag.py` — `is_empty_workspace` payload test.
- `tests/integration/test_cold_start_scan.py` — full-journey TestClient test.
- `web/src/components/ColdStartCard.tsx` — consent card.
- `web/src/components/ColdStartCard.test.tsx` — Vitest unit test for the card.
- `web/e2e/cold-start-card.spec.ts` — Playwright @375×667.

**Modify:**
- `agent_os/agent/tools/grep_tool.py` — export `_find_ripgrep` (rename to public `find_ripgrep`, keep alias) so the walker reuses it.
- `agent_os/agent/prompt_builder.py` — add `PromptContext.cold_start: bool`; branch in `_onboarding_or_directive`.
- `agent_os/daemon_v2/agent_manager.py` — `start_agent(cold_start, cold_start_skeleton)`; set `PromptContext.cold_start`; set `Session` origin; append skeleton system message.
- `agent_os/api/routes/agents_v2.py` — new `POST /agents/{project_id}/cold-start-scan`; `is_empty_workspace` in `_redact_project`; guard the `instructions → project_goals.md` sync in `update_project`.
- `web/src/types.ts` — `Project.is_empty_workspace?: boolean`.
- `web/src/hooks/useAgent.ts` — `coldStartScan(projectId)` calling the new endpoint.
- `web/src/components/ChatView.tsx` — mount `<ColdStartCard>` in the empty state.

---

## Task 1: Make ripgrep resolver reusable

**Files:**
- Modify: `agent_os/agent/tools/grep_tool.py:55-94`

- [ ] **Step 1: Add a public alias for the resolver**

In `grep_tool.py`, immediately after the existing `def _find_ripgrep() -> str | None:` definition ends (after line 94), add:

```python
# Public alias so other modules (e.g. workspace_scan) can reuse the vendored
# ripgrep resolution logic without importing a private symbol.
def find_ripgrep() -> str | None:
    """Public wrapper around the vendored-ripgrep resolver."""
    return _find_ripgrep()
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from agent_os.agent.tools.grep_tool import find_ripgrep; print(bool(find_ripgrep()))"`
Expected: prints `True` on this dev machine (vendored rg present).

- [ ] **Step 3: Commit**

```bash
git add agent_os/agent/tools/grep_tool.py
git commit -m "refactor(grep): expose find_ripgrep for reuse by workspace_scan"
```

---

## Task 2: Deterministic skeleton walker (Stage 1)

**Files:**
- Create: `agent_os/agent/workspace_scan.py`
- Test: `tests/regression/test_workspace_scan.py`

**Contract:** `scan_workspace(workspace: str, *, max_files: int = 1000) -> str` returns a human/agent-readable inventory: every gitignore-respected file with its size, **self-bounded** — if the tree exceeds `max_files`, keep the largest files + the shallowest paths and append a truncation notice. Never raises on a large tree; returns a notice instead.

- [ ] **Step 1: Write the failing tests**

```python
# tests/regression/test_workspace_scan.py
import os
from agent_os.agent.workspace_scan import scan_workspace


def _touch(path: str, size: int = 0) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("x" * size)


def test_lists_files_with_sizes(tmp_path):
    _touch(str(tmp_path / "README.md"), 100)
    _touch(str(tmp_path / "src" / "main.py"), 250)
    out = scan_workspace(str(tmp_path))
    assert "README.md" in out
    assert "src/main.py" in out or os.path.join("src", "main.py") in out
    assert "100" in out and "250" in out


def test_respects_gitignore(tmp_path):
    _touch(str(tmp_path / ".gitignore"), 0)
    with open(tmp_path / ".gitignore", "w") as f:
        f.write("ignored/\n*.log\n")
    _touch(str(tmp_path / "keep.py"), 10)
    _touch(str(tmp_path / "ignored" / "secret.py"), 10)
    _touch(str(tmp_path / "debug.log"), 10)
    out = scan_workspace(str(tmp_path))
    assert "keep.py" in out
    assert "secret.py" not in out
    assert "debug.log" not in out


def test_self_bounds_large_tree(tmp_path):
    for i in range(50):
        _touch(str(tmp_path / f"f{i}.txt"), i)
    out = scan_workspace(str(tmp_path), max_files=10)
    # Truncation notice present; output does not list all 50 files.
    assert "more" in out.lower() or "truncat" in out.lower()
    assert out.count("\n") < 50


def test_empty_workspace_returns_notice(tmp_path):
    out = scan_workspace(str(tmp_path))
    assert out.strip() != ""
    assert "no files" in out.lower() or "empty" in out.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/regression/test_workspace_scan.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_os.agent.workspace_scan'`.

- [ ] **Step 3: Implement the walker**

```python
# agent_os/agent/workspace_scan.py
# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic, gitignore-respecting, self-bounding workspace skeleton.

Stage 1 of the cold-start scan. No LLM. Produces the one fact the agent
cannot reliably guess — how much is here and where — as a bounded inventory
string suitable for injection as a system message.

Reuses the vendored ripgrep (`rg --files`, which honors .gitignore) for the
file listing, then stats each path for size. Falls back to an os.walk that
skips .git if ripgrep is unavailable.
"""
from __future__ import annotations

import os
import subprocess

from agent_os.agent.tools.grep_tool import find_ripgrep

_DEFAULT_MAX_FILES = 1000
_RG_TIMEOUT_SECONDS = 10


def _list_files(workspace: str) -> list[str]:
    """Return workspace-relative file paths, gitignore-respected when possible."""
    rg = find_ripgrep()
    if rg is not None:
        try:
            proc = subprocess.run(
                [rg, "--files", "--hidden", "--glob", "!.git"],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=_RG_TIMEOUT_SECONDS,
            )
            if proc.returncode in (0, 1):  # 1 == no files matched
                return [ln for ln in proc.stdout.splitlines() if ln.strip()]
        except (OSError, subprocess.SubprocessError):
            pass
    # Fallback: os.walk skipping .git (no gitignore semantics).
    out: list[str] = []
    for root, dirs, files in os.walk(workspace):
        dirs[:] = [d for d in dirs if d != ".git"]
        for name in files:
            rel = os.path.relpath(os.path.join(root, name), workspace)
            out.append(rel)
    return out


def scan_workspace(workspace: str, *, max_files: int = _DEFAULT_MAX_FILES) -> str:
    """Build a bounded inventory string of (path, size) for the workspace."""
    rels = _list_files(workspace)
    sized: list[tuple[str, int]] = []
    for rel in rels:
        try:
            size = os.path.getsize(os.path.join(workspace, rel))
        except OSError:
            size = 0
        sized.append((rel, size))

    total = len(sized)
    if total == 0:
        return "[WORKSPACE SKELETON]\nNo files found (empty workspace)."

    truncated = total > max_files
    if truncated:
        # Keep the largest files (high signal) and the shallowest paths
        # (top-level structure). Union, then cap.
        by_size = sorted(sized, key=lambda t: t[1], reverse=True)[: max_files]
        by_depth = sorted(sized, key=lambda t: t[0].count(os.sep))[: max_files]
        seen: set[str] = set()
        kept: list[tuple[str, int]] = []
        for item in by_depth + by_size:
            if item[0] in seen:
                continue
            seen.add(item[0])
            kept.append(item)
            if len(kept) >= max_files:
                break
        sized = kept

    sized.sort(key=lambda t: t[0])
    total_bytes = sum(s for _, s in sized)
    lines = [
        f"[WORKSPACE SKELETON] {total} files"
        + (f" (showing {len(sized)} — largest + top levels)" if truncated else "")
        + f", ~{total_bytes} bytes shown.",
    ]
    for rel, size in sized:
        lines.append(f"{rel}\t{size}")
    if truncated:
        lines.append(
            f"[... {total - len(sized)} more files not shown; "
            "use list/glob/grep to explore specific paths]"
        )
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/regression/test_workspace_scan.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add agent_os/agent/workspace_scan.py tests/regression/test_workspace_scan.py
git commit -m "feat(scan): deterministic gitignore-respecting self-bounding workspace skeleton"
```

---

## Task 3: Cold-start prompt mode

**Files:**
- Modify: `agent_os/agent/prompt_builder.py:28-50` (add field), `:433-493` (branch)
- Test: `tests/regression/test_cold_start_prompt.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/regression/test_cold_start_prompt.py
from agent_os.agent.prompt_builder import PromptBuilder, PromptContext
from agent_os.agent.autonomy import Autonomy


def _ctx(tmp_path, **kw):
    base = dict(
        workspace=str(tmp_path), model="m", autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[], tool_names=["read", "write"], os_type="macos",
        datetime_now="2026-06-08T00:00:00", project_name="Demo",
        project_instructions="",
    )
    base.update(kw)
    return PromptContext(**base)


def test_cold_start_emits_scan_stages(tmp_path):
    # No project_goals.md present + cold_start=True -> 3-stage scan prompt.
    section = PromptBuilder()._onboarding_or_directive(_ctx(tmp_path, cold_start=True))
    assert "COLD-START" in section.upper()
    assert "skeleton" in section.lower()
    # Stage 3 ownership rules: must NOT instruct writing user_directives.md.
    assert "user_directives.md" not in section


def test_non_cold_start_keeps_reactive_onboarding(tmp_path):
    section = PromptBuilder()._onboarding_or_directive(_ctx(tmp_path, cold_start=False))
    assert "ONBOARDING MODE" in section
    assert "COLD-START" not in section.upper()


def test_existing_goals_still_directive(tmp_path):
    import os
    gp = tmp_path / "orbital" / "instructions"
    os.makedirs(gp, exist_ok=True)
    (gp / "project_goals.md").write_text("Mission: do X")
    section = PromptBuilder()._onboarding_or_directive(_ctx(tmp_path, cold_start=True))
    # Goals exist -> directive wins even under cold_start (idempotent re-entry).
    assert "PROJECT DIRECTIVE" in section
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/regression/test_cold_start_prompt.py -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'cold_start'`.

- [ ] **Step 3a: Add the field**

In `prompt_builder.py`, in the `PromptContext` dataclass after line 50 (`turns_since_last_update`), before `def __post_init__`, add:

```python
    cold_start: bool = False  # first-session import scan mode (Stage 1-3)
```

- [ ] **Step 3b: Branch in `_onboarding_or_directive`**

In `prompt_builder.py`, replace the `if content is None:` line (currently line 438) opening so the cold-start branch is checked first. Immediately after `content = self._read_truncated(goals_path)` (line 437), insert:

```python
        if content is None and context.cold_start:
            return (
                "## COLD-START WORKSPACE SCAN\n\n"
                "This is an imported project with an existing workspace. The user has\n"
                "consented to a one-time scan. A deterministic [WORKSPACE SKELETON] (every\n"
                "gitignore-respected file + size) has been provided to you as a system\n"
                "message. Work through three stages:\n\n"
                "STAGE 2 — READ (informed by the skeleton sizes):\n"
                "- Use the skeleton's sizes to plan BEFORE opening files. Small total →\n"
                "  read broadly. Large total → read high-signal files (README, config,\n"
                "  entry points, the largest meaningful sources) and sample the rest.\n"
                "- You have NO precise token meter. You will see a coarse 'Context usage:'\n"
                "  line each turn. When it crosses ~70%, STOP reading and move to Stage 3\n"
                "  with what you have. State what you skipped.\n\n"
                "STAGE 3 — PROPOSE → CONFIRM → WRITE:\n"
                f"- Propose, in chat, your read of the project (descriptive State) and a\n"
                "  DRAFT of suggested Goals. Report which files you read and which you skipped.\n"
                "- State is yours to assert. Goals are a SUGGESTION the user owns — invite edits.\n"
                "- Do NOT propose or write prescriptive Instructions; a scan cannot infer intent.\n"
                "- Write NOTHING until the user confirms (ok / yes / looks good / any affirmative).\n"
                "- On confirmation: (1) write the agreed Goals to\n"
                f"  {context.workspace}/orbital/instructions/project_goals.md using the `write`\n"
                "  tool (Mission, Triggers, Scope, Rules, Preferences; under 1500 words), then\n"
                "  (2) call the `checkpoint_state` tool to persist PROJECT_STATE.md and CONTEXT.md.\n"
                "- Do NOT write user_directives.md. After writing, announce readiness and begin."
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/regression/test_cold_start_prompt.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add agent_os/agent/prompt_builder.py tests/regression/test_cold_start_prompt.py
git commit -m "feat(prompt): cold-start scan mode branch gated on PromptContext.cold_start"
```

---

## Task 4: Thread cold-start through start_agent (mode + origin + skeleton injection)

**Files:**
- Modify: `agent_os/daemon_v2/agent_manager.py:351-358` (signature), `:477-494` (PromptContext), `:519-534` (Session.new origin), and the pre-loop region (~before line 722) for the skeleton append.

- [ ] **Step 1: Extend the signature**

In `agent_manager.py`, change the `start_agent` signature (lines 351-358) to add two params after `session`:

```python
    async def start_agent(self, project_id: str, config: AgentConfig,
                          initial_message: str | None = None,
                          trigger_source: str | None = None,
                          trigger_name: str | None = None,
                          initial_nonce: str | None = None,
                          session_id: str | None = None,
                          queue_state: str = "chat",
                          session: "Session | None" = None,
                          cold_start: bool = False,
                          cold_start_skeleton: str | None = None) -> None:
```

- [ ] **Step 2: Set `cold_start` on PromptContext**

In the `PromptContext(...)` construction (~lines 477-494), add the field (after `project_id=project_id,`):

```python
        cold_start=cold_start,
```

- [ ] **Step 3: Tag the session origin**

In the `Session.new(...)` call (~line 526-534), change the `origin=` argument to honor cold-start:

```python
        origin=("cold_start" if cold_start else ("queue" if queue_state == "running" else "chat")),
```

- [ ] **Step 4: Inject the skeleton as a system message before the loop runs**

Locate where the loop task is created (`task = asyncio.create_task(loop.run(...))`, ~line 722). Immediately BEFORE that line, insert:

```python
        if cold_start and cold_start_skeleton:
            # Stage-1 inventory enters the conversation as system context so the
            # agent plans Stage-2 reads against measured sizes. Persisted in JSONL.
            session.append_system(cold_start_skeleton)
```

- [ ] **Step 5: Smoke-check it imports/threads**

Run: `python -c "import agent_os.daemon_v2.agent_manager as m; import inspect; print('cold_start' in inspect.signature(m.AgentManager.start_agent).parameters)"`
Expected: prints `True`.

- [ ] **Step 6: Commit**

```bash
git add agent_os/daemon_v2/agent_manager.py
git commit -m "feat(manager): thread cold_start mode, origin, and skeleton injection through start_agent"
```

---

## Task 5: Cold-start scan endpoint

**Files:**
- Modify: `agent_os/api/routes/agents_v2.py` (new route near the other `/agents/{project_id}/...` routes, after `:732`)
- Test: covered by integration Task 9.

- [ ] **Step 1: Add the endpoint**

In `agents_v2.py`, after the `inject` route block, add:

```python
@router.post("/agents/{project_id}/cold-start-scan", status_code=201)
async def cold_start_scan(project_id: str):
    """Mint the project's first session and start the agent in cold-start mode.

    Runs the deterministic skeleton walk synchronously (so a walker failure
    surfaces as an HTTP error), then starts a content-less loop whose prompt
    drives the 3-stage scan. No user message is fabricated.
    """
    from agent_os.agent.workspace_scan import scan_workspace
    project = _project_store.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="project not found")
    workspace = project.get("workspace", "")
    if not workspace or not os.path.isdir(workspace):
        raise HTTPException(status_code=400, detail="workspace missing")

    skeleton = scan_workspace(workspace)
    config = _agent_manager._build_agent_config_from_project(project_id)
    minted = await _agent_manager.new_session(project_id)
    session_id = minted["session_id"]
    await _agent_manager.start_agent(
        project_id, config,
        initial_message=None,
        session_id=session_id,
        cold_start=True,
        cold_start_skeleton=skeleton,
    )
    return {"status": "started", "session_id": session_id}
```

- [ ] **Step 2: Commit**

```bash
git add agent_os/api/routes/agents_v2.py
git commit -m "feat(api): POST /agents/{id}/cold-start-scan starts a cold-start session"
```

---

## Task 6: `is_empty_workspace` on the Project payload

**Files:**
- Modify: `agent_os/api/routes/agents_v2.py:337-348` (`_redact_project`)
- Test: `tests/regression/test_project_workspace_empty_flag.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/regression/test_project_workspace_empty_flag.py
from agent_os.api.routes.agents_v2 import _redact_project


def test_empty_workspace_flagged_true(tmp_path):
    p = {"project_id": "p1", "workspace": str(tmp_path), "api_key": ""}
    assert _redact_project(p)["is_empty_workspace"] is True


def test_nonempty_workspace_flagged_false(tmp_path):
    (tmp_path / "README.md").write_text("hi")
    p = {"project_id": "p1", "workspace": str(tmp_path), "api_key": ""}
    assert _redact_project(p)["is_empty_workspace"] is False


def test_orbital_only_workspace_still_empty(tmp_path):
    # An orbital/ scaffold dir alone does not count as "imported content".
    (tmp_path / "orbital").mkdir()
    p = {"project_id": "p1", "workspace": str(tmp_path), "api_key": ""}
    assert _redact_project(p)["is_empty_workspace"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/regression/test_project_workspace_empty_flag.py -q`
Expected: FAIL — `KeyError: 'is_empty_workspace'`.

- [ ] **Step 3: Compute the flag in `_redact_project`**

In `agents_v2.py`, inside `_redact_project`, before `return result` (line 348), add:

```python
    workspace = result.get("workspace", "")
    result["is_empty_workspace"] = _workspace_is_empty(workspace)
```

And add this helper just above `_redact_project` (before line 337):

```python
def _workspace_is_empty(workspace: str) -> bool:
    """True if the workspace has no user content (ignoring the orbital/ scaffold
    and dotfiles). Used by the frontend to decide whether to offer a cold-start scan.
    """
    if not workspace or not os.path.isdir(workspace):
        return True
    try:
        for name in os.listdir(workspace):
            if name == "orbital" or name.startswith("."):
                continue
            return False
    except OSError:
        return True
    return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/regression/test_project_workspace_empty_flag.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add agent_os/api/routes/agents_v2.py tests/regression/test_project_workspace_empty_flag.py
git commit -m "feat(api): expose is_empty_workspace on project payload for cold-start gating"
```

---

## Task 7: Guard the `instructions → project_goals.md` sync

**Files:**
- Modify: `agent_os/api/routes/agents_v2.py:498-511`
- Test: `tests/regression/test_goals_sync_guard.py`

**Decision (approved):** keep the legacy fallback for never-scanned projects, but make it **non-clobbering** — only sync `instructions` → `project_goals.md` when the file does **not** already exist. Once the scan (or onboarding) authors it, disk is canonical.

- [ ] **Step 1: Write the failing test**

```python
# tests/regression/test_goals_sync_guard.py
import os
from agent_os.agent.project_paths import ProjectPaths
from agent_os.api.routes.agents_v2 import _maybe_sync_instructions_to_goals


def test_sync_writes_goals_when_absent(tmp_path):
    ws = str(tmp_path)
    _maybe_sync_instructions_to_goals(ws, goals_content=None, instructions="do X")
    assert "do X" in open(ProjectPaths(ws).project_goals).read()


def test_sync_does_not_clobber_existing_goals(tmp_path):
    ws = str(tmp_path)
    pp = ProjectPaths(ws)
    os.makedirs(pp.instructions_dir, exist_ok=True)
    open(pp.project_goals, "w").write("SCAN-AUTHORED GOALS")
    _maybe_sync_instructions_to_goals(ws, goals_content=None, instructions="stale field")
    assert open(pp.project_goals).read() == "SCAN-AUTHORED GOALS"


def test_explicit_goals_content_always_wins(tmp_path):
    ws = str(tmp_path)
    pp = ProjectPaths(ws)
    os.makedirs(pp.instructions_dir, exist_ok=True)
    open(pp.project_goals, "w").write("old")
    _maybe_sync_instructions_to_goals(ws, goals_content="explicit new", instructions="ignored")
    assert open(pp.project_goals).read() == "explicit new"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/regression/test_goals_sync_guard.py -q`
Expected: FAIL — `ImportError: cannot import name '_maybe_sync_instructions_to_goals'`.

- [ ] **Step 3: Extract + guard the sync**

In `agents_v2.py`, add this helper above `update_project` (near `_write_workspace_file`):

```python
def _maybe_sync_instructions_to_goals(workspace: str, *, goals_content: str | None,
                                      instructions: str | None) -> None:
    """Sync the legacy `instructions` field to project_goals.md, guarded.

    Explicit goals_content always wins. Otherwise the legacy field only seeds
    project_goals.md when it does NOT already exist — so a scan- or onboarding-
    authored file is never clobbered by a later unrelated Settings save.
    """
    from agent_os.agent.project_paths import ProjectPaths
    effective = goals_content
    if effective is None and instructions is not None:
        if os.path.exists(ProjectPaths(workspace).project_goals):
            return  # disk is canonical; do not clobber
        effective = instructions
    if effective is not None:
        _write_workspace_file(workspace, "project_goals.md", effective)
```

Then replace the inline sync in `update_project` (lines 506-509) with:

```python
    _maybe_sync_instructions_to_goals(
        workspace, goals_content=goals_content, instructions=updates.get("instructions"),
    )
```

(Keep the `rules_content` / `user_directives.md` write on line 510-511 unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/regression/test_goals_sync_guard.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add agent_os/api/routes/agents_v2.py tests/regression/test_goals_sync_guard.py
git commit -m "fix(api): guard instructions->project_goals.md sync against clobbering scan output"
```

---

## Task 8: Frontend — types + hook + consent card

**Files:**
- Modify: `web/src/types.ts` (Project), `web/src/hooks/useAgent.ts`
- Create: `web/src/components/ColdStartCard.tsx`, `web/src/components/ColdStartCard.test.tsx`

- [ ] **Step 1: Add the type field**

In `web/src/types.ts`, in the `Project` interface (lines 26-46), add:

```typescript
  is_empty_workspace?: boolean;
```

- [ ] **Step 2: Add the hook call**

In `web/src/hooks/useAgent.ts`, mirroring `newSession` (lines 68-78), add:

```typescript
  const coldStartScan = useCallback(async (projectId: string) => {
    return api<{ status: string; session_id?: string }>(
      `/api/v2/agents/${projectId}/cold-start-scan`,
      { method: 'POST' },
    );
  }, []);
```

Add `coldStartScan` to the hook's returned object.

- [ ] **Step 3: Write the failing Vitest unit test**

```tsx
// web/src/components/ColdStartCard.test.tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { ColdStartCard } from './ColdStartCard';

describe('ColdStartCard', () => {
  it('renders folder name and Scan/Skip', () => {
    render(<ColdStartCard folderName="my-repo" onScan={vi.fn()} onSkip={vi.fn()} />);
    expect(screen.getByText(/my-repo/)).toBeTruthy();
    expect(screen.getByRole('button', { name: /scan/i })).toBeTruthy();
    expect(screen.getByRole('button', { name: /skip/i })).toBeTruthy();
  });

  it('calls onScan when Scan clicked', async () => {
    const onScan = vi.fn();
    render(<ColdStartCard folderName="r" onScan={onScan} onSkip={vi.fn()} />);
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: /scan/i })); });
    expect(onScan).toHaveBeenCalledOnce();
  });
});
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd web && npx vitest run src/components/ColdStartCard.test.tsx`
Expected: FAIL — cannot resolve `./ColdStartCard`.

- [ ] **Step 5: Implement the card** (Tailwind, matching `ApprovalCard` conventions)

```tsx
// web/src/components/ColdStartCard.tsx
interface ColdStartCardProps {
  folderName: string;
  onScan: () => void;
  onSkip: () => void;
  busy?: boolean;
}

export function ColdStartCard({ folderName, onScan, onSkip, busy }: ColdStartCardProps) {
  return (
    <div className="border border-accent/30 border-l-[3px] border-l-accent rounded-lg bg-background">
      <div className="px-4 py-1.5 border-b border-accent/20 text-accent/80 text-xs uppercase tracking-[0.6px] font-semibold">
        Scan this workspace?
      </div>
      <div className="px-4 py-3 space-y-3 text-sm">
        <p className="text-secondary">
          <span className="font-medium text-primary">{folderName}</span> has existing files.
          I can scan it to understand the project and propose a summary for you to confirm —
          nothing is written until you approve.
        </p>
        <div className="flex gap-2">
          <button
            onClick={onScan}
            disabled={busy}
            className="bg-primary text-white px-4 py-1.5 rounded max-md:min-h-[44px] disabled:opacity-50"
          >
            {busy ? 'Scanning…' : 'Scan'}
          </button>
          <button
            onClick={onSkip}
            disabled={busy}
            className="bg-background text-secondary border border-border px-4 py-1.5 rounded max-md:min-h-[44px]"
          >
            Skip
          </button>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd web && npx vitest run src/components/ColdStartCard.test.tsx`
Expected: PASS.

- [ ] **Step 7: TypeScript check**

Run: `cd web && npx tsc --noEmit`
Expected: zero errors.

- [ ] **Step 8: Commit**

```bash
git add web/src/types.ts web/src/hooks/useAgent.ts web/src/components/ColdStartCard.tsx web/src/components/ColdStartCard.test.tsx
git commit -m "feat(web): cold-start consent card + coldStartScan hook + is_empty_workspace type"
```

---

## Task 9: Mount the card in the empty chat state

**Files:**
- Modify: `web/src/components/ChatView.tsx` (empty-state region ~`:1998-2002`; also needs `project` + `sessions.length` + `coldStartScan` in scope — confirm exact props at execution).

**Note for executor:** Read `ChatView.tsx:1740-2010` before editing to confirm the empty-state JSX and what props/hooks are in scope (`project`, `sessionId`, `items`, `useAgent`). The card mounts only when `project?.is_empty_workspace === false && items.length === 0 && sessionId === undefined`.

- [ ] **Step 1: Import the card**

Add near the other component imports at the top of `ChatView.tsx`:

```tsx
import { ColdStartCard } from './ColdStartCard';
```

- [ ] **Step 2: Wire the handlers + render**

Replace the empty-state block (currently lines 1998-2002, the "No messages yet" `<div>`) with:

```tsx
{!loading && items.length === 0 && !stream && (
  project?.is_empty_workspace === false && sessionId === undefined ? (
    <ColdStartCard
      folderName={(project.workspace || '').split('/').filter(Boolean).pop() || 'this folder'}
      busy={coldStartBusy}
      onScan={async () => {
        setColdStartBusy(true);
        try {
          const r = await coldStartScan(project.project_id);
          if (r.session_id) {
            setActiveSessionId(r.session_id);
            setRoute((prev) => prev.name === 'project'
              ? { ...prev, sessionId: r.session_id } : prev);
          }
        } finally { setColdStartBusy(false); }
      }}
      onSkip={() => setColdStartDismissed(true)}
    />
  ) : (
    <div className="text-secondary text-sm text-center mt-12">
      No messages yet. Send a message to get started.
    </div>
  )
)}
```

Add the local state near the other `useState` hooks in `ChatView`:

```tsx
const [coldStartBusy, setColdStartBusy] = useState(false);
const [coldStartDismissed, setColdStartDismissed] = useState(false);
```

And include `coldStartDismissed` in the card-visibility condition (`&& !coldStartDismissed`) so Skip hides it for the session. Pull `coldStartScan` from `useAgent()` where the other agent actions are destructured, and confirm `project`, `setActiveSessionId`, `setRoute` are already in scope (they are used elsewhere in this component).

- [ ] **Step 3: TypeScript check**

Run: `cd web && npx tsc --noEmit`
Expected: zero errors.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/ChatView.tsx
git commit -m "feat(web): mount cold-start consent card in empty imported-project view"
```

---

## Task 10: Integration test — full journey (backend)

**Files:**
- Create: `tests/integration/test_cold_start_scan.py`

**Note for executor:** Read `tests/integration/conftest.py` and one existing integration test (e.g. `test_chat_reasoning_seam.py`) first to match the fixture style (TestClient construction, credential/keyring stubs per the headless-keychain memory: set `PYTHON_KEYRING_BACKEND=in-memory` + `AGENT_OS_API_KEY`). The LLM turn is stubbed — assert wiring/file effects, not real model output.

- [ ] **Step 1: Write the test (failing until all prior tasks land)**

```python
# tests/integration/test_cold_start_scan.py
import os


def test_empty_workspace_has_no_scan_flag(client, tmp_workspace):
    # tmp_workspace fixture: empty dir. Project payload flags it empty.
    pid = _create_project(client, tmp_workspace)
    proj = client.get(f"/api/v2/projects/{pid}").json()
    assert proj["is_empty_workspace"] is True


def test_imported_workspace_flag_and_scan_starts_session(client, tmp_workspace):
    open(os.path.join(tmp_workspace, "README.md"), "w").write("# Real project")
    pid = _create_project(client, tmp_workspace)

    proj = client.get(f"/api/v2/projects/{pid}").json()
    assert proj["is_empty_workspace"] is False

    # No sessions yet.
    assert client.get(f"/api/v2/projects/{pid}/sessions").json()["sessions"] == []

    r = client.post(f"/api/v2/agents/{pid}/cold-start-scan")
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "started" and body["session_id"]

    # A session now exists for the project.
    sessions = client.get(f"/api/v2/projects/{pid}/sessions").json()["sessions"]
    assert any(s.get("session_id") == body["session_id"] for s in sessions)


def test_confirmation_writes_flip_onboarding(client, tmp_workspace):
    # With the LLM stub configured to (a) write project_goals.md and
    # (b) call checkpoint_state, after the scan turn completes both gates flip.
    open(os.path.join(tmp_workspace, "README.md"), "w").write("# Real project")
    pid = _create_project(client, tmp_workspace)
    client.post(f"/api/v2/agents/{pid}/cold-start-scan")
    _drain_agent(client, pid)  # helper: await loop completion in the harness

    pp_state = os.path.join(tmp_workspace, "orbital", "PROJECT_STATE.md")
    pp_goals = os.path.join(tmp_workspace, "orbital", "instructions", "project_goals.md")
    assert os.path.exists(pp_goals)   # prompt off-switch
    assert os.path.exists(pp_state)   # is_onboarding_complete gate
```

- [ ] **Step 2: Implement the helpers + fixtures**

Add module-level `_create_project`, `_drain_agent`, and any `client` / `tmp_workspace` fixtures by copying the conventions from `tests/integration/conftest.py`. Configure the stubbed provider so the cold-start turn writes `project_goals.md` and calls `checkpoint_state` (consult `test_checkpoint_state_tool_engages_cooldown.py` for how the stub triggers a refresh).

- [ ] **Step 3: Run the integration test**

Run: `PYTHON_KEYRING_BACKEND=in-memory AGENT_OS_API_KEY=test python -m pytest tests/integration/test_cold_start_scan.py -q`
Expected: PASS (3 passed).

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_cold_start_scan.py
git commit -m "test(integration): full cold-start scan journey — flag, start, confirm-flip"
```

---

## Task 11: Playwright — consent card @ 375×667

**Files:**
- Create: `web/e2e/cold-start-card.spec.ts`

**Note for executor:** Read `web/e2e/chat-reasoning-capsule.spec.ts` and `web/playwright.config.ts` first to reuse the isolated-daemon + Vite boot + API-seed harness. Seed two projects: one empty workspace, one with a file.

- [ ] **Step 1: Write the test**

```typescript
// web/e2e/cold-start-card.spec.ts
import { test, expect } from '@playwright/test';
// Reuse the boot/seed helpers from chat-reasoning-capsule.spec.ts.

test('shows consent card on imported (non-empty) workspace, not on empty', async ({ page }) => {
  // Seed: projImported (workspace has README.md), projEmpty (empty workspace), 0 sessions each.
  // ... boot daemon + vite + seed via API (copy harness) ...

  await page.goto(`${APP_URL}/#/project/${projImported}/chat`);
  await expect(page.getByText('Scan this workspace?')).toBeVisible();
  await expect(page.getByRole('button', { name: /scan/i })).toBeVisible();

  await page.goto(`${APP_URL}/#/project/${projEmpty}/chat`);
  await expect(page.getByText('Scan this workspace?')).toHaveCount(0);
  await expect(page.getByText('No messages yet')).toBeVisible();
});

test('Scan spawns a session; Skip leaves the composer', async ({ page }) => {
  await page.goto(`${APP_URL}/#/project/${projImported}/chat`);
  await page.getByRole('button', { name: /skip/i }).click();
  await expect(page.getByText('Scan this workspace?')).toHaveCount(0);
  // Re-seed / second project for Scan to avoid cross-test state.
  await page.goto(`${APP_URL}/#/project/${projImported2}/chat`);
  await page.getByRole('button', { name: /scan/i }).click();
  await expect(page.getByText('Scan this workspace?')).toHaveCount(0); // card gone, session active
});
```

- [ ] **Step 2: Run it**

Run: `cd web && npx playwright test e2e/cold-start-card.spec.ts`
Expected: PASS (2 passed) at viewport 375×667.

- [ ] **Step 3: Commit**

```bash
git add web/e2e/cold-start-card.spec.ts
git commit -m "test(e2e): cold-start consent card visibility + Scan/Skip @375x667"
```

---

## Task 12: Full suite + daemon smoke (per CLAUDE.md)

- [ ] **Step 1: Backend regressions + the new tests**

Run: `PYTHON_KEYRING_BACKEND=in-memory AGENT_OS_API_KEY=test python -m pytest tests/unit/ tests/platform/ tests/regression/test_workspace_scan.py tests/regression/test_cold_start_prompt.py tests/regression/test_goals_sync_guard.py tests/regression/test_project_workspace_empty_flag.py tests/integration/test_cold_start_scan.py -q`
Expected: green except the 3 documented pre-existing env-fails (see CLAUDE.md).

- [ ] **Step 2: Frontend typecheck + unit**

Run: `cd web && npx tsc --noEmit && npx vitest run src/components/ColdStartCard.test.tsx`
Expected: zero TS errors; Vitest passes.

- [ ] **Step 3: Live daemon smoke (MANDATORY per CLAUDE.md §3)**

Restart the daemon (`bash scripts/restart-daemon.sh`), create a project pointed at a real non-empty folder, GET the project and confirm `is_empty_workspace: false`, POST `/cold-start-scan`, and confirm via `/api/v2/projects/{pid}/sessions` that a session appears and the agent greets + proposes (no 4xx/5xx in daemon logs). Provide the QR code per CLAUDE.md for mobile verification of the consent card.

- [ ] **Step 4: Final commit (if any smoke fixes)**

```bash
git add -A && git commit -m "chore: cold-start scan — suite green + daemon smoke verified"
```

---

## Self-Review (against TASK-context-md-cold-start)

- **Consent card, no tokens until click** → Tasks 8-9 (card), Task 6 (`is_empty_workspace` gate), Task 11 (visibility e2e). ✅
- **Skip spawns no session; Scan spawns the working session** → Task 9 (`onSkip` dismiss vs `onScan` → endpoint), Task 5 (endpoint mints + starts), Task 11. ✅
- **Empty workspace → normal reactive onboarding** → Task 6 flag false-gates the card; Task 3 keeps reactive onboarding when `cold_start` not set. ✅
- **Stage 1 deterministic, gitignore, sized, self-bounding** → Task 2 (+reuse Task 1). ✅
- **Stage 2 bounded by coarse ~70% signal, no token tool** → Task 3 prompt text encodes the bound; no `context_remaining` tool built (out of scope). ✅
- **Stage 3 ownership split (State=agent, Goals=user draft, Instructions abstain)** → Task 3 prompt; Task 3 test asserts `user_directives.md` not instructed. ✅
- **Confirm-time write via existing plumbing, not duplicated** → Task 3 routes State/Context through `checkpoint_state`→`run_session_end_routine` (`_occ_write_metadata`+`_cap_context_tokens`); Goals via `write` tool (off-switch). ✅
- **Nothing written before confirmation** → Task 3 prompt: "Write NOTHING until the user confirms." ✅
- **Goals de-entanglement (guard, approved)** → Task 7. ✅
- **Tests: regression / integration / Playwright @375×667** → Tasks 2,3,6,7 (regression), 10 (integration), 11 (e2e). ✅
- **Out of scope honored** (no maintenance cadence, no scan-after-skip, no `context_remaining` tool, no dedicated artifact editor, no parallel scan). ✅

**Highest-latitude area (flagged):** the Stage-3 finalize mechanism (agent `write` for Goals + `checkpoint_state` for State/Context) is the part most likely to need tuning during execution — verify in Task 10/12 that the stubbed/live agent actually produces a non-empty `PROJECT_STATE.md` (the `is_onboarding_complete` gate) when it calls `checkpoint_state`. If the session-end extraction yields empty state on a thin cold-start transcript, fall back to having the agent write `PROJECT_STATE.md` directly via the `write` tool in the Stage-3 prompt.
```
