# INVESTIGATION: INDEX.md Feature — Current State Audit

*Created: 2026-06-02*
*Purpose: Gather implementation facts needed to write TASK-index-md-feature.md*
*Status: COMPLETE*

---

## Background

We are introducing INDEX.md as a first-class Layer 1 workspace file — the agent's living map of its project/workspace. It replaces the existing CONTEXT.md (which exists but is not in Layer 1 and not actively maintained by the agent). Additionally, we need the upload endpoint to notify the agent when a user uploads a file.

This investigation answers all questions below before we write the implementation spec.

---

## Questions to Answer

### Q1: Session-End Routine — CONTEXT.md Handling — **ANSWERED**

**File:** `agent_os/agent/workspace_files.py` — `run_session_end_routine()`

**Answer:** CONTEXT.md **IS fully implemented** in the session-end routine. Contrary to the framing in the Background ("exists but is not in Layer 1 and not actively maintained"), CONTEXT.md is requested in the LLM prompt, OCC-baselined, and written in its own dedicated code section, exactly like DECISIONS.md and LESSONS.md.

**Evidence:**

- **File mapping** — `workspace_files.py:146`: includes `"context": "CONTEXT.md"`.
- **Read-in** — `workspace_files.py:467`: existing CONTEXT.md is read and passed into the LLM prompt.
- **OCC baseline** — `workspace_files.py:666`: CONTEXT.md mtime baseline captured pre-LLM alongside the other files.
- **Write section** — `workspace_files.py:797–812`: a **separate, independent** conditional write block (`if result.get("context", "").strip():`), using the same `_apply_sanity_checks()` dedup/cap pattern as DECISIONS and LESSONS.
  - Entry pattern `_CONTEXT_ENTRY_PATTERN = r"^-\s+"` at `workspace_files.py:181`; cap 25 entries; keep strategy `"last"` (`:804`).

**Exact prompt template** (`build_session_end_prompt()`, the CONTEXT field, `workspace_files.py:534–550`):

```
5. "context" (string, empty to preserve existing): The COMPLETE updated
   CONTEXT.md file. This REPLACES the existing file entirely.
   Scope: external entities relevant to this project — people, services,
   platforms, third-party APIs, persistent environmental constraints.
   Exclusions (do NOT include):
     - Workspace files or internal project artifacts
     - One-shot session errors or transient tool failures
     - In-progress work or current task state (belongs in PROJECT_STATE)
     - Decisions or rationale (belongs in DECISIONS)
     - Patterns or advice (belongs in LESSONS)
   - Carry forward every still-relevant entry
   - Add genuinely new external entities discovered THIS SESSION
   - Drop entries not referenced in the last 10 sessions
   - Merge duplicates
   - Cap: 25 entries
   - Return empty string "" ONLY to indicate "no updates needed, preserve
     existing file."
```

- System message (`workspace_files.py:678`): `"You maintain workspace memory files for an AI agent. Respond with ONLY valid JSON."`
- The full user prompt (`workspace_files.py:492–578`) requests 5 JSON fields: `project_state`, `decisions`, `session_log_entry`, `lessons`, `context`.

**Files produced by the routine (5 total):**

1. PROJECT_STATE.md — full overwrite, always if non-empty (`workspace_files.py:711–715`)
2. DECISIONS.md — append if LLM returns non-empty (`:717–733`)
3. SESSION_LOG.md — append + truncate at cap, under `asyncio.Lock` (`:735–778`)
4. LESSONS.md — append if non-empty (`:780–795`)
5. CONTEXT.md — replace if non-empty (`:797–812`)

**Separate section or lumped?** CONTEXT.md has its **own** separate conditional code block (`:797–812`), not lumped with the others. Each of DECISIONS / LESSONS / CONTEXT uses identical `if result.get("<key>","").strip():` structure but in distinct blocks.

---

### Q2: Upload Endpoint — Post-Upload Notification — **ANSWERED**

**File:** `agent_os/api/routes/files_v2.py` — the upload handler (`upload_file`, `files_v2.py:143–170`)

**Answer:** **NOT IMPLEMENTED.** The upload handler has zero integration with the session, agent manager, task queue, or WebSocket. After writing the file to disk it returns immediately.

**Evidence — full flow trace:**

1. Route — `files_v2.py:143`: `@router.post("/projects/{project_id}/files/upload")`; signature `:144` `async def upload_file(project_id, file: UploadFile, path="/uploads/")`.
2. Path resolve/validate — `:145` `_resolve_path(...)` (helper `:37–53`); read with size cap `:148`; `HTTPException(413)` if > 10 MB `:149–150`.
3. Dir + filename sanitize — `os.makedirs(..., exist_ok=True)` `:153`; `os.path.basename()` sanitize `:156–158`.
4. Path-safety recheck — `:160–164`, `HTTPException(400)` if escapes workspace.
5. **Write to disk** — `:166–167` `with open(dest,"wb") as f: f.write(data)` — the **only** side effect.
6. **Return** — `:170` `return {"path": rel_path, "size": len(data)}`.

**No session JSONL write, no task-queue injection, no WebSocket broadcast.** The handler's only injected dependency is `project_store` (configured `files_v2.py:25–28`, wired from `app.py:333`). No references to `ws_manager`, `agent_manager`, `trigger_manager`, or any `Session` object anywhere in the file.

**Related (not connected):** an independent `file_watch` trigger system exists at `agent_os/daemon_v2/trigger_manager.py` (watchdog observers on directories), but it is **not** wired to this HTTP upload endpoint.

---

### Q3: Compaction/Checkpoint — Pre-Compaction Writes — **ANSWERED**

**Files:** `agent_os/agent/context.py`, `agent_os/agent/loop.py`, `agent_os/agent/compaction.py`

**Answer:** Compaction **DOES** trigger Layer 1 workspace writes first. It is not a bare truncate. A "checkpoint" concept also exists with three triggers.

**Threshold check** — `context.py:79–80`:
```python
def should_compact(self) -> bool:
    return self._last_usage_pct > 0.80
```
Usage computed at `context.py:306–311` as `total_tokens / available_budget` (budget = model limit − response reserve).

**Trigger in main loop** — `loop.py:846–847` (runs at bottom of each iteration, after tool execution):
```python
# Check compaction
if self._context_manager.should_compact():
```

**Ordered compaction flow (`loop.py:847–939`):**

1. **Token-pressure refresh FIRST** (`:850–873`) — fires `_run_refresh("token_pressure", iteration)` *before* compaction, exempt from cooldown; this calls `on_session_end_refresh` → `run_session_end_routine()` (writes all 5 Layer 1 files, `workspace_files.py:710–813`, OCC-gated).
2. **Pre-compaction memory flush** (`:875–925`) — appends `MEMORY_FLUSH_PROMPT` and gives the agent one turn to write/edit state; executes any tool calls the agent emits (including write/edit).
3. **Actual compaction** (`:927`) — `compaction_mod.run(...)` summarizes messages.
4. **Post-compaction reorientation** (`:933–939`) — `compaction_mod.inject_reorientation(workspace, session)` re-injects project_goals.md + PROJECT_STATE.md (`compaction.py:83–120`, each capped 3000 chars).

**MEMORY_FLUSH_PROMPT** (`compaction.py:26–33`):
```python
MEMORY_FLUSH_PROMPT = (
    "Pre-compaction memory flush. "
    "Your context window is nearly full and history will be summarised shortly. "
    "Write any critical working state — current task position, active decisions, "
    "in-progress work, and anything you must not forget — to PROJECT_STATE.md now. "
    "Use the write or edit tool. "
    "If there is nothing important to save, reply with exactly: <silent>"
)
```

**Checkpoint concept — YES, three trigger types** (all call `_run_refresh` → `run_session_end_routine()`):
- **Agent-decided:** tool `checkpoint_state` registered at `agent_manager.py:685–690` → `loop.trigger_checkpoint()` → `_run_refresh("agent_decided", ...)`.
- **Turn-count:** every `COOLDOWN_TURNS` (50) — `loop.py:43`.
- **Token-pressure:** the >80% path above (`loop.py:847–873`).

---

### Q4: System Prompt — Workspace File Maintenance Instructions — **ANSWERED**

**File:** `agent_os/agent/prompt_builder.py`

**Answer:** The system prompt instructs maintenance of **three** files (PROJECT_STATE.md, DECISIONS.md, LESSONS.md) with both during-work and session-end timing. **CONTEXT.md is NOT mentioned in the system prompt.** No mention of "index", "map", or "workspace understanding"; "orientation" appears once, unrelated.

**Exact prompt text:**

PROJECT_STATE.md — `prompt_builder.py:481–482`:
```
- PROJECT_STATE.md: Living summary of project status, pending work, key files.
  Update after completing significant work. Keep under 1K tokens.
```
Also referenced in token-pressure nudges at `:639` and `:644`:
```
You are using significant context. Consider updating PROJECT_STATE.md now.
...
URGENT: Save all important state to PROJECT_STATE.md immediately.
```

DECISIONS.md — `prompt_builder.py:483`:
```
- DECISIONS.md: Key decisions with brief reasoning. Append when you make non-obvious choices.
```

LESSONS.md — `prompt_builder.py:484–486`:
```
- LESSONS.md: Force-injected every turn. Auto-consolidated at session end.
  You may append mid-session when you recover from errors or discover non-obvious
  workarounds. Keep entries under 100 words. Session-end routine handles dedup.
```

CONTEXT.md — **NOT IMPLEMENTED** (no reference in `prompt_builder.py`).

**Timing — BOTH during work and session end.** During-work wording: "Update after completing significant work" (PROJECT_STATE), "Append when you make non-obvious choices" (DECISIONS), "You may append mid-session when you recover from errors" (LESSONS). Session-end wording: "Auto-consolidated at session end" (LESSONS). Closing nudge `:487–488`: "These files are your memory across sessions. If you don't maintain them, you'll lose context when the session restarts. Update them proactively."

**Keywords "index"/"map"/"orientation"/"workspace understanding":** Only "orientation" appears, once, at `prompt_builder.py:348` — and only describing the displayed workspace path ("The workspace path shown above is for orientation only."). No "index", "map", or "workspace understanding" anywhere.

---

### Q5: Mid-Session Message Injection — **ANSWERED**

**Files:** `agent_os/agent/session.py`, `agent_os/agent/loop.py`, `agent_os/api/routes/agents_v2.py` (note: actual filename is `agents_v2.py`, not `agent_v2.py`)

**Answer (system mid-stream):** There is **no** mechanism to inject a SYSTEM message into the LLM context *while a stream is in flight*. The loop owns a single-writer contract on the session during streaming. The closest facility is **deferred messages**, inserted after the current tool batch completes — not mid-stream.

**Deferred-message facility:**
- `Session.defer_message(content, role="system", source="daemon")` — `session.py:544–553`; payload dict `{role, content, source, timestamp}` stored in `self._deferred_messages`.
- `Session.pop_deferred_messages()` — `session.py:555–559`.
- Drained in loop **after tool execution** — `loop.py:803–804`:
  ```python
  for msg in self._session.pop_deferred_messages():
      self._session.append(msg)
  ```

**USER message injection — full path:**
- **Endpoint:** `POST /api/v2/agents/{project_id}/inject` — handler `agents_v2.py:717–838`.
- **Request model:** `InjectRequest` — `agents_v2.py:196–232` (fields `content`, `target`, `nonce`, `attachments`, `session_id`).
- **Handler → manager:** calls `_agent_manager.inject_message(project_id, content, nonce, session_id, queue_state)` — `agent_manager.py:1069–1320`. Session resolved via `_sid_inject()` (`:1095`).
  - **Loop running:** `handle.session.queue_message(content, nonce=nonce)` — `agent_manager.py:1149`; appends to `self._queue: list[tuple[str, str|None]]` (`session.py:114`, method `:530–532`).
  - **Idle/paused for approval:** auto-deny pending approval, append directly, resume — `:1152–1247`.
  - **No session:** auto-start agent with the message — `:1130–1135`.
  - **Slot conflict:** `ValueError` → route returns HTTP 202 with `slot_held` payload (`agents_v2.py:815–831`).
- **Loop drains queue:** `queued = self._session.pop_queued_messages()` — `loop.py:292` (`pop_queued_messages` at `session.py:534–538`); each tuple becomes a `role="user"`, `source="user"` message appended to session (`loop.py:298–310`). Simple in-memory list, drained at top of the while loop.

**Mechanism shape (for reuse — descriptive only, no design):**

| Aspect | User (`queue_message`) | System (`defer_message`) |
|--------|------------------------|--------------------------|
| Container | `self._queue: list[tuple[str, str\|None]]` | `self._deferred_messages: list[dict]` |
| Payload | tuple `(content, nonce)` | dict `{role, content, source, timestamp}` |
| Role | implicit user (set in loop) | explicit (`role="system"`/`"daemon"`) |
| Consumed by | main queue-drain `loop.py:292` | post-tool batch `loop.py:803` |
| Timing | next iteration boundary | after current tool batch |
| Inject API | `session.queue_message(content, nonce)` | `session.defer_message(content, role, source)` |

Both injection APIs are callable by the daemon today from outside the loop; neither requires mid-stream injection.

---

### Q6: Onboarding Flow — Workspace Orientation — **ANSWERED**

**File:** `agent_os/agent/prompt_builder.py` — `_onboarding_or_directive()`

**Answer:** **NOT IMPLEMENTED.** After onboarding completes and project_goals.md is written, the agent does **no** automatic workspace exploration and **no** file mapping/listing.

**Evidence:**
- Onboarding forbids tool use until done — `prompt_builder.py:420–422`:
  ```
  DO NOT use any tools (read, shell, write, edit, browser, etc.) until onboarding is complete.
  The only tool call you make during onboarding is the final `write` to create project_goals.md.
  After writing project_goals.md, announce that you're ready and begin working.
  ```
- After onboarding the prompt transitions to the PROJECT DIRECTIVE section (`:424–430`), which injects project_goals.md content. There is no exploration step, no file-listing tool call, and no index/map generation.
- The onboarding sequence (`:399–423`) covers only: greet user → clarify goals via dialogue → write project_goals.md → announce readiness.

---

## Summary Table

| Q | Topic | Status |
|---|-------|--------|
| Q1 | CONTEXT.md in session-end routine | **Fully implemented** (prompt + OCC + dedicated write block; 5 files total) |
| Q2 | Upload endpoint notifies agent | **NOT IMPLEMENTED** (disk write + return only) |
| Q3 | Pre-compaction workspace writes / checkpoint | **Implemented** (token-pressure refresh + memory flush before compaction; 3 checkpoint triggers) |
| Q4 | System prompt workspace-file maintenance | PROJECT_STATE/DECISIONS/LESSONS yes; **CONTEXT.md not mentioned**; no "index"/"map" |
| Q5 | Mid-session injection | USER queue implemented; **no system mid-stream injection** (deferred = post-tool only) |
| Q6 | Onboarding workspace orientation | **NOT IMPLEMENTED** (no exploration/mapping after project_goals.md) |

## Key Facts for the Implementation Spec

1. **CONTEXT.md already exists end-to-end** in the session-end routine (`workspace_files.py`) but is **absent from the system prompt** (`prompt_builder.py`) — so the agent is never told it exists or how to use it. An INDEX.md feature can model its session-end write block, OCC baseline, and sanity-check wiring directly on the existing CONTEXT.md pattern.
2. **Compaction already has a checkpoint/flush pipeline** (token-pressure refresh → memory flush → compaction → reorientation). Any INDEX.md write-before-compaction would hook the existing `run_session_end_routine()` / `_run_refresh` path rather than need new plumbing.
3. **Upload → agent notification is greenfield.** The upload handler is fully isolated; reuse of `session.queue_message()` (user-role, next-iteration) or `session.defer_message()` (system/daemon-role, post-tool-batch) are the two existing injection seams — neither touched by the upload handler today.
4. **No workspace-orientation/mapping step exists** anywhere (onboarding or otherwise). "index"/"map"/"workspace understanding" appear nowhere in the prompt.

---

*This document reports current state only. No solutions or implementation changes proposed, per the investigation scope.*
