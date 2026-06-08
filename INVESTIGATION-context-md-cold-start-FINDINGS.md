# INVESTIGATION-context-md-cold-start — FINDINGS

**Type:** Findings doc (diagnosis only — no code edits made)
**Serves:** writing `TASK-context-md-cold-start` hermetically
**Method:** live code reads + on-disk artifact mining. Prior Claude implementation claims were treated skeptically and re-verified against source. Every file:line below was read directly this session.
**Status:** Complete. All 8 questions resolved. Q6 and Q7 (the design-redirecting ones) answered definitively.

---

## TL;DR — the two findings that change the design

- **Q6 (REDIRECT):** The agent *does* get a context-usage signal, but it is **coarse, estimated (`len/4`), and one turn stale** — plus a single emergency "pre-compaction flush" turn. There is **no on-demand budget tool and no precise remaining-token count**. So *"agent plans within one context window and stops near compaction"* is **only partially implementable**: express the bound in terms of the existing 70%/85% thresholds + the flush turn, or add a new tool. Do **not** specify fine-grained "I have N tokens left" planning — that signal does not exist.

- **Q7 (REDIRECT, in your favor):** CONTEXT.md **maintenance EXISTS, is deterministically wired, is test-covered, and left a real on-disk write on this very repo** (Apr 27 12:52 rewrite of CONTEXT.md+DECISIONS.md+SESSION_LOG.md together). It is **not** dead prompt-only code. **BUT** it only fires when a trigger is met: **≥50 turns**, **token pressure (>80%)**, or the agent **explicitly calling `checkpoint_state`**. A short "import a workspace, understand it, done" session **hits none of these**, so maintenance will not cover the cold-start gap — which **confirms the cold-start scan is worth building**. Also: maintenance writes via a **utility-provider session-end routine → `_occ_write_metadata`**, **not** the agent's `write` tool — reusable plumbing for the cold-start write.

- **Reuse verdicts:** Q2 deterministic gitignore-walker = **NET-NEW**. Q3 propose→confirm-**with-edit**→write surface = **NET-NEW** (approve/deny exists; edit-before-write does not).

---

## Q1 — First-session trigger point & whether a content-less start path exists

**First-turn origin (chat):** `inject_message()` Case 3 in `agent_os/daemon_v2/agent_manager.py:1186-1191`. When no handle and no on-disk session exist, it calls `start_agent(... initial_message=content ...)`. This is the *only* path that auto-starts a fresh chat agent today, and every current caller passes user `content`.

**Project creation does NOT start anything:** `create_project` (`agent_os/api/routes/agents_v2.py:387-455`) writes `projects.json`, installs skills, and calls `_ensure_dispatcher` — no session, no loop, no message.

**Content-less start path EXISTS at the function level:** `start_agent` signature is
```python
# agent_os/daemon_v2/agent_manager.py:351-358
async def start_agent(self, project_id, config,
                      initial_message: str | None = None,   # ← defaults to None
                      ... session_id=None, queue_state="chat", session=None): ...
```
So a loop *can* be started with no user message. (System-only prompts are already handled downstream by `_KICKOFF_CONTENT = "Begin."` in `agent_os/agent/providers/openai_compat.py`.) No current caller exercises the content-less first turn, but the plumbing supports it.

**→ Evidence for the task author:** the earliest hook to inject a consent prompt *before* any agent work is the tail of `create_project` (`agents_v2.py:455`) — mint the chat session there and either (a) start the loop content-less in a "consent" mode, or (b) have the frontend render a consent prompt and only call `start_agent` on "yes". A content-less start path exists; the first-turn reactive origin is `inject_message` Case 3.

---

## Q2 — Deterministic gitignore-respecting tree-walk: reuse vs net-new

### VERDICT: **NET-NEW.** No existing walker combines recursive descent + .gitignore respect + per-file sizes + self-bounding.

**Every directory-traversal site found, and why each fails the requirement:**

| Candidate | File:line | Recursive | .gitignore | Sizes | Verdict |
|---|---|---|---|---|---|
| `ReadTool._list_directory` | `agent_os/agent/tools/read.py:66-80` (`os.listdir`, `os.path.getsize` at :76) | No (1 level) | No | Yes | one-level only |
| `GlobTool` | `agent_os/agent/tools/glob_tool.py:66` (`Path.glob`), cap `_MAX_RESULTS=1000` at :17 | Yes (`**`) | **No** | No | no gitignore, no sizes |
| `files_v2.list_files` (powers FileExplorer) | `agent_os/api/routes/files_v2.py:68-91` (`os.listdir`, `stat.st_size` at :86) | No (1 level) | No | Yes | one-level only |
| `skills.py` | `agent_os/agent/skills.py:30` (`os.scandir`) | No (1 level) | No | No | one-level only |
| `GrepTool` (ripgrep) | `agent_os/agent/tools/grep_tool.py` | Yes (via `rg`) | **Yes** (rg honors .gitignore) | No | search, not a sized listing |
| `os.walk` in repo | `agent_os/api/routes/agents_v2.py:2119` | — | — | — | tmpdir cleanup only, unrelated |

**Dependency check (no gitignore lib present):** `pyproject.toml` dependencies = openai, anthropic, fastapi, uvicorn, httpx, trafilatura, keyring, python-multipart, websockets, croniter, psutil, aiofiles, watchdog, claude-agent-sdk. **No `pathspec` / gitignore parser.** Net-new walker must add a dep **or** hand-roll gitignore **or** shell out to ripgrep (`rg --files` honors .gitignore and is already vendored — see `agent_os/vendor/rg/`).

**Self-bounding guard is feasible — mirror existing cap patterns:**
- `glob_tool.py:17` `_MAX_RESULTS=1000` + truncation notice `"[... N+ more paths, refine your pattern]"` (:101-107)
- `grep_tool.py:35-38` `_MAX_MATCHES=100`, `_MAX_COUNT_PER_FILE=50`, `_TIMEOUT_SECONDS=10`, truncation notice (:220-228)
- Size via `os.stat(p).st_size` (already used `read.py:76`, `files_v2.py:86`).

**→ Note for task author:** `rg --files` (vendored ripgrep, gitignore-aware) + a `os.stat` size pass is the cheapest path to a gitignore-respecting sized skeleton without a new Python dep. Hand-rolling gitignore is the fragile alternative.

---

## Q3 — Propose → confirm → write surface: reuse vs net-new

### VERDICT: **NET-NEW for "confirm-WITH-EDIT-before-write."** An approve/deny gate exists; an *edit-the-proposed-content* gate does not.

**What exists (approve/deny only):**
- Interceptor: `agent_os/daemon_v2/autonomy.py` `AutonomyInterceptor` — `should_intercept` (:53), `on_intercept` (:103) broadcasts `approval.request` and stashes `_pending_approvals[tool_call_id]`.
- SDK permission gate: `agent_os/agent/transports/sdk_transport.py:483` `_handle_permission`, payload keys `request_id`, `tool_name`, `tool_input` (:500-504).
- API: `ApproveRequest` (`agent_os/api/routes/agents_v2.py:1392`) fields = `tool_call_id`, `reply_text`, `response_payload`, `approve_all`. **No `edited_content` / `modified_tool_args` field.** `reply_text` is *guidance to the agent*, not edited artifact content.
- Execution uses the **original** `tool_args` (`agent_manager.py:1638-1651`) — there is no path that swaps in user-edited content.
- Frontend `ApprovalCard.tsx` renders what/tool/args + an optional "guidance" textbox (:218-224). No artifact editor.

**How `project_goals.md` is created today:** during onboarding the agent calls the ordinary `write` tool (`agent_os/agent/tools/write.py`) directly — **no confirm step, no user gate**. (Prompt instruction at `prompt_builder.py:457`.)

**→ For the design's "user edits Goals before write," net-new is required:** (1) a new response field carrying edited content, (2) a frontend editor for the proposed structured artifact (not the guidance textbox), (3) a write path that consumes user-edited content. The existing approve/deny interceptor can be a *scaffold* but not the surface as-is.

---

## Q4 — Where State / Goals / Instructions live, and separability

| Artifact | Disk path (via `agent_os/agent/project_paths.py`) | Write path | Read path | Separable? |
|---|---|---|---|---|
| **State** | `orbital/PROJECT_STATE.md` (`project_state`, :58) | session-end routine `_occ_write_metadata(...,"state",...)`; also gates `is_onboarding_complete` (`agent_manager.py:2669-2682`) | injected every turn (Layer-1 / `_memory`) | **YES — clean, agent-owned** |
| **Goals** | `orbital/instructions/project_goals.md` (`project_goals`, :86-87) | onboarding agent `write` tool; **AND** Settings-UI sync from the `instructions` field (see below) | `prompt_builder._onboarding_or_directive` (:436) — present ⇒ "## PROJECT DIRECTIVE"; absent ⇒ onboarding | **ENTANGLED** |
| **Instructions (user, prescriptive)** | `orbital/instructions/user_directives.md` (`user_directives`, :90-91) | Settings UI `user_directives_content` → `_write_workspace_file` (`agents_v2.py:511`); agent may append (`prompt_builder.py:572`) | `prompt_builder.py:523-527`, rendered as **"## Project Instructions"** | **YES — user-owned** |
| **CONTEXT.md** | `orbital/CONTEXT.md` (`context`, :74-75) | session-end routine (see Q5/Q7) | Layer-1 every turn | agent-owned |
| `instructions` (projects.json field) | `projects.json` (not a file) | `create_project` (`agents_v2.py:405`), `update_project` | injected into onboarding as `context.project_instructions` (`prompt_builder.py:445`) | **syncs into Goals** |

### ⚠️ ENTANGLEMENT + NAMING TRAP (load-bearing for the design)

Verbatim code comment, `agent_os/api/routes/agents_v2.py:501-507`:
> "The Settings UI writes to the `instructions` field. Sync it to instructions/project_goals.md so the prompt builder (which reads from disk) sees it… The `instructions` key itself stays in `updates` so it is still persisted in projects.json for backward compatibility."

```python
# agents_v2.py:506-511
if goals_content is None and updates.get("instructions") is not None:
    goals_content = updates["instructions"]
if goals_content is not None:
    _write_workspace_file(workspace, "project_goals.md", goals_content)
if rules_content is not None:
    _write_workspace_file(workspace, "user_directives.md", rules_content)
```

Two traps the task author must not trip on:
1. **`instructions` (projects.json) → maps to GOALS (`project_goals.md`), not to "Instructions."** The field name lies relative to the design vocabulary.
2. The design's **"Instructions" (prescriptive, user-owned) = `user_directives.md`**, which the prompt renders under the heading **"## Project Instructions."** Different artifact, colliding name.

**Mapping the design's hard ownership split onto current storage:**
- **State (agent-owned)** → `PROJECT_STATE.md`. ✅ Already clean and separable.
- **Goals (agent drafts, user edits & owns)** → `project_goals.md`, **but** dual-sourced with the `projects.json:instructions` field. ⚠️ The task must decide: retire/ignore the field, or treat `project_goals.md` as canonical and stop syncing.
- **Instructions (user owns, agent abstains)** → `user_directives.md`. ✅ Separable; agent must be told not to write it during cold-start.

---

## Q5 — CONTEXT.md write + read path

- **Disk path:** `orbital/CONTEXT.md` — `project_paths.py:74-75` (confirmed).
- **Write cap:** `_CONTEXT_TOKEN_CAP = 1500` (`agent_os/agent/workspace_files.py:347`), applied via `_cap_context_tokens` truncating at the last complete line under the budget (:350+). Prompt *targets* <1000 tokens; cap is a 1500-token backstop. The comment notes it preserves the input string object to avoid thrashing the prefix cache.
- **Injection:** CONTEXT.md is a Layer-1 file (`agent_os/agent/context.py:33`, `_LAYER1_FILES`), injected **in full, every turn, with NO read-time truncation**:
```python
# context.py:224-230
for key, filename in _LAYER1_FILES:
    content = self._workspace_files.read(key)
    if content:
        layer_messages.append({"role": "system", "content": f"[{filename}]\n{content}"})
```
**Consequence:** an oversized CONTEXT.md (e.g., a user hand-edit beyond 1500 tokens, which the cap does not police on read) is re-injected wholesale on every turn. Write-time cap ≠ read-time cap.

---

## Q6 — Can the agent observe proximity to compaction? **DEFINITIVE: PARTIALLY — coarse, estimated, one turn stale. NO precise/real-time/on-demand signal.**

**The mechanism:**
- Compaction trigger: `should_compact()` = `_last_usage_pct > 0.80` (`agent_os/agent/context.py:81`).
- Usage is an **estimate**: `total_tokens = sum(estimate_message_tokens(m) …) / available_budget` (`context.py:308-312`), and `estimate_message_tokens` is `len(text)/4`. **Not** real provider token counts.
- Compaction = LLM summarization keeping the recent tail (`agent_os/agent/compaction.py`).

**What the agent CAN see (this corrects an earlier first-pass claim that it sees nothing):**
- Every turn, `_context_budget` injects a usage line into the prompt (`agent_os/agent/prompt_builder.py:711-725`):
  - always: `"Context usage: ~{pct}%."`
  - `>70%`: "Consider updating PROJECT_STATE.md and CONTEXT.md now."
  - `>85%`: "URGENT: Save all important state… Context will be compacted soon."
- This is the **truly-dynamic** section appended last each turn (`context.py:304-305`).
- One **emergency flush turn** before compaction: `MEMORY_FLUSH_PROMPT` is appended and a flush LLM call runs (`loop.py:991-999`), giving the agent exactly one extra chance to persist state.

**What the agent CANNOT do (the redirect):**
- **No tool** to query remaining budget on demand (tool inventory in `agent_os/agent/tools/` has read/write/edit/glob/grep/shell/browser/agent_message/checkpoint_state/triggers/request_access/request_credential/notify/queue_signals — none expose context budget).
- The number is **estimated** (`len/4`) and **one turn stale** (computed from this turn's assembled context, read by the model next turn). Near the limit it is unreliable.

**Falsifiable statement (verified true):** `grep -rn "context_usage_pct" agent_os/agent/` shows the value reaches the model **only** through `_context_budget` text injection (`prompt_builder.py:711-725`); no tool and no other message surfaces it.

**→ Design consequence for the task author:** Replace *"plan within one context window and stop near compaction (a precise bound)"* with one of:
1. **Use the existing coarse bound:** instruct the cold-start agent to read until the injected usage crosses ~70%, then stop and propose — accepting ±estimation slop and the one-turn lag; rely on the pre-compaction flush as the safety net.
2. **Add a net-new `context_remaining` tool** (real provider usage, queryable mid-run) if a tighter bound is required. This is new work; flag it in the task.
Either way, do not assume the agent can count tokens.

---

## Q7 — Does CONTEXT.md maintenance actually fire during normal work? **DEFINITIVE: YES — path exists, is deterministically wired, is test-covered, and left a real on-disk write. Caveat: only when a trigger condition is met.**

**The maintenance path (write goes through the session-end routine, NOT the agent `write` tool):**
- `run_session_end_routine` (`agent_os/agent/workspace_files.py`) calls a **utility-provider LLM** that returns structured JSON (`state`/`decisions`/`lessons`/`context`/`session_log`); each non-empty field is written deterministically via `_occ_write_metadata` under optimistic-concurrency control. CONTEXT.md write: `workspace_files.py:861-868` (`if result.get("context","").strip(): _cap_context_tokens(...) → _occ_write_metadata(..., "context", ...)`).
- **Three deterministic triggers in the loop, all → `_run_refresh → session_end_refresh_callback → run_session_end_routine(bypass_idempotency=True)` (callback at `agent_manager.py:576-598`):**
  1. **Turn-count:** `loop.py:403-415`, fires when `_turns_since_last_update >= COOLDOWN_TURNS`, and `COOLDOWN_TURNS = 50` (`loop.py:55`).
  2. **Token-pressure:** `loop.py:968-978`, fires before compaction when `should_compact()` (>80%).
  3. **Agent-decided:** the `checkpoint_state` tool → `trigger_checkpoint` → `_run_refresh("agent_decided", …)`.

**LIVE on-disk evidence it fired (not a prompt mention, not just tests):**
- This repo's own workspace `orbital/`:
  - `CONTEXT.md` (1876 B), `DECISIONS.md` (1896 B), `SESSION_LOG.md` (803 B) **all mtime `Apr 27 12:52:49`** — the session-end routine writes these together, so the shared timestamp is its fingerprint.
  - `PROJECT_STATE.md` and `LESSONS.md` mtime `Apr 3 14:08` — the *earlier* creation cluster. The Apr 27 rewrite of the other three is therefore a **maintenance** event distinct from initial creation.
  - `SESSION_LOG.md` content has structured "Session 1 / Session 2 — Completed/Attempted/Clarified" entries → the routine's LLM summary output, not hand-written.
- **Why there is no JSONL tool-call trace:** maintenance writes via `_occ_write_metadata` (a direct file write from the utility-provider JSON), **not** the agent's `write` tool — so grepping session JSONL for a CONTEXT.md `write` call correctly finds nothing even though maintenance fired. (The only surviving session JSONL, `orbital/sessions/orbital-bug-log_3f80036b.jsonl`, is 13 lines from Apr 3 — far short of the 50-turn trigger — and predates the Apr 27 write.)
- Test coverage exercising the firing: `tests/.../test_periodic_refresh_30_turn_session.py` (asserts turn-count refresh events) and `test_context_md_lifecycle.py` (asserts an existing CONTEXT.md is rewritten and re-read). (Reported by sub-agent; not re-run this session — the on-disk artifact above is the primary live evidence.)

**The caveat that matters for the design:** maintenance fires **only** at ≥50 turns, >80% context, or an explicit `checkpoint_state` call. A typical "import workspace → understand it → confirm → done" flow is short and triggers **none** of these. **So existing maintenance does not close the cold-start gap — it reinforces the need for the scan.** Deprioritizing maintenance is safe; building cold-start is not redundant with it.

**Reusable for cold-start:** the `run_session_end_routine` → `_occ_write_metadata` plumbing already does "produce structured content via a utility LLM, then write CONTEXT.md/STATE/etc. with OCC and a token cap." The cold-start "propose → write" stage can reuse `_occ_write_metadata` + `_cap_context_tokens` rather than routing through the agent's `write` tool.

---

## Q8 — Current onboarding gate and its off-switch

- Gate: `prompt_builder._onboarding_or_directive` (`agent_os/agent/prompt_builder.py:433-493`).
- **Off-switch = presence of `project_goals.md`:**
```python
# prompt_builder.py:436-438
goals_path = ProjectPaths(context.workspace).project_goals
content = self._read_truncated(goals_path)
if content is None:        # no project_goals.md → ONBOARDING MODE
    ...                    # else → "## PROJECT DIRECTIVE" (the file's content)
```
- Onboarding body (`:439-486`) tells the agent to greet, set goals, **write `project_goals.md`**, then (non-scratch) "explore the workspace… create CONTEXT.md (under 1000 tokens)." The CONTEXT.md instruction at `:472-485` is **pure natural-language**, no deterministic walk (ties back to Q2).
- `_read_truncated` cap is `_BOOTSTRAP_TRUNCATE = 20_000` chars (read-back of goals).

**→ What to repurpose vs replace:** the gate is a clean seam — writing `project_goals.md` is what flips a project out of onboarding. The cold-start scan should slot in *around* this gate: the scan produces the deterministic skeleton + proposed CONTEXT.md/State, while Goals (writing `project_goals.md`) remains the onboarding off-switch and stays user-owned per the design.

---

## Reuse-vs-net-new ledger (for the task author)

**Reuse:**
- Vendored ripgrep `rg --files` (gitignore-aware) + `os.stat` sizes → cheapest skeleton source (Q2).
- Cap/truncation patterns: `glob_tool._MAX_RESULTS`, `grep_tool._MAX_MATCHES` notices (Q2).
- Write plumbing: `_occ_write_metadata` + `_cap_context_tokens` from the session-end routine (Q5/Q7).
- Onboarding gate seam: `project_goals.md` presence (Q8).
- Coarse context signal: `_context_budget` 70%/85% thresholds + pre-compaction flush (Q6).
- Content-less `start_agent` entry (Q1).

**Net-new:**
- Deterministic gitignore-respecting **sized** tree-walk with self-bounding truncate-with-notice (Q2).
- Propose→**edit**→confirm→write surface: `ApproveRequest.edited_content` field + frontend artifact editor + edited-content write path (Q3).
- (Conditional) `context_remaining` tool if a precise read-budget bound is required instead of the coarse one (Q6).
- A first-session **consent-to-scan** trigger wired at/after `create_project` (Q1) — nothing fires today.

## Open decisions a fresh instance still must make (not blockers, but call them out)
1. Goals/instructions de-entanglement: retire or ignore the `projects.json:instructions`→`project_goals.md` sync, or make `project_goals.md` canonical (Q4).
2. Bound for stage-2 reading: accept the coarse 70% signal, or invest in a `context_remaining` tool (Q6).
3. Skeleton source: vendored `rg --files` vs hand-rolled gitignore vs new `pathspec` dep (Q2).
4. Consent UI: reuse the approval interceptor scaffold or build a dedicated consent prompt (Q1/Q3).
