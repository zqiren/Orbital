# Investigation: `/new` + credential-deny spawns a duplicate, uncancellable agent

**Project:** orbital-marketing (`proj_e1164e981494`)
**Status:** Diagnosis complete. **No code changed** (investigation-only, per project rules).
**Date:** 2026-05-25
**Related branch:** `fix/stop-button-turn-cancel` (this bug lives in the same cancel/new-session/loop-lifecycle area)

---

## 1. Reported symptom

> "I typed `/new` previously, then typed something that triggered a credential
> request, which I denied. Then I saw one assistant agent (blue label) **and**
> another **Agent** producing the same message. The Agent is still showing
> browser tool calls in the interface."

In short: after `/new` followed by a denied credential request, **two agents run
at once on one project**, emit duplicate output, and one of them gets stuck and
cannot be stopped.

---

## 2. Evidence gathered (live system + transcripts)

### 2.1 The user mashed `/new` four times in 35 seconds
Session files created in `orbital/sessions/` (first→last entry timestamps, UTC):

| Session | Entries | First | Last | Contents |
|---|---|---|---|---|
| `75f62f4a` | 1 | 02:22:45 | 02:22:45 | meta only (empty) |
| `70f26a8f` | 1 | 02:22:54 | 02:22:54 | meta only (empty) |
| `dcf4295c` | 1 | 02:23:04 | 02:23:04 | meta only (empty) |
| `272f5780` | 3 | 02:23:19 | 02:24:33 | meta + **the two user task messages** |
| `e141c85a` | 114 | 2026-05-20 | **02:24:51** | the long-running OLD session |

Four `/new` invocations, ~10s apart.

### 2.2 User messages and agent responses landed in *different* sessions
- **New** session `272f5780` holds the user's actual task:
  - `02:23:59` user: *"能登陆我的小红书帐号吗？…帮我涨涨粉"* (log into my xiaohongshu, comment on AI posts, grow followers)
  - `02:24:33` user: *"不记得了，你可以直接用浏览器登陆"* (don't remember creds, just use the browser)
  - **Zero assistant responses.**
- **Old** session `e141c85a` holds *all* the responses to that task:
  - `02:24:21` assistant: *"好，我来试试。首先需要你的小红书登录凭证。"* → triggers `request_credential`
  - `02:24:38` assistant: *"好的，我来试试。先导航到小红书看看登录页面。"* → browser route
  - `02:24:41` **three** `CANCELLED: This tool call was not executed.`
  - `02:24:48` browser navigate **FAILED** (`ERR_ABORTED`) + another `CANCELLED`
  - `02:24:51` browser navigate **SUCCEEDED** ("Navigated to: 小红书 explore")

Two "我来试试" openers and two competing browser navigations (one fails, one
succeeds) interleaved in a single session = **two concurrent loop tasks writing
to `e141c85a`**.

### 2.3 Corroborating artifacts
- `orbital/approval_history.jsonl`: two `request_credential` **denials** at
  `02:24:14` and `02:24:41` — matches the user denying the credential.
- Orphaned sub-agent process **PID 40037** (`claude … --input-format stream-json`
  for `…/orbital-marketing`), parent = Orbital.app, **alive 16h47m**, sleeping —
  the stuck "Agent" that "is still showing browser tool calls" and cannot be
  dismissed. Its own transcript (`sub_agents/claude-code/820e5ff2.jsonl`) stopped
  updating at 17:58 the prior day — i.e. wedged, not working.
- Live monitor (8 min, 31 samples this morning): steady `idle` / 1 sub-agent —
  the incident is **not actively recurring**, but the orphan persists.

---

## 3. Root cause

The defect is a **cluster** in the management-agent lifecycle. Four
independent weaknesses combine; the headline cause is **(C) the preserved loop
object is never rebound to the new session**, but (A), (B), and (D) are what let
a *second* loop survive and run concurrently.

### (A) No per-project lifecycle lock
`agent_manager.py` has **no `asyncio.Lock`** guarding `new_session`,
`inject_message`, `cancel_message`, or `approve` for a given project (grep for
`Lock`/`async with` in that file returns nothing). Firing `/new` four times in
35s runs **four `new_session()` coroutines concurrently**, interleaved with the
inject/approve coroutines.

### (B) `new_session()` lets a non-terminating loop survive, then proceeds anyway
`agent_manager.py:1316-1324`:
```python
if handle.task is not None and not handle.task.done():
    await handle.loop.terminate()
    try:
        await asyncio.wait_for(asyncio.shield(handle.task), timeout=10.0)
    except (asyncio.TimeoutError, Exception):
        logger.warning("new_session(%s): loop did not stop gracefully", project_id)
```
`asyncio.shield(...)` means that when the 10s `wait_for` times out, **the loop
task is NOT cancelled** — it keeps running. The code only logs a warning and
continues. A loop that is paused on an approval, mid-tool, or inside a slow
browser call routinely misses the 10s window and survives.

Then `new_session()` runs a **synchronous session-end LLM summary with a 200s
timeout** (`:1336-1344`) *before* swapping the session (step 8 at `:1407`). So
for up to 200s after `/new`, `handle.session` still points at the **old**
session while the user can keep typing and mashing `/new`.

### (C) The preserved `handle.loop` is never rebound to the new session  ← core
`new_session()`'s docstring (`:1305-1311`) says it "**preserves the project
handle (loop, …) but swaps the session object**." It updates:
- `cm._session = new_session` (context manager, `:1396`)
- `handle.session = new_session` (`:1407`)

…but **never `handle.loop._session = new_session`.** The loop stores its session
once in `__init__` (`loop.py:97` → `self._session = session`) and that is the
**only** assignment to `loop._session` anywhere. Consequently the preserved loop
keeps reading and **writing to the original session object** (`e141c85a`) for
the rest of its life. This is exactly why every response landed in `e141c85a`
while the user's messages went to `272f5780` (delivered via `handle.session`).

### (D) `loop.run()` has no re-entrancy guard
`loop.py:200-204`:
```python
async def run(self, initial_message=None, initial_nonce=None) -> None:
    self._running = True                  # unconditional — no "if already running: return"
    self._task = asyncio.current_task()   # OVERWRITES the previous run's task handle
```
Nothing prevents a second concurrent `run()` on the same loop object, and the
second call **overwrites `self._task`**. `terminate()` only cancels
`self._task` (`loop.py:971-981`), so after a re-entry it can cancel **only the
latest** task — the earlier concurrent `run()` becomes an **orphan that
terminate()/`/new`/Stop can no longer cancel**. That orphan is the "Agent" that
won't go away and the leaked sub-agent process (PID 40037).

---

## 4. How the four combine to produce the exact symptom

1. `e141c85a`'s loop is mid-task and **pauses on `request_credential`** approval.
2. User mashes `/new` ×4 (no lock → 4 overlapping `new_session()` coroutines).
   Each `terminate()`+`shield(10s)` fails to kill the approval-paused/mid-tool
   loop in time → **the old loop survives** (B). Each then blocks on a slow
   pre-flush summary, so `handle.session` lags behind for many seconds.
3. The session swap eventually points `handle.session` at `272f5780`, but
   **`handle.loop._session` is never rebound** (C) → the surviving loop still
   writes to `e141c85a`.
4. User's task messages are injected to `handle.session` (`272f5780`); the denied
   credential resolves the approval → `resume()` → `_start_loop()` starts
   **another** `handle.loop.run()` (D) concurrently with the survivor.
5. Two `run()` tasks now execute on the same loop/session → **duplicate "我来试试"
   openers, two browser navigations (one CANCELLED/`ERR_ABORTED`, one success),
   triple `CANCELLED` markers** — all in `e141c85a`.
6. UI shows the **new** session `272f5780` (blue "Assistant", where the user
   types) **and** the **old** still-running loop (`e141c85a`, the "Agent" stuck
   on browser tool calls). The orphaned `run()` task (D) is uncancellable →
   stuck Agent + leaked sub-agent process (PID 40037, alive 16h47m) + wedged
   browser page.

Every observed datum is accounted for.

---

## 5. Reproduction (minimal)

1. Start the orbital-marketing agent on a task that will call a tool requiring
   approval (e.g. `request_credential` or any browser/credential action).
2. When the approval card appears, **deny** it.
3. Immediately press **New session (`/new`) several times in a row** while the
   turn is still resolving.
4. Send a new task message.

Expected (bug): two agents render; output appears in the previous session, not
the new one; one agent gets stuck on a tool call and cannot be stopped; a
`claude … stream-json` sub-agent process leaks.

---

## 6. Current residual state

- `proj_e1164e981494` run-status: **idle** (not actively duplicating now).
- **PID 40037** orphaned sub-agent still alive (16h47m). Safe to reap with
  `kill 40037` once you confirm you don't need its wedged browser page. (Not done
  here — investigation only.)
- Stale new-session files `75f62f4a` / `70f26a8f` / `dcf4295c` are empty
  (meta-only) and harmless.

---

## 7. Recommended fixes (NOT implemented — for review)

Ordered by leverage:

1. **(C) Rebind the loop's session in `new_session()`** — set
   `handle.loop._session = new_session` (and any other component holding a
   session ref) right where `cm._session` is updated. This alone stops the
   cross-session write that defines the bug. Best as an explicit
   `loop.rebind_session(new_session)` method rather than poking a private attr.
2. **(D) Make `loop.run()` non-re-entrant** — at the top: `if self._running:
   logger.error(...); return` (or assert), so a second concurrent `run()` can
   never orphan the first task.
3. **(A) Add a per-project lifecycle lock** — serialize `new_session` /
   `inject_message` / `cancel_message` / `approve` with one `asyncio.Lock` per
   project so rapid `/new` can't interleave.
4. **(B) Don't let a non-terminating loop survive silently** — drop the
   `shield`, or after the timeout actually `handle.task.cancel()` and verify it
   is `done()` before swapping; consider doing the (potentially 200s) pre-flush
   summary *after* the session swap / in the background so `/new` returns fast.
5. **Lifecycle test** — a regression test that fires N concurrent `new_session()`
   while a turn is approval-paused and asserts: exactly one live loop task, the
   loop's `_session is handle.session`, and no leaked sub-agent.

Per the project's `test-gated-bugfix` rule, any fix should start from a failing
test reproducing §5.
