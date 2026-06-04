# INVESTIGATION — 2026-05-28 round 3, slot enforcement during sub-agent dispatch

Investigator: backend bug-hunt sub-agent. No code changes performed. The
coordinator believed `bf3c97c` would fully fix R3-1 as a downstream
consequence of the SDK on_completed routing repair. The user retested and
reports the slot is still not enforced. This report lists three potential
root causes with raw evidence.

---

## Top-line verdict

**`bf3c97c` IS loaded in the running .app.** Confirmed by disassembling the
PYZ archive from `/Applications/Orbital.app/Contents/MacOS/Orbital` (PID
39197, binary mtime `May 28 13:51:11 2026`, ELAPSED 05:44 at observation
time) and inspecting the frozen bytecode signatures for the functions the
fix touched:

```
=== process_manager ===
  ProcessManager.start:           argcount=5  kwonly=1
                                  varnames[:8]=('self', 'project_id', 'handle',
                                                 'adapter', 'transcript',
                                                 'session_id', 'key', 'consume')
=== sub_agent_manager ===
  SubAgentManager.start:          argcount=4  kwonly=1
                                  varnames[:8]=('self', 'project_id', 'handle',
                                                 'depth', 'session_id', 'sk',
                                                 'current_count', 'config')
  SubAgentManager._start_from_registry:
                                  argcount=4  kwonly=1
                                  varnames=(... 'session_id', ...)
```

Both `_process_manager.start(...)` call sites in `SubAgentManager` forward
`session_id` as a kwarg in the frozen bytecode (CALL_KW with kw-name tuple
`('transcript', 'session_id')` at both call sites — `:221` and `:596`).
Extraction: `/tmp/orbital-r3/{process_manager,sub_agent_manager,
agent_manager}.code` (marshal-dumped from
`PyInstaller.archive.readers.ZlibArchiveReader('PYZ.pyz').extract(...)`).

So the loop-wake fix really is in the running process. The single-slot bug
persists *despite* the wake working — see hypotheses below.

---

## Test sessions on disk

Latest two `Quick Tasks` JSONLs (mtime 13:55, both):

```
quick_tasks_b1aa8a5e.jsonl  35,797 B   (session B — viewed and "leaked into")
quick_tasks_93304278.jsonl  11,018 B   (session A, F1=sess_e4277f84 —
                                        the session that dispatched)
```

The relevant tails:

```
SESSION A (quick_tasks_93304278 / F1=sess_e4277f84)
 14  2026-05-28T05:55:14.316Z  user/user        "再让他列一下20-40之间的质数"
 15  2026-05-28T05:55:17.865Z  assistant/mgmt   tool_call agent_message{action:send,agent:claude-code}
 16  2026-05-28T05:55:17.866Z  tool/mgmt        "Error: agent 'claude-code' not running"
 17  2026-05-28T05:55:20.369Z  assistant/mgmt   tool_call agent_message{action:start,agent:claude-code}
 18  2026-05-28T05:55:21.753Z  tool/mgmt        "Started Claude Code"
 19  2026-05-28T05:55:21.753Z  system/daemon    "[Sub-agent] claude-code started …"
 20  2026-05-28T05:55:24.948Z  assistant/mgmt   tool_call agent_message{action:send,...}
 21  2026-05-28T05:55:24.952Z  tool/mgmt        "Dispatched to claude-code. Awaiting completion…"
 22  2026-05-28T05:55:24.952Z  system/daemon    "[Sub-agent] Message sent to claude-code: …"
 23  2026-05-28T05:55:28.302Z  system/daemon    "[Sub-agent] claude-code completed. Summary: 20到40之间…23, 29, 31, 37"
 24  2026-05-28T05:55:31.799Z  assistant/mgmt   "已派发给 Claude Code，等待结果中。…23, 29, 31, 37 这 4 个质数 🎯"

SESSION B (quick_tasks_b1aa8a5e)
 18  2026-05-28T05:55:34.840Z  user/user        "what happened？"
 19  2026-05-28T05:55:55.249Z  assistant/mgmt   "Nothing bad — the file write needed your approval, …"
```

The wake fix is observably working — record 23 (`[Sub-agent] claude-code
completed. Summary: …23, 29, 31, 37`) lands in session A's JSONL, and the
management agent's response at record 24 follows ~3.5 s later. This is the
exact signal that was missing in the round-2 investigation; B1 is fixed.

---

## Daemon log — the round-3 window

`~/Library/Application Support/Orbital/logs/daemon.log`, verbatim
(`2026-05-28 13:55:14` through `13:55:35`):

```
13:55:14,282  inject_message(proj_6be4f16fb272): hydrating session sess_e4277f84 (uuid quick_tasks_93304278) from disk
13:55:14,310  Enabled sub-agents for project proj_6be4f16fb272: ['claude-code', 'codex', 'gemini-cli']
13:55:14,315  Sleep prevention enabled: Orbital: agent(s) running
13:55:14,769  POST api.deepseek.com 200 OK
13:55:17,862  [CACHE_AUDIT] input=10835 cached=6400 output=128
13:55:18,182  POST api.deepseek.com 200 OK
13:55:19,390  project_store: flushed runtime updates
13:55:20,369  [CACHE_AUDIT] input=8867 cached=4480 output=97
13:55:22,048  POST api.deepseek.com 200 OK
13:55:23,152  inject_message(proj_6be4f16fb272): hydrating session quick_tasks_b1aa8a5e (uuid quick_tasks_b1aa8a5e) from disk    ← FIRST B INJECT
13:55:24,946  [CACHE_AUDIT] input=9236 cached=2304 output=99
13:55:24,952  yield_turn: dispatch tool ended the management turn                                                                ← A yields
13:55:24,953  Sleep prevention disabled
13:55:28,637  POST api.deepseek.com 200 OK                                                                                       ← A resumed (push wake)
13:55:29,393  project_store: flushed runtime updates
13:55:31,796  [CACHE_AUDIT] input=9547 cached=2304 output=81                                                                     ← A's resume reply done
13:55:34,818  inject_message(proj_6be4f16fb272): hydrating session quick_tasks_b1aa8a5e (uuid quick_tasks_b1aa8a5e) from disk    ← SECOND B INJECT
13:55:34,834  Enabled sub-agents for project proj_6be4f16fb272: ['claude-code', 'codex', 'gemini-cli']
13:55:34,839  Sleep prevention enabled
13:55:35,147  POST api.deepseek.com 200 OK
13:55:55,249  [CACHE_AUDIT] input=15592 cached=6400 output=1253
13:58:29,380  evicting idle project proj_6be4f16fb272 session sess_e4277f84
```

Two telling observations:

1. **The 13:55:23.152 inject to session B is followed by NO
   `Enabled sub-agents` line.** `Enabled sub-agents` logs at
   `agent_manager.py:383`, which is reached only AFTER the slot guard at
   `:298` returns cleanly. So the slot guard fired and `start_agent`
   raised `ValueError("Slot held by session sess_e4277f84")`. The HTTP
   route (`agents_v2.py:812-827`) translated the exception into a 202
   `{"status":"slot_held","holding_session_id":"sess_e4277f84",…}`
   response.

2. **The 13:55:34.818 inject to session B IS followed by
   `Enabled sub-agents` at `13:55:34.834` and a real LLM call.** By
   13:55:34 the slot is genuinely empty: session A's resume turn finished
   at 13:55:31.799, no sub-agent is busy, no idle-poll alive — so the
   guard correctly lets B through.

This means **the backend slot guard IS working** for the case the user
exercised. The user-perceived bug ("slot wasn't enforced") most likely
maps to one of three causes I rank below. The first two displace the
candidate hypotheses I was asked to consider; the third is the prior
investigation's §9 latent issue, which I keep in third place because it
does not fit the *single*-session-dispatch case the user actually ran.

---

## Hypothesis 1 (MOST LIKELY) — Frontend silently swallowed the 202 SlotHeldNotice trigger so the user saw "no enforcement"

### Hypothesis
The backend correctly returned `202 {status:"slot_held", …}` at 13:55:23,
but the React `ChatView`'s `setSlotHeldNotice({...})` either didn't render,
got dismissed by an immediately-following state effect, or the user
clicked through it so fast the notice flashed and vanished. The user
then re-sent at 13:55:34 *after* A had naturally finished, perceived the
second inject succeeding, and concluded the slot was never enforced.

### Mechanism (file:line traced)

1. `agents_v2.py:808-811` — HTTP route awaits
   `_agent_manager.inject_message(...)`.
2. `agent_manager.py:1015-1018` — inject hydrates session B's history then
   calls `start_agent(project_id, config, initial_message=content,
   initial_nonce=nonce, session_id=loaded.session_id, …)`. The log line
   at 13:55:23.152 is the `logger.info("inject_message(%s): hydrating
   session %s (uuid %s) from disk", …)` at `agent_manager.py:1011`. This
   confirms the request reached this branch.
3. `agent_manager.py:298-300` — `start_agent` reaches the slot guard:
   ```python
   holder = self.current_holder_session_id(project_id)
   if holder is not None and holder != session_id:
       raise ValueError(f"Slot held by session {holder}")
   ```
   At 13:55:23.152, session A's handle is in `_handles` and `handle.task`
   is alive (next LLM POST `13:55:22.048 → cache_audit 13:55:24.946`
   spans the inject), so `current_holder_session_id` returns
   `"sess_e4277f84"` per `agent_manager.py:1257-1262`:
   ```python
   if handle.task is not None and not handle.task.done():
       return sid
   ```
   `holder != session_id` (`"sess_e4277f84" != "quick_tasks_b1aa8a5e"`)
   → raises.
4. `agents_v2.py:812-827` — the HTTP route catches `ValueError`,
   re-checks the holder, returns `JSONResponse(status_code=202,
   content={"status":"slot_held","holding_session_id":holder,…})`.
5. `web/src/config.ts:43-107` — the frontend `api<T>()` wrapper treats
   202 as success (`response.ok` is true for 2xx) and parses JSON.
6. `web/src/components/ChatView.tsx:1634-1673` — `injectMessage` callback
   recognises the slot_held shape and calls `setSlotHeldNotice({...})`
   with the holding session id.
7. `web/src/components/ChatView.tsx:2073-2150` — the notice renders
   inline above the composer.

### Supporting evidence
- **Backend rejected the inject at 13:55:23.** Direct proof: the missing
  `Enabled sub-agents` log line for the 13:55:23 inject vs. its presence
  for both the 13:55:14 (A) and 13:55:34 (B-retry) injects. `Enabled
  sub-agents` is `agent_manager.py:383` — only reachable past the slot
  guard.
- **The JSONL never received a 13:55:23 user-message record.** Session B
  JSONL's next user record after the prior April 27 history is
  `2026-05-28T05:55:34.840Z "what happened？"` — the *13:55:34* (second)
  inject. Slot-rejected injects don't write user messages because
  `start_agent` raises before any persistence happens.
- **The user's report mentions a SECOND missing UX symptom**: "the
  approval request wasnt rendered." That suggests the frontend is having
  general trouble surfacing intermediate notices on this build —
  consistent with SlotHeldNotice also failing to render at 13:55:23.
- The chat UI hook used by `setSlotHeldNotice` was last touched by F1 in
  `bf3c97c` (the rawMessages optimistic-strip on slot_held rollback) —
  the change is plausible to have introduced a subtle render-timing bug.

### Ruling out
- I cannot rule this out from backend logs alone — the WS connection
  carries no record of which 202 the frontend actually saw, and the
  202 response body is not logged. The user's description ("slot wasnt
  enforced") is ambiguous between "I saw no notice and message went
  through" (this hypothesis) and "I saw nothing happen at the backend
  at all" (hypotheses 2/3).
- I did NOT find a state reset that would clear `slotHeldNotice` on
  session switch. `grep -n "setSlotHeldNotice(null)"
  web/src/components/ChatView.tsx` returns only the two explicit user
  paths (Wait button at :2077, CancelAndSend at :2090). So if the state
  *was* set at 13:55:23, it should have stayed visible until the user
  acted. But: the state-setting in `setSlotHeldNotice({...})` and the
  optimistic raw-message strip at :1655-1663 both run in the same
  `injectMessage` callback. A throw or React batching anomaly between
  them could land in a half-rendered intermediate state.

### Diagnostic to confirm (do NOT run yet)
Open Safari/Chrome DevTools Network tab, then in session A dispatch a
sub-agent and immediately try sending in session B. Observe the
`/api/v2/agents/{pid}/inject` row: if it shows `202 Accepted` with the
slot_held body, this hypothesis is the cause and the bug is purely
frontend rendering. If it shows `200 OK`, hypothesis 2 or 3 is correct.

---

## Hypothesis 2 (LIKELY) — Slot-guard gap between `loop.run()` returning and `_on_loop_done` registering the idle-poll

### Hypothesis
After `yield_turn` fires, the management loop's `task.done()` becomes
True *before* the done-callback runs and populates
`_idle_poll_tasks[project_id]`. During that gap, `current_holder_
session_id` returns `None` because none of its three conditions match:
the task is done (cond 1 false), there's no approval pause (cond 2
false), and the poll task hasn't been registered yet (cond 3 false).
A concurrent inject from session B in that gap escapes the slot guard.

This is hypothesis (c) you asked me to consider, sharpened: the
`current_holder_session_id` predicate doesn't cover the post-yield,
pre-callback window.

### Mechanism
1. `loop.py:732-735` — yield_turn ends the management turn:
   ```python
   if result.meta and result.meta.get("yield_turn"):
       logger.info("yield_turn: dispatch tool ended the management turn")
       exit_outer = True
       break
   ```
2. `loop.run()` returns. The asyncio task transitions to done.
3. The done callback `_on_loop_done` registered at
   `agent_manager.py:624` is queued — but `add_done_callback` does NOT
   run the callback synchronously. It schedules it via the loop's
   `call_soon`. So between the task transitioning to done and
   `_on_loop_done` actually executing, an arbitrary number of other
   ready coroutines may run.
4. `_on_loop_done` body (`agent_manager.py:2591-2606`) is what would
   register the poll:
   ```python
   busy = [a for a in active if a.get("status") != "idle"]
   if busy:
       self._broadcast(... status="waiting" ...)
       poll_task = asyncio.ensure_future(
           self._check_sub_agents_done(project_id, session_id=session_id)
       )
       self._idle_poll_tasks[project_id] = poll_task
   ```
   Until this line lands, `_idle_poll_tasks[project_id]` either is
   absent or holds a *previous* completed poll task.
5. If session B's inject HTTP handler runs in that gap,
   `current_holder_session_id` (`agent_manager.py:1257-1270`) walks
   `_handles`:
   - `handle.task.done() == True` → cond 1 false.
   - `handle.session._paused_for_approval == False` → cond 2 false.
   - `_idle_poll_tasks.get(project_id)` is `None` (or `done()`) →
     cond 3 false.
   Returns `None`. Slot guard at `:298-300` does not raise. **Leak.**

### Supporting evidence
- The relevant log lines:
  ```
  13:55:22,048  POST api.deepseek.com 200 OK     ← LLM call N for A
  13:55:23,152  inject_message(...): hydrating quick_tasks_b1aa8a5e   ← B inject
  13:55:24,946  [CACHE_AUDIT] input=9236 …       ← LLM N response processed
  13:55:24,952  yield_turn: dispatch tool ended  ← A yields
  ```
  In *this particular* run, B's inject lands at 13:55:23 — well BEFORE
  A's yield at 13:55:24.952. So the task was still alive (cond 1 held)
  and the slot guard fired. That's why the bug didn't visibly manifest in
  this trace — the timing missed the gap.
- But the gap is real and racy. Under load (e.g. user smashing send on
  session B while A is mid-yield, or queue-dispatcher kicking a retry),
  the inject could land precisely in the few microseconds between
  `loop.run()` returning and `_on_loop_done` running. The window is
  small but non-zero — Python's `call_soon` queue ordering is not
  bounded.
- The architecture explicitly relies on the *poll* being the third
  holding-condition for the post-yield waiting state
  (`agent_manager.py:1241-1245` docstring). Without a synchronous
  registration, that condition has a hole.

### Ruling out
- The specific timing in this user's session does NOT fit this
  hypothesis — B's inject was 1.8 s *before* A's yield, so the
  cond-1 check returned True. I cannot rule out that the user
  experienced this gap on a DIFFERENT attempt that didn't make it into
  the current JSONL trace (e.g. an earlier attempt where the user-as-
  observer perceived enforcement failing); I can only rule it out for
  the 13:55:23/13:55:34 sequence.
- A defensive fix would be to set a synchronous flag in the
  loop-done callback's enclosing scope (or have `current_holder_session_id`
  treat "task just done + sub-agents alive" as still holding via a
  `_sub_agent_manager.list_active(project_id, session_id=sid)` call as
  a fallback). But this requires code change and is out of scope here.

### Diagnostic to confirm
Add a microsecond-resolution timestamp to the existing log lines around
`loop.py:733` (`yield_turn`) and the `_on_loop_done` callback entry, then
fire 50 simultaneous injects to session B while session A is
yield_turn-ing repeatedly. Plot the inject-arrival-time histogram
against the [yield_turn, callback-runs] window. If any inject lands in
that window and slips past the guard, this is the bug.

---

## Hypothesis 3 (LATENT, NOT TODAY'S BUG) — `_idle_poll_tasks` keyed by bare `project_id` collides across sessions

### Hypothesis
The §9 latent issue from the round-2 investigation: when two sessions
within the same project both reach `_on_loop_done` with busy sub-agents,
the second `_on_loop_done` overwrites `_idle_poll_tasks[project_id]`
with its own poll task. The first session's poll is leaked (and
worse, condition 3 of `current_holder_session_id` now resolves to "the
*other* session's poll is alive" — which still returns the wrong sid
because the iteration in `current_holder_session_id` is over `_handles`
in dict-iteration order).

### Mechanism
- `agent_manager.py:101`:
  ```python
  self._idle_poll_tasks: dict[str, asyncio.Task] = {}  # project_id -> poll task
  ```
- `agent_manager.py:2606`:
  ```python
  self._idle_poll_tasks[project_id] = poll_task
  ```
- `agent_manager.py:1702-1703` — explicit acknowledgement that the key
  is bare:
  ```
  ``_idle_poll_tasks`` is keyed by bare ``project_id``
  (NOT ``SessionKey``), so the pop uses ``project_id`` directly.
  ```
- `current_holder_session_id` `:1267` reads
  `self._idle_poll_tasks.get(project_id)` inside its `for (pid, sid),
  handle in self._handles.items()` loop. So when it finds session A's
  handle and looks up the poll, it ALWAYS gets the project-scoped poll
  — which under collision is the session that registered LAST, not A.
  The return value is still `sid` from the outer loop variable (which
  is whichever handle the iteration is currently on). Result: the
  reported holder may be the wrong session, or "no holder" if the
  surviving poll exits while the leaked one still references a real
  busy state.

### Supporting evidence
- Code lines above. The shape of the bug is consistent: two-session
  dispatch over the same project would collide.
- Round-2 investigation §9 (line 245 of
  `INVESTIGATION-2026-05-28-backend-still-broken.md`) called this out
  as a "latent design issue ... Not triggered by the current bug but
  ... worth tightening".

### Ruling out for the current user-reported run
- The user's R3 trace has **exactly one** session dispatching at a time
  (session A dispatched claude-code at 13:55:21; session B never
  reached `start_agent` until 13:55:34, by which time A's sub-agent
  was long done). There is no concurrent dispatch from two distinct
  sessions in the same project during the relevant window. So the
  bare-key collision cannot have been the trigger HERE.
- A multi-session-dispatch repro would require: session A starts a
  sub-agent and yields; session B *also* starts a sub-agent and
  yields, before A's sub-agent finishes. Then the second `_on_loop_done`
  blots out A's poll. The user's current bug doesn't put us in that
  state — session B never even gets a sub-agent because its first
  inject is rejected (per hypothesis 1) and its second inject is a
  bare "what happened？" with no agent_message tool call.

### Diagnostic to confirm
Run two parallel sub-agent dispatches in the same project from two
different sessions (e.g. spin up session A and session B in Quick
Tasks, have each ask claude-code to count primes). Watch for which
`_idle_poll_tasks[project_id]` survives — the loser session will
silently lose its poll. This is the latent bug, separate from the
user's R3 report.

---

## Summary table

| # | Hypothesis | Fits R3 trace? | Loaded-bytecode confirmed? | Likely true cause? |
|---|---|---|---|---|
| 1 | Frontend swallowed 202 slot_held | YES — backend log shows guard fired at 13:55:23, no JSONL persistence; user saw apparent "no enforcement" | n/a (frontend) | Best fit |
| 2 | Yield-to-callback gap in `current_holder_session_id` | Plausible window exists, but the specific run shows guard fired (cond 1 was true) | Yes — guard predicate as documented at `agent_manager.py:1257-1270` | Likely-on-other-runs |
| 3 | `_idle_poll_tasks` bare-`project_id` collision | NO — only one session dispatched in this run | Yes — `agent_manager.py:101, :2606`; explicit comment at `:1702` | Latent only |

If only one root cause can be addressed first, **start with hypothesis 1
(frontend SlotHeldNotice render path)**. The backend log evidence already
proves the slot guard worked in this run; the user's perception of "not
enforced" is most cleanly explained by the notice failing to surface. If
that turns out to be a red herring, hypothesis 2 is the next-most-likely
backend mechanism and would justify hardening `current_holder_session_id`
to fall back to `_sub_agent_manager.list_active(project_id, session_id=sid)`
when conditions 1-3 all miss.

---

## Appendix — raw inspection commands and outputs

PYZ extraction:
```
python3 -c "
from PyInstaller.archive.readers import CArchiveReader
car = CArchiveReader('/Applications/Orbital.app/Contents/MacOS/Orbital')
data = car.extract('PYZ.pyz')
open('/tmp/orbital-r3/PYZ.pyz','wb').write(data)
"     # → 22,581,128 bytes
```

Module extraction:
```
python3 -c "
from PyInstaller.archive.readers import ZlibArchiveReader
import marshal
zar = ZlibArchiveReader('/tmp/orbital-r3/PYZ.pyz')
for t in ['agent_os.daemon_v2.process_manager',
          'agent_os.daemon_v2.sub_agent_manager',
          'agent_os.daemon_v2.agent_manager']:
    code = zar.extract(t)
    open(f'/tmp/orbital-r3/{t.split(\".\")[-1]}.code','wb').write(marshal.dumps(code))
"
```

Signature confirmation (excerpt):
```
=== process_manager ===
  ProcessManager.start: argcount=5 kwonly=1
    varnames[:8]=('self','project_id','handle','adapter','transcript',
                  'session_id','key','consume')
=== sub_agent_manager ===
  SubAgentManager.start.dis includes:
    LOAD_FAST                4 (session_id)
    LOAD_CONST              19 (('transcript', 'session_id'))
    CALL_KW                  5
  SubAgentManager._start_from_registry.dis includes the same pattern.
```

Process and binary:
```
$ ps -p 39197 -o pid,etime,command
  PID ELAPSED COMMAND
39197   05:44 /Applications/Orbital.app/Contents/MacOS/Orbital
$ stat -f "%Sm %N" /Applications/Orbital.app/Contents/MacOS/Orbital
May 28 13:51:11 2026 /Applications/Orbital.app/Contents/MacOS/Orbital
```

Files referenced:
- `/Users/keanezhou/Desktop/orbital-test/agent_os/daemon_v2/agent_manager.py`
  (slot guard `:298-300`, `current_holder_session_id` `:1224-1270`,
  `_on_loop_done` `:2496-2627`, `_idle_poll_tasks` `:101, :2606, :1267,
  :1702-1706`, `inject_system_message` `:872-902`, `_start_loop`
  `:2438-2480`).
- `/Users/keanezhou/Desktop/orbital-test/agent_os/api/routes/agents_v2.py`
  (`inject_message` route `:716-834`, slot_held translation `:812-827`).
- `/Users/keanezhou/Desktop/orbital-test/agent_os/daemon_v2/process_manager.py`
  (fix bf3c97c kw-arg addition, lines `:31-:50, :87-93, :95-100`).
- `/Users/keanezhou/Desktop/orbital-test/agent_os/daemon_v2/sub_agent_manager.py`
  (kw-forward at `:221, :596`).
- `/Users/keanezhou/Desktop/orbital-test/agent_os/agent/loop.py`
  (`yield_turn` `:732-735`).
- `/Users/keanezhou/Desktop/orbital-test/web/src/components/ChatView.tsx`
  (`slot_held` handling `:1634-1673`, render `:2073-2150`, state init
  `:350`).
- `/Users/keanezhou/Desktop/orbital-test/web/src/config.ts`
  (`api<T>` wrapper `:43-107`).
- `/Users/keanezhou/Library/Application Support/Orbital/scratch/orbital/sessions/quick_tasks_93304278.jsonl`
  (session A).
- `/Users/keanezhou/Library/Application Support/Orbital/scratch/orbital/sessions/quick_tasks_b1aa8a5e.jsonl`
  (session B).
- `/Users/keanezhou/Library/Application Support/Orbital/logs/daemon.log`
  (round-3 window 13:55:14 — 13:58:29).
