# Feedback row 4 — Windows workspace grant froze the daemon (verify on Windows)

**Report.** Feedback row 4, 2026-11-07 07:56 UTC, Windows 11, v0.20.0
(`inst_bf94c768a37f`). The user signed in with TokenDance, used hy4, and imported
a large project folder on `D:\` (`AgentWorkPlace\AI+CodingProgram`, with git
repos nested inside it). The scan "never started". The app froze three times
(15:17, 15:35, 15:37). The user restarted twice, and the third time it came
back after **1016 s**. hy4 then finished the scan in about 50 s, so the model
was not the cause.

**Cause.** All three hang-log dumps show the event-loop thread inside
`AgentManager.start_agent` → `WindowsPlatformProvider.grant_folder_access` →
`PermissionManager.grant_access` → `icacls <workspace> /grant AgentOS-Worker:(OI)(CI)F /T /C /Q`.
The callers were the `cold-start-scan` and `start-onboarding` routes.

- `/T` (from v0.1.0) writes an explicit ACE on every file, one at a time. Spec
  109 removed it for the toolchain roots (measured at 182 s with it, 23 s
  without) but kept it here, on the belief that "the workspace is empty at
  setup". An imported project (spec 160) is not empty.
- It ran on **every** session start, not just the first one. Every `/grant`
  re-propagates through the whole tree even when the ACE is already there
  (spec 109 V1).
- It ran synchronously inside an `async def`, so chat, the UI, the websocket
  and every other session froze with it.

## What changed (batch/1107-160-161)

1. `permissions.py` `grant_access`:
   - **no `/T`**: the inheritable root ACE propagates, the same rule as the
     toolchain roots.
   - **skip**: when the root already has an explicit, non-inherited
     `(OI)(CI)` allow for the worker whose mask covers the mode, nothing is
     written. That costs one DACL read. The masks are `R` = `0x120089` and
     `F` = `0x1F01FF`, with generic bits mapped. `harden_writable_root` still
     runs first, and it has its own skip.
2. Off the event loop (`asyncio.to_thread`) at every async caller:
   - `start_agent` (the workspace grant and the scratch read portals)
   - the `request_access` approval
   - `POST /api/v2/platform/folders/grant`
   - `PUT …/sessions/{sid}/scope`
3. Tests: `tests/unit/test_windows_permissions.py` (grant shape, skip, no
   skip) and `tests/unit/test_grant_off_event_loop.py` (thread identity at
   each caller).

**Not done:** no UI status line while the first grant runs. With the app
responsive, the scan or onboarding request just stays pending. Add one only if
the first grant is still long after V2.

## Verify on Windows (Claude on the Windows box)

Use the **CI-built** installer from a dispatch of this branch
(`gh workflow run ci.yml --ref batch/1107-160-161`), not a local build. Get a
big test tree: copy a real repo tree with many files (more than 50k, with
nested `.git` and `node_modules`) to `D:\orbital-grant-test` (or `C:\` if
there is no D: drive). **Never point a test project at a real workspace**
(deleting a test project deletes its data).

**V1 — the mask values the skip expects (the skip rests on this).**
```powershell
New-Item -ItemType Directory C:\t\g -Force | Out-Null
icacls C:\t\g /grant "AgentOS-Worker:(OI)(CI)F" /Q
icacls C:\t\g /grant "AgentOS-Worker:(OI)(CI)R" /Q   # on a second fresh folder
.venv\Scripts\python -c "from agent_os.platform.windows import acl; [print(hex(a.mask), hex(a.flags), a.sid) for a in acl.read_dacl(r'C:\t\g')]"
```
Pass: the worker's ACE reads `0x1f01ff` (F) or `0x120089` (R), with flags
`0x3`. Any other value means `_MODE_MASKS` / `_file_rights` in
`permissions.py` needs it, or the skip never fires. That is safe but slow, so
report it.

**V2 — first grant time, before and after.** For both v0.20.0 and the new
build, time the old and new command on the test tree:
```powershell
Measure-Command { icacls D:\orbital-grant-test /grant "AgentOS-Worker:(OI)(CI)F" /T /C /Q }   # old
icacls D:\orbital-grant-test /remove AgentOS-Worker /Q
Measure-Command { icacls D:\orbital-grant-test /grant "AgentOS-Worker:(OI)(CI)F" /C /Q }      # new
```
Then check that a deep file shows `AgentOS-Worker:(I)(F)`:
`icacls D:\orbital-grant-test\<deep>\<file>`. Record both times. If the
`/T`-less grant does NOT reach the deep file, say so loudly. The fix is then
wrong and `/T` has to come back (off the event loop).

**V3 — the app stays responsive during the first grant.** In the installed
app, import `D:\orbital-grant-test` as a new project and start the scan. While
it is starting, switch projects, open another session and type in its
composer. Pass: the UI responds the whole time, `hang.log` gains no dump whose
loop stack is in `grant_access`, and the scan starts once the grant finishes.

**V4 — repeat sessions are instant.** Open a second new session in the same
project. In `orbital.log`, the gap between the session's user row and the
first LLM call should be well under a second, with no `Granted AgentOS-Worker
access` line for that start (the skip returns before logging).

**V5 — the worker can still write.** In that project, have the agent run a
shell command that writes a file in a deep subfolder and in a nested repo's
working tree. Pass: both writes succeed. W3 still holds: `git -C <nested repo>
config user.name x` run by the worker is denied.

**V6 — W3 pass cost, for the record.** The `_protect_control_files` pass after
the grant (`find_repository_roots`, depth 3, plus DACL reads) now also runs off
the loop. Read the `refresh_sandbox_grants(): … in N ms` line at daemon start.
On the user's machine it was 181 s and 65 s. If it is minutes on the test tree
too, report it as its own issue, since it slows the start of every session
(though it no longer freezes the app).

Report results in `REPORT-feedback4-windows.md` in the worktree. Fix only
inside `agent_os/platform/windows/permissions.py` and its tests. If V2 fails
(no propagation without `/T`), stop and report rather than redesigning.
