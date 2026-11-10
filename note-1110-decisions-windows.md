# batch/1110 Windows pass: items that need the owner's decision

Written 2026-11-10 by the Windows coordinator session. Everything that was a plain bug is fixed and
committed on `batch/1110` (see `note-1110-results-windows.md`, "Fixes made in this pass"). The items
below are product or architecture calls, or need something only you have. None of them blocks the
batch; each has a suggested default so you can answer in one word.

## Outcomes (owner's walk-through, 2026-11-10 morning, implemented on Windows)

| Item | Owner's call | Landed |
|---|---|---|
| 1 | say it, don't plumb it: an ⓘ beside Autonomy Level explains the level applies to Orbital's agent and other agents keep their own permission models | `3b8112b` |
| 2 | go | `285348f` (需要改 prefills the reply; prefixed answer is still the guarded change turn) |
| 3 | go | `44fa468`, `786d2a0` (refuse before persisting; `no_runner` names both remedies) |
| 4 | fix the note | `f7a626f` |
| 5 | leave | — |
| 6 | go | `9a46c79` |
| 7 | no guessed default ("what if 8000 is taken?"): read the bound port | `cb1b652` (`BoundPortProbe`; `/mcp` always attached) |
| 8 | go | `b4a5ef5` (`goals_draft_pending` stamp on the project) |
| 9, 10, 15 | accept | — |
| 11 | nothing needed: the reader already strips the BOM | — |
| 12 | keep `/schedule` listed, mark it | `109cc67` (Orbital cannot list or delete Claude's cloud routines, so showing them in Orbital's automation list is not possible today) |
| 13 | go | `f276d26` |
| 14 | go | `c8da35d` |
| 16 | copy ok | upgrade half done (results note §"Second pass"); downgrade half still needs an admin install of the v0.21 exe |
| 17 | install dsh with the OpenCode Go key | installed; ACP rows in the results note |
| 18 | fix the recipe | `f7a626f` |
| §C | the two test-side fixes | `1fb22e3` |

## A. Product / architecture (found by the smoke)

1. **Supervised autonomy is ignored for pinned workers (C3, same as the Mac).**
   `agent_os/daemon_v2/sub_agent_manager.py:1296` forces `Autonomy.HANDS_OFF` for every sub-agent, so a
   project set to Supervised still lets a pinned Claude Code create files with no approval card, and
   Codex on Windows runs `on-request` + Orbital auto-accept (`codex_transport.py:91–103`).
   Suggested default: pass the project's autonomy through for `user_pinned` sends (Claude Code's SDK
   `can_use_tool` already supports an approval pause); keep HANDS_OFF for manager-dispatched children.

2. **The confirm card's 需要改 sends at once, without the correction (167-W4).** `TaskCard.tsx:221`
   sends every option immediately, so 需要改 costs one worker turn that only asks "what's wrong?", and
   the real correction is a second turn. Typing the correction into the card's reply field answers the
   ask in one turn. Suggested default: for the card's `change` option, focus or prefill the reply
   field ("需要改：") instead of sending.

3. **Keyless with every worker disabled (A6): a send leaves a dead session and blames the key.** The
   typed 400 `missing_api_key` is right, but the session (user row + `start_failed` row) stays in the
   list, and the message only says "add an API key" when re-enabling a worker is the other remedy.
   Suggested default: refuse before persisting when no worker and no key exist, and name both remedies.

4. **W2 sign beside Skip shows the coin and the count, not the worker's name.** The 1110 note says
   "the coin sign with '2' and the worker name sits beside Skip"; the built sign (`TokenCostSign.tsx`,
   kind `importAgent`) shows only the count, with the name in the hover sentence. Either the note's
   wording or the sign needs to change. Suggested default: keep the sign as built (one glyph beside
   each button), fix the note.

5. **Changed-files hint misses shell writes.** Codex on Windows writes files with PowerShell
   `Set-Content`, so `worker_turn_changed_files` (`agent_manager.py:115`) reports "none" for a turn
   that created `notes.md`. Goals were still drafted (the trigger is `project_goals_missing`). Fixing
   it means parsing shell commands or diffing the workspace; not a quick patch. Suggested default: leave.

6. **Orbital's own tool details land in project memory.** After one mis-shaped `dispatch` call
   (`message` instead of `task`), Claude Code wrote a LESSONS.md entry about the Orbital MCP tool's
   schema. 167 keeps such rules out of the goals file; LESSONS.md has no such guard. Suggested
   default: a clearer validation error on the tool (name the `task` field) and leave LESSONS.md alone.

7. **A daemon started without AGENT_OS_PORT degrades silently.** Fixed for the isolated launcher
   (commit `2ad4f1a`), but `python -m uvicorn agent_os.api.app:create_app --factory --port 8000` (the
   command in CLAUDE.md) still attaches no dispatch endpoint, and the fallback is a shell `codex exec`
   plus a MEMORY.md recipe that primes later sessions to bypass Orbital. Suggested default: derive the
   dispatch URL from the bound port at startup when the variable is missing (uvicorn's `--port` is
   knowable from the app factory only with a lifespan hook), or at least refuse to attach the seat
   note when the tool is absent.

8. **Goals card never posts when the goals were written outside the record turn** (review finding,
   `loop.py` `_maybe_confirm_goals`). If the main run itself writes `project_goals.md` and the record
   turn is skipped (a queued user message, `_memory_record_wanted` false), the next run start resets
   the flag and the card is lost. Likewise `_worker_goals_drafts` is in-memory only, so a daemon
   restart mid record turn loses the card. Suggested default: post the card from the run end whenever
   `goals_missing_at_run_start` and the file exists now, record-turn or not; persist nothing.

9. **D4 wording.** "Hand this to Orbital: say hi." makes Claude Code answer in words that there is no
   Orbital agent to dispatch to, without calling the tool; the `unknown agent 'orbital'` path only
   runs when asked to call the tool explicitly. Nothing breaks. Suggested default: accept.

10. **The record turn's "nothing to record" reply.** It is an ordinary assistant row in `GET /chat`
    (hidden by the UI, `chatTransform.ts:1189`), for Claude Code and Codex. Fine for the UI; any
    other reader of `/chat` (the phone relay, exports) shows it. Suggested default: leave.

11. **Codex's goals file carries a UTF-8 BOM on Windows** (`Set-Content` default). Readers tolerate
    it (`read_text_lenient`), the first heading renders as `﻿# First draft` in raw viewers.
    Suggested default: strip a leading BOM when the daemon reads the goals/memory files.

12. **Keyless `/` picker offers Claude Code's `/schedule` and `/loop` (spec 166).** `/automate` is hidden when
    keyless (7423b9c), but with Claude Code pinned the worker's own `/schedule` ("Create, update, list, or run
    scheduled cloud agents (routines)…") is listed; that is the command behind the Mac's B8 side effect (a real
    cloud routine on the owner's account). Suggested default: hide worker commands whose description or name
    matches schedule / routine / cron when keyless, or at least `/schedule` by name, and say so in the picker.

13. **A daemon restart during a worker's FIRST turn loses the native thread (E3).** The resume record is written
    only when a turn completes (`lifecycle_observer.py:846`), so a first turn killed mid-run leaves no
    `sub_agent_thread` meta row and `_determine_resume` (`sub_agent_manager.py:759–767`) starts fresh although
    Codex created the thread on disk; the recap hides most of the damage. A restart during a later turn resumes
    correctly. Suggested default: record the thread id as soon as `thread/start` (Codex) or the SDK init (Claude)
    returns it, keep the completion-time update for model and pid.

14. **A busy Claude Code worker outlives a hard daemon kill and races the resumed scan (E3, Windows).** The
    in-flight import-scan `claude.exe` (ppid = dead daemon) finished its turn ≈55 s later, writing
    `PROJECT_STATE.md`, `INDEX.md`, `project_goals.md` while the restarted daemon's resumed scan wrote the same
    files (last writer won). Idle workers and Codex exit on stdin EOF; a Claude turn in flight does not. Related
    to the known "stop_all leaves claude.exe alive" note, but this is the hard-kill path. Suggested default: put
    spawned workers in a Windows Job Object with kill-on-close, or have the startup resume skip a scan whose
    previous process is still alive.

15. **Import-scan summaries sometimes come back in English with `import_scan_lang: zh`** (claude-code ws-w1/w2,
    codex ws-c1; the card question and options are zh). Model output under the LANGUAGE section; the Mac saw
    zh. Suggested default: leave; revisit if users report it.

19. **Idle worker processes are never reaped while the daemon runs** (dsh pass). Every session that ever ran
    dsh keeps its `cmd.exe → DeepSeek Harness.exe` alive; 10 at once on rig D, the oldest idle 25 minutes.
    `_reap_idle_sub_agents` (`agent_manager.py:~6884`) runs only when the owning session goes away; the
    same holds for Claude Code and Codex (the Mac saw ~20 at peak). Resume makes stopping them lossless.
    Suggested default: an idle timeout per worker adapter (say 10 minutes), stop → resume on the next send.

20. **dsh's Windows sandbox repairs ACLs on the user's folder** (external, dsh 0.2.2-rc.1). Its first shell call
    on a fresh folder fails (`SetNamedSecurityInfoW failed (Win32 5): grantWrite(<ws>)`); it then loads its
    bundled `diagnose-windows-sandbox-acl` skill and runs the repair with `danger-full-access`, leaving
    `acl-recovery\` folders (once outside the workspace) and new ACEs (`Everyone:(CI)(DENY)(S,DC)`, a
    capability SID with W,D,DC, `Low Mandatory Level`). Not Orbital's code, but an Orbital user on Windows
    sees it. Suggested default: raise with the dsh team; consider a one-line warning in Orbital's dsh card.

21. **One `POST /mcp` 400 per worker spawn** (every dsh and Claude Code spawn, before the 200 initialize): the
    MCP SDK's stateless streamable-HTTP transport answering the client's first probe. Dispatch works.
    CLAUDE.md asks for no 400s in daemon logs. Suggested default: document it; or answer the probe with a
    200 in our ASGI route if the SDK exposes a hook.

22. **Job object and `claude.exe` on the test rigs (item 14 follow-up).** Rig D's log shows `windows job: pid …
    not assigned (error 5)` for Claude Code spawns only (dsh and Codex assigned fine). The daemon on that
    rig was launched from inside a Claude Code session, whose harness keeps its children in a job; a process
    that broke away from that job cannot be added to a job nested under it. From a plain console a native
    `claude.exe` assigns fine (verified). Expected to be clean under the desktop app; verify once on the
    CI-built installer (a `windows job:` warning in the packaged daemon's log would mean it is not).

23. **Mixed path separators** when a project's workspace was created through the API with forward slashes
    (`C:/orbital-scratch/D/ws-w3\orbital\sub_agents\…`). Cosmetic; the UI sends backslashes. Suggested
    default: `os.path.normpath` on create; left alone because tests compare the stored workspace verbatim.

## B. Needs something only you have

16. **167-W6 (upgrade / downgrade of a v0.21 data dir)** could not run: the only v0.21-era data dir on
    this machine is the real app's `%APPDATA%\Orbital` (serving port 8000, real projects), which I did
    not touch, and installing an older `.exe` needs admin. If you want it, point me at a copy of a
    v0.21 data dir with a half-way interview (or say it's fine to copy the real one to a scratch dir).

17. **ACP agent on Windows.** No ACP agent is installed here (`dsh` is a manual `dsh-acp-demo.cmd`
    stub, not ready; Kimi / MiniMax / Cursor / Pi absent), so every ×3 ACP row, D3, 168-5 and 166-5
    are BLOCKED. The OpenCode Go key you gave was used for the keyed cases (E1/E2, 168-6, 166-8)
    through an `opencode-go` / `deepseek-v4.1-flash` card; it was never written to disk outside the
    test daemons' in-memory keyring. If you want the ACP rows on Windows, say which agent to install.

18. **The runbook's history recipe.** `claude -p "Reply with just OK"` and the Codex equivalent are
    filtered out as probe noise (`history.py:417–439` `_PROBE_RE`), so folders seeded that way never
    reach the wizard; that is why the Mac had to rewrite the import payload. A real first message
    ("The code word for this folder is MAPLE. Remember it.") makes them importable through the real
    UI. Suggested default: change the recipe in `note-keyless-workbench.md`.

## C. Test-suite reds on Windows that are not this batch's

`pytest tests/unit tests/platform --timeout 120` on the tip: 34 failed / 7055 passed. 31 of the 34
fail identically on `origin/main` in a clean worktree (symlink privilege `WinError 1314`, Windows
path length in the pi/kimi session fixtures, `test_version` reading stale installed metadata 0.18.0,
`C:\` being a git work tree on this machine). The other three:

- `test_subagent_transcript_endpoint::test_400_for_path_traversal_shaped_handle` passes on main only
  because that worktree has no built `web/dist`: with the SPA present the unmatched URL is served as
  `index.html` (200, text/html). The route's handle regex still rejects traversal. Test-env artifact;
  a sharper assertion (not `application/json`) would make it deterministic.
- `test_shutdown_hardening::test_an_earlier_snapshot_reaches_a_process_a_stop_already_forgot` (new
  on 1109): the venv's `python.exe` on Windows is a launcher that spawns the real interpreter, so the
  snapshot sees two pids where the test expects one. Test-env artifact (use `sys._base_executable`).
- `test_worker_commands::test_publish_get_and_workspace_is_realpathed` (new on 166) needs a symlink.

I left these three alone; say if you want the two test-side fixes.
