# Test note: batch/1110 on Windows (specs 166, 167, 168 + the keyless onboarding pass)

**Branch under test:** `batch/1110`, at the commit that adds this file. It is the 2026-11-09 tip (`b260bfd`:
v0.21.0 + batch 1109 + spec 156 + the keyless smoke fixes) plus three merges:

| Spec | What changed for the user | Branch |
| --- | --- | --- |
| 168 | A pinned Claude Code reliably hands work to another agent through Orbital (it gets the same first-turn dispatch line Codex, ACP agents and Pi already got). It no longer runs `codex exec` from its shell. | `fix/168-claude-code-seat-note` |
| 167 | Work first: no interview, no agent-speaks-first greeting, no Scan card, no onboarding gate. Goals are drafted by the end-of-run memory turn of the first real run, then confirmed on the existing card 这样理解对吗？ 对 \| 需要改. The wizard's keyless import scans by default (a switch beside Skip turns it off). A failed import scan is final and stays visible with its cause. | `feat/167-work-first-goals` |
| 166 | With a worker pinned, the `/` picker also lists that worker's own commands (Claude Code commands, Codex skills as `$name`, ACP agents' advertised commands). A pick prefills the composer; Enter sends it as plain text. | `feat/166-worker-commands` |

Run this **together with `note-keyless-workbench.md`** (same branch). That note still applies, except the cases
listed under "Superseded" below, which 167 changed by design. Specs are in `BACKLOG/specs/166-…`, `167-…`, `168-…` on
the Mac (not committed); everything you need is in this file.

Write results to `note-1110-results-windows.md` on this branch, one line per case:
`167-W3-codex PASS — evidence` / `FAIL — what happened` / `BLOCKED — why`. If you fix something, commit the fix
separately with a regression test, and say which case it fixes.

## Safety

The safety rules and setup in `note-keyless-workbench.md` apply unchanged: scratch folders only
(`C:\orbital-scratch\`), an isolated daemon with a fresh data dir, no `AGENT_OS_API_KEY`, back up and mtime-check
`%USERPROFILE%\.orbital\sub_agent_config.json`, `~/orbital/user_memory.md` and `user_preferences.md`, tiny worker
turns, never run an interactive login. One addition: 166 writes `worker-commands.json` in the **test** data dir;
check nothing new appears under `%USERPROFILE%\.orbital\`.

## What the Mac already verified (so you know what is Windows-only)

- **Unit and platform suites** on the merged branch: 7,257 passed. The only reds are the 2 known pre-existing
  `test_settings_key_telemetry` failures (one hangs without `--timeout`; run pytest with `--timeout 120`).
- **Web:** `tsc -b` clean, vitest 2,763/2,763, `check-i18n` clean (7 pre-existing missing-zh warnings).
- **168:** 16/16 pinned Claude Code runs dispatched (5 with the wording that failed before, 4 other wordings
  incl. 「让 Codex 看一下」, 3 with a `codex exec` recipe seeded in its MEMORY.md, a daemon restart), 0 shell
  `codex exec`. Pinned Codex → Claude Code and pinned Pi → Claude Code still dispatch. A manager-dispatched Claude
  Code gets no line and no tool.
- **167:** all §8 cases on keyed and keyless rigs with real Claude Code, Codex and dsh. Goals drafted 7/7 (manager),
  6/6 each (Claude Code, Codex, dsh). After one fix, 0/5 drafts copied Orbital's own operating rules into the goals.
- **166:** Claude Code commands and Codex `$skills` listed, picked and executed; Kimi (15) and Cursor (56) advertised
  their commands and the lists survive a restart (running a command was blocked by those accounts' plans). Unpinned
  catalog byte-identical to before. 390 px zh: no overflow.
- **Integration (all three merged):** see the last section.

## Superseded cases in `note-keyless-workbench.md` (do not run as written)

- **W4, A2** ("Skip means skip", no scan after a wizard import): replaced by 167-W2 below.
- **A3** (interview on a new empty project): replaced by 167-W3.
- **A4** (first-use Scan card): replaced by 167-W4. The Scan card no longer exists.
- **A5** (sidebar import): replaced by 167-W5 (same flow plus failure visibility).
- **A7** (paths in the onboarding instruction): replaced by 167-W1 and 167-W7.
- **E1**, the part "the wizard's imported projects are not scanned now (`import_scan` is `off`)": with the switch
  on (default) they were already scanned; with it off they stay `off`. Everything else in E1 still applies.

Everything else (A1, A6, all of B, C, D, the rest of E) runs as written. D1/D2 overlap with 168 below; run them once.

## 168 — Claude Code seat dispatch (×1, Claude Code + Codex; Pi if set up)

1. Pin Claude Code with Codex enabled. In a new session: "fix the sign in bug, then have codex review it" (any tiny
   scratch project with an obvious bug works). Pass: one `codex sent:` row, Codex's bubbles in the same chat,
   Claude Code's follow-up after the hand-back. No "Other agents in this project…" text anywhere in the chat.
2. Same with 「修好登录的 bug，然后让 Codex 看一下」.
3. Restart Orbital mid-session and ask again in the same session. Still dispatches.
4. Pin Codex and ask it to have Claude Code review something. Still dispatches; no `claude -p` in Codex's shell.
5. Pin Pi (if Pi is set up) and ask it to have Claude Code review. Still dispatches.
6. No pin (keyed only): ask the manager to have Claude Code fix something. Claude Code works normally and nothing
   about dispatch shows.
7. After 1–3, `%USERPROFILE%\.codex\sessions` has no new `codex exec` sessions (only Orbital-started ones).

## 167 — Work first (keyless and keyed; ×3 = Claude Code, Codex, one ACP agent)

- **167-W1 Paths.** Keyless import scan on a `D:\` (or non-`C:`) folder: the first dispatch row and the worker prompt
  show `D:/…/orbital/instructions/project_goals.md` (forward slashes, drive letter kept). After a first task on a
  project with no goals, the worker writes `<ws>\orbital\instructions\project_goals.md`, not a cwd guess.
- **167-W2 Wizard keyless import.** Fresh install, no key, import 2 scratch folders. On the key step the coin sign
  with "2" and the worker name sits beside Skip; hover/tap shows the sentence and the switch.
  Switch on, Skip: 项目梳理 sessions run on the worker, at most 2 at once, each ending with the zh card.
  Switch off, Skip: no scan; the first message in a project brings the card.
- **167-W3 New empty project ×3** (sticky pin set). No greeting and no session until you send. Send one small task:
  the worker does it, the hidden memory turn writes `project_goals.md` ("First draft"), and the card
  这样理解对吗？ 对 | 需要改 appears in the session and on the Workbench. Tap 对: the reply is recorded and **no**
  run starts (log: no spawn). Open the goals file: it describes the user's project, never Orbital's rules
  (no `orbital/bin`, background jobs, MCP or dispatch).
- **167-W4 New project on a folder with files ×3** (not imported). No Scan/Skip card. The first task works on the
  files; the memory turn drafts goals from folder + conversation. Tap 需要改 with a correction: exactly one turn,
  and it changes only files under `orbital\`.
- **167-W5 Sidebar import ×3.** As the old A5 (sign names the worker; 项目梳理 session; ≤ 2 at once; zh card;
  对 starts no run; 需要改 is one turn inside `orbital\`; 对，做第 1 步 is one turn). New: a scan on a signed-out
  worker stays as a 项目梳理 session with a sign-in cause on the Workbench card; it is **not** re-run after a
  settings save or an app restart; the project's first real message drafts the goals.
- **167-W6 Upgrade.** A v0.21 data dir with an interview half-way (if you have one from an earlier Windows pass):
  opens, continues as a normal chat, and its next run drafts goals. Also open this branch's data with the v0.21
  installer once (downgrade) and confirm it loads. Neither was verified on the Mac.
- **167-W7 From the old A7.** Codex can list the workspace during the scan; two worker starts in a row do not stall
  on permissions.

## 166 — Worker commands in the `/` picker

1. Claude Code pinned, fresh daemon: `/` lists Claude's own commands (put a scratch
   `.claude\commands\orbital-probe.md` in the workspace; it shows as "(project)" with its argument hint). The cold
   probe spawns claude via the native CLI resolution: check an npm `claude.cmd` install still probes, through
   `claude.exe`.
2. Codex pinned: `/` lists Codex skills as `$name`. The cold probe spawns `codex app-server` with cwd = the workspace:
   check an npm `codex.cmd` install.
3. Pick a Claude command: composer shows `/name ` with the caret at the end, no chip. Enter: the worker runs it.
4. Pick a Codex skill: composer shows `$name `. Enter: Codex uses the skill (its transcript shows the skill read).
5. Kimi or Cursor pinned (if either has a working plan): `/` lists their commands after the first message, and still
   lists them after an app restart. Pick one and send it; record what the agent does (never verified anywhere).
6. `worker-commands.json` is written in the test data dir. Its workspace keys are realpaths: check that a workspace
   path with different drive-letter case (`d:\…` vs `D:\…`) still matches.
7. No console window flashes during the probes.
8. Unpinned: the `/` list is exactly what it was before (Orbital's catalog only).

## Known, not fixed (do not file as new; note if Windows behaves differently)

- On the first message to a newly started worker, Orbital's recap block comes before the user's text, and the recap
  repeats the message. `/name` and `$name` still worked for Claude Code and Codex; Claude built-ins (`/compact`) and
  ACP agents in that position are untested. Candidate follow-up spec.
- Two sessions of the same worker in one project share one transcript file, so "this run wrote memory" can count the
  other session's rows (once skipped a memory turn on the Mac).
- A manager import scan that dies on an LLM error gets no Workbench card (its row has no `_meta.event`).
- The empty-chat "Didn't finish" cards never list a scan session (the Workbench card does).

## Mac integration check (all three merged)

Isolated daemon on the merged commit (`ce7c16d`), scratch HOME and workspaces, real Claude Code and Codex,
2026-11-10 01:19–01:25.

- **166:** `skill-catalog?worker=claude-code` → 81 entries with `/name ` invokes, including the scratch project command
  as `/orbital-probe `; `?worker=codex` → 25 entries as `$name `; unpinned → 0 worker entries. Sending
  `/orbital-probe hello` through the pin → `PONG-INT hello`. PASS.
- **167 × 168** (the case no single builder could run): empty project, Claude Code pinned, Codex enabled, one line
  "write a four-line haiku about autumn into haiku.md, then have codex review it and apply its notes", ×2.
  Both runs: Claude Code dispatched to Codex through Orbital (`Message sent to codex` row, Codex's review, hand-back,
  notes applied); no shell `codex exec`; the seat line never appears in the chat; the memory turn drafted
  `project_goals.md` and the confirm card opened. Neither goals file contains Orbital's operating rules (one records
  the user's own write → codex review → apply loop, which is the user's workflow). PASS.
- Not checked in the integration run: a browser pass (the builders did theirs on their branches), keyed manager
  paths (167's builder did), ACP agents.
