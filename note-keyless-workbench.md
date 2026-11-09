# Test note — keyless onboarding × Workbench

**Branch under test:** `batch/1109`, at the commit that adds this file (built on the
2026-11-09 installer `0d14735` plus one memory fix). Everything below was written from the
code on this branch; the Mac pass records its results in `note-keyless-results-mac.md`
beside this file. Windows: write yours to `note-keyless-results-windows.md` on this branch,
pass/fail per item with evidence (status codes, JSONL rows, log lines, screenshots named
by case). If you fix something, commit the fix separately with a regression test.

Test from source. Agents in scope: **Claude Code** (SDK transport), **Codex**
(app-server transport), and **one ACP agent** that has working credit on the machine
(Kimi Code, MiniMax Code, Cursor or dsh). Cases marked ×3 run once per agent.

## What keyless means (so you know what "right" looks like)

The wizard's Orbital API key step has a Skip. After Skip, Orbital's own agent (the
"manager") cannot run anywhere, so:

- Every new session starts pinned to the project's sticky pin, or, with none, to the
  first available worker. The chat's empty state says who it is talking to.
- The pin switcher keeps the Orbital row with an "API key required" pill; clicking it
  opens Global Settings at the LLM cards.
- Onboarding of a new project (the interview) and the first-use Scan card run on the
  pinned worker, which writes `orbital/instructions/project_goals.md`,
  `orbital/PROJECT_STATE.md` and `orbital/INDEX.md` with its own file tools.
- The Workbench has no greeting. Its project chip never shows 自动 (routing needs a
  model): it starts on Quick Tasks, then on the last project sent to. The picker shows
  自动 greyed with 需要 API Key; clicking it opens the LLM settings. The agent chip's
  自动 means the project's pin.
- The wizard's Skip means skip: projects imported in the wizard get `orbital/` on disk but
  no background 项目梳理 scan. A sidebar Import after the wizard does scan, on the
  pinned worker, two at a time, and the daemon posts a confirm card
  (这样理解对吗？ · 对 | 需要改, or with next steps: 对，做第 1 步 | 对，先不做 | 需要改).
- The pinned worker at the top of a session has one `dispatch` tool (the `orbital` MCP
  server over HTTP) to hand work to another agent; the result comes back as its next
  turn. Orbital never wakes for it.

## Watch these four closely (the code suggests they are weak)

- **W1 Agent chip "Orbital" while keyless.** The project picker greys 自动 but the agent
  picker still lists Orbital as selectable. A send with it injects to a manager with no
  key. Record exactly what the user sees (error with text back? dead session + failed
  card?).
- **W2 ACP seat dispatch has never been seen working.** The MCP server mounts and
  `tools/list` succeeds on all five ACP agents, but no ACP model has ever called
  `dispatch`.
- **W3 Codex gets no Orbital system prompt**, only a first-turn note. The ASKS.md grammar
  and the run-end memory contract may never reach it. Does a Codex worker ever produce an
  ask card or write PROJECT_STATE?
- **W4 Wizard Skip means no scans.** Confirm nothing scans by accident after a wizard
  import, and that the first-use Scan card is there instead.

## Safety rules (read before touching anything)

- **Never import a real project folder into a test daemon.** Use scratch folders only
  (`C:\orbital-scratch\` on Windows, `~/orbital-wt/run/<id>/` on the Mac). Deleting a
  project deletes its `orbital\` folder; opening a session writes `AGENTS.md` +
  `orbital\` into the folder. On the wizard's Projects step **uncheck every real folder**.
- Run the daemon isolated with a **fresh data dir** so the wizard shows, and **do not set
  `AGENT_OS_API_KEY`** (that would make the daemon keyed):
  - Windows: `python scripts/dev_daemon_isolated.py --port 8338 --data-dir C:\orbital-scratch\data`
    then `cd web && set ORBITAL_DEV_API_PORT=8338 && npx vite --port 5173`.
  - Mac: the `dev-daemon.py` launcher with the memkeyring env, a spare port and a scratch
    data dir (see the Mac results file for the exact command).
- **`%USERPROFILE%\.orbital\sub_agent_config.json` (`~/.orbital/` on the Mac) and
  `~/orbital/user_memory.md` are shared with the real app.** Back them up first, never
  type into About You / User Memory, never use the locate card or a per-agent model
  change on the test daemon, and check their mtimes at the end.
- Keep worker turns tiny ("Reply with the single word PONG") — they bill the real
  Claude / ChatGPT plans. Never run `claude auth login`, `codex login` or any
  interactive login.
- For case 29 only, a real Orbital key is needed. Add it through the normal UI card form;
  never paste it into a results file.

## Setup

1. Back up `sub_agent_config.json`, `user_memory.md`, `user_preferences.md`; note mtimes.
2. Create three scratch workspaces and give each native history so the wizard can
   import them: `git init` each, then inside run
   `claude -p "Reply with just OK"` and `codex exec --skip-git-repo-check "Reply with just OK"`
   (one of each per folder; the ACP agent's history is optional).
3. Start the isolated daemon on a fresh data dir, then Vite against it. Confirm the
   listening PID on the port is the daemon you just started.
4. Open the UI, go through the wizard: Agents → Projects (uncheck every real folder, keep
   the three scratch ones) → key step **Skip** → finish.

Record the agents the daemon reports available (`GET /api/v2/agents/available`).

---

## A. Entry and first project

**A1 Skip persists.** Relaunch the UI (and restart the daemon once). Pass: no wizard; Global
Settings shows the keyless wording with the key form; `GET /api/v2/settings` has
`setup.api_key_skipped: true` and `llm.api_key_set: false`; in a chat the pin switcher's
Orbital row shows "API key required" and clicking it opens Global Settings at the LLM cards.

**A2 Wizard import keyless (W4).** Pass: each imported scratch folder has `orbital/` on disk
right after the wizard; no session titled 项目梳理 / Project overview exists in any of
them; each project's empty chat shows the first-use Scan card; the daemon log has no
"import scan" start line.

**A3 New empty project ×3.** Create a project on an empty scratch folder with the pin
set to the agent under test (set the project's sticky pin first, or disable the others).
Pass: start-onboarding returns 201 with `runner: "worker"`; the session is pinned to that
agent; the empty chat shows the keyless hint naming it; the worker's first reply is the
interview questions and it **stops** (no files written yet, no shell). Answer in two lines.
Pass: it writes `project_goals.md` at the absolute path it was given (not under its cwd
guess), then ends with a proposed first step and waits. Your next message resumes the
same native thread (one worker process per session in the log, `resume` present).

**A4 First-use Scan card ×3.** On an imported scratch project, click Scan. Pass: the scan
runs on the pinned worker (log: worker spawn, no manager loop); `project_goals.md`,
`PROJECT_STATE.md`, `INDEX.md` written; the reply ends with a question or proposal, and
the worker does **not** start editing project files on its own. Note: this path has no
daemon confirm card; that is expected.

**A5 Sidebar Import keyless ×3.** After the wizard, import another scratch folder (with
history) from the sidebar's Import entry. Pass: the token sign beside Import names the
worker that will read it; after Import a session titled 项目梳理 / Project overview
appears, pinned to that worker, running; with several imports at most two run at once;
when it finishes the Workbench shows the card unread with the daemon's confirm card
(这样理解对吗？ with options). Then:
- tap 对 (or 对，先不做): the ask closes, the user row is recorded, **no** worker run
  starts (log shows no spawn, JSONL has no new assistant row);
- on another import tap 需要改 and type a correction: exactly one worker turn, and it
  edits files under `orbital/` only;
- on an import with next steps tap 对，做第 1 步: one worker turn that does step 1.

**A6 No worker available.** Disable every worker in the project's settings (or sign all
out). Pass: creating a new empty project returns the missing-key 400 (code
`missing_api_key`); the UI shows a readable message, not a spinner; no orphan session is
left in the project's session list.

**A7 Windows only.** In A3/A4 the paths in the worker's instruction use forward slashes
with the drive letter kept (check the first user row of the JSONL); Codex can list the
workspace folder; starting a worker twice in a row does not stall on permissions.

## B. Workbench composer keyless

**B1 Home.** Pass: no greeting line; the composer sits around the golden section; a fresh
browser shows the project chip on Quick Tasks; after a send to project X, a reload shows
the chip on X.

**B2 Project picker.** Pass: 自动 is visible, greyed, with 需要 API Key; clicking it opens
Global Settings at the LLM cards and sends nothing; the other rows pick normally.

**B3 Agent 自动 ×3.** Chip on a project whose sticky pin is the agent under test; type
"Reply with the single word PONG"; send. Pass: a pending card appears at once; a new
session is minted with `pinned_target` = that agent (`GET .../sessions`); the card shows
the agent's logo (never Orbital's); it turns running then unread with PONG as its result
line; the summary line counts it.

**B4 Explicit agent pick.** On a project whose sticky pin is Claude Code, pick Codex in the
agent chip and send PONG. Pass: that session's `pinned_target` is codex; the project's
`default_pinned_target` is unchanged (`GET /api/v2/projects/{pid}`); the next 自动 send
goes to Claude Code.

**B5 Agent "Orbital" (W1).** Pick Orbital in the agent chip and send. Record: is the row
selectable at all; what the composer shows; whether a session was minted and what its
JSONL holds; whether a card stays on the Workbench and what it says. Expected good
behaviour: the text comes back with a clear "needs API key" line, no dead session.

**B6 Attachments ×3.** Stage a small text file in the composer and send "Reply with the
first line of the attached file". Pass: the upload happens after the project is known
(network tab: route is **not** called keyless, upload then new-session then inject); the
worker reads the file and answers with its first line. Desktop app: attach by path
(drag a 20 MB file) and the worker reads it without an upload.

**B7 Quick Tasks ×3.** Chip on Quick Tasks; send PONG. Pass: the worker runs with the Quick
Tasks folder as cwd (the worker's JSONL or log shows it); the reply arrives; nothing in the
log mentions a manager or a missing key.

**B8 `/automate` keyless.** Type `/automate every day at 9 say hi` with the pin on a
worker. Pass: the text goes to the pinned worker as plain text (no manager run, no 400);
no trigger is created (`GET /api/v2/projects/{pid}/triggers` unchanged); record what the
worker answers. Product question to note: should the skill be hidden when keyless?

**B9 External task endpoint.** `POST /api/v2/workbench/tasks` with `{"text":"PONG"}`
only. Pass: 200 with `by: "fallback"`, `reason: "no API key"`, `project_id` = Quick Tasks,
`agent` = the pin. With `agent: "<a disabled worker>"`: 400. With `agent: "orbital"`:
record the outcome (same gap as B5).

## C. Cards, asks and follow-ups

**C1 Worker ask ×3 (W3 for Codex).** In a worker session say: "Before you continue you need
my decision: red or blue? Record it the way Orbital asked you to and stop." Pass: the
worker appends `- open … options: 红 | 蓝` (or similar) to `orbital/ASKS.md`; the Workbench
card shows the question with option pills; tapping one closes the ask (`- done <id>
"<words>"` line appears or the ask is marked answered) and sends the words into the same
session; the chat shows the reply under the ask. For Codex, record whether it ever wrote
ASKS.md at all.

**C2 Reply from the card.** Type into the card's reply field and press Enter. Pass: the text
becomes the next turn of that pinned session (same worker, resumed); the dot updates when
it answers; ✓ clears the unread state; opening the session also clears it.

**C3 Supervised Claude Code approval.** Pin Claude Code in supervised mode; ask it to create
a file. Pass: the card shows the pending approval; the card's reply field is hidden;
clicking the card opens the session; approving continues the run and the card updates.
Codex: ask for the same and confirm the behaviour matches Codex's own sandbox setting
(Orbital does not override it).

**C4 Stop causes ×3.** Force a stop (sign the agent out in another terminal, or exhaust a
limit if one is near, or kill the worker process). Pass: the card shows the stop cause
(signed out / limit / interrupted), the project's empty state shows a left-off card for
it, re-pinning to another worker and sending continues with a recap of the conversation.

**C5 Memory record turn ×3.** After a worker run ends, wait 30 s. Pass: the daemon sent the
"[Orbital memory]" message (JSONL row with that marker); PROJECT_STATE.md gained an entry;
the Workbench card did **not** become unread again because of it, and no "nothing to
record" bubble shows in the chat.

**C6 Parallel.** Send PONG to two different projects within 5 s, then two sends into one
project 5 s apart. Pass: all four run; the two in one project do not share a session; the
cards stay distinct and each clears on its own ✓; the log shows no lock error (a
`lock_wait` line is fine).

**C7 Imported native sessions.** In an imported scratch project, open a virtual Claude Code
row (shown with the agent mark, no chip), then send "What code word did I tell you?" to it.
Pass: the row becomes a real session, the native thread is resumed (the agent knows the
earlier content), and it appears on the Workbench as a card with the right logo. Codex:
same. If the machine has a Codex automation, let it fire: the card shows Codex and no
manager ran.

## D. Dispatch from the seat keyless

**D1 Claude Code seat → Codex.** Pin Claude Code; send "Ask Codex to reply with the single
word PONG and tell me what it said." Pass: Claude Code calls `mcp__orbital__dispatch` (log
line / JSONL tool row), Codex runs as a child (depth 1), the result arrives as Claude
Code's next turn and it reports PONG; no manager run and no missing-key error anywhere; the
Workbench shows one card for the parent, whose status reflects the child while it runs.

**D2 Codex seat → Claude Code.** Same with the roles swapped. Pass as D1, and Codex must not
run `claude -p` in its shell (grep the JSONL for a shell command containing `claude`).

**D3 ACP seat (W2).** Pin the ACP agent; same request. Record: did the agent list the tool
(daemon log `POST /mcp` initialize + tools/list), did it call it, did the result come back.
Any outcome is useful; this has never been observed.

**D4 Dispatch to Orbital.** From a worker seat: "Hand this to Orbital: say hi." Pass: the
tool returns an error string the worker relays; nothing hangs; no manager run.

**D5 Child has no tool; stop the parent.** While D1's child runs, press Stop on the parent.
Pass: the child keeps running to its end (log), the child's own session never shows a
dispatch tool, and record where its result lands (expected: a `worker_result` row in the
parent's session, no wake).

## E. Transitions

**E1 Add a key later.** Click the greyed 自动 row → add a real card in Global Settings.
Pass: the Workbench shows the greeting, 自动 is selectable, the Orbital row works in the pin
switcher; an existing project with a sticky worker pin still starts new sessions on that
worker until the user re-pins; the wizard's imported projects are **not** scanned now
(`import_scan` is `off`); a new sidebar import is scanned by Orbital.

**E2 Remove the key.** Delete the default card. Pass: back to keyless everywhere (no
greeting, chip on the last project, Orbital row with the pill); an in-flight manager
session ends with a readable provider error, not a hang.

**E3 Daemon restart mid-run.** Start a worker run and restart the daemon. Pass: the session
reopens with its rows intact; a new send resumes the native thread; an import scan that
was pending or running resumes (log "import scan" resume line).

**E4 Phone via relay.** Open the Workbench on a phone. Pass: the card reply row is always
visible; no greeting; the project chip default is right; a send works end to end.

---

## Results format

One line per case: `A3-claude PASS — evidence` / `FAIL — what happened` /
`BLOCKED — why`. Attach the JSONL excerpt or log lines that prove it. End with the
mtimes of the three shared HOME files before and after, and the list of worktree paths
found (there should be none) in `sub_agent_config.json`.
