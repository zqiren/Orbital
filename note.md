# Windows smoke note — v0.25.0 release candidate (2026-11-25)

Branch-only file: it is dropped before this branch merges to main. It replaces the 2026-11-24
note for `dd1fc782`; that night's results stay in `note-windows-results.md` and are referred to
below as "the 1124 results".

**Your job (Windows Claude session):** smoke the checks below on the CI installer, fix what
fails, and write the results. The Mac session has stopped here; nothing below has been run on
Windows unless it says so.

## Build under test

- Branch `test/1125-all-fixes`, commit `9c55fa86` (41 commits on top of published v0.24.1 `232212a`).
- CI run `45567085778`: every job green (frontend, backend unit tests on Windows and macOS, both
  installers, fresh-install and upgrade scenarios on both systems).
- Installer: the `Orbital-Windows-installer` artifact of that run, `Orbital-Setup-0.25.0.exe`,
  sha256 `472cd61cfbbb8e048428a8be1754cfa703bb0549a781d6805daf757b5a20a43a`.
  `gh run download 45567085778 -n Orbital-Windows-installer`
- Test this CI-built `.exe` only, never a local build. Settings → About must read 0.25.0.

## Rules

- **Protect the machine's data.** Before any fresh-data part, quit Orbital and rename
  `%APPDATA%\Orbital` to `Orbital.bak-1125`; restore it before part B. Use scratch folders
  (`C:\orbital-scratch\1125\…`) for test projects; never send test messages into real projects.
- Read "Testing notes for the next Windows session" at the bottom of `note-windows-results.md`
  first (the Claude desktop app's private `%APPDATA%` copy, the UAC prompt).
- **Fixing.** Commit fixes on `test/1125-all-fixes`, each with a test that was red first. Follow
  `CLAUDE.md`: unit tests, `npx tsc -b`, vitest for touched frontend code, a restarted daemon
  before retesting. After fixes, push, run `gh workflow run ci.yml --ref test/1125-all-fixes`,
  and re-check the fixed items on the new CI installer (record its sha256). Two failed attempts
  at one bug: stop and write down what was tried.
- **Do not** tag, publish, merge to main or touch the release aliases. The owner releases.
- A failure that is a product decision, not a bug (wording, behaviour that is debatable), goes
  in the results as a question for the owner, unfixed.
- Results go in a new file `note-windows-results-1125.md` beside this one, committed and pushed:
  one row per check with PASS / FAIL / NOT RUN, what you saw, the `daemon.log` lines for any
  failure, every fix with its commit, and a plain list of what was not run and why. First line:
  `STATUS: READY` or `STATUS: NOT READY — <what blocks the release>`.

**Windows paths:** app data `%APPDATA%\Orbital\` · daemon log `%APPDATA%\Orbital\logs\daemon.log` ·
install `C:\Orbital\bin\` · Claude Code history `%USERPROFILE%\.claude\projects\` · Codex `%USERPROFILE%\.codex\`

## Already proven on Windows (do not repeat)

- By CI on this exact build: the installer builds; fresh install (Welcome → untick import →
  Start with no key → Workbench → Codex reply → add a provider card); upgrade over v0.24.1 with
  a chat turn; the backend unit suite.
- By the 1124 results, on a build this one contains: Welcome page agent detection, the projects
  line, key test with a real and a wrong key, folding, Start with an import, keyless routing by
  Claude Code, Recent, quit mid-import, upgrade keeps data, downgrade to v0.24.1 keeps data,
  Antigravity not-installed and installed-signed-out rows.

Everything since then is new: the dispatch cards, undo additions, usage-limit state, design
look D, the composer, the status fold, honest agent status, the import and installer fixes, the
Codex thread names and the pin-menu model fix.

---

## A. Release blockers (run these first)

**A1 — Upgrade over v0.24.1 cleans the manifests folder.** With the real data restored and
published v0.24.1 installed, install the 0.25.0 `.exe` over it.
Pass: `C:\Orbital\bin\_internal\agent_os\agents\manifests\` holds no `gemini-cli.yaml` (it is
emptied before the copy); `antigravity.yaml` is there; the app opens with every project, session,
card and setting intact; no Welcome page; no "Gemini CLI" anywhere in Settings → Sub-agents.
Then install v0.24.1 back over it: it opens the same data and does NOT list Antigravity (the
1124 results' finding 1). Finish on 0.25.0 again.
Also record: an aborted install (cancel at the UAC prompt) leaves the old app working.

**A2 — Look D in the real window.** Design look D (tinted sidebar and rail, the reading area on
one inset rounded sheet, 8px radius) was only ever seen in Chromium and WebKit on a Mac.
Check in the installed app, light and dark, 中文 and English, at 100 %, 125 % and 150 % display
scaling: (a) the native title bar against the tinted chrome — its colour was not built, so
report how it looks, with a screenshot, rather than fixing it; (b) the sheet's thin ring is
visible and even at each scaling; (c) no banding in the chrome's gradient; (d) the sheet's bottom
corners against the window's corners; (e) Microsoft YaHei text does not overflow buttons, the
sidebar rows, the 新建会话 button or the collapsed rail; (f) the collapsed right panel's handle
sits inside the sheet. Screenshots of each go in `evidence\`.

**A3 — Type scale on Windows fonts.** Same screens as A2 plus Settings (model providers,
Sub-agents, About), the Workbench and one long chat. Pass: no clipped or overlapping text, no
line of Chinese cut at the top or bottom, with ClearType on.

**A4 — A plain manager turn and a worker turn.** In a scratch project with a real key card: one
manager turn that reads and writes a file, then pin Codex and Claude Code in turn and get a
reply from each. Pass: replies arrive, no console window flashes, no 400/500 in `daemon.log`.

**A5 — Status fold.** Use a model that writes `[STATUS: …]` lines (the manager on a multi-step
task: "list the files here, read two of them, then write a summary to summary.md"). Pass: the
text `[STATUS` never appears in the chat, live or after a reload; a run of status-only steps
shows as one line that changes while it runs and opens to its steps; the final reply's bubble
and its copy button carry no tag. Never run with a real model on any system: report anything odd.

## B. New features

**B1 — Composer.** One slot: Stop while a turn runs and the box is empty, Send once there is
text; Esc stops the turn. Type three messages while a turn runs. Pass: they show as rows inside
the composer with edit, delete and send now; editing one changes what is sent; delete removes it;
they go out in order when the turn ends. Type Chinese with the Microsoft Pinyin IME, in the
composer and while editing a queued row: Enter that confirms a candidate must not send.
Known and accepted, do not file: Stop sends the waiting messages at once as the next turn; send
now on a row sends all waiting rows with that one first.

**B2 — Dispatch cards (spec 195).** Ask the manager for something it hands to a worker ("ask
Codex to list the files and report back"). Pass: the dispatch shows as a card in the chat; the
docked panel shows the worker's steps; when the worker asks a question it reaches you as a
question card; while you are pinned to another agent, Orbital's manager does not talk over it.
Record how a finished card looks (no "改了 N 个文件" line is expected).

**B3 — Undo (spec 196).** In a scratch project: have an agent create a text file, a PNG and a
PDF (under 20 MB) and edit an existing file across two turns. Pass: undo of the latest turn
restores the edit and removes the created files; "undo since a message" on the earlier message
undoes both turns; the agent's next turn knows the files were restored. Backslash paths and a
folder name with Chinese characters and a space must work.
Timing specific to Windows: file events arrive about 1 s late here. A file deleted and
re-created within about 3 s is treated as an overwrite and left on disk by undo — known and
accepted. A delete and re-create more than 5 s apart must count as a creation: check it.

**B4 — Usage-limit state (spec 197).** Needs an account at its limit; if none is, mark NOT RUN.
With a limited Claude Code or Codex: the agent reads as limited with its reset time in this
PC's local time zone (compare with the clock), and the one-click switch moves the chat to
another agent. Never seen with a real limited account on any system.

**B5 — Honest agent status.** (a) Sign Codex out for real (`codex logout`; an empty
`USERPROFILE` does not sign it out). Pass: Welcome (fresh data), Settings → Sub-agents and the
pin menu each say it is signed out and offer Sign in; a keyless send with no ready agent names
the agent and the reason, never "no agent is enabled". Sign back in from the row: it turns
ready without a relaunch. (b) Not-detected card for an agent that is not installed: it leads
with the install command, and for Antigravity that is the PowerShell `irm … | iex` line.
(c) After "Check again" on the Welcome page, no empty gap is left under the agent chips (seen on
WebView2 in the 1124 results, never reproduced on a Mac). (d) Status sweep time with all agents
installed: record it, and that no console window opens during a sweep.

**B6 — Pin-menu model switcher on a fresh install.** Fresh data, finish Welcome with the import
unticked. In Quick Tasks, before any session exists, open the pin menu. Pass: each worker's
model line has a chevron and opens its list (or its free-text field); pick a non-default model
for Codex, send, and confirm the turn ran on it (the session's `sub_agent_thread` row); the pick
survives an app restart.

**B7 — Import no longer lists Orbital's own threads.** Fresh data on this PC, where earlier
Orbital runs left Codex and Claude Code threads. Pass: the import list and Recent show no
session titled "Other agents in this project: …" or "Skills you can use (from Orbital): …"
(the 1124 results' finding 2). Record the time from Start to the list being complete on first
launch; the first scan after a daemon start now reads every session file once.
Also record whether old pending asks still flood the Workbench after an import (finding 3 —
not fixed in this build; count them).

**B8 — Codex thread names.** Pinned send to Codex in a new session. Pass: `daemon.log` has
"named thread …" after the turn starts, and the ChatGPT desktop app's sidebar shows the Orbital
session's title for that thread, not the preamble. With the older npm Codex forced (set its path
in the Codex card), the turn still works if naming is refused.

**B9 — Codex newest binary and background agents.** (a) With npm Codex and the ChatGPT app
both installed, the pin menu lists the models the desktop copy lists (gpt-6.2-*), and a path set
by hand in the card still wins. (b) Ask Claude Code to "run a background agent that waits 60 s,
and reply now". Pass: its sub-agent text never appears in the chat, the Background chip shows,
and a second message you send is answered at once instead of waiting.

**B10 — Sign in with ChatGPT.** Needs the owner at the browser; if he is not there, NOT RUN.
Settings → model providers → ChatGPT tile. Pass: the default browser opens on OpenAI's consent
page; after consent a card appears with the account email and a model; a manager turn runs on
it; deleting the card signs out. The whole click-through has never completed on any system, and
whether the packaged app can verify the token (bundled `jwt` / `cryptography` / `certifi`) is
unknown: report exactly where it stops, with the log lines.

## C. Antigravity

Run `note-antigravity-windows.md` steps 2–6 on this installer if the Google sign-in works and
the network exit is eligible (US, Japan, Singapore). The 1124 session could not sign in.
Additions for this build:
- On an ineligible exit: after the first refused turn, Antigravity must read "Not available in
  your region" on Welcome, in Settings and in the pin menu, and Start must not count it as a
  ready agent; "Check again" forgets it until the next turn.
- One `agy.exe` console window appeared during the first sweep after install in the 1124
  results. See whether it recurs and, if it does, whether Orbital or agy opens it.
- The login job tail showed "…" as `â€¦` (UTF-8 read as cp1252): fix if it is still there.
- If C passes fully, set `STATUS: READY` in `note-antigravity-windows.md`. If it cannot be run,
  leave it PENDING and say so; the owner decides whether Antigravity ships as it is.

## D. If time allows

- `docs/verification/in-app-upgrade-windows.md`: the one-click upgrade from v0.24.1 to this
  build needs a published release or a feed you control; only run it if that is easy to set up.
- A phone on the relay is not part of this note.

## Known and accepted (do not file)

- Version 0.25.0 is not published yet: the update pill may offer nothing or v0.24.1.
- The SmartScreen warning on the unsigned installer.
- The Background chip reads Idle for 3–4 s between a job ending and the
  follow-up turn.
- A message held behind an outside agent is lost if the daemon crashes before dispatch.
