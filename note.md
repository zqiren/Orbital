# Windows test note — one-page Welcome + Workbench landing (2026-11-24)

Branch-only file: it is dropped before this branch merges to main.

**Build under test:** `feat/1123-one-page-welcome` @ `cf0e0548`, CI run 45201197729.
Installer: the `Orbital-Windows-installer` artifact of that run, `Orbital-Setup-0.24.1.exe`,
sha256 `f02bbb08c791ab922403d490d3c2020a2ac0e84c3a2df977f004c1e2d4cbc6d9`.
Test this CI-built `.exe` only, never a local build. It still says version 0.24.1 (same as the
published release); identify it by the sha256.

**What this build changes**
- First run is ONE Welcome page (agents, "Import N projects" line, Orbital Agent connect panel
  with provider tiles + key field + ChatGPT sign-in) with one Start button. The four-page wizard,
  the Accounts page, the Create Project hand-off and the automatic tour are gone.
- Start lands on the Workbench at once; the import runs in the background.
- Workbench has a headline and a Recent section (latest sessions across projects, below the live cards).
- With no API key, a Workbench message on Auto is routed to a project by the first signed-in
  Claude Code or Codex (a one-off CLI call, 15 s budget, then Quick Tasks).
- A signed-out agent now says so and offers Sign in.
- Underneath: the new connect screen for model providers and Sign in with ChatGPT.

**Already proven by CI on Windows (do not repeat):** installer builds; the fresh-install scenario
(Welcome page → untick import → Start with no key → Workbench → Codex reply; add a provider card)
and the upgrade-over-previous-release scenario both pass. Backend unit tests: 8399 passed, 1 failed
(`test_history_service.py::TestWatcherAttribution::test_burst_of_20_writes_in_2s_is_one_version`,
a timing test in file history, passed on the previous run of the same code; being rerun).

**Windows paths**
- App data: `%APPDATA%\Orbital\`  ·  Daemon log: `%APPDATA%\Orbital\logs\daemon.log`
- Claude Code history: `%USERPROFILE%\.claude\projects\`  ·  Codex: `%USERPROFILE%\.codex\`

**Protect the machine's data.** Before part A, quit Orbital and rename `%APPDATA%\Orbital` to
`Orbital.bak-1124` (restore it before part B). In part A, do not send messages into imported real
projects unless the check says so; use a scratch folder.

For every check report: PASS / FAIL / NOT RUN, what you saw, and the matching `daemon.log` lines
for any failure. Say plainly what you did not run.

---

## A. Fresh install (no `%APPDATA%\Orbital`)

**A1 — Welcome page appears and detects agents.** Launch after install.
Pass: one page, no step strip. Every agent installed on this PC is listed; signed-in ones show a
check, signed-out ones a Sign in button. Codex is found even when it is the desktop app's
hash-versioned copy. No console window flashes while it detects. Record how long the agent list
takes to appear.

**A2 — Sign in from the page.** For one signed-out agent, click Sign in and finish in the browser.
Pass: the row turns to a check without relaunching; Start becomes enabled if it was disabled.
(This click has never been run on any OS.)

**A3 — Projects line.** Pass: "Import N projects" is pre-checked and N looks right for this PC's
history (drive-letter and backslash paths, OneDrive-redirected Desktop/Documents). "Review all"
opens the list, search works, unticking changes the Start label. Esc / Done restores the page.

**A4 — Orbital Agent panel.** (a) Paste a real provider key: the test runs by itself (no Test
button), a model row appears, the summary shows provider + model. (b) A wrong key shows a clear
failure and Start is not blocked when an agent is ready. (c) ChatGPT tile: complete the sign-in in
the browser. Pass: a card appears with the account email and a model; a manager turn runs on it.
(The full browser click-through has never completed anywhere: report exactly where it stops if it
does.) (d) TokenDance tile sign-in, if an account is available.

**A5 — Folding.** Click each band header: it folds to one line (agent logos / import line /
provider + model). Clicking into the key field folds Agents. Nothing folds by itself on load.
Check in 中文 and English, at a narrow window.

**A6 — Start.** With the import ticked. Pass: the Workbench shows within about a second; a small
spinner sits by the project count; projects appear in the sidebar as they import; Recent fills
without a reload; no New project dialog; no tour. Record the import time and project count.

**A7 — Keyless routing (no API key, Claude Code or Codex signed in).** Create two scratch projects
in empty folders with clearly different names (e.g. `invoices`, `thesis`). From the Workbench on
Auto send "add a line to the invoice notes". Pass: the project slot shimmers, then settles on
`invoices` in roughly 6–11 s; the agent replies; NO console window flashes for the routing call;
no new session appears in the agent's own history for it (count `%USERPROFILE%\.claude\projects`
folders and Codex threads before and after). Repeat 5 times with different tasks; report
hits/misses and times. If a call exceeds 15 s it must land in Quick Tasks, not hang.

**A8 — Signed-out error.** Sign the only enabled agent out (no API key), send a message in a
project. Pass: the notice names the agent and has a Sign in button; no "no agent is enabled" text.

**A9 — Recent.** Open an imported session from Recent. Pass: its transcript opens; sending a
message there moves it to the live cards; after you open the reply it returns to Recent.

**A10 — Quit mid-import.** Fresh data again, click Start with a large import, quit the app after
5 s, relaunch. Pass: no Welcome page; whatever was imported is there; "Import projects" in the
sidebar brings the rest; no duplicates.

## B. Upgrade (restore `Orbital.bak-1124` to `%APPDATA%\Orbital` first, then install over v0.24.1)

**B1.** Pass: no Welcome page; all projects, sessions, cards and settings intact; the Workbench
shows the headline and Recent; the tour does not open by itself; Settings → About → Take the tour
still works.

**B2 — Cards.** Global Settings → model providers. Pass: existing cards intact and still working
(send one manager turn); adding a card through the new connect screen works; deleting a ChatGPT
card signs out.

**B3 — Downgrade.** Reinstall v0.24.1 over it. Pass: the app opens the same data with nothing lost.
(A ChatGPT card, if created, may show as an unknown provider there: report what it shows.)
