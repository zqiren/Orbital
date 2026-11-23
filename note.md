# Windows test note — one-page Welcome + Workbench landing, with the Antigravity worker (2026-11-24)

Branch-only file: it is dropped before this branch merges to main.

**Eligible network required for part C.** Antigravity refuses accounts whose network exits in a
region Google excludes (Hong Kong and mainland China among them): the sign-in itself works, but
every reply is a refusal. Run part C on a US, Japan or Singapore exit, or on no VPN outside the
excluded regions. Parts A and B need no special network.

**Build under test:** branch `test/1124-welcome-antigravity` (`feat/1123-one-page-welcome` with
`feat/antigravity-worker` merged in), commit `<COMMIT>`, CI run `<RUN>`.
Installer: the `Orbital-Windows-installer` artifact of that run, `Orbital-Setup-0.24.1.exe`,
sha256 `<SHA256>`.
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
- Merged in: the Antigravity worker (Google's `agy` CLI over ACP) replaces Gemini CLI. Its sign-in
  asks for a code: agy opens the browser itself, then the agent row shows one "Paste the code" /
  "粘贴授权码" field. That field is on the Welcome page, in Settings → Sub-agents, and in a chat's
  "signed out" notice.

**Already proven by CI on Windows (do not repeat), for the Welcome build before the merge:** installer builds; the fresh-install scenario
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

## C. Antigravity (merged build only)

The full Antigravity steps (install path, sign-in, PONG, approvals, resume, model switch, what to
record) are in `note-antigravity-windows.md` beside this file: run them on this installer. The
checks below exist only because both features are in one build. Fresh data (`%APPDATA%\Orbital`
renamed away, as in part A) unless a check says otherwise.

**C1 — Not installed.** Before installing agy, open the Welcome page.
Pass: no Antigravity row among the agents; "Add another agent" lists an Antigravity chip, and its
card shows "Not detected" with the locate card. No "Gemini CLI" anywhere.

**C2 — Installed, signed out.** In PowerShell run `irm https://antigravity.google/cli/install.ps1 | iex`
(do not sign in), then on the Welcome page press "Check again" under "Add another agent" (or quit
and relaunch). Pass: an Antigravity row with a Sign in button, no check. If no other agent is ready
and no key is set, Start stays disabled. Record whether it was found without the locate card and
the path `agy.exe` landed at.

**C3 — Sign in from the Welcome page.** Press Sign in on the Antigravity row.
Pass: a browser tab opens on Google sign-in, opened by agy (Orbital opens nothing; use that tab, a
code from a second tab fails with "Invalid code verifier"). Under the row a "Paste the code" field
appears with Submit (the row's neighbour keeps its height; Start stays on screen, also with the
window at about 900 px wide). Paste the code Google shows, press Submit or Enter. The field goes,
the row turns to a check without a relaunch, and Start becomes enabled if Antigravity is the only
ready agent. Record any console window that flashes. The code must show as `[redacted]` in
`GET /api/v2/settings/sub-agents/login-jobs` and must not appear in `daemon.log`.
Also press Sign in on Codex or Claude Code if one is signed out: no code field must appear for them.

**C4 — Start gate with Antigravity only.** Sign Claude Code and Codex out (or test on a PC without
them), no key, Antigravity signed in. Pass: Start is enabled; Start lands on the Workbench; a chat
pinned to Antigravity gets a reply (on an eligible network). On an ineligible network Antigravity
still reads signed in here (its probe cannot see the region) and the first reply is the error
"Antigravity isn't available in your region": report exactly what the Welcome page and that first
message showed.

**C5 — Keyless routing ignores Antigravity.** No API key, Antigravity signed in AND Claude Code or
Codex signed in. Repeat part A7 (two scratch projects, Auto, five tasks).
Pass: every routed message is answered by the routing call of Claude Code or Codex (the
`/api/v2/workbench/route` answer in the daemon log names `claude-code` or `codex`, never
`antigravity`), routing takes about as long as in A7, and no new conversation appears in agy's own
history for the routing calls. With ONLY Antigravity signed in (C4's setup), an Auto message lands
in Quick Tasks.

**C6 — Signed-out notice for Antigravity.** Antigravity installed and signed out, no key, Claude Code
and Codex signed out or absent. Send a message in a scratch project. Pass: the notice names
Antigravity and offers Sign in; Sign in opens agy's browser tab and the same paste-the-code field
appears under the notice's buttons; after Submit the notice goes and the next message runs.

