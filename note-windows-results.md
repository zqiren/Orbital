# Windows results — one-page Welcome + Antigravity (2026-11-24)

Branch-only file, like `note.md`. Not committed yet.

## STATUS: A and B done on the CI installer except the sign-in rows; part C only up to the sign-in (C1, C2, code field)

- Build under test: `test/1124-welcome-antigravity` @ `dd1fc782`, CI run `45220698491`.
- Installer: `Orbital-Setup-0.24.1.exe`, sha256
  `351ae6cd4fe864c25b6c632fb82b33becfe7f40c0ff4a6efb7dc15f6ec04ae4a` (the run's
  `Orbital-Windows-installer` artifact; `C:\Orbital\bin\Orbital.exe` dated 2026-11-23 16:55 UTC after install).
- CI on that run: frontend, backend unit tests (Windows + macOS), both installer builds, Windows
  fresh-install + upgrade scenarios green. macOS fresh-install jobs were still queued at 01:20.
- Network exit: US. Windows 10 Home 19045. Agents on this PC: Claude Code 2.1.303, Codex (npm
  0.162.1 and desktop 0.171.0-alpha.13), DeepSeek Harness 0.2.2-rc.1. No agy.
- The UI was driven in a browser on `http://127.0.0.1:8000` against the installed app's daemon
  (as the CI scenario does); the native window was open too. A watcher logged every visible console
  window for the whole run (`evidence\winwatch-ci.log`); it caught a control `cmd` window and nothing else.
- Machine state now: published v0.24.1 reinstalled, real data back in `%APPDATA%\Orbital`, agy
  1.5.15 left installed (signed out). v0.24.1 now shows an Antigravity agent as installed, because
  of finding 1.

## Part A — fresh data, CI build

| Check | Result | What was seen |
|---|---|---|
| A1 Welcome, agents | PASS | One page, no step strip. Claude Code, Codex, DeepSeek Harness with checks. Codex resolved to the desktop copy `%LOCALAPPDATA%\OpenAI\Codex\bin\31fb36cfc4b38f05\codex.exe` over `%APPDATA%\npm\codex.CMD`. Daemon answered 5 s after launch; first agent sweep 1.51 s (slowest claude-code 0.85 s), so the list is there when the page paints. No console window. |
| A2 Sign in from the page | NOT RUN | No agent is signed out on this PC. Dev-daemon rehearsal with a scratch home: Sign in on Claude Code started a login job, spinner + "Open the sign-in page" link, no code field, no console window; sign-in not completed. |
| A3 Projects line | PASS | "Import 49 projects" pre-checked (34 → 49 while the scan finished: 262 sessions in 2.2 s, repo walk 0.5 s). Drive-letter/backslash paths correct. Search filters; select-all acts on the filtered rows; unticking all turns the button into "Start"; Esc restores the page. The button reads "View all", not "Review all". No OneDrive-redirected folders on this PC. |
| A4a real key | PASS | OpenCode Go key: tested itself, green check, model row `kimi-k3`. Key was typed + Enter (the test also fires on blur); a real clipboard paste was not exercised. Folded summary (logo + model) seen on the dev daemon only. |
| A4b wrong key | PASS | "Invalid key · 401" under the field; Start stayed enabled. |
| A4c ChatGPT, A4d TokenDance | NOT RUN | Need the owner's browser sign-in. |
| A5 Folding | PASS (partly on dev daemon) | CI build: clicking the key field folds Agents to the logo line; nothing folded on load; 中文 at about 800 px fits. Each band's fold/unfold and the 900 px English check were done on the dev daemon. |
| A6 Start | PASS, with a smaller import | Workbench at once, headline + Recent, no New project dialog, no tour. Imported 36 of the 49 (all `C:\orbital-scratch\…` and `D:\orbital-smoke…` folders; the 13 real folders were left unticked so the test would not write `orbital\` into them): 36 projects in 3.4 s, sidebar filling as they landed. The spinner by the count was not captured (it finished too fast). |
| A7 Keyless routing | PASS | UI send "add a line to the invoice notes": slot showed Auto, settled on `invoices` in 7.2 s, Claude Code replied. Five more through `POST /workbench/route`: 4 of 4 project tasks right in 7.2–7.3 s, an unrelated question to Quick Tasks in 7.2 s. Log: `workbench route: <pid> by claude-code in 7.2s`. 38 candidate projects. No console window. `~\.claude\projects` gained only the `invoices` project's own folder; Codex rollouts unchanged. Codex as the router was not exercised (Claude Code comes first). No >15 s call occurred. |
| A8 Signed-out error | NOT RUN | Needs a real sign-out. Dev-daemon rehearsal: notice "Claude Code 未登录" with 登录 / 关闭, no "no agent is enabled" text. |
| A9 Recent | PASS | Opened an imported Codex session from Recent ("Imported from Codex · Nov 19"); transcript shown; sent a message, Codex replied PONG in it; back on the Workbench it sat at the top of Recent. The live-card step in between was seen on the dev daemon, not here (the reply was read in the session). |
| A10 Quit mid-import | PASS | Fresh data again, 36 ticked, app killed with 9 imported. Relaunch: no Welcome page (`setup.api_key_skipped: true`), the 9 are there, the import does not resume by itself. "Import projects…" offered the other 40 with the 9 excluded; importing 2 more gave 12 projects, no duplicate workspace. |

## Part B — real data (last used 2026-11-07, 5 projects, one OpenCode Go card)

| Check | Result | What was seen |
|---|---|---|
| B1 | PASS | No Welcome page. All 5 projects, sessions and the card intact. Workbench shows the headline and Recent. Tour did not open by itself; Settings → About → Take the tour started it (step 1 of 5). |
| B2 Cards | PASS, ChatGPT part NOT RUN | Existing card healthy; a manager turn in Quick Tasks replied PONG on it. Added a second card through the new connect screen (key tested itself, model row, Save); existing card stayed default. Deleted the test card again (through the API: the test browser blocks the native confirm dialog). No ChatGPT card to delete. |
| B3 Downgrade | PASS | Published v0.24.1 (sha256 `4c1855ad…c6e704`) installed over it: opens the same data, card, 5 projects and tonight's Quick Tasks session all there; old headline back. See finding 1. |

## Part C — Antigravity, CI build, fresh data (second install of the test build, 06:17)

The owner could not sign in to Google ("we can't sign in for some reason"), so everything that
needs a signed-in Antigravity is NOT RUN: no real reply has been seen on Windows either.

| Check | Result | What was seen |
|---|---|---|
| C1 Not installed | PASS | No Antigravity row; chip under "添加其他 Agent"; card "未检测到" with the locate card. No "Gemini CLI" in the page or in `/settings/sub-agents`, although the stale `gemini-cli.yaml` from the older install was still in the manifests folder (finding 1). |
| C2 Installed, signed out | PASS, Start gate NOT RUN | `irm https://antigravity.google/cli/install.ps1 \| iex` put `agy.exe` 1.5.15 at `%LOCALAPPDATA%\agy\bin\agy.exe` and added that folder to the user PATH (not to the running shell). "重新检测" on the Welcome page: Antigravity row with a 登录 button, no check, found without the locate card (`source: path`). Start gate with no other agent ready could not be tested (Claude Code, Codex and dsh are signed in). |
| C3 Sign in from the Welcome page | PARTIAL | Pressing 登录: spinner, an "打开登录页面" link (Google OAuth URL from agy), and under the row a "粘贴授权码" field with 提交; the neighbouring row kept its height and Start stayed on screen at 1280 px. Login job tail: "Open this link in a browser…", "Then paste the authorization code…". A made-up code + Enter: field went away, row back to 登录 with "登录未完成。请重试，或在终端运行该 CLI 的登录命令。"; job `failed / nonzero_exit`, tail `couldn't log in: token exchange failed: oauth2: "invalid_grant" "Malformed auth code."`. The code appears neither in the job tail nor in `daemon.log`. No console window for `agy acp --login`. Real code, check mark, Start enabling: NOT RUN. Codex/Claude Code show no code field (dev-daemon rehearsal). Whether agy itself opened the browser tab was not observed. |
| C4, C5, C6 | NOT RUN | Need a signed-in Antigravity and/or Claude Code and Codex signed out. |
| note-antigravity step 1 | PASS | As C2. `setup_actions` carried the `irm …` command before the install (seen on the dev daemon). |
| note-antigravity steps 2–6 | NOT RUN | Need the sign-in. |

Recorded for the Antigravity note:
- Status sweep with agy installed: 2.3–2.8 s for 10 agents (agy 1.6–2.0 s; `agy models` exits 1
  with "Please sign in to view available models").
- Console windows: one `agy.exe` console window appeared during the first sweep after the install
  (06:22:54.9, pid 22828). Three later sweeps and the login job showed none. Every spawn in
  `agent_os/agents/*.py` and `api/routes/settings.py` passes `win_no_window_flags()`, so the window
  was probably opened by agy itself (a child it starts on its first run); that is an inference.
- The login job tail shows the URL's "…" as `â€¦` (UTF-8 read as cp1252).
- After "重新检测" the space where the Antigravity card had been stayed as an empty gap under the chips.

## Findings

1. **The installer never removes files a newer build added.** After B3, v0.24.1 lists
   "Antigravity" among its agents: `C:\Orbital\bin\_internal\agent_os\agents\manifests\antigravity.yaml`
   stays behind and v0.24.1 loads every `.yaml` in that folder. The same gap means an upgrade to
   this build leaves `gemini-cli.yaml` in place (it is `listed: false`, so it does not show).
   `installer/agentos-setup.iss` has no `[InstallDelete]`. Confirmed on the second install of the
   test build: `gemini-cli.yaml` dated 2026-11-19 sat beside the new `antigravity.yaml`. Not
   checked: whether a loaded stale `gemini-cli` manifest gets in the way of the retired-slug mapping.
2. **Orbital's own Codex turns come back as imported sessions.** After an import, Recent and the
   session list show sessions titled "Other agents in this project: codex, dsh. When…", marked
   "Imported from Codex", whose only content is Orbital's injected preamble and memory prompt,
   shown as "you → @codex". Seen on folders that were Orbital projects before.
3. **Old pending asks flood the Workbench after an import.** 23 "这样理解对吗？" cards appeared
   from the imported folders' existing `orbital\` state, in Chinese under an English UI. Specific
   to folders that were Orbital projects before.
4. The locate card for a not-installed Antigravity asks the user to "copy the prompt to
   Antigravity and paste its answer", which cannot be done before it is installed (dev daemon).
5. Codex still reads signed in when `USERPROFILE` points at an empty folder, so the signed-out
   checks for Codex need a real `codex logout`.

## Testing notes for the next Windows session

- Shells started from the Claude desktop app see a private copy of `%APPDATA%`
  (`%LOCALAPPDATA%\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Roaming`). A stale `Orbital` folder
  there (2026-10-27) shadowed the real one in every listing. It is renamed to
  `Orbital.stale-layer-1124`. Orbital itself, launched from such a shell, wrote to the real folder.
- CI builds the installer with Inno Setup 6.7.1; innoextract 1.9 cannot unpack it, so there is no
  way around the UAC prompt.

## Left on this PC

- agy 1.5.15 at `%LOCALAPPDATA%\agy\bin\`, and that folder on the user PATH.
- `C:\orbital-scratch\1124\`: both installers, `appdata-run1`, `appdata-run2` and
  `appdata-run3-partC` (the fresh data folders), `realdata-safety-copy` (the real data before part B, minus browsers/webview),
  dev-daemon data, scratch workspaces, `evidence\`.
- New Claude Code project folders `~\.claude\projects\C--orbital-scratch-1124-ws-invoices` and
  `…-1124-ci-invoices`; one extra Codex turn in `C:\orbital-scratch\1118-slash\proj`.
- `orbital\sessions` / `orbital\history` refreshed inside the imported scratch and smoke folders.
- Real data: one new Quick Tasks session ("Reply with the single word PONG…"); the tour was opened
  and skipped in AI CELL.
- innoextract 1.9 installed per-user through winget (did not help; safe to remove).
- A Claude sign-in page may have opened in the default browser from the A2 rehearsal.
