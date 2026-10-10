# Windows session, 2026-10-10 evening → 2026-10-11 night (v0.16.0)

Follow-up to `109-windows-report-2026-10-10.md`. Branch `batch/1009-specs-102-110`.

## Installer to test

`…\scratchpad\ci-38074658058\Orbital-Setup-0.16.0.exe`, CI run **38074658058** (commit `8539651`),
404,769,518 bytes, SHA-256 `8CE1FB4BDB8C5853229E8AD5EFEAC93C5E8D824BFA5A33D3F170AC69D9581365`.
CI: frontend green; backend green on windows-latest (4494 passed) and macos-14 (4604 passed) apart from
the known `test_browser_live` red.

**Your machine is set up for a true first run.** Orbital is closed, and its two data folders were moved aside
(not deleted):
`%APPDATA%\Orbital` → `Orbital.saved-before-first-run`, and `%USERPROFILE%\orbital` →
`orbital.saved-before-first-run`. Install the build (UAC), let it launch, and you get the setup wizard, then
the first-journey tour once you create a project. The first launch re-extracts the bundled Chromium in the
background, as it would for a new user. To get your projects back afterwards, quit Orbital and run
`powershell -ExecutionPolicy Bypass -File …\scratchpad\orbital-restore.ps1`.

The tour state and the language choice now **persist across restarts** (fixed tonight), so a restart
mid-tour no longer resets them.

## Fixed this session (all on the branch, each with tests)

| Commit | What |
|---|---|
| `e312f5b` | Claude Code ran on no Windows machine with an npm install: the SDK refuses `claude.CMD`; now handed the native `claude.exe` |
| `b17edb8` | A failed pinned dispatch is logged and shown in the chat instead of vanishing |
| `6b57a82` | Codex gets a **Login** (ChatGPT) button beside Set API Key |
| `4213c7e` | No empty console window when Claude Code runs |
| `c866fff` | Claude model labels say what runs ("Opus 5.5"); model-list probes no longer leak `claude.exe` / `codex.exe` |
| `c89e4fd` | An abandoned Login no longer leaves `claude auth login` running |
| `f793601` | The test suite no longer starts a real `claude auth login` on dev machines |
| `7c00aa5` | **Signed-in Codex shows as signed in on Windows**, so its usage appears (the check was POSIX shell, run by cmd.exe) |
| `6a4f4ed` | Codex usage read no longer leaks `node` + `codex.exe` |
| `8539651` | The window keeps localStorage across restarts (language, tour state) |

Also: updated your npm CLIs (codex 0.162.1, Claude Code 2.1.296), so the live model lists are current.

## Verified tonight in an isolated bench (today's code, separate data dir, a local mock model)

Your OpenCode key did not come through in the message, so I ran Orbital against a local mock
OpenAI-compatible model instead. It used throwaway data and touched none of your projects.

Passed: setup wizard (EN/ZH switch, custom provider, Test Connection, connectors step, project scan), project
creation, the first-journey tour (all stops), chat with the main agent, shell tool, Files tab and preview with a
Chinese filename, Settings → Sub-agents (Codex **Logged in**, Claude **Logged in**, versioned model labels),
Codex usage in the agent picker ("Week · 63% left"), **pinned Codex and Claude Code both answered**, the
session appeared at once, and worker bubbles survived a reload.

## Issues for you to decide (not fixed)

1. **Chinese text in shell output becomes `?`.** On Chinese Windows, PowerShell started without a console
   writes CJK as literal `?`: `dir` lists `中文笔记.txt` as `????.txt`, so the agent cannot read Chinese
   filenames from shell output. Reproduced outside Orbital with the exact `cmd /c … powershell -Command … >
   file` shape. Prefixing the command with `[Console]::OutputEncoding=[Text.Encoding]::UTF8;` returns
   `中文笔记.txt`. The installed app's sandboxed path uses the same shape. *Proposed:* add that prefix in the
   Windows PowerShell wrapper (`agent_os/agent/tools/shell.py`). High impact for zh users.
2. **A workspace path with a space counts as "outside the workspace".** `_WIN_ABS_RE` stops at whitespace,
   so for workspace `…\My Project`, `dir "…\My Project"` is flagged as outside (`…\My`). The `[focus]`
   warning then tells the agent not to read or explore its own project. Common on Windows. *Proposed:* match
   quoted paths whole and compare against the workspace by prefix.
3. **Import scan offers your home folder (`C:\Users\qiren`) as a project.** Importing it makes the profile a
   workspace, which grants the sandbox account full control of it. It also offers Orbital's own Quick Tasks
   folder and a `%TEMP%` folder. *Proposed:* exclude home, drive roots, Orbital's data dir and temp dirs from
   suggestions, and refuse home or drive roots as a workspace at creation.
4. **Import scan shows `\\?\C:\…` paths and duplicates them.** Codex records `\\?\`-prefixed paths. The
   dedupe key keeps the prefix, so the same folder appears twice (once per source) with the raw prefix shown.
   *Proposed:* strip `\\?\` before the realpath key and for display.
5. **The agent picker is disabled until the first message, but tour step 3 tells you to use it.** It is
   disabled while the chat has no session (`disabled={sessionId === undefined}`), and a new project has none.
6. **Clipped sessions column at the default window size.** At 1200×800 with the preview panel open, the
   sessions list is squeezed and cut off ("Automations", "0s ago", "sion").
7. **Onboarding:** pressing **Next** before **Test Connection** sends a real chat request, then still asks you
   to test (a wasted call on a paid provider).
8. Small: Codex version shows as "vcodex-cli 0.162.1". Tour step 7 points at the collapsed edge strip, not at
   Quick Tasks. "Claude Code, Codex" should read "and". The "claude-code" name wraps in the sent/completed rows.
   The folder picker flashes "Root / No folders here" while it loads. A reload doesn't reopen the last project.
9. Environment, not Orbital: `npm i -g @openai/codex@latest @anthropic-ai/claude-code@latest` (both at once)
   died silently twice on this machine, leaving the commands missing. One package at a time worked.

## Not done

- Anything needing your real model key (none was received).
- The new installer on the real machine: install, V3 sign-in screen and the first-run experience. These need
  you (UAC).
