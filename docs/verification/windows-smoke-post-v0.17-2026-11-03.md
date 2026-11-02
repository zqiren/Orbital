# Windows smoke: features since v0.17 (2026-11-03)

Machine: Windows 10 Home 19045, system locale en-US (code page 1252), UI culture zh-CN,
UTC+8, Git 2.35.1, Claude Code 2.1.303, Codex 0.162.1. Not a Chinese system locale.

Driven against the installed v0.20.0 RC (`dc24b47`) with its OpenCode Go card, plus isolated dev
daemons for keyless and import flows. Import used a fake Claude Code and Codex history.

## Results

| Feature | Result | Notes |
|---|---|---|
| Chinese project names and paths, spaces, long names | Pass | Manager, Claude Code, Codex, triggers |
| Keyless mode, "API key required" pill | Pass | |
| Session import, Claude Code and Codex | Pass | Chinese folder; opening converts the transcript |
| Codex automation becomes a trigger | **Fixed** | Took over correctly, but timezone was UTC. See bug 1 |
| Project memory record turns | Pass | |
| Locate card | Pass | Section B of the v0.20.0 check |
| Claude Code continuing after a background shell | Pass | |
| Worker told of prior session logs | Pass | Forward slashes on Windows |
| Left-off cards | **Fixed** | Provider refusals read "Stopped". See bug 3 |
| `/` skill menu and catalog | Pass | Manager and worker; a path outside the catalog gets 400 |
| Background jobs | Pass | 25 s ping tracked and woke the agent; other routes stayed near 20 ms |
| Non-UTF-8 files | Pass | GBK workspace file and GBK PROJECT_STATE, on code page 1252 only |
| Asks in composer and Workbench | Pass | Row and actions render; ASKS.md stamped |
| Triggers, manager and worker | Pass | Chinese-named project |
| Safe-mode shell | **Fixed, needs CI build** | Started in `C:\`. See bug 2 |
| Codex shell commands | **Blocked here** | See decision 1 |
| Chinese UI layout, 1180 px | Pass | Keyless, slash menu, resume card, Automations, pin menu |

## Bugs fixed on this branch

1. **Windows local timezone was always UTC.** `_local_tz_name()` only knew `TZ` and `/etc/localtime`.
   A Codex automation set for 10:30 would fire at 18:30 on a UTC+8 machine, and Workbench overdue
   dates used the wrong midnight. Now uses the existing system detector.
2. **Safe-mode shell ran in `C:\`.** The worker cannot read the parents of a workspace under a user
   profile, so PowerShell falls back to the drive root and `Set-Location` is refused. Relative
   commands hit `C:\`. A PowerShell drive rooted at the workspace fixes it with no ACL change; run by
   hand as the real worker on the installed build. `(Get-Location)` now prints `workspace:\`.
   This is unverified end to end until a CI build, because dev daemons have no sandbox account.
3. **Provider refusals read "Stopped" on left-off cards.** "Model is unavailable" and the region
   refusal now read "Model not available".
4. **Python crashed printing Chinese in the agent shell.** Redirected output used code page 1252.
   The script now sets `PYTHONIOENCODING=utf-8` unless it is already set.

## Decisions for you

1. **What should Orbital do when the user's own Codex sandbox is broken?** With
   `[windows] sandbox = "elevated"` in `~/.codex/config.toml`, every command on this machine fails with
   `helper_unknown_error: setup refresh had errors`, in plain `codex exec` too and in any path.
   `-c windows.sandbox="unelevated"` works. Options: repair Codex's setup on this machine, have Orbital
   show this error plainly, or have Orbital override the setting (a security trade-off).
2. **`[STATUS: …]` shows as chat text.** The marker is never stripped, and tests expect it. Hide it?
3. **`deepseek-v4-flash` is in OpenCode Go's `suggested_models` but answers 400 "Model is
   unavailable".** `deepseek-v4.1-flash` and `deepseek-v4-pro` work. Changing the registry needs real
   spec values for the new id.
4. **Raw model ids flash in the pin menu** while the cached live list refreshes: labels such as
   "Opus 5.5" are replaced by `opus`, `claude-opus-5-5`. Cosmetic, from spec 136's fallback.
5. **Automation sessions are named `[Triggered by schedule …`.** Listed as a known issue in v0.19.0.

## Not tested

- DeepSeek Harness, Pi, Kimi Code, MiniMax Code: not installed here.
- The real 403 region refusal: this OpenCode workspace allows Global, so the fix is covered by unit
  tests using the gateway's text from an earlier field report.
- Resuming an imported session against a real CLI: the fake history has no real thread.
- Create-project folder picker (native dialog), the quit prompt for jobs, phone layout on a device.
- `dubious ownership` for the worker: this machine's Git predates the check.
- Sticky agent by clicking through several sessions: seen in passing only.
- A Chinese system locale. This machine's is en-US.
