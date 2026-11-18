# batch/1118 on Windows: results

**Status: READY** for the five features merged at `5e472a5b` (v0.24.0), with one
fix commit on top (`8b49c824`, browser live view). Re-run CI on the tip before
tagging; the installers tested here were built from `5e472a5b`.

Not verified on Windows, by design or for lack of a second UAC click: the
elevated half of the in-app upgrade (see UPG-8 below). It cannot fail for
v0.24.0 users on this release (0.24.0 is only ever upgraded *from*); it first
runs for real when 0.24.1 is published.

Later pushes to this branch get their own section at the end of this file.

- Machine: Windows 10 Home 10.0.19045 (zh-CN), Python 3.13.2, not elevated, UAC on.
- Run: 2026-11-19, 00:30 to 01:40 +0800.
- Installer under test: CI run 44277014359 (`Orbital-Setup-0.24.0.exe`,
  sha256 `6f0f7a79…a44e0de`), installed over a real 0.20.0 at `C:\Orbital`.

## Suites

| Suite | Result |
| --- | --- |
| `pytest tests/unit tests/platform` (minus the two sandbox files) | 8106 passed, 316 skipped, 33 failed, 12 errors. Every red is local to this machine: 31 are `WinError 1314` (no symlink privilege), the rest are the known long-path (`.pi` fixtures, `LongPathsEnabled=0`), `C:\` being a git work tree, a stale ignored `agent_os/_version.py` (0.18.0), and `test_pid_claim_lock::test_a_claim_in_progress_is_never_seen_empty`, which passes alone. CI's windows-latest job is green on the same commit. |
| `npx tsc -b` | 0 errors |
| `npm test` (vitest) | 197 files, 3174 passed |
| `check-i18n` | OK, 5 missing zh (warnings) |

The first pytest run aborted at collection: `zstandard` is a new dependency.
`pip install -e ".[dev]"` fixed it.

## In-app upgrade (`docs/verification/in-app-upgrade-windows.md`)

Rig: the CI 0.24.0 app installed at `C:\Orbital`, started with a scratch
profile (`USERPROFILE`/`APPDATA`/`LOCALAPPDATA` under `C:\orbital-scratch\1118-upg`).
No code was patched. The feed was a local HTTPS proxy that answers
`api.github.com/.../releases/tags/v0.24.1` and the `Orbital-Setup.exe` download
with the same CI installer, offered as 0.24.1 (`HTTPS_PROXY` + `SSL_CERT_FILE`
+ `AGENT_OS_UPDATE_URL`; httpx honours all three). The pill was driven in the
app's own web UI on port 8000.

| Step | Result |
| --- | --- |
| UPG-1 install | PASS. `/SILENT` install of CI 0.24.0 over 0.20.0, exit 0, Restart Manager found nothing in use, `--setup-sandbox` exit 0. Apps & features: one entry, "Orbital version 0.24.0" (`Orbital_is1`), so the explicit `AppId` matches the old implicit one. |
| UPG-2 pill | PASS. `can_install: true` on `C:\Orbital\bin\Orbital.exe`; pill reads 更新到 v0.24.1. |
| UPG-3 download | PASS. 下载中 n% rising, `.part` growing in `%LOCALAPPDATA%\Orbital\updates\`. |
| UPG-4 resume | PASS. Connection dropped at 120 MB: 更新失败 / 重试 / 手动下载, `.part` kept at 120,000,000 bytes. Retry sent `Range: bytes=120000000-` with `If-Range` and finished. Cosmetic: the pill shows 下载中 0% for about a second after Retry before it jumps to the resumed percentage. |
| UPG-5 verify, UAC | PASS. Straight to 正在重启…, `consent.exe` up, no SmartScreen step before it. No `Zone.Identifier` stream on the downloaded file. Setup command line is exactly `setup_command()`'s, with `/ORBITALRELAUNCH="C:\Orbital\bin\Orbital.exe"`. `pending.json` written. |
| UPG-6 decline | PASS (by UAC timeout, about 2 min, nobody clicked). App stayed on 0.24.0, pill 重启以更新, `pending.json` removed, `C:\Orbital` mtimes unchanged. |
| UPG-7 busy prompt | NOT RUN (native dialog, needs a person). Unit-tested only. |
| UPG-8 to 12, 14 to 16 (past UAC) | NOT RUN. Needs a second UAC click and a NEW installer that carries the handshake under a higher version; no such published release exists before v0.24.0 ships. Suggestion: run these in CI on windows-latest (admin, no UAC prompt) with a local feed like the one above; the rig script is small. |
| UPG-13 hash mismatch | PASS. One byte flipped in transit: `error: hash`, pill 更新失败, the `.part` deleted, no Setup process, no UAC. |
| UPG-17 portable copy | PASS. `bin` copied without `unins000.exe`: `can_install: false`, pill is the link 新版本 v0.24.1 →. |

No Defender notification during any of this.

## Worker slash commands (`15100229`)

Live Claude Code and Codex workers, keyless, pinned.

- SLASH-1 claude-code PASS. First message `/context` and a project command
  `/pong` reached the worker alone (its own transcript shows only the command),
  real output came back.
- SLASH-2 claude-code **DEVIATION, not a blocker**. The postponed first-turn
  note and skills note are not lost, but they ride on Orbital's own
  `[Orbital memory]` turn that follows the command, not on the next ordinary
  user message as the commit message says. `_sends_alone`
  (`sub_agent_manager.py:2467`) holds back only user-addressed commands; the
  memory prompt is sent with `daemon_turn=True` (`agent_manager.py:3185`) and
  takes the held notes. The worker still has them before the user's next
  message. Decide whether that is intended.
- SLASH-3 PASS. Commands on later turns run, sent alone.
- SLASH-4 PASS (claude-code and codex). Workbench `/` picker lists the pinned
  worker's commands (85 + project ones; 42 Codex `$skills`), prefills, runs.
  `fast` and `focus` are filtered out of both pickers.
- SLASH-5 PASS. `/usage`, which answers only in a local-command-stdout frame,
  becomes the reply bubble.
- SLASH-6 partly BLOCKED (no API key on that rig, so no unpinned recipient).
  `/tmp/foo is missing…` goes as ordinary text; `/nosuchcmd` is sent alone and
  the worker treats it as text.
- SLASH-7 PASS at 390 px zh. Pre-existing: the `/schedule` row's zh warning is
  clipped without an ellipsis (`SlashMenu.tsx:234`).
- Codex `$skill` messages are not sent alone (only `/` counts as a command);
  they work because Codex finds `$skill` anywhere in the text.

## Column chevrons (`f99b2681`)

All PASS at 1440x900, 1024x768, zh, and 390x844: project chevron, session
hover handle at the exact vertical centre, 20 px rail (measured), rail "+"
starts a session, waiting dot appears and clears, rail Workbench glyph only
while the list is hidden, no `title` tooltips, aria-labels present in both
languages, keyboard operable, no horizontal scroll with both folded.

- UX gap at 1024 wide (not a blocker): with the session column folded there is
  no way to open the right panel or reach the project gear, because both
  buttons live in the column header (`ProjectDetail.tsx:341`, about `:520`).
- With the panel overlay open, the handle draws undimmed above the scrim.
- Pre-existing: `strip.projects` has no zh entry.

## Sources (`0f9af53e`)

- SRC-189 PASS. 17 delete-during-check loops (slow 200, 500 with retry, and
  with the UI open): the source never came back in the list, the UI,
  `sources.json` or after a daemon restart. No 5xx, no tracebacks.
- SRC-190 PASS on Windows. Calendar is listed as unavailable for the platform,
  create returns 409, the add-source list has no Calendar entry, and no
  permission state appears. The strict EventKit read itself is macOS-only.

## Browser live view (`11ee28e0`)

Real Chrome, `opencode-go` / `deepseek-v4.1-flash` as the manager.

- PASS: the view fills the panel (772x822 view, 772x822 page, 0 px off) and
  follows maximize, restore and window resizes; frames are 2x (1544x1644),
  agent screenshots are CSS size (772x822 watched, 1280x720 unwatched); clicks
  land at every size (8/8); the page goes home to 1280x720 when the last viewer
  leaves; the idle-closed page reopens once in 1.2 to 1.6 s and never loops;
  two viewers do not fight over the size; zh labels fit; no orphaned
  chrome/node. `tests/platform/test_browser_live_chrome.py`: 4 passed.
- Divider drag was not driven in the UI (covered by the platform test only).
- **Bug found and fixed (`8b49c824`).** On the dev rig, opening the view on a
  page that already existed failed 7/7 with
  `could not attach to the browser page: … Screencast is already active`, and
  the view stayed at 1280x720 with no navigation state. Cause: the first
  viewport message arrived while `_attach` was still between creating the CDP
  session and starting the screencast; `_apply_viewport` took the missing cap
  for a density change and started the screencast first. The CI-built 0.24.0
  did not show it (0/22 opens), so `5e472a5b` as built is not broken, but the
  race is real and a slow attach could hit it. The fix restarts only a running
  screencast and clears the cap on detach; the regression test reproduces the
  exact error. Browser unit and real-Chrome tests: 106 passed with the fix.
- Left alone: a failed attach is not retried (`_attach_to` keeps `_page`), so
  any other attach failure also leaves the stream stuck until it reconnects.
- `scripts/dev_daemon_isolated.py` does not isolate the agent's browser
  profile (`~/orbital/browser-profile` is used regardless of `--data-dir`).

## State left on this machine

- `C:\Orbital` is now the CI-built 0.24.0 (was 0.20.0). It has not been
  launched against the owner's real profile.
- Scratch data under `C:\orbital-scratch\1118-*`. Evidence (logs, screenshots,
  rig scripts) is in each folder's `evidence\`.
