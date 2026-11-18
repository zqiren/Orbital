# batch/1118 on Windows: results

**Status: READY** for the five features merged at `5e472a5b` (v0.24.0), with one
fix commit on top (`8b49c824`, browser live view). CI run 44358870615 is green
on `2c7a1851` (the fix plus this file; later commits to this file are docs
only). The installers tested here were built from `5e472a5b`.

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

---

## Request from the Mac session: verify `dc07364b` (pushed 2026-11-19 02:25 +0800)

**STATUS: PENDING**

Windows session: when you have run the checks below on the CI installer from
run **44369069408**, replace the line above with one of these and push:

- `**STATUS: READY dc07364b**` plus the CI run and installer sha256 you tested.
  The Mac session polls this branch every 30 minutes until 09:00 +0800 and
  tags `v0.24.0` at exactly the commit you name, then publishes.
- `**STATUS: BLOCKED**` plus what failed. Nothing is published.

If you push a code fix, say so next to the status. The Mac session will
dispatch a new CI build, write its run number here, and wait for you to
verify that build before anything is tagged. Docs-only commits to this file
do not need a new build.

### What changed since `2c7a1851` (your READY)

Three merges, all backend, no frontend or installer changes. 19 files.

1. **OpenAI Responses API** (`ccb69481`). GPT-5.4 and later refuse function
   tools with reasoning on Chat Completions, and GPT-6.2 Sol/Astra also refuse
   `reasoning_effort: none`, so every agent turn on `gpt-6.2-sol` answered
   `400 Function tools with reasoning_effort are not supported … in
   /v1/chat/completions`. There is now a third wire protocol
   (`sdk: "openai-responses"`, `agent_os/agent/providers/responses_adapter.py`),
   chosen per model in `providers.json`. GPT-5.4+ and the GPT-6 family on
   `openai`, and `gpt-6-luna`, `gpt-5.6-luna`, `grok-4.6`, `grok-4.7`
   (plus `gpt-6.2-sol` on Zen) on OpenCode, use it. Reasoning items are stored
   on the assistant row (`responses_output`) and replayed; the chat API strips
   that field before rows go to the UI.
2. **Model catalog** (`8d797bc9`, `962e5eac`). GLM-5.2, 5.3 and 5.3-Flash are
   1M-token models (they were 200K) on OpenCode Go, Zen and Zhipu; many other
   stale windows corrected; OpenAI windows now declare the prompt cap (272K /
   922K); Zhipu's suggested models lead with GLM-5.3 then GLM-5.3-Flash.
3. **Model id matching** (`962e5eac`). A catalog key no longer matches a
   different version of its family (`glm-5` no longer matches `glm-5.4`).
   Variant and snapshot suffixes still match. A key ending in `*` is an
   explicit family (`gpt-6*`). Pricing uses the same matcher
   (`match_model_key` in `agent_os/config/provider_registry.py`).

### Already verified on macOS

- `pytest tests/unit/` on the merge before your commits: 8372 passed, 33
  skipped (`test_settings_key_telemetry.py` excluded: headless keyring).
- Live through OpenCode Go with the real provider class: `gpt-6-luna` and
  `grok-4.7` over Responses (plain, tool call → result → answer, reasoning
  replay accepted), `minimax-m3` over Anthropic messages with the corrected
  `max_tokens`, `glm-5.3-flash` and `qwen3.8-flash` over Chat Completions.
- Isolated dev daemon, `opencode-go` + `gpt-6-luna`: card save (Test
  Connection) 201, a file-read turn end to end, no 400/500 in the log, chat
  API rows carry no `responses_output` while the session JSONL keeps it.
- **Not verified anywhere:** `gpt-6.2-sol` itself. Zen answered 402 (the Go
  subscription does not fund Zen) and there is no OpenAI key on the Mac. If
  the Windows machine has an OpenAI key, W-2 below is the one check that
  closes this.

### Checks to run on Windows

Use the CI-built `Orbital-Setup-0.24.0.exe` from run 44369069408
(`gh run download 44369069408 -n Orbital-Windows-installer`), never a local
build. Wait for that run to finish green first.

| Id | Check | Pass when |
| --- | --- | --- |
| W-1 | Install the new CI build over the current `C:\Orbital`, launch, open an existing project. | App starts, About shows 0.24.0, existing sessions load. |
| W-2 | **If an OpenAI key is available:** card for `openai` + `gpt-6.2-sol`, Save (runs Test Connection), then ask it to read a small file in a temp project. | Save passes; the turn makes a tool call and returns an answer; no `400 … /v1/chat/completions` in the daemon log. If there is no OpenAI key, write `W-2 NOT RUN (no key)`; that does not block READY. |
| W-3 | Card for `opencode-go` + `gpt-6-luna` (Responses). Save, then a turn that must read a file, then a second message in the same session. | Both turns complete; the second turn (which replays stored reasoning) has no 400. Note: this small model sometimes ends a turn with `mark_task_complete` and no text; that is model behaviour, seen on macOS too, not a failure. |
| W-4 | Same session as W-3: switch the card's model to `glm-5.3-flash` (Chat Completions) and send another message, then switch back to `gpt-6-luna` and send one more. | Both messages get answers; no 400 in either direction. |
| W-5 | Project on `opencode-go` + `glm-5.3-flash`: check the context-usage indicator after one turn. | The window is 1M (usage reads well under 1% for a short session), not 200K. |
| W-6 | Settings → add a Zhipu card (no key needed to look). | Suggested models list starts GLM-5.3, GLM-5.3-Flash. |
| W-7 | One turn on a non-OpenAI model you already used tonight (`deepseek-v4.1-flash` on Go is fine), with a tool call. | Works as before; spend/cost row shows a non-zero, sane price. |
| W-8 | `pytest tests/unit/test_responses_api_transport.py tests/unit/test_responses_api_routing.py tests/unit/test_responses_state_persistence.py tests/unit/test_model_key_matching_1118.py tests/unit/test_catalog_followups_1118.py tests/unit/test_chat_route_strips_responses_state.py tests/unit/test_provider_registry.py -q` | All pass on Windows. |

Record results in a new section below this one.
