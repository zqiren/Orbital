# Windows results — batch/1110 (specs 166, 167, 168 + the keyless pass)

Tested 2026-11-10 01:25 → 03:30 (local, UTC+8) against `note-1110-windows.md` and
`note-keyless-workbench.md` on this branch, from source.

| Item | Value |
|---|---|
| Code under test | `batch/1110` @ `ca770dc` (+ this pass's fixes, listed below, which the rigs picked up on restart) |
| Windows | Windows 10 Home China 22H2, build 19045 |
| Python / Node | Python 3.13.2 (`.venv`), Node v22, Playwright 1.58.1 (Chromium 1208, headless) |
| Agents | Claude Code 2.1.303 (`%APPDATA%\npm\claude.CMD` → native `claude.exe`, signed in), Codex CLI 0.162.1 (`codex.CMD`, signed in). **No ACP agent** (dsh is a manual `dsh-acp-demo.cmd` stub, `installed: false`; Kimi / MiniMax / Cursor / Pi absent). |
| Orbital key | none on the rigs (keyless) until the keyed cases, then an `opencode-go` / `deepseek-v4.1-flash` card (test `ok: true`) in the daemon's in-memory keyring |
| Rigs | three isolated daemons (`scripts/dev_daemon_isolated.py`, fresh data dirs under `C:\orbital-scratch\{A,B,C}\data`, ports 8338/8339/8340) each with its own Vite (5173/5174/5175) and its own Playwright profile; one tester per rig (A: keyless core + 167; B: Workbench, imports, E; C: 168, D, 166) |
| Evidence | `C:\orbital-scratch\<L>\results.md`, `shots\<case>.png`, `daemon.log`, `data\logs\daemon.log`, session JSONL under each scratch workspace |

**Rig note that affected every tester (fixed, commit `2ad4f1a`):** `scripts/dev_daemon_isolated.py`
never set `AGENT_OS_PORT`, so the first daemon of each rig attached no dispatch MCP endpoint ("AGENT_OS_PORT
is not set: the dispatch MCP endpoint is not attached, so pinned workers get no dispatch tool"). On that
daemon a pinned Claude Code ran `codex exec` from its own shell and saved the recipe to its MEMORY.md
(168-1/2/7 FAIL as briefed). Every rig was restarted on the fixed launcher; each tester lists which cases
ran before the restart. All PASS lines below are from the corrected rig unless marked.

## Fixes made in this pass (each its own commit, on `batch/1110`)

| Commit | What | Case |
|---|---|---|
| `4638c70` | Wizard Skip clicked before the key form's settings read landed sent `import_scan:false`, turning the deferred imports off for good; Skip now reads the settings itself when the form has not reported (test: `SetupWizard.test.tsx`) | 167-W2 (review) |
| `67505ce` | `skill-catalog?worker=` cold-probed (spawned a CLI for) a worker in the project's `disabled_sub_agents`; now folds nothing in and spawns nothing (test: `test_inject_skill.py`) | 166 (review) |
| `2ad4f1a` | `dev_daemon_isolated.py` exports `AGENT_OS_PORT`, so pinned workers on an isolated daemon get the dispatch tool | 168-1/2/7 |
| `1d75d9e` | The import scan's Stage 3 goals path was `D:\…\ws/orbital/instructions/project_goals.md` (raw workspace + `/`); it now goes through `_prompt_path` like the preamble (test: `test_worker_onboarding.py`) | 167-W1 |
| `88c5436` | `## Next steps / - None pending; awaiting the user's next instruction.` (what Claude Code's record turn writes, 3/3 drafts) became step 1 of the confirm card (对，做第 1 步 \| 对，先不做 \| 需要改); items opening with none / nothing / n/a / 无 / 暂无 / 没有 are placeholders now (test: `test_import_scan_160.py`) | 167-W2/W3 |
| `583e49b` | composer placeholder showed the pin's slug (「发消息给 claude-code…」); now the product name like the keyless hint under it (test: `ChatView.test.tsx`) | cosmetic (A) |
| `9098e44` | a worker's record turn could read another session's turn from the shared transcript ("Files this turn changed: none" after an edit, or a foreign file list): a turn of this session counts even before its boundary row lands, and with none of it in the file the hint is none (test: `test_worker_memory_record.py`) | 167-W5 / Known #2 (A, B) |
| `b481a11` | a refused resume of an idle session (keyless Orbital session, `missing_api_key`) appended the user row and no failure row, so a reload showed an unanswered message; now the same `start_failed` row as the fresh-start path (test: `test_message_write_ahead.py`) | E2 (B) |
| (test only) | `test_a_scan_records_its_session_on_the_project` waited 50 bare ticks for work that crosses `asyncio.to_thread`; it now waits for the RUNNING state | suite |

Static review of `b260bfd..ca770dc` (8 angles) found the first two above plus items 8 and the
efficiency note in `note-1110-decisions-windows.md`; the Codex `skills/list` parser returning None for a
group without a `skills` list is the builder's explicit choice (its test asserts it) and was left.

Everything that needs a product or architecture call is in **`note-1110-decisions-windows.md`**
(18 items, each with a suggested default).

## Second pass (2026-11-10 morning): the decisions, implemented on Windows

The owner went through `note-1110-decisions-windows.md` from first principles; each item's outcome is
recorded in that file. What landed on the branch, each its own commit with a regression test:

| Commit | Decision | What |
|---|---|---|
| `c8da35d` | 14 | Windows: every worker process (and its shells) is assigned to one kill-on-close Job Object, so a hard daemon kill takes the workers with it (`agent_os/utils/windows_job.py`, ctypes only; `test_windows_job.py` proves a child inherits membership) |
| `9a46c79` | 6 | the dispatch tool's `task` argument says it carries the text and that there is no `message` argument |
| `cb1b652` | 7 | `/mcp` is always attached; with no `AGENT_OS_PORT` the URL comes from the bound port in the first request's ASGI scope (`BoundPortProbe`), never a guessed default |
| `f276d26` | 13 | the native thread id is recorded when the worker gets it (Claude's init frame, Codex's `thread/start`) through a transport sink → `on_thread_update`, so a restart during the FIRST turn resumes |
| `44fa468`, `786d2a0` | 3 | a start the provider would refuse writes no row (keyless A6 leaves no dead session); no key AND no enabled worker is `no_runner`, with its own UI message naming both remedies |
| `b4a5ef5` | 8 | the goals card is never lost: `goals_draft_pending` is stamped on the project at every goal-less run start and when a worker is asked to draft; the run end (record turn or not) and the worker's memory terminal post the card whenever goals exist and a draft was owed; posting clears the stamp |
| `285348f` | 2 | 需要改 / Needs changes on the goals card (Workbench and in-chat) prefills the reply with 「需要改：」 and focuses it; the daemon classifies the prefixed line as the guarded change turn; the feed and session-asks payloads carry `change_option` |
| `3b8112b` | 1 | the owner chose wording over plumbing: an ⓘ beside Autonomy Level (project settings and the create form) says the level applies to the Orbital agent, other agents keep their own permission models |
| `109cc67` | 12 | `/schedule` stays in the keyless picker (owner's call) with a warning that it creates a routine on the user's account outside Orbital |
| `f7a626f` | 4, 18 | the note's W2 wording; the runbook's history recipe is non-probe |
| `1fb22e3` | §C | the two Windows test artifacts made deterministic (shutdown snapshot spawns the real interpreter; the transcript traversal test accepts the SPA's text/html but never JSON) |

Not changed, by the owner's decision or because nothing was needed: 5 (shell writes in the changed-files
hint), 9, 10, 15; 11 (the lenient reader already strips a UTF-8 BOM, so only raw viewers see it).

**ACP agent (item 17):** the owner approved installing DeepSeek Harness 0.2.2-rc.1 on this machine
(`deepseek-harness-0.2.2-rc.1-win-x64.exe`, 280 MB from the vendor URL in `dsh.yaml`, Authenticode
signature valid, per-user NSIS `/S /currentuser`, no UAC). The stale `dsh` demo override was removed from
`~/.orbital/sub_agent_config.json` (the only edit to a shared HOME file in this pass, backed up); Orbital
then detects `…\DeepSeek Harness\resources\runtime\cli\bin\dsh.cmd` as installed and ready. The OpenCode Go
route rides on a scratch `DSH_HOME` (`C:\orbital-scratch\dsh-home\profiles\desktop\cordis.patch.yml`: an
`llm-pi-ai` providers row `opencode-go` with `api: anthropic-messages`, `baseURL https://opencode.ai/zen/go`,
`apiKeyEnv OPENCODE_API_KEY`, header `x-opencode-session: orbital`, model `deepseek-v4.1-flash`; and
`agent-default-model` pointing at it), which Orbital's mirror copies into every spawn; the key lives only in
the test daemon's environment. First round trip: pinned dsh answered PONG, its memory turn drafted
`project_goals.md` and the 这样理解对吗 card opened. The dsh ACP rows are in the section below.

**Upgrade check (item 16, upgrade half):** the owner approved copying the real app's data dir
(`%APPDATA%\Orbital`, `data_version: 1`, the v0.21 layout) to `C:\orbital-scratch\upgrade\data` without
`browsers/` and `webview/`, with `projects.json` cut to three projects whose workspaces were also copied to
scratch (`smoke1022`, `open117`, the data dir's own `scratch`), so no real folder could be touched. This
branch's daemon opened it on port 8341 with no error or migration line, listed every old session (3 + 3) as
idle chats, and a first task on goal-less `smoke1022` with Claude Code pinned created the file, ran the
memory turn, wrote `orbital/instructions/project_goals.md` ("First draft … smoke-test the agent harness")
and opened the card. Not done: the downgrade half (opening this branch's data with the v0.21 installer),
which needs an admin install the owner was not present for. No half-way interview session existed in that
data dir, so that specific transition was not exercised.

## dsh ACP rows (tester D, dsh 0.2.2-rc.1 via OpenCode Go)

Rig: daemon `127.0.0.1:8342` (keyless, `C:\orbital-scratch\D\data`, `DSH_HOME=C:\orbital-scratch\dsh-home`), Vite 5176,
wizard done with every real folder unchecked. Claude Code and Codex also enabled.

- 167-W3-dsh PASS — empty project pinned dsh: `sessions: []`, no greeting (「正在与 DeepSeek Harness 对话…」); first task created hello.txt and answered DONE; the memory turn carried "This project has no goals yet…" and dsh answered "Recorded."; `project_goals.md` "# First draft" describes the scratch workspace (no orbital/bin, MCP, dispatch); card 251134 这样理解对吗？ `options: 对 | 需要改`; tapping 对 → `inject {ask_id}`, one `import_scan_confirmed` row, `- done 251134 … by:user 对`, transcript unchanged (no run).
- 167-W4-dsh PASS — folder with files, no Scan card; dsh added `remove()` to groc.py (3 min, mostly its own sandbox repair, below); memory turn drafted goals from folder + conversation and opened card 6a968d; the correction typed as 「需要改：这只是我学 Python 的练习…」 → `inject {ask_id}`, exactly one dsh turn, 「已经更新，只动了 orbital/ 下的文件」; mtime diff: nothing outside `orbital\` changed, goals got a `supersedes:` line.
- 167-W7-dsh PASS — dsh listed the workspace (`Get-ChildItem -Force`); two starts 5 s apart answered in 21 s / 6 s, no stall, no prompt.
- B3-dsh PASS — Workbench 自动 send on a dsh-pinned project: pending card at once, `new-session` → `inject {target: dsh}`, PONG in 4 s, card with the DeepSeek Harness logo, counted in 「6 件等你」.
- B6-dsh PASS — upload → new-session → inject with `attachments`, no `/workbench/route`; dsh answered the file's first line (KIWI-77).
- B7-dsh PASS — Quick Tasks + DeepSeek Harness chip: PONG in 3 s, dsh session dir shows cwd = the Quick Tasks folder; no manager / missing-key line.
- C1-dsh PASS (the Mac's dsh FAILED this) — dsh appended `- open Which one should I go with: red or blue? options: red | blue` to ASKS.md and stopped; card with pills and the dsh logo; tapping red → `inject {ask_id, ask_answer: red}`, `- done d0425a … by:user red`, same session continued ("red it is"), DECISIONS.md updated by the memory turn.
- C2-dsh PASS — card reply → next turn of the same session (same native dsh session dir), card result PING; ✓ dropped it from 「8 件等你」 to 7; opening a session cleared its card.
- C4-dsh PASS — killed the session's `DeepSeek Harness.exe` 30 s into `Start-Sleep 90`: row "dsh stopped with error: Error: Connection closed", card 「w7-dsh · 已中断」, left-off card 「未完成 · DeepSeek Harness · 已中断」; a send with `target: claude-code` into the same session got the recap ("You had asked it to run Start-Sleep -Seconds 90 and then reply DONE").
- C5-dsh PASS — every run got the `[Orbital memory]` turn (goals on first runs, PROJECT_STATE / DECISIONS later, "nothing to record" otherwise); `last_reply_at` stays on the real reply so the card did not turn unread again; no memory bubble in the chat.
- **D3 PASS, both directions** — dsh seat → Codex: `POST /mcp` initialize/initialized, `ListToolsRequest`, `CallToolRequest`, `Message sent to codex: "Reply with the single word PONG…"`, dsh's turn ended at once, Codex `completed. Summary: PONG`, `worker_result` row, dsh's next turn "Codex's reply came back as exactly the single word: PONG". Claude Code seat → dsh child: `CallToolRequest`, `Message sent to dsh`, dsh PONG, Claude's follow-up "dsh replied as asked"; the depth-1 dsh child logged no `ListToolsRequest` (no tool, as designed).
- 166-5-dsh BLOCKED — dsh 0.2.2-rc.1 advertises no commands: `skill-catalog?worker=dsh` → `[]` after 12 spawns, `worker-commands.json` has no dsh entry, and a raw ACP probe (`initialize` + `session/new`, 25 s listen) received no `session/update` at all, so no `available_commands_update`. Nothing to pick or persist.
- C7-dsh N/A (no imported dsh history).

Findings from the dsh pass: (plain, fixed below) the empty-state left-off card quoted the ask-answer wire text `[Answer to ask d0425a, already closed: "…"] red`; `POST /projects` silently ignored `default_pinned_target` (UI unaffected); the 对，做第 1 步 option was offered for a step that reads "Wait for the user's actual request". (product, decisions §A19–21) idle worker processes are never reaped while the daemon runs (10 dsh processes alive, one per session that ever ran); dsh's own Windows sandbox fails its first shell call (`SetNamedSecurityInfoW … grantWrite`) and then runs an unconfined ACL repair that leaves `acl-recovery\` folders (one outside the workspace) and new deny ACEs on the user's folder; one `POST /mcp` 400 per worker spawn from the MCP SDK's stateless transport. The job-object `error 5` lines for `claude.exe` on this rig are explained in §A22 (the daemon ran inside the Claude Code harness's own job; a plain-console probe assigns `claude.exe` fine).

HOME after the dsh pass: the three Orbital files unchanged; `~/.dsh` exists since the first `dsh -V` probe (empty template, no key); `~/.dsh/profiles/acp/cordis.yml` was rewritten once at 10:15 by a `dsh --profile acp` launch without `DSH_HOME` (the upgrade-rig daemon, which had none; harmless: the file is dsh's empty template). Everything else dsh wrote is under `C:\orbital-scratch\dsh-home` and the projects' `orbital\sub_agents\dsh\dsh-sessions\`.

## Suites

- `pytest tests/unit tests/platform --timeout 120` (excluding the two `sandbox-exec` files): **34 failed /
  7055 passed / 249 skipped** on the tip before this pass's fixes, and **35 failed / 7056 passed** after them
  (the extra red is the timing-sensitive test below, since made deterministic). 31 of the 34 fail identically
  on `origin/main` in a clean worktree (symlink privilege `WinError 1314`, Windows path length in the pi/kimi
  fixtures, `test_version` reading stale installed metadata 0.18.0, `C:\` being a git work tree here). The other
  three are test-environment artifacts, explained in the decisions file §C. None is this batch's code.
- `tsc -b` clean; `vitest` 2761 → 2765 passed (new wizard tests, updated ChatView test); `check-i18n` 7 pre-existing missing-zh warnings.
- After the second pass, on the pushed tip `bfe99da`: `pytest` **33 failed / 7074 passed** (the two
  test-side artifacts are green now; the card-shape test gained the `change_option` key). 32 of the 33 are the
  pre-existing Windows reds listed above; the one other,
  `test_provider_integration::test_run_process_overwrites_terminates_old`, is flaky under load (passes on
  rerun), as is `test_history_sources_138::TestMiniMax::test_sub_sessions_and_archived_are_skipped` and the
  Vitest `SubAgentSettings` update-floor case (each failed once in a full run and passed alone, on untouched
  code). `tsc -b` clean; `vitest` **2774 passed / 178 files**; `check-i18n` 1555 keys, the same 7 pre-existing
  missing-zh warnings.
- `test_import_scan_160::test_a_scan_records_its_session_on_the_project` waited 50 bare event-loop ticks for a
  run that goes through `asyncio.to_thread` twice: passed in one full run, failed 3/3 alone on a quiet machine
  (also on the untouched `ca770dc` file). It now waits for the RUNNING state (commit below); 3/3 green after.

## Overall tally

| | PASS | PARTIAL | FAIL | BLOCKED |
|---|---|---|---|---|
| 168 (C) | 1, 2, 3, 4, 6, 7 | | (1, 2, 7 on the rig as briefed; fixed) | 5 |
| D (C) | 1, 2, 4, 5 | | | 3 |
| 166 (C) | 1, 2, 3, 4, 6, 7, 8 | | | 5 |
| 167 (A, B) | W2-on, W3-claude ×2, W3-codex, W4-codex, W5-claude, W5-codex, W5-signed-out, W7 | W2-off, W4-claude | W1 (fixed) | W6 |
| A / C (A) | A1, C1 ×2, C2 ×2, C3-codex, C4 ×2, C5 ×2, C6, C7 ×2 | A6 | C3-claude (same as Mac) | ACP rows |
| B / E (B) | B1–B9 (claude, codex), E1, E2 | E3 | | B3/B6/B7-acp, E4 |

Every FAIL that was a plain bug is fixed on the branch; the rest (C3-claude, the PARTIAL causes, E3's two
findings) is in `note-1110-decisions-windows.md`.

---

## 168 — Claude Code seat dispatch (tester C)

- 168-1 PASS — `proj_e8c340056448` (ws-login3), session `c-ws-login3_99879d64`, "fix the sign in bug, then have codex review it". Claude transcript: `ToolSearch select:mcp__orbital__dispatch` then `mcp__orbital__dispatch {"agent":"codex","task":"Please review a one-line bug fix …"}`; chat: one `Message sent to codex` row, Codex's review bubble, `codex completed`, `worker_result` hand-back, Claude's follow-up. "Other agents in this project" appears 0 times in `/chat`, the JSONL and both transcripts. No `codex exec`. (FAIL on the seat-less rig: `codex exec --sandbox read-only --skip-git-repo-check …` from Bash, recipe saved to MEMORY.md.)
- 168-2 PASS — ws-login4, 「修好登录的 bug，然后让 Codex 看一下」. `mcp__orbital__dispatch` called twice (first with `message` instead of `task`, validation error; second with `task`). One `Message sent to codex` row, Codex's zh review, hand-back, zh follow-up. Seat text 0 times. (Same FAIL on the seat-less rig.)
- 168-3 PASS — daemon stopped (pid 17872) and restarted (25012); 「再让 Codex 看一下 login.py…」 into the same session: `resumed prior session`, `mcp__orbital__dispatch` again, Codex's verdict, hand-back, follow-up.
- 168-4 PASS (= D2) — ws-codex, Codex pinned: `orbital.dispatch {"agent":"claude-code",…}`, `Message sent to claude-code`, Claude's review, hand-back, Codex's follow-up. `commandExecution` commands containing `claude`: only `Get-Content CLAUDE.md`; `claude -p` / `claude.exe` / `claude.CMD` 0 times.
- 168-5 BLOCKED — no Pi.
- 168-6 PASS (keyed) — ws-login-mgr, no pin: the manager called `agent_message send claude-code`, Claude fixed `=` → `==`; Claude's native session has no seat line, `mcp__orbital__*` 0 times, "Other agents" 0 times; the memory turn drafted the goals.
- 168-7 PASS (corrected rig) — every rollout started by it has `originator=orbital, source=vscode`; the only `originator=codex_exec, source=exec` rollouts under `C:\orbital-scratch\C\` are the two from the seat-less rig (ws-login1/2). The `skills/list` probes created no rollout.

## D — dispatch from the seat keyless (tester C)

- D1 PASS — ws-d, `mcp__orbital__dispatch {"agent":"codex","task":"Reply with the single word PONG…"}`; Codex replied PONG, `worker_result` row, Claude's next turn "Codex replied with exactly the single word requested" (as on the Mac, Claude does not repeat the word); no manager run, no missing-key error; `/workbench/feed` one card (claude-code); `POST /mcp` 200/202 per spawn (plus one 400 per Claude spawn before the handshake, from the MCP SDK's stateless transport; harmless).
- D2 PASS — see 168-4.
- D3 BLOCKED — no ACP agent.
- D4 PASS (note) — with the exact wording Claude read the tool schema and answered in words that there is no "Orbital" agent, without calling; asked explicitly it called `{"agent":"orbital"}`, got `Error: unknown agent 'orbital'` and relayed it verbatim. Nothing hung, no manager run.
- D5 PASS — `Start-Sleep 40` then PONG: the parent's turn ended at once, `sub-agents/status` claude-code `idle` / codex `running` 01:47:54–01:48:42, child transcript only `commandExecution pwsh -Command 'Start-Sleep 40'`, no dispatch tool in the child, `worker_result` row 10 of the parent JSONL, one parent turn after. No manager wake. Stop-on-parent not tried (nothing to stop, as on the Mac).

## 166 — worker commands in the `/` picker (tester C)

- 166-1 PASS — Claude pinned on ws-cmd with `.claude\commands\orbital-probe.md`: `skill-catalog?worker=claude-code` → 89 entries, all `source: worker`, incl. `{"name":"orbital-probe","invoke":"/orbital-probe ","hint":"word","description":"probe (project)"}`; log `Claude Code CLI: using native …\node_modules\@anthropic-ai\claude-code\bin\claude.exe instead of shim …\claude.CMD`; `worker-commands.json` in `C:\orbital-scratch\C\data\`, nothing new under `~/.orbital`. UI: `/` requests `?worker=claude-code`, filtering shows `orbital-probe · word · probe (project)` (`166-1-filtered.png`). The hint shows `word` without the `[ ]` of `argument-hint: [word]`.
- 166-2 PASS — Codex pinned: 25 entries as `$name `; a cold probe on a new project with `.agents\skills\orbital-pong\SKILL.md` listed `$orbital-pong ` (26, 0.6 s): `codex app-server` through `codex.CMD`, cwd = workspace, no child left, no rollout.
- 166-3 PASS — pick → composer `/orbital-probe ` caret 15/15, no chip; typed `hello`, Enter → `inject {"content":"/orbital-probe hello","target":"claude-code","pinned":true}`; transcript `[Using tool: Skill]`; reply `PONG-INT hello`.
- 166-4 PASS — pick → `$orbital-pong ` caret 14/14, no chip; `hi` → reply `SKILL-PONG hi`; the rollout has the `<skill><name>orbital-pong</name><path>…SKILL.md</path>` item (skill read). (The skill was added after the cold probe and appeared once Codex's live session republished.)
- 166-5 BLOCKED — no Kimi / Cursor.
- 166-6 PASS — project on `c:\orbital-scratch\C\ws-lower` (stored lowercase); after `?worker=claude-code` the json key is `C:\orbital-scratch\C\ws-lower` (`worker_commands.py:134` `os.path.realpath`); `get()` with `C:\…`, `c:\…` and `c:\orbital-scratch\c\WS-LOWER` all return the 88 commands; second request served from the store (0.1 s, no spawn).
- 166-7 PASS (by code) — both probes pass `creationflags=win_no_window_flags()` (`claude_models.py:183`, `codex_models.py:122`); the live transports too (`codex_transport.py:929/1452`, `install_sdk_no_window_spawn` `sdk_transport.py:85`); teardown via psutil spawns nothing.
- 166-8 PASS — keyless unpinned catalog on ws-cmd: only Orbital's (`orbital-pong`, `source: workspace`, 0 worker entries). Keyed, Orbital pin: UI requests `skill-catalog` with no `?worker=`, list shows only the built-in `automate`.

## 167 — work first (tester A, keyless, Claude Code + Codex)

- 167-W2-off PARTIAL — fresh data dir, wizard: Agents → Projects (10 rows; every real folder unchecked, only `C:\orbital-scratch\A\ws-a`, `ws-b` kept; payload `{"folders":[…ws-b, …ws-a],"groups":[],"lang":"zh","defer_scan":true}`, no rewrite needed) → key step: the sign beside Skip shows the coin glyph and "2" (`W2-12-skip-sign-crop.png`); hover: 「导入后，Orbital 会用 Claude Code 把这 2 个项目各读一遍，理出目标和现状。仅此一次，会用掉 Claude Code 的额度。」 with the switch 梳理这些项目 (on). Switched OFF (count hides, slash shows, `W2-07`); Skip sent `PUT /settings {"setup_api_key_skipped":true,"import_scan":false}`. `GET /settings`: `setup.api_key_skipped: true`, `llm.api_key_set: false`, `unscanned_imports: 0`, `import_scan_agent: claude-code`; both projects `import_scan: "off"` (still after two restarts). Log: only `import scan: key step skipped, 2 project(s) not read`; no 项目梳理 session. First message in ws-a → notes.md → `[Orbital memory]` turn → `project_goals.md` "## First draft" → `ask_opened 这样理解对吗？` in the session and on `/workbench/feed`. PARTIAL: the worker name is not beside Skip (only in the hover sentence; decisions §A4), and the card had three options because of the "None pending" step (fixed, `88c5436`).
- 167-W2-on — see tester B.
- 167-W3-claude PASS ×2 — w3-cc (seat-less) and w3-cc2 (with the seat): `sessions: []` until the send; empty state has no greeting (「发送消息，开启工作。」 + 「正在与 Claude Code 对话…」); send → session pinned claude-code, notes.md HELLO, memory turn wrote `project_goals.md` "## First draft", `ask_opened` in session and feed; tapping the card's yes option → `inject {ask_id, ask_answer}`, row `import_scan_confirmed`, ASKS `- done … by:user`, **no** SDKTransport.start after. w3-cc2 goals file: grep `orbital/bin|background|mcp|dispatch|seat|other agents` finds nothing.
- 167-W3-codex PASS — w3-cx: same flow; Codex's memory turn wrote `# First draft` (UTF-8 BOM, decisions §A11); card 对 | 需要改; 对 → confirmed row, no `turn/start` after. Goals grep clean.
- 167-W4-claude PARTIAL — w4-cc (calc.py bug, README, todo.txt, not imported): no Scan/Skip card; first task fixed calc.py; memory turn drafted goals from folder + chat; card opened. Tapping 需要改 sends a bare "需要改" at once (`TaskCard.tsx:221`), the worker asks what is wrong, the correction is a second turn (decisions §A2); that turn changed only `orbital\instructions\project_goals.md` (+ DECISIONS.md by the memory turn); calc.py untouched.
- 167-W4-codex PASS (+ 167-W1 location half) — `D:\orbital-scratch-A\ws-d`: no Scan card; Codex listed `orbital/, AGENTS.md, calc.py, README.md, todo.txt` and fixed calc.py; goals drafted at `D:\orbital-scratch-A\ws-d\orbital\instructions\project_goals.md`; correction typed into the card's reply field → one worker turn (`turn/start` 01:53:35) + memory turn; files changed only under `orbital\`.
- 167-W1 FAIL → fixed (`1d75d9e`) — the first-run path is relative (`_GOALS_REL`) and landed at `<ws>\orbital\instructions\project_goals.md` (no cwd guess); the scan instruction's Stage 3 line was `D:\orbital-scratch-A\ws-d/orbital/instructions/project_goals.md` (mixed separators) while the preamble said `D:/…`. After the fix both lines render `D:/x/ws/orbital/instructions/project_goals.md`. Not re-run as a live D:\ import scan (none in A's scope).
- 167-W6 BLOCKED — the only v0.21-era data dir here is the real app's `%APPDATA%\Orbital` (port 8000, real projects); not touched, no installer downloaded (decisions §B16).
- 167-W7 PASS — Codex listed the workspace during its run; back-to-back starts (ws-d ×2 5 s apart, plus w3-cx and w3-cc2) all answered PONG; no permission prompt, no stall, no lock error.

## A / C — keyless entry, cards, asks, follow-ups (tester A)

- A1 PASS — across two restarts: no wizard; `setup.api_key_skipped: true`, `llm.api_key_set: false`; pin switcher's Orbital row with the 「需要 API 密钥」 pill; clicking it opens 全局设置 at 模型服务 (`A1-after-click.png`).
- A6 PARTIAL — project with `disabled_sub_agents: ["claude-code","codex"]`: creation is 201 (nothing to fail since 167); the first send → `inject` 400 `{"code":"missing_api_key"}`; the UI shows 「项目和全局均未配置 API 密钥。…」 + 打开设置, not a spinner; but session `a6_3aabb82a` (user row + `start_failed`) stays in the list and the message names only the key (decisions §A3).
- C1-claude PASS — the note's exact wording was refused twice by the model (`API Error: … safeguards flagged this message`), surfaced as a readable stop, not a hang; reworded ("I still have to choose a colour for the notes header: red or blue. Don't pick for me. Record this open question…"): `- open Which colour should the notes header be: red or blue? options: red | blue`, `ask_opened` row, feed card with red/blue pills; tap red → `inject {ask_id, ask_answer:"red"}`, `- done ed1c2c … by:user red`, same session "Noted: the notes header colour is red."
- C1-codex PASS (W3: Codex writes ASKS.md) — exact wording → `- open Which color should we choose before continuing? options: red | blue`; pills; blue → `- done 0bff89 … blue`; "Blue it is."
- C2 PASS (claude, codex) — reply typed in the card's field → next turn of the same pinned session (Claude: live process reused; Codex: same thread `01a1c172-…`), result line updated, ✓ (`POST /workbench/checked`) removed only that card, opening the session also cleared it.
- C3-claude FAIL (same as the Mac) — `autonomy=supervised`, "Create hello.txt": created at once, no approval card. `sub_agent_manager.py:1296` forces HANDS_OFF (decisions §A1).
- C3-codex PASS (as the Mac) — Codex created the file without asking; note that Orbital does set Codex's policy (HANDS_OFF → `on-request` + auto-accept on Windows, `codex_transport.py:79–103`), so Supervised is ignored for Codex too.
- C4-claude PASS (better than the Mac) — killed the daemon's `claude.exe` child mid-`ping -n 90`: terminal row "claude-code stopped with error: Command failed with exit code 4294967295", feed `stop.cause: interrupted`, card 已中断, left-off card "Claude Code · 已中断" (`C4-cc-leftoff.png`); re-pinned to Codex → recap of the ping request. (The killed claude.exe's own pwsh/PING children survived; removed by hand.)
- C4-codex PASS — killed cmd → node → codex.exe mid-ping: "codex stopped with error: turn ended without a completion signal", `stop.cause: interrupted`; re-pinned to Claude Code → recap.
- C5 PASS (claude, codex) — `[Orbital memory]` row (`_kind: memory_record`) after each run; PROJECT_STATE gained entries; ✓ then 75 s past the memory turn: the card did not return (record rows excluded by `is_reply_row`, `session_index.py:172`); "nothing to record" present in `/chat` but hidden in the UI (`chatTransform.ts:1189`).
- C6 PASS — PONG to w3-cx and w3-cc2 3 s apart, then ws-d ×2 5 s apart: four distinct sessions and cards; no lock error.
- C7 PASS (claude, codex) — virtual rows in wizard-imported ws-b: "What code word did I tell you?" → `SDKTransport.start … resume=True`, `materialized: true`, "…BIRCH."; Codex: `session import: materialized imp_codex_…`, `turn/start thread=01a1c15f-…`, "ASPEN."; feed cards with the right logo. No Codex automation exists here.
- Known #1 (recap before user text, repeated message): same on Windows (Codex rollout: recap first, user text at offset 1113, the question twice).
- Known #2 (two sessions of one worker share one transcript): same (all ws-d codex sessions write `sub_agents\codex\5d693dac.jsonl`); likely why a memory turn for w3-cc2 said "Files this turn changed: none" despite a `Write …\hello.txt` row with that session's `dispatch_id`.
- Known #3, #4: not exercisable on the keyless switch-off rig.

## 167-W2 (switch on) and 167-W5 — wizard and sidebar imports (tester B)

Rig note: B seeded its scratch folders with non-probe one-liners ("Read README.md and tell me in one short
sentence what this project is.") because "Reply with just OK" is filtered as a probe (decisions §B18); all six
folders then listed in the wizard and the real UI payload was sent, no rewrite. Case 1 and the claude-code
half of case 2 ran on the seat-less daemon (import scans do not need dispatch; results stand).

- 167-W2-on PASS — wizard (zh): Agents → Projects (10 candidates; every real folder and the other testers' unchecked; `ws-w1..3` kept; payload `{"folders":[…],"lang":"zh","defer_scan":true}`) → key step: two signs, the one beside Skip `3` + 「暂时跳过，使用已有 Agent」; hover 「导入后，Orbital 会用 Claude Code 把这 3 个项目各读一遍，理出目标和现状。仅此一次，会用掉 Claude Code 的额度。」 with 梳理这些项目 `role=switch aria-checked=true` (`W2-keystep*.png`). Skip → `PUT /settings {"setup_api_key_skipped":true,"import_scan":true}`; log `import scan: key step skipped, 3 project(s) to read with the user's agent`; ws-w3 01:35:49→01:36:20, ws-w1 01:35:50→01:36:17, ws-w2 01:36:19 (after ws-w1)→01:36:41: max 2 at once; each `Import scan dispatched directly to 'claude-code'`, session 项目梳理 pinned claude-code `origin: cold_start`, last row `ask_opened … import_scan_confirm:true 这样理解对吗？`; Workbench cards with 对，做第 1 步 | 对，先不做 | 需要改 and the steps as detail; worker prompt path `C:/orbital-scratch/B/ws-w1` (167-W1 pass on `C:`). Each project `import_scan: done` with `import_scan_session` / `import_scan_ask`. Observation: the scan's summary/steps came back in English for ws-w1/ws-w2 despite `import_scan_lang: zh` (model output; the card chrome is zh).
- 167-W5-claude PASS — sidebar 从其他 Agent 导入 → `ws-s1..3` only; sign `3` + the Claude Code sentence, switch on; `POST /onboarding/import {"folders":[…],"lang":"zh"}`; ws-s3 01:41:05→01:41:32, ws-s2 01:41:05→01:41:41, ws-s1 01:41:34→01:42:06 (≤2 at once); three 项目梳理 sessions pinned claude-code, cards unread (`checked_reply_at: null`, "6 件等你") with the three options. 对，先不做 (ws-s1): `import_scan_confirmed` user row, no dispatch, no assistant row, card gone. 需要改 (ws-s2): exactly one turn (`Message sent` with `<orbital_turn_note>` → one `completed`); files changed only under `orbital\`. 对，做第 1 步 (ws-s3): one turn (`<orbital_picked_step>`), fixed `auth.py`, one `completed`. (Desktop UI: the option pills show on hover only.)
- 167-W5-codex PASS — the daemon picks the scan agent, so Codex was reached by hiding Claude Code (`PUT /settings {"sub_agents_hidden":["claude-code"]}` → the new project got `disabled_sub_agents:["claude-code"]`): sign 「…会用 Codex 把这 1 个项目各读一遍…」; `Import scan dispatched directly to 'codex'` 01:47:55→01:48:38; card a96a40 with the three options; `project_goals.md` written; 对，先不做 → confirmed row, no run. (Codex's summary text English despite zh.)
- 167-W5-signed-out PASS — simulated without touching real credentials: the daemon restarted with `CLAUDE_CONFIG_DIR=C:/orbital-scratch/B/fake-claude-cfg` holding a fake `.credentials.json` (bogus OAuth tokens, future expiry), so the probe says `loggedIn: true` (ready, like a real expired login) and every run fails `Failed to authenticate: OAuth session expired and could not be refreshed`. Import of `ws-x1`: `Import scan dispatched directly to 'claude-code'` 01:49:40, `stopped with error … Failed to authenticate` 01:49:44, `import scan: proj_… failed; its first real run drafts the goals`; the session stays (项目梳理, pinned claude-code, last row `sub_agent_terminal kind:error`); Workbench `stop.cause: network_auth`, card 「项目梳理 / ws-x1 · 网络或登录出错」 (`W5-signedout-card.png`); `import_scan: failed`. Not re-run after `PUT /settings {"sub_agents_hidden":[],"import_scan":true}` (no scan line, still `failed`), not after E1's card save, not after a daemon restart. First real message ("Fix the bug in calc.py…") on the normal daemon: claude-code fixed `calc.py`, the `[Orbital memory]` turn carried "This project has no goals yet. First, draft…", wrote `project_goals.md` ("## First draft … small Python scratch project", no Orbital rules) and `PROJECT_STATE.md`, then `ask_opened … 这样理解对吗？`. The cause wording is the classifier's 网络或登录出错, not a specific "signed out".

## B — Workbench composer keyless (tester B; daemon on `2ad4f1a`)

- B1 PASS — fresh profile: `.wb-greeting` present but empty (`aria-hidden`); composer top at 43% of an 860 px viewport (Mac ≈ 38%); chips 「项目: 快速任务 | Agent: 自动」 (`B1-home.png`); after sends the profile reopens on the last project sent to.
- B2 PASS — project picker: 自动 row 「自动 / 需要 API Key」 `aria-disabled="true"`; click → Global Settings at 模型服务 (「还没有模型服务」); zero non-GET requests; other rows pick normally.
- B3-claude PASS — ws-w1 (sticky claude-code), 自动, PONG: `new-session` → `inject {target:"claude-code",pinned:true}`; card at once 「ws-w1 · 正在开始…」 (`data-dot=run`) then 「ws-w1 · PONG」 with the Claude mark; `pinned_target: claude-code`; summary 「9 件等你」.
- B3-codex PASS — ws-w2 (sticky codex): `target:"codex"`, PONG, Codex mark. B3-acp BLOCKED.
- B4 PASS — ws-w1, picked Codex: session `pinned_target: codex`; `default_pinned_target` still claude-code; next 自动 send went to claude-code.
- B5 PASS (7423b9c) — agent picker: 「Orbital / 需要 API Key」 `aria-disabled`; click opens Global Settings; no request, no session minted.
- B6-claude PASS — attach.txt (BANANA-42): `POST /projects/{pid}/files/upload?path=/uploads/` → `new-session` → `inject {attachments:[{path:"uploads/…"}], target:"claude-code"}`; no `/workbench/route`; reply names BANANA-42 (the card's result line shows only the reply's first line, cosmetic). B6-codex PASS, same order. B6-acp BLOCKED; desktop drag-by-path not run (web).
- B7-claude PASS — Quick Tasks + 自动: `target:"claude-code"`, PONG, cwd = `data1\scratch` (`~/.claude/projects/C--orbital-scratch-B-data1-scratch`). B7-codex PASS (rollout `cwd: C:\orbital-scratch\B\data1\scratch`). No manager / missing-key line. B7-acp BLOCKED.
- B8 PASS (as amended: hidden, never sent) — status line 「还没有自动任务」 without the `/automate` hint; `/` menu (85 rows on Claude Code, 25 on Codex) has no `automate`; `/auto` shows only Claude's `auto-mode-setup`, `autocompact`; `skill-catalog?worker=claude-code` has no `automate`; `triggers` → `[]`. Product note: the same keyless `/` list offers Claude Code's own `/schedule` (cloud routines; the command behind the Mac's B8 side effect) and `/loop` (decisions §A12).
- B9 PASS — `{"text":"…PONG"}` → 200 `{project_id: Quick Tasks, agent:"claude-code", by:"fallback", reason:"no API key"}`, PONG; `{"agent":"gemini-cli"}` → 400 "not available in this project"; `{"agent":"orbital"}` → 400 `missing_api_key`, Quick Tasks session count unchanged (no orphan, 7423b9c).

## E — transitions (tester B)

- E1 PASS — greyed 自动 → Global Settings → 添加模型服务: OpenCode Go (base URL prefilled `https://opencode.ai/zen/go/v1`), model `deepseek-v4.1-flash` typed (the model list came back empty; accepted), Save disabled until 测试连接 → 「已连接到 OpenCode Go，使用 deepseek-v4.1-flash」; Save → `POST /settings/cards` 201 (`E1-tested.png`, `E1-after-save.png`); `api_key_set: true`. Workbench: greeting back (「说事就行，我来安排」), chips 自动 | 自动, project 自动 selectable, agent picker 「Orbital / deepseek-v4.1-flash」; pin switcher Orbital row 「管理者 ✓」 with no pill, selecting it PATCHes `pinned_target:null`. Sticky worker pin kept (ws-w1 自动 → claude-code). No re-scan after the save (ws-w1/2/3 `done`, ws-x1 `failed`). New sidebar import ws-e1: sign 「…会用这个 Key 把这 1 个项目各读一遍…会消耗 token。」, log `reading proj_… with orbital (session ws-e1_…)` → done, confirm card; rows `source: management`.
- E2 PASS — with a manager essay in flight: `DELETE /settings/cards/{id}` → 400 `card_is_default`; `DELETE /settings/api-key` → 200; the in-flight turn finished normally 10 s later (+ its "nothing to record" memory turn), no hang; `api_key_set: false`; Workbench keyless again (empty greeting, chip on the last project, 自动 「需要 API Key」, agent picker Orbital 「需要 API Key」); pin switcher Orbital row 「OpenCode Go · deepseek-v4.1-flash / 需要 API 密钥」 → Global Settings; `POST /workbench/tasks {agent:"orbital"}` → 400, session count unchanged; UI send in the existing Orbital session → `inject` 400 with the inline alert 「项目和全局均未配置 API 密钥。…」. The refused message stayed as an unanswered user row (fixed, `b481a11`).
- E3 PARTIAL — run 1 (restart during a FIRST turn): codex `Start-Sleep 90` (thread `01a1c191-…`) + a keyless import scan of ws-r1 reading on claude-code; daemon killed 02:28:20, restarted. Rows intact. Import scan resumed with ONE 项目梳理 session (`Removed orphaned session lock sentinel`, `import scan: resume queued 1 project(s)`, `removed session ws-r1_aa983f8b`, `minted … ws-r1_b425cf4d`, done, confirm card; 596995d verified). Native thread NOT resumed: the next send was "fresh session — first spawn", new thread `01a1c192-…` (the recap carried the earlier text, so the reply still named `Start-Sleep 90`); decisions §A13. Orphan: `claude.exe 11176` (the running scan worker, ppid = dead daemon) kept working ≈55 s and wrote `PROJECT_STATE.md`, `INDEX.md`, `project_goals.md` while the resumed scan wrote the same files (last writer won; two scans ran on one project); decisions §A14. Run 2 (restart during a LATER turn): rows intact; next send "resumed prior session", `turn/start thread=01a1c192-…` (same thread), answered the earlier question; 0 orphans.
- E4 BLOCKED — no phone.

Other B observations: after the wizard and after each sidebar import the SPA sends `POST /agents/{landing project}/new-session {}` whose minted id never materialises (harmless); the Windows sandbox banner 「尚未配置Agent沙箱…」 shows on every page (no sandbox password on this machine; unrelated); the first-journey tour overlay covers the composer on a fresh profile until skipped.

## Differences from the Mac (all testers)

- The Mac's launcher exported the port; the isolated launcher here did not (fixed). Once set, 168/D match the Mac exactly.
- C4-claude: Windows shows `stop.cause: interrupted` / 已中断 and a left-off card where the Mac showed the raw `-9` text (the 1109 fix at work).
- Scratch folders imported through the real wizard UI once their native history used a non-probe first message (`_PROBE_RE` filters "Reply with just OK"; decisions §B18); the Mac had to rewrite the payload.
- Codex on Windows writes files with PowerShell `Set-Content`: UTF-8 BOM on the goals file; the changed-files hint misses those writes (decisions §A5, §A11).
- Codex rollouts: `originator=orbital, source=vscode` (app-server) vs `originator=codex_exec, source=exec`; `commandExecution` rows appear twice (start + completion).
- D4: Claude declined in words before calling the tool; the Mac's Claude called it at once.
- 166: Claude cold probe 89 commands (Mac 81), Codex 25 (same); npm `claude.CMD` is swapped for the bundled `claude.exe`; `codex.CMD` runs directly; `realpath` normalises drive-letter case.
- The model refused C1's exact wording with a safeguards error (model side; surfaced as a readable stop).
- Codex hands-off on Windows is `on-request` + auto-accept by design; no permission stall in back-to-back starts.

## Observations (no action taken)

- HMR: the coordinator's edit to `SetupWizard.tsx` during A's wizard remounted it to step 1 (the import had already posted; re-entering showed both folders as imported).
- The daemon log says "import scan: confirm card … posted" for a first-run goals card on a non-imported project (wording only).
- Projects created by API with a forward-slash workspace show mixed transcript paths (`C:/…/w3-cx\orbital\sub_agents…`); the UI sends backslashes.
- The left-off card counts "23 个未提交的文件" for a non-repo because `C:\` is a git work tree on this machine (environment).
- First `POST /mcp` per Claude spawn is a 400 before the 200/202 handshake (MCP SDK stateless transport; dispatch works).

## HOME check

Before (01:2x) and after (end of run), unchanged on every rig:
```
2026-11-03 00:16:33.572604000 +0800 /c/Users/qiren/.orbital/sub_agent_config.json
2026-05-26 23:49:09.927632100 +0800 /c/Users/qiren/orbital/user_preferences.md
~/orbital/user_memory.md: does not exist (before and after)
ls -la ~/.orbital/ -> sub_agent_config.json only (182 bytes)   [no worker-commands.json]
```
`sub_agent_config.json` holds `claude-code: {effort: medium}` and the manual `dsh` binary entry; no worktree
or scratch paths. `worker-commands.json` exists only under each test data dir. No real folder was imported
on any rig (the wizard's real-folder rows were unchecked every time; A guarded the import request and
would have aborted on any non-scratch path). `~/.codex/sessions` gained only Orbital-started rollouts
plus the two `codex exec` ones from the seat-less rig.
