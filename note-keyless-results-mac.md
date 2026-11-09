# Keyless x Workbench: Mac results

- Commit under test: 5e69b77 (batch/1109), worktree clean before and after the run.
- Date: 2026-11-09. Run dir: /Users/keanezhou/orbital-wt/run/1109-keyless/ (data/, ws-*, shots/, out-*.txt, pw/ scripts left in place).
- Daemon (keyless, port 8590, fresh data dir, Vite 5590 proxied to it):
  `PYTHONPATH=.../memkeyring PYTHON_KEYRING_BACKEND=memkeyring.MemoryKeyring AGENT_OS_PORT=8590 .venv/bin/python run/dev-daemon.py --worktree .../1109-batch --port 8590 --data-dir .../run/1109-keyless/data`
  No AGENT_OS_API_KEY. user_memory_path was PUT to a scratch file (accepted; confirmed by GET).
- Agents ready on /agents/available: built-in, claude-code, codex, cursor, dsh, kimi-code, minimax-code, pi (installed, not used). Not ready: codebuddy, gemini-cli.
  Real turns: claude-code and codex. ACP agents refuse turns (minimax: insufficient balance 1008; kimi: 403 plan; cursor: "Upgrade your plan", RetriableError).
- Browser: the shared Playwright MCP was held by another session. I drove the UI with an isolated playwright-core Chromium (scripts in run dir pw/). Screenshots named by case in shots/.
- Rig deviations (all fail-closed):
  - The wizard's Projects step and the sidebar Import dialog list only REAL folders (the history scan ran at daemon start and did not include my scratch folders). I drove the real UI but intercepted POST /onboarding/import and rewrote `folders` to my scratch folders only (a stub box was ticked, every real folder unchecked; any rewrite error aborted the request). The UI really sent a real folder in that payload each time (ws-a..c wizard: `.../1107-attach/probe/cwd`; sidebar imports: `/Users/keanezhou/Desktop/orbital-test`); the rewrite replaced it. I verified afterwards: 16 projects, none outside the run dir, no orbital/ created in either real folder.
  - Wizard step state is not persisted: a reload mid-wizard restarts at step 1. After "Skip for now" on step 3 the wizard is done.
  - B8 had a real side effect on the owner's Claude account (a cloud routine was created); see B8.

Per-case verdicts are below; the FAIL / PARTIAL / BLOCKED lists at the end are the summary.

## A. Entry and first project

- A1 PASS. After Skip the reload shows no wizard (shots/A1-reload.png). Daemon restarted once (kill + relaunch, same data dir): `GET /settings` -> `setup.api_key_skipped: true`, `llm.api_key_set: false`. Pin switcher shows Orbital row with orange "API key required" pill; clicking it lands on Global Settings at LLM Providers (shots/A1-pin-switcher.png, A1-after-pill, B2-after-click-auto.png). Not separately read: the keyless wording of the Global Settings key form (the page shows "No providers yet. Add one...").
- A2 PARTIAL (W4 held, scan card did not). Via the wizard: `orbital/` exists on disk in ws-a/b/c right after (only `history/` inside until first use; `orbital/sessions` etc. appear on first send). Projects have `import_scan: off`. No session titled 项目梳理/Project overview in any. Log: `import scan: key step skipped, 3 project(s) not read` (no "reading" lines). Divergence: the project's empty chat shows NO first-use Scan card, only the generic empty state with example prompts. Cause in code: `ChatView.tsx` ~3835 hides the cold-start card when `project.imported` ("Spec 153: an imported project skips the scan offer"), and also when the folder has no files (`is_empty_workspace`). The note's "first-use Scan card" does not exist for imported projects in this build (shots/A2-wsa.png, A2-wsa-with-file.png). The imported virtual sessions "Reply with just OK" are listed (claude + codex per folder).
- A3-claude PASS (with a flow difference). `POST /agents/{pid}/start-onboarding` -> 201 `runner: "claude-code"`, session a3-claude_56ddab75. First reply is the interview (3 questions), stops, no files written, no shell (out-A3c-1.txt). After my 2-line answer the worker did NOT write files: it replied with a restatement and said "say ok and I'll write this down". After "ok": `ws-d/orbital/instructions/project_goals.md` (workspace-absolute path), PROJECT_STATE.md, INDEX.md, DECISIONS.md written; reply proposes a first step ("say go") and waits. One SDKTransport.start in the log for 4 messages (same worker, same native thread afcc6bbe), no re-spawn. The "writes goals right after the answer" in the note is really "after the confirm".
- A3-codex PASS. start-onboarding 201 `runner: "codex"`. Same interview then confirm flow; goals/state/index written after "ok"; CodexTransport turn/start on one thread (01a1bff6-...) for every turn. One memory-record turn answered "(no output)".
- A3-acp BLOCKED (ACP agents refuse turns).
- A4-claude PASS (note differs). Non-imported project with files (ws-g, a4-scan), UI Scan card shown, click -> `POST /agents/{pid}/cold-start-scan` 201 `runner: "claude-code"`. Reply summarises, drafts goals, asks three questions and says "I'll write nothing until you confirm". No project files edited. project_goals.md / PROJECT_STATE.md / INDEX.md were NOT written by the scan (the note expects them written; the Scan card copy says "nothing is saved until you approve"). On an imported project the Scan card is not offered at all (see A2); I did not call cold-start-scan on an imported project.
- A4-codex PASS. cold-start-scan 201 `runner: "codex"` on ws-i; reply ends "Does this draft match your intent?", "Nothing has been written".
- A4-acp BLOCKED.
- A5-claude PASS. Sidebar menu -> "Import projects from your agents". Sign beside Import: tooltip "Importing lets Claude Code read these 7 projects once ... uses your Claude Code quota" (shots/A5-sign-hover.png). After Import: session "Project overview" pinned claude-code, log `reading ... with worker`. Three imports at once: log shows 2 started at 19:43:24, the third started at 19:43:43 after one finished (max two at once). Each finished with a daemon `ask_card` "Did I get this right?" with options `Yes, do step 1 | Yes, not now | Needs changes`; Workbench shows the card unread (orange) with the options on hover (shots/A5-card.png).
  - "Yes, not now" (ws-e): ask closed (`- done a77bf0 ... by:user Yes, not now`), user row `import_scan_confirmed`, spawn count unchanged, JSONL has no new assistant row. PASS.
  - "Needs changes" (ws-j): clicking the option sends one worker turn at once ("Needs changes"), the worker answers "What did I get wrong?"; my correction typed into the card reply is a second turn. Files changed only under orbital/ (git status: AGENTS.md, README.md, orbital/ only). PASS, but there is no correction field before the first turn.
  - "Yes, do step 1" (ws-k): exactly one worker turn; it tried step 1 and asked for the missing topic instead of writing the intro. PASS.
- A5-codex / A5-acp BLOCKED: the daemon picked Claude Code as the import-scan agent (`import_scan_agent` in /settings); there is no way to choose another.
- A6 PASS (API), UI part not run. Project with every worker in `disabled_sub_agents` -> `POST start-onboarding` 400 `{"code":"missing_api_key"}`, `GET .../sessions` -> `[]`, no `orbital/sessions` dir created. I did not drive the create-project UI to read the message.
- A7 not applicable (Windows only).

## B. Workbench composer keyless

- B1 PASS. Fresh browser profile: no greeting, composer top at ~38% of the viewport, chip "Quick Tasks" (shots/B1-home-fresh.png). After a send to ws-b a reload shows chip ws-b (shots/B3-home-cards.png).
- B2 PASS. Project picker shows "Auto" greyed with "Needs an API key" (`aria-disabled`); a real mouse click opens Global Settings at LLM Providers; the only non-GET calls were the UI's own `PUT /settings/ui-language`, nothing sent. Other rows pick normally (shots/B2-picker.png, B2-after-click-auto.png).
- B3-claude PASS. UI send on ws-a: `POST new-session`, then `POST inject {target:"claude-code", pinned:true}`; card appears at once ("1 running"), Claude logo, then unread with "ws-a · PONG". Session `pinned_target: claude-code`. Summary line counts it ("2 waiting on you").
- B3-codex PASS. Same on ws-b, Codex logo, `pinned_target: codex`.
- B3-acp BLOCKED.
- B4 PASS. ws-a (sticky pin claude-code), picked Codex: session ws-a_865da172 `pinned_target: codex`; `GET /projects/{pid}` still `default_pinned_target: claude-code`; the next Auto send went to claude-code (ws-a_878f1d2e).
- B5 FAIL (W1 confirmed). The Orbital row in the agent chip is selectable and nothing marks it unavailable (shots/B5-agent-chip-open.png). Send on ws-c: `new-session` 200, `inject` (no target) -> 400 `{"code":"missing_api_key"}`. Session ws-c_f5f78d13 is left behind with JSONL rows: user text + `[The agent could not be started for this message — ProviderConfigError: No LLM API key configured... Your message was saved; send it again...]`. On the Workbench the card showed "ws-c · Starting..." with a green running dot, then disappeared on its own within seconds; no error text and no way back to the text (shots/B5-after-send.png, B5-home-later.png). The project's empty chat later lists it as a left-off card "Orbital · No reply". Daemon log has a ProviderConfigError traceback for the loop task.
- B6-claude PASS. UI: stage attach.txt, send. Network order: `POST /projects/{pid}/files/upload` -> `new-session` -> `inject` with `attachments`; no `/workbench/route` call. Worker answered "BANANA-42 is the first line". B6-codex PASS, same order and answer. B6-acp BLOCKED. Desktop drag-by-path not run (web only).
- B7-claude PASS. Quick Tasks chip, Auto: pinned claude-code, answered PONG, cwd = data/scratch (`~/.claude/projects/...-data-scratch` created). B7-codex PASS (picked Codex): rollout cwd = data/scratch. No manager or missing-key line in the log for these. B7-acp BLOCKED.
- B8 PASS on the criteria, FAIL on side effect. `/automate every day at 9 say hi` on ws-a went to claude-code as plain text, no manager run, no 400, `GET /projects/{pid}/triggers` stayed `[]`. The worker replied "There's no /automate command, so I used /schedule" and CREATED A REAL CLOUD ROUTINE on the owner's Claude account: "Daily 9am hi", id `trig_01KTrTogazcyXAZfmLgW6GSN`, runs daily 01:00 UTC from Nov 10, all account connectors auto-attached. Per the worker's own note it cannot be deleted by API: delete it at https://claude.ai/code/routines/trig_01KTrTogazcyXAZfmLgW6GSN. I did not delete it. Product answer for the note's question: yes, hide `/automate` when keyless (and the worker should not get a skill name it can resolve to a real account action).
- B9 PASS (first two), outcome recorded (third). `{"text":...}` -> 200 `by:"fallback"`, `reason:"no API key"`, Quick Tasks, `agent:"claude-code"`. `agent:"gemini-cli"` (not ready) -> 400 "not available in this project". `agent:"orbital"` -> 400 `missing_api_key`, but a session (quick_tasks_29ce6f5b, "PONG", idle, pinned null) was already minted: same orphan as B5.

## C. Cards, asks, follow-ups

- C1-claude PASS (pills not exercised). Worker appended `- open Decide: red or blue? ...` to orbital/ASKS.md, daemon stamped `open ceeee4 2026-11-09`; the Workbench card is orange with the question. My prompt did not make the worker write `options:` so there were no pills; answering through the card reply field sent `[Answer to ask ceeee4, already closed...] red` to the same session (ws-a_812aa28c), the log has `- done ceeee4 ... by:user red`, and the worker acknowledged and recorded the decision in DECISIONS.md (shots/C1-claude-home.png, C1-hover.png, C1-reply-after.png).
- C1-codex PASS (W3: Codex DID write ASKS.md). `- open User decision required...: red or blue?` stamped `7308b5`; card reply "blue" -> `- done 7308b5 ... by:user blue`, same session continued ("Blue it is."), PROJECT_STATE updated. Codex had no ASKS grammar header of its own but followed the seeded one. The ask had no `options:` here either.
- C1-acp BLOCKED.
- C2 PASS (claude, codex). Card reply field + Enter became the next turn of the same pinned session (resumed, same transcript). Opening a session in the chat clears its unread dot (shots/C5-chat.png). The card's ✓ itself was not clicked.
- C3-claude FAIL. Project `autonomy: "supervised"` (the settings backend field; the Autonomy radio in project settings maps to it) with Claude Code pinned: asked it to create hello.txt, it created it with no approval card and no pause (`ws-h/hello.txt` exists; session rows show a straight completion). Code: `sub_agent_manager.py` ~1179 forces `Autonomy.HANDS_OFF` for every sub-agent, so project autonomy never reaches a pinned worker. The Supervised setting is silently ignored keyless.
- C3-codex PASS. Codex created hello-codex.txt without a prompt (its own workspace-write sandbox); Orbital did not override it.
- C4-claude PARTIAL. Killed the worker process (SIGKILL on the claude child of my daemon) while a long command ran. Card shows the raw text "ws-c · Error: Command failed with exit code -9 (exit code: -9)" with a normal blue unread dot, not a "stopped / interrupted" cause (shots/C4-claude-home.png). Re-pinning to Codex and sending continued with a recap: the answer named the earlier `sleep 90` request. The empty-state left-off cards exist (screenshot shots/A1-pin-switcher.png shows "Didn't finish" cards with causes "Network or sign-in error", "Usage limit", "No reply").
- C4-codex PASS. Killed the codex app-server mid-run: JSONL row `codex stopped with error: turn ended without a completion signal`, card "ws-b · Stopped"; re-pin to Claude Code continued and recalled `sleep 80`. The "exhaust a limit" variant was not run.
- C4-acp BLOCKED. (Kimi/MiniMax stop causes did render as labelled cards: "Network or sign-in error", "Usage limit".)
- C5-claude PASS with one FAIL. After every worker run the daemon sent the `[Orbital memory]` message (JSONL rows `_kind: memory_record`), 2 to 4 s after the run ended, not 30 s. PROJECT_STATE.md gained entries (ws-a, ws-b, ws-d, ws-f). The chat shows no "nothing to record" bubble (shots/C5-chat.png) and the card did not turn unread again. FAIL: for sessions that have no user message (the interview and the Scan session, e.g. a4-scan), the Workbench card TITLE is the memory prompt text: "[Orbital memory] Your last turn is ending. Decide..." (shots/C5-memory-title-leak.png, A5-menu.png).
- C5-codex PASS. Same rows; the record turn's reply was "nothing to record" or "(no output)" (a bare "(no output)" row appears in the JSONL after some runs).
- C5-acp BLOCKED.
- C6 PASS. Two projects 1 s apart and two sends into ws-c 5 s apart: 4 distinct sessions, all answered PONG; no lock error and no `lock_wait` line in data/logs/daemon.log. Per-card ✓ not clicked.
- C7-claude PASS. Virtual row imp_claude-code_... in ws-a, sent "what was the very first thing I asked": response header "resumed prior session"; answer "reply with just OK". C7-codex PASS, same. The Workbench card logo check and a Codex automation firing were not run.

## D. Dispatch from the seat

- D1 PASS. Claude Code seat, "Ask Codex to reply PONG": transcript has `mcp__orbital__dispatch` (agent codex), a `Message sent to codex` row, then a `worker_result` user row, then Claude's next turn "Codex replied as asked". No manager run, no missing-key error. (Claude did not quote the word PONG itself; Codex's summary row says PONG.) Daemon log `POST /mcp` 200/202.
- D2 PASS. Codex seat -> Claude Code: Codex transcript has `orbital.dispatch`, child ran, `worker_result` row, Codex said "Claude Code replied exactly as requested". Codex shell commands only read CLAUDE.md/AGENTS.md; no `claude -p` command.
- D3 BLOCKED (call), mount verified. Minimax, Kimi and Cursor were pinned and sent the request: minimax and kimi refused before listing tools; each spawn made POST /mcp initialize (200) + initialized (202); for Cursor the log shows `ListToolsRequest` processed, so tools/list succeeds. No ACP model called `dispatch` (they refused or errored).
- D4 PASS. Claude seat dispatch to Orbital: tool returned `Error: unknown agent 'orbital'` and the worker relayed it verbatim; nothing hung; no manager run. Note: the error's list of valid ids includes codebuddy and gemini-cli, which are not ready.
- D5 PARTIAL. The parent's turn ends the moment the dispatch returns (summary row "I've sent the task..."), so there is no running parent turn to press Stop on: `POST /agents/{pid}/cancel` -> `{"status":"no_agent"}`, `.../sub-agents/claude-code/stop` -> 409 "No active session". The child ran to its end (`sleep 40`, DONE). Its transcript has only `commandExecution` (no dispatch tool). Its result landed as a `worker_result` user row and the parent got a normal next turn (it woke: "I checked Codex's session log..."). The "stop the parent while the child runs" case could not be exercised because there is nothing to stop; the Stop button in the UI was not tried.

## E. Transitions

- E1 PASS. Created a card via POST /settings/cards (opencode-go, deepseek-v4.1-flash; test ok). Workbench now shows "Drop it here, it finds its place.", chip "Auto" (shots/E1-home.png). Orbital row works: `POST /workbench/tasks {agent:"orbital"}` ran the manager loop, replied PONG. ws-a/ws-b keep `import_scan: off` and sticky worker pins (`effective_pinned_target` claude-code/codex). A new sidebar import (ws-n) was scanned by Orbital: log `reading ... with orbital`, confirm card posted. The pin switcher Orbital row working was proven through the API, not by clicking it.
- E2 PARTIAL. The sole default card cannot be deleted: `DELETE /settings/cards/{id}` -> 400 `card_is_default` "Make another card the default first." So "delete the default card" is not reachable with one card. I removed the key instead (`DELETE /settings/api-key` -> `api_key_set:false`) with a manager essay in flight. The in-flight turn finished normally (the request was already out); it did not end with an error. After removal the Workbench is keyless again: no greeting; the new card stays in the list (key removed, still named in the Orbital row). A new Orbital send then fails as in B5. The "readable error for an in-flight manager session" part was not observed.
- E3 PASS (with two notes). Killed the daemon (SIGTERM) with a codex `sleep 60` run and an import scan running, relaunched on the same data dir. Session reopened with rows intact (status idle), a new send resumed the native thread (answer named the sleep 60 request). Import scan: log `import scan: resume queued 1 project(s)` and a new "Project overview" session ws-m_48b0489c finished and posted its card. Notes: (1) the abandoned first scan session ws-m_d238187e stays, so the project has two sessions titled Project overview; (2) after the SIGTERM an orphan `claude` worker process (PID 73584, parent 1) survived; I killed it. No other orphans found at the end.
- E4 not run (phone).

## Other observations
- Pin switcher and agent chip list Pi although the wizard unchecked it and the projects carry `disabled_sub_agents: ["pi"]`.
- Many worker processes stay alive (one claude or codex app-server per session, about 20 at peak).
- Workbench card title for the dead B5 session and the empty-project left-off cards are reasonable ("No reply").
- With the key removed the Orbital row still shows the old card name ("e1-card").

## FAIL / PARTIAL / BLOCKED list
- FAIL: B5 (Orbital agent chip keyless: dead session, card vanishes, no message), B9-orbital (orphan session), C3-claude (Supervised ignored), C5 title leak (memory prompt as card title), A2 scan card missing on imported projects, B8 side effect (real cloud routine), E2 delete-default-card unreachable.
- PARTIAL: A4 files not written (by copy), C4-claude (raw error text), D5, E2, A2.
- BLOCKED: every ACP case (A3, B3, B6, B7, C1, C4, C5 for ACP), A5-codex/acp, D3 call, A7, E4.

## HOME files
Before (home-mtimes-before.txt) and after are identical:

```
Nov  9 16:13:58 2026 /Users/keanezhou/.orbital/sub_agent_config.json
Nov  9 16:13:27 2026 /Users/keanezhou/orbital/user_memory.md
Nov  7 12:35:33 2026 /Users/keanezhou/orbital/user_preferences.md
```
All three are byte-identical to the backups. `sub_agent_config.json` holds only claude-code opus, codex gpt-6-astra, minimax-code MiniMax-M3 and no worktree or run paths. The scratch user_memory.md was never created.
