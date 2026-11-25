STATUS: NOT READY — the fixes made tonight (B7 import, B8 Codex thread names, B10 ChatGPT sign-in on Windows, themed title bar) work from the repo but are not yet verified on a CI installer

# Windows results — v0.25.0 release candidate (2026-11-26, 00:00–02:00 local)

Branch-only file, like `note.md`. Build under test: `Orbital-Setup-0.25.0.exe` from CI run
`45567085778` (commit `9c55fa86`), sha256 `472cd61c…5a20a43a` (matches the note; the artifact zip
matched GitHub's digest `3c3bb611…`). Settings → About read `Orbital v0.25.0`.

How it was driven: the installed app was started with
`WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS=--remote-debugging-port=9222`, and the real WebView2 window
was driven over CDP (clicks and typing in the actual app, not a browser on port 8000). Native
screenshots were taken from the screen. Evidence is in `evidence\1125\` (git-ignored) and
`C:\orbital-scratch\1125\evidence\`.

## What blocks the release

1. **B10 Sign in with ChatGPT cannot work on Windows in `9c55fa86`.** The consent completes, the
   token exchange, JWKS fetch and model list all return 200, then
   `POST /api/v2/providers/chatgpt/signin` answers 500 and the dialog shows only `连接失败 · 500`.
   Cause: Windows Credential Manager refuses a secret over 1280 characters (`CredWrite` error
   1783; measured here: 1280 fits, 1300 does not) and the token set is several thousand. Fixed in
   `fffa18b0`, round-tripped against the real vault from the repo, **not** yet run through the
   sign-in on a CI installer. So the bundled `jwt` / `cryptography` / `certifi` question is
   answered (they work), and the remaining unknown is everything after the card is stored.
2. **B7 still lists Orbital's own threads** (33 "Other agents in this project: …" and 14 recap-titled
   sessions across 38 folders, two of them in Recent). Fixed in `1b64ae31` + `33509d86`, checked
   from the repo against the same folders, not on a CI installer.
3. **B8 no Codex thread is ever named** with Codex 0.171 on Windows. Fixed in `69880045`, verified
   on a dev daemon, not on a CI installer.
4. **A1, downgrade leg**: published v0.24.1 installed over 0.25.0 still lists Antigravity. This
   cannot be fixed from this build (see A1). Owner's call whether it matters.

## Results

| Check | Result | What was seen |
|---|---|---|
| A1 upgrade over v0.24.1 | PASS | Manifests folder: no `gemini-cli.yaml`, `antigravity.yaml` present, every file dated by the new build. All 6 projects with the same session counts (1/1/26/3/3/1), the OpenCode-go card, language and settings intact. No Welcome page. "Gemini" appears nowhere in Settings. |
| A1 v0.24.1 back over it | **FAIL as written** | Same data opens (same counts, card). But v0.24.1 **does list Antigravity**: its installer has no `[InstallDelete]`, so `antigravity.yaml` (dated by 0.25.0) stays and `gemini-cli.yaml` comes back. The new `[InstallDelete]` only helps a downgrade *to* 0.25.0 or later. Not fixable without republishing v0.24.1. Question for the owner. |
| A1 finish on 0.25.0 | PASS | Reinstalled, same data, manifests clean again. |
| A1 aborted install | PASS | "No" at the UAC prompt: installer exit code 2, no log, registry still 0.24.1, old app starts and serves its projects. Also seen: a silent install while Orbital is running stops by itself ("Setup has detected that Orbital is currently running"), old app untouched. |
| A2 look D | PASS with findings | 96 screenshots: light/dark × 中文/English × 100 / 125 / 150 %, eight screens each. See "A2 detail". |
| A3 type scale | PASS with one finding | No clipped or overlapping text found by eye or by a DOM scan (scrollWidth/Height against clientWidth/Height on every text node's element) on any of the 96 screens. One layout defect in Settings dropdowns (finding 5). |
| A4 manager turn + workers | PASS | Manager read `notes.txt` and wrote `out.txt` (53 s). Pinned Codex and Claude Code each replied PONG. No console window during these turns, no 4xx/5xx or ERROR in `daemon.log`. |
| A5 status fold | PASS, oddities | The model wrote four `[STATUS: …]` tags (session file). `[STATUS` never appeared in the chat text or the DOM, sampled twice a second while running, nor after a reload. The fold showed one line that changed ("Reading notes.txt and todo.md · 2 步" → "Writing summary.md · 3 步") and opened to its three steps. Copy on all three bubbles carried no tag. Oddities: finding 6. |
| B1 composer | PASS, IME by simulation only | Empty box while running: Stop; with text: Send. Three messages typed during a turn became rows with 编辑 / 删除 / 停止并立即发送. Delete removed a row. The rows went out in order when the turn ended. Esc stopped a turn ("已被你停止"). Questions: finding 7. IME: a simulated composition confirm (keyCode 229) did not send; **the real Microsoft Pinyin IME was not typed by a person** (see not run). |
| B2 dispatch cards | PASS, one part not run | "Ask Codex to list the files…" showed a dispatch card (Codex icon, brief, 17s); opening it docks the worker's steps (its text and `commandExecution`). A worker's question ("Which fruit do you prefer, apple or pear?") arrived as a question card with apple / pear buttons; answering "pear" was passed on and `fruit.txt` was written. Finished card: one line with the brief and the duration, no "改了 N 个文件". Not run: manager staying quiet while pinned to another agent. New defect: finding 8. |
| B3 undo | PASS with notes | In `C:\orbital-scratch\1125\ws\撤销 测试 b3` (Chinese name, a space, backslash paths). Undo of the latest turn restored the edited `existing.txt`. "Undo since" the first message listed the text file, PNG and PDF as created; ticking them removed all three. The agent's next turn said `existing.txt` "was reverted". Delete, 8 s wait, re-create of `notes2.txt`: the sheet showed it as "改了 2 行" and undo brought the original content back. Notes: finding 9. |
| B4 usage limit | NOT RUN | No account at its limit (Codex showed "本周 · 剩余 66–67 %"). |
| B5a Codex signed out | PARTIAL | After a real `codex logout`: Welcome row shows 登录 and no check; Settings shows 未登录 + 登录; pin menu shows "需要登录". Sending to it: `codex 失败：Error: 'codex' is installed but not signed in — open Settings → Sub-agents` plus "发送消息失败。请重试。" (names the agent and the reason, never "no agent is enabled"). Sign-in from the row: the card turned 已登录 60 s later without a relaunch. Gaps: finding 10. "No ready agent" could not be set up (Claude Code and dsh are signed in). |
| B5b not-detected card | PASS for CodeBuddy | Card leads with 安装 `npm install -g @tencent-ai/codebuddy-code` and 复制. Antigravity's `irm … | iex` line not re-checked: agy is installed here. |
| B5c gap after Check again | PASS, original case not reproduced | After 重新检测 the CodeBuddy card stayed and no empty gap appeared (16 px between the chips and the import row). The 1124 gap needed a card to disappear after detection; nothing changed state tonight. |
| B5d sweep | PASS, console windows seen | 10 agents in 1.7–2.3 s per sweep (agy the slowest at 1.4–1.9 s), one at 5.9 s while agy updated itself 1.5.15 → 1.5.17. Console windows: finding 11. |
| B6 pin-menu models, fresh data | PASS | Fresh data, import unticked, Quick Tasks with 0 sessions: all four workers show a model line with a chevron. Picked `gpt-6.1-terra` for Codex; `daemon.log`: `turn/start … model=gpt-6.1-terra`; reply PONG; after an app restart the line still read `gpt-6.1-terra`. |
| B7 import | **FAIL → fixed** | See blocker 2 and "Fixes". Timing: Start with 38 scratch folders ticked → project list complete in about 5 s. Finding 3 of 1124 still there: 25 pending "这样理解对吗？ / Did I get this right?" cards on the Workbench after the import. Only scratch folders were imported (14 real folders unticked by hand). |
| B8 Codex thread names | **FAIL → fixed** | No "named thread" line; `thread/read` on Orbital's threads returned `name: None`. The turn itself worked. Older npm Codex forced by hand: not run. See "Fixes". |
| B9a newest Codex binary | PASS, override not run | Orbital uses the desktop copy (`%LOCALAPPDATA%\OpenAI\Codex\bin\…\codex.exe`, 0.171.0-alpha.13) over npm 0.162.1 and lists `gpt-6.2-sol`, `gpt-6.2-astra`, …. A path set by hand winning: not run. `daemon.log` warns on every Codex start: `codex 0.171.0 != pinned 0.162.1`. |
| B9b background agent | PASS | Claude Code launched a 60 s background agent and replied at once; chip read 后台运行 from 18 s; a second message sent at 21 s was answered QUICK in about 5 s while the job still ran; the sub-agent's own text never showed; Claude Code relayed DONE-BG at the end. |
| B10 Sign in with ChatGPT | **FAIL → fixed, not re-run** | Stops at storing the tokens: see blocker 1. Log: `ChatGPT sign-in: waiting for the browser callback on 127.0.0.1:56231`, `POST …/oauth/token 200`, `GET …/jwks.json 200`, `GET …/v1/models 200`, then `POST /api/v2/providers/chatgpt/signin 500` with no traceback in either log. |
| C Antigravity sign-in (step 2) | PASS | 登录 → owner signed in with Google and pasted the code in the card; job `complete`, `verified: true`, card 已登录 with no note. Job tail: the two prompts and "Logged in as <email>"; the code appears nowhere in the tail or `daemon.log`. The "…" in the URL now reads correctly (no `â€¦`). |
| C ineligible exit | PASS, Welcome part not run | This network is refused. Pinned PONG ended as `antigravity 因错误停止：Antigravity 在你所在的地区不可用`; log: `turn refused by Antigravity (region_unavailable): '⚠ You do not have a valid license … (#3501)'`. Pin menu: "Antigravity, 所在地区不可用". Settings: 已登录 + 所在地区不可用. 刷新 cleared it (`ready: true`) until the next turn. Not run: the Welcome row and Start not counting it. No console window during the turn. |
| C steps 3–6 | NOT RUN | Need an eligible exit (US, Japan, Singapore). `note-antigravity-windows.md` stays PENDING. |
| D in-app upgrade | NOT RUN | No published 0.25.0 and no feed was set up. |

## A2 detail (look D in the real window)

- **How the scalings were made.** 100 % is the primary monitor. For 125 % and 150 % the app was
  started with `--force-device-scale-factor=1.25` / `1.5`, which gives the page the same device
  pixel ratio a 125 % / 150 % Windows would; the native title bar stayed at 100 %. The owner
  declined changing the system scaling. A true system-wide 125 / 150 % was therefore **not** seen.
- (a) **Title bar**: the native title bar is plain white with black text in both themes. On the
  light chrome it reads as a slightly brighter band; **on the dark theme it is a white bar on a
  near-black window.** `A2\*-dark-*` and `A2\_corners-zoom.png`. Reported, not fixed.
- (b) Ring: a 1 px ring on the sheet's left and top edge at 100 % and 125 % (`#d0d9e5` / `#cfd4dd`
  beside white in light), visible in both themes at all three scales.
- (c) Banding: none. Down the chrome the colour moves one level at a time (`#dde4f0` → `#e8e6f0`
  in light, `#131825` → `#111018` in dark), longest flat run 30 px.
- (d) Sheet bottom corners: rounded and inset from the window edge at 100 % and 150 %; Windows 10
  windows have square corners, so nothing clashes. At 125 % my window overlapped the taskbar by
  10 px, so that row of shots does not show the bottom edge.
- (e) Microsoft YaHei: no overflow in buttons, sidebar rows, 新建会话 or the collapsed rail.
- (f) The collapsed right panel's handle (显示工作区) sits on the sheet's right edge, inside it.
- **Finding 12 (not look D, older):** Orbital is only system-DPI aware
  (`GetProcessDpiAwareness` = 1). On a second monitor whose scaling differs from the primary one
  Windows stretches the whole window as a bitmap, so everything is blurred. Seen on the 125 %
  monitor here.

## Findings (not fixed; questions for the owner unless marked)

5. **Settings dropdowns that are not full width draw their arrow far from the box** (权限模式 on
   every sub-agent card, the connector 身份验证 menu). `Select.tsx` wraps the `<select>` in a
   full-width `relative block` span and pins the chevron to the span's right edge; a select
   without `w-full` shrinks to its text (65–103 px) while the arrow sits 640 px away. Both
   themes. Older than this build (`8dc134b4`).
6. A5 oddities: the fold's time counted up to "26秒" while running and settled at "19秒". A
   finished thinking row reads "思考中… · <1秒" (present tense). A reply that links a file
   mid-sentence renders the file card as a block, leaving a lone "." on its own line. After the
   reply a single status row ("Summary complete · <1秒") stands alone.
7. B1: editing a queued row moves its text back into the box and Enter re-queues it **at the
   end**, so the order changes. The waiting messages go out together as one turn; the manager
   answered only the last one ("THREE").
8. B2: when the manager's own read step targeted the folder ".", the right panel opened a preview
   titled "." showing "无法加载文件".
9. B3: created files start unticked in the undo sheet (by design in `TurnUndoSheet`), so "undo
   since a message" with the default ticks removes nothing that was created. A script the agent
   created and deleted inside one turn is listed as "删除了这个文件" and pre-ticked; restoring it
   does nothing. A one-word edit is counted as "改了 2 行".
10. B5a: (i) the pin menu says 需要登录 but offers no Sign in there. (ii) The failed send is in
    English with a raw `Error:` prefix under a Chinese UI. (iii) With the ChatGPT/Codex desktop
    app running, Orbital's `codex login` job ends after 2.4 s with "Error logging in: Login
    cancelled" (the app takes over port 1455), the browser tab still completes the sign-in, and
    the card then shows 已登录 together with the stale "登录未完成。请重试…" line.
11. **agy.exe console windows**: 8 visible `agy.exe` console windows in about 80 minutes, each a
    moment before a status sweep finished (00:17, 00:42, 00:58, 01:14, three at 01:27, 01:30).
    Every spawn in `agent_os` passes `CREATE_NO_WINDOW` (checked call by call), and 12 runs of
    `agy models` plus 3 of `agy acp` with that flag from a plain Python process opened none. The
    spacing (roughly every 15–25 minutes) fits a periodic job of agy's own, probably its update
    check: agy updated itself during the session. So agy opens it; that is an inference.
13. Pinned sends show two raw rows per turn: `codex 已发送："…". Transcript: C:\…\x.jsonl` and
    `codex 已完成：…. Transcript: …`. A first turn in a new project ends with a "这样理解对吗？"
    card whose only step is "No further steps are currently requested."
14. A successful login job is logged at WARNING (`login job … ended: status=complete`).

## Fixes (all on `test/1125-all-fixes`, each with a test that was red first)

| Commit | What | Verified |
|---|---|---|
| `69880045` | Codex: `thread/name/set` is retried after 1 s, 3 s and 8 s when refused. Codex 0.171 answers "rollout … is empty" 50 ms after `turn/start`; Orbital tried once and logged the failure at DEBUG. | Dev daemon: "refused … again in 1.0s" then `named thread …`; `thread/read` shows the session title. |
| `1b64ae31`, `33509d86` | Import: a native session whose title starts with one of Orbital's own openers is skipped. The 33 threads were not Orbital's Codex threads: the Codex desktop app imports Claude Code sessions into its own store (`external-import-turn-1`, originator "Codex Desktop", new ids), so neither the folder's thread records nor the originator matched. | From the repo: `list_native_sessions` on three of the affected folders returns none of them. |
| `fffa18b0` | Windows: a card secret over 1000 characters is split over `card:<id>#1..n`. | A 6004-character secret round-trips through the real Credential Manager; shorter replacement and delete leave no parts. |
| `029261d7` | Windows title bar follows the light / dark theme (owner's request). | Real window from the repo on Windows 10: see "Later the same night". |

Local unit runs: the touched test files pass. Five tests in `test_session_import.py`, two in
`test_onboarding_rows_138.py`, two in `test_browser_manager.py` and one in `test_left_off_138.py`
fail on this PC with and without these commits (the known local baseline); CI decides.

## Later the same night (02:00–02:40)

- **B10 from the repo, real Windows vault, with the owner's consent: PASS.** A repo daemon
  (`fffa18b0`, isolated data dir, the real `WinVaultKeyring`) ran the sign-in end to end:
  `POST /providers/chatgpt/signin` 200, card "ChatGPT · gpt-6.2-sol" with the account email,
  `auth: signin`, connection test "Connected to chatgpt using gpt-6.2-sol", default card. The
  vault holds `orbital-parts:v1:4` and four parts (1000, 1000, 1000, 917 characters). A manager
  turn on that card read `fact.txt` and answered with the code word. The card was kept
  (`C:\orbital-scratch\1125\dev-realvault\data`, card `card_35086af1`) so the installer check
  can run a turn on it and then delete it without a second consent. Not yet seen: deleting the
  card signs out; any of this on a CI installer.
- **Title bar (A2a), at the owner's request: it now follows the theme** (`029261d7`). The page's
  `data-theme` is reported to the shell on load and on every change; the window asks Windows for
  its dark title bar in the dark theme, and on Windows 11 also for the chrome's colour. Checked
  on this Windows 10 (build 19045) with a real pywebview window from the repo: bar white in
  light, black in dark, white again, each within a second of the theme change
  (`evidence\titlebar-*.png`). On Windows 10 the light theme is unchanged (Windows offers only
  its own light or dark bar there); the Windows 11 colour path is unit-tested only. Removing the
  bar (frameless) was discussed and left for later.
- CI: run `45670454550` (head `74034c06`, the three fixes) built its Windows installer; it was not
  installed because the title bar commit followed. Run `45672108561` (head `029261d7`) is the
  build to install. **Still not done: installing it (one UAC click) and re-checking B7, B8, B10
  and the title bar on it.**

## Not run, and why

- A2/A3 at a real system-wide 125 % and 150 % (emulated instead; the owner declined the setting change).
- B1 with the real Microsoft Pinyin IME, in the composer and while editing a queued row (needs hands on the keyboard).
- B2 manager staying quiet while pinned to another agent.
- B4 (no limited account). B5a "no ready agent" case. B5b Antigravity install line (agy is installed).
- B8 with the older npm Codex forced. B9a path set by hand.
- B10 after the token store, on any installer.
- C steps 3–6, and the Welcome row for a region-refused Antigravity.
- D.
- Whether a stale `gemini-cli.yaml` loaded by v0.24.1 after a downgrade does any harm.

## Testing notes for the next Windows session

- **Do not tap Alt to bring the Orbital window forward.** It starts Windows' menu mode on the
  window and the WebView2 page (and its debug port) freezes until the next key. That cost an hour.
  Use `SetWindowPos(HWND_TOPMOST, SWP_NOACTIVATE)`.
- The WebView2 debug port (`WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS`) drives the real window;
  `--force-device-scale-factor` in the same variable emulates scaling.
- The third fresh-data launch of the night wrote its `Orbital` folder into the Claude desktop
  app's private layer (`…\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Roaming`) instead of the real
  Roaming folder; the first two did not. Check both places before and after each reset.
- `gh run download` managed about 2 MB/min here. The artifact's blob URL takes ranged requests:
  32 parallel `curl -r` parts finished 408 MB in under two minutes.
- Bash heredocs still eat backslashes; write scripts with a file tool.
- Two UAC prompts meant to be declined were accepted; ask for the decline with a blocking question.

## Left on this PC

- Orbital 0.25.0 (the CI build) installed, running on the real data. Real data now also holds two
  scratch projects, `smoke1125-a4` (11 sessions) and `smoke1125-b3`, both under `C:\orbital-scratch\1125\ws\`.
- `C:\orbital-scratch\1125\`: the installer, `appdata-run1` (38 scratch folders imported),
  `appdata-run2`, `appdata-run3` (the three fresh data folders), `realdata-safety-copy`, a dev
  daemon's data, workspaces, tools, `evidence\`.
- Codex was signed out and is signed in again. Antigravity (agy 1.5.17) is now signed in with the
  owner's Google account; its card has no sign-out button.
- The ChatGPT consent was given once; nothing was stored. A `chatgpt-host-id` file is in `appdata-run3`.
- New Codex threads in `~\.codex` for the scratch folders (two of them named by the probe and the
  dev daemon), new Claude Code sessions for `C--orbital-scratch-1125-ws-a4`.
- `orbital\` state was refreshed inside the 38 imported scratch folders.
