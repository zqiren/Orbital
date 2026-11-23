# T3 Code and desktop-app coding agents: how it integrates them, and what Orbital could do

Date: 2026-11-24. Research only; no Orbital code was changed. Branch-only file, like `note.md` (`docs/research/` is gitignored).

Evidence tags used throughout:
- **[src]** read in source code (file:line given)
- **[docs]** read in the project's own docs
- **[issue]** read in a GitHub issue/PR thread (a user's or maintainer's statement, not code)
- **[inference]** my conclusion, not directly stated anywhere

## Summary

**T3 Code does not integrate any vendor desktop app.** It does not connect to a running Antigravity,
ChatGPT/Codex, Claude or Cursor desktop app. It does not use their IPC, sockets or deep links, and it
does not look inside their bundles for a CLI. Every agent runs as a process T3 starts itself (ACP over
stdio, `codex app-server`, the Claude Agent SDK, the Cursor SDK, `opencode serve`, `pi --mode rpc`). It
finds that process through PATH, a user-set **Binary path**, or **a runtime T3 downloads and manages
itself**. Its maintainers turned down automatic discovery of the CLI inside ChatGPT.app on macOS and
Windows (PRs #14110 and #33016, both closed unmerged). They called it a product decision because the
bundled build "can be chatgpt-desktop-specific, and its versioned path could change with desktop
updates" [issue].

So the answer to "is there a better way than CLI-only?" is mostly **no, not through the desktop apps**.
The vendors' supported automation surfaces are the CLIs and SDKs. The one place T3 tried to reach into
a desktop app (ChatGPT's browser-use channel) was refused by the app itself (#27083) [issue]. In this
respect Orbital is already ahead of T3: it auto-discovers the desktop apps' bundled CLIs (Codex on both
OSes, DSH, and Claude's copy on macOS), and it forks a Codex thread when the desktop app holds the
writer lock.

T3 does several things around the CLIs better than Orbital, and they are worth copying:
1. a **stdio MCP bridge**, so ACP agents without HTTP MCP (Antigravity) still get the host's tools.
   This closes Orbital's "Antigravity cannot dispatch" gap.
2. **isolated per-instance profiles** for agy (`GEMINI_HOME`, file token storage), which allow
   several accounts and keep sign-in deterministic.
3. a **managed Antigravity runtime** downloaded from Google's own release manifest, so no
   `irm | iex` step.
4. **capturing the sign-in URL** by pointing `BROWSER` at a helper, instead of letting agy open the
   browser.

## Sources read

- T3 Code: `github.com/pingdotgg/t3code`, default branch `main`, commit
  **`42c9107983d8f65c38117501cbcde1ee5c82ef14`** (2026-11-23, "fix(server): storage cleanup removes
  merged subagent worktrees (#41694)"). MIT license. Shallow clone at
  `C:\orbital-scratch\t3code-research\repo`. Read only; nothing installed or run.
- Paths below are relative to that repo. Line numbers are at that commit.
- Orbital: branch `test/1124-welcome-antigravity` at `D:\orbital-public`.

---

## 1. T3 Code: per-agent integration

Built-in drivers: Codex, Claude, Cursor, Grok, fx, OpenCode, Antigravity, Kiro, Pi, Muse, plus a
generic ACP Registry driver [src `apps/server/src/provider/builtInDrivers.ts:35-47`]. The README says
the same and adds "any ACP agent" [docs `README.md:28`].

| Agent | How T3 connects | How the binary is found | Sign-in | Desktop app used? |
|---|---|---|---|---|
| **Antigravity** | `agy acp --gemini-dir <profile> --app-data-dir antigravity-acp` over stdio ACP [src `apps/server/src/provider/antigravityAuthSupport.ts:545-572`] | 1) user Binary path; 2) **T3-managed runtime** downloaded from Google's CLI auto-updater manifest into `<baseDir>/tools/antigravity-acp/<platform>-<arch>/versions/<sha>` [src `AntigravityInstallation.ts:198-205`, `antigravityCliRelease.ts:3-4,43`]; 3) `agy` on PATH [src `AntigravityInstallation.ts:663-666`] | Own flow over `agy acp --login` / ACP `authenticate`, with T3 capturing the URL (details below) | **No.** Error text: "The Antigravity app, IDE, and CLI are separate installations." [src `AntigravityInstallation.ts:670`]. The IDE is only an "Open in editor" target [src `packages/shared/src/editor.ts:27-28`] |
| **Codex** | `codex app-server` JSON-RPC [src `apps/server/src/provider/codexLaunchArgs.ts:17`, `CodexInstallation.ts:545`; package `packages/effect-codex-app-server`] | `codex` on PATH [src `CodexInstallation.ts:500`] or a T3-managed copy downloaded from npm/GitHub releases into `<baseDir>/tools/codex` [src `CodexInstallation.ts:39-40,208`], or the Binary path | Existing `codex login` (shared `~/.codex`), or T3's own "Connect with ChatGPT" OAuth (scope `chatgpt.tokens.use.direct`) feeding Codex through a custom `model_providers` entry with an auth command [src `CodexChatGptAuth.ts:39-41`, `CodexManagedRuntime.ts:26`; docs `docs/user/providers-codex.md:5-34`] | **No auto-discovery.** Users can point Binary path at `ChatGPT.app/.../codex` by hand (#10514 workaround) [issue]. Auto-discovery PRs #14110 (macOS) and #33016 (Windows) closed unmerged [issue]. The one thing added: an app-bundled copy is never offered a one-click update (commit `b8e8e0d`, fixes #16351) [issue] |
| **Claude Code** | `@anthropic-ai/claude-agent-sdk` `query()` with `pathToClaudeCodeExecutable` [src `apps/server/src/orchestration-v2/Adapters/ClaudeAdapterV2.ts:68,1222,1372-1374`] | Binary path, default `claude`; Windows `.cmd`/WinGet shims resolved to the native exe [src `Drivers/ClaudeExecutable.ts:1-45`] | Claude Code's own login; extra accounts via `CLAUDE_CONFIG_DIR` [docs `docs/user/providers-claude.md:3-38`] | **No.** No reference to Claude desktop's bundled Claude Code. The only `/Library/Application Support/ClaudeCode` paths are Claude Code's managed-policy files [src `Drivers/ClaudeHome.ts:96`] |
| **Cursor** | Official `@cursor/sdk` local-agent runtime, not ACP and not the `agent` binary [docs `docs/user/cursor.md:1-4,87-89`; src `packages/provider-cursor/src/server/CursorAgentSdk.ts:11`] | n/a (SDK in process/worker) | SDK `Cursor.auth.login({openBrowser:false, apiKeyName:"T3 Code - …"})`, which mints a named API key [src `packages/provider-cursor/src/server/auth.ts:205-211`] | **No.** "Your Cursor editor and CLI login are separate" [docs `cursor.md:35-36`]. MCP OAuth done in the Cursor desktop does not carry over [docs `cursor.md:18-27`] |
| OpenCode | `opencode serve --hostname --port` started by T3, or **an existing OpenCode server URL** + password [src `packages/provider-opencode/src/server/OpenCodeRuntime.ts:610,821-836`; docs `providers-opencode.md:32-56`] | PATH / Binary path | OpenCode's own | Talks to an already-running OpenCode server over HTTP. This is the only "attach to a running process" integration, and it uses OpenCode's documented server API |
| Pi | `pi --mode rpc` [src `packages/provider-pi/src/server/mcpInjection.ts:275-276`] | PATH | Pi's own | No |
| Grok | `grok agent stdio` (ACP) [src `packages/provider-grok/src/server/acpSupport.ts:50-54`] | PATH | own | No |
| Kiro, fx, Muse, ACP Registry | ACP over stdio [src `packages/provider-kiro/src/server/acpSupport.ts`, `provider-fx/.../acpSupport.ts`; docs `providers-acp.md`] | PATH / registry | own | No |

### 1.1 Antigravity in detail (all [src] unless marked)

- **Isolated profile per provider instance.** Profile dir `<stateDir>/providers/antigravity/<sha256(instanceId)>`
  (`antigravityAuthSupport.ts:185-196`). Spawn env (`antigravityAuthSupport.ts:202-247`):
  - `GEMINI_HOME=<profile>` (235)
  - `AGY_ACP_FORCE_FILE_STORAGE=1` (236)
  - `AGY_CLI_DISABLE_AUTO_UPDATE=1` (237)
  - `BROWSER=<helper>` (238)
  - `ELECTRON_RUN_AS_NODE=1` (240)
  - T3-owned `TEMP`/`TMP`, because "each launch unpacks about 1 GB" (221, 241-243)
  - For Google sign-in, a **fake `SSH_CONNECTION`**. The code says agy 1.5 "ignores
    AGY_ACP_FORCE_FILE_STORAGE and keeps Google sign-ins in one OS keyring entry for every profile,
    unless it believes it runs over SSH" (230-233).
- **Ambient credentials stripped.** `GEMINI_API_KEY`, `GOOGLE_API_KEY`,
  `GOOGLE_APPLICATION_CREDENTIALS`, the GCP project vars and others are removed before spawn
  (`antigravityAuthSupport.ts:62-82`). Docs say the same (`providers-antigravity.md:39-41`) [docs].
- **User config still shared.** Global skills (directory junctions on Windows) are linked in, and
  `AGENTS.md`, `GEMINI.md` and `config/skills.json` are copied from the real `~/.gemini`
  (`antigravityAuthSupport.ts:255-340`).
- **Sign-in.** agy's browser launch goes to a Node one-liner set as `BROWSER`. The one-liner prints a
  marker plus the URL to stderr (`antigravityAuthSupport.ts:55-60`), and T3 checks it with a preflight
  URL before trusting it (`:476-506`). T3 then shows the URL itself and accepts two kinds of answer:
  - a loopback redirect URL (`127.0.0.1:<port>`), which T3 forwards to agy's listener
    (`antigravityCallback.ts`)
  - or the code shown on `https://antigravity.google/oauth-callback`, which it writes to
    `agy acp --login` stdin (`antigravityAuthSupport.ts:575-640`, `AntigravityAuth.ts:89,289-310`)

  Docs: "Paste that code, or the page's full address" (`providers-antigravity.md:19-26`) [docs].
- **Other sign-in methods:** Gemini Enterprise, Gemini API key, Agent Platform/Vertex
  (`antigravityAuthSupport.ts:108-140`; docs table `providers-antigravity.md:30-37`).
- **Model list:** taken from the ACP session's `configOptions`/`models` after `session/new`, not
  from `agy models` (`AntigravityProvider.ts:82-96,438`). The docs require "health checks must stay
  passive" (`docs/internals/03_providers.md:43-45`) [docs].
- **Version gate.** Needs agy ≥ 1.5, recommends ≥ 1.5.12 [docs `providers-antigravity.md:65-70`].
  Withdrawn releases are refused (`AntigravityInstallation.ts:248,655-660`).
- **Host tools without HTTP MCP.** The ACP adapter always prepares a **stdio MCP server that is T3's
  own binary run as `acp-mcp-bridge`**, relaying to T3's HTTP MCP endpoint with a bearer header
  (`packages/provider-acp/src/server/adapter.ts:1093-1106`). It uses HTTP only when the agent
  advertises `mcpCapabilities.http` (`AcpSessionRuntime.ts:2364-2374`), and otherwise falls back to
  the stdio server. Antigravity opts into HTTP-when-advertised (`AntigravityAdapterV2.ts:224-225`).
  Bridge: `apps/server/src/mcp/AcpMcpStdioBridge.ts:12` ("ACP mandates stdio MCP support; HTTP/SSE
  support varies"), CLI entry `apps/server/src/cli/acpMcpBridge.ts:7`.
  - [inference] Orbital's probe found agy advertises `http:false`. If so, T3 gives Antigravity its
    tools through the stdio path.
- **Reading agy's private on-disk state.** For subagent "child work" and session ownership, T3 opens
  agy's native session SQLite read-only and decodes its protobuf blobs by field number
  (`acp/AntigravitySessionFiles.ts:1,18`, `acp/AntigravityChildWork.ts:1,19`). This is undocumented
  format reverse-engineering.
- **Usage limits: private Google endpoints.** T3 reads the CLI's OAuth token from agy's keyring entry
  (`"gemini"/"antigravity"` via `@napi-rs/keyring`; a separate Windows reader) or from
  `antigravity-oauth-token` in the profile (`antigravityUsageLimits.ts:43-44,231-243,331-337`). It
  then calls `cloudcode-pa.googleapis.com` / `daily-cloudcode-pa.googleapis.com`
  `retrieveUserQuotaSummary` / `fetchAvailableModels` with a User-Agent that claims to be
  `antigravity/hub/2.24.0` (`antigravityUsageLimits.ts:20-31,470-473`).
  - [inference] This impersonates Google's own client against an unpublished API: a ToS and
    stability risk. Open bug #34381 is about stale quota when the CLI is used as the binary [issue].
- **Edit from here:** agy cannot rewind, so T3 starts a fresh agy session and sends the kept
  conversation [docs `providers-antigravity.md:88-91`].
- **History import:** none. Importable sources are only `claudeAgent`, `codex`, `pi` [src
  `packages/contracts/src/agentSessions.ts:5`].

### 1.2 Codex and the ChatGPT desktop app

- **Shared `~/.codex` is the only coupling.** T3's app-server and the desktop app share `CODEX_HOME`,
  so threads, config and plugins leak both ways [issue #37757, #38148]:
  - Delegated T3 threads show up in ChatGPT's Codex sidebar, tagged `originator="T3 Code"` in
    `state_5.sqlite` (#38148, open).
  - The desktop app's bundled `codex-app-tools@openai-bundled` plugin, which is enabled in the shared
    `config.toml`, fails in T3's app-server because only the app passes `CODEX_APP_TOOLS_PIPE_PATH`
    (#37757, open). Suggested workaround: `-c plugins.codex-app-tools@openai-bundled.enabled=false`.
- **Writer lock.** When the app holds a thread, T3 shows "This conversation is open in another Codex
  client, such as the Codex app… Quit that client" [src
  `apps/server/src/orchestration-v2/Adapters/CodexAdapterV2.ts:405-418,8302`]. It does not fork.
- **Desktop-app private channel refused.** A Linux user copied the ChatGPT app's
  `BROWSER_USE_ACCESS_*` / `CODEX_APP_TOOLS_PIPE_PATH` env into T3's Codex. The app answered with a
  signed `{"allowed": false}`: "an access decision inside the app for sessions it didn't start"
  (#27083, closed) [issue]. The reporter says browser use worked on macOS when T3 ran the app's
  bundled Codex; I did not verify that.
- **Read-state sync:** #38421 (closed) proposes Codex app-server's experimental
  `thread/readState/update`. I found no implementation in source (grep for `readState` in the
  server: no hits outside unrelated projection code) [src, negative].
- **History import:** reads `~/.claude/projects/*` and Codex rollouts, plus Codex's
  `state_5.sqlite` for the project of each chat [src
  `apps/server/src/project/AgentSessionScanner.ts:201,1499-1504,1587`]. Because the ChatGPT desktop
  app writes to the same `~/.codex`, desktop-app Codex threads are importable too [inference].

### 1.3 What depends on a desktop app being installed or running

Nothing in T3's agent paths. The only desktop-app touch points are:
1. "Open in editor" can launch the Antigravity IDE, Cursor, VS Code and others
   [src `packages/shared/src/editor.ts:24-31`].
2. The shared Codex home, with the failure modes above.
3. Docs that send users to the ChatGPT desktop app to set up the Chrome extension
   [docs `providers-codex.md:141-155`].

T3's own desktop app has a local socket for `t3 app` [src `apps/server/src/cli/app.ts:40-189`], but
that is T3's own app, not a vendor's.

---

## 2. Orbital today

| Agent | Manifest | Transport | Binary discovery | Sign-in probe / login | Desktop app |
|---|---|---|---|---|---|
| Antigravity | `agent_os/agents/manifests/antigravity.yaml` | `acp-sdk`, `agy acp` (`:51-57`) → `agent_os/agent/transports/acp_sdk_transport.py` | PATH, then `%LOCALAPPDATA%\agy\bin\agy.exe`, `~/.local/bin/agy` and others (`:83-92`); install is the user's `irm … \| iex` / `curl \| bash` (`:67-72`) | `agy models` exit code; `agy acp --login` with a paste-the-code field; agy opens the browser itself (`:24-33,73-82`) | None. Uses the user's real `~/.gemini` and keyring (no profile isolation) |
| Codex | `codex.yaml` | `codex-appserver` (`:27-36`) → `codex_transport.py` (pinned `SUPPORTED_CODEX_VERSION = "0.162.1"`, `:54`) | npm shim, then `%LOCALAPPDATA%\OpenAI\Codex\bin\*\codex.exe` (`:114-119`), then macOS ChatGPT.app by bundle id `com.openai.codex` (`:106-112`); `binary_choice: newest` (`:105`) | `codex login status` exit code; `codex login` (`:77-83`) | **Yes: auto-discovers and may prefer the app's bundled CLI.** Writer-lock refusal answered with `thread/fork` (`codex_transport.py:56-60`, spec 126) |
| Claude Code | `claude_code.yaml` | `sdk` → `sdk_transport.py` (claude-agent-sdk) | npm, native `~/.local/bin/claude.exe`; macOS also `~/Library/Application Support/Claude/claude-code/*/*/claude.app/...` (`:73-101`) | `claude auth status --json` `loggedIn` (`:59-72`) | Yes on macOS (Claude desktop's downloaded Claude Code, lowest priority). **Not on Windows** |
| Cursor | `cursor.yaml` | `acp-sdk`, `agent acp` (`:13-18`) | `%LOCALAPPDATA%\Programs\cursor\resources\app\bin\agent.cmd` and npm (`:36-57`) | `agent status --format json` `isAuthenticated` (`:24-33`) | Partly: the Windows path is inside the Cursor IDE install |
| DSH | `dsh.yaml` | `acp-sdk` | `desktop_apps` with `order`, macOS bundle id `com.deepseek.dsh`, Windows uninstall-registry `DisplayName` + `install_dirs` (`:41-56`) | n/a here | Yes (vendor's primary distribution) |
| Kimi, MiniMax, CodeBuddy | `kimi-code.yaml`, `minimax-code.yaml`, `codebuddy.yaml` | `acp-sdk` | PATH / auto_detect | per manifest | No |
| Pi | `pi.yaml` | `pi-rpc` → `pi_rpc_transport.py` | PATH | per manifest | No |
| Aider, Cline, Continue, Copilot, Goose | respective yaml | `pty` | PATH | per manifest | No |

Mechanism: `agent_os/agents/desktop_apps.py`:
- macOS finds apps by bundle id (Spotlight `mdfind` plus a fallback scan of the Applications
  folders, validated against Info.plist).
- Windows finds them through the uninstall registry's `InstallLocation`, with default dirs as a
  fallback (`:5-24,163-287`).
- A locate card maps a user-picked `.app` or folder to the CLI inside (`:322-357`).
- `SetupEngine.resolve_binary` orders the user pin, a `before_cli` app, PATH, auto_detect and an
  `after_cli` app. With `newest`, the highest `--version` wins. A pinned path is re-resolved when it
  disappears (`agent_os/agents/setup_engine.py:1119-1150`).

MCP gap: Orbital mounts its dispatch server only as HTTP MCP and gives up when the agent does not
advertise `mcp_capabilities.http` (`acp_sdk_transport.py:421-438`). That is why an Antigravity-pinned
chat cannot dispatch (`note-antigravity-windows.md`, "Antigravity has no HTTP MCP").

History import: `agent_os/onboarding/sources/` covers claude_code, codex, kimi_code, minimax_code
and pi. Antigravity has none.

---

## 3. Side by side

| Topic | T3 Code | Orbital |
|---|---|---|
| Talks to a running vendor desktop app | No | No |
| Uses CLI bundled in a desktop app | Only if the user types the path; auto-discovery declined | Yes, automatically (Codex both OSes, DSH, Claude macOS) |
| Vendor runtime install | Downloads and manages Codex and Antigravity itself (checksummed, versioned, rollback) | User runs the vendor installer; Orbital detects it |
| Antigravity profile | Private `GEMINI_HOME` per instance, file token storage, ambient creds stripped, multi-account | User's own `~/.gemini` and keyring |
| Antigravity sign-in URL | Captured via `BROWSER` helper; T3 shows it; loopback or code | agy opens the browser; Orbital shows only a code field |
| Antigravity model list | From the ACP session's config options | `agy models` as the probe; picker from session options |
| Host tools for ACP agents without HTTP MCP | stdio bridge (own binary as `acp-mcp-bridge`) | None: dispatch not mounted |
| Codex writer lock held by desktop app | Error: "quit that client" | `thread/fork` and continue |
| Codex/desktop shared-home side effects | Known open bugs (#37757, #38148) | Same exposure, not handled [inference] |
| Usage/quota for Antigravity | Private Google endpoints with spoofed UA | None |
| History import | Claude, Codex, Pi | Claude, Codex, Kimi, MiniMax, Pi |

---

## 4. Options for Orbital, ranked by value vs effort

### A. stdio MCP bridge for ACP agents without HTTP MCP (high value, low-medium effort)
- **What:** when `mcp_capabilities.http` is false, pass an ACP `McpServerStdio` entry whose command
  is Orbital's own executable in a small relay mode. The relay forwards JSON-RPC over stdio to the
  daemon's HTTP MCP URL with the bearer token, given through env, not argv. This gives Antigravity
  (and any stdio-only ACP agent) the dispatch tool.
- **Takes:**
  - a relay entry point in the frozen app (PyInstaller: a `--mcp-bridge` flag on `Orbital.exe` or a
    tiny console script)
  - a branch in `acp_sdk_transport.py:_mcp_servers`
  - tests that the token never lands in argv or logs
- **Risks:** low. It is the ACP spec's mandatory path (T3 relies on the same assumption,
  `AcpMcpStdioBridge.ts:12`). On Windows the relay process needs CREATE_NO_WINDOW.
- **Not verified:** that agy 1.5.x actually loads stdio MCP servers passed in `session/new`
  [inference from ACP spec + T3 design]. Probe with one run.

### B. Isolated Antigravity profile, with Orbital-shown sign-in URL (medium-high value, medium effort)
- **What:** spawn `agy acp --gemini-dir <orbital profile> --app-data-dir antigravity-acp` with
  `GEMINI_HOME`, `AGY_ACP_FORCE_FILE_STORAGE=1` and `AGY_CLI_DISABLE_AUTO_UPDATE=1`. Set `BROWSER` to
  a helper that hands the URL to Orbital. Strip ambient `GEMINI_API_KEY`/`GOOGLE_*`. Copy or link
  `~/.gemini` skills and `GEMINI.md`.
- **Gains:**
  - several Google accounts
  - sign-in never collides with the user's terminal agy
  - Orbital controls the URL, so the "a second tab gives Invalid code verifier" trap disappears
  - works from the phone/relay, since the user can paste the loopback redirect URL
- **Risks:**
  - T3's fake `SSH_CONNECTION` trick (`antigravityAuthSupport.ts:230-233`) depends on undocumented
    agy behaviour and could break on any agy release.
  - The user's existing terminal sign-in is no longer reused: one extra sign-in.
  - Windows path length (T3 hashes and shortens dirs for the 260-char limit, `antigravityAuthSupport.ts:180-183`).
- **Not verified:** whether the flags and env vars exist in the agy build Orbital probed (1.5.13);
  T3 targets ≥ 1.5.12 [docs].

### C. Managed Antigravity runtime download (medium value, medium-high effort)
- **What:** fetch Google's manifest
  (`antigravity-cli-auto-updater-…run.app/manifests`, `antigravityCliRelease.ts:3-4`). Check that the
  URL is under `storage.googleapis.com/antigravity-public/antigravity-cli/` and verify the sha512.
  Install into Orbital's data dir and keep the previous release for rollback.
- **Gains:** removes the `irm … | iex` step from the Welcome page.
- **Risks:**
  - The manifest endpoint is an undocumented Cloud Run URL.
  - About 230 MB download, and about 1 GB unpacked per launch (T3 comment, `antigravityAuthSupport.ts:221`).
  - Orbital then owns updates.
- **Not verified:** that Google permits redistribution-by-download; T3 does it, but I found no Google
  statement either way.

### D. Defuse Codex shared-home side effects (medium value, low effort)
- **(1)** Pass `-c plugins.codex-app-tools@openai-bundled.enabled=false` when the desktop app's
  plugin is enabled. This avoids the `codex_app` MCP startup failure that #37757 reports on Windows
  Store builds.
- **(2)** Check how Orbital-originated threads show in ChatGPT's sidebar (#38148 analysis: the app
  hides only ephemeral/`chatgpt_hidden`/child threads). Consider `ephemeral` for Orbital's one-off
  routing calls (note.md A7 already requires "no new session appears in the agent's own history").
- **Risks:** relies on `-c` key syntax quoted in an issue (unquoted form works, quoted form silently
  ignored) [issue]. Re-check on the pinned Codex version.
- **Not verified:** whether Orbital's app-server sessions hit #37757 today.

### E. Claude desktop's bundled Claude Code on Windows (low-medium value, low effort, needs a probe)
- Orbital finds it on macOS only (`claude_code.yaml:97-101`). Find where Claude desktop for Windows
  keeps its downloaded Claude Code, then add a glob.
- **Not verified:** that the Windows desktop app ships Claude Code at all, and where. I did not check
  this machine.

### F. Antigravity history import (low value, medium effort)
- agy keeps sessions in a SQLite DB with protobuf blobs (T3 decodes them by field number,
  `AntigravitySessionFiles.ts:18`). It is fragile and undocumented. Wait until Google documents an
  export or ACP `session/list` with history is enough.

### G. Talking to the desktop apps themselves (not recommended)
- No vendor publishes a local API for its desktop app. The one observed attempt was refused by the
  ChatGPT app with a signed denial (#27083) [issue].
- Private IPC, pipes and app-internal plugins change with every app update. Using them is likely
  outside the terms of use [inference].
- Reusing a desktop app's stored credentials directly (keychain entries, token files) is the same
  class of risk. T3 does it only to read Antigravity quota.

### H. Usage/quota for Antigravity (not recommended as T3 does it)
- T3 calls unpublished `cloudcode-pa` endpoints with a User-Agent pretending to be Google's Hub app.
  It is useful, but it impersonates a first-party client. Skip unless Google publishes a quota API or
  ACP carries usage.

---

## 5. Not verified / could not establish

- Whether agy 1.5.13 honours `--gemini-dir`, `--app-data-dir`, `GEMINI_HOME`,
  `AGY_ACP_FORCE_FILE_STORAGE` and the `SSH_CONNECTION` keyring bypass. Read in T3 source; not run.
- Whether agy loads stdio MCP servers from `session/new` (option A). Inferred from the ACP spec and
  T3's design.
- Whether T3's Antigravity actually ends up on the stdio bridge. It does if agy advertises
  `http:false`, as Orbital's probe recorded. I did not check what agy advertises on Windows.
- Whether T3 sessions using ChatGPT.app's bundled Codex on macOS get browser use (claimed in #27083,
  not verified).
- Whether the Claude desktop app on Windows bundles Claude Code, and where.
- Whether Google's CLI manifest URL and the `cloudcode-pa` quota calls are allowed under Google's
  terms. No statement found either way; I did not search Google's ToS.
- T3 issue/PR discussion beyond the threads cited (#10514, #14110, #16351, #27083, #33016, #37757,
  #38148, #38421, #34381) was not read exhaustively.
- I did not search for other open-source tools that integrate vendor desktop apps, so I name none.
- Nothing in T3 was executed. Every T3 behaviour above is from reading code, docs and issues.
