<p align="center">
  <img src="docs/subagent-dispatch.gif" alt="Delegating work to Claude Code from inside an Orbital project" width="800">
</p>
<p align="center"><em>You assign work to the project. The agent plans, delegates to Claude Code, and reports back.</em></p>

<p align="center">
  <img src="docs/approval.gif" alt="Claude Code requests approval to create a folder — approved from phone" width="800">
</p>
<p align="center"><em>Consequential action — approval request to your phone — work continues.</em></p>

<h2 align="center">Give your agent a project, not a prompt.</h2>
<p align="center">The project workspace you and your agent share — with memory that persists,<br>sandbox boundaries you set, and approvals you control.</p>

<p align="center">
  <a href="https://github.com/zqiren/Orbital/releases/download/v0.5.1/Orbital-Setup-1.0.0.exe"><strong>Windows Installer (.exe)</strong></a> &nbsp;&middot;&nbsp;
  <a href="https://github.com/zqiren/Orbital/releases/download/v0.5.1/Orbital-1.0.0-macOS.dmg"><strong>macOS Installer (.dmg)</strong></a> &nbsp;&middot;&nbsp;
  <a href="https://www.youtube.com/watch?v=ranTQFmW6vU"><strong>Watch the demo</strong></a>
</p>
<p align="center">Set up in under 5 minutes. No Python or Node required.</p>

<p align="center">
  <img src="docs/screenshots/orbital-logo.png" alt="Orbital" width="80">
</p>

# Orbital

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](#license) ![Platform: Windows](https://img.shields.io/badge/Platform-Windows-0078D6?logo=windows) ![Platform: macOS](https://img.shields.io/badge/Platform-macOS-000000?logo=apple) ![Status: Alpha](https://img.shields.io/badge/Status-Alpha-orange)

---

## Why this exists

Working with AI agents today is micromanaging an intern.

You paste context into every session. You watch every action because you don't trust it won't break something. You copy outputs between tools because the agents can't see each other's work. You re-explain the project every morning because nothing persisted. You run one thing at a time because running more means losing track.

A real project isn't a chat session. It's a workspace with a goal, a budget, boundaries, and a history. You hand it to someone you trust, check in when it matters, and get out of the way. That's what Orbital does for agents.

---

## What makes Orbital different

**A project, not a prompt.** Each project is a folder on your machine with its own workspace, instructions, memory, budget, sandbox, and approval rules. Context persists across sessions. Decisions accumulate. A project you opened yesterday knows what happened yesterday.

**Delegate, don't micromanage.** Pick an autonomy preset — hands-off, check-in, or supervised. The agent runs within the boundaries you set. Consequential actions surface to you for approval, with full context. Everything else, the agent handles. You supervise from your desk or your phone.

**One workspace, many agents.** Your Orbital project can dispatch work to Claude Code, Codex, Gemini CLI, or any CLI agent — all working on the same files, with access to the same decisions, instructions, and history. Stop copy-pasting between a chat tab and a terminal tab. The agents see each other's work because they're in the same project.

---

## At a Glance

- **Project-based agent management** — each project is a folder with its own workspace, instructions, budget, and autonomy level
- **Sub-agent delegation** — the management agent monitors your workspace, evaluates progress against goals, and dispatches work to Claude Code, Codex, Gemini CLI, or any CLI agent
- **Triggers** — set up a cron job or file watcher so the management agent checks in regularly and kicks off sub-agents without you
- **Approval workflows** — agents pause before risky actions; approve from desktop or phone
- **Sandboxed execution** — agents only access folders you specify (Windows sandbox user, macOS Seatbelt)
- **Mobile supervision** — manage agents from your phone via QR code pairing
- **Budget controls** — per-project spending limits with configurable actions
- **13 built-in LLM providers** — Anthropic, OpenAI, DeepSeek, Moonshot (Kimi), Groq, Google Gemini, xAI, Mistral, Together, OpenRouter, Zhipu, Qwen, plus custom endpoints
- **Browser automation** — 26 browser actions via Patchright with anti-detection
- **Persistent context** — PROJECT_STATE.md, DECISIONS.md, LESSONS.md maintained across sessions
- **Self-improving skills** — agent creates reusable skills from multi-step workflows and consults them before repeating similar tasks
- **Credential store** — API keys and website passwords in OS keychain, never exposed to chat

---

## Quick Start

1. **Launch Orbital** — the setup wizard guides you through three steps:

   **Step 1 — LLM Provider:** Connect your API key. Supports Anthropic, OpenAI, Moonshot, DeepSeek, and a dozen other providers.

   <p align="center">
     <img src="docs/screenshots/apikey-setup.png" alt="Setup wizard step 1 — configure your LLM provider and API key" width="700">
   </p>

   **Step 2 — Sandbox:** Orbital creates an isolated user account so agents can't access your personal files or network without permission.

   <p align="center">
     <img src="docs/screenshots/sandbox-setup.png" alt="Setup wizard step 2 — sandbox isolation confirmation" width="700">
   </p>

   **Step 3 — Browser Warm-up:** Sign into sites your agents will need (Google, GitHub, etc.) so they can browse without getting blocked by CAPTCHAs.

   <p align="center">
     <img src="docs/screenshots/browser-warm-up.png" alt="Setup wizard step 3 — browser warm-up for agent web access" width="700">
   </p>

2. **Create a project** — give it a name, pick a workspace directory, set an autonomy level

<p align="center">
  <img src="docs/screenshots/new-project-setting.png" alt="New project creation with workspace and autonomy settings" width="700">
</p>

3. **Chat** — type a task in the chat bar and the management agent handles it
4. **Approve or automate** — review tool calls in the approval card, or set autonomy to hands-off

---

## Screenshots

<p align="center">
  <img src="docs/screenshots/2A-dashboard-all-running.png" alt="Orbital dashboard with multiple projects running in parallel" width="800">
</p>
<p align="center"><em>Multiple projects running in parallel — each with its own workspace, triggers, and session history</em></p>

<p align="center">
  <img src="docs/screenshots/5B2-mobile-approval-card.png" alt="Mobile approval card — approve agent actions from your phone" width="350">
</p>
<p align="center"><em>Approve agent actions from your phone — with full context and optional guidance</em></p>

<p align="center">
  <img src="docs/screenshots/3B-subagent-delegation.png" alt="Sub-agent executing delegated work and reporting back" width="800">
</p>
<p align="center"><em>Management agent delegates Phase 1 to @claudecode, monitors progress, and reviews the result — all inside the same project workspace</em></p>

---

## How It Works

Orbital treats each unit of agent work as a **project** — not a chat session. A project binds a workspace directory, evolving instructions, an autonomy preset, a budget, approval rules, and persistent state into one supervised unit. The agent works inside the project. You supervise from anywhere.

```mermaid
flowchart TB
    UI["<b>Frontend (React SPA)</b><br/>Chat UI · Approval Cards · Settings · Files"]

    subgraph daemon["Daemon (FastAPI + uvicorn)"]
        direction TB
        AM["AgentManager<br/><i>lifecycle</i>"]
        SAM["SubAgentManager<br/><i>delegation</i>"]
        TM["TriggerManager<br/><i>cron · file watch</i>"]
        Loop["Agent Loop<br/><i>streaming · safety guards</i>"]
        TR["Transports<br/>Pipe · PTY · SDK · ACP"]
        LLM["LLM Provider<br/><i>OpenAI + Anthropic SDK</i>"]
        Tools["Tool Registry<br/><i>shell · file · browser · triggers</i>"]
        Auto["Autonomy Interceptor<br/><i>approve · deny · bypass</i>"]

        AM --> Loop
        SAM --> TR
        TM --> AM
        Loop --> LLM
        Loop --> Tools
        Loop --> Auto
    end

    Platform["<b>Platform Layer</b><br/>Windows sandbox user · macOS Seatbelt · Linux bubblewrap (planned)"]
    Relay["<b>Cloud Relay (Node.js, optional)</b><br/>REST proxy · Event forwarding · Push notifications · Pairing"]
    Phone["Phone"]

    UI <-->|REST + WS| AM
    UI <-->|REST + WS| SAM
    Tools --> Platform
    AM -.WebSocket tunnel.-> Relay
    Relay -.WebSocket.-> Phone
```

**Key design decisions:**
- **Isolation**: OS-level sandboxing (Windows sandbox user, macOS Seatbelt, Linux bubblewrap planned)
- **Fail-closed interceptor**: Any approval system error results in DENY, never ALLOW
- **Single daemon**: PID file enforcement prevents multiple instances
- **Local-first**: Your files and project state live on your disk. The cloud relay, when enabled, proxies approvals and events — not your files.

---

## How Orbital compares

| | Orbital | Claude Projects | OpenClaw | Claude Cowork |
| --- | --- | --- | --- | --- |
| Project lives on your machine | ✅ (workspace is a folder you own) | ❌ (cloud-hosted) | ✅ (agent workspace) | Partial (folder access, VM-sandboxed) |
| Agent can update the project | ✅ (memory, decisions, lessons maintained by the agent) | ❌ (human-only edits) | Partial (MEMORY.md, no structured state) | ❌ (session-scoped) |
| Structured project state across sessions | ✅ (PROJECT_STATE.md, DECISIONS.md, LESSONS.md) | ❌ | Partial | ❌ |
| Delegate to external CLI agents | ✅ (Claude Code, Codex, Gemini CLI, any CLI agent) | ❌ | Partial (child sessions, not external CLI) | ❌ (internal Claude sub-agents only) |
| Multiple agents share one workspace | ✅ | ❌ | ❌ | ❌ |
| Approval workflow with mobile oversight | ✅ (configurable autonomy, phone approval) | ❌ | Partial (exec-only, IM inline buttons) | ❌ |
| Per-project budget caps (real USD) | ✅ | ❌ | ❌ | ❌ (subscription-based) |
| Sandboxed execution by default | ✅ (Windows sandbox user, macOS Seatbelt) | N/A (cloud) | Opt-in (Docker, not default) | ✅ (VM, Computer Use runs outside it) |
| Triggers (cron + file watch) | ✅ | ❌ | ✅ (`openclaw cron`) | ✅ (`/schedule`) |
| Open source | GPL-3.0 | ❌ | MIT | ❌ |

**The short version:** Claude Projects proved the mental model. OpenClaw proved local agents work. Cowork proved people want agents to run autonomously. Orbital is all three in one place, on your machine, where agents can actually update the project and collaborate inside the same workspace.

---

## Feature Deep Dives

<details>
<summary><strong>Orbital Is / Is Not</strong></summary>

| Orbital **IS** | Orbital **IS NOT** |
| --- | --- |
| A project workspace where you and your agents share the same files, history, and context | A cloud service — everything runs on your machine |
| A sub-agent coordinator: Claude Code, Codex, Gemini CLI (supports [ACP](https://agentcommunicationprotocol.dev/) transport) + [claude-agent-sdk](https://github.com/anthropics/anthropic-sdk-python) | An OpenClaw fork — custom agent loop, built from scratch |
| Remote supervision: approve actions, browse workspace files, upload from phone | A chat wrapper — agents run continuously via cron and file watchers |
| Budget controls, autonomy presets, credential management (OS keychain) | Fully autonomous God Mode (yet) — scheduler-driven today, full autonomy on the roadmap |

</details>

<details>
<summary><strong>Project & Workspace Model</strong></summary>

Each project maps to a workspace directory and maintains its own sessions, triggers, and configuration.

<p align="center">
  <img src="docs/screenshots/files.png" alt="File explorer — browse and upload files in the project workspace" width="800">
</p>
<p align="center"><em>Browse, preview, and upload files in each project's workspace</em></p>

```
{workspace}/
+-- orbital/                            # Operational metadata
    +-- sessions/
    |   +-- {session_id}.jsonl          # Append-only session log
    +-- instructions/
    |   +-- project_goals.md
    |   +-- user_directives.md
    +-- skills/                         # Project skills
    +-- sub_agents/                     # Sub-agent transcripts
    +-- tool-results/                   # Tool output artifacts
    +-- output/                         # Agent work artifacts
    |   +-- screenshots/                # Browser screenshots
    |   +-- pdfs/                       # Saved PDFs
    |   +-- shell-output/               # Shell command output
    +-- PROJECT_STATE.md                # Current task state
    +-- DECISIONS.md                    # Decision log
    +-- LESSONS.md                      # Learned patterns
    +-- SESSION_LOG.md                  # Last 3 session summaries
    +-- CONTEXT.md                      # External reference material

~/orbital/                              # Home global (daemon infrastructure)
+-- daemon.pid                          # Singleton enforcement
+-- daemon-state.json                   # Agent heartbeat state
+-- device.json                         # Device identity
+-- browser-profile/                    # Shared browser profile
+-- credential-meta.json                # Credential metadata
```

**Session format**: One JSON line per message (role, source, content, timestamp, tool_calls). Append-only with file locks. Never modified except during compaction.

</details>

<details>
<summary><strong>Quick Tasks</strong></summary>

The sidebar includes a **Quick Task** section for fire-and-forget interactions. Scratch projects skip the full project creation flow — useful for one-off tasks that don't need a dedicated workspace.

<p align="center">
  <img src="docs/screenshots/quick-task.png" alt="Quick Task — browsing Hacker News and returning structured results" width="800">
</p>

</details>

<details>
<summary><strong>Autonomy & Approval System</strong></summary>

Three autonomy presets control how much supervision agents receive:

| Preset | Shell | File Write | Browser | Description |
|--------|-------|-----------|---------|-------------|
| **Hands-off** | Auto | Auto | Auto | Maximum autonomy. Only `request_access` requires approval. |
| **Check-in** | Approval | Approval | Write only | Balanced. Default for external agents. |
| **Supervised** | Approval | Approval | All except read | Maximum oversight. |

<p align="center">
  <img src="docs/screenshots/budget.png" alt="Project settings — autonomy presets and budget controls" width="700">
</p>
<p align="center"><em>Pick an autonomy level and set budget limits per project</em></p>

**Approval flow:**
1. Interceptor catches tool call based on autonomy rules
2. Frontend shows an **Approval Card** with tool name, arguments, and context
3. User can **Approve**, **Deny**, or **Auto-approve for 10 minutes**
4. Per-action bypass: same tool+args auto-approved for 60 seconds

<p align="center">
  <img src="docs/screenshots/5B2-mobile-approval-card.png" alt="Mobile approval card — approve agent actions from your phone" width="350">
</p>
<p align="center"><em>Approve agent actions from your phone — with full context and optional guidance</em></p>

</details>

<details>
<summary><strong>Sub-Agent Delegation</strong></summary>

Orbital is not tied to a single AI tool. The management agent plans and delegates, while specialized sub-agents execute. Any CLI-based agent can be registered via a manifest file.

<p align="center">
  <img src="docs/screenshots/3A-chat-file-creation.png" alt="Management agent creating a plan and delegating to sub-agents" width="800">
</p>
<p align="center"><em>The management agent creates an implementation plan...</em></p>

<p align="center">
  <img src="docs/screenshots/3B-subagent-delegation.png" alt="Sub-agent executing delegated work and reporting back" width="800">
</p>
<p align="center"><em>...delegates Phase 1 to @claudecode, monitors progress, and reviews the result</em></p>

**Transport types:**

| Transport | Use Case |
|-----------|----------|
| **Pipe** | stdin/stdout subprocess, JSON streaming |
| **PTY** | Pseudo-terminal for interactive agents — Gemini CLI, Codex, Copilot CLI, Cline, Goose |
| **SDK** | Direct Claude SDK integration |
| **ACP** | [Agent Communication Protocol](https://agentcommunicationprotocol.dev/) — supported but not the current default |

> **Note:** ACP transport is implemented in the daemon but agent manifests currently default to PTY for stability. Switching any ACP-compatible agent (Gemini CLI, Codex, Copilot CLI, Cline, Goose) to ACP is a one-line manifest change — see `docs/acp-migration.md` (coming soon).

</details>

<details>
<summary><strong>Built-in Tool Suite</strong></summary>

The management agent has access to these tool categories:

| Category | Tools | Description |
|----------|-------|-------------|
| **Shell** | `shell` | Command execution with network-aware detection |
| **File** | `read`, `write`, `edit`, `glob`, `grep` | File operations and search within workspace |
| **Browser** | 26 actions via Patchright | Navigate, click, type, extract, screenshot, multi-tab, PDF, web search, URL fetch |
| **Triggers** | `create_trigger`, `list_triggers`, `update_trigger`, `delete_trigger` | Schedule and file-watch triggers via natural language |
| **Credentials** | `request_credential` | Agent-initiated credential request — opens secure modal |
| **Delegation** | `agent_message` | Route tasks to sub-agents |
| **Access** | `request_access` | Request sandbox portal to a path outside the workspace |

</details>

<details>
<summary><strong>Browser Automation</strong></summary>

Built on **Patchright** (a Playwright fork with anti-bot-detection):

- **Stealth mode**: Anti-automation detection scripts injected into every browser context
- **Shared profile**: One browser profile across all projects — log into services once, all agents share cookies
- **Accessibility-first**: `snapshot` returns an accessibility tree with `[ref=eN]` element references for reliable interaction
- **26 browser actions**: navigate, click, type, fill, press, hover, select, drag, upload, snapshot, screenshot, extract, search (page), evaluate, tab management, go back/forward, reload, wait, PDF export, web search, URL fetch, batch

<p align="center">
  <img src="docs/screenshots/5A-mobile-browsing-activity.png" alt="Agent browsing arxiv.org and scanning research papers" width="350">
</p>
<p align="center"><em>An agent browsing arxiv.org — scanning for AI reasoning papers on a daily schedule</em></p>

</details>

<details>
<summary><strong>Continuous Operation & Triggers</strong></summary>

Agents run continuously via triggers — no manual intervention needed. Create triggers through **natural language** in the chat:

> *"Watch the uploads/ folder for new .jpg files and analyze them"*
> *"Run a research scan every morning at 6 AM"*

The management agent translates this into a `create_trigger` tool call with the appropriate type and parameters.

**Trigger types:**

| Type | Configuration | Example |
|------|--------------|---------|
| **Schedule** | Cron expression + timezone | `0 6 * * *` (daily at 6 AM) |
| **File Watch** | Path + glob patterns + debounce | `uploads/*.jpg`, 5s debounce |

<p align="center">
  <img src="docs/screenshots/file-watch-trigger.png" alt="File watch trigger — monitoring auth/ directory for .py changes" width="800">
</p>
<p align="center"><em>File watch trigger: monitors auth/ for .py changes, runs tests on every save. 22 runs so far.</em></p>

<p align="center">
  <img src="docs/screenshots/scheduled-trigger.png" alt="Scheduled trigger — daily research scan at 6 AM" width="800">
</p>
<p align="center"><em>Scheduled trigger: scans arxiv, Hacker News, and tech blogs every day at 6 AM. 12 runs.</em></p>

**Real-world example — Health Tracker with file watch:**

<p align="center">
  <img src="docs/screenshots/4B-mobile-meal-chat1.jpg" alt="Setting up a meal photo file watcher from phone" width="350">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="docs/screenshots/4B-mobile-meal-chat2.jpg" alt="Agent automatically analyzing a meal photo" width="350">
</p>
<p align="center"><em>Left: "Watch uploads/ for meal photos and track calories." Right: Drop a photo, get instant nutritional analysis.</em></p>

</details>

<details>
<summary><strong>Context Management & Compaction</strong></summary>

**Five workspace files** maintained by the LLM at session boundaries:

| File | Purpose |
|------|---------|
| `PROJECT_STATE.md` | Current task, in-progress work |
| `DECISIONS.md` | Decision log with rationale |
| `LESSONS.md` | Learned patterns and pitfalls |
| `SESSION_LOG.md` | Last 3 session summaries |
| `CONTEXT.md` | External references, API docs |

**Cold resume**: On session start, these files are assembled into the system prompt to reorient the agent — no context lost between sessions.

**Compaction** (when context usage exceeds 80%): memory flush, LLM-driven summarization of older messages, recent messages kept intact, post-compaction reorientation with project goals and current state.

**Prefix caching** (v0.4.2): the system prompt is split into static, semi-stable, and truly-dynamic sections so up to ~95% of input tokens hit the provider's prefix cache on follow-up turns. See the [v0.4.2 release notes](https://github.com/zqiren/Orbital/releases/tag/v0.4.2) for benchmark numbers.

</details>

<details>
<summary><strong>Mobile Remote Control</strong></summary>

Control agents from your phone on the local network or via a cloud relay.

<p align="center">
  <img src="docs/screenshots/4A-mobile-dashboard.png" alt="Mobile dashboard — all projects at a glance" width="350">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="docs/screenshots/5C-mobile-approved.png" alt="Agent completing work after mobile approval" width="350">
</p>
<p align="center"><em>Left: Project dashboard on phone. Right: Agent completes its research after you approve from anywhere.</em></p>

**Local network**: Scan the QR code in Settings to open Orbital on your phone via LAN.

<p align="center">
  <img src="docs/screenshots/qr-code-lan-pairng.png" alt="QR code for mobile access on local network" width="700">
</p>
<p align="center"><em>Scan to open Orbital on your phone — same Wi-Fi network required</em></p>

**Cloud relay** (optional): Deploy a relay server for access outside your home network. Push notifications for approval requests, budget alerts, and agent status changes.

</details>

<details>
<summary><strong>Cost Controls & Budget Limits</strong></summary>

Per-project budget limits prevent runaway spending:

| Setting | Description |
|---------|-------------|
| `Budget Limit (USD)` | Maximum spend for the project |
| `Budget Action` | `ask` (pause and prompt user) or `stop` (halt the agent) |
| `Spent` | Running total with reset option |

The agent loop tracks cumulative token usage and computes cost using per-model pricing from the provider registry. When the budget threshold is reached, the configured action fires and a push notification is sent.

</details>

<details>
<summary><strong>Credential Management</strong></summary>

<p align="center">
  <img src="docs/screenshots/credential-store.png" alt="Credential store — website passwords stored in system keychain" width="700">
</p>
<p align="center"><em>Website credentials stored in your system keychain. Agents always ask permission before using them.</em></p>

- **API keys**: Stored in OS keychain (`keyring`), masked in API responses, per-project BYOK override
- **Website credentials**: Metadata in `credential-meta.json`, values in OS keychain. The `request_credential` tool lets agents request credentials mid-session via a secure modal — credentials never appear in chat history.

</details>

<details>
<summary><strong>LLM Provider Routing & BYOK</strong></summary>

**13 providers** supported out of the box:

Anthropic, OpenAI, DeepSeek, Moonshot (Kimi), Groq, Google Gemini, xAI, Mistral, Together, OpenRouter, Zhipu, Qwen, plus a `custom` entry for any OpenAI-compatible endpoint (e.g., Ollama, Azure OpenAI, self-hosted models).

- **SDK routing**: Anthropic SDK for Anthropic, OpenAI SDK for OpenAI-compatible providers
- **Per-model metadata**: Display name, tier, context window, max output, capabilities (vision, tool use, streaming), pricing
- **Fallback rotation**: When the primary provider fails, the loop rotates to fallback providers with error classification (transient, rate limit, abort)

</details>

<details>
<summary><strong>Loop Safety Guards</strong></summary>

The agent loop includes multiple safety mechanisms to prevent runaway execution:

| Guard | Threshold | Behavior |
|-------|-----------|----------|
| **Token budget** | 100M tokens (configurable) | Hard stop on cumulative usage |
| **Repetition detection** | 5 identical action hashes | Forces different approach |
| **Ping-pong detection** | 3 identical consecutive pairs | Breaks alternating cycles |
| **Circuit breaker** | 2 consecutive identical errors | Blocks tool until new user message |
| **Context overflow** | 3 consecutive overflows | Hard stop after progressive reduction |

</details>

<details>
<summary><strong>Desktop App & System Tray</strong></summary>

Orbital ships as a desktop application bundled with PyInstaller:

- **System tray**: Agent activity status, quick access menu, running port in tooltip
- **Native window**: Embeds the React frontend via `pywebview` — no browser needed
- **Daemon lifecycle**: Desktop app spawns the daemon on launch, manages port allocation, cleans up on exit
- **Sleep prevention**: Blocks system sleep while agents are active (Windows `SetThreadExecutionState`), re-allows when idle

**Skills system**: Agents create reusable skills from multi-step workflows and consult matching skills before starting similar tasks. Skills are stored as SKILL.md files in the workspace and managed through the Settings UI.

<p align="center">
  <img src="docs/screenshots/skills.png" alt="Skills — operational patterns the agent follows" width="700">
</p>
<p align="center"><em>Skills like Efficient Execution, Learning Capture, and Task Planning shape how the agent works</em></p>

</details>

---

## Installation

### Windows

1. Download [`Orbital-Setup-1.0.0.exe`](https://github.com/zqiren/Orbital/releases/download/v0.5.1/Orbital-Setup-1.0.0.exe) from [Releases](https://github.com/zqiren/Orbital/releases/tag/v0.5.1)
2. Run the installer and follow the prompts
3. Launch Orbital from the Start Menu or desktop shortcut

<details>
<summary>Windows SmartScreen Warning</summary>

Orbital is not yet code-signed, so Windows will show a security warning:

> **Windows protected your PC** — Microsoft Defender SmartScreen prevented an unrecognized app from starting.

Click **"More info"** then **"Run anyway"**. Code signing will be added in a future release.
</details>

### macOS

1. Download [`Orbital-1.0.0-macOS.dmg`](https://github.com/zqiren/Orbital/releases/download/v0.5.1/Orbital-1.0.0-macOS.dmg) from [Releases](https://github.com/zqiren/Orbital/releases/tag/v0.5.1)
2. Open the DMG and drag Orbital to your Applications folder
3. Launch Orbital from Applications or Spotlight

Requires macOS 13 (Ventura) or later. Apple Silicon and Intel supported.

<details>
<summary>macOS Gatekeeper Warning</summary>

Orbital is not yet code-signed, so macOS will block it on first launch:

> **"Orbital" can't be opened because Apple cannot check it for malicious software.**

To proceed:
1. Open **System Settings → Privacy & Security**
2. Scroll down — you'll see "Orbital was blocked"
3. Click **"Open Anyway"**

This is only needed once. Code signing will be added in a future release.
</details>

### From Source

```bash
# Clone the repository
git clone https://github.com/zqiren/Orbital.git && cd Orbital

# Install Python dependencies (Python 3.11+)
pip install -e ".[desktop]"

# Install frontend dependencies (Node.js 18+)
cd web && npm install && cd ..

# Start the daemon
python -m uvicorn agent_os.api.app:create_app --factory --port 8000

# Start the frontend dev server (separate terminal)
cd web && npx vite --host 127.0.0.1 --port 5173
```

Open `http://localhost:5173` in your browser. The setup wizard runs on first launch.

### Note on Sleep/Shutdown

Orbital prevents system sleep while agents are actively working (via OS-level sleep inhibition on Windows and macOS). When all agents are idle, sleep is re-allowed. The system tray icon shows current agent activity status.

---

## Development

### Backend

```bash
# Start daemon
python -m uvicorn agent_os.api.app:create_app --factory --port 8000

# Restart with fresh code
bash scripts/restart-daemon.sh
```

### Frontend

```bash
cd web
npm install
npx vite --host 127.0.0.1 --port 5173
```

### Key Paths

| Component | Path |
|-----------|------|
| FastAPI app factory | `agent_os/api/app.py` |
| Agent loop | `agent_os/agent/loop.py` |
| Tool implementations | `agent_os/agent/tools/` |
| Autonomy interceptor | `agent_os/daemon_v2/autonomy.py` |
| LLM providers | `agent_os/agent/providers/` |
| Trigger manager | `agent_os/daemon_v2/trigger_manager.py` |
| Browser manager | `agent_os/daemon_v2/browser_manager.py` |
| Desktop entry point | `agent_os/desktop/main.py` |
| System tray | `agent_os/desktop/tray.py` |
| Frontend components | `web/src/components/` |

---

## Testing

```bash
# Unit + platform tests
python -m pytest tests/unit/ tests/platform/ -q \
  --ignore=tests/platform/test_consumer3_wiring.py

# TypeScript check (zero errors expected)
cd web && npx tsc --noEmit

# Daemon integration test
bash scripts/restart-daemon.sh
curl http://localhost:8000/api/v2/projects
```

**Known pre-existing test notes:**
- `test_consumer3_wiring.py` — requires Windows sandbox user configuration
- `test_e2e.py`, `test_user_stories.py` — require a real LLM API key set via `AGENT_OS_TEST_API_KEY`

---

## Roadmap

### Shipped

- Multi-provider LLM routing with fallback rotation
- Three autonomy presets with cascade to sub-agents
- Streaming chat with real-time WebSocket events
- Browser automation with anti-detection (Patchright)
- Continuous operation via schedule and file-watch triggers
- Natural language trigger creation
- Cloud relay with push notifications and device pairing
- Context compaction with pre-compaction memory flush
- Prefix-cache-optimized prompt assembly (v0.4.2)
- Per-project budget limits and cost tracking
- Credential management (API keys + website credentials)
- Desktop app with system tray and native window
- Agent loop safety guards (iteration cap, repetition, ping-pong, circuit breaker)
- OS-level sleep prevention during agent activity
- Sub-agent delegation with @mention routing

### Next

- **Webhook triggers** — HTTP endpoint that fires agent tasks on incoming webhooks
- **Pipeline triggers** — Chain project outputs as inputs to other projects
- **Network isolation** — Per-project domain allowlists enforced at OS level
- **Linux sandboxing** — bubblewrap enforcement
- **Code signing** — Eliminate SmartScreen warnings on Windows
- **Auto-resume on daemon restart** — Restore in-progress sessions

---

## Why I built this

I loved Claude Projects. I hated that I couldn't let an agent update the project, and that it didn't live on my machine.

I loved OpenClaw. I hated the lack of control — no budget, no sandbox, no way to supervise from my phone when I stepped away.

Orbital is the thing I wanted. A project you can hand to an agent. Memory that persists. Sandbox, budget, approvals you control. The phone to check in when you're not at your desk. Claude Code, Codex, Gemini CLI working as sub-agents in the same workspace.

Built nights and weekends while working full-time. Still very early. Feedback and issues welcome.

---

## License

Orbital is licensed under the [GNU General Public License v3.0](LICENSE).

```
Orbital — The project workspace you and your agent share.
Copyright (C) 2026 Orbital Contributors

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
```
