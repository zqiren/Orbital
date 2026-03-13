# Agent OS: API Reference for Frontend & Cloud Relay

*Generated: 2026-02-11*
*Status: ACTIVE — authoritative reference for all consumers of the daemon API*

This document contains everything a frontend or cloud relay developer needs to connect to the Agent OS daemon. No Python internals, no class hierarchies — just HTTP endpoints, WebSocket events, JSON shapes, and lifecycle rules.

---

## Connection Model

```
Frontend (Electron / Mobile Web / Cloud Relay)
    ↕ REST (HTTP) + WebSocket (WS)
Daemon (localhost:8000 or relay)
```

- **REST** for actions: create project, start/stop agent, inject messages, approve/deny, read chat
- **WebSocket** for real-time events: streaming tokens, agent activity, status changes, approval requests

Both interfaces use **snake_case** exclusively for all field names. No exceptions.

---

## REST API

Base path: `/api/v2/`

All request and response bodies are JSON. Error responses use `{"detail": "..."}`.

---

### POST /api/v2/projects — Create Project

Creates a new project. Returns the full project object.

**Request:**
```json
{
  "name": "My App",
  "workspace": "/Users/dev/my-app",
  "model": "claude-sonnet-4-5-20250929",
  "api_key": "sk-...",
  "base_url": "https://api.anthropic.com/v1",
  "autonomy": "hands_off",
  "instructions": "Focus on the backend API"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Display name for the project |
| `workspace` | string | yes | Absolute path to the workspace folder. Must exist on disk. |
| `model` | string | yes | LLM model identifier (e.g. `claude-sonnet-4-5-20250929`, `gpt-4o`, `kimi-k2.5`) |
| `api_key` | string | yes | API key for the LLM provider |
| `base_url` | string | no | Custom API base URL. If set, uses OpenAI SDK direct. If omitted, uses LiteLLM routing. |
| `autonomy` | string | no | `"hands_off"` (default), `"check_in"`, or `"supervised"`. Controls which tool calls require user approval. |
| `instructions` | string | no | Project-specific instructions injected into the agent's system prompt. |

**Response: 201 Created**
```json
{
  "project_id": "proj_a1b2c3d4e5f6",
  "name": "My App",
  "workspace": "/Users/dev/my-app",
  "model": "claude-sonnet-4-5-20250929",
  "api_key": "sk-...",
  "base_url": "https://api.anthropic.com/v1",
  "autonomy": "hands_off",
  "instructions": "Focus on the backend API"
}
```

**Error: 400**
```json
{"detail": "Workspace path does not exist"}
```

---

### GET /api/v2/projects — List Projects

**Response: 200**
```json
[
  {
    "project_id": "proj_a1b2c3d4e5f6",
    "name": "My App",
    "workspace": "/Users/dev/my-app",
    "model": "claude-sonnet-4-5-20250929",
    "autonomy": "hands_off"
  }
]
```

---

### POST /api/v2/agents/start — Start Agent

Starts the management agent loop for a project. The agent runs as a background task.

**Request:**
```json
{
  "project_id": "proj_a1b2c3d4e5f6",
  "initial_message": "Read the codebase and summarize the architecture"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `project_id` | string | yes | Project to start the agent on |
| `initial_message` | string | no | First user message. If omitted, agent resumes from existing session. |

**Response: 200**
```json
{"status": "started"}
```

**Error: 404** — Project not found
**Error: 400** — Agent already running for this project

---

### POST /api/v2/agents/{project_id}/inject — Send Message

Sends a message to a running or idle agent.

**Request:**
```json
{
  "content": "Now refactor the auth module",
  "target": "claudecode"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `content` | string | yes | The message text |
| `target` | string | no | Sub-agent handle for @mention routing. Omit for management agent. |

**Response: 200**
```json
{"status": "queued"}
```

| Status | Meaning |
|--------|---------|
| `"queued"` | Agent loop is running. Message will be processed between tool calls. |
| `"delivered"` | Agent loop was idle. New loop started with this message. |

**Error: 404** — No active session for project

---

### POST /api/v2/agents/{project_id}/stop — Stop Agent

Signals the agent to stop. The agent finishes its current operation, resolves pending tool calls, and halts.

**Response: 200**
```json
{"status": "stopping"}
```

The actual `stopped` confirmation arrives via WebSocket as an `agent.status` event.

**Error: 404** — No active session for project

---

### POST /api/v2/agents/{project_id}/approve — Approve Tool Call

Approves a pending tool call that was intercepted by the autonomy system.

**Request:**
```json
{
  "tool_call_id": "call_abc123"
}
```

**Response: 200**
```json
{"status": "approved"}
```

After approval, the agent loop resumes and re-executes the approved tool. Same tool+args combinations are auto-approved for 60 seconds (bypass window).

---

### POST /api/v2/agents/{project_id}/deny — Deny Tool Call

Denies a pending tool call. The agent receives the denial reason and adjusts its approach.

**Request:**
```json
{
  "tool_call_id": "call_abc123",
  "reason": "Do not modify production config"
}
```

**Response: 200**
```json
{"status": "denied"}
```

---

### GET /api/v2/agents/{project_id}/chat — Get Chat History

Returns the full conversation history for the current session.

**Response: 200**
```json
[
  {
    "role": "user",
    "content": "Read the codebase and summarize the architecture",
    "source": "user",
    "timestamp": "2026-02-11T14:30:00.000Z"
  },
  {
    "role": "assistant",
    "content": null,
    "source": "management",
    "timestamp": "2026-02-11T14:30:05.000Z",
    "tool_calls": [
      {
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "read",
          "arguments": "{\"path\": \"src/main.py\"}"
        }
      }
    ],
    "_status": "Using read"
  },
  {
    "role": "tool",
    "content": "def main():\n    print('hello')\n",
    "tool_call_id": "call_abc123",
    "source": "management",
    "timestamp": "2026-02-11T14:30:06.000Z"
  },
  {
    "role": "assistant",
    "content": "The codebase has a single entry point in src/main.py...",
    "source": "management",
    "timestamp": "2026-02-11T14:30:10.000Z"
  }
]
```

**Error: 404** — No active session for project

### Message Format

Every message in the chat array has these fields:

| Field | Type | Present | Description |
|-------|------|---------|-------------|
| `role` | string | always | `"user"`, `"assistant"`, `"tool"`, `"agent"`, `"system"` |
| `content` | string or null | always | Message text. Null for assistant messages that only contain tool_calls. |
| `source` | string | always | Who produced this: `"user"`, `"management"`, or a sub-agent handle like `"claudecode"` |
| `timestamp` | string | always | ISO 8601 timestamp |
| `target` | string | sometimes | Who the message is for (sub-agent handle). Absent = management agent. |
| `tool_calls` | array | assistant only | Raw LLM tool_calls when agent invokes tools. See Tool Call Format below. |
| `tool_call_id` | string | tool only | Binds this result to the tool_call that produced it. |
| `_status` | string | assistant only | Human-readable status (e.g. `"Using read"`, `"Editing file"`). For UI status indicators. |
| `_meta` | object | tool only | Optional metadata from tool execution (e.g. `{"network": true, "domains": ["github.com"]}`). |

### Tool Call Format (inside `tool_calls` array)

```json
{
  "id": "call_abc123",
  "type": "function",
  "function": {
    "name": "read",
    "arguments": "{\"path\": \"src/main.py\"}"
  }
}
```

Note: `arguments` is a **JSON string**, not a parsed object. This is the raw LLM format. Parse it on the frontend if you need structured access.

### Role Guide

| Role | Source | Meaning |
|------|--------|---------|
| `user` | `"user"` | Human input |
| `assistant` | `"management"` | Management agent response (text or tool calls) |
| `tool` | `"management"` | Tool execution result |
| `agent` | `"{handle}"` | Sub-agent output (e.g. Claude Code) |
| `system` | `"management"` | System messages (iteration limits, errors) |

---

## WebSocket Protocol

### Connection

Connect to: `ws://localhost:8000/ws` (or `wss://` via relay)

After connecting, subscribe to projects you want events for:

```json
→ {"type": "subscribe", "project_ids": ["proj_a1b2c3d4e5f6"]}
← {"type": "subscribed", "project_ids": ["proj_a1b2c3d4e5f6"]}
```

You can re-subscribe at any time to change which projects you receive events for. Each subscribe replaces the previous subscription set.

---

### Event: agent.status

Agent lifecycle state transitions.

```json
{
  "type": "agent.status",
  "project_id": "proj_a1b2c3d4e5f6",
  "status": "running",
  "source": "management"
}
```

```json
{
  "type": "agent.status",
  "project_id": "proj_a1b2c3d4e5f6",
  "status": "error",
  "reason": "LLM error after 3 retries: rate limit exceeded"
}
```

| Status | Meaning |
|--------|---------|
| `running` | Agent loop started or resumed |
| `idle` | Agent loop completed normally (waiting for next message) |
| `stopped` | Agent was stopped by user |
| `error` | Agent loop crashed (see `reason` field) |

---

### Event: chat.stream_delta

Real-time LLM token streaming. Use these to show the agent's response as it's being generated.

```json
{
  "type": "chat.stream_delta",
  "project_id": "proj_a1b2c3d4e5f6",
  "text": "The",
  "source": "management",
  "is_final": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | The text chunk (may be a single token or word) |
| `source` | string | `"management"` for the management agent |
| `is_final` | boolean | `true` on the last chunk of the response |

**Frontend implementation:** Concatenate `text` fields in order to build the streamed response. When `is_final` is true, the response is complete. Note: the full response also appears in chat history (via REST) after streaming finishes.

---

### Event: agent.activity

Tool usage and results. Use these to show "Agent is reading file...", "Agent is running command...", etc.

```json
{
  "type": "agent.activity",
  "project_id": "proj_a1b2c3d4e5f6",
  "id": "evt_abc123",
  "category": "file_read",
  "description": "Reading src/main.py",
  "tool_name": "read",
  "source": "management",
  "timestamp": "2026-02-11T14:30:05.000Z"
}
```

| Category | Trigger |
|----------|---------|
| `file_read` | Agent reads a file |
| `file_write` | Agent writes a file |
| `file_edit` | Agent edits a file |
| `command_exec` | Agent runs a shell command |
| `web_search` | Agent searches the web |
| `web_fetch` | Agent fetches a URL |
| `request_access` | Agent requests access to a resource |
| `agent_message` | Agent sends a message to a sub-agent |
| `tool_result` | A tool result was received |
| `tool_use` | Any other tool call |
| `agent_output` | Sub-agent produced output |

---

### Event: approval.request

The agent wants to execute a tool that requires user approval. Show an approval card in the UI.

```json
{
  "type": "approval.request",
  "project_id": "proj_a1b2c3d4e5f6",
  "what": "Agent wants to run: shell",
  "tool_name": "shell",
  "tool_call_id": "call_xyz789",
  "tool_args": {
    "command": "rm -rf node_modules && npm install"
  },
  "recent_activity": [
    {"role": "assistant", "content": "I need to reinstall dependencies..."}
  ]
}
```

The frontend should render this as an approval card with "Approve" and "Deny" buttons. Use `tool_call_id` when calling the approve/deny REST endpoints.

---

### Event: approval.resolved

An approval request was resolved (approved or denied).

```json
{
  "type": "approval.resolved",
  "project_id": "proj_a1b2c3d4e5f6",
  "resolution": "approved"
}
```

---

## Agent Lifecycle State Machine

```
                    ┌──────────────────────────┐
                    │                          │
                    ▼                          │
  POST /start → [running] ──────────────→ [idle]
                    │                     ▲    │
                    │                     │    │ POST /inject (delivered)
                    │                     │    └──→ [running]
                    │                     │
                    │  (interceptor)      │
                    ▼                     │
              [paused for approval]       │
                    │                     │
            ┌───────┴───────┐             │
            │               │             │
     POST /approve    POST /deny          │
            │               │             │
            └───────┬───────┘             │
                    │                     │
                    └─────────────────────┘

  POST /stop → [stopping] → [stopped]

  (unrecoverable error) → [error]
```

### Key transitions:
- **start** → `running` (broadcasts `agent.status: running`)
- **loop completes** → `idle` (broadcasts `agent.status: idle`)
- **inject while idle** → `running` again (hot resume on same session)
- **inject while running** → message queued, processed between tool calls
- **intercepted tool** → `paused` (broadcasts `approval.request`)
- **approve/deny** → `running` again (loop resumes)
- **stop** → `stopping` → `stopped` (broadcasts `agent.status: stopped`)
- **unrecoverable error** → `error` (broadcasts `agent.status: error` with `reason`)

---

## Autonomy Presets

Controls which tool calls pause for user approval.

| Preset | Approves automatically | Requires approval |
|--------|----------------------|-------------------|
| `hands_off` | All tools except `request_access` | `request_access` only |
| `check_in` | `read`, `edit`, `web_search`, `web_fetch`, `agent_message` | `shell`, `write`, `request_access` |
| `supervised` | `read` only | Everything else |

After a tool+args combination is approved, it auto-approves for 60 seconds (bypass window).

---

## Available Tools

These are the tools the management agent can use. Frontend should be aware of them for activity rendering and approval cards.

| Tool | Description | Takes workspace | Activity category |
|------|-------------|-----------------|-------------------|
| `read` | Read a file from workspace | yes | `file_read` |
| `write` | Write/create a file in workspace | yes | `file_write` |
| `edit` | Search-and-replace edit on a file | yes | `file_edit` |
| `shell` | Execute a shell command | yes | `command_exec` |
| `web_search` | Search the web (requires search API key) | no | `web_search` |
| `web_fetch` | Fetch and extract content from a URL | no | `web_fetch` |
| `request_access` | Request access to a portal/resource | no | `request_access` |
| `agent_message` | Send a message to a sub-agent | no | `agent_message` |

---

## Workspace & Session Storage

```
{workspace}/
├── .agent-os/
│   ├── sessions/
│   │   └── {session_id}.jsonl     ← conversation log (append-only)
│   ├── instructions/              ← user-provided context (*.md)
│   ├── reference/                 ← agent-readable reference files
│   ├── PROJECT_STATE.md           ← agent's long-term memory
│   └── DECISIONS.md               ← agent's decision log
├── src/                           ← user's actual project files
└── ...
```

- **Session JSONL** is the single source of truth for conversation state
- **PROJECT_STATE.md** and **DECISIONS.md** are written by the agent for cold resume (new session boots from these)
- **instructions/*.md** are always injected into the system prompt (Layer 3 context)

---

## Cloud Relay Notes

For the cloud relay (mobile access over internet), the relay acts as a transparent proxy:

1. **REST** — Forward all `/api/v2/*` requests to the local daemon. The relay adds authentication but does not modify request/response bodies.
2. **WebSocket** — Maintain a persistent WS connection to the daemon. Forward events to authenticated mobile clients. The subscribe/broadcast protocol is the same.
3. **Authentication** — The relay handles auth (JWT, OAuth, etc). The daemon does not authenticate — it trusts localhost.
4. **API keys** — The `api_key` field in create_project is the LLM provider key, NOT the relay auth token. These are separate concerns. The relay should never expose `api_key` to the mobile client.

---

## Quick Start Example (Frontend)

```javascript
// 1. Create project
const project = await fetch('/api/v2/projects', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    name: 'My App',
    workspace: '/Users/dev/my-app',
    model: 'claude-sonnet-4-5-20250929',
    api_key: 'sk-...',
    autonomy: 'hands_off'
  })
}).then(r => r.json());

// 2. Connect WebSocket and subscribe
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'subscribe',
    project_ids: [project.project_id]
  }));
};

// 3. Handle events
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  switch (data.type) {
    case 'subscribed':
      console.log('Subscribed to', data.project_ids);
      break;
    case 'agent.status':
      updateStatusBadge(data.status);  // "running", "idle", "stopped", "error"
      break;
    case 'chat.stream_delta':
      appendToStreamingBubble(data.text);
      if (data.is_final) finalizeStreamingBubble();
      break;
    case 'agent.activity':
      showActivityIndicator(data.description);  // "Reading src/main.py"
      break;
    case 'approval.request':
      showApprovalCard(data);  // tool_name, tool_args, tool_call_id
      break;
  }
};

// 4. Start agent
await fetch('/api/v2/agents/start', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    project_id: project.project_id,
    initial_message: 'Analyze the codebase and suggest improvements'
  })
});

// 5. Send follow-up message
await fetch(`/api/v2/agents/${project.project_id}/inject`, {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({content: 'Focus on the auth module specifically'})
});

// 6. Handle approval
async function onApprove(toolCallId) {
  await fetch(`/api/v2/agents/${project.project_id}/approve`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({tool_call_id: toolCallId})
  });
}

async function onDeny(toolCallId, reason) {
  await fetch(`/api/v2/agents/${project.project_id}/deny`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({tool_call_id: toolCallId, reason: reason})
  });
}

// 7. Get full chat history
const messages = await fetch(`/api/v2/agents/${project.project_id}/chat`)
  .then(r => r.json());
```

---

## Verified By

All interfaces documented here have been verified by the following test suites:

- **331 unit tests** — component-level isolation tests
- **4 wiring tests** — real LLM integration (kimi-k2.5)
- **16 e2e tests** — full daemon stack with real LLM
- **7 user story tests** — end-to-end user workflows with full tracing

See `docs/user-story-test-report.md` for detailed traces of real user workflows against these interfaces.
