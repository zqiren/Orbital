# FINDINGS — Sub-Agent Context & Persistence

**Investigation:** TASK-investigate-sub-agent-context-and-persistence.md
**Date:** 2026-05-10
**Test machine:** Windows 10 Home, Git Bash + PowerShell. Cwd-resolved scratch at `/tmp/orbital-investigation/` (= `C:\Users\qiren\AppData\Local\Temp\orbital-investigation\`).
**Versions under test:**

| Tool | Version | Auth state |
|---|---|---|
| `claude` (Anthropic Claude Code) | 2.1.138 | logged in via Claude.ai (`max` subscription) |
| `codex` (OpenAI Codex CLI) | 0.114.0 | **not credentialed** (returns 401) |
| `gemini` | not installed | n/a |

**Common probe shape:** `env -u CLAUDECODE claude -p [opts] "<prompt>" < /dev/null`. `--model haiku` was used to bound cost; spot-checks against the default model showed identical observable behavior.

---

## Tier 1 — claude-code SDK (empirical)

### Q1. What does the SDK preserve across `--resume` with the same `sessionId`?

**Finding: full message history is preserved across process death and reuse.**

```bash
SID=$(python -c "import uuid; print(uuid.uuid4())")
# Turn 1 (process A)
env -u CLAUDECODE claude -p --session-id "$SID" --model haiku --output-format json \
  "Remember this number: 84217. Reply with exactly the words: OK, remembered." \
  > turn1.json
# Process A exits.
# Turn 2 (fresh process B)
env -u CLAUDECODE claude -p --resume "$SID" --model haiku --output-format json \
  "What number did I ask you to remember in the previous turn? Reply with just the digits."
```

Turn 2 result: `'84217'`. `session_id` echoed in both responses is identical. Sessions are persisted on disk at `~/.claude/projects/<sanitized-cwd>/<session-uuid>.jsonl` — JSONL of every user/assistant/tool message, plus queue-ops and `last-prompt` markers.

### Q2. Does claude-code read `CLAUDE.md` from cwd at session start?

**Finding: yes — and it walks ancestors all the way to `~/.claude/CLAUDE.md`.**

| Setup | Prompt | Result |
|---|---|---|
| `CLAUDE.md` in cwd with `"shibboleth" -> PINEAPPLE-9847-CTX` | `shibboleth` | `'PINEAPPLE-9847-CTX'` ✓ |
| Empty cwd, no nearby CLAUDE.md | `shibboleth` | generic explanation referencing **superpowers framework** (= `~/.claude/CLAUDE.md`) leaked in |
| `CLAUDE.md` in *parent* dir, child as cwd | `ancestor-test` | `'ELDERBERRY-3120-PARENT'` ✓ |

Confirms (a) cwd CLAUDE.md is auto-discovered and (b) ancestor walk reaches all the way to the user-level `~/.claude/CLAUDE.md`. **Implication for orbital: the sub-agent inherits the user's personal Claude config by default; if orbital doesn't want that, it must spawn from a workspace dir whose ancestor chain doesn't traverse `$HOME` — but that's effectively impossible on Unix/macOS since everything is under `/`. The only suppression knob documented is `--bare`, which requires an API key (OAuth incompatible).**

`--bare` was rejected for further probing because it disables OAuth login on this machine; on a sub-agent that orbital launches in production, `--bare` plus an explicit `ANTHROPIC_API_KEY` would be the only clean way to suppress ancestor CLAUDE.md walk.

### Q3. SDK given `--system-prompt` (or `--append-system-prompt`) AND CLAUDE.md exists in cwd — what wins?

**Finding: CLAUDE.md wins on direct conflict, every time.**

| Probe | CLAUDE.md says | `--system-prompt` says | Reply |
|---|---|---|---|
| Q3.A baseline | `color → BLUE-FROM-CLAUDEMD` | (none) | `'BLUE-FROM-CLAUDEMD'` |
| Q3.B append-conflict | `color → BLUE-FROM-CLAUDEMD` | `--append-system-prompt 'color → RED-FROM-APPEND'` | `'BLUE-FROM-CLAUDEMD'` |
| Q3.C full-replace | `color → BLUE-FROM-CLAUDEMD` | `--system-prompt 'color → GREEN-FROM-SYS'` | `'BLUE-FROM-CLAUDEMD'` |
| Q3.D non-conflict + CLAUDE.md | `color → BLUE-FROM-CLAUDEMD` | `--system-prompt 'be terse'` | `'BLUE-FROM-CLAUDEMD'` |
| Q3.E append on a NEW word | (silent on `fruit`) | `--append-system-prompt 'fruit → KIWI-FROM-APPEND'` | `'KIWI-FROM-APPEND'` |
| Q3.F sys-prompt on a NEW word | (silent on `shape`) | `--system-prompt 'shape → HEX-FROM-SYS'` | `'HEX-FROM-SYS'` |

So both flags work — they inject instructions the model honors — but **CLAUDE.md beats both on direct contradiction** even when `--system-prompt` claims to *replace* the default. This is consistent with CLAUDE.md being injected as a separate, model-facing instruction block (likely tied to the user-message turn) rather than as part of the replaceable default system prompt.

**Implication: orbital cannot use `--system-prompt` to override project-level CLAUDE.md content. It can only ADD to it. Use CLAUDE.md as the authoritative injection point.**

### Q4. ACP transport — same probes

**Finding: ACP cannot be empirically tested against claude-code 2.1.138 — it has no ACP server mode.**

```bash
env -u CLAUDECODE claude acp
# ✘ unknown command "acp"
#   └ Did you mean claude mcp?
```

Orbital's `ACPTransport` (`agent_os/agent/transports/acp_transport.py`) sends `claude` with `args=["acp"]`, expecting JSON-RPC 2.0 over stdio. The current production claude binary rejects this. Orbital's existing ACP unit tests (`tests/unit/test_acp_transport.py`) and e2e test (`tests/test_e2e_acp.py`) use a Python `dummy_acp_agent.py` fixture, not real claude-code — so the regression has been masked.

The README documents ACP as the gemini-cli pathway: *"Claude Code, Codex, Gemini CLI (supports ACP transport)"*. Empirically validating ACP requires installing gemini-cli, which the spec explicitly forbids ("do NOT install new agents to expand the test matrix"). **Therefore ACP-specific Q1/Q2/Q3/Q5 answers are deferred until a gemini-cli-equipped probe machine is available.**

Negative finding flagged for the architecture: **if claude-code remains the default sub-agent, the ACP transport is dead code on the claude-code path; orbital should route claude-code through SDKTransport (which already works), and ACP should be tagged as gemini-only.**

### Q5. Does the SDK persist tool-call history across resume?

**Finding: yes — file paths, tool inputs, and tool results all survive `--resume`.**

```bash
echo "marker: KOALA-OBSERVATORY-1138" > secret.txt
SID=$(python -c "import uuid; print(uuid.uuid4())")
env -u CLAUDECODE claude -p --session-id "$SID" --model haiku --output-format json \
  --permission-mode bypassPermissions \
  "Use the Read tool to read 'secret.txt'. Tell me what it contains."
# → "...unique marker token: KOALA-OBSERVATORY-1138..."
env -u CLAUDECODE claude -p --resume "$SID" --model haiku --output-format json \
  "What file did you read in the previous turn, and what was the unique marker?"
# → "I read secret.txt, which contained the unique marker token: KOALA-OBSERVATORY-1138."
```

The session JSONL on disk contains both the `tool_use` (Read input) and `tool_result` (file content) blocks. **Implication: orbital does NOT need to build a parallel session log of sub-agent tool history — Claude Code already does it durably.**

### Q6. When orbital daemon restarts, does the SDK session ID survive?

**Finding: yes, IF the cwd at resume matches the cwd at creation. Otherwise resume fails.**

Sessions are keyed by `~/.claude/projects/<sanitized-cwd>/<session-uuid>.jsonl`. Sanitization replaces `\` and `/` with `-`.

| Resume context | Outcome |
|---|---|
| Same cwd as creation, after delay | ✓ `'84217'` (Q6.C) |
| **Different cwd** (resume from sibling dir with same sessionId) | ✗ `No conversation found with session ID: <uuid>` (Q6.B) |

So orbital daemon restarts are safe **as long as the sub-agent workspace cwd is preserved**. If orbital ever relocates a sub-agent's workspace (move project folder, switch drive letter, etc.), the session is unrecoverable. **Implication: store both `sessionId` AND `workspace_cwd` in orbital's per-sub-agent state. Validate cwd matches before attempting resume; if not, treat as session-loss and start fresh.**

### Q7. What is the practical max for `system_prompt` size before degradation?

**Finding: 1KB / 5KB / 10KB / 20KB inline all work. 50KB inline fails at the OS argv limit on Windows. File-based variant (`--append-system-prompt-file`) accepts 50KB without issue.**

| Size | Mechanism | Result | `cache_creation_tokens` |
|---|---|---|---|
| 1 KB | `--append-system-prompt` | `'SIZE-1KB-OK'` | 7,928 |
| 5 KB | `--append-system-prompt` | `'SIZE-5KB-OK'` | 8,815 |
| 10 KB | `--append-system-prompt` | `'SIZE-10KB-OK'` | 9,924 |
| 20 KB | `--append-system-prompt` | `'SIZE-20KB-OK'` | 12,143 |
| 50 KB | `--append-system-prompt` (inline) | **OS error: `Argument list too long`** | n/a |
| 50 KB | `--append-system-prompt-file <path>` | `'SIZE-50KB-FILE-OK'` | 18,801 |

The Anthropic API itself has no problem with 50 KB system prompts — the failure was at the Windows shell argv layer. **Implication: orbital should always pass large prompts via the `-file` variants (`--system-prompt-file`, `--append-system-prompt-file`), never inline.** The `-file` flags are not in `claude --help` but are referenced in the `--bare` description and were verified to work on 2.1.138.

---

## Tier 2 — codex (skipped, documented)

The codex CLI is installed (`codex-cli 0.114.0`) but not credentialed on this machine — `codex exec` returns `401 Unauthorized: Missing bearer or basic authentication in header`. Per Tier 2 spec ("Run only if codex is installed and credentialed on the test machine. If not, document this and skip."), no Q1/Q2/Q3 results were collected.

What was extracted from `codex --help` for Tier 5 below: codex has its own `--config` override flag (TOML), `codex exec` for non-interactive, `codex exec resume <SESSION_ID> [PROMPT]` for resume, `codex resume`/`codex fork` interactive variants, `--all` flag explicitly notes that it "disables cwd filtering" — i.e. **codex also scopes resume by cwd by default, just like claude-code**.

---

## Tier 3 — documentation only

### gemini-cli (Google)

1. **Context file:** `GEMINI.md` (default, configurable via `context.fileName` in `settings.json`).
2. **Lookup locations (concatenated, not ranked):**
   1. Global: `~/.gemini/GEMINI.md`
   2. Workspace + ancestor walk
   3. Just-in-time: when a tool touches a file/dir, the CLI scans ancestors of that path up to a trusted root.
3. **Precedence:** *"loads files in the following order: 1. Global … 2. Environment/workspace … 3. JIT"* — but they are concatenated, not overridden. Conflict semantics are not documented.
4. **System-prompt CLI flag:** **none.** System-prompt customization is environment-driven only:
   - `GEMINI_SYSTEM_MD=true|1` → use `.gemini/system.md` as project default
   - `GEMINI_SYSTEM_MD=/path/to/file` → custom path
   - `GEMINI_SYSTEM_MD=false|0` → disable
   - "The override is a full replacement, not a merge."
5. **Resume:** `gemini --resume` (most recent), `gemini --resume <index>`, `gemini --resume <uuid>`, plus `--list-sessions` / `--delete-session`. No `--continue` / `-c` (those are claude-code conventions).

Sources: docs/cli/gemini-md.md, docs/cli/system-prompt.md, docs/cli/session-management.md, docs/cli/cli-reference.md (all on `main`).

### cline (CLI surface)

1. **Context file:** rule files. `.clinerules` (single file) or `.clinerules/` (directory of `.md`/`.txt` — *all files concatenated*). Also auto-detected: `.cursorrules`, `.windsurfrules`, `AGENTS.md`. Notably **`CLAUDE.md` is NOT documented as a recognized format**.
2. **Lookup locations (no documented ancestor walk):**
   - Workspace rules: `.clinerules` "at your project root."
   - Global rules: per-OS user dir (`Documents\Cline\Rules` on Windows, `~/Documents/Cline/Rules` on macOS/Linux; an open issue notes `~/Cline/Rules` is also seen in practice).
3. **Precedence:** *"Workspace rules take precedence when they conflict with global rules."* All recognized files appear in the Rules panel and are merged unless toggled off.
4. **System-prompt CLI flag:** **none documented.** No `--system-prompt`, `--instructions`, `--rules`. Custom instructions come exclusively from rule files plus GUI/`cline config`.
5. **Resume:** `--continue` (most recent task) or `--taskId <id>` / `-T <id>`. No `--resume` or `--session-id`.

Source: docs.cline.bot/customization/cline-rules, docs.cline.bot/cline-cli/getting-started.

### GitHub Copilot CLI

Two surfaces. `gh copilot` (suggest/explain extension) is **deprecated, EOL 2025-10-25** — stateless, no instruction file, no resume. The new agentic Copilot CLI (`@github/copilot`, GA 2026-02) is what matters:

1. **Context files:** `AGENTS.md` (primary, also recognized: `CLAUDE.md`, `GEMINI.md`), `.github/copilot-instructions.md`, `.github/instructions/**/*.instructions.md` (path-specific with YAML frontmatter). Disable with `--no-custom-instructions`.
2. **Lookup locations:** repo root, `.github/`, cwd, nested dirs, `$HOME/.copilot/copilot-instructions.md`, custom dirs via `COPILOT_CUSTOM_INSTRUCTIONS_DIRS` (comma-separated).
3. **Precedence:** *"Instructions in the AGENTS.md file in the root directory, if found, are treated as primary instructions"*; nested AGENTS.md files are *"additional"*; *"Copilot's choice between conflicting instructions is non-deterministic"* — explicit ambiguity documented.
4. **System-prompt CLI flag:** **none.** Closest is `-p PROMPT` / `--prompt=PROMPT` (one-shot user prompt, not system), `--add-dir=PATH` (file-access scope, not instructions).
5. **Resume:** `--resume[=VALUE]` (picker or by ID), `--continue` (most recent in cwd), `/resume` slash command. Sessions persisted at `~/.copilot/session-state/<id>/` with `events.jsonl` + `workspace.yaml`, plus a SQLite store.

Sources: docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/add-custom-instructions, docs.github.com/en/copilot/concepts/agents/copilot-cli/chronicle, github/copilot-cli repo.

---

## Tier 5 — invocation parameters & login flow

### claude-code

| Parameter | Flag / mechanism | Default | Stability |
|---|---|---|---|
| Model | `--model <alias\|id>` (e.g. `haiku`, `sonnet`, `opus`, `claude-sonnet-4-6`) | user/system config (`opus` for max sub) | stable |
| Effort / "thinking" | `--effort <low\|medium\|high\|xhigh\|max>` | session config | stable |
| Reasoning trace | `--include-partial-messages` (with `--print --output-format=stream-json`) | off | stable |
| Max budget (cost cap) | `--max-budget-usd <n>` (only with `--print`) | unset | stable |
| Fallback model | `--fallback-model <model>` (only with `--print`) | unset | stable |
| Tool whitelist | `--allowedTools "Bash(git *) Edit"` | full default set | stable |
| Tool blacklist | `--disallowedTools <list>` | empty | stable |
| Tool subset | `--tools <list>` (`""` = none, `"default"` = all, or names) | default | stable |
| Permission mode | `--permission-mode <acceptEdits\|auto\|bypassPermissions\|default\|dontAsk\|plan>` | `default` (interactive prompts) | stable |
| Skip-permissions kill switch | `--dangerously-skip-permissions` | off | stable, scary-named |
| System prompt (REPLACE) | `--system-prompt <prompt>` or `--system-prompt-file <path>` | Claude Code default | stable; `-file` variant undocumented in `--help` but works on 2.1.138 |
| System prompt (APPEND) | `--append-system-prompt <prompt>` or `--append-system-prompt-file <path>` | none | same as above |
| Session ID (NEW) | `--session-id <uuid>` | auto-generated | stable |
| Resume | `--resume <uuid\|title>` (cwd-scoped) | n/a | stable |
| Continue | `-c` / `--continue` (most recent in cwd) | n/a | stable |
| Suppress CLAUDE.md auto-discovery | `--bare` (also disables hooks/LSP/plugins/auto-memory; requires API key, not OAuth) | off | stable |
| Custom agents | `--agents '{"name": {"description":"...","prompt":"..."}}'` | empty | stable |
| MCP config | `--mcp-config <files...>`, `--strict-mcp-config` | merged from settings | stable |
| Settings | `--settings <file-or-json>`, `--setting-sources user,project,local` | all sources | stable |
| Workdir scope add | `--add-dir <dirs...>` | cwd only | stable |
| Plan/JSON output | `--output-format text\|json\|stream-json`; `--input-format text\|stream-json`; `--json-schema <schema>` | `text` | stable |
| **Temperature** | **not exposed on CLI** — use `--system-prompt` to nudge style. | API default | n/a |

#### Login flow

```bash
# Trigger login (opens browser, runs local callback server)
claude auth login                       # default: Claude.ai subscription
claude auth login --console             # API/billing console route
claude auth login --sso                 # force SSO
claude auth login --email <addr>        # pre-fill email field
claude setup-token                      # long-lived API token, requires Claude subscription

# Detect post-login state (machine-readable JSON)
claude auth status
# → {"loggedIn":true,"authMethod":"claude.ai","apiProvider":"firstParty","email":"...","orgId":"...","subscriptionType":"max"}

claude auth logout                      # remove credentials
```

- **Blocking:** yes — `claude auth login` opens a browser and waits for the callback. Whether the URL is also printed to stdout/stderr was not empirically tested (the test machine is already logged in; not retesting). Safe assumption: orbital should expect a blocking child process and surface its stderr to the user.
- **Success signal:** `claude auth status` returns JSON with `loggedIn: true`. Credential file lives at `~/.claude/.credentials.json` (471 bytes on this machine). Polling either is fine; the JSON status command is the contractual API.

### codex (OpenAI)

| Parameter | Flag / mechanism | Default | Stability |
|---|---|---|---|
| Model | `-m, --model <id>`, or `-c model="o3"` | config | stable |
| Generic config override (TOML) | `-c key=value` (e.g. `-c shell_environment_policy.inherit=all`) | n/a | stable, the primary tuning surface |
| Feature flag | `--enable <name>` / `--disable <name>` (= `-c features.<name>=true/false`) | per defaults | stable |
| OSS provider | `--oss`, `--local-provider <lmstudio\|ollama>` | off | stable |
| Image attachment | `-i, --image <FILE...>` | none | stable |
| Sandbox | `--sandbox read-only\|...` (and `codex sandbox` subcommand) | depends on default config | stable |
| Skip git-repo check | `--skip-git-repo-check` | enforced | stable |
| Output schema | `--output-schema <FILE>` (JSON Schema for final response) | none | stable on `exec` |
| JSON event stream | `--json` (stdout JSONL of events) | off | stable on `exec` |
| Session resume | `codex exec resume <UUID> [PROMPT]` or `codex exec resume --last` | n/a | stable |
| Session resume (interactive) | `codex resume`, `codex fork` | picker | stable |
| Disable cwd filter on resume | `--all` flag on `resume` | off (cwd-scoped, like claude) | stable |
| MCP server mode | `codex mcp-server` | off | stable |
| App server (experimental) | `codex app-server` | off | **experimental** (marked in --help) |
| Cloud tasks | `codex cloud` | off | **experimental** (marked in --help) |
| **Thinking level / temperature** | not first-class; expressed via `-c` against `~/.codex/config.toml` keys | n/a | n/a |

#### Login flow

```bash
codex login                              # default OAuth flow (browser)
codex login --device-auth                # device-code flow (no browser; prints code+URL to stderr)
printenv OPENAI_API_KEY | codex login --with-api-key   # API key from stdin

codex login status                       # current auth state
codex logout                             # remove credentials
```

- **Blocking:** `codex login` (default) opens a browser and waits for callback. **`codex login --device-auth` is the orchestrator-friendly path** — it prints the code + verification URL and lets the user activate from any other device.
- **Success signal:** credentials land at `~/.codex/auth.json` (does not exist on this machine, since not logged in). `codex login status` is the JSON-friendly check.

### gemini-cli (documentation only, not installed on test machine)

| Parameter | Mechanism | Default | Stability |
|---|---|---|---|
| Model | `--model` / `-m`, `model.name` in `settings.json`, `GEMINI_MODEL` env | from settings | stable |
| Thinking level | `thinkingConfig.thinkingLevel` (LOW/MEDIUM/HIGH) under `modelConfigs.aliases.<alias>.modelConfig.generateContentConfig.thinkingConfig` in settings.json | HIGH (Gemini 3); 8192 budget on chat-base-2.5 | stable in config; CLI exposure ambiguous (open issues #21974, #25122) |
| Max session turns | `model.maxSessionTurns` in settings.json; possibly `--max-session-turns` (referenced by exit-code 53 but not in `--help`) | -1 (unlimited) | stable in config; CLI ambiguous |
| Temperature | `generateContentConfig.temperature` in settings.json (no CLI flag) | 0 (base), 1 (chat-base) | stable |
| Top-P | `generateContentConfig.topP` in settings.json | undocumented | stable |
| Auto-approve / YOLO | `--yolo` / `-y` (CLI only — not settable in settings.json), or `--approval-mode default\|auto_edit\|plan\|yolo` | `default` | stable |
| Approval (config) | `general.defaultApprovalMode` in settings.json | `default` | stable |
| Allowed tools | `tools.allowed` in settings.json or `--allowed-tools` | unset | stable |
| Sandbox | `--sandbox` / `-s`, `tools.sandbox` config, `GEMINI_SANDBOX` env | off | stable |
| Headless prompt | `-p` / `--prompt` | n/a | stable |
| Output format | `--output-format text\|json` | text | stable |
| Resume | `--resume`, `--resume <index>`, `--resume <uuid>`, `--list-sessions`, `--delete-session` | n/a | stable |

#### Login flow (gemini-cli)

- **No `gemini auth login` subcommand.** Authentication is interactive on first launch (chooser between "Login with Google" and "Use Gemini API key"). Inside the TUI, **`/auth`** opens the auth-method dialog.
- **OAuth** spins up a local HTTP callback server. **Whether the URL is also written to stdout/stderr is ambiguous** — docs do not state this explicitly, no flag forces URL-only output.
- **Headless / orbital-friendly path:** pre-cache creds via a prior interactive login OR set environment:
  - `GEMINI_API_KEY` — Gemini API key
  - `GOOGLE_API_KEY` — Vertex AI Express
  - `GOOGLE_GENAI_USE_VERTEXAI=true` — force Vertex
  - `GOOGLE_GENAI_USE_GCA=true` — force OAuth (Google Code Assist)
  - `GOOGLE_APPLICATION_CREDENTIALS=<path>` — service-account JSON
- **Success signal:** cached OAuth tokens land at `~/.gemini/tokens.json` (one source) or `~/.gemini/oauth_creds.json` (another) — **filename is ambiguous between sources**. No documented `gemini auth status`. Closest contractual probe: `gemini -p "ping"` and read exit code (41 = auth not configured, 42 = file-read fail, 53 = max-session-turns exceeded).

---

## Cleanup

Scratch dir at `/tmp/orbital-investigation/` (= `C:\Users\qiren\AppData\Local\Temp\orbital-investigation\`) contains all probe transcripts: `q1/`, `q2/`, `q3/`, `q5/`, `q6/`, `q7/`, `t2/`, `t4/`. Per the spec's CLEANUP CHECKLIST, this directory should be deleted after these findings are reviewed. No test API keys were used; only the user's existing OAuth login was exercised. Generated session files at `~/.claude/projects/C--Users-qiren-AppData-Local-Temp-orbital-investigation-*` will linger until the user runs `claude project` cleanup or deletes them manually — they contain only the probe transcripts above and are not sensitive.
