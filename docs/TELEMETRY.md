# Telemetry — published schema

Orbital sends **one anonymous aggregate per day**. This document is the
complete, authoritative description of everything that ever leaves your
machine. The exact outbound JSON is also inspectable in-app, verbatim, under
**Settings → Data & privacy** — before and after it is sent.

## Principles

- **Raw events never leave the machine.** A local spool
  (`{data_dir}/telemetry/events.jsonl`) records coarse product events; only
  the daily aggregate below is transmitted.
- **No content, ever.** No prompts, no file paths, no model output, no
  project/session identifiers, no free text. Counters, enums, booleans, and
  ISO dates only.
- **Random identity, resettable.** `install_id` is a random token minted at
  first run (`{data_dir}/telemetry/install.json`). It fingerprints nothing;
  deleting that file mints a new one.
- **Off means off.** The Settings toggle stops all counter collection and all
  sends. (Structured LLM error codes keep spooling locally for your own
  debugging — they are local-only and never transmitted while disabled.)

## The daily ping

```json
{
  "schema": 1,
  "install_id": "inst_9f3c2a1b7d4e",
  "account_id": null,
  "version": "0.8.4",
  "os": "darwin",
  "date": "2026-08-07",
  "first_seen": "2026-08-01",
  "milestones": {
    "key_set": true,
    "first_project": true,
    "first_session": true,
    "first_turn": true,
    "key_skipped": false,
    "first_import": true,
    "first_leftoff_open": true,
    "first_imported_send": false
  },
  "counters": {
    "app_starts": 2,
    "projects_created": 1,
    "sessions": 3,
    "turns": 41,
    "errors": { "provider_unreachable": 2 },
    "errors_by_provider": { "deepseek": 2 },
    "tokens_by_provider": { "deepseek": { "in": 120000, "out": 8000 } },
    "login_attempted": 2,
    "login_failed": 1,
    "login_unverified": 0,
    "login_by_agent": {
      "claude-code": { "attempted": 1, "failed": 1, "unverified": 0 },
      "codex": { "attempted": 1, "failed": 0, "unverified": 0 }
    },
    "login_failed_by_reason": { "nonzero_exit": 1 },
    "update_attempted": 1,
    "update_failed": 0,
    "subagent_dispatches": 4,
    "subagent_failed": 1,
    "subagent_failed_by_reason": { "update_needed": 1 },
    "subagent_by_agent": {
      "claude-code": { "dispatched": 3, "failed": 1, "failed_by_reason": { "update_needed": 1 } },
      "codex": { "dispatched": 1, "failed": 0, "failed_by_reason": {} }
    },
    "recent_failures": [
      {
        "ts": "2026-08-07T09:12:03+00:00",
        "event": "login_failed",
        "agent": "claude-code",
        "reason": "nonzero_exit",
        "exit_code": 1,
        "cli_version": "2.1.100",
        "install_kind": "npm",
        "duration_ms": 840,
        "error_class": null,
        "stderr_category": "unknown_command"
      },
      {
        "ts": "2026-08-07T09:13:40+00:00",
        "event": "subagent_failed",
        "agent": "claude-code",
        "reason": "update_needed",
        "exit_code": null,
        "cli_version": "2.1.100",
        "install_kind": "npm",
        "duration_ms": 12,
        "error_class": null,
        "stderr_category": "other"
      }
    ],
    "wizard_step_reached": { "agents": 1, "projects": 1, "key": 1, "accounts": 1, "done": 1 },
    "agents_detected": 2,
    "agents_located": 0,
    "history_sessions_bucket": "100+",
    "projects_imported": { "folder": 3, "group": 1 },
    "projects_unchecked": 2,
    "leftoff_shown": 4,
    "leftoff_opened": { "last_time": 0, "unfinished": 1 },
    "unfinished_opened_by_cause": {
      "usage_limit": 1, "model_unavailable": 0, "network_auth": 0,
      "interrupted": 0, "no_reply": 0
    },
    "agent_switch_after_open": 1,
    "notes_found": { "claude-code": 145 }
  }
}
```

| Field | Type | Meaning |
|---|---|---|
| `schema` | int | Payload schema version (currently `1`). |
| `install_id` | string | Random install token (`inst_` + 12 hex chars). Not derived from hardware, account, or anything identifying. |
| `account_id` | null | Reserved for a future opt-in account system. Always `null` today. |
| `version` | string | Orbital app version. |
| `os` | enum | `darwin` \| `windows` \| `linux`. |
| `date` | ISO date | The UTC day this aggregate covers. |
| `first_seen` | ISO date | When this install id was minted. |
| `milestones` | booleans | Lifetime activation flags: API key set, first project, first session, first turn. Since spec 138 also: `key_skipped` (the setup wizard's Orbital-agent key step was skipped), `first_import` (projects were first imported from the user's agent history), `first_leftoff_open` (a "Last time" / "Didn't finish" card was first opened), `first_imported_send` (a first message was sent in a conversation that came from another agent's history). |
| `counters.app_starts` | int | Daemon starts that day. |
| `counters.projects_created` | int | Projects created (the auto-created scratch workspace is not counted). |
| `counters.sessions` | int | Chat sessions created. |
| `counters.turns` | int | Management-agent LLM responses. |
| `counters.errors` | map | LLM failures by stable error code (`missing_api_key`, `invalid_api_key`, `model_not_found`, `provider_unreachable`, `provider_error`). Codes only — never messages. |
| `counters.errors_by_provider` | map | The **same** failures as `errors`, counted by provider name (`deepseek`, `minimax`, `custom`, …) instead of by code. `unknown` when the failure happened before the provider was resolved. Both maps are published: `errors` is the series with history, `errors_by_provider` is what makes a failure attributable. |
| `counters.tokens_by_provider` | map | Daily input/output token totals per provider (no per-model detail; that stays local in the budget ledger). |
| `counters.login_attempted` | int | Sub-agent CLI sign-in jobs started from Settings (Claude Code, Codex, …). |
| `counters.login_failed` | int | Of those, how many failed: the CLI could not be spawned, went idle past the timeout, or exited non-zero. **Since v0.17.0** a job that exited 0 but whose sign-in could not be confirmed is counted under `login_unverified` instead (earlier releases counted it here). |
| `counters.login_unverified` | int | Sign-in jobs that exited 0 while the CLI's own status probe still reported signed-out — "signed in, but couldn't confirm", the same wording the card shows. Not a failure; listed separately so a probe that breaks on one platform cannot masquerade as users failing to sign in. |
| `counters.login_by_agent` | map | The same three counts per CLI: `{agent: {attempted, failed, unverified}}`. Keys follow the `subagent_by_agent` allowlist (built-in sub-agent names only, `other` otherwise). |
| `counters.login_failed_by_reason` | map | `login_failed` split by a fixed enum: `spawn_error`, `timeout`, `nonzero_exit` (`unknown` for rows written before the enum existed). Never a message. |
| `counters.update_attempted` | int | In-app "Update <CLI>" jobs started (the CLI's own updater, run from the sub-agent card). |
| `counters.update_failed` | int | Of those, how many did not end with the CLI at or above the version Orbital needs — decided by re-reading `--version`, never by the updater's exit code. |
| `counters.subagent_dispatches` | int | Prompts handed to a sub-agent (by the management agent, a direct send, a queue item, or a fanout worker). |
| `counters.subagent_failed` | int | Sub-agent work that failed: a sub-agent that could not be started, or a task that ended in an error. A deliberate stop is not a failure. Because a failed start is never dispatched, this is not a subset of `subagent_dispatches`. |
| `counters.subagent_failed_by_reason` | map | `subagent_failed` split by a fixed enum: `spawn` (the process never started), `runtime_error` (the task errored), `adapter_broken` (Orbital's own transport failed), `not_authenticated` (refused before dispatch: the CLI is installed but not signed in / has no key), `update_needed` (refused before dispatch: the CLI is older than the version Orbital needs), `unknown` (rows from before the enum). |
| `counters.subagent_by_agent` | map | Per sub-agent: `{agent: {dispatched, failed, failed_by_reason}}`, with `failed_by_reason` the same enum as above. Keys are built-in sub-agent names only (`claude-code`, `codex`, `gemini-cli`, …), `native` for Orbital's own fanout workers, and `other` for any name that is not built in — so no free text leaves the machine. |
| `counters.recent_failures` | list | The day's last 20 `login_failed` / `login_unverified` / `update_failed` / `subagent_failed` events with structured diagnostics, so a failing install can be understood without asking for logs. Every field is an enum or a number: `ts` (ISO time), `event`, `agent` (allowlisted as above), `reason` (the enums above), `exit_code` (int or null), `cli_version` (the dotted version number read from `--version`, or null), `install_kind` (`npm` / `native` / `unknown`, judged from where the binary was found), `duration_ms`, `error_class` (an exception class name from a fixed set, `other`, or null), `stderr_category` (`unknown_command` / `not_logged_in` / `network` / `timeout` / `batch_refused` / `other` — the CLI's output mapped through fixed patterns; the text itself never leaves). |

| `counters.wizard_step_reached` | map | Setup-wizard steps reached that day: `agents`, `projects`, `key`, `accounts`, `done` (a fixed enum; one count per arrival). |
| `counters.agents_detected` | int or null | How many supported coding agents (Claude Code, Codex, …) were found installed when the wizard read the history — the day's latest figure; `null` when not measured that day. A count only, never which ones. |
| `counters.agents_located` | int | Agents the user pointed Orbital at by hand ("Installed but not found?"). |
| `counters.history_sessions_bucket` | enum or null | How many past conversations Orbital found on this machine, as a bucket: `0`, `1-10`, `11-100`, `100+`. Never the conversations, their folders or their text — those are read locally and never leave the machine. |
| `counters.projects_imported` | map | Projects created by the import: `folder` (a folder the user had worked in) and `group` (scattered conversations brought together). Counts only. |
| `counters.projects_unchecked` | int | Suggested projects the user unchecked before importing. |
| `counters.leftoff_shown` | int | Times the "where you left off" cards were shown in a project's new-session view. |
| `counters.leftoff_opened` | map | Cards opened: `last_time` or `unfinished`. |
| `counters.unfinished_opened_by_cause` | map | Opened "Didn't finish" cards by why the conversation stopped (fixed enum: `usage_limit`, `model_unavailable`, `network_auth`, `interrupted`, `no_reply`). Never the error text. |
| `counters.agent_switch_after_open` | int | Times the user picked a different agent after opening a card. |
| `counters.notes_found` | map | Private notes each coding agent keeps on this computer that no other agent can read (today only Claude Code's `memory/` notes), as the setup wizard's agent row counts them: `{agent: n}`, the day's latest count per agent. Keys follow the `subagent_by_agent` allowlist. A count only — never the notes, their titles or the folders they are about. |

`counters.sessions` counts sessions at the point they are minted, so every
path produces one: the "+ New session" button, the first-run cold-start scan,
the workbench spawn, and the queue dispatcher (which runs each queue item in
its own fresh session). Earlier releases counted only the button, so the
figure was biased against exactly the heaviest users.

## When Orbital does not send

Beyond the Settings toggle, the sender refuses to start at all for processes
that are not somebody's Orbital — they were flooding the dataset with
single-ping phantom installs:

- **`AGENT_OS_TELEMETRY_DISABLED`** — set to any truthy value (`1`, `true`, …)
  to suppress all transmission for that process. Set by the test suite
  (`tests/conftest.py`), CI (`.github/workflows/ci.yml`), and the repo dev
  daemon (`scripts/restart-daemon.sh`).
- **A data dir under a temp path** (`tempfile.gettempdir()`, `/tmp`,
  `/private/var/folders`, symlinks resolved) — a structural backstop for
  processes that forget the env var. No real install lives there.

Both guards suppress **transmission only**: the local spool, the daily
snapshot, and the Settings → Data & privacy viewer keep working, and each
guard logs once at startup when it fires.

Maintainers trying an installer on a VM or a spare machine should set
`AGENT_OS_TELEMETRY_DISABLED=1` first: a fresh install mints a fresh
`install_id`, and each such try-out otherwise shows up as a real install that
never came back.

## Transport & retention

- Sent to Orbital's ingest endpoint over HTTPS on daemon startup, every hour
  while running, and once more when the app quits (bounded to three seconds, so
  quitting is never held up). Offline days queue locally and send late; resends
  are idempotent (the server upserts on `install_id` + `date`).
- Past days are re-aggregated from the local event spool at the next send and
  re-sent until the server acknowledges them (`telemetry/acks.json`), so a
  session closed between two sends, or one that crossed UTC midnight, still
  lands in its day's counters. The spool keeps roughly the last 5–10 MB of
  events; a day older than that is sent as whatever snapshot it last had.
- Server-side, raw daily rows are kept for 13 months, then rolled into
  monthly aggregates.
