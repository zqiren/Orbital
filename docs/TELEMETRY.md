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
    "first_turn": true
  },
  "counters": {
    "app_starts": 2,
    "projects_created": 1,
    "sessions": 3,
    "turns": 41,
    "errors": { "provider_unreachable": 2 },
    "errors_by_provider": { "deepseek": 2 },
    "tokens_by_provider": { "deepseek": { "in": 120000, "out": 8000 } },
    "login_attempted": 1,
    "login_failed": 0
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
| `milestones` | booleans | Lifetime activation flags: API key set, first project, first session, first turn. |
| `counters.app_starts` | int | Daemon starts that day. |
| `counters.projects_created` | int | Projects created (the auto-created scratch workspace is not counted). |
| `counters.sessions` | int | Chat sessions created. |
| `counters.turns` | int | Management-agent LLM responses. |
| `counters.errors` | map | LLM failures by stable error code (`missing_api_key`, `invalid_api_key`, `model_not_found`, `provider_unreachable`, `provider_error`). Codes only — never messages. |
| `counters.errors_by_provider` | map | The **same** failures as `errors`, counted by provider name (`deepseek`, `minimax`, `custom`, …) instead of by code. `unknown` when the failure happened before the provider was resolved. Both maps are published: `errors` is the series with history, `errors_by_provider` is what makes a failure attributable. |
| `counters.tokens_by_provider` | map | Daily input/output token totals per provider (no per-model detail; that stays local in the budget ledger). |
| `counters.login_attempted` | int | Sub-agent CLI sign-in jobs started (Claude Code, Codex, …). The agent slug stays local in the spool — only the daily total is sent. |
| `counters.login_failed` | int | Of those, how many ended in failure. Together with `login_attempted` this is the only visibility into whether sub-agent setup is where activation stalls. |

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

## Transport & retention

- Sent to Orbital's ingest endpoint over HTTPS on daemon startup and every six
  hours while running. Offline days queue locally and send late; resends are
  idempotent (the server upserts on `install_id` + `date`).
- Server-side, raw daily rows are kept for 13 months, then rolled into
  monthly aggregates.
