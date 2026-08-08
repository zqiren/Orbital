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
    "tokens_by_provider": { "deepseek": { "in": 120000, "out": 8000 } }
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
| `counters.tokens_by_provider` | map | Daily input/output token totals per provider (no per-model detail; that stays local in the budget ledger). |

## Transport & retention

- Sent to Orbital's ingest endpoint over HTTPS on daemon startup and every six
  hours while running. Offline days queue locally and send late; resends are
  idempotent (the server upserts on `install_id` + `date`).
- Server-side, raw daily rows are kept for 13 months, then rolled into
  monthly aggregates.
