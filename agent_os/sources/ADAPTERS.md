# Writing a task-source adapter

A task source turns an outside service into Workbench tasks. The core
(`agent_os/sources/core.py`) does everything shared — storage, secrets, de-duplication,
merging, holding while paused, one-at-a-time sending, gaps, failure state, the API and
the WebSocket pushes. An adapter only talks to its service.

## Where it goes

One module per type, named after the type: `agent_os/sources/adapters/<type>.py`
(`telegram.py`, `email.py`, …). It exports `ADAPTER`, a subclass of
`agent_os.sources.adapters.Adapter`. The registry imports it by name; nothing else
changes. Until the module exists the catalogue reports the type `available: false,
unavailable_reason: "not_built"`.

The connect form (fields, kinds, setup link, agent prompt) is already in
`agent_os/sources/catalog.py`. To change your type's entry without editing that file,
export `DESCRIPTOR = {...}` from your module: its top-level keys replace the catalogue's
(for example a different `fields` list or `kinds`).

## The interface

```python
from agent_os.sources.adapters import Adapter
from agent_os.sources.model import Item, Resolve, Verified, VerifyError, SourceFailure, Choice

class TelegramAdapter(Adapter):
    mode = "stream"        # "poll" (core calls poll() every interval_s) | "stream"
    interval_s = None      # poll only; None = the catalogue's interval_s
    baseline = True        # poll only: a kind's first poll records items as seen, sends none
    lossless = True        # stream only: the service replays what was missed -> no gaps
    platforms = None       # e.g. ["darwin"]; None = the catalogue's list

    async def verify(self, ctx) -> Verified: ...          # connect / reconnect / settings change
    async def poll(self, ctx) -> list[Item | Resolve]: ...  # mode "poll"
    async def stream(self, ctx) -> None: ...                # mode "stream"
    async def choices(self, ctx, list_name, params) -> list[Choice]: ...   # pickers
    async def accounts(self) -> Accounts: ...              # cli_account types
    async def qr_start(self, ctx) -> dict: ...              # qr types
    async def qr_poll(self, ctx, handle) -> dict: ...
ADAPTER = TelegramAdapter
```

### `ctx` (`agent_os/sources/context.py`)

| | |
|---|---|
| `ctx.config` | non-secret form values (dict copy) |
| `ctx.secret(key)` | a secret field's value (already read from the keychain) |
| `ctx.kinds` | `{kind_key: params}` — only the kinds the user switched on |
| `ctx.account` | the pinned CLI account (cli types) |
| `ctx.label` | what your last `verify` returned as the label |
| `ctx.state` | your own small dict, persisted by the core after each poll / emit. Use it for a cursor or an up/down flag, nothing big. |
| `await ctx.run(argv, timeout=, env=, cwd=)` | run a program off the loop, no shell, killed on timeout → `RunResult(code, stdout, stderr)`; raises `CommandNotFound` / `CommandTimeout` |
| `await ctx.to_thread(fn, …)` | any blocking library call (imaplib, an SDK) |
| `await ctx.emit([Item, …])` | stream: hand arrivals to the core |
| `ctx.connected()` | stream: call once the connection is up (closes a gap, clears a failure) |
| `ctx.now()` | aware UTC datetime |
| `ctx.lang` | `"zh"` / `"en"`: the language for a title Orbital itself writes (not for the service's own text) |

Never block the event loop: every socket, subprocess and SDK call goes through
`ctx.run`, `ctx.to_thread` or an asyncio client.

### What you return

`Item(id, kind, title, subject=None, body="", url=None, sender=None, place=None, at=None)`

- `id`: stable per item (message id, `issue#n`, IMAP UID + UIDVALIDITY). The core records
  each id once per source, so re-reading an overlapping window is fine and you need no
  cursor of your own (keep one in `ctx.state` only if your API needs it to page).
- `kind`: which switched-on kind it matched (`"dm"`, `"at"`, `"assigned"`, …). Items of a
  kind that is off are dropped.
- `subject`: what it is about (thread id, issue). Same subject while its task is open →
  merged into one history entry with a count. Default: the id.
- `title`: one line (the core cleans and caps it). `body`: the text worth reading; it
  reaches the agent as quoted material, never as instructions.
- bots: `sender` (display name) and `place`: `{"kind": "dm"}`, `{"kind": "group", "name":
  …}` or `{"kind": "channel", "name": "#support"}`.

`Resolve(subject)`: the condition that opened that subject's entry is over (a website
back up). The same entry is marked recovered; no new task.

### Errors

- In `verify`: `raise VerifyError(reason, field, message)`; `reason` ∈ `rejected`,
  `not_found`, `signed_out`, `cli_missing`, `permission`, `unreachable`, `invalid`,
  `error`; `field` = the form field or kind param to mark. Nothing is created.
- Later (`poll` / `stream`): `raise SourceFailure(code, params, retryable=True)` with a
  contract code (`token_rejected`, `auth_expired`, `repo_not_found`, `calendar_permission`,
  `unreachable`, `rate_limited`, …). `retryable=False` when checking again can only fail
  again (a revoked token): the core stops the source until the user reconnects and offers
  no 「再检查」. Any other exception is a retryable `error`.
- Messages must never contain a secret. Log through `ctx.log` without secrets.

### Streams

`stream(ctx)` holds the connection: call `ctx.connected()` when it is up, then
`await ctx.emit([...])` per arrival, and return or raise when the connection ends. The
core restarts it with backoff (1 s doubling to 5 min, reset after a minute connected),
records gaps for lossy types, and marks the source `unreachable` after three quick
failures in a row. Cancel-safe: the core cancels the task on stop / remove / reconnect.
Streams have no baseline (a message is new by definition).

### What the core guarantees you

Secrets are stored in the keychain only after `verify` passed; the source is paused or
not without you knowing (arrivals are held by the core); one arrival is sent at a time;
`verify` runs again when the user changes kinds, account or config.

## Tests

Put them in `tests/unit/test_sources_<type>.py`. Fake the service, not the core:
`tests/unit/test_sources_adapters.py` shows a faked CLI (`FakeGh`) and a local HTTP
server; `tests/unit/test_sources_pipeline.py` shows a whole source through
`SourcesCore` with `AdapterRegistry({"<type>": adapter})`.
