# Integration test harness

This directory hosts the live-daemon integration harness used by the
SDK-lifecycle, background-send, and CLI-adapter bug-fix tasks. The
harness spawns a real uvicorn subprocess running the FastAPI app
factory (`agent_os.api.app:create_app`), exposes a typed HTTP +
WebSocket client, and provides cross-platform process-tree utilities
built on `psutil`.

## Running the self-tests

From the repo root:

```bash
python -m pytest tests/integration/test_harness_selftest.py -m live_daemon -v
```

The `live_daemon` marker is registered in `conftest.py`; tests without
that marker run under the normal suite and do not spawn subprocesses.

To exclude daemon-based tests from a broader run:

```bash
python -m pytest tests/ -m "not live_daemon"
```

## Prerequisites

### All platforms

- Python ≥ 3.11 (matches `pyproject.toml`).
- The daemon's runtime dependencies installed: `pip install -e .[dev]`
  pulls `httpx`, `websockets`, `psutil`, `pytest`, and
  `pytest-asyncio`, all of which the harness relies on.
- No other daemon listening on the ephemeral port the OS hands out.

### macOS

- `psutil` wheels ship for macOS; no special setup required.

### Windows

- The harness spawns the daemon with `CREATE_NEW_PROCESS_GROUP` so it
  can send `CTRL_BREAK_EVENT` on shutdown. This does not require
  Administrator rights but may trigger a Windows Defender / firewall
  prompt the first time `python.exe` binds to a local port —
  allow it for the loopback interface.
- If `psutil` fails to install from source, install the Microsoft
  C++ Build Tools; the published wheels cover CPython 3.11/3.12 on
  x64 and most users never hit the source path.

## How isolation works

- **Ephemeral port**: `DaemonProcess.start()` asks the OS for a free
  TCP port via `socket.bind(("127.0.0.1", 0))` so two daemons can
  coexist (e.g. the developer's own daemon on :8000 plus the test
  daemon on an OS-assigned port).
- **PID-file isolation**: The daemon writes a singleton PID file at
  `~/orbital/daemon.pid`. The harness overrides `HOME` and
  `USERPROFILE` on the spawned subprocess so each test run gets its
  own throwaway `~/orbital/` inside a `tempfile.TemporaryDirectory`.
  That directory is also where `~/orbital/browser-profile` and
  `daemon-state.json` land, so the harness can't corrupt the real
  developer workspace.
- **Cloud relay**: If the developer has `AGENT_OS_RELAY_URL` set in
  their shell, the harness unsets it on the subprocess so test
  daemons never reach out to Railway.
- **No production code changes**: All isolation is achieved via
  environment variables and process signals. Nothing under
  `agent_os/` is modified by the harness.

## API endpoints used

Endpoint paths are inspected directly from
`agent_os/api/routes/agents_v2.py` and `agent_os/api/app.py`. The task
spec references an `API-reference.md` file that does not exist in this
repo; do not trust that path.

| Capability           | Method | Path                                     |
| -------------------- | ------ | ---------------------------------------- |
| Health / settings    | GET    | `/api/v2/settings`                       |
| Create project       | POST   | `/api/v2/projects`                       |
| List projects        | GET    | `/api/v2/projects`                       |
| Get project          | GET    | `/api/v2/projects/{id}`                  |
| Delete project       | DELETE | `/api/v2/projects/{id}`                  |
| Start agent          | POST   | `/api/v2/agents/start`                   |
| Inject message       | POST   | `/api/v2/agents/{id}/inject`             |
| Stop agent           | POST   | `/api/v2/agents/{id}/stop`               |
| Run status           | GET    | `/api/v2/agents/{id}/run-status`         |
| WebSocket events     | WS     | `/ws` (subscribe protocol — see app.py)  |

There is intentionally **no** `/shutdown` endpoint on the daemon; the
harness teardown path uses `CTRL_BREAK_EVENT` (Windows) / `SIGTERM`
(POSIX) followed by a psutil-based tree kill.

## Known limitations

- **Self-test 3 (`test_daemon_shutdown_reaps_children`)** spawns a
  synthetic child of the test process rather than a real sub-agent
  dispatched via the daemon. Dispatching a real sub-agent requires a
  configured LLM provider with a valid API key, which CI does not
  have. The test still validates the critical property —
  `DaemonProcess.shutdown()` reaps the daemon PID and every
  descendant captured before teardown. The follow-up bug-fix tasks
  that *do* have LLM credentials can dispatch real sub-agents and
  assert on the full tree.
- **Windows grandchildren after hard-kill**: on Windows, if a child
  of the daemon spawns its own grandchild and the harness has to
  fall back to `TerminateProcess` (i.e. the graceful `CTRL_BREAK`
  path failed), grandchildren may survive because Windows does not
  propagate the kill down the tree automatically. The harness
  mitigates this by calling `kill_process_tree` itself (walks the
  tree via `psutil`) after the root dies. If you still see
  stragglers, they are pre-existing product-level leaks and should
  be tracked separately.

## Debugging — keeping the daemon up by hand

To reproduce an interactive session with the exact configuration a test
would see:

```python
from tests.integration.harness import DaemonProcess
d = DaemonProcess()
d.start()
print(f"daemon up: pid={d.pid} port={d.port}")
# do stuff — d.base_url is 'http://127.0.0.1:<port>'
# ...
d.shutdown()
```

Drop that into an `ipython` or `python -i` session. The harness prints
captured daemon logs via `d.log_lines()` so you can inspect startup
output without grepping stderr.
