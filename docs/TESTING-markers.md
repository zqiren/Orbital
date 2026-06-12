# Test resource markers

The default suite run (`python -m pytest tests/`) is green-by-construction:
tests that need a resource this machine/CI may lack are marked and **skipped
by default** with an explicit reason. Markers are registered in
`pyproject.toml`; the skip logic lives in `tests/conftest.py`
(`pytest_collection_modifyitems`).

Recommended default invocation (avoids macOS Keychain prompts/hangs in
headless shells):

```bash
PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring python -m pytest tests/ -q
```

| Marker | Resource needed | How to run the group manually |
|---|---|---|
| `requires_windows` | Real Windows machine (powershell, `ctypes.windll`, win32 APIs) | On Windows: `python -m pytest -m requires_windows tests/` (skips auto-lift on win32) |
| `requires_keychain` | Functional, unlocked OS keychain | `ORBITAL_KEYCHAIN_TESTS=1 python -m pytest -m requires_keychain tests/` (run in a GUI session, not headless) |
| `requires_relay` | Relay server source tree at `./relay` (deployed on Railway; not part of this checkout) | Check out the relay source to `./relay`, `npm install` there, then `python -m pytest -m requires_relay tests/e2e/relay/` (skip auto-lifts when `relay/src/index.ts` exists) |
| `live_sandbox` | Real macOS `sandbox-exec` runs. **Hazard:** probes the real `~/Desktop` and has EPERM-locked the dev tree mid-suite (see `ACTIVE-seam3-test-sweep-resume.md`) | From a **throwaway account/machine only**: `ORBITAL_LIVE_SANDBOX_TESTS=1 python -m pytest -m live_sandbox tests/platform/` |
| `live_daemon` | Spawns real `claude`/`codex`/uvicorn processes; costs live LLM turns | `ORBITAL_LIVE_DAEMON_TESTS=1 python -m pytest -m live_daemon tests/integration/` |

Notes:

- `tests/test_e2e.py`, `tests/test_wiring.py`, `tests/test_user_stories.py`
  self-skip unless `AGENT_OS_TEST_API_KEY` is set (live-LLM e2e; unchanged by
  the triage).
- `tests/platform/test_macos_provider_integration.py::test_portal_readonly` is
  a documented expected env-fail even on a real sandbox-capable Mac.
- Setting `AGENT_OS_API_KEY` in the environment changes the api-key status
  endpoint's reported source to `environment`; the suite tolerates this since
  the 2026-06-12 triage, but prefer leaving it unset for suite runs.
- Tests skipped with reason `DELETION CANDIDATE: …` are pending review in the
  2026-06-12 triage report — they assert deliberately-removed behavior and are
  proposed for deletion, not retirement-by-skip.
