# AGENTS.md

A guide for coding agents (Codex, Cursor, Claude Code and others) working in
this repository.

## What this repository is

Orbital is a desktop app that runs a management agent and delegates work to
coding-agent CLIs (Claude Code, Codex, Cursor, Pi and others) in one
conversation per project.

- `agent_os/` holds the Python daemon (FastAPI, Python 3.11+). The app factory
  is `agent_os/api/app.py`, the agent loop and worker transports are under
  `agent_os/agent/`, and the desktop shell is `agent_os/desktop/`.
- `web/` holds the React + TypeScript + Vite frontend.
- `tests/unit/` and `tests/platform/` hold the pytest suites (pytest-asyncio).
- `scripts/` and `installer/` hold the build and release tooling.

**`CLAUDE.md` is the full development guide.** It covers the test gates, how to
run the daemon, i18n, and the release process. Read it before changing
`agent_os/` or `web/`. This file is the short version.

## Contributions

Orbital does not merge outside pull requests (see `CONTRIBUTING.md`). If you are
helping someone who found a bug, help them open an **issue**, with the file, the
line, and what is wrong. A precise diagnosis gets the fix shipped fastest.

## Setup and checks

```bash
pip install -e ".[desktop]"
cd web && npm install && cd ..

# Backend tests. The null keyring stops the macOS Keychain from hanging headless runs.
PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring python -m pytest tests/unit/ -q

# Frontend: types, unit tests, translation catalog
cd web && npx tsc -b && npm test && node scripts/check-i18n.mjs
```

Run the daemon with `bash scripts/restart-daemon.sh` (port 8000), and the
frontend with `cd web && npx vite --host 127.0.0.1 --port 5173`. Restart the
daemon after every backend change, because a running daemon keeps the old code.

## Things that bite

- **One daemon per machine.** The daemon holds `~/orbital/daemon.pid`, so a
  dev daemon and an installed Orbital.app can't both run. Port 8000 is often
  the installed app, not your code: check which process is listening before
  you trust a result.
- **Some platform tests use the real sandbox.** `tests/platform/test_e2e_agent_isolation.py`
  and `test_macos_provider_integration.py` run `sandbox-exec` against
  `~/Desktop` and can lock the checkout out of its own files. Leave them out
  unless you need them.
- **User-facing strings go through `t('key')`** (`web/src/i18n/`). Never put a
  bare string literal in JSX. Add the English entry; Chinese can follow later.
  Never edit `docs/i18n/ui-terms.zh-Hans.csv` in Excel.
- **Path checks need `realpath` on both sides**, with a trailing separator, or
  symlinks and prefix matches get through.
- **Don't trust a local build for release behaviour.** Release installers are
  Developer-ID signed with hardened runtime and notarized in CI. A local
  `scripts/build-*.sh` build is ad-hoc signed and can behave differently.

## If there is an `orbital/` folder

That means this checkout is open as a project in Orbital itself: Orbital is
developed with Orbital. `orbital/` holds that project's memory and is
gitignored. When Orbital hands you a task, the instructions in your prompt say
which of its files to read and update. Don't commit anything under `orbital/`.
