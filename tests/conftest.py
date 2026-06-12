"""Top-level test fixtures.

Autouse: patches ``agent_os.api.app.acquire_pid_file`` so tests that call
``create_app()`` don't fail with ``DaemonAlreadyRunning`` when a real Orbital
daemon happens to be running on the dev machine. Tests that intentionally
exercise the PID-file behaviour (``test_session_file_locking.py``) import
``acquire_pid_file`` directly from ``agent_os.utils.pid_file`` and are
unaffected by patching the re-export in ``agent_os.api.app``.

Also enforces the resource markers declared in pyproject.toml: marked tests
are skipped by default unless the resource is present or explicitly opted
into via env var. See docs/TESTING-markers.md for the run commands.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _opted_in(env_var: str) -> bool:
    return os.environ.get(env_var, "") == "1"


def pytest_collection_modifyitems(config, items):
    skip_windows = pytest.mark.skip(
        reason="requires a real Windows machine (run on Windows: pytest -m requires_windows)")
    skip_keychain = pytest.mark.skip(
        reason="requires a functional unlocked OS keychain (opt in: ORBITAL_KEYCHAIN_TESTS=1 pytest -m requires_keychain)")
    skip_relay = pytest.mark.skip(
        reason="requires the relay server source at ./relay (deployed on Railway; not in this checkout)")
    skip_live_sandbox = pytest.mark.skip(
        reason="runs a REAL sandbox-exec against ~/Desktop and can EPERM-lock the dev tree "
               "(opt in from a throwaway account: ORBITAL_LIVE_SANDBOX_TESTS=1 pytest -m live_sandbox)")
    skip_live_daemon = pytest.mark.skip(
        reason="spawns a real claude/codex/uvicorn process and costs live turns "
               "(opt in: ORBITAL_LIVE_DAEMON_TESTS=1 pytest -m live_daemon)")

    has_relay_source = (_REPO_ROOT / "relay" / "src" / "index.ts").is_file()

    for item in items:
        if "requires_windows" in item.keywords and sys.platform != "win32":
            item.add_marker(skip_windows)
        if "requires_keychain" in item.keywords and not _opted_in("ORBITAL_KEYCHAIN_TESTS"):
            item.add_marker(skip_keychain)
        if "requires_relay" in item.keywords and not has_relay_source:
            item.add_marker(skip_relay)
        if "live_sandbox" in item.keywords and not _opted_in("ORBITAL_LIVE_SANDBOX_TESTS"):
            item.add_marker(skip_live_sandbox)
        if "live_daemon" in item.keywords and not _opted_in("ORBITAL_LIVE_DAEMON_TESTS"):
            item.add_marker(skip_live_daemon)


@pytest.fixture(autouse=True)
def _bypass_daemon_pid_guard():
    """Bypass the create_app() singleton daemon PID guard for the whole suite."""
    with patch("agent_os.api.app.acquire_pid_file"):
        yield
