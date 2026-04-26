"""Top-level test fixtures.

Autouse: patches ``agent_os.api.app.acquire_pid_file`` so tests that call
``create_app()`` don't fail with ``DaemonAlreadyRunning`` when a real Orbital
daemon happens to be running on the dev machine. Tests that intentionally
exercise the PID-file behaviour (``test_session_file_locking.py``) import
``acquire_pid_file`` directly from ``agent_os.utils.pid_file`` and are
unaffected by patching the re-export in ``agent_os.api.app``.
"""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _bypass_daemon_pid_guard():
    """Bypass the create_app() singleton daemon PID guard for the whole suite."""
    with patch("agent_os.api.app.acquire_pid_file"):
        yield
