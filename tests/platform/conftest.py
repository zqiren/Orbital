# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared fixtures and markers for platform isolation tests."""

import os
import sys
import tempfile
import shutil
import uuid

import pytest

# --- Module-level booleans ---

IS_WINDOWS = sys.platform == "win32"

IS_ADMIN = False
if IS_WINDOWS:
    try:
        import ctypes
        IS_ADMIN = ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception:
        IS_ADMIN = False

HAS_SANDBOX_USER = False
if IS_WINDOWS:
    try:
        import subprocess
        result = subprocess.run(
            ["net", "user", "AgentOS-Worker"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
        )
        HAS_SANDBOX_USER = result.returncode == 0
    except Exception:
        HAS_SANDBOX_USER = False

# --- Skip markers ---

skip_not_windows = pytest.mark.skipif(
    not IS_WINDOWS, reason="Requires Windows"
)

skip_not_admin = pytest.mark.skipif(
    not IS_ADMIN, reason="Requires administrator privileges"
)

skip_no_sandbox = pytest.mark.skipif(
    not HAS_SANDBOX_USER, reason="Requires AgentOS-Worker user to exist"
)

# --- Fixtures ---


@pytest.fixture
def temp_workspace(tmp_path):
    """Creates a temporary workspace directory, cleaned up after the test."""
    workspace = tmp_path / f"test_workspace_{uuid.uuid4().hex[:8]}"
    workspace.mkdir(parents=True, exist_ok=True)
    yield str(workspace)
    # tmp_path is cleaned up automatically by pytest


@pytest.fixture(scope="class")
def ensure_sandbox_account():
    """Ensure the sandbox user exists with a valid password.

    Previous test modules (e.g. test_sandbox_account) may delete the user.
    This fixture re-creates it so downstream tests can run.
    Only effective when running as admin.
    """
    if not IS_ADMIN:
        if not HAS_SANDBOX_USER:
            pytest.skip("Sandbox user not available and not admin to create it")
        yield
        return

    from agent_os.platform.windows.credentials import CredentialStore
    from agent_os.platform.windows.sandbox import SandboxAccountManager

    cs = CredentialStore()
    mgr = SandboxAccountManager(cs)
    status = mgr.ensure_account_exists()
    if not status.exists or not status.password_valid:
        pytest.skip(f"Could not ensure sandbox account: {status.error}")
    yield


@pytest.fixture
def temp_file(tmp_path):
    """Creates a temporary directory with a test.txt containing 'hello'.

    Yields the directory path (not the file path).
    """
    test_dir = tmp_path / f"test_dir_{uuid.uuid4().hex[:8]}"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_file = test_dir / "test.txt"
    test_file.write_text("hello", encoding="utf-8")
    yield str(test_dir)
    # tmp_path is cleaned up automatically by pytest
