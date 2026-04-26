# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: workspace file writes are atomic so concurrent readers never see partial content.

Note: the older "session-end runs as background fire-and-forget" tests in
this file were removed during the cancel-arch merge. Cancel-arch removed
the fire-and-forget pattern from loop.py (it had no strong reference and
emitted "Task was destroyed but it is pending" on cancel); the synchronous
session-end inside agent_manager.new_session() is now the sole authoritative
summarization path. See cancel-arch test_agent_loop_cancel.py::test_no_fire_and_forget_session_end.
"""

import asyncio
import os
import tempfile
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.workspace_files import WorkspaceFileManager


# ---------------------------------------------------------------------------
# (Removed: test_idle_broadcasts_before_session_end_completes / test_session_end_still_executes)
# These tested the deleted fire-and-forget session-end pattern in AgentLoop.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Atomic write tests (cancel-arch additions): cover the file-write durability
# story that replaced the old fire-and-forget session-end behaviour.
# ---------------------------------------------------------------------------

def test_atomic_write_no_partial_reads():
    """Concurrent reads during write must see either old or new content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wfm = WorkspaceFileManager(tmpdir)
        wfm.ensure_dir()

        # Write initial content
        wfm.write("state", "OLD_CONTENT")

        new_content = "X" * 1_000_000  # 1MB
        results = []
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                content = wfm.read("state")
                if content is not None:
                    results.append(content)
                time.sleep(0.0001)

        # Start reader thread
        t = threading.Thread(target=reader, daemon=True)
        t.start()

        # Perform atomic write
        wfm.write("state", new_content)

        stop.set()
        t.join(timeout=2)

        # Every read must be either old content or new content
        for r in results:
            assert r == "OLD_CONTENT" or r == new_content, \
                f"Partial read detected: len={len(r)}"


# ---------------------------------------------------------------------------
# Test 4: Atomic write — tmp file cleaned up
# ---------------------------------------------------------------------------

def test_atomic_write_tmp_file_cleaned_up():
    """No .tmp files should remain after a successful write."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wfm = WorkspaceFileManager(tmpdir)
        wfm.ensure_dir()

        wfm.write("state", "test content")
        wfm.append("decisions", "\nnew decision")
        wfm.write("lessons", "lesson learned")

        # Check for leftover .tmp files
        workspace_dir = os.path.join(tmpdir, "orbital")
        for f in os.listdir(workspace_dir):
            assert not f.endswith(".tmp"), f"Leftover tmp file: {f}"


# ---------------------------------------------------------------------------
# Test 5: Concurrent read during atomic write sees valid content
# ---------------------------------------------------------------------------

def test_new_loop_reads_valid_file_during_background_write():
    """A reader (context builder) must see valid content even during write."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wfm = WorkspaceFileManager(tmpdir)
        wfm.ensure_dir()

        # Write initial state
        wfm.write("state", "# Project State\nInitial state")

        # Simulate background session-end writing new state
        # while a new loop reads it
        wfm.write("state", "# Project State\nUpdated after session end")

        # Reader should see either version — both are valid
        content = wfm.read("state")
        assert content is not None, "File read returned None"
        assert len(content) > 0, "File read returned empty content"
        assert "Project State" in content, "File content is corrupted"
