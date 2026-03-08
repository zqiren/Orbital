# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Regression tests for file-watch triggers.

Covers:
1. Observer creation and lifecycle
2. File events firing triggers with debounce
3. Glob pattern filtering
4. Path security validation (workspace boundary)
5. Changed files included in trigger message
"""

import asyncio
import os
import tempfile
from unittest.mock import MagicMock, AsyncMock

import pytest

from agent_os.daemon_v2.project_store import ProjectStore
from agent_os.daemon_v2.trigger_manager import (
    TriggerManager,
    validate_trigger,
    validate_watch_path,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_project_store(triggers=None, workspace=None):
    """Create a real ProjectStore with a temp directory and one project."""
    tmpdir = workspace or tempfile.mkdtemp()
    store = ProjectStore(data_dir=tmpdir)
    pid = store.create_project({
        "name": "Test Project",
        "workspace": tmpdir,
        "model": "gpt-4",
        "api_key": "sk-test",
    })
    if triggers is not None:
        store.update_project(pid, {"triggers": triggers})
    return store, pid, tmpdir


def _make_file_watch_trigger(trigger_id="trg_fw001", watch_path="incoming",
                              patterns=None, recursive=False,
                              debounce_seconds=1, enabled=True):
    return {
        "id": trigger_id,
        "name": "File Watcher",
        "enabled": enabled,
        "type": "file_watch",
        "watch_path": watch_path,
        "patterns": patterns or [],
        "recursive": recursive,
        "debounce_seconds": debounce_seconds,
        "task": "Process the new files",
        "autonomy": None,
        "last_triggered": None,
        "trigger_count": 0,
        "created_at": "2026-03-03T00:00:00Z",
    }


# ===========================================================================
# Validation Tests
# ===========================================================================

class TestFileWatchValidation:

    def test_valid_file_watch_trigger(self):
        trigger = _make_file_watch_trigger()
        assert validate_trigger(trigger) is None

    def test_file_watch_requires_watch_path(self):
        trigger = _make_file_watch_trigger(watch_path="")
        error = validate_trigger(trigger)
        assert error is not None
        assert "watch_path" in error

    def test_file_watch_rejects_outside_workspace(self):
        """watch_path outside workspace should be rejected."""
        tmpdir = tempfile.mkdtemp()
        trigger = _make_file_watch_trigger(watch_path="/etc/passwd")
        error = validate_trigger(trigger, workspace=tmpdir)
        assert error is not None
        assert "outside workspace" in error

    def test_file_watch_rejects_traversal(self):
        """Path traversal via .. should be rejected."""
        tmpdir = tempfile.mkdtemp()
        trigger = _make_file_watch_trigger(watch_path="../../etc")
        error = validate_trigger(trigger, workspace=tmpdir)
        assert error is not None
        assert "outside workspace" in error

    def test_file_watch_accepts_subdirectory(self):
        """Subdirectory within workspace should be accepted."""
        tmpdir = tempfile.mkdtemp()
        trigger = _make_file_watch_trigger(watch_path="incoming/photos")
        error = validate_trigger(trigger, workspace=tmpdir)
        assert error is None

    def test_validate_watch_path_within_workspace(self):
        tmpdir = tempfile.mkdtemp()
        assert validate_watch_path("subdir", tmpdir) is None

    def test_validate_watch_path_outside_workspace(self):
        tmpdir = tempfile.mkdtemp()
        error = validate_watch_path("/etc/", tmpdir)
        assert error is not None


# ===========================================================================
# Observer Lifecycle Tests
# ===========================================================================

class TestFileWatchObserver:

    @pytest.mark.asyncio
    async def test_file_watch_creates_observer(self):
        """Creating a file_watch trigger should start an Observer."""
        trigger = _make_file_watch_trigger(debounce_seconds=1)
        store, pid, tmpdir = _make_project_store(triggers=[trigger])
        mock_agent_mgr = MagicMock()
        mock_agent_mgr.is_running = MagicMock(return_value=False)
        tm = TriggerManager(store, mock_agent_mgr)

        await tm.start()

        assert "trg_fw001" in tm._file_observers
        observer = tm._file_observers["trg_fw001"]
        assert observer.is_alive()

        # Verify incoming directory was created
        assert os.path.isdir(os.path.join(tmpdir, "incoming"))

        await tm.stop()
        assert "trg_fw001" not in tm._file_observers

    @pytest.mark.asyncio
    async def test_file_watch_disable_stops_observer(self):
        """Disabling a file_watch trigger should stop its observer."""
        trigger = _make_file_watch_trigger(debounce_seconds=1)
        store, pid, tmpdir = _make_project_store(triggers=[trigger])
        mock_agent_mgr = MagicMock()
        tm = TriggerManager(store, mock_agent_mgr)

        await tm.start()
        assert "trg_fw001" in tm._file_observers

        # Unregister (simulating disable)
        tm.unregister_trigger("trg_fw001")
        assert "trg_fw001" not in tm._file_observers

        await tm.stop()

    @pytest.mark.asyncio
    async def test_register_file_watch_replaces_existing(self):
        """Re-registering a trigger should replace the old observer."""
        trigger = _make_file_watch_trigger(debounce_seconds=1)
        store, pid, tmpdir = _make_project_store(triggers=[trigger])
        mock_agent_mgr = MagicMock()
        tm = TriggerManager(store, mock_agent_mgr)

        await tm.start()
        old_observer = tm._file_observers["trg_fw001"]

        # Re-register with updated trigger
        trigger2 = _make_file_watch_trigger(debounce_seconds=2)
        tm.register_trigger(pid, trigger2)
        new_observer = tm._file_observers["trg_fw001"]

        assert new_observer is not old_observer
        assert new_observer.is_alive()

        await tm.stop()


# ===========================================================================
# File Event & Debounce Tests
# ===========================================================================

class TestFileWatchFiring:

    @pytest.mark.asyncio
    async def test_file_watch_fires_on_new_file(self):
        """Creating a file in the watched directory should eventually fire the trigger."""
        trigger = _make_file_watch_trigger(debounce_seconds=1)
        store, pid, tmpdir = _make_project_store(triggers=[trigger])
        mock_agent_mgr = MagicMock()
        mock_agent_mgr.is_running = MagicMock(return_value=False)
        mock_agent_mgr.start_agent = AsyncMock()
        tm = TriggerManager(store, mock_agent_mgr)

        await tm.start()

        # Create a file in the watched directory
        watch_dir = os.path.join(tmpdir, "incoming")
        test_file = os.path.join(watch_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("hello")

        # Wait for debounce (1s) + processing margin
        await asyncio.sleep(2.5)

        # Agent should have been started
        assert mock_agent_mgr.start_agent.call_count >= 1
        call_kwargs = mock_agent_mgr.start_agent.call_args
        assert call_kwargs.kwargs["trigger_source"] == "file_watch"
        assert "test.txt" in call_kwargs.kwargs["initial_message"]

        await tm.stop()

    @pytest.mark.asyncio
    async def test_file_watch_debounce(self):
        """Multiple rapid file creates should be debounced into a single trigger fire."""
        trigger = _make_file_watch_trigger(debounce_seconds=1)
        store, pid, tmpdir = _make_project_store(triggers=[trigger])
        mock_agent_mgr = MagicMock()
        mock_agent_mgr.is_running = MagicMock(return_value=False)
        mock_agent_mgr.start_agent = AsyncMock()
        tm = TriggerManager(store, mock_agent_mgr)

        await tm.start()

        # Create 5 files rapidly
        watch_dir = os.path.join(tmpdir, "incoming")
        for i in range(5):
            with open(os.path.join(watch_dir, f"file{i}.txt"), "w") as f:
                f.write(f"content {i}")
            await asyncio.sleep(0.1)

        # Wait for debounce + processing
        await asyncio.sleep(2.5)

        # Should fire exactly once (all files collected in one batch)
        assert mock_agent_mgr.start_agent.call_count == 1

        await tm.stop()

    @pytest.mark.asyncio
    async def test_file_watch_pattern_filter(self):
        """Only files matching patterns should trigger."""
        trigger = _make_file_watch_trigger(patterns=["*.jpg", "*.png"], debounce_seconds=1)
        store, pid, tmpdir = _make_project_store(triggers=[trigger])
        mock_agent_mgr = MagicMock()
        mock_agent_mgr.is_running = MagicMock(return_value=False)
        mock_agent_mgr.start_agent = AsyncMock()
        tm = TriggerManager(store, mock_agent_mgr)

        await tm.start()

        watch_dir = os.path.join(tmpdir, "incoming")

        # Create a .txt file — should NOT trigger
        with open(os.path.join(watch_dir, "readme.txt"), "w") as f:
            f.write("text")
        await asyncio.sleep(2.0)
        assert mock_agent_mgr.start_agent.call_count == 0

        # Create a .jpg file — SHOULD trigger
        with open(os.path.join(watch_dir, "photo.jpg"), "wb") as f:
            f.write(b"\xff\xd8\xff")
        await asyncio.sleep(2.5)
        assert mock_agent_mgr.start_agent.call_count == 1

        await tm.stop()

    @pytest.mark.asyncio
    async def test_file_watch_changed_files_in_message(self):
        """Trigger message should include the names of changed files."""
        trigger = _make_file_watch_trigger(debounce_seconds=1)
        store, pid, tmpdir = _make_project_store(triggers=[trigger])
        mock_agent_mgr = MagicMock()
        mock_agent_mgr.is_running = MagicMock(return_value=False)
        mock_agent_mgr.start_agent = AsyncMock()
        tm = TriggerManager(store, mock_agent_mgr)

        await tm.start()

        watch_dir = os.path.join(tmpdir, "incoming")
        with open(os.path.join(watch_dir, "report.pdf"), "wb") as f:
            f.write(b"PDF")

        await asyncio.sleep(2.5)

        call_kwargs = mock_agent_mgr.start_agent.call_args
        msg = call_kwargs.kwargs["initial_message"]
        assert "Changed files:" in msg
        assert "report.pdf" in msg
        assert "Process the new files" in msg

        await tm.stop()
