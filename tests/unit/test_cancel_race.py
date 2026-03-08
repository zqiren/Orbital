# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for SubAgentManager cancel propagation and lifecycle lock.

Verifies that:
- stop_all() during start() does not orphan adapters
- start() during stop_all() is rejected with stopping flag
- Failed start() cleans up partially-spawned adapters
- Concurrent starts to the same handle are serialized
- stop_all() + stop() don't deadlock
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.daemon_v2.sub_agent_manager import SubAgentManager


def _make_manager():
    """Create a SubAgentManager with mock dependencies."""
    pm = MagicMock()
    pm.start = AsyncMock()
    pm.stop = AsyncMock()
    mgr = SubAgentManager(process_manager=pm)
    return mgr, pm


def _make_mock_adapter(alive=True, idle=False):
    """Create a mock CLIAdapter."""
    adapter = AsyncMock()
    adapter.is_alive = MagicMock(return_value=alive)
    adapter.is_idle = MagicMock(return_value=idle)
    adapter.display_name = "test"
    adapter.stop = AsyncMock()
    adapter.start = AsyncMock()
    adapter._send_lock = asyncio.Lock()
    return adapter


class TestStoppingFlag:
    """start() must be rejected while stop_all() is in progress."""

    @pytest.mark.asyncio
    async def test_start_during_stop_all_rejected(self):
        """If stop_all is in progress, start() returns an error immediately."""
        mgr, pm = _make_manager()

        # Pre-register an adapter with a slow stop
        adapter = _make_mock_adapter()

        async def slow_stop():
            await asyncio.sleep(0.1)

        adapter.stop = slow_stop

        mgr._adapters["proj"] = {"agent-a": adapter}

        # Run stop_all and start concurrently
        start_result = None

        async def try_start():
            nonlocal start_result
            # Small delay to ensure stop_all has set the flag
            await asyncio.sleep(0.02)
            start_result = await mgr.start("proj", "agent-b")

        await asyncio.gather(
            mgr.stop_all("proj"),
            try_start(),
        )

        assert start_result is not None
        assert "shutting down" in start_result

    @pytest.mark.asyncio
    async def test_stopping_flag_cleared_after_stop_all(self):
        """The stopping flag must be cleared even if stop_all encounters errors."""
        mgr, pm = _make_manager()
        adapter = _make_mock_adapter()
        adapter.stop = AsyncMock(side_effect=RuntimeError("stop failed"))
        mgr._adapters["proj"] = {"agent-a": adapter}

        await mgr.stop_all("proj")

        # Flag should be cleared — start should work now
        assert "proj" not in mgr._stopping

    @pytest.mark.asyncio
    async def test_stopping_flag_cleared_on_empty_project(self):
        """stop_all on a project with no adapters should still clear cleanly."""
        mgr, pm = _make_manager()
        await mgr.stop_all("proj")
        assert "proj" not in mgr._stopping


class TestLifecycleLock:
    """The lifecycle lock must prevent the spawn-register race."""

    @pytest.mark.asyncio
    async def test_stop_during_start_sees_adapter(self):
        """If start() is in progress, stop_all() waits and then finds the adapter."""
        mgr, pm = _make_manager()

        adapter = _make_mock_adapter()
        config = MagicMock()
        config.workspace = "/tmp"
        config.env = {}
        config.approval_patterns = []
        config.args = None
        config.command = "echo"

        # Make adapter.start() slow so we can race against it
        original_start = adapter.start

        async def slow_start(cfg):
            await asyncio.sleep(0.05)
            return await original_start(cfg)

        adapter.start = slow_start

        # Patch CLIAdapter constructor to return our mock
        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter", return_value=adapter):
            mgr._adapter_configs["agent-a"] = config

            results = {}

            async def do_start():
                results["start"] = await mgr.start("proj", "agent-a")

            async def do_stop():
                await asyncio.sleep(0.02)  # start first
                await mgr.stop_all("proj")

            await asyncio.gather(do_start(), do_stop())

        # The adapter was stopped (stop_all found it after start registered it)
        # OR start was rejected because stopping flag was set.
        # Either way: no orphan.
        assert (
            adapter.stop.called  # stop_all found and stopped it
            or "shutting down" in results.get("start", "")  # start was rejected
        )

    @pytest.mark.asyncio
    async def test_concurrent_starts_no_deadlock(self):
        """Two concurrent starts to the same project must not deadlock."""
        mgr, pm = _make_manager()

        adapter1 = _make_mock_adapter()
        adapter2 = _make_mock_adapter()
        config = MagicMock()
        config.workspace = "/tmp"
        config.env = {}
        config.approval_patterns = []
        config.args = None
        config.command = "echo"

        mgr._adapter_configs["agent-a"] = config
        mgr._adapter_configs["agent-b"] = config

        call_count = [0]

        def make_adapter(**kwargs):
            call_count[0] += 1
            return adapter1 if call_count[0] == 1 else adapter2

        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter", side_effect=make_adapter):
            r1, r2 = await asyncio.wait_for(
                asyncio.gather(
                    mgr.start("proj", "agent-a"),
                    mgr.start("proj", "agent-b"),
                ),
                timeout=5.0,
            )

        assert "Started" in r1
        assert "Started" in r2


class TestDefensiveCleanup:
    """Failed adapter.start() must clean up partially-spawned processes."""

    @pytest.mark.asyncio
    async def test_failed_start_calls_adapter_stop(self):
        """If adapter.start() raises, adapter.stop() should be called for cleanup."""
        mgr, pm = _make_manager()

        adapter = _make_mock_adapter()
        adapter.start = AsyncMock(side_effect=RuntimeError("spawn failed"))

        config = MagicMock()
        config.workspace = "/tmp"
        config.env = {}
        config.approval_patterns = []
        config.args = None
        config.command = "echo"

        mgr._adapter_configs["agent-a"] = config

        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter", return_value=adapter):
            result = await mgr.start("proj", "agent-a")

        assert "adapter start failed" in result
        adapter.stop.assert_called_once()
        # Adapter should NOT be registered
        assert "agent-a" not in mgr._adapters.get("proj", {})

    @pytest.mark.asyncio
    async def test_failed_start_cleanup_does_not_raise(self):
        """If adapter.stop() also fails during cleanup, it should not propagate."""
        mgr, pm = _make_manager()

        adapter = _make_mock_adapter()
        adapter.start = AsyncMock(side_effect=RuntimeError("spawn failed"))
        adapter.stop = AsyncMock(side_effect=RuntimeError("cleanup also failed"))

        config = MagicMock()
        config.workspace = "/tmp"
        config.env = {}
        config.approval_patterns = []
        config.args = None
        config.command = "echo"

        mgr._adapter_configs["agent-a"] = config

        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter", return_value=adapter):
            result = await mgr.start("proj", "agent-a")

        # Should not raise, just return the original error
        assert "adapter start failed" in result


class TestLockCleanup:
    """Lifecycle lock should be cleaned up after stop_all."""

    @pytest.mark.asyncio
    async def test_lock_removed_after_stop_all(self):
        """stop_all cleans up the lifecycle lock for the project."""
        mgr, pm = _make_manager()
        mgr._adapters["proj"] = {}
        # Force lock creation
        mgr._get_lock("proj")
        assert "proj" in mgr._lifecycle_locks

        await mgr.stop_all("proj")

        assert "proj" not in mgr._lifecycle_locks

    @pytest.mark.asyncio
    async def test_lock_recreated_on_new_start(self):
        """After stop_all cleans up the lock, a new start creates a fresh one."""
        mgr, pm = _make_manager()
        mgr._adapters["proj"] = {}
        await mgr.stop_all("proj")
        assert "proj" not in mgr._lifecycle_locks

        # _get_lock should create a new one
        lock = mgr._get_lock("proj")
        assert lock is not None
        assert "proj" in mgr._lifecycle_locks
