# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Step 1 — dispatcher lifecycle decoupled from agent lifecycle.

The dispatcher is a per-project session-lifecycle manager: created at daemon
startup (one per project), surviving agent stop/start, and torn down only at
daemon shutdown or project deletion. These tests lock in that contract:

  - start_all_dispatchers() creates one dispatcher per project.
  - stop_agent() does NOT destroy the project's dispatcher.
  - shutdown() sweeps and shuts down every dispatcher (even with no agent).
  - shutdown_dispatcher() tears down a single project's dispatcher (delete path).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle


def _make_manager(tmp_path, project_store=None):
    mgr = AgentManager(
        project_store=project_store or MagicMock(),
        ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(),
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
        platform_provider=None,
        registry=MagicMock(),
        setup_engine=MagicMock(),
        settings_store=None,
        credential_store=None,
    )
    mgr._state_file = tmp_path / "daemon-state.json"
    return mgr


def _make_handle(task_is_none=True):
    session = MagicMock()
    task = None if task_is_none else MagicMock(spec=asyncio.Task)
    return ProjectHandle(
        session=session,
        loop=MagicMock(),
        provider=MagicMock(),
        registry=MagicMock(),
        context_manager=MagicMock(),
        interceptor=MagicMock(),
        task=task,
        config_snapshot={"workspace": "/tmp/test", "model": "gpt-4"},
        started_at="2026-01-01T00:00:00+00:00",
    )


@pytest.mark.asyncio
async def test_start_all_dispatchers_creates_one_per_project(tmp_path):
    """start_all_dispatchers iterates the project store and ensures a
    dispatcher for every project with a workspace."""
    ps = MagicMock()
    ps.list_projects.return_value = [
        {"project_id": "p1", "workspace": str(tmp_path / "w1")},
        {"project_id": "p2", "workspace": str(tmp_path / "w2")},
        {"project_id": "p3", "workspace": ""},  # no workspace -> skipped
    ]
    mgr = _make_manager(tmp_path, project_store=ps)
    mgr._ensure_dispatcher = AsyncMock()

    await mgr.start_all_dispatchers()

    called_projects = {c.args[0] for c in mgr._ensure_dispatcher.await_args_list}
    assert called_projects == {"p1", "p2"}  # p3 skipped (no workspace)


@pytest.mark.asyncio
async def test_stop_agent_preserves_dispatcher(tmp_path):
    """stop_agent must NOT tear down the project's dispatcher — it outlives
    any individual agent."""
    mgr = _make_manager(tmp_path)
    mgr._sub_agent_manager.stop_all = AsyncMock()
    fake_dispatcher = MagicMock()
    fake_dispatcher.shutdown = AsyncMock()
    mgr._dispatchers["proj-1"] = fake_dispatcher
    mgr._handles[("proj-1", "default")] = _make_handle(task_is_none=True)

    await mgr.stop_agent("proj-1")

    assert "proj-1" in mgr._dispatchers, "dispatcher must survive agent stop"
    fake_dispatcher.shutdown.assert_not_awaited()


@pytest.mark.asyncio
async def test_shutdown_sweeps_all_dispatchers(tmp_path):
    """Daemon shutdown shuts down every dispatcher, including ones with no
    active agent handle (which the _handles-based teardown would miss)."""
    mgr = _make_manager(tmp_path)
    d1, d2 = MagicMock(), MagicMock()
    d1.shutdown = AsyncMock()
    d2.shutdown = AsyncMock()
    mgr._dispatchers["p1"] = d1
    mgr._dispatchers["p2"] = d2
    # No handles -> the existing handle teardown loop is a no-op.

    await mgr.shutdown()

    d1.shutdown.assert_awaited_once()
    d2.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_shutdown_dispatcher_tears_down_one(tmp_path):
    """shutdown_dispatcher pops and shuts down a single project's dispatcher
    (used by the project-delete path)."""
    mgr = _make_manager(tmp_path)
    fake = MagicMock()
    fake.shutdown = AsyncMock()
    mgr._dispatchers["gone"] = fake

    await mgr.shutdown_dispatcher("gone")

    assert "gone" not in mgr._dispatchers
    fake.shutdown.assert_awaited_once()
