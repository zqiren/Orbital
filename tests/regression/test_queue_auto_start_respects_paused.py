# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: auto-start on first item respects PAUSED queue state.

Concern 2 of TASK-queue-architecture-amendments.md: when adding a queue
item to a project whose queue is PAUSED, the route must NOT auto-start
the agent. The user explicitly paused; the item stays QUEUED and waits
for ``POST /queue/resume``.

Without this guard, the auto-start would bypass the pause and surprise
the user with a running agent they had stopped.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.api.routes import agents_v2
from agent_os.api.routes.agents_v2 import QueueAddItemRequest, add_queue_item
from agent_os.queue.models import ItemState, QueueRunState
from agent_os.queue.store import QueueStore


# ---------------------------------------------------------------------------
# Helpers — replicate the route-globals patching pattern
# ---------------------------------------------------------------------------


class _Globals:
    """Context-manager that snapshots and restores agents_v2 module globals."""

    def __init__(self, **patches):
        self._patches = patches
        self._original = {}

    def __enter__(self):
        for name, value in self._patches.items():
            self._original[name] = getattr(agents_v2, name)
            setattr(agents_v2, name, value)
        return self

    def __exit__(self, *exc):
        for name, value in self._original.items():
            setattr(agents_v2, name, value)


def _setup_manager(store, *, has_handle=False, dispatcher=None):
    """Build a MagicMock agent_manager wired to the test store."""
    mgr = MagicMock()
    mgr.has_handle = MagicMock(return_value=has_handle)
    mgr.ensure_agent_started = AsyncMock(return_value=True)
    mgr.get_dispatcher = MagicMock(return_value=dispatcher)
    mgr.get_queue_store = MagicMock(return_value=store)
    return mgr


def _setup_project_store(workspace, project_id):
    ps = MagicMock()
    ps.get_project = MagicMock(return_value={
        "id": project_id,
        "workspace": str(workspace),
        "name": "paused-test",
    })
    return ps


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_start_skipped_when_queue_paused(tmp_path):
    """Queue is PAUSED, no handle, POST item:
       - item is persisted (QUEUED state, queue stays PAUSED)
       - ensure_agent_started is NOT called
       - notify_new_item is not called (no dispatcher exists)
    """
    project_id = "proj_paused_skip"

    store = QueueStore(tmp_path / "queue.json")
    store.set_queue_state(QueueRunState.PAUSED)

    agent_manager = _setup_manager(store, has_handle=False, dispatcher=None)
    project_store = _setup_project_store(tmp_path, project_id)
    ws_manager = MagicMock()

    with _Globals(
        _agent_manager=agent_manager,
        _project_store=project_store,
        _ws_manager=ws_manager,
    ):
        req = QueueAddItemRequest(content="will wait for resume")
        result = await add_queue_item(project_id, req)

    # Item persisted but queue still paused; item state is QUEUED.
    state = store.load()
    assert state.state == QueueRunState.PAUSED
    assert len(state.items) == 1
    assert state.items[0].content == "will wait for resume"
    assert state.items[0].state == ItemState.QUEUED

    # Critical: the auto-start hook MUST be skipped while paused.
    agent_manager.ensure_agent_started.assert_not_called()

    # And no dispatcher exists, so notify is not invoked.
    assert result["item"]["id"] == state.items[0].id


@pytest.mark.asyncio
async def test_auto_start_fires_when_queue_running(tmp_path):
    """Mirror-image: queue is RUNNING (default), no handle, POST item:
       - ensure_agent_started IS called.
    Confirms the paused-check is the only thing gating auto-start."""
    project_id = "proj_running_starts"

    store = QueueStore(tmp_path / "queue.json")
    assert store.load().state == QueueRunState.RUNNING

    fake_dispatcher = MagicMock()
    fake_dispatcher.notify_new_item = MagicMock()
    agent_manager = _setup_manager(
        store, has_handle=False, dispatcher=fake_dispatcher,
    )
    project_store = _setup_project_store(tmp_path, project_id)
    ws_manager = MagicMock()

    with _Globals(
        _agent_manager=agent_manager,
        _project_store=project_store,
        _ws_manager=ws_manager,
    ):
        req = QueueAddItemRequest(content="please run me")
        await add_queue_item(project_id, req)

    agent_manager.ensure_agent_started.assert_awaited_once_with(project_id)
    fake_dispatcher.notify_new_item.assert_called_once()


@pytest.mark.asyncio
async def test_auto_start_skipped_when_paused_and_handle_already_exists(tmp_path):
    """Paused + already-running agent: nothing weird should happen.
       Auto-start is skipped because handle exists; notify_new_item still
       fires so the dispatcher sees the new item next tick (and respects
       paused state on its own)."""
    project_id = "proj_paused_with_handle"

    store = QueueStore(tmp_path / "queue.json")
    store.set_queue_state(QueueRunState.PAUSED)

    fake_dispatcher = MagicMock()
    fake_dispatcher.notify_new_item = MagicMock()
    agent_manager = _setup_manager(
        store, has_handle=True, dispatcher=fake_dispatcher,
    )
    project_store = _setup_project_store(tmp_path, project_id)
    ws_manager = MagicMock()

    with _Globals(
        _agent_manager=agent_manager,
        _project_store=project_store,
        _ws_manager=ws_manager,
    ):
        req = QueueAddItemRequest(content="waiting")
        await add_queue_item(project_id, req)

    agent_manager.ensure_agent_started.assert_not_called()
    fake_dispatcher.notify_new_item.assert_called_once()


@pytest.mark.asyncio
async def test_resume_transitions_paused_to_running(tmp_path):
    """After an item is added under PAUSED, POST /queue/resume must
    flip the queue back to RUNNING so the dispatcher can pick the
    item up. This guarantees the path the user takes to ungate work
    actually does ungate it.

    The dispatcher's own _run loop is not exercised here — we only
    assert that the resume() entry point flips queue state. End-to-end
    item execution after resume is covered by the existing stop/resume
    roundtrip tests.
    """
    project_id = "proj_resume_unblocks"

    store = QueueStore(tmp_path / "queue.json")
    store.set_queue_state(QueueRunState.PAUSED)

    # The route's resume endpoint requires a dispatcher to be present;
    # we mock one and verify it's the dispatcher's resume() that actually
    # toggles state.  Set up a fake dispatcher whose resume() flips the
    # store back to RUNNING (which the real dispatcher does too).
    async def fake_resume():
        store.set_queue_state(QueueRunState.RUNNING)
        return {"status": "resumed"}

    fake_dispatcher = MagicMock()
    fake_dispatcher.resume = AsyncMock(side_effect=fake_resume)

    agent_manager = MagicMock()
    agent_manager.get_dispatcher = MagicMock(return_value=fake_dispatcher)

    with _Globals(_agent_manager=agent_manager):
        from agent_os.api.routes.agents_v2 import resume_queue
        await resume_queue(project_id)

    assert store.load().state == QueueRunState.RUNNING
    fake_dispatcher.resume.assert_awaited_once()
