# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: queue state transitions respect PAUSED.

After the items-route auto-start was removed, PAUSED still has special
behavior at the explicit /queue/start endpoint: rather than ignoring the
paused state, /queue/start treats PAUSED as a resume request and
delegates to dispatcher.resume(), which hot-resumes any parked attempt.

These tests verify:
- Adding items while paused: persists, no state change, no auto-start.
- /queue/start while paused: delegates to resume (parked session reused).
- /queue/resume (legacy alias): still works for the same purpose.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.api.routes import agents_v2
from agent_os.api.routes.agents_v2 import (
    QueueAddItemRequest,
    add_queue_item,
    resume_queue,
    start_queue,
)
from agent_os.queue.models import ItemState, QueueRunState
from agent_os.queue.store import QueueStore


class _Globals:
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


def _write_project_state(workspace):
    orbital = os.path.join(str(workspace), "orbital")
    os.makedirs(orbital, exist_ok=True)
    with open(os.path.join(orbital, "PROJECT_STATE.md"), "w", encoding="utf-8") as f:
        f.write("# state\n")


def _setup_manager(store, *, has_handle=False, dispatcher=None):
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
# Items route while paused — staging-only, no state changes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_items_route_while_paused_persists_only(tmp_path):
    """Queue PAUSED, POST item: item is persisted (QUEUED), queue stays
    PAUSED, no auto-start fires (the items route no longer auto-starts
    regardless of state)."""
    project_id = "proj_paused_persist"
    store = QueueStore(tmp_path / "queue.json")
    store.set_queue_state(QueueRunState.PAUSED)

    agent_manager = _setup_manager(store, has_handle=False, dispatcher=None)
    project_store = _setup_project_store(tmp_path, project_id)

    with _Globals(
        _agent_manager=agent_manager,
        _project_store=project_store,
        _ws_manager=MagicMock(),
    ):
        await add_queue_item(project_id, QueueAddItemRequest(content="will wait"))

    state = store.load()
    assert state.state == QueueRunState.PAUSED
    assert len(state.items) == 1
    assert state.items[0].state == ItemState.QUEUED
    agent_manager.ensure_agent_started.assert_not_called()


@pytest.mark.asyncio
async def test_items_route_while_paused_notifies_existing_dispatcher(tmp_path):
    """Paused + existing dispatcher (agent up but queue paused): item
    is persisted and dispatcher gets a nudge. Dispatcher's own _run
    loop ignores nudges while state is PAUSED, so this is harmless."""
    project_id = "proj_paused_notify"
    store = QueueStore(tmp_path / "queue.json")
    store.set_queue_state(QueueRunState.PAUSED)

    fake_dispatcher = MagicMock()
    fake_dispatcher.notify_new_item = MagicMock()

    agent_manager = _setup_manager(
        store, has_handle=True, dispatcher=fake_dispatcher,
    )
    project_store = _setup_project_store(tmp_path, project_id)

    with _Globals(
        _agent_manager=agent_manager,
        _project_store=project_store,
        _ws_manager=MagicMock(),
    ):
        await add_queue_item(project_id, QueueAddItemRequest(content="waiting"))

    agent_manager.ensure_agent_started.assert_not_called()
    fake_dispatcher.notify_new_item.assert_called_once()


# ---------------------------------------------------------------------------
# /queue/start while paused — delegates to dispatcher.resume()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_queue_while_paused_calls_resume(tmp_path):
    """POST /queue/start on PAUSED queue → dispatcher.resume() runs,
    queue flips to RUNNING. Mirrors the old /queue/resume semantics so
    parked attempts hot-resume rather than starting fresh sessions."""
    project_id = "proj_paused_to_running"
    _write_project_state(tmp_path)

    store = QueueStore(tmp_path / "queue.json")
    store.set_queue_state(QueueRunState.PAUSED)

    async def fake_resume():
        store.set_queue_state(QueueRunState.RUNNING)
        return {"status": "running", "resumed_item_id": None}

    fake_dispatcher = MagicMock()
    fake_dispatcher.resume = AsyncMock(side_effect=fake_resume)

    agent_manager = _setup_manager(
        store, has_handle=True, dispatcher=fake_dispatcher,
    )
    project_store = _setup_project_store(tmp_path, project_id)

    with _Globals(
        _agent_manager=agent_manager,
        _project_store=project_store,
        _ws_manager=MagicMock(),
    ):
        await start_queue(project_id)

    fake_dispatcher.resume.assert_awaited_once()
    assert store.load().state == QueueRunState.RUNNING


@pytest.mark.asyncio
async def test_resume_alias_still_works(tmp_path):
    """The legacy /queue/resume endpoint stays as an alias for
    dispatcher.resume() so existing frontends keep working."""
    project_id = "proj_legacy_resume"
    store = QueueStore(tmp_path / "queue.json")
    store.set_queue_state(QueueRunState.PAUSED)

    async def fake_resume():
        store.set_queue_state(QueueRunState.RUNNING)
        return {"status": "running"}

    fake_dispatcher = MagicMock()
    fake_dispatcher.resume = AsyncMock(side_effect=fake_resume)

    agent_manager = MagicMock()
    agent_manager.get_dispatcher = MagicMock(return_value=fake_dispatcher)

    with _Globals(_agent_manager=agent_manager):
        await resume_queue(project_id)

    fake_dispatcher.resume.assert_awaited_once()
    assert store.load().state == QueueRunState.RUNNING
