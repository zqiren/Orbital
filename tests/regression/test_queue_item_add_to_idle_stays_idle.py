# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: adding an item to an IDLE queue does NOT auto-transition.

After the items route stopped auto-starting the agent, IDLE→RUNNING is
ONLY triggered by POST /queue/start. This test pins that contract: a
fresh add to an idle queue must leave state=idle, item=queued, and no
dispatcher activity. Without this guarantee, the queue would silently
restart whenever the user dragged in a new task even though they had
intentionally let the queue drain to idle.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.api.routes import agents_v2
from agent_os.api.routes.agents_v2 import QueueAddItemRequest, add_queue_item
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


@pytest.mark.asyncio
async def test_add_item_to_idle_queue_stays_idle(tmp_path):
    project_id = "proj_idle_add"

    store = QueueStore(tmp_path / "queue.json")
    store.set_queue_state(QueueRunState.IDLE)

    agent_manager = MagicMock()
    agent_manager.has_handle = MagicMock(return_value=False)
    agent_manager.ensure_agent_started = AsyncMock()
    agent_manager.get_dispatcher = MagicMock(return_value=None)
    agent_manager.get_queue_store = MagicMock(return_value=store)

    project_store = MagicMock()
    project_store.get_project = MagicMock(return_value={
        "id": project_id, "workspace": str(tmp_path), "name": "idle",
    })

    with _Globals(
        _agent_manager=agent_manager,
        _project_store=project_store,
        _ws_manager=MagicMock(),
    ):
        await add_queue_item(project_id, QueueAddItemRequest(content="stays queued"))

    state = store.load()
    assert state.state == QueueRunState.IDLE
    assert len(state.items) == 1
    assert state.items[0].state == ItemState.QUEUED
    agent_manager.ensure_agent_started.assert_not_called()


@pytest.mark.asyncio
async def test_add_item_to_idle_with_existing_handle_notifies_only(tmp_path):
    """If the agent happens to be up (e.g., from a chat session) but the
    queue is IDLE, adding an item still must not flip state. The
    dispatcher gets a notify (harmless — it checks state on every tick)."""
    project_id = "proj_idle_with_handle"

    store = QueueStore(tmp_path / "queue.json")
    store.set_queue_state(QueueRunState.IDLE)

    fake_dispatcher = MagicMock()
    fake_dispatcher.notify_new_item = MagicMock()

    agent_manager = MagicMock()
    agent_manager.has_handle = MagicMock(return_value=True)
    agent_manager.ensure_agent_started = AsyncMock()
    agent_manager.get_dispatcher = MagicMock(return_value=fake_dispatcher)
    agent_manager.get_queue_store = MagicMock(return_value=store)

    project_store = MagicMock()
    project_store.get_project = MagicMock(return_value={
        "id": project_id, "workspace": str(tmp_path), "name": "idle-handle",
    })

    with _Globals(
        _agent_manager=agent_manager,
        _project_store=project_store,
        _ws_manager=MagicMock(),
    ):
        await add_queue_item(project_id, QueueAddItemRequest(content="add me"))

    state = store.load()
    assert state.state == QueueRunState.IDLE
    assert state.items[0].state == ItemState.QUEUED
    agent_manager.ensure_agent_started.assert_not_called()
    fake_dispatcher.notify_new_item.assert_called_once()
