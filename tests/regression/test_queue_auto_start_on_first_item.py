# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: adding the first item to an empty project auto-starts the agent.

Concern 2 of TASK-queue-architecture-amendments.md: previously
``POST /api/v2/projects/{pid}/queue/items`` only nudged the dispatcher
*if it existed*. With no agent running, no dispatcher existed and the
item sat idle until the user manually started the agent. The fix wires
the route to call ``AgentManager.ensure_agent_started`` after the item
is persisted, so the dispatcher comes up and drains the queue without
manual intervention.

These tests cover the route-level behavior: that the auto-start hook
fires (with no ``initial_message``, because the dispatcher must wrap the
item with a ``[QUEUE ITEM]`` header — Concern 3), and that the helper on
``AgentManager`` honors its contract (no-op when a handle already
exists, raises on missing project).
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.api.routes import agents_v2
from agent_os.api.routes.agents_v2 import QueueAddItemRequest, add_queue_item
from agent_os.queue.models import QueueRunState
from agent_os.queue.store import QueueStore


# ---------------------------------------------------------------------------
# Route-level: POST /queue/items auto-starts the agent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_queue_item_auto_starts_agent_when_no_handle(tmp_path):
    """No agent, no dispatcher, POST item → ensure_agent_started fires
    and the new dispatcher is notified about the item."""
    project_id = "proj_auto_start_first"

    store = QueueStore(tmp_path / "queue.json")

    # Spy on ensure_agent_started: don't actually start anything; just
    # record the call and pretend a handle came into existence so the
    # subsequent get_dispatcher path runs.
    fake_dispatcher = MagicMock()
    fake_dispatcher.notify_new_item = MagicMock()

    agent_manager = MagicMock()
    # has_handle returns False the first time (auto-start path), True
    # afterwards (so the route's existing notify-dispatcher branch sees a
    # live agent). The route only calls has_handle once, but be safe.
    agent_manager.has_handle = MagicMock(return_value=False)
    agent_manager.ensure_agent_started = AsyncMock(return_value=True)
    agent_manager.get_dispatcher = MagicMock(return_value=fake_dispatcher)

    project_store = MagicMock()
    project_store.get_project = MagicMock(return_value={
        "id": project_id,
        "workspace": str(tmp_path),
        "name": "auto-start project",
    })

    ws_manager = MagicMock()

    # Patch module globals (configure() normally sets these at app boot).
    orig = (
        agents_v2._agent_manager,
        agents_v2._project_store,
        agents_v2._ws_manager,
    )
    agents_v2._agent_manager = agent_manager
    agents_v2._project_store = project_store
    agents_v2._ws_manager = ws_manager

    # Route uses _resolve_queue_store which goes through
    # agent_manager.get_queue_store(). Wire that to our temp store so the
    # add_item call writes to disk and we can inspect it.
    agent_manager.get_queue_store = MagicMock(return_value=store)

    try:
        req = QueueAddItemRequest(content="run the migration")
        result = await add_queue_item(project_id, req)
    finally:
        (
            agents_v2._agent_manager,
            agents_v2._project_store,
            agents_v2._ws_manager,
        ) = orig

    # 1. Item is in the store.
    state = store.load()
    assert len(state.items) == 1
    assert state.items[0].content == "run the migration"

    # 2. ensure_agent_started was awaited exactly once for this project.
    agent_manager.ensure_agent_started.assert_awaited_once_with(project_id)

    # 3. Dispatcher (whatever ensure_agent_started conceptually created)
    #    was nudged about the new item.
    fake_dispatcher.notify_new_item.assert_called_once()

    # 4. The route returned the persisted item.
    assert result["item"]["id"] == state.items[0].id


@pytest.mark.asyncio
async def test_post_queue_item_does_not_restart_when_handle_exists(tmp_path):
    """When an agent is already running, the route MUST NOT auto-start;
    it should only nudge the existing dispatcher (today's behavior)."""
    project_id = "proj_already_running"

    store = QueueStore(tmp_path / "queue.json")

    fake_dispatcher = MagicMock()
    fake_dispatcher.notify_new_item = MagicMock()

    agent_manager = MagicMock()
    agent_manager.has_handle = MagicMock(return_value=True)
    agent_manager.ensure_agent_started = AsyncMock()
    agent_manager.get_dispatcher = MagicMock(return_value=fake_dispatcher)
    agent_manager.get_queue_store = MagicMock(return_value=store)

    project_store = MagicMock()
    project_store.get_project = MagicMock(return_value={
        "id": project_id, "workspace": str(tmp_path), "name": "running",
    })

    ws_manager = MagicMock()

    orig = (
        agents_v2._agent_manager,
        agents_v2._project_store,
        agents_v2._ws_manager,
    )
    agents_v2._agent_manager = agent_manager
    agents_v2._project_store = project_store
    agents_v2._ws_manager = ws_manager

    try:
        req = QueueAddItemRequest(content="do the thing")
        await add_queue_item(project_id, req)
    finally:
        (
            agents_v2._agent_manager,
            agents_v2._project_store,
            agents_v2._ws_manager,
        ) = orig

    agent_manager.ensure_agent_started.assert_not_called()
    fake_dispatcher.notify_new_item.assert_called_once()


# ---------------------------------------------------------------------------
# AgentManager.has_handle + ensure_agent_started behavior
# ---------------------------------------------------------------------------


def _make_agent_manager_with_project(workspace, project_id="proj_x"):
    """Build a minimal AgentManager whose project store returns one project
    so ``_build_agent_config_from_project`` succeeds."""
    from agent_os.config.provider_registry import ProviderRegistry
    from agent_os.daemon_v2.agent_manager import AgentManager

    project_store = MagicMock()
    project_store.get_project = MagicMock(return_value={
        "id": project_id,
        "workspace": str(workspace),
        "name": "ensure-test",
        "model": "gpt-4o",
        "api_key": "sk-test",
        "autonomy": "hands_off",
        "sdk": "openai",
        "provider": "custom",
    })

    ws = MagicMock()
    ws.broadcast = MagicMock()
    ws.add_broadcast_hook = MagicMock()

    mgr = AgentManager(
        project_store=project_store,
        ws_manager=ws,
        sub_agent_manager=MagicMock(),
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
        platform_provider=None,
        registry=MagicMock(),
        setup_engine=None,
        settings_store=None,
        credential_store=None,
        browser_manager=None,
        user_credential_store=None,
        provider_registry=ProviderRegistry(),
    )
    return mgr


def test_has_handle_returns_false_when_no_handle(tmp_path):
    mgr = _make_agent_manager_with_project(tmp_path)
    assert mgr.has_handle("proj_x") is False


def test_has_handle_returns_true_when_handle_present(tmp_path):
    mgr = _make_agent_manager_with_project(tmp_path)
    mgr._handles["proj_x"] = MagicMock()
    assert mgr.has_handle("proj_x") is True


@pytest.mark.asyncio
async def test_ensure_agent_started_starts_without_initial_message(tmp_path):
    """``ensure_agent_started`` must call ``start_agent`` with NO
    initial_message — the queue dispatcher injects the item itself."""
    mgr = _make_agent_manager_with_project(tmp_path)

    captured = []

    async def fake_start(pid, config, **kwargs):
        captured.append({"pid": pid, "config": config, "kwargs": kwargs})

    mgr.start_agent = fake_start

    started = await mgr.ensure_agent_started("proj_x")

    assert started is True
    assert len(captured) == 1
    assert captured[0]["pid"] == "proj_x"
    # No initial_message should be passed — the dispatcher wraps the
    # queue item with a [QUEUE ITEM] header before injection (Concern 3).
    assert captured[0]["kwargs"].get("initial_message") is None


@pytest.mark.asyncio
async def test_ensure_agent_started_is_no_op_when_handle_exists(tmp_path):
    """If a handle already exists, ``ensure_agent_started`` must not
    re-start the agent."""
    mgr = _make_agent_manager_with_project(tmp_path)
    mgr._handles["proj_x"] = MagicMock()

    start_calls = []

    async def fake_start(pid, config, **kwargs):
        start_calls.append(pid)

    mgr.start_agent = fake_start

    started = await mgr.ensure_agent_started("proj_x")

    assert started is False
    assert start_calls == []


@pytest.mark.asyncio
async def test_ensure_agent_started_raises_when_project_missing(tmp_path):
    mgr = _make_agent_manager_with_project(tmp_path)
    mgr._project_store.get_project = MagicMock(return_value=None)

    with pytest.raises(KeyError):
        await mgr.ensure_agent_started("does_not_exist")


# ---------------------------------------------------------------------------
# Sanity: signature checks so the helper stays callable from the route
# ---------------------------------------------------------------------------


def test_ensure_agent_started_is_async():
    from agent_os.daemon_v2.agent_manager import AgentManager
    assert inspect.iscoroutinefunction(AgentManager.ensure_agent_started)


def test_has_handle_is_sync_returns_bool():
    from agent_os.daemon_v2.agent_manager import AgentManager
    assert not inspect.iscoroutinefunction(AgentManager.has_handle)
