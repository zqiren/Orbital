# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: queue items route is staging-only; /queue/start drives drain.

Earlier amendments (Concern 2) had ``POST /queue/items`` auto-start the
agent and the dispatcher. That coupling produced a race during the
agent's onboarding turn — the loop's first run exited text-only before
the queue item was injected, and the dispatcher's _await_and_handle
mis-attributed the text-only exit to the queued item as a contract
violation.

The fix moves auto-start to an explicit endpoint:

- ``POST /queue/items`` — staging only. Persists the item; never starts
  the agent or transitions queue state. Notifies an existing dispatcher
  (no-op if none).
- ``POST /queue/start`` — entry point for draining. Onboarding-gated.
  When queue is IDLE, auto-starts the agent (if needed) and flips state
  to RUNNING. When PAUSED, delegates to dispatcher.resume().

These tests cover both halves of the contract:
1. The items route does NOT auto-start anymore.
2. The /queue/start endpoint does, gated by onboarding completion.
3. The underlying AgentManager helpers (has_handle, ensure_agent_started)
   remain available since /queue/start uses them.
"""

from __future__ import annotations

import inspect
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.api.routes import agents_v2
from agent_os.api.routes.agents_v2 import (
    QueueAddItemRequest,
    add_queue_item,
    start_queue,
)
from agent_os.queue.models import QueueRunState
from agent_os.queue.store import QueueStore


def _write_project_state(workspace):
    """Mimic the onboarding-complete signal: PROJECT_STATE.md on disk."""
    orbital = os.path.join(str(workspace), "orbital")
    os.makedirs(orbital, exist_ok=True)
    with open(os.path.join(orbital, "PROJECT_STATE.md"), "w", encoding="utf-8") as f:
        f.write("# state\n")


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


# ---------------------------------------------------------------------------
# Items route: staging-only, never auto-starts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_items_route_does_not_auto_start_when_no_handle(tmp_path):
    """POST /queue/items with no agent must persist the item and stop.
    Auto-start is the explicit /queue/start endpoint's job."""
    project_id = "proj_no_auto"
    store = QueueStore(tmp_path / "queue.json")

    agent_manager = MagicMock()
    agent_manager.has_handle = MagicMock(return_value=False)
    agent_manager.ensure_agent_started = AsyncMock()
    agent_manager.get_dispatcher = MagicMock(return_value=None)
    agent_manager.get_queue_store = MagicMock(return_value=store)

    project_store = MagicMock()
    project_store.get_project = MagicMock(return_value={
        "id": project_id, "workspace": str(tmp_path), "name": "no auto",
    })

    with _Globals(
        _agent_manager=agent_manager,
        _project_store=project_store,
        _ws_manager=MagicMock(),
    ):
        req = QueueAddItemRequest(content="run the migration")
        result = await add_queue_item(project_id, req)

    # Item persisted.
    state = store.load()
    assert len(state.items) == 1
    assert state.items[0].content == "run the migration"
    # CRITICAL: no auto-start happened.
    agent_manager.ensure_agent_started.assert_not_called()
    # Returned the item.
    assert result["item"]["id"] == state.items[0].id


@pytest.mark.asyncio
async def test_items_route_notifies_dispatcher_when_one_exists(tmp_path):
    """If an agent is already running, the items route still nudges the
    dispatcher (so it picks up the new item on its next tick)."""
    project_id = "proj_existing_dispatcher"
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

    with _Globals(
        _agent_manager=agent_manager,
        _project_store=project_store,
        _ws_manager=MagicMock(),
    ):
        await add_queue_item(project_id, QueueAddItemRequest(content="do it"))

    agent_manager.ensure_agent_started.assert_not_called()
    fake_dispatcher.notify_new_item.assert_called_once()


# ---------------------------------------------------------------------------
# /queue/start: gated by onboarding, drives auto-start when IDLE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_queue_rejects_before_onboarding(tmp_path):
    """No PROJECT_STATE.md on disk → start_queue refuses with 409 and a
    clear message pointing the user at the Chat tab."""
    from fastapi import HTTPException

    project_id = "proj_pre_onboarding"
    store = QueueStore(tmp_path / "queue.json")
    # Intentionally do NOT call _write_project_state — fresh project.

    agent_manager = MagicMock()
    agent_manager.has_handle = MagicMock(return_value=False)
    agent_manager.ensure_agent_started = AsyncMock()
    agent_manager.get_dispatcher = MagicMock(return_value=None)
    agent_manager.get_queue_store = MagicMock(return_value=store)

    project_store = MagicMock()
    project_store.get_project = MagicMock(return_value={
        "id": project_id, "workspace": str(tmp_path), "name": "fresh",
    })

    with _Globals(
        _agent_manager=agent_manager,
        _project_store=project_store,
        _ws_manager=MagicMock(),
    ), pytest.raises(HTTPException) as ei:
        await start_queue(project_id)

    assert ei.value.status_code == 409
    assert "onboarding" in ei.value.detail.lower()
    agent_manager.ensure_agent_started.assert_not_called()


@pytest.mark.asyncio
async def test_start_queue_auto_starts_when_idle_and_onboarded(tmp_path):
    """PROJECT_STATE.md present, queue IDLE, no handle:
    /queue/start auto-starts agent, flips state to RUNNING, notifies dispatcher."""
    project_id = "proj_idle_start"
    _write_project_state(tmp_path)

    store = QueueStore(tmp_path / "queue.json")
    store.set_queue_state(QueueRunState.IDLE)

    fake_dispatcher = MagicMock()
    fake_dispatcher.notify_new_item = MagicMock()

    agent_manager = MagicMock()
    agent_manager.has_handle = MagicMock(return_value=False)
    agent_manager.ensure_agent_started = AsyncMock(return_value=True)
    agent_manager.get_dispatcher = MagicMock(return_value=fake_dispatcher)
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
        result = await start_queue(project_id)

    agent_manager.ensure_agent_started.assert_awaited_once_with(project_id)
    assert store.load().state == QueueRunState.RUNNING
    fake_dispatcher.notify_new_item.assert_called_once()
    assert result == {"status": "running"}


@pytest.mark.asyncio
async def test_start_queue_no_op_when_already_running(tmp_path):
    """If queue state is already RUNNING, /queue/start short-circuits
    with already_running and does not start anything."""
    project_id = "proj_already_running"
    _write_project_state(tmp_path)

    store = QueueStore(tmp_path / "queue.json")
    # Default state is RUNNING

    agent_manager = MagicMock()
    agent_manager.has_handle = MagicMock(return_value=True)
    agent_manager.ensure_agent_started = AsyncMock()
    agent_manager.get_dispatcher = MagicMock(return_value=MagicMock())
    agent_manager.get_queue_store = MagicMock(return_value=store)

    project_store = MagicMock()
    project_store.get_project = MagicMock(return_value={
        "id": project_id, "workspace": str(tmp_path), "name": "running",
    })

    with _Globals(
        _agent_manager=agent_manager,
        _project_store=project_store,
        _ws_manager=MagicMock(),
    ):
        result = await start_queue(project_id)

    assert result == {"status": "already_running"}
    agent_manager.ensure_agent_started.assert_not_called()


@pytest.mark.asyncio
async def test_start_queue_delegates_to_resume_when_paused(tmp_path):
    """PAUSED queue → /queue/start calls dispatcher.resume() so the parked
    attempt session hot-resumes (same semantics as the old /queue/resume)."""
    project_id = "proj_paused_resume"
    _write_project_state(tmp_path)

    store = QueueStore(tmp_path / "queue.json")
    store.set_queue_state(QueueRunState.PAUSED)

    async def fake_resume():
        store.set_queue_state(QueueRunState.RUNNING)
        return {"status": "running", "resumed_item_id": None}

    fake_dispatcher = MagicMock()
    fake_dispatcher.resume = AsyncMock(side_effect=fake_resume)

    agent_manager = MagicMock()
    agent_manager.has_handle = MagicMock(return_value=True)
    agent_manager.ensure_agent_started = AsyncMock()
    agent_manager.get_dispatcher = MagicMock(return_value=fake_dispatcher)
    agent_manager.get_queue_store = MagicMock(return_value=store)

    project_store = MagicMock()
    project_store.get_project = MagicMock(return_value={
        "id": project_id, "workspace": str(tmp_path), "name": "paused",
    })

    with _Globals(
        _agent_manager=agent_manager,
        _project_store=project_store,
        _ws_manager=MagicMock(),
    ):
        result = await start_queue(project_id)

    fake_dispatcher.resume.assert_awaited_once()
    assert store.load().state == QueueRunState.RUNNING
    assert result["status"] == "running"


# ---------------------------------------------------------------------------
# AgentManager helpers (unchanged contract; /queue/start still uses them)
# ---------------------------------------------------------------------------


def _make_agent_manager_with_project(workspace, project_id="proj_x"):
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

    return AgentManager(
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


def test_has_handle_returns_false_when_no_handle(tmp_path):
    mgr = _make_agent_manager_with_project(tmp_path)
    assert mgr.has_handle("proj_x") is False


def test_has_handle_returns_true_when_handle_present(tmp_path):
    mgr = _make_agent_manager_with_project(tmp_path)
    # F7: handles are keyed by SessionKey == (project_id, session_id)
    # since #22's multi-session foundation.
    from agent_os.daemon_v2.models import DEFAULT_SESSION_ID, make_session_key
    mgr._handles[make_session_key("proj_x", DEFAULT_SESSION_ID)] = MagicMock()
    assert mgr.has_handle("proj_x") is True


@pytest.mark.asyncio
async def test_ensure_agent_started_starts_without_initial_message(tmp_path):
    mgr = _make_agent_manager_with_project(tmp_path)

    captured = []

    async def fake_start(pid, config, **kwargs):
        captured.append({"pid": pid, "kwargs": kwargs})

    mgr.start_agent = fake_start
    started = await mgr.ensure_agent_started("proj_x")

    assert started is True
    assert len(captured) == 1
    assert captured[0]["pid"] == "proj_x"
    assert captured[0]["kwargs"].get("initial_message") is None


@pytest.mark.asyncio
async def test_ensure_agent_started_is_no_op_when_handle_exists(tmp_path):
    mgr = _make_agent_manager_with_project(tmp_path)
    # F7: handles are keyed by SessionKey == (project_id, session_id).
    from agent_os.daemon_v2.models import DEFAULT_SESSION_ID, make_session_key
    mgr._handles[make_session_key("proj_x", DEFAULT_SESSION_ID)] = MagicMock()

    calls = []

    async def fake_start(pid, config, **kwargs):
        calls.append(pid)

    mgr.start_agent = fake_start
    started = await mgr.ensure_agent_started("proj_x")

    assert started is False
    assert calls == []


def test_ensure_agent_started_is_async():
    from agent_os.daemon_v2.agent_manager import AgentManager
    assert inspect.iscoroutinefunction(AgentManager.ensure_agent_started)


def test_has_handle_is_sync_returns_bool():
    from agent_os.daemon_v2.agent_manager import AgentManager
    assert not inspect.iscoroutinefunction(AgentManager.has_handle)
