# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 079 §3.3 — choosing which agent runs an automation.

An assigned trigger fires down the same funnel an assigned queue item and a
chat @mention take: the task goes straight to the chosen worker, and the
management agent is woken by the worker's terminal event to act on the result
and notify — which is what its turn does today anyway. Unlike the queue there
is no verdict to await, so there is no slot hold here.

An UNASSIGNED trigger must keep starting the management agent exactly as
before; that is pinned explicitly below.
"""

import tempfile

import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_os.daemon_v2.project_store import ProjectStore
from agent_os.daemon_v2.trigger_manager import TriggerManager


def _project_with_trigger(agent=None):
    tmpdir = tempfile.mkdtemp()
    store = ProjectStore(data_dir=tmpdir)
    pid = store.create_project({
        "name": "Test Project",
        "workspace": tmpdir,
        "model": "gpt-4",
        "api_key": "sk-test",
    })
    trigger = {
        "id": "trg_aaa", "name": "Nightly", "type": "schedule", "enabled": True,
        "schedule": {"cron": "0 7 * * *", "human": "Daily at 7am"},
        "task": "Do the thing", "trigger_count": 0, "last_triggered": None,
    }
    if agent is not None:
        trigger["agent"] = agent
    store.update_project(pid, {"triggers": [trigger]})
    return store, pid


def _manager(send_result="Message sent to codex."):
    sub_mgr = MagicMock()
    sub_mgr.send = AsyncMock(return_value=send_result)
    mgr = MagicMock()
    mgr.is_running = MagicMock(return_value=False)
    mgr.start_agent = AsyncMock()
    mgr.new_session = AsyncMock(return_value={"session_id": "proj_sess1"})
    mgr.get_sub_agent_manager = MagicMock(return_value=sub_mgr)
    return mgr, sub_mgr


@pytest.mark.asyncio
async def test_assigned_trigger_dispatches_to_the_worker():
    store, pid = _project_with_trigger(agent="codex")
    mgr, sub_mgr = _manager()
    tm = TriggerManager(store, mgr)

    await tm._fire_trigger(pid, "trg_aaa")

    mgr.start_agent.assert_not_awaited()
    sub_mgr.send.assert_awaited_once()
    args, kwargs = sub_mgr.send.await_args
    assert args[1] == "codex"
    assert "Do the thing" in args[2]
    assert kwargs["session_id"] == "proj_sess1"
    # queue_item (spec 079), not user_mention: same direct-send funnel, but
    # the dispatch marker must not wake the manager — only the worker's
    # terminal event does.
    assert kwargs["initiator"] == "queue_item"

    # The fire still counts — the automation ran, it just ran elsewhere.
    updated = store.get_project(pid)["triggers"][0]
    assert updated["trigger_count"] == 1
    assert updated["last_triggered"] is not None


@pytest.mark.asyncio
async def test_the_worker_receives_the_trigger_context_line():
    """The "[Triggered by schedule 'Nightly' …]" preamble is what tells the
    runner why it woke up; the worker needs it as much as the manager did."""
    store, pid = _project_with_trigger(agent="codex")
    mgr, sub_mgr = _manager()
    tm = TriggerManager(store, mgr)

    await tm._fire_trigger(pid, "trg_aaa")

    dispatched = sub_mgr.send.await_args.args[2]
    assert "Triggered by schedule 'Nightly'" in dispatched


@pytest.mark.asyncio
async def test_unassigned_trigger_still_starts_the_management_agent():
    store, pid = _project_with_trigger(agent=None)
    mgr, sub_mgr = _manager()
    tm = TriggerManager(store, mgr)

    await tm._fire_trigger(pid, "trg_aaa")

    sub_mgr.send.assert_not_awaited()
    mgr.start_agent.assert_awaited_once()
    kwargs = mgr.start_agent.await_args.kwargs
    assert kwargs["trigger_source"] == "schedule"
    assert kwargs["trigger_name"] == "Nightly"
    assert "Do the thing" in kwargs["initial_message"]


@pytest.mark.asyncio
async def test_a_stale_slug_surfaces_as_an_agent_status_error():
    """An agent uninstalled since the automation was saved must not fail
    silently — this path was log-only before triggers grew error reporting,
    and a dispatch that never happened is exactly that class of failure."""
    store, pid = _project_with_trigger(agent="ghost")
    mgr, sub_mgr = _manager(
        send_result="Error: agent 'ghost' not running for project 'p'",
    )
    ws = MagicMock()
    tm = TriggerManager(store, mgr, ws_manager=ws)

    await tm._fire_trigger(pid, "trg_aaa")

    statuses = [
        c.args[1] for c in ws.broadcast.call_args_list
        if c.args[1].get("type") == "agent.status"
    ]
    assert statuses, "a failed dispatch must reach the UI"
    assert statuses[0]["status"] == "error"
    assert statuses[0]["source"] == "trigger"

    fired = [
        c.args[1] for c in ws.broadcast.call_args_list
        if c.args[1].get("type") == "trigger.fired"
    ]
    assert not fired, "a dispatch that failed must not report as fired"
