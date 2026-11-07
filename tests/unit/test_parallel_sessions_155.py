# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 155 §5.2/§5.5/§3.7: sessions of one project run in parallel.

A second session's message just runs (no 202, no pending FIFO, no owed wake),
a worker terminal for an idle session wakes it now, a trigger fires while
another session runs, and the places that assumed one running session per
project act on the session they name: run-status, project deletion, the
queue-item delete gate, session-less input, queue pause and retry, the
ledger's spend broadcast and the browser's pages.
"""

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.agent.session import Session
from agent_os.api.routes import agents_v2
from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle
from agent_os.daemon_v2.models import AgentConfig, make_session_key

PID = "proj155"
A = "sess-155-aaaa"
B = "sess-155-bbbb"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _config(workspace: str) -> AgentConfig:
    return AgentConfig(workspace=workspace, model="stub-model", api_key="k",
                       provider="stub-provider", sdk="openai", project_name="proj")


def _manager(workspace: str) -> AgentManager:
    sub = MagicMock()
    sub.list_active = MagicMock(return_value=[])
    sub.stop_all = AsyncMock()
    sub.running_session_ids = MagicMock(return_value=[])
    mgr = AgentManager(project_store=MagicMock(), ws_manager=MagicMock(),
                       sub_agent_manager=sub, activity_translator=MagicMock(),
                       process_manager=MagicMock())
    mgr._ws.broadcast = MagicMock()
    mgr._project_store.get_project.return_value = {
        "project_id": PID, "name": "proj", "workspace": workspace,
    }
    mgr._platform_provider = None
    mgr._build_agent_config_from_project = lambda pid: _config(workspace)
    return mgr


def _plant(mgr, sid, workspace, *, running=True, paused=False, waiting=False,
           trigger_name=None):
    """A live handle for ``sid``: a running loop task, a session paused on an
    approval, or a session waiting on its workers (idle-poll alive)."""
    session = Session.new(sid, workspace, session_id=sid)
    session._paused_for_approval = paused
    task = None
    stopped = asyncio.Event()
    if running:
        task = asyncio.get_running_loop().create_task(stopped.wait())
    loop = MagicMock()

    async def _terminate():          # the loop ends its run on Stop
        session.stop()
        stopped.set()

    loop.terminate = AsyncMock(side_effect=_terminate)
    loop.yield_memory_record = MagicMock(return_value=False)
    handle = ProjectHandle(session=session, loop=loop, provider=None, registry=None,
                           context_manager=None, interceptor=None, task=task,
                           trigger_source="schedule" if trigger_name else None,
                           trigger_name=trigger_name)
    mgr._handles[make_session_key(PID, sid)] = handle
    if waiting:
        mgr._idle_poll_tasks[make_session_key(PID, sid)] = (
            asyncio.get_running_loop().create_task(asyncio.sleep(3600)))
    return handle


def _cleanup(mgr):
    for handle in mgr._handles.values():
        if handle.task is not None and not handle.task.done():
            handle.task.cancel()
    for task in mgr._idle_poll_tasks.values():
        task.cancel()


class _CapturingStart:
    def __init__(self):
        self.calls: list[dict] = []

    async def __call__(self, project_id, config, **kwargs):
        self.calls.append(kwargs)


def _rows(workspace, sid):
    path = os.path.join(workspace, "orbital", "sessions", f"{sid}.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# 5. A second session runs; a wake for an idle session runs now; triggers fire
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_agent_for_a_second_session_passes_while_another_runs(tmp_path):
    ws = str(tmp_path)
    mgr = _manager(ws)
    _plant(mgr, A, ws)

    class _PastTheGuards(Exception):
        pass

    # Cut start_agent off just after its entry guards: the provider build is
    # the first thing past them.
    mgr._build_llm_providers = MagicMock(side_effect=_PastTheGuards())
    session = Session.new(B, ws, session_id=B)
    try:
        with pytest.raises(_PastTheGuards):
            await mgr.start_agent(PID, _config(ws), initial_message="hello",
                                  session_id=B, session=session)
    finally:
        _cleanup(mgr)


@pytest.mark.parametrize("state", ["running", "paused", "waiting"])
@pytest.mark.asyncio
async def test_a_message_to_a_second_session_writes_and_starts(tmp_path, state):
    """Whatever A is doing — streaming, paused on an approval, waiting on its
    workers — a message to B is written and B starts: no 202, no queue."""
    ws = str(tmp_path)
    mgr = _manager(ws)
    _plant(mgr, A, ws, running=state == "running", paused=state == "paused",
           waiting=state == "waiting")
    start = _CapturingStart()
    mgr.start_agent = start
    try:
        result = await mgr.inject_message(PID, "second session's task",
                                          nonce="n-b", session_id=B)
    finally:
        _cleanup(mgr)
    assert result == "started"
    assert [c["session_id"] for c in start.calls] == [B]
    assert [r["content"] for r in _rows(ws, B) if r.get("role") == "user"] == [
        "second session's task"]


@pytest.mark.asyncio
async def test_a_worker_terminal_for_an_idle_session_wakes_it_now(tmp_path):
    ws = str(tmp_path)
    mgr = _manager(ws)
    _plant(mgr, A, ws)                          # A runs
    _plant(mgr, B, ws, running=False)           # B idle, handle live
    mgr._start_loop = AsyncMock()
    try:
        status = await mgr.inject_system_message(
            PID, "[Sub-agent] codex completed", session_id=B,
            meta={"event": "sub_agent_terminal", "handle": "codex"})
    finally:
        _cleanup(mgr)
    assert status == "delivered"
    mgr._start_loop.assert_awaited_once_with(PID, session_id=B)


@pytest.mark.asyncio
async def test_a_worker_terminal_for_a_disk_only_session_wakes_it_now(tmp_path):
    ws = str(tmp_path)
    mgr = _manager(ws)
    _plant(mgr, A, ws)
    on_disk = Session.new(B, ws, session_id=B)
    mgr._hydrate_session = AsyncMock(return_value=on_disk)
    mgr._abuild_agent_config_from_project = AsyncMock(return_value=_config(ws))
    start = _CapturingStart()
    mgr.start_agent = start
    try:
        status = await mgr.inject_system_message(
            PID, "job ended", session_id=B, meta={"event": "job_terminal"})
    finally:
        _cleanup(mgr)
    assert status == "delivered"
    assert [c["session_id"] for c in start.calls] == [B]


@pytest.mark.asyncio
async def test_a_trigger_fires_while_another_session_runs(tmp_path):
    from agent_os.daemon_v2.trigger_manager import TriggerManager

    ws = str(tmp_path)
    mgr = _manager(ws)
    _plant(mgr, A, ws)                          # the user's chat runs
    trigger = {"id": "trg1", "name": "daily digest", "type": "schedule",
               "enabled": True, "task": "write the digest", "schedule": {}}
    mgr._project_store.get_project.return_value = {
        "project_id": PID, "name": "proj", "workspace": ws, "triggers": [trigger],
    }
    mgr.start_agent = AsyncMock()
    tm = TriggerManager(mgr._project_store, mgr, ws_manager=MagicMock())
    try:
        await tm._fire_trigger(PID, "trg1")
        mgr.start_agent.assert_awaited_once()
        # Its own previous run still going holds it (no stacked runs).
        _plant(mgr, "sess-trigger-run", ws, trigger_name="daily digest")
        mgr.start_agent.reset_mock()
        await tm._fire_trigger(PID, "trg1")
        mgr.start_agent.assert_not_awaited()
    finally:
        _cleanup(mgr)


# ---------------------------------------------------------------------------
# Status: per session, and the project aggregate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_status_is_per_session_and_the_project_aggregates(tmp_path):
    ws = str(tmp_path)
    mgr = _manager(ws)
    _plant(mgr, A, ws, running=False, paused=True)
    _plant(mgr, B, ws, running=False, waiting=True)
    try:
        assert mgr.get_run_status(PID, session_id=A) == "pending_approval"
        assert mgr.get_run_status(PID, session_id=B) == "waiting"
        assert mgr.get_run_status(PID) == "waiting"
        _plant(mgr, "sess-155-cccc", ws)
        assert mgr.get_run_status(PID) == "running"
        assert set(mgr.live_session_ids(PID)) == {A, B, "sess-155-cccc"}
    finally:
        _cleanup(mgr)
    mgr._handles.clear()
    mgr._idle_poll_tasks.clear()
    assert mgr.get_run_status(PID) == "idle"


def _client(mgr, sub=None):
    saved = {k: getattr(agents_v2, k) for k in
             ("_project_store", "_agent_manager", "_ws_manager", "_sub_agent_manager")}
    sub = sub or MagicMock(has_running_sub_agents=MagicMock(return_value=False))
    agents_v2.configure(project_store=mgr._project_store, agent_manager=mgr,
                        ws_manager=MagicMock(), sub_agent_manager=sub)
    app = FastAPI()
    app.include_router(agents_v2.router)
    return TestClient(app), saved


def _restore(saved):
    for k, v in saved.items():
        setattr(agents_v2, k, v)


@pytest.mark.asyncio
async def test_run_status_route_answers_for_the_named_session(tmp_path):
    """B3: run-status?session_id=B used to return the holder's state."""
    ws = str(tmp_path)
    mgr = _manager(ws)
    _plant(mgr, A, ws)
    _plant(mgr, B, ws, running=False)
    client, saved = _client(mgr)
    try:
        body_b = client.get(f"/api/v2/agents/{PID}/run-status", params={"session_id": B}).json()
        body_a = client.get(f"/api/v2/agents/{PID}/run-status", params={"session_id": A}).json()
        body = client.get(f"/api/v2/agents/{PID}/run-status").json()
    finally:
        _restore(saved)
        _cleanup(mgr)
    assert body_b["status"] == "idle"
    assert body_a["status"] == "running"
    assert body["status"] == "running"
    assert body["session_statuses"] == {A: "running"}


# ---------------------------------------------------------------------------
# Stop in B stops only B
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_in_one_session_leaves_the_other_and_its_pages(tmp_path):
    ws = str(tmp_path)
    mgr = _manager(ws)
    handle_a = _plant(mgr, A, ws)
    _plant(mgr, B, ws)
    bm = MagicMock()
    bm.close_session_pages = AsyncMock()
    bm.close_project_pages = AsyncMock()
    mgr._browser_manager = bm
    platform = MagicMock()
    platform.stop_process = AsyncMock()
    mgr._platform_provider = platform
    try:
        await mgr.stop_agent(PID, session_id=B)
        assert make_session_key(PID, B) not in mgr._handles
        assert not handle_a.task.done()
        bm.close_session_pages.assert_awaited_once_with(PID, B)
        bm.close_project_pages.assert_not_called()
        # A is still live: the project's sandbox process / proxy stays.
        platform.stop_process.assert_not_awaited()
        await mgr.stop_agent(PID, session_id=A)
        platform.stop_process.assert_awaited_once_with(PID)
    finally:
        _cleanup(mgr)


# ---------------------------------------------------------------------------
# 9. Deletion: the project stops every session; a queue item's own status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_all_sessions_stops_every_live_session(tmp_path):
    ws = str(tmp_path)
    mgr = _manager(ws)
    _plant(mgr, A, ws)
    _plant(mgr, B, ws, running=False, paused=True)
    _plant(mgr, "sess-idle", ws, running=False)
    try:
        stopped = await mgr.stop_all_sessions(PID)
    finally:
        _cleanup(mgr)
    assert set(stopped) == {A, B}
    assert make_session_key(PID, A) not in mgr._handles
    assert make_session_key(PID, B) not in mgr._handles


def test_project_delete_routes_stop_every_session():
    import inspect
    src = inspect.getsource(agents_v2.delete_project) + inspect.getsource(
        agents_v2.bulk_delete_projects)
    assert src.count("stop_all_sessions(") == 2
    assert "current_holder_session_id" not in src


@pytest.mark.asyncio
async def test_queue_item_delete_is_refused_while_its_attempt_runs_beside_a_chat(tmp_path):
    from agent_os.queue.models import AttemptRecord, ItemState

    ws = str(tmp_path)
    mgr = _manager(ws)
    store = mgr.get_queue_store(PID, workspace=ws)
    item = store.add_item("research the market")
    store.append_attempt(item.id, AttemptRecord(session_id=B))
    store.set_item_state(item.id, ItemState.RUNNING)
    _plant(mgr, A, ws)            # a chat session (the first live handle)
    _plant(mgr, B, ws)            # the item's attempt, also running
    client, saved = _client(mgr)
    try:
        resp = client.delete(f"/api/v2/projects/{PID}/queue/items/{item.id}")
    finally:
        _restore(saved)
        _cleanup(mgr)
    assert resp.status_code == 409
    assert any(it.id == item.id for it in store.load().items)


# ---------------------------------------------------------------------------
# 10. Session-less input goes to the chat session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_session_less_inject_lands_in_the_chat_session_while_the_queue_drains(tmp_path):
    from agent_os.queue.models import QueueRunState

    ws = str(tmp_path)
    mgr = _manager(ws)
    store = mgr.get_queue_store(PID, workspace=ws)
    store.set_queue_state(QueueRunState.RUNNING)
    mgr._dispatchers[PID] = MagicMock(_store=store)
    _plant(mgr, "sess-queue-attempt", ws)        # the queue's attempt runs
    mgr._ensure_chat_session = MagicMock(return_value="sess-chat")
    try:
        assert mgr._sid_inject(PID, None) == "sess-chat"
    finally:
        _cleanup(mgr)


# ---------------------------------------------------------------------------
# 11. Same-session queued messages still list and cancel (↑ recall)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_same_session_queued_messages_list_and_cancel(tmp_path):
    ws = str(tmp_path)
    mgr = _manager(ws)
    handle = _plant(mgr, A, ws)
    try:
        assert await mgr.inject_message(PID, "follow-up", nonce="n1", session_id=A) == (
            "queued_same_session")
        assert mgr.list_pending(PID) == [{
            "session_id": A, "nonce": "n1", "content": "follow-up",
            "position": 1, "kind": "same"}]
        assert mgr.cancel_pending_inject(PID, A, nonce="n1") == {
            "status": "cancelled", "removed": True}
        assert mgr.list_pending(PID) == []
        assert handle.session.list_queued() == []
    finally:
        _cleanup(mgr)


def test_the_cross_session_queue_is_gone():
    for name in ("enqueue_pending_inject", "_maybe_dispatch_pending", "_release_slot",
                 "purge_pending", "_maybe_owe_wake", "_deliver_owed_wake",
                 "has_pending_inject", "_pending_session_entries", "switch_session"):
        assert not hasattr(AgentManager, name), name


# ---------------------------------------------------------------------------
# Ledger (B10): the older loop keeps broadcasting when the newer one ends
# ---------------------------------------------------------------------------


def test_ledger_hook_ownership_returns_to_the_older_loop(tmp_path):
    from agent_os.budget import ledger

    ws = str(tmp_path)
    seen: list[str] = []

    class _Loop:
        def __init__(self, name):
            self.name = name

        def hook(self, project_dir, event):
            seen.append(self.name)

    older, newer = _Loop("older"), _Loop("newer")
    ledger.register_append_hook(ws, older.hook)
    ledger.register_append_hook(ws, newer.hook)
    ledger._fire_append_hook(ws, object())
    assert seen == ["newer"]                     # one owner, no double notify
    ledger.unregister_append_hook(ws, newer.hook)  # bound methods: equal, not identical
    ledger._fire_append_hook(ws, object())
    assert seen == ["newer", "older"]
    ledger.register_append_hook(ws, older.hook)  # re-arm: still one entry
    ledger.unregister_append_hook(ws, older.hook)
    ledger._fire_append_hook(ws, object())
    assert seen == ["newer", "older"]
    assert ws not in ledger._append_hooks
