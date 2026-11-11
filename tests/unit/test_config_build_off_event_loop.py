# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The agent config build never runs on the event loop, and the awaits it
adds cannot double-start a session.

Spec 171 A2: ``_build_agent_config_from_project`` can block for seconds (a
cold agent check probes every CLI by subprocess; each card key is a Keychain
read). Spec 101 gave the inject paths ``_abuild_agent_config_from_project``;
the start route, the import scan, triggers, the pinned consolidation pass,
the queue's worker dispatch, Workbench routing and the Memory session still
built it on the loop. Each now builds in a worker thread.

Every new await is a window in which a second request can start the same
session. ``start_agent`` itself already awaited before storing its handle
(the folder grant, connector reflection), so two starts of one session could
both register — two loops on one session. It now holds a reservation from
its guard to the handle store; inject waits for an in-flight start and then
takes the live-handle path; a trigger fires once at a time.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from agent_os.daemon_v2.models import AgentConfig, make_session_key


def _assert_off_loop(seen: list[int]) -> None:
    assert seen, "the config build did not run"
    assert threading.get_ident() not in seen, "the config build ran on the event loop"


def _manager(platform_provider=None):
    from agent_os.daemon_v2.agent_manager import AgentManager
    from tests.card_doubles import FakeCardStore

    sub_agent_mgr = MagicMock()
    sub_agent_mgr.list_active = MagicMock(return_value=[])
    sub_agent_mgr.stop = AsyncMock()
    sub_agent_mgr.stop_all = AsyncMock()
    return AgentManager(
        project_store=MagicMock(),
        ws_manager=MagicMock(),
        sub_agent_manager=sub_agent_mgr,
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
        platform_provider=platform_provider,
        settings_store=FakeCardStore.with_default(
            provider="anthropic", model="claude-x", key="sk-test"),
    )


def _recording_build(seen: list[int], config=None, delay: float = 0.0):
    def build(_project_id):
        seen.append(threading.get_ident())
        if delay:
            time.sleep(delay)
        return config if config is not None else AgentConfig(
            workspace="/tmp/ws", model="m", api_key="k")
    return build


def _running_handle():
    task = MagicMock()
    task.done.return_value = False
    session = MagicMock()
    session.is_stopped.return_value = False
    session._paused_for_approval = False
    return SimpleNamespace(task=task, session=session, loop=MagicMock(),
                           interceptor=None, last_activity=0.0)


# --------------------------------------------------------------------------
# start_agent: one start per session
# --------------------------------------------------------------------------

def _slow_grant_provider():
    from agent_os.platform.types import PermissionResult

    provider = MagicMock()

    def grant(path, mode, *, scope):
        time.sleep(0.1)
        return PermissionResult(success=True, path=path)

    provider.grant_folder_access.side_effect = grant
    provider.get_capabilities.return_value = MagicMock(
        isolation_method="seatbelt", setup_complete=True)
    return provider


@pytest.mark.asyncio
async def test_two_starts_of_one_session_start_one_loop(tmp_path):
    """The folder grant awaits before the handle is stored. Before the
    reservation both starts got past the guard and both ran a loop."""
    mgr = _manager(_slow_grant_provider())
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace)
    config = AgentConfig(workspace=workspace, model="gpt-4", api_key="sk-test")
    runs: list[int] = []

    def make_loop(*_a, **_k):
        loop = MagicMock()

        async def run():
            runs.append(1)
            await asyncio.sleep(0.5)

        loop.run = run
        return loop

    with patch("agent_os.daemon_v2.agent_manager.LLMProvider"), \
         patch("agent_os.daemon_v2.agent_manager.ToolRegistry") as MockReg, \
         patch("agent_os.daemon_v2.agent_manager.PromptBuilder"), \
         patch("agent_os.daemon_v2.agent_manager.Session") as MockSession, \
         patch("agent_os.daemon_v2.agent_manager.ContextManager"), \
         patch("agent_os.daemon_v2.agent_manager.AgentLoop", side_effect=make_loop), \
         patch("agent_os.daemon_v2.agent_manager.AutonomyInterceptor"):
        MockSession.new.return_value = MagicMock()
        MockReg.return_value.tool_names.return_value = ["read"]
        results = await asyncio.gather(
            mgr.start_agent("proj_1", config, session_id="sess_1"),
            mgr.start_agent("proj_1", config, session_id="sess_1"),
            return_exceptions=True)
        await asyncio.sleep(0.05)

    assert results.count(None) == 1, results
    refused = [r for r in results if isinstance(r, ValueError)]
    assert len(refused) == 1 and "already running" in str(refused[0])
    assert len(runs) == 1, "two loops ran on one session"
    assert mgr._handles.get(make_session_key("proj_1", "sess_1")) is not None
    assert not mgr._start_reservations(), "the reservation outlived the start"
    for task in [h.task for h in mgr._handles.values()]:
        task.cancel()


@pytest.mark.asyncio
async def test_a_failed_start_releases_the_session():
    mgr = _manager()

    async def boom(*_a, **_k):
        raise RuntimeError("setup failed")

    mgr._start_agent_reserved = boom
    config = AgentConfig(workspace="/tmp/ws", model="m", api_key="k")
    with pytest.raises(RuntimeError):
        await mgr.start_agent("proj_1", config, session_id="sess_1")
    assert not mgr._start_reservations()
    assert mgr._start_rejection("proj_1", "sess_1") is None


@pytest.mark.asyncio
async def test_two_sends_to_one_new_session_start_it_once():
    """Two chat sends to a session with no handle: both enter the auto-start
    branch and await the config build. The second must not start a second
    loop; it waits for the first start and is queued into its live loop."""
    mgr = _manager()
    seen: list[int] = []
    mgr._build_agent_config_from_project = _recording_build(seen, delay=0.05)
    mgr._hydrate_session = AsyncMock(return_value=None)
    mgr._preflight_provider = MagicMock()
    mgr._new_session_object = MagicMock(
        side_effect=lambda _cfg, sid, **_k: MagicMock(session_uuid=sid))
    mgr._wire_session_observers = MagicMock()
    sk = make_session_key("proj_1", "sess_1")
    starts: list[str] = []
    release = asyncio.Event()

    async def fake_body(project_id, config, *, session_id, sk, **_kw):
        starts.append(session_id)
        await release.wait()
        mgr._handles[sk] = _running_handle()

    mgr._start_agent_reserved = fake_body
    with patch("agent_os.daemon_v2.agent_manager.persist_user_row") as persisted:
        first = asyncio.create_task(
            mgr.inject_message("proj_1", "one", session_id="sess_1"))
        second = asyncio.create_task(
            mgr.inject_message("proj_1", "two", session_id="sess_1"))
        while not starts:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.1)   # the second is past its build and waiting
        release.set()
        results = await asyncio.gather(first, second)

    _assert_off_loop(seen)
    assert starts == ["sess_1"], "the session was started twice"
    assert sorted(results) == ["queued_same_session", "started"]
    # Whichever send's build finished first started the session; the other
    # was queued into its loop. Each message lands exactly once.
    persisted.assert_called_once()
    started_with = persisted.call_args.args[1]
    queued = "two" if started_with == "one" else "one"
    mgr._handles[sk].session.queue_message.assert_called_once_with(queued, nonce=None)


@pytest.mark.asyncio
async def test_start_route_builds_off_the_loop_and_refuses_a_second_start(monkeypatch):
    from agent_os.api.routes import agents_v2

    mgr = _manager()
    seen: list[int] = []
    mgr._build_agent_config_from_project = _recording_build(seen, delay=0.05)
    release = asyncio.Event()

    async def fake_body(project_id, config, *, session_id, sk, **_kw):
        await release.wait()
        mgr._handles[sk] = _running_handle()

    mgr._start_agent_reserved = fake_body
    project_store = MagicMock()
    project_store.get_project.return_value = {"project_id": "proj_1"}
    monkeypatch.setattr(agents_v2, "_project_store", project_store)
    monkeypatch.setattr(agents_v2, "_agent_manager", mgr)
    req = agents_v2.StartAgentRequest(project_id="proj_1", session_id="sess_1")

    first = asyncio.create_task(agents_v2.start_agent(req))
    second = asyncio.create_task(agents_v2.start_agent(req))
    await asyncio.sleep(0.2)
    release.set()
    results = await asyncio.gather(first, second, return_exceptions=True)

    _assert_off_loop(seen)
    assert {"status": "started"} in results
    refused = [r for r in results if isinstance(r, HTTPException)]
    assert len(refused) == 1 and refused[0].status_code == 400


@pytest.mark.asyncio
async def test_manager_scan_builds_off_the_loop(monkeypatch):
    from agent_os.api.routes import agents_v2

    mgr = _manager()
    seen: list[int] = []
    mgr._build_agent_config_from_project = _recording_build(seen)
    mgr.validate_provider_config = MagicMock()
    mgr.new_session = AsyncMock(return_value={"session_id": "sess_scan"})
    mgr.start_agent = AsyncMock()
    monkeypatch.setattr(agents_v2, "_agent_manager", mgr)

    out = await agents_v2._start_manager_scan("proj_1", "skeleton", flavour="guess")
    assert out["session_id"] == "sess_scan"
    _assert_off_loop(seen)


# --------------------------------------------------------------------------
# Triggers
# --------------------------------------------------------------------------

def _trigger_store():
    import tempfile

    from agent_os.daemon_v2.project_store import ProjectStore

    tmpdir = tempfile.mkdtemp()
    store = ProjectStore(data_dir=tmpdir)
    pid = store.create_project({"name": "T", "workspace": tmpdir, "model": "gpt-4"})
    store.update_project(pid, {"triggers": [{
        "id": "trg_1", "name": "Watch", "type": "file_watch", "enabled": True,
        "task": "Summarise the change", "trigger_count": 0,
    }]})
    return store, pid


@pytest.mark.asyncio
async def test_trigger_builds_off_the_loop():
    from agent_os.daemon_v2.trigger_manager import TriggerManager

    store, pid = _trigger_store()
    seen: list[int] = []
    am = MagicMock()
    am.is_running = MagicMock(return_value=False)
    am._build_agent_config_from_project = MagicMock(side_effect=_recording_build(seen))
    am.start_agent = AsyncMock()
    tm = TriggerManager(store, am)
    await tm._fire_trigger(pid, "trg_1", changed_files=["a.txt"])
    am.start_agent.assert_awaited_once()
    _assert_off_loop(seen)


@pytest.mark.asyncio
async def test_two_fires_of_one_trigger_start_one_run():
    """Two file-watch flushes are separate tasks. The busy gate only sees a
    run whose handle is stored, so both used to pass it while the first was
    still building its config."""
    from agent_os.daemon_v2.trigger_manager import TriggerManager

    store, pid = _trigger_store()
    seen: list[int] = []
    am = MagicMock()
    am.is_running = MagicMock(return_value=False)
    am._build_agent_config_from_project = MagicMock(
        side_effect=_recording_build(seen, delay=0.1))
    am.start_agent = AsyncMock()
    tm = TriggerManager(store, am)

    await asyncio.gather(
        tm._fire_trigger(pid, "trg_1", changed_files=["a.txt"]),
        tm._fire_trigger(pid, "trg_1", changed_files=["b.txt"]))

    assert am.start_agent.await_count == 1, "the automation started twice"
    assert store.get_project(pid)["triggers"][0]["trigger_count"] == 1
    # The next fire after it is a new fire.
    await tm._fire_trigger(pid, "trg_1", changed_files=["c.txt"])
    assert am.start_agent.await_count == 2


# --------------------------------------------------------------------------
# Pinned consolidation
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_consolidation_pass_builds_off_the_loop_and_reads_the_live_session(tmp_path):
    from agent_os.daemon_v2.pinned_consolidation import PinnedConsolidationCoordinator

    seen: list[int] = []
    disk_copy, live_copy = MagicMock(name="disk"), MagicMock(name="live")
    am = MagicMock()
    am.manager_ready = MagicMock(return_value=True)
    state = {"live": None}
    am.get_session = MagicMock(side_effect=lambda *_a, **_k: state["live"])
    am._load_session_from_disk = MagicMock(return_value=disk_copy)

    def build(pid):
        seen.append(threading.get_ident())
        state["live"] = live_copy   # a user message hydrated it meanwhile
        return AgentConfig(workspace=str(tmp_path), model="m", api_key="k")

    am._build_agent_config_from_project = MagicMock(side_effect=build)
    am._build_llm_providers = MagicMock(return_value=(MagicMock(), [], MagicMock(), None))
    coord = PinnedConsolidationCoordinator(am, MagicMock())
    with patch("agent_os.agent.workspace_files.run_session_end_routine",
               new_callable=AsyncMock, return_value="ok") as routine:
        await coord._run_pass(("proj_1", "sess_1"), "test")

    _assert_off_loop(seen)
    assert routine.await_args.kwargs["session"] is live_copy


# --------------------------------------------------------------------------
# Queue: dispatch to a worker
# --------------------------------------------------------------------------

def _queue_dispatcher(tmp_path, mgr):
    from agent_os.queue.dispatcher import QueueDispatcher
    from agent_os.queue.store import QueueStore

    store = QueueStore(tmp_path / "queue.json")
    item = store.add_item("write hello.txt", agent="codex")
    d = QueueDispatcher("proj_1", store, mgr, workspace=str(tmp_path))
    d._send_to_worker_and_hold = AsyncMock()
    d._persist_queue_user_row = MagicMock()
    return d, store, item


@pytest.mark.asyncio
async def test_worker_dispatch_builds_off_the_loop(tmp_path):
    seen: list[int] = []
    mgr = MagicMock()
    mgr._handles = {}
    mgr._build_agent_config_from_project = MagicMock(side_effect=_recording_build(seen))
    d, _store, item = _queue_dispatcher(tmp_path, mgr)

    await d._dispatch_to_worker(item, "sess_q", "codex",
                                wrapped_content="row", worker_content="task")

    _assert_off_loop(seen)
    assert d._persist_queue_user_row.call_args.args[:2] == ("sess_q", "row")
    assert d._persist_queue_user_row.call_args.kwargs["config"] is not None
    d._send_to_worker_and_hold.assert_awaited_once()


@pytest.mark.asyncio
async def test_worker_dispatch_does_not_write_into_a_session_started_meanwhile(tmp_path):
    mgr = MagicMock()
    mgr._handles = {}

    def build(pid):
        mgr._handles[make_session_key("proj_1", "sess_q")] = _running_handle()
        return AgentConfig(workspace=str(tmp_path), model="m", api_key="k")

    mgr._build_agent_config_from_project = MagicMock(side_effect=build)
    d, store, item = _queue_dispatcher(tmp_path, mgr)
    from agent_os.queue.models import AttemptRecord, ItemState
    store.set_item_state(item.id, ItemState.RUNNING)
    store.append_attempt(item.id, AttemptRecord(session_id="sess_q"))

    await d._dispatch_to_worker(item, "sess_q", "codex",
                                wrapped_content="row", worker_content="task")

    d._persist_queue_user_row.assert_not_called()
    d._send_to_worker_and_hold.assert_not_awaited()
    assert store.load().items[0].state == ItemState.QUEUED   # reclaimed


# --------------------------------------------------------------------------
# Workbench routing
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_workbench_route_provider_builds_off_the_loop(monkeypatch):
    from agent_os.api.routes import workbench

    seen: list[int] = []
    am = MagicMock()
    am.manager_ready = MagicMock(return_value=True)
    am._build_agent_config_from_project = MagicMock(side_effect=_recording_build(seen))
    utility = object()
    am._build_llm_providers = MagicMock(return_value=(None, [], utility, None))
    monkeypatch.setattr(workbench, "_agent_manager", am)
    assert await workbench._route_provider("proj_scratch") is utility
    _assert_off_loop(seen)


# --------------------------------------------------------------------------
# The Memory session
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_memory_session_mint_builds_off_the_loop(tmp_path):
    mgr = _manager()
    mgr._project_store.get_project.return_value = {"workspace": str(tmp_path)}
    mgr.list_sessions = MagicMock(return_value=[])
    seen: list[int] = []
    mgr._build_agent_config_from_project = _recording_build(seen)

    session = await mgr._amemory_session("proj_1")

    _assert_off_loop(seen)
    assert session is not None and session.origin == "memory"


@pytest.mark.asyncio
async def test_memory_session_started_meanwhile_wins_over_a_fresh_one(tmp_path):
    from agent_os.agent.session import Session

    mgr = _manager()
    mgr._project_store.get_project.return_value = {"workspace": str(tmp_path)}
    mgr.list_sessions = MagicMock(return_value=[])
    live = Session.new("mem_live", str(tmp_path), session_id="mem_live", origin="memory")

    def build(pid):
        mgr._handles[make_session_key("proj_1", "mem_live")] = SimpleNamespace(
            session=live, task=None)
        return AgentConfig(workspace=str(tmp_path), model="m", api_key="k")

    mgr._build_agent_config_from_project = build
    assert await mgr._amemory_session("proj_1") is live
