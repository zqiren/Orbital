# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for SubAgentManager.dispatch_fanout: cap validation (including the
atomic-under-lock TOCTOU close), successful dispatch (adapter/transcript
registration + fanout.started broadcast), the worker_deps_factory /
fanout_registry unwired-gates, and the full observer-hook wiring (absorb
suppresses per-worker injection, join fires exactly once).

Also covers the session-list filtering requirement (spec 009 §3a): worker
sessions carry a `session_kind: worker` meta row and must not appear in
AgentManager.list_sessions.

Spec 009 (subagent fanout), Task 2 brief.
"""

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.providers.types import StreamChunk, TokenUsage
from agent_os.agent.tools.base import ToolResult
from agent_os.daemon_v2.fanout import FanoutRegistry
from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver
from agent_os.daemon_v2.native_worker import WorkerDeps
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

SID = "s1"
PID = "p1"


# ---------------------------------------------------------------------------
# Stubs (mirror tests/unit/test_native_worker.py's style: real Session/
# ContextManager/AgentLoop, only the provider and tool registry are stubbed)
# ---------------------------------------------------------------------------

class _StubProvider:
    def __init__(self, text: str):
        self.provider = "stub"
        self.model = "stub-model"
        self.sdk = "stub-sdk"
        self._text = text

    async def stream(self, messages, tools=None):
        yield StreamChunk(text=self._text)
        yield StreamChunk(is_final=True, usage=TokenUsage(input_tokens=10, output_tokens=5))


class _RaisingProvider:
    def __init__(self):
        self.provider = "stub"
        self.model = "stub-model"
        self.sdk = "stub-sdk"

    async def stream(self, messages, tools=None):
        raise RuntimeError("boom: provider unreachable")
        yield  # pragma: no cover — makes this an async generator


class _StubToolRegistry:
    def schemas(self) -> list[dict]:
        return []

    def is_async(self, name: str) -> bool:
        return False

    def execute(self, name: str, arguments: dict) -> ToolResult:
        return ToolResult(content="ok")

    async def execute_async(self, name: str, arguments: dict) -> ToolResult:
        return ToolResult(content="ok")

    def tool_names(self) -> list[str]:
        return []

    def reset_run_state(self) -> None:
        pass


def _make_registry_factory():
    def _make_tool_registry(allowed_paths, forbidden_paths):
        return _StubToolRegistry()
    return _make_tool_registry


class _StubProjectStore:
    def __init__(self, workspace: str):
        self._workspace = workspace

    def get_project(self, project_id: str) -> dict:
        return {"workspace": self._workspace}


class _Recorder:
    """Doubles as both the ``agent_manager`` (inject_system_message) and the
    ``ws_manager`` (broadcast) collaborators — simpler to assert against than
    threading two separate MagicMocks through call_args_list."""

    def __init__(self):
        self.injected: list[tuple] = []
        self.broadcasts: list[tuple] = []

    async def inject_system_message(self, project_id, content, *, session_id=None):
        self.injected.append((project_id, content, session_id))
        return "delivered"

    def broadcast(self, project_id, payload):
        self.broadcasts.append((project_id, payload))


def _make_manager(tmp_path, *, provider_factory=None):
    """A SubAgentManager wired for fanout, with the worker registry factory
    stubbed. ``provider_factory()`` builds the (shared) provider each fanout
    batch's WorkerDeps carries; defaults to an always-succeeding stub."""
    pm = MagicMock()
    pm.start = AsyncMock()
    pm.stop = AsyncMock()
    rec = _Recorder()
    project_store = _StubProjectStore(str(tmp_path))
    mgr = SubAgentManager(
        process_manager=pm, project_store=project_store, ws_manager=rec,
    )
    if provider_factory is None:
        provider_factory = lambda: _StubProvider("done: 42")

    def _worker_deps_factory(project_id, session_id):
        return WorkerDeps(
            provider=provider_factory(),
            workspace=str(tmp_path),
            project_id=project_id,
            parent_session_id=session_id,
            make_tool_registry=_make_registry_factory(),
        )

    mgr._worker_deps_factory = _worker_deps_factory
    mgr._fanout_registry = FanoutRegistry(
        inject=rec.inject_system_message, broadcast=rec.broadcast,
        stop_worker=mgr.stop,
    )
    return mgr, rec


def _two_tasks():
    return [
        {"brief": "investigate A", "label": "task-a"},
        {"brief": "investigate B", "label": "task-b"},
    ]


async def _wait_until(cond, *, timeout: float = 2.0, interval: float = 0.01):
    loop = asyncio.get_event_loop()
    start = loop.time()
    while not cond():
        if loop.time() - start > timeout:
            raise AssertionError(f"condition not met within {timeout}s")
        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Cap / structural validation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_single_task_rejected(tmp_path):
    mgr, _ = _make_manager(tmp_path)
    result = await mgr.dispatch_fanout(
        PID, [{"brief": "x", "label": "a"}], session_id=SID)
    assert result.startswith("Error")


@pytest.mark.asyncio
async def test_six_tasks_rejected(tmp_path):
    mgr, _ = _make_manager(tmp_path)
    tasks = [{"brief": f"b{i}", "label": f"l{i}"} for i in range(6)]
    result = await mgr.dispatch_fanout(PID, tasks, session_id=SID)
    assert result.startswith("Error")


@pytest.mark.asyncio
async def test_missing_brief_or_label_rejected(tmp_path):
    mgr, _ = _make_manager(tmp_path)
    tasks = [{"brief": "x", "label": "a"}, {"brief": "y"}]  # task 1 missing label
    result = await mgr.dispatch_fanout(PID, tasks, session_id=SID)
    assert result.startswith("Error")


@pytest.mark.asyncio
async def test_atomic_cap_with_existing_adapters(tmp_path):
    """2 existing adapters + 4 requested tasks = 6 > MAX_CONCURRENT_SUBAGENTS
    (5) — must be rejected, and the message must mention capacity/limit."""
    mgr, _ = _make_manager(tmp_path)
    sk = (PID, SID)
    mgr._adapters[sk] = {"claude-code": object(), "codex": object()}
    tasks = [{"brief": f"b{i}", "label": f"l{i}"} for i in range(4)]
    result = await mgr.dispatch_fanout(PID, tasks, session_id=SID,
                                       max_runtime_s=3600)
    assert result.startswith("Error")
    assert "limit" in result.lower() or "capacity" in result.lower()
    # No partial registration — a rejected batch must not leave stray adapters.
    assert len(mgr._adapters[sk]) == 2


@pytest.mark.asyncio
async def test_worker_factory_unwired(tmp_path):
    mgr, _ = _make_manager(tmp_path)
    mgr._worker_deps_factory = None
    result = await mgr.dispatch_fanout(PID, _two_tasks(), session_id=SID)
    assert result.startswith("Error")
    assert "worker factory unwired" in result


@pytest.mark.asyncio
async def test_fanout_registry_unwired(tmp_path):
    mgr, _ = _make_manager(tmp_path)
    mgr._fanout_registry = None
    result = await mgr.dispatch_fanout(PID, _two_tasks(), session_id=SID)
    assert result.startswith("Error")


# ---------------------------------------------------------------------------
# Successful dispatch: registration + broadcast
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_successful_dispatch_registers_and_broadcasts(tmp_path):
    mgr, rec = _make_manager(tmp_path)
    sk = (PID, SID)
    result = await mgr.dispatch_fanout(PID, _two_tasks(), session_id=SID)

    assert result.startswith("Fanout ")
    assert "dispatched" in result
    assert "2 tasks" in result
    assert "task-a" in result and "task-b" in result

    adapters = mgr._adapters.get(sk, {})
    assert len(adapters) == 2
    handles = list(adapters.keys())
    assert all(h.startswith("worker:") for h in handles)

    for handle in handles:
        assert (PID, SID, handle) in mgr._transcripts

    started = [p for (pid, p) in rec.broadcasts if p["type"] == "fanout.started"]
    assert len(started) == 1
    assert started[0]["project_id"] == PID
    assert started[0]["session_id"] == SID
    assert {t["label"] for t in started[0]["tasks"]} == {"task-a", "task-b"}
    assert set(started[0]["fanout_id"]) <= set("0123456789abcdef")
    assert len(started[0]["fanout_id"]) == 8

    # Let the background sends drain so the test doesn't leak pending tasks.
    await _wait_until(lambda: all(
        not a.is_running() for a in adapters.values()
    ))


def _sub_agent_running_like_dispatcher(active) -> bool:
    """Verbatim copy of QueueDispatcher._sub_agent_running's reduction over
    list_active()'s return value (agent_os/queue/dispatcher.py, the
    ``_continuation_pending`` -> ``_sub_agent_running`` call chain at
    dispatcher.py:1335 -> dispatcher.py:~1330). Kept as an exact copy (not
    imported) so this test pins the ACCESS PATTERN dispatcher.py uses, not
    just whatever list_active() happens to return."""
    return any((a or {}).get("status") == "running" for a in (active or []))


@pytest.mark.asyncio
async def test_list_active_and_status_report_registered_workers(tmp_path):
    """Regression pinning the REAL consumer path, not just
    NativeWorkerAdapter.is_alive()/is_idle() in isolation:
    SubAgentManager.list_active()/status() are exactly what
    QueueDispatcher._continuation_pending (agent_os/queue/dispatcher.py:1335)
    calls on every registered adapter, so this must not raise and must
    correctly report freshly-dispatched fanout workers as running."""
    mgr, rec = _make_manager(tmp_path)
    result = await mgr.dispatch_fanout(PID, _two_tasks(), session_id=SID)
    assert result.startswith("Fanout ")
    adapters = mgr._adapters[(PID, SID)]
    handles = list(adapters.keys())
    assert len(handles) == 2

    # The background-send tasks were just created (asyncio.create_task) but
    # haven't been scheduled to actually run yet — one event-loop tick gets
    # them into send()'s body, where _running flips True before the turn's
    # own first real await.
    await asyncio.sleep(0)

    # list_active/status must not raise, and must report the in-flight
    # workers as running.
    active = mgr.list_active(PID, session_id=SID)
    assert {a["handle"] for a in active} == set(handles)
    assert all(a["status"] in ("running", "idle") for a in active)
    for h in handles:
        assert mgr.status(PID, h, session_id=SID) in ("running", "idle")

    # dispatcher.py:1335's EXACT access pattern: sub_mgr.list_active(...)
    # then any(a.get("status") == "running" for a in active). Must not
    # raise and must report True while the fanout batch is still in flight
    # — this is the slot-hold signal the running queue relies on to keep a
    # session parked until its dispatched work (including fanout workers)
    # actually finishes.
    assert _sub_agent_running_like_dispatcher(active) is True

    await _wait_until(lambda: all(not a.is_running() for a in adapters.values()))

    # Once the turns complete, still alive (not popped/broken) and idle.
    active_after = mgr.list_active(PID, session_id=SID)
    assert {a["handle"] for a in active_after} == set(handles)
    for h in handles:
        assert mgr.status(PID, h, session_id=SID) == "idle"
    assert _sub_agent_running_like_dispatcher(active_after) is False


@pytest.mark.asyncio
async def test_dispatch_fanout_accepts_files_scope(tmp_path):
    mgr, rec = _make_manager(tmp_path)
    tasks = [
        {"brief": "x", "label": "a",
         "files_scope": {"allowed": ["src/"], "forbidden": ["secrets/"]}},
        {"brief": "y", "label": "b"},
    ]
    result = await mgr.dispatch_fanout(PID, tasks, session_id=SID)
    assert result.startswith("Fanout ")
    adapters = mgr._adapters[(PID, SID)]
    await _wait_until(lambda: all(not a.is_running() for a in adapters.values()))


# ---------------------------------------------------------------------------
# Full observer-hook wiring: absorb suppresses per-worker inject, join fires
# exactly once.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_lifecycle_all_succeed_joins_once(tmp_path):
    mgr, rec = _make_manager(
        tmp_path, provider_factory=lambda: _StubProvider("done: 42"))
    observer = LifecycleObserver(rec, rec)
    observer.fanout_registry = mgr._fanout_registry
    mgr._lifecycle_observer = observer

    await mgr.dispatch_fanout(PID, _two_tasks(), session_id=SID)

    await _wait_until(lambda: len(rec.injected) >= 1)
    # Exactly one injection for the whole batch — absorb_terminal suppressed
    # the two individual "[Sub-agent] ... completed" injections that would
    # otherwise fire per worker.
    assert len(rec.injected) == 1
    content = rec.injected[0][1]
    assert "2/2 succeeded" in content

    completed = [p for (_, p) in rec.broadcasts if p["type"] == "fanout.completed"]
    assert len(completed) == 1
    assert completed[0]["succeeded"] == 2
    assert completed[0]["total"] == 2

    # The per-worker WS broadcast (sub_agent.completed) still fires — the
    # progress card needs it even though the session injection is absorbed.
    per_worker = [p for (_, p) in rec.broadcasts if p["type"] == "sub_agent.completed"]
    assert len(per_worker) == 2

    task_updates = [p for (_, p) in rec.broadcasts if p["type"] == "fanout.task_update"]
    assert len(task_updates) == 2
    assert all(t["status"] == "completed" for t in task_updates)


@pytest.mark.asyncio
async def test_full_lifecycle_all_error_joins_with_partial_report(tmp_path):
    mgr, rec = _make_manager(
        tmp_path, provider_factory=lambda: _RaisingProvider())
    observer = LifecycleObserver(rec, rec)
    observer.fanout_registry = mgr._fanout_registry
    mgr._lifecycle_observer = observer

    await mgr.dispatch_fanout(PID, _two_tasks(), session_id=SID)

    await _wait_until(lambda: len(rec.injected) >= 1)
    assert len(rec.injected) == 1
    content = rec.injected[0][1]
    assert "0/2 succeeded" in content
    assert "boom: provider unreachable" in content

    per_worker_errors = [p for (_, p) in rec.broadcasts if p["type"] == "sub_agent.error"]
    assert len(per_worker_errors) == 2


# ---------------------------------------------------------------------------
# Session-list filtering (agent_manager.list_sessions / _disk_session_entries)
# ---------------------------------------------------------------------------

def _write_session_jsonl(sessions_dir: str, uuid: str, *, worker: bool) -> None:
    os.makedirs(sessions_dir, exist_ok=True)
    path = os.path.join(sessions_dir, f"{uuid}.jsonl")
    lines = [
        {"role": "meta", "event": "session_start", "session_id": uuid,
         "session_uuid": uuid, "origin": "chat", "timestamp": "2026-07-04T00:00:00+00:00"},
    ]
    if worker:
        lines.append({
            "role": "meta", "event": "session_kind", "kind": "worker",
            "parent_session_id": "parent-1", "fanout_id": "abcd1234",
            "task_label": "some task", "timestamp": "2026-07-04T00:00:01+00:00",
        })
    lines.append({
        "role": "user", "content": "hello", "source": "user",
        "timestamp": "2026-07-04T00:00:02+00:00",
    })
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


def test_list_sessions_excludes_worker_sessions(tmp_path):
    from agent_os.daemon_v2.agent_manager import AgentManager
    from agent_os.agent.project_paths import ProjectPaths

    workspace = str(tmp_path)
    sessions_dir = ProjectPaths(workspace).sessions_dir
    _write_session_jsonl(sessions_dir, "normal_sess_1", worker=False)
    _write_session_jsonl(sessions_dir, "worker_sess_1", worker=True)

    project_store = _StubProjectStore(workspace)
    mgr = AgentManager(
        project_store=project_store, ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(), activity_translator=MagicMock(),
        process_manager=MagicMock(),
    )
    sessions = mgr.list_sessions(PID)
    session_ids = {s["session_id"] for s in sessions}
    assert "normal_sess_1" in session_ids
    assert "worker_sess_1" not in session_ids
