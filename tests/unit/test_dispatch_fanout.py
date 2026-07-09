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


class _SlowProvider:
    """Provider whose stream() blocks on a long sleep before ever yielding —
    models a worker turn genuinely still in flight when stop_all interrupts
    it mid-turn (IMPORTANT 3, round-3 review)."""

    def __init__(self):
        self.provider = "stub"
        self.model = "stub-model"
        self.sdk = "stub-sdk"

    async def stream(self, messages, tools=None):
        await asyncio.sleep(10)
        yield StreamChunk(text="should never get here")  # pragma: no cover
        yield StreamChunk(is_final=True, usage=TokenUsage(input_tokens=1, output_tokens=1))


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

    async def inject_system_message(self, project_id, content, *,
                                    session_id=None, meta=None):
        # ``meta`` mirrors the real AgentManager.inject_system_message
        # signature (Task 6: fanout join summaries pass
        # meta={"display_content": ...}); recorded as a 4th tuple element so
        # existing positional assertions (indices 0-2) are untouched.
        self.injected.append((project_id, content, session_id, meta))
        return "delivered"

    def broadcast(self, project_id, payload):
        self.broadcasts.append((project_id, payload))


def _make_manager(tmp_path, *, provider_factory=None, registry_factory=None):
    """A SubAgentManager wired for fanout, with the worker registry factory
    stubbed. ``provider_factory()`` builds the (shared) provider each fanout
    batch's WorkerDeps carries; defaults to an always-succeeding stub.
    ``registry_factory`` overrides the ``make_tool_registry`` callable
    itself (default: always returns a fresh ``_StubToolRegistry``) — used
    to simulate a mid-batch construction failure."""
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
    if registry_factory is None:
        registry_factory = _make_registry_factory()

    def _worker_deps_factory(project_id, session_id):
        return WorkerDeps(
            provider=provider_factory(),
            workspace=str(tmp_path),
            project_id=project_id,
            parent_session_id=session_id,
            make_tool_registry=registry_factory,
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


@pytest.mark.asyncio
async def test_construction_failure_returns_error_and_cleans_up(tmp_path):
    """IMPORTANT (round-3 review): a mid-batch construction failure (task
    index 1 of 3's NativeWorkerAdapter raises, via a make_tool_registry that
    fails on its 2nd call) must return an Error string — never raise out of
    dispatch_fanout — and must leave NO residue: no adapters registered
    (the bulk self._adapters update happens only after the whole batch
    succeeds), no leaked transcript entries for the task that DID construct
    successfully before the failure, and no group registered with the
    FanoutRegistry."""
    calls = {"n": 0}

    def _failing_registry_factory(allowed_paths, forbidden_paths):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom: registry construction failed")
        return _StubToolRegistry()

    mgr, rec = _make_manager(tmp_path, registry_factory=_failing_registry_factory)
    tasks = [
        {"brief": "x", "label": "a"},
        {"brief": "y", "label": "b"},
        {"brief": "z", "label": "c"},
    ]
    result = await mgr.dispatch_fanout(PID, tasks, session_id=SID)

    assert result.startswith("Error")
    assert calls["n"] == 2  # confirms it failed on task index 1, not task 0

    # No adapter residue at all — the bulk self._adapters update never ran.
    assert mgr._adapters.get((PID, SID), {}) == {}
    # No leaked transcript for task 0, which DID construct successfully
    # before task 1 raised.
    assert not any(
        key[0] == PID and key[1] == SID for key in mgr._transcripts
    )
    # No group registered with the registry either.
    assert mgr._fanout_registry._groups == {}
    assert mgr._fanout_registry._by_handle == {}


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

    # D1 (issues 2+3, round 2): each fanout.started task entry carries the
    # worker's real session_uuid so the frontend can chat-shape drill-in
    # while the batch is still mid-flight (before the registry entry is
    # popped at resolution).
    by_handle = {t["handle"]: t for t in started[0]["tasks"]}
    for handle, adapter in adapters.items():
        assert adapter.session_uuid is not None
        assert adapter.session_uuid.startswith("worker_")
        assert by_handle[handle]["session_uuid"] == adapter.session_uuid

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
async def test_dispatch_fanout_forbids_orbital_dir_for_every_worker(tmp_path):
    """Spec 013 race #5: workers must not write Layer-1 memory files —
    the management session owns memory. The orbital dir is auto-appended
    to every task's forbidden_paths, including tasks with no files_scope,
    and user-supplied forbidden entries survive."""
    captured: list[tuple] = []

    def _recording_factory(allowed_paths, forbidden_paths):
        captured.append((allowed_paths, forbidden_paths))
        return _StubToolRegistry()

    mgr, rec = _make_manager(tmp_path, registry_factory=_recording_factory)
    tasks = [
        {"brief": "investigate A", "label": "task-a",
         "files_scope": {"allowed": ["src/"], "forbidden": ["src/x"]}},
        {"brief": "investigate B", "label": "task-b"},   # no files_scope
    ]
    await mgr.dispatch_fanout(PID, tasks, session_id=SID)

    from agent_os.agent.project_paths import ProjectPaths
    expected = ProjectPaths(str(tmp_path)).orbital_dir
    assert len(captured) == 2
    for allowed, forbidden in captured:
        assert forbidden is not None and expected in forbidden
    assert "src/x" in captured[0][1]          # user entry preserved
    assert captured[0][0] == ["src/"]         # allowed untouched
    assert captured[1][0] is None             # scopeless task: allowed stays None


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
    # B1 (issue 1, round 2): terminal task_update broadcasts must carry
    # completed_at_ms so the frontend can freeze each row's countdown
    # independently instead of sharing one never-freezing timer.
    assert all(isinstance(t.get("completed_at_ms"), int) for t in task_updates)

    # fanout.completed's per-task entries carry the same field.
    assert all(
        isinstance(t.get("completed_at_ms"), int) for t in completed[0]["tasks"]
    )


@pytest.mark.asyncio
async def test_full_lifecycle_reaps_adapters_after_join(tmp_path):
    """CRITICAL fix: a completed fanout's worker adapters must be reaped
    from ``_adapters`` once the join resolves — workers are one-shot
    (native_worker.py), so leaving terminal adapters registered forever
    only wastes a MAX_CONCURRENT_SUBAGENTS slot. Before the fix, both
    handles stay in ``_adapters``/``list_active`` as 'idle' after the join,
    so a second same-session fanout hits the cap error with nothing
    actually running."""
    mgr, rec = _make_manager(
        tmp_path, provider_factory=lambda: _StubProvider("done: 42"))
    observer = LifecycleObserver(rec, rec)
    observer.fanout_registry = mgr._fanout_registry
    mgr._lifecycle_observer = observer

    sk = (PID, SID)
    await mgr.dispatch_fanout(PID, _two_tasks(), session_id=SID)
    await _wait_until(lambda: len(rec.injected) >= 1)

    assert mgr._adapters.get(sk, {}) == {}
    assert mgr.list_active(PID, session_id=SID) == []

    # A second fanout of 5 tasks in the same session must now succeed — a
    # cap check that still sees the first batch's leaked terminal adapters
    # would reject this with a capacity error.
    tasks = [{"brief": f"b{i}", "label": f"l{i}"} for i in range(5)]
    result = await mgr.dispatch_fanout(PID, tasks, session_id=SID)
    assert result.startswith("Fanout "), result

    adapters = mgr._adapters.get(sk, {})
    await _wait_until(lambda: all(not a.is_running() for a in adapters.values()))


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


@pytest.mark.asyncio
async def test_stop_all_mid_flight_reports_interrupted_not_error(tmp_path):
    """IMPORTANT 3 (round-3 review): a worker torn down by stop_all (project
    stop / user stop mid-flight) must be reported 'interrupted', not
    'error' — a deliberate stop is not a failure. Traced path: stop_all ->
    SubAgentManager.stop() -> NativeWorkerAdapter.stop() sets
    _stop_requested THEN cancels the turn -> AgentLoop.run() returns
    normally with exit_reason='cancelled' -> _read_final_response()'s
    fallback produces an 'Error: task was cancelled...' response ->
    _background_send's routing must recognize _stop_requested and route to
    on_turn_interrupted instead of on_error."""
    mgr, rec = _make_manager(tmp_path, provider_factory=lambda: _SlowProvider())
    observer = LifecycleObserver(rec, rec)
    observer.fanout_registry = mgr._fanout_registry
    mgr._lifecycle_observer = observer

    await mgr.dispatch_fanout(PID, _two_tasks(), session_id=SID)
    # Let both background sends actually enter their turn (reach the
    # provider's blocking stream()) before tearing down mid-flight.
    await asyncio.sleep(0.05)

    await mgr.stop_all(PID, session_id=SID)

    await _wait_until(lambda: len(rec.injected) >= 1)
    assert len(rec.injected) == 1
    content = rec.injected[0][1]
    # Both tasks interrupted, never "error" — a deliberate stop is not a
    # failure (this is the exact mislabeling the fix corrects).
    assert "[interrupted]" in content
    assert "[error]" not in content

    per_worker_interrupted = [
        p for (_, p) in rec.broadcasts
        if p["type"] == "sub_agent.turn_interrupted"
    ]
    assert len(per_worker_interrupted) == 2
    per_worker_errors = [
        p for (_, p) in rec.broadcasts if p["type"] == "sub_agent.error"
    ]
    assert len(per_worker_errors) == 0

    completed = [
        p for (_, p) in rec.broadcasts if p["type"] == "fanout.completed"
    ]
    assert len(completed) == 1
    assert completed[0]["succeeded"] == 0


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
