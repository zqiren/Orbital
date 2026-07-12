# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests: NativeWorkerAdapter + worker loop construction.

Exercises the adapter against a real ``AgentLoop``/``Session`` (not mocked)
with a stub provider and a stub tool registry, mirroring the real-loop test
style in ``test_loop_ledger_emission.py``. Covers the duck-type contract
``SubAgentManager._dispatch_async``'s ``_background_send`` fallback relies on
(sub_agent_manager.py:939-1036): ``_transport is None``, ``send()`` blocks for
one turn and never raises for task-level failures, ``_last_response`` carries
the outcome.
"""

import json

import pytest

from agent_os.agent.providers.types import StreamChunk, TokenUsage
from agent_os.agent.tools.base import ToolResult
from agent_os.daemon_v2.native_worker import (
    NativeWorkerAdapter,
    WorkerDeps,
    make_worker_handle,
)


def test_make_worker_handle():
    assert make_worker_handle("a1b2c3d4", 0) == "worker:a1b2c3d4-0"
    assert make_worker_handle("a1b2c3d4", 2) == "worker:a1b2c3d4-2"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _StubProvider:
    """Single-turn provider with provider/model/sdk identity attributes,
    matching the shape AgentLoop reads (see TextProvider in
    test_loop_ledger_emission.py)."""

    def __init__(self, text: str):
        self.provider = "stub"
        self.model = "stub-model"
        self.sdk = "stub-sdk"
        self._text = text

    async def stream(self, messages, tools=None):
        yield StreamChunk(text=self._text)
        yield StreamChunk(is_final=True, usage=TokenUsage(input_tokens=10, output_tokens=5))


class _ToolCallProvider:
    """Provider that emits a single ``mark_task_complete`` tool call — models
    a turn that ends via a tool signal rather than a trailing text-only
    assistant message, so ``_read_final_response()``'s completion-state
    fallback is exercised for real. ``mark_task_complete``/``mark_task_blocked``
    are detected by AgentLoop.run() directly off the response's tool_calls
    (loop.py's "queue signal short-circuit"), so this needs no tool actually
    registered in the stub registry."""

    def __init__(self, tool_name: str, arguments: dict):
        self.provider = "stub"
        self.model = "stub-model"
        self.sdk = "stub-sdk"
        self._tool_name = tool_name
        self._arguments = arguments

    async def stream(self, messages, tools=None):
        yield StreamChunk(tool_calls_delta=[{
            "index": 0,
            "id": "tc1",
            "type": "function",
            "function": {
                "name": self._tool_name,
                "arguments": json.dumps(self._arguments),
            },
        }])
        yield StreamChunk(is_final=True, usage=TokenUsage(input_tokens=5, output_tokens=5))


class _MultiToolCallProvider:
    """Provider that emits ``num_tool_calls`` REAL tool calls (a distinct
    ``some_tool`` call per iteration — routed through the tool registry's
    ``execute()``, unlike ``_ToolCallProvider``'s ``mark_task_complete``
    short-circuit) before ending the turn with a plain text response. Models
    a multi-tool-call turn for exercising per-tool-call activity bumps."""

    def __init__(self, num_tool_calls: int):
        self.provider = "stub"
        self.model = "stub-model"
        self.sdk = "stub-sdk"
        self._num_tool_calls = num_tool_calls
        self._iteration = 0

    async def stream(self, messages, tools=None):
        self._iteration += 1
        if self._iteration <= self._num_tool_calls:
            yield StreamChunk(tool_calls_delta=[{
                "index": 0,
                "id": f"tc{self._iteration}",
                "type": "function",
                "function": {"name": "some_tool", "arguments": "{}"},
            }])
        else:
            yield StreamChunk(text="all done")
        yield StreamChunk(is_final=True, usage=TokenUsage(input_tokens=5, output_tokens=5))


class _RaisingProvider:
    """Provider whose stream() raises mid-turn — simulates a task-level
    failure (network error, API error, etc.) inside the worker's loop."""

    def __init__(self):
        self.provider = "stub"
        self.model = "stub-model"
        self.sdk = "stub-sdk"

    async def stream(self, messages, tools=None):
        raise RuntimeError("boom: provider unreachable")
        yield  # pragma: no cover — makes this an async generator


class _StubToolRegistry:
    """Minimal ToolRegistryLike stub — no tools registered, just the shape
    AgentLoop's run() loop calls unconditionally every turn."""

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


def _make_registry_factory(registry):
    def _make_tool_registry(allowed_paths, forbidden_paths, worker_handle=None):
        return registry
    return _make_tool_registry


@pytest.fixture
def stub_worker_deps(tmp_path):
    return WorkerDeps(
        provider=_StubProvider("done: 42"),
        workspace=str(tmp_path),
        project_id="proj-1",
        parent_session_id="parent-sess-1",
        make_tool_registry=_make_registry_factory(_StubToolRegistry()),
    )


@pytest.fixture
def stub_worker_deps_err(tmp_path):
    return WorkerDeps(
        provider=_RaisingProvider(),
        workspace=str(tmp_path),
        project_id="proj-1",
        parent_session_id="parent-sess-1",
        make_tool_registry=_make_registry_factory(_StubToolRegistry()),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_sets_last_response(tmp_path, stub_worker_deps):
    a = NativeWorkerAdapter(deps=stub_worker_deps, handle="worker:x-0",
                            display_name="t0", allowed_paths=None,
                            forbidden_paths=None)
    assert a._transport is None
    assert a._idle is True
    await a.send("compute the answer")
    assert a._last_response == "done: 42"
    assert a._idle is True          # idle restored after turn
    assert a.is_running() is False


@pytest.mark.asyncio
async def test_send_strips_inline_think_block_from_last_response(tmp_path):
    """QUALITY fix: a provider that emits raw inline ``<think>...</think>``
    reasoning (e.g. a reasoning-model config gap that skips the upstream
    openai_compat.py split) must not leak that reasoning into
    ``_last_response`` — and therefore not into transcripts or fanout join
    summaries. Only the visible answer survives."""
    deps = WorkerDeps(
        provider=_StubProvider("<think>reasoning</think>\nactual answer"),
        workspace=str(tmp_path),
        project_id="proj-1",
        parent_session_id="parent-sess-1",
        make_tool_registry=_make_registry_factory(_StubToolRegistry()),
    )
    a = NativeWorkerAdapter(deps=deps, handle="worker:x-2",
                            display_name="t2", allowed_paths=None,
                            forbidden_paths=None)
    await a.send("compute the answer")
    assert a._last_response == "actual answer"

    # The RAW text (think block included) is still what the worker's own
    # session JSONL stores — the strip is applied only where
    # _read_final_response derives _last_response, never to persisted
    # storage.
    raw_contents = [
        msg.get("content") for msg in a._session.get_messages()
        if msg.get("role") == "assistant"
    ]
    assert any(
        isinstance(c, str) and "<think>" in c for c in raw_contents
    ), raw_contents


@pytest.mark.asyncio
async def test_send_failure_encoded_not_raised(tmp_path, stub_worker_deps_err):
    a = NativeWorkerAdapter(deps=stub_worker_deps_err, handle="worker:x-1",
                            display_name="t1", allowed_paths=None,
                            forbidden_paths=None)
    await a.send("boom")            # provider raises inside the loop
    assert a._last_response.startswith("Error")
    assert a._broken is True
    assert a.is_running() is False


@pytest.mark.asyncio
async def test_worker_session_tagged(tmp_path, stub_worker_deps):
    a = NativeWorkerAdapter(deps=stub_worker_deps, handle="worker:x-0",
                            display_name="t0", allowed_paths=None,
                            forbidden_paths=None)
    # session JSONL exists and carries the kind meta row
    text = open(a.session_path).read()
    assert '"session_kind"' in text and '"worker"' in text


@pytest.mark.asyncio
async def test_make_tool_registry_receives_path_scopes(tmp_path, stub_worker_deps):
    """The adapter forwards allowed_paths/forbidden_paths/handle verbatim to
    deps.make_tool_registry — Task 2's spawner-provided factory is the one
    that turns those into an enforced restricted registry (plus a
    handle-keyed BrowserTool); this task only guarantees the call happens
    with the right arguments."""
    seen = {}

    def _make_tool_registry(allowed_paths, forbidden_paths, worker_handle=None):
        seen["allowed"] = allowed_paths
        seen["forbidden"] = forbidden_paths
        seen["handle"] = worker_handle
        return _StubToolRegistry()

    stub_worker_deps.make_tool_registry = _make_tool_registry
    NativeWorkerAdapter(deps=stub_worker_deps, handle="worker:x-0",
                        display_name="t0",
                        allowed_paths=["src/"], forbidden_paths=["secrets/"])
    assert seen == {
        "allowed": ["src/"], "forbidden": ["secrets/"], "handle": "worker:x-0",
    }


@pytest.mark.asyncio
async def test_stop_cancels_turn(tmp_path, stub_worker_deps):
    """stop() delegates to loop.cancel_turn() and leaves the adapter in a
    non-running, idle state — safe to call even with no turn in flight."""
    a = NativeWorkerAdapter(deps=stub_worker_deps, handle="worker:x-0",
                            display_name="t0", allowed_paths=None,
                            forbidden_paths=None)
    await a.stop()
    assert a.is_running() is False
    assert a._idle is True


def test_is_alive_and_is_idle_fresh_adapter(tmp_path, stub_worker_deps):
    """A freshly-constructed worker is alive (not broken) and idle (no turn
    in flight) — the state every _adapters[sk] scanner (list_active/status,
    and QueueDispatcher._continuation_pending outside this package) expects
    on registration, mirroring CLIAdapter.is_alive()/is_idle()'s call shape
    (plain methods returning bool)."""
    a = NativeWorkerAdapter(deps=stub_worker_deps, handle="worker:x-7",
                            display_name="t7", allowed_paths=None,
                            forbidden_paths=None)
    assert a.is_alive() is True
    assert a.is_idle() is True


@pytest.mark.asyncio
async def test_is_idle_false_while_running(tmp_path, stub_worker_deps):
    """is_idle() reflects is_running() live, not just at construction."""
    a = NativeWorkerAdapter(deps=stub_worker_deps, handle="worker:x-8",
                            display_name="t8", allowed_paths=None,
                            forbidden_paths=None)
    a._running = True  # simulate a turn in flight (mirrors the reentrancy test)
    assert a.is_idle() is False
    assert a.is_alive() is True


@pytest.mark.asyncio
async def test_is_alive_false_once_broken(tmp_path, stub_worker_deps_err):
    """A worker whose turn raised (task-level failure) is marked _broken —
    is_alive() must reflect that so status()/list_active() report it as
    'stopped' rather than crashing or reporting a live worker."""
    a = NativeWorkerAdapter(deps=stub_worker_deps_err, handle="worker:x-9",
                            display_name="t9", allowed_paths=None,
                            forbidden_paths=None)
    await a.send("boom")
    assert a._broken is True
    assert a.is_alive() is False
    assert a.is_idle() is True  # the turn ended; not running any more


@pytest.mark.asyncio
async def test_tool_completion_falls_back_to_completion_state(tmp_path):
    """A turn that ends via mark_task_complete (not a trailing text-only
    assistant message) exercises _read_final_response()'s fallback to
    loop.get_completion_state() for real — the loop's own tool-call
    short-circuit path, not a mocked _read_final_response()."""
    deps = WorkerDeps(
        provider=_ToolCallProvider("mark_task_complete", {"summary": "did the thing"}),
        workspace=str(tmp_path),
        project_id="proj-1",
        parent_session_id="parent-sess-1",
        make_tool_registry=_make_registry_factory(_StubToolRegistry()),
    )
    a = NativeWorkerAdapter(deps=deps, handle="worker:x-2", display_name="t2",
                            allowed_paths=None, forbidden_paths=None)
    await a.send("do it")
    assert a._last_response == "did the thing"


@pytest.mark.asyncio
async def test_send_reentrancy_guard(tmp_path, stub_worker_deps):
    """A second send() while a turn is already in flight fails fast with an
    Error response instead of running two loop.run() invocations on one
    Session — a worker is one-shot by construction, so this is a defensive
    guard against a caller bug, not a normal code path."""
    a = NativeWorkerAdapter(deps=stub_worker_deps, handle="worker:x-3",
                            display_name="t3", allowed_paths=None,
                            forbidden_paths=None)
    a._running = True  # simulate a turn already in flight
    await a.send("second message")
    assert a._last_response == "Error: worker is already running a task"
    # The guard must not touch the in-flight turn's own state.
    assert a.is_running() is True


@pytest.mark.asyncio
async def test_on_activity_fires_at_turn_start_and_end(tmp_path, stub_worker_deps):
    """Task 2 activity plumbing: WorkerDeps.on_activity, when set, fires once
    at send() start and once at send() end — the turn-boundary bumps. This
    turn's stub provider makes no tool calls, so no additional per-tool-call
    bumps fire (see test_on_activity_fires_per_tool_call for that case)."""
    calls = []
    stub_worker_deps.on_activity = lambda: calls.append("tick")
    a = NativeWorkerAdapter(deps=stub_worker_deps, handle="worker:x-4",
                            display_name="t4", allowed_paths=None,
                            forbidden_paths=None)
    await a.send("compute the answer")
    assert calls == ["tick", "tick"]


@pytest.mark.asyncio
async def test_on_activity_fires_per_tool_call(tmp_path, stub_worker_deps):
    """Correction (team-lead, spec 009): turn-boundary-only bumps are
    insufficient for a single-turn worker — a long multi-tool-call turn
    would never advance last_activity until the whole turn ends, defeating
    the point of the stall watchdog for exactly the long investigative
    tasks it's meant to protect. WorkerDeps.on_activity must also fire per
    tool execution, via the registry wrap (_ActivityTrackingRegistry) —
    verified here with a provider that makes 2 real tool calls (routed
    through the stub registry's execute(), NOT the mark_task_complete/
    blocked short-circuit) before ending the turn with text."""
    calls = []
    stub_worker_deps.on_activity = lambda: calls.append("tick")
    stub_worker_deps.provider = _MultiToolCallProvider(num_tool_calls=2)
    a = NativeWorkerAdapter(deps=stub_worker_deps, handle="worker:x-11",
                            display_name="t11", allowed_paths=None,
                            forbidden_paths=None)
    await a.send("do multi-step work")
    assert a._last_response == "all done"
    # 2 turn-boundary bumps (start + end) + at least 2 per-tool-call bumps.
    assert calls.count("tick") >= 4


@pytest.mark.asyncio
async def test_on_activity_none_is_safe(tmp_path, stub_worker_deps):
    """The default (no on_activity set) must not error — most callers
    (non-fanout construction, existing tests) never set it."""
    assert stub_worker_deps.on_activity is None
    a = NativeWorkerAdapter(deps=stub_worker_deps, handle="worker:x-5",
                            display_name="t5", allowed_paths=None,
                            forbidden_paths=None)
    await a.send("compute the answer")
    assert a._last_response == "done: 42"


@pytest.mark.asyncio
async def test_on_activity_exception_does_not_break_turn(tmp_path, stub_worker_deps):
    """A broken caller-supplied on_activity hook must not corrupt the
    worker's own turn result."""
    def _boom():
        raise RuntimeError("hook exploded")
    stub_worker_deps.on_activity = _boom
    a = NativeWorkerAdapter(deps=stub_worker_deps, handle="worker:x-6",
                            display_name="t6", allowed_paths=None,
                            forbidden_paths=None)
    await a.send("compute the answer")
    assert a._last_response == "done: 42"


# ---------------------------------------------------------------------------
# Streaming (drill-in live view): worker deltas broadcast as chat.stream_delta
# addressed to the WORKER's own session_uuid. ChatView filters strictly by
# viewed session id, so these never leak into the main chat pane; the drill-in
# subscribes for exactly this id.
# ---------------------------------------------------------------------------

class _Chunk:
    def __init__(self, text="", reasoning_content="", is_final=False):
        self.text = text
        self.reasoning_content = reasoning_content
        self.is_final = is_final


def test_worker_streams_deltas_over_broadcast(tmp_path):
    events = []
    deps = WorkerDeps(
        provider=_StubProvider("done"),
        workspace=str(tmp_path),
        project_id="proj-1",
        parent_session_id="parent-sess-1",
        make_tool_registry=_make_registry_factory(_StubToolRegistry()),
        broadcast=lambda pid, payload: events.append((pid, payload)),
    )
    a = NativeWorkerAdapter(deps=deps, handle="worker:x-0", display_name="t0",
                            allowed_paths=None, forbidden_paths=None)
    a._session.notify_stream(_Chunk(text="Hel"))
    a._session.notify_stream(_Chunk(text="lo", is_final=True))
    a._session.notify_stream(_Chunk(text="next turn"))
    assert len(events) == 3
    pid, first = events[0]
    assert pid == "proj-1"
    assert first["type"] == "chat.stream_delta"
    assert first["session_id"] == a.session_uuid
    assert first["source"] == "worker:x-0"
    assert first["text"] == "Hel"
    assert first["seq"] == 1
    assert events[1][1]["is_final"] is True
    assert events[1][1]["seq"] == 2
    # Per-worker seq counter resets after a final delta (mirrors the
    # ActivityTranslator convention without sharing its per-project counter,
    # which the management stream owns).
    assert events[2][1]["seq"] == 1


def test_worker_without_broadcast_has_no_stream_observer(tmp_path, stub_worker_deps):
    a = NativeWorkerAdapter(deps=stub_worker_deps, handle="worker:x-0",
                            display_name="t0", allowed_paths=None,
                            forbidden_paths=None)
    assert a._session.on_stream is None


def test_worker_stream_broadcast_errors_are_swallowed(tmp_path):
    def _boom(pid, payload):
        raise RuntimeError("ws down")
    deps = WorkerDeps(
        provider=_StubProvider("done"),
        workspace=str(tmp_path),
        project_id="proj-1",
        parent_session_id="parent-sess-1",
        make_tool_registry=_make_registry_factory(_StubToolRegistry()),
        broadcast=_boom,
    )
    a = NativeWorkerAdapter(deps=deps, handle="worker:x-0", display_name="t0",
                            allowed_paths=None, forbidden_paths=None)
    a._session.notify_stream(_Chunk(text="x"))  # must not raise


# ---------------------------------------------------------------------------
# Browser scope teardown (Plan 3 Task 3): WorkerDeps.close_browser_scope, when
# set, is called with the worker's own handle after the one-shot turn ends
# (send()'s finally block) AND on stop() — BrowserManager.close_worker_scope
# (Task 1) is idempotent, so firing from both paths is safe even if a caller
# stops a worker mid-turn.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_closes_browser_scope_after_turn(tmp_path, stub_worker_deps):
    closed = []

    async def fake_close(scope):
        closed.append(scope)

    stub_worker_deps.close_browser_scope = fake_close
    a = NativeWorkerAdapter(deps=stub_worker_deps, handle="worker:f1-0",
                            display_name="t0", allowed_paths=None,
                            forbidden_paths=None)
    await a.send("do the task")
    assert closed == ["worker:f1-0"]


@pytest.mark.asyncio
async def test_stop_closes_browser_scope(tmp_path, stub_worker_deps):
    closed = []

    async def fake_close(scope):
        closed.append(scope)

    stub_worker_deps.close_browser_scope = fake_close
    a = NativeWorkerAdapter(deps=stub_worker_deps, handle="worker:f1-1",
                            display_name="t1", allowed_paths=None,
                            forbidden_paths=None)
    await a.stop()
    assert closed == ["worker:f1-1"]


@pytest.mark.asyncio
async def test_send_browser_scope_cleanup_failure_does_not_break_turn(
    tmp_path, stub_worker_deps
):
    """A broken close_browser_scope hook (e.g. Playwright teardown raising)
    must not corrupt the worker's own turn result — mirrors the
    on_activity-exception safety guarantee above."""
    async def fake_close(scope):
        raise RuntimeError("browser close boom")

    stub_worker_deps.close_browser_scope = fake_close
    a = NativeWorkerAdapter(deps=stub_worker_deps, handle="worker:f1-2",
                            display_name="t2", allowed_paths=None,
                            forbidden_paths=None)
    await a.send("compute the answer")
    assert a._last_response == "done: 42"
