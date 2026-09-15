# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for backlog #23 D1 + #28 and spec 091 §4: a sub-agent
terminal event that lands while the management turn is in flight must wake
the management loop once that turn ends.

A terminal lifecycle event injected mid-turn is deferred
(``Session.defer_message``) for safe insertion after the current tool batch.
Backlog #23 tagged those markers (``_meta = {"event": "sub_agent_terminal",
"kind": ...}``) and taught ``_on_loop_done`` to hot-resume the loop when its
drain of the deferred buffer found one; backlog #28 widened the tag from the
negative terminals to every terminal (``completed``, ``interrupted``, the
fanout join summary).

Spec 091 §4: that drain almost never finds anything. ``AgentLoop.run()``'s own
``finally`` drains the same buffer first, and ``_on_loop_done`` is the loop
task's done-callback, so it always runs after that ``finally`` — the buffer is
already empty and the wake never fired (found live by the v0.13.0 cut smoke).
The earlier consumer tests here drove ``_on_loop_done`` with a mock task, so
the loop's drain never ran and they passed falsely. The fix decides at defer
time: ``inject_system_message`` flags the handle, and ``_on_loop_done``
consumes the flag.

The consumer tests therefore run a REAL ``AgentLoop.run()`` task with a
scripted LLM, inject while the turn is genuinely in flight, and let the real
done-callback decide. A few callback-level tests remain below them for the
branches that need no loop (the rare row deferred after the loop's drain, the
idle and queued-message paths). Producer-side tests (does each terminal carry
the tag, without losing the #24 ``display_content`` split?) are unchanged.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.context import ContextManager
from agent_os.agent.loop import AgentLoop
from agent_os.agent.prompt_builder import Autonomy, PromptContext
from agent_os.agent.providers.types import StreamChunk, TokenUsage
from agent_os.agent.session import Session, persist_user_row
from agent_os.agent.tools.base import ToolResult
from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle
from agent_os.daemon_v2.fanout import FanoutRegistry, FanoutTask
from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver
from agent_os.daemon_v2.models import make_session_key


PROJECT = "proj_test"
# Explicit session id: the "default" sentinel is retired (seam 3 / D1); bare
# project-level calls no longer resolve to a planted handle.
SID = "sess-wake-deferred-0001"


@pytest.fixture
def manager():
    ws = MagicMock()
    ws.broadcast = MagicMock()
    project_store = MagicMock()
    sub_agent_manager = MagicMock()
    sub_agent_manager.list_active = MagicMock(return_value=[])
    activity_translator = MagicMock()
    process_manager = MagicMock()
    mgr = AgentManager(
        project_store=project_store,
        ws_manager=ws,
        sub_agent_manager=sub_agent_manager,
        activity_translator=activity_translator,
        process_manager=process_manager,
    )
    return mgr


def _terminal_meta(kind: str, **extra) -> dict:
    """The ``_meta`` the LifecycleObserver terminals and the fanout join
    actually stamp."""
    return {"event": "sub_agent_terminal", "kind": kind, **extra}


def _terminal_marker(content: str, kind: str) -> dict:
    """A deferred-buffer row shaped the way ``Session.defer_message`` builds
    one for a tagged terminal."""
    return {
        "role": "system",
        "content": content,
        "source": "daemon",
        "timestamp": "2026-07-22T00:00:00+00:00",
        "_meta": _terminal_meta(kind),
    }


async def _wait_until(cond, *, timeout: float = 2.0, interval: float = 0.005):
    """Poll ``cond`` — the fanout stubs below suspend for real (mirroring
    test_fanout_registry.py), so a group needs more than one loop tick to
    resolve."""
    loop = asyncio.get_event_loop()
    start = loop.time()
    while not cond():
        if loop.time() - start > timeout:
            raise AssertionError(f"condition not met within {timeout}s")
        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Real-loop harness
# ---------------------------------------------------------------------------


class _PromptBuilder:
    def build(self, context):
        return ("cached-system-prefix", "semi-stable-suffix", "dynamic-runtime")


def _prompt_context(workspace: str) -> PromptContext:
    return PromptContext(
        workspace=workspace,
        model="test-model",
        autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[],
        tool_names=[],
        os_type="linux",
        datetime_now="2026-01-01T00:00:00",
        context_usage_pct=0.0,
    )


_USAGE = TokenUsage(input_tokens=10, output_tokens=5)


def _reply(text: str) -> list[StreamChunk]:
    return [StreamChunk(text=text),
            StreamChunk(is_final=True, usage=_USAGE, finish_reason="stop")]


def _tool_call(call_id: str, name: str) -> list[StreamChunk]:
    return [
        StreamChunk(tool_calls_delta=[{
            "index": 0, "id": call_id, "type": "function",
            "function": {"name": name, "arguments": "{}"},
        }]),
        StreamChunk(is_final=True, usage=_USAGE, finish_reason="tool_calls"),
    ]


class _ScriptedLLM:
    """Replays one scripted response per call. The call numbered
    ``gate_call`` blocks until ``release`` is set, so a test can inject while
    that LLM call is genuinely in flight. Records what every call was sent."""

    provider = "test"
    model = "test-model"
    sdk = "openai"

    def __init__(self, script, *, gate_call=None):
        self._script = script
        self._gate_call = gate_call
        self.calls = 0
        self.sent: list[list[dict]] = []
        self.in_flight = asyncio.Event()
        self.release = asyncio.Event()

    async def stream(self, messages, tools=None):
        idx = self.calls
        self.calls += 1
        self.sent.append(list(messages))
        if idx == self._gate_call:
            self.in_flight.set()
            await self.release.wait()
        for chunk in self._script[min(idx, len(self._script) - 1)]:
            yield chunk


class _SlowTool:
    """A one-tool registry whose execution blocks until released — the
    in-flight tool batch. ``yield_turn`` makes the result end the turn the
    way a dispatch tool (``agent_message``) does."""

    def __init__(self, *, yield_turn: bool = False):
        self._yield_turn = yield_turn
        self.running = asyncio.Event()
        self.release = asyncio.Event()

    def schemas(self) -> list[dict]:
        return [{"type": "function", "function": {
            "name": "slow_tool", "description": "blocks until released",
            "parameters": {"type": "object", "properties": {}},
        }}]

    def tool_names(self) -> list[str]:
        return ["slow_tool"]

    def is_async(self, name: str) -> bool:
        return True

    def execute(self, name: str, arguments: dict) -> ToolResult:
        raise AssertionError("slow_tool is async-only")

    async def execute_async(self, name: str, arguments: dict) -> ToolResult:
        self.running.set()
        await self.release.wait()
        meta = {"yield_turn": True} if self._yield_turn else None
        return ToolResult(content="tool done", meta=meta)

    def reset_run_state(self) -> None:
        pass


def _run_turn(manager, handle) -> asyncio.Task:
    """Start ``loop.run()`` exactly as ``start_agent``/``_start_loop`` do: a
    task whose done-callback is the manager's real ``_on_loop_done``."""
    task = asyncio.create_task(handle.loop.run())
    task.add_done_callback(manager._on_loop_done(PROJECT, session_id=SID))
    handle.task = task
    return task


def _start_real_turn(manager, tmp_path, llm, tools=None):
    session = Session.new(SID, str(tmp_path))
    tools = tools or _SlowTool()
    ctx = ContextManager(session, _PromptBuilder(),
                         _prompt_context(str(tmp_path)))
    loop = AgentLoop(session, llm, tools, ctx,
                     project_dir=str(tmp_path), max_iterations=10)
    persist_user_row(session, "hello")
    handle = ProjectHandle(
        session=session, loop=loop, provider=llm, registry=tools,
        context_manager=ctx, interceptor=None, task=None,
    )
    manager._handles[make_session_key(PROJECT, SID)] = handle
    _run_turn(manager, handle)
    return handle, session


def _spy_wake(manager) -> AsyncMock:
    """Replace the hot-resume with a spy: the wake is the call, not the
    follow-up turn it would start."""
    manager._start_loop = AsyncMock()
    return manager._start_loop


async def _turn_ended(handle) -> None:
    """Await the loop task AND its done-callback, which the event loop
    schedules a tick after the task resolves."""
    await handle.task
    for _ in range(5):
        await asyncio.sleep(0)


def _rows_with(session, content: str) -> list[dict]:
    return [m for m in session.get_messages() if m.get("content") == content]


# ---------------------------------------------------------------------------
# Consumer side, through a real AgentLoop.run() task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind",
    # #23's negative terminals…
    ["error", "failed", "stopped",
     # …and #28's three siblings (completed / interrupted / fanout join).
     "completed", "interrupted", "fanout_join"],
)
async def test_terminal_deferred_during_the_final_llm_call_wakes_after_the_turn(
        manager, tmp_path, kind):
    """The live bug (spec 091 §4): the terminal lands while the turn's last,
    text-only LLM call streams. The loop's own ``finally`` appends it after
    the reply and the turn ends, so nothing in that turn ever read it — the
    manager must wake for it."""
    wake = _spy_wake(manager)
    llm = _ScriptedLLM([_reply("the reply")], gate_call=0)
    handle, session = _start_real_turn(manager, tmp_path, llm)
    await asyncio.wait_for(llm.in_flight.wait(), timeout=2)

    content = f"[Sub-agent] codex {kind}"
    status = await manager.inject_system_message(
        PROJECT, content, session_id=SID, meta=_terminal_meta(kind))
    assert status == "deferred"

    llm.release.set()
    await _turn_ended(handle)

    # The bug's precondition, pinned so this can never drift back to a mock:
    # the loop itself drained the row — after the reply, before the callback.
    order = [(m.get("role"), m.get("content")) for m in session.get_messages()]
    assert order.index(("assistant", "the reply")) < order.index(("system", content))
    assert session.pop_deferred_messages() == []
    assert llm.calls == 1

    wake.assert_called_once_with(PROJECT, session_id=SID)


@pytest.mark.asyncio
async def test_terminal_deferred_during_a_turn_ending_tool_batch_wakes(
        manager, tmp_path):
    """Same gap, other drain point: the terminal lands during a tool batch
    that ends the turn (a dispatch tool's ``yield_turn``). The loop drains it
    right after the batch and exits with no further LLM call, so it is unread
    — the manager must wake."""
    wake = _spy_wake(manager)
    tools = _SlowTool(yield_turn=True)
    llm = _ScriptedLLM([_tool_call("tc1", "slow_tool"), _reply("unreached")])
    handle, session = _start_real_turn(manager, tmp_path, llm, tools)
    await asyncio.wait_for(tools.running.wait(), timeout=2)

    content = "[Sub-agent] codex stopped with error: boom"
    status = await manager.inject_system_message(
        PROJECT, content, session_id=SID, meta=_terminal_meta("error"))
    assert status == "deferred"

    tools.release.set()
    await _turn_ended(handle)

    assert llm.calls == 1
    assert len(_rows_with(session, content)) == 1
    wake.assert_called_once_with(PROJECT, session_id=SID)


@pytest.mark.asyncio
async def test_terminal_read_by_a_later_llm_call_of_the_same_turn_does_not_wake_again(
        manager, tmp_path):
    """A terminal deferred during an ordinary tool batch is drained right
    after it and read by the turn's NEXT LLM call — that turn has already
    answered it. Waking again would re-run a turn with nothing new in it."""
    wake = _spy_wake(manager)
    tools = _SlowTool()
    llm = _ScriptedLLM([_tool_call("tc1", "slow_tool"),
                        _reply("the worker failed; here is what happened")])
    handle, session = _start_real_turn(manager, tmp_path, llm, tools)
    await asyncio.wait_for(tools.running.wait(), timeout=2)

    content = "[Sub-agent] codex stopped with error: boom"
    status = await manager.inject_system_message(
        PROJECT, content, session_id=SID, meta=_terminal_meta("error"))
    assert status == "deferred"

    tools.release.set()
    await _turn_ended(handle)

    assert llm.calls == 2
    assert content in json.dumps(llm.sent[1], default=str)
    wake.assert_not_called()


@pytest.mark.asyncio
async def test_pinned_terminal_deferred_mid_turn_lands_without_waking(
        manager, tmp_path):
    """suppress_wake (spec 074): a pinned dispatch's terminal still lands in
    the session but never starts a management turn, deferred or not."""
    wake = _spy_wake(manager)
    llm = _ScriptedLLM([_reply("the reply")], gate_call=0)
    handle, session = _start_real_turn(manager, tmp_path, llm)
    await asyncio.wait_for(llm.in_flight.wait(), timeout=2)

    content = "[Sub-agent] codex completed. Summary: done."
    await manager.inject_system_message(
        PROJECT, content, session_id=SID,
        meta=_terminal_meta("completed", suppress_wake=True))

    llm.release.set()
    await _turn_ended(handle)

    assert len(_rows_with(session, content)) == 1
    wake.assert_not_called()


@pytest.mark.asyncio
async def test_untagged_marker_deferred_mid_turn_does_not_wake(manager, tmp_path):
    """Scope guard: the TAG is what wakes, not the fact that something was
    deferred. Non-terminal lifecycle chatter (a "started" marker, or the
    pre-#28 join shape that carried only display_content) still lands
    silently — otherwise every routine notification would restart the loop."""
    wake = _spy_wake(manager)
    llm = _ScriptedLLM([_reply("the reply")], gate_call=0)
    handle, session = _start_real_turn(manager, tmp_path, llm)
    await asyncio.wait_for(llm.in_flight.wait(), timeout=2)

    started = "[Sub-agent] cursor started (initiated by: management_agent)."
    joined = "[Fanout f] 1/1 succeeded.\n- [completed] a (worker:f-0)"
    await manager.inject_system_message(PROJECT, started, session_id=SID)
    await manager.inject_system_message(
        PROJECT, joined, session_id=SID,
        meta={"display_content": "[Fanout f] 1/1 succeeded."})

    llm.release.set()
    await _turn_ended(handle)

    assert len(_rows_with(session, started)) == 1
    assert len(_rows_with(session, joined)) == 1
    wake.assert_not_called()


@pytest.mark.asyncio
async def test_the_wake_is_not_repeated_by_the_turn_it_started(manager, tmp_path):
    """No double-processing: the turn the wake starts reads the terminal and
    ends with nothing newly deferred — it must not wake a third turn."""
    wake = _spy_wake(manager)
    llm = _ScriptedLLM([_reply("the reply"), _reply("addressed the failure")],
                       gate_call=0)
    handle, session = _start_real_turn(manager, tmp_path, llm)
    await asyncio.wait_for(llm.in_flight.wait(), timeout=2)
    await manager.inject_system_message(
        PROJECT, "[Sub-agent] codex stopped with error: boom",
        session_id=SID, meta=_terminal_meta("error"))
    llm.release.set()
    await _turn_ended(handle)
    wake.assert_called_once()

    # The woken turn, started the way the real _start_loop would.
    _run_turn(manager, handle)
    await _turn_ended(handle)

    assert llm.calls == 2
    wake.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("event", ["completed", "interrupted"])
async def test_each_new_producer_wakes_through_a_real_turn(
        manager, tmp_path, event):
    """End-to-end with no marker shapes hand-written: the observer injects
    while a real turn is in flight, ``inject_system_message`` defers, the loop
    drains, and the done-callback wakes."""
    wake = _spy_wake(manager)
    llm = _ScriptedLLM([_reply("the reply")], gate_call=0)
    handle, session = _start_real_turn(manager, tmp_path, llm)
    await asyncio.wait_for(llm.in_flight.wait(), timeout=2)
    observer = LifecycleObserver(manager, MagicMock())

    if event == "completed":
        await observer.on_completed(
            PROJECT, "cursor", "all green", "/tmp/t.jsonl", session_id=SID)
    else:
        await observer.on_turn_interrupted(
            PROJECT, "cursor", "/tmp/t.jsonl", session_id=SID)
    assert len(session._deferred_messages) == 1

    llm.release.set()
    await _turn_ended(handle)

    contents = [m.get("content") or "" for m in session.get_messages()]
    assert any(c.startswith("[Sub-agent] cursor") for c in contents)
    wake.assert_called_once_with(PROJECT, session_id=SID)


@pytest.mark.asyncio
async def test_an_all_failed_fanout_join_wakes_through_a_real_turn(
        manager, tmp_path):
    """Same end-to-end, driven by the fanout registry wired to the real
    ``inject_system_message`` — every worker failed, so the join summary is
    the only thing that can tell the owner anything."""
    wake = _spy_wake(manager)
    llm = _ScriptedLLM([_reply("the reply")], gate_call=0)
    handle, session = _start_real_turn(manager, tmp_path, llm)
    await asyncio.wait_for(llm.in_flight.wait(), timeout=2)

    async def stop_worker(project_id, worker, session_id=None):
        await asyncio.sleep(0)

    registry = FanoutRegistry(
        inject=manager.inject_system_message,
        broadcast=lambda *a, **k: None,
        stop_worker=stop_worker,
    )
    registry.create_group(
        PROJECT, SID,
        [FanoutTask(handle="worker:f-0", label="a", brief="x")],
        max_runtime_s=3600,
    )
    registry.absorb_terminal(PROJECT, "worker:f-0", SID, kind="error",
                             summary="ProviderError: 429",
                             transcript_path="t0")
    await _wait_until(lambda: bool(session._deferred_messages))

    llm.release.set()
    await _turn_ended(handle)

    contents = [m.get("content") or "" for m in session.get_messages()]
    assert any(c.startswith("[Fanout f] 0/1 succeeded.") for c in contents)
    wake.assert_called_once_with(PROJECT, session_id=SID)


# ---------------------------------------------------------------------------
# Callback-level branches that need no loop. These call ``_on_loop_done``
# directly on a mock handle, so they say nothing about rows the loop drains
# itself — that is what the real-loop tests above are for.
# ---------------------------------------------------------------------------


def _handle_with_deferred(manager, deferred_messages, *, queued=None):
    session = MagicMock()
    session.is_stopped.return_value = False
    session._paused_for_approval = False
    session.pop_deferred_messages.return_value = deferred_messages
    session.pop_queued_messages.return_value = queued or []
    session.append = MagicMock()

    handle = MagicMock()
    handle.session = session
    handle.loop = MagicMock(last_llm_error=None)

    task = MagicMock()
    task.exception.return_value = None
    handle.task = task

    manager._handles[make_session_key(PROJECT, SID)] = handle
    return handle, task


def _run_callback(manager, task):
    callback = manager._on_loop_done(PROJECT, session_id=SID)
    mock_future = MagicMock()
    with patch("asyncio.ensure_future", return_value=mock_future) as mock_ensure:
        callback(task)
        if mock_ensure.call_args:
            coro = mock_ensure.call_args[0][0]
            coro.close()
    return mock_ensure


@pytest.mark.parametrize("kind", ["error", "completed", "fanout_join"])
def test_row_still_in_the_buffer_at_callback_time_wakes(manager, kind):
    """The buffer scan ``_on_loop_done`` keeps: a terminal deferred after the
    loop's own drain already ran, but before the done-callback, is appended
    here and wakes the loop."""
    handle, task = _handle_with_deferred(
        manager, [_terminal_marker(f"[Sub-agent] cursor {kind}", kind)])

    mock_ensure = _run_callback(manager, task)

    handle.session.append.assert_called_once()
    appended = handle.session.append.call_args[0][0]
    assert appended["content"] == f"[Sub-agent] cursor {kind}"
    mock_ensure.assert_called_once()


def test_untagged_row_in_the_buffer_does_not_wake(manager):
    started = {
        "role": "system",
        "content": "[Sub-agent] cursor started (initiated by: user).",
        "source": "daemon",
        "timestamp": "2026-07-22T00:00:00+00:00",
    }
    handle, task = _handle_with_deferred(manager, [started])

    mock_ensure = _run_callback(manager, task)

    handle.session.append.assert_called_once()
    mock_ensure.assert_not_called()


def test_idle_path_unchanged_when_nothing_deferred(manager):
    """No regression: an ordinary idle turn-end (no deferred, no queued,
    no busy sub-agents) still broadcasts idle and does not spuriously wake."""
    handle, task = _handle_with_deferred(manager, [])

    mock_ensure = _run_callback(manager, task)

    handle.session.append.assert_not_called()
    mock_ensure.assert_not_called()
    manager._ws.broadcast.assert_called()
    call_args = manager._ws.broadcast.call_args[0]
    assert call_args[1]["status"] == "idle"


def test_queued_user_message_still_wakes_without_any_deferred(manager):
    """Regression guard: the pre-existing queued-user-message resume path is
    untouched when there is nothing deferred at all."""
    handle, task = _handle_with_deferred(
        manager, [], queued=[("go ahead", None)])

    mock_ensure = _run_callback(manager, task)

    handle.session.append.assert_called_once()
    appended = handle.session.append.call_args[0][0]
    assert appended["role"] == "user"
    assert appended["content"] == "go ahead"
    mock_ensure.assert_called_once()


# ---------------------------------------------------------------------------
# Producer side (backlog #28) — the three terminals that shipped untagged.
# Each asserts the wake tag AND that nothing else about the marker moved:
# the content strings are consumed by the chat renderer's parity fixture, and
# `display_content` is #24's user-visible/agent-facing split.
# ---------------------------------------------------------------------------


class _RecordingManager:
    """Captures every injection exactly as ``inject_system_message`` receives
    it, so producer-side tests assert on the same ``meta`` dict the consumer
    later reads off ``_meta``."""

    def __init__(self):
        self.injections: list[tuple[str, dict]] = []

    async def inject_system_message(self, project_id, content, **kwargs):
        self.injections.append((content, kwargs))


@pytest.mark.asyncio
async def test_on_completed_carries_the_wake_tag_and_keeps_display_content():
    """(a) A mid-turn completion used to be appended silently."""
    agent_manager = _RecordingManager()
    observer = LifecycleObserver(agent_manager, MagicMock())

    await observer.on_completed(
        PROJECT, "cursor", "all green", "/tmp/t.jsonl", session_id=SID)

    content, kwargs = agent_manager.injections[0]
    meta = kwargs["meta"]
    assert meta["event"] == "sub_agent_terminal"
    assert meta["kind"] == "completed"
    # The #24 split survives the added keys: display_content is still the
    # clean marker, and the LLM-facing content still carries the guidance.
    assert meta["display_content"] == (
        "[Sub-agent] cursor completed. Summary: all green. "
        "Transcript: /tmp/t.jsonl."
    )
    assert content.startswith(meta["display_content"])
    assert "do NOT repeat or re-summarize" in content


@pytest.mark.asyncio
async def test_on_completed_no_output_variant_also_carries_the_wake_tag():
    """The empty-summary branch takes a different guidance string — and must
    not take a different meta."""
    agent_manager = _RecordingManager()
    observer = LifecycleObserver(agent_manager, MagicMock())

    await observer.on_completed(
        PROJECT, "cursor", "", "/tmp/t.jsonl", session_id=SID)

    content, kwargs = agent_manager.injections[0]
    assert kwargs["meta"]["event"] == "sub_agent_terminal"
    assert kwargs["meta"]["kind"] == "completed"
    assert content.startswith(kwargs["meta"]["display_content"])


@pytest.mark.asyncio
async def test_on_turn_interrupted_carries_the_wake_tag():
    """(b) The silence its own docstring calls the Piece-3 Part-C
    silent-hang class: the management session may be AWAITING this result."""
    agent_manager = _RecordingManager()
    observer = LifecycleObserver(agent_manager, MagicMock())

    await observer.on_turn_interrupted(
        PROJECT, "cursor", "/tmp/t.jsonl", session_id=SID)

    content, kwargs = agent_manager.injections[0]
    assert kwargs["meta"]["event"] == "sub_agent_terminal"
    assert kwargs["meta"]["kind"] == "interrupted"
    # Marker text unchanged (no display split on this one).
    assert content.startswith(
        "[Sub-agent] cursor was stopped before completing its current task")
    assert "display_content" not in kwargs["meta"]


def _fanout_registry(injections):
    """A registry whose collaborators suspend for real — see
    test_fanout_registry.py's module docstring on why that matters."""
    async def inject(project_id, content, **kwargs):
        await asyncio.sleep(0)
        injections.append((content, kwargs))

    async def stop_worker(project_id, handle, session_id=None):
        await asyncio.sleep(0)

    return FanoutRegistry(inject=inject, broadcast=lambda *a, **k: None,
                          stop_worker=stop_worker)


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_kind", ["completed", "error"])
async def test_fanout_join_summary_carries_the_wake_tag(worker_kind):
    """(c) The join summary is the group's terminal event. The all-failed
    group (worker_kind="error") is the case that matters most: nothing else
    will ever wake the owner to report the failure."""
    injections: list[tuple[str, dict]] = []
    registry = _fanout_registry(injections)
    registry.create_group(
        PROJECT, SID,
        [FanoutTask(handle="worker:f-0", label="a", brief="x"),
         FanoutTask(handle="worker:f-1", label="b", brief="y")],
        max_runtime_s=3600,
    )
    for i in (0, 1):
        registry.absorb_terminal(
            PROJECT, f"worker:f-{i}", SID, kind=worker_kind,
            summary=f"s{i}", transcript_path=f"t{i}")

    await _wait_until(lambda: bool(injections))

    content, kwargs = injections[0]
    meta = kwargs["meta"]
    assert meta["event"] == "sub_agent_terminal"
    assert meta["kind"] == "fanout_join"
    succeeded = 2 if worker_kind == "completed" else 0
    assert content.startswith(f"[Fanout f] {succeeded}/2 succeeded.")
    # FROZEN join format + the display split are both untouched.
    assert meta["display_content"].startswith(f"[Fanout f] {succeeded}/2")
    assert "Synthesize these results" not in meta["display_content"]
    assert "Synthesize these results" in content
