# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 129 §5A: the claude-code transport reads its stream for the life of
the process, not per dispatch.

Observed 2026-10-22: a worker that started a ``run_in_background`` shell and
ended its turn went idle in Orbital, then ran three turns of its own when the
shell exited (the CLI enqueues a ``<task-notification>`` and runs a model
turn with no ``query()`` from us). Nothing read between dispatches, so those
turns were stranded; after ~100 unread messages the SDK's reader blocked on
its bounded channel and stopped answering tool permissions — a hard freeze
until the user's next prompt drained the backlog and mislabelled every stale
turn as the reply.

One reader, started at connect and cancelled at stop. A ``ResultMessage``
closes the pending dispatch when one is pending; otherwise it ends a
self-initiated turn (``turn_complete`` carries ``origin``).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.transports.base import TransportEvent
from agent_os.agent.transports.sdk_transport import HAS_SDK, SDKTransport

pytestmark = [pytest.mark.asyncio,
              pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")]


# ---------------------------------------------------------------------------
# SDK message builders
# ---------------------------------------------------------------------------

def _assistant(text: str):
    from claude_agent_sdk.types import AssistantMessage, TextBlock
    return AssistantMessage(content=[TextBlock(text=text)], model="claude-x",
                            parent_tool_use_id=None, error=None)


def _result(*, is_error=False, session_id="sdk-s-1"):
    from claude_agent_sdk.types import ResultMessage
    return ResultMessage(
        subtype="error" if is_error else "success",
        duration_ms=10, duration_api_ms=8, is_error=is_error, num_turns=1,
        session_id=session_id, total_cost_usd=0.0, usage={}, result=None,
        structured_output=None,
    )


def _task_started(*, task_id: str, task_type: str):
    from claude_agent_sdk.types import TaskStartedMessage
    return TaskStartedMessage(
        subtype="task_started", data={}, task_id=task_id,
        description="d", uuid="u-" + task_id, session_id="sdk-s-1",
        tool_use_id="tu-" + task_id, task_type=task_type,
    )


def _task_notification(*, task_id: str, status: str = "completed"):
    from claude_agent_sdk.types import TaskNotificationMessage
    return TaskNotificationMessage(
        subtype="task_notification", data={}, task_id=task_id, status=status,
        output_file="", summary="done", uuid="n-" + task_id,
        session_id="sdk-s-1", tool_use_id="tu-" + task_id, usage=None,
    )


class _ChannelClient:
    """A ClaudeSDKClient over a live channel: ``receive_messages()`` is the
    SDK's unbounded iterator (yields until the channel ends, raises when
    the channel carries an exception). ``receive_response()`` is a trap —
    the persistent reader must never call it."""

    _END = object()

    def __init__(self):
        self._chan: asyncio.Queue = asyncio.Queue()
        self.queries: list[str] = []
        self.disconnect_called = False

    async def query(self, message, *a, **k):
        self.queries.append(message)

    def feed(self, *messages):
        for m in messages:
            self._chan.put_nowait(m)

    def end(self):
        self._chan.put_nowait(self._END)

    def fail(self, exc: Exception):
        self._chan.put_nowait(exc)

    async def receive_messages(self):
        while True:
            item = await self._chan.get()
            if item is self._END:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    def receive_response(self):
        raise AssertionError("receive_response() must not be used by the persistent reader")

    async def disconnect(self):
        self.disconnect_called = True


async def _transport(workspace: str) -> tuple[SDKTransport, _ChannelClient]:
    transport = SDKTransport()
    client = _ChannelClient()
    transport._client = client
    transport._alive = True
    transport._workspace = workspace  # real path → the display-only usage ledger is quiet
    transport._ensure_reader()
    return transport, client


async def _settle(n: int = 20) -> None:
    for _ in range(n):
        await asyncio.sleep(0)


def _drain(transport) -> list[TransportEvent]:
    events = []
    while not transport._event_queue.empty():
        events.append(transport._event_queue.get_nowait())
    return events


def _turn_completes(events):
    return [e for e in events if e.event_type == "turn_complete"]


def _texts(events):
    return [e.raw_text for e in events if e.event_type == "message"]


async def _stop(transport):
    with patch("agent_os.agent.transports.process_kill.kill_process_tree", new=AsyncMock()):
        await transport.stop()


# ---------------------------------------------------------------------------
# §6.1 — unsolicited messages are read with no dispatch pending
# ---------------------------------------------------------------------------

async def test_messages_with_no_pending_dispatch_are_emitted_as_a_self_turn(tmp_path):
    transport, client = await _transport(str(tmp_path))
    try:
        client.feed(_assistant("the shell finished; frames look fine"), _result())
        await _settle()
        events = _drain(transport)

        assert _texts(events) == ["the shell finished; frames look fine"]
        tcs = _turn_completes(events)
        assert len(tcs) == 1
        assert tcs[0].data["origin"] == "self"
        assert tcs[0].data["cause"] == "success"
        assert tcs[0].data["session_id"] == "sdk-s-1"
        assert client.queries == []
    finally:
        await _stop(transport)


async def test_self_turn_error_result_is_cause_error_never_stopped(tmp_path):
    transport, client = await _transport(str(tmp_path))
    try:
        client.feed(_assistant("x"), _result(is_error=True))
        await _settle()
        tcs = _turn_completes(_drain(transport))
        assert [(t.data["origin"], t.data["cause"]) for t in tcs] == [("self", "error")]
    finally:
        await _stop(transport)


# ---------------------------------------------------------------------------
# §6.2 — a dispatch's result closes the dispatch; the next result is self
# ---------------------------------------------------------------------------

async def test_dispatch_result_closes_the_dispatch_and_a_later_result_is_self(tmp_path):
    transport, client = await _transport(str(tmp_path))
    try:
        await transport.dispatch("go")
        assert transport.dispatch_pending is True
        client.feed(_assistant("answer"), _result())
        await _settle()
        events = _drain(transport)
        tcs = _turn_completes(events)
        assert len(tcs) == 1
        assert tcs[0].data["origin"] == "dispatch"
        assert tcs[0].data["cause"] == "success"
        assert transport.dispatch_pending is False

        client.feed(_assistant("woke up on my own"), _result())
        await _settle()
        events = _drain(transport)
        assert _texts(events) == ["woke up on my own"]
        assert [t.data["origin"] for t in _turn_completes(events)] == ["self"]
    finally:
        await _stop(transport)


async def test_dispatch_error_result_is_cause_error(tmp_path):
    transport, client = await _transport(str(tmp_path))
    try:
        await transport.dispatch("go")
        client.feed(_result(is_error=True))
        await _settle()
        tcs = _turn_completes(_drain(transport))
        assert [(t.data["origin"], t.data["cause"]) for t in tcs] == [("dispatch", "error")]
    finally:
        await _stop(transport)


async def test_a_second_dispatch_while_one_is_pending_closes_the_first_as_stopped(tmp_path):
    """Today's contract, kept: the prior turn's boundary is written (cause
    stopped) before the new query so the dispatch-id FIFO stays paired."""
    transport, client = await _transport(str(tmp_path))
    try:
        await transport.dispatch("first")
        await transport.dispatch("second")
        events = _drain(transport)
        assert [(t.data["origin"], t.data["cause"]) for t in _turn_completes(events)] == [
            ("dispatch", "stopped")]
        assert transport.dispatch_pending is True
        assert client.queries == ["first", "second"]
        client.feed(_result())
        await _settle()
        assert [t.data["origin"] for t in _turn_completes(_drain(transport))] == ["dispatch"]
    finally:
        await _stop(transport)


async def test_blocking_send_returns_the_dispatch_turns_text(tmp_path):
    transport, client = await _transport(str(tmp_path))
    try:
        waiter = asyncio.ensure_future(transport.send("hello"))
        await _settle()
        client.feed(_assistant("Hello from SDK!"), _result(session_id="sdk-session-1"))
        text = await asyncio.wait_for(waiter, timeout=2.0)
        assert text == "Hello from SDK!"
        assert transport.session_id == "sdk-session-1"
        # The events still reach read_stream consumers.
        events = _drain(transport)
        assert _texts(events) == ["Hello from SDK!"]
        assert [t.data["origin"] for t in _turn_completes(events)] == ["dispatch"]
    finally:
        await _stop(transport)


# ---------------------------------------------------------------------------
# §6.3 — background tasks never hold the dispatch (batch 1122: agent tasks
# no longer do either; their end comes back as a self turn)
# ---------------------------------------------------------------------------

async def test_agent_task_does_not_hold_the_dispatch_and_its_continuation_is_a_self_turn(tmp_path):
    transport, client = await _transport(str(tmp_path))
    try:
        await transport.dispatch("go")
        client.feed(_assistant("Workflow started (wf123)."),
                    _task_started(task_id="wf123", task_type="workflow"),
                    _result())
        await _settle()
        tcs = _turn_completes(_drain(transport))
        assert [(t.data["origin"], t.data["cause"]) for t in tcs] == [("dispatch", "success")]
        assert transport.dispatch_pending is False
        assert transport.background_agents == ["d"]

        client.feed(_task_notification(task_id="wf123"),
                    _assistant("Wrote results to agent_output/report.md"),
                    _result())
        await _settle()
        events = _drain(transport)
        assert any("agent_output/report.md" in t for t in _texts(events))
        tcs = _turn_completes(events)
        assert [(t.data["origin"], t.data["cause"]) for t in tcs] == [("self", "success")]
        assert transport.background_agents == []
    finally:
        await _stop(transport)


async def test_local_bash_is_not_awaited_and_its_continuation_is_a_self_turn(tmp_path):
    transport, client = await _transport(str(tmp_path))
    try:
        await transport.dispatch("go")
        client.feed(_assistant("Started the dev server in the background."),
                    _task_started(task_id="bg1", task_type="local_bash"),
                    _result())
        await _settle()
        events = _drain(transport)
        assert [t.data["origin"] for t in _turn_completes(events)] == ["dispatch"]
        assert transport.dispatch_pending is False

        # The shell exits later and the CLI runs a turn on its own: read,
        # emitted, and ended as the worker continuing — not stranded.
        client.feed(_task_notification(task_id="bg1"),
                    _assistant("server process exited"),
                    _result())
        await _settle()
        events = _drain(transport)
        assert _texts(events) == ["server process exited"]
        assert [t.data["origin"] for t in _turn_completes(events)] == ["self"]
    finally:
        await _stop(transport)


async def test_a_never_finishing_agent_task_never_pins_the_dispatch(tmp_path):
    """Was the 1800 s continuation budget: a stuck task held the dispatch
    (and the user's next prompt) until it expired. Now the result closes the
    dispatch at once and the task is only reported as background work."""
    transport, client = await _transport(str(tmp_path))
    try:
        await transport.dispatch("go")
        client.feed(_task_started(task_id="wf9", task_type="workflow"), _result())
        await _settle()
        tcs = _turn_completes(_drain(transport))
        assert [(t.data["origin"], t.data["cause"]) for t in tcs] == [("dispatch", "success")]
        assert transport.dispatch_pending is False
        assert transport.background_agents == ["d"]
        await transport.dispatch("next")
        assert client.queries == ["go", "next"]
        assert _turn_completes(_drain(transport)) == []  # nothing superseded
    finally:
        await _stop(transport)


# ---------------------------------------------------------------------------
# §6.6 — stop() cancels the reader
# ---------------------------------------------------------------------------

async def test_stop_cancels_the_reader_and_closes_a_pending_dispatch_as_stopped(tmp_path):
    transport, client = await _transport(str(tmp_path))
    reader = transport._reader_task
    assert reader is not None and not reader.done()
    await transport.dispatch("go")

    loop = asyncio.get_running_loop()
    started = loop.time()
    await _stop(transport)
    assert loop.time() - started < transport.READER_EXIT_GRACE_S + 5.0
    assert reader.done()
    assert transport._reader_task is None
    assert transport._alive is False
    assert client.disconnect_called
    tcs = _turn_completes(_drain(transport))
    assert [(t.data["origin"], t.data["cause"]) for t in tcs] == [("dispatch", "stopped")]


async def test_stop_with_nothing_pending_emits_no_boundary(tmp_path):
    transport, client = await _transport(str(tmp_path))
    await _stop(transport)
    assert _turn_completes(_drain(transport)) == []


async def test_start_spawns_the_reader_once(tmp_path):
    transport = SDKTransport()
    client = _ChannelClient()
    with patch("agent_os.agent.transports.sdk_transport.ClaudeSDKClient", return_value=client), \
         patch("agent_os.agent.transports.sdk_transport.ClaudeAgentOptions"):
        client.connect = AsyncMock()
        await transport.start("claude", [], str(tmp_path))
    try:
        reader = transport._reader_task
        assert reader is not None and not reader.done()
        transport._ensure_reader()
        assert transport._reader_task is reader
        # Live even with no dispatch: the whole point.
        client.feed(_assistant("unprompted"), _result())
        await _settle()
        assert _texts(_drain(transport)) == ["unprompted"]
    finally:
        await _stop(transport)
