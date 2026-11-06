# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 098: a claude-code image Read over the SDK's stdout buffer.

Observed on a live daemon (2026-09-27, before the fix): the Read's tool-result
line (1.2 MB for a 659 KB PNG) overflowed the SDK's 1 MiB default buffer. The
turn closed with the buffer error, but the transport kept reporting alive, so a
follow-up on the same pin went to a client whose message reader was dead: the
dispatch was acknowledged and then nothing ever came back — no reply, no error,
no turn_complete, badge stuck on "running", claude CLI still running.

These tests drive the SDK's real ``Query`` reader through a fake SDK-level
transport, so the fatal/non-fatal split is the SDK's own behaviour: an
exception inside the reader ends it for good; a ``MessageParseError`` is raised
in the consumer's parse step and leaves the reader running.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, CLIJSONDecodeError
from claude_agent_sdk._internal.query import Query

from agent_os.agent.transports.sdk_transport import SDKTransport
from agent_os.daemon_v2.models import make_session_key
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

BUFFER_ERROR = "JSON message exceeded maximum buffer size of 1048576 bytes"


class _ScriptedSDKTransport:
    """SDK-level ``Transport``: yields scripted stdout frames, then either
    raises (a fatal read error) or stays open like a live CLI."""

    def __init__(self, frames, raise_after=None):
        self._frames = frames
        self._raise_after = raise_after
        self.release = asyncio.Event()

    async def connect(self):
        pass

    async def write(self, data):
        pass

    def read_messages(self):
        return self._read()

    async def _read(self):
        for frame in self._frames:
            yield frame
        if self._raise_after is not None:
            raise self._raise_after
        await self.release.wait()

    async def close(self):
        self.release.set()

    def is_ready(self):
        return True

    async def end_input(self):
        pass


async def _transport_over(sdk_transport) -> SDKTransport:
    """An SDKTransport as start() leaves it, over a real Query reader: the
    SDK's stdout reader and Orbital's persistent reader (spec 129) both
    running."""
    query = Query(transport=sdk_transport, is_streaming_mode=True)
    await query.start()
    client = ClaudeSDKClient(options=ClaudeAgentOptions())
    client._transport = sdk_transport
    client._query = query
    transport = SDKTransport()
    transport._client = client
    transport._alive = True
    transport._ensure_reader()
    return transport


async def _reader_settled(transport) -> None:
    """Wait for the persistent reader to end (a fatal stream error)."""
    await asyncio.wait_for(
        asyncio.gather(transport._reader_task, return_exceptions=True), timeout=5.0)


async def _settle(n: int = 30) -> None:
    for _ in range(n):
        await asyncio.sleep(0)


def _drain(transport) -> list:
    events = []
    while not transport._event_queue.empty():
        events.append(transport._event_queue.get_nowait())
    return events


_BUFFER_OVERFLOW = CLIJSONDecodeError(
    BUFFER_ERROR, ValueError("Buffer size 1221219 exceeds limit 1048576"))


class TestFatalReaderError:
    @pytest.mark.asyncio
    async def test_buffer_overflow_emits_error_and_turn_complete_then_not_alive(self):
        sdk = _ScriptedSDKTransport(
            frames=[{"type": "assistant", "message": {
                "model": "claude-fable-5",
                "content": [{"type": "tool_use", "id": "t1", "name": "Read",
                             "input": {"file_path": "memory-context.png"}}]}}],
            raise_after=_BUFFER_OVERFLOW)
        transport = await _transport_over(sdk)

        await transport.dispatch("read the screenshots")
        await _reader_settled(transport)
        events = _drain(transport)

        # The loud part the user already sees stays exactly as it was.
        errors = [e for e in events if e.event_type == "error"]
        assert len(errors) == 1
        assert BUFFER_ERROR in errors[0].raw_text
        assert events[-1].event_type == "turn_complete"
        assert events[-1].data["cause"] == "error"
        assert events[-1].data["origin"] == "dispatch"
        # The fix: a client whose reader is gone no longer claims to be alive,
        # so the manager rebuilds it instead of dispatching into silence.
        assert transport.is_alive() is False

    @pytest.mark.asyncio
    async def test_reader_death_kills_the_claude_process(self):
        # Not-alive must mean the process is gone: list_active() evicts a
        # not-alive adapter WITHOUT stop(), which would otherwise orphan a
        # claude CLI still holding the session.
        transport = await _transport_over(
            _ScriptedSDKTransport(frames=[], raise_after=_BUFFER_OVERFLOW))
        transport._proc = MagicMock(name="psutil.Process")

        with patch("agent_os.agent.transports.process_kill.kill_process_tree",
                   new=AsyncMock()) as kill:
            await transport.dispatch("go")
            await _reader_settled(transport)

        kill.assert_awaited_once()
        assert kill.await_args.args[0] is transport._proc

    @pytest.mark.asyncio
    async def test_process_exit_mid_turn_is_not_alive(self):
        # The CLI dying under the reader (ProcessError / plain EOF) ends the
        # reader the same way — same verdict.
        sdk = _ScriptedSDKTransport(frames=[], raise_after=RuntimeError(
            "Command failed with exit code 1"))
        transport = await _transport_over(sdk)

        await transport.dispatch("go")
        await _reader_settled(transport)

        assert transport.is_alive() is False

    @pytest.mark.asyncio
    async def test_reader_death_between_dispatches_is_a_self_boundary(self):
        # Spec 129 §6.5: nothing is pending, so the boundary is the worker's
        # own turn ending in error — and the verdict is the same: not alive,
        # process killed, rebuilt with resume on the next send.
        transport = await _transport_over(
            _ScriptedSDKTransport(frames=[], raise_after=_BUFFER_OVERFLOW))
        transport._proc = MagicMock(name="psutil.Process")

        with patch("agent_os.agent.transports.process_kill.kill_process_tree",
                   new=AsyncMock()) as kill:
            await _reader_settled(transport)
        events = _drain(transport)

        assert [e.event_type for e in events] == ["error", "turn_complete"]
        assert events[-1].data["cause"] == "error"
        assert events[-1].data["origin"] == "self"
        assert transport.is_alive() is False
        kill.assert_awaited_once()


class _TrickleSDKTransport(_ScriptedSDKTransport):
    """Yields its frames one scheduler tick apart, like a CLI writing lines."""

    async def _read(self):
        for frame in self._frames:
            yield frame
            await asyncio.sleep(0)
        await self.release.wait()


class TestUnsolicitedBacklog:
    """Spec 129 §6.4 / §2.3: the SDK's channel holds 100 messages and its
    reader answers tool permissions inline, so 100 unread messages between
    dispatches freeze the next tool call. The persistent reader keeps the
    channel drained."""

    @pytest.mark.asyncio
    async def test_150_unsolicited_messages_are_emitted_and_the_channel_never_fills(self):
        frames = [{"type": "assistant", "message": {
            "model": "claude-fable-5",
            "content": [{"type": "text", "text": f"self-wake line {i}"}]}}
            for i in range(150)]
        sdk = _TrickleSDKTransport(frames=frames)
        transport = await _transport_over(sdk)
        stats = transport._client._query._message_receive.statistics
        buffered: list[int] = []
        put = transport._event_queue.put

        async def spy_put(event):
            buffered.append(stats().current_buffer_used)
            await put(event)

        transport._event_queue.put = spy_put
        try:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 5.0
            while transport._event_queue.qsize() < 150 and loop.time() < deadline:
                await asyncio.sleep(0.005)
            events = _drain(transport)
            texts = [e.raw_text for e in events if e.event_type == "message"]
            assert texts == [f"self-wake line {i}" for i in range(150)]
            assert max(buffered) < 100, f"SDK channel filled: peak {max(buffered)}"
            assert transport.is_alive() is True
        finally:
            sdk.release.set()
            with patch("agent_os.agent.transports.process_kill.kill_process_tree",
                       new=AsyncMock()):
                await transport.stop()


class TestNonFatalParseError:
    @pytest.mark.asyncio
    async def test_malformed_frame_keeps_the_transport_alive_and_reading(self):
        # An assistant frame missing its content fails parse_message() in the
        # consumer; the SDK reader itself keeps running, so the client is
        # still usable and must NOT be torn down. Spec 129: the persistent
        # reader reports the bad frame and keeps reading — the turn's own
        # ResultMessage still closes the dispatch (no flush-on-next-send).
        sdk = _ScriptedSDKTransport(frames=[
            {"type": "assistant", "message": {}},
            {"type": "assistant", "message": {
                "model": "claude-fable-5",
                "content": [{"type": "text", "text": "after the bad frame"}]}},
            {"type": "result", "subtype": "success", "duration_ms": 1,
             "duration_api_ms": 1, "is_error": False, "num_turns": 1,
             "session_id": "s-parse", "total_cost_usd": 0.0, "usage": {},
             "result": None},
        ])
        transport = await _transport_over(sdk)
        transport._proc = MagicMock(name="psutil.Process")

        with patch("agent_os.agent.transports.process_kill.kill_process_tree",
                   new=AsyncMock()) as kill:
            await transport.dispatch("go")
            await _settle()
        events = _drain(transport)

        assert [e.event_type for e in events] == ["error", "message", "turn_complete"]
        assert events[-1].data["cause"] == "success"
        assert events[-1].data["origin"] == "dispatch"
        assert transport.is_alive() is True
        kill.assert_not_awaited()
        sdk.release.set()
        with patch("agent_os.agent.transports.process_kill.kill_process_tree",
                   new=AsyncMock()):
            await transport.stop()


class TestBufferCeiling:
    @pytest.mark.asyncio
    async def test_options_raise_the_stdout_buffer_above_one_mib(self):
        transport = SDKTransport(resume_session_id="3ab97fd0")
        with patch("agent_os.agent.transports.sdk_transport.ClaudeSDKClient") as MockClient, \
             patch("agent_os.agent.transports.sdk_transport.ClaudeAgentOptions") as MockOptions:
            MockClient.return_value.connect = AsyncMock()
            await transport.start("claude", [], "/workspace")

        kwargs = MockOptions.call_args.kwargs
        assert kwargs["max_buffer_size"] > 1024 * 1024
        assert kwargs["resume"] == "3ab97fd0"


class _Adapter:
    def __init__(self, transport):
        self._transport = transport
        self._idle = True
        self._resume_status = ("resumed", None)

    def is_alive(self):
        return self._transport.is_alive()


class _DispatchRecorder:
    def __init__(self):
        self.dispatched = []

    async def dispatch(self, message):
        self.dispatched.append(message)

    def is_alive(self):
        return True


class TestNextDispatchRebuilds:
    @pytest.mark.asyncio
    async def test_send_to_a_dead_handle_stops_it_and_respawns_with_resume(self):
        dead_transport = await _transport_over(
            _ScriptedSDKTransport(frames=[], raise_after=_BUFFER_OVERFLOW))
        await _reader_settled(dead_transport)
        assert dead_transport.is_alive() is False

        mgr = SubAgentManager(process_manager=MagicMock())
        sk = make_session_key("proj", "sess")
        dead = _Adapter(dead_transport)
        mgr._adapters[sk] = {"claude-code": dead}
        calls = {"stop": [], "start": []}

        async def fake_stop(project_id, handle, *, session_id=None):
            calls["stop"].append(handle)
            mgr._adapters[make_session_key(project_id, session_id)].pop(handle, None)
            return f"Stopped {handle}"

        async def fake_start(project_id, handle, depth=0, *, session_id=None,
                             announce=True, fresh=False, pinned=False, initiator="management_agent"):
            calls["start"].append({"fresh": fresh, "pinned": pinned})
            mgr._adapters[make_session_key(project_id, session_id)][handle] = \
                _Adapter(_DispatchRecorder())
            return "Started claude-code"

        mgr.stop = fake_stop
        mgr.start = fake_start

        result = await mgr.send("proj", "claude-code", "Reply with PONG.",
                                session_id="sess", initiator="user_pinned")

        assert calls["stop"] == ["claude-code"]
        # fresh=False is the resume path: start() consults the persisted
        # thread record and hands its session id to the new transport
        # (test_default_start_consults_determine_resume_and_forwards_record).
        assert calls["start"] == [{"fresh": False, "pinned": True}]
        rebuilt = mgr._adapters[sk]["claude-code"]
        assert rebuilt is not dead
        assert rebuilt._transport.dispatched == ["Reply with PONG."]
        assert result.startswith("Message sent to claude-code")

    @pytest.mark.asyncio
    async def test_send_to_a_live_handle_does_not_rebuild(self):
        mgr = SubAgentManager(process_manager=MagicMock())
        sk = make_session_key("proj", "sess")
        live = _Adapter(_DispatchRecorder())
        mgr._adapters[sk] = {"claude-code": live}
        mgr.stop = AsyncMock()
        mgr.start = AsyncMock()

        await mgr.send("proj", "claude-code", "continue", session_id="sess")

        mgr.stop.assert_not_awaited()
        mgr.start.assert_not_awaited()
        assert live._transport.dispatched == ["continue"]


class TestDeadBeforeTurnComplete:
    @pytest.mark.asyncio
    async def test_not_alive_by_the_time_turn_complete_is_seen(self):
        # The queued-prompt drain runs when the consumer sees turn_complete,
        # so the dead verdict must already hold at that moment.
        transport = await _transport_over(
            _ScriptedSDKTransport(frames=[], raise_after=_BUFFER_OVERFLOW))
        alive_at_turn_complete = []
        put_nowait = transport._event_queue.put_nowait

        def spy_put(event):
            if event.event_type == "turn_complete":
                alive_at_turn_complete.append(transport.is_alive())
            put_nowait(event)

        transport._event_queue.put_nowait = spy_put
        await transport.dispatch("go")
        await _reader_settled(transport)

        assert alive_at_turn_complete == [False]


def _queued(message, n):
    from agent_os.daemon_v2.sub_agent_manager import _QueuedPrompt
    return _QueuedPrompt(message=message, dispatch_id=f"sess:{n}",
                         transcript_path="t.jsonl", initiator="user_pinned")


class _DeadAdapter:
    def __init__(self):
        self._transport = _DispatchRecorder()
        self._idle = True

    def is_alive(self):
        return False


class TestQueuedPromptBehindTheFatalTurn:
    """A message sent while the fatal turn ran sits in the handle's FIFO and
    drains when that turn closes. It must reach a rebuilt client, not the
    dead one."""

    def _mgr_with_queue(self, *prompts):
        from collections import deque
        mgr = SubAgentManager(process_manager=MagicMock())
        key = ("proj", "sess", "claude-code")
        dead = _DeadAdapter()
        mgr._adapters[make_session_key("proj", "sess")] = {"claude-code": dead}
        mgr._prompt_active.add(key)
        mgr._prompt_queues[key] = deque(prompts)
        return mgr, key, dead

    @pytest.mark.asyncio
    async def test_drain_rebuilds_with_resume_then_dispatches_in_order(self):
        mgr, key, dead = self._mgr_with_queue(_queued("first", 1), _queued("second", 2))
        calls = {"stop": 0, "start": []}

        async def fake_stop(project_id, handle, *, session_id=None):
            calls["stop"] += 1
            mgr._prompt_active.discard(key)
            mgr._adapters[make_session_key(project_id, session_id)].pop(handle, None)
            return f"Stopped {handle}"

        async def fake_start(project_id, handle, depth=0, *, session_id=None,
                             announce=True, fresh=False, pinned=False, initiator="management_agent"):
            calls["start"].append({"fresh": fresh, "pinned": pinned})
            mgr._adapters[make_session_key(project_id, session_id)][handle] = \
                _Adapter(_DispatchRecorder())
            return "Started claude-code"

        mgr.stop = fake_stop
        mgr.start = fake_start

        await mgr._on_prompt_turn_closed("proj", "claude-code",
                                         session_id="sess", cause="error")
        await asyncio.gather(*mgr._rebuild_tasks)

        assert dead._transport.dispatched == []
        assert calls["stop"] == 1
        assert calls["start"] == [{"fresh": False, "pinned": True}]
        rebuilt = mgr._adapters[make_session_key("proj", "sess")]["claude-code"]
        assert rebuilt._transport.dispatched == ["first"]
        # The second waits its turn behind the first, as before the death.
        assert [p.message for p in mgr._prompt_queues[key]] == ["second"]
        assert [p.dispatch_id for p in mgr._prompt_queues[key]] == ["sess:2"]

    @pytest.mark.asyncio
    async def test_failed_rebuild_leaves_a_row_for_every_queued_prompt(self):
        mgr, key, dead = self._mgr_with_queue(_queued("first", 1), _queued("second", 2))

        async def fake_stop(project_id, handle, *, session_id=None):
            mgr._adapters[make_session_key(project_id, session_id)].pop(handle, None)
            return f"Stopped {handle}"

        async def fake_start(*a, **k):
            return "Error: adapter start failed: claude not found"

        mgr.stop = fake_stop
        mgr.start = fake_start
        marked = []

        async def spy_mark(dropped, project_id, handle, *, session_id, why):
            marked.append(([p.message for p in dropped or []], why))

        mgr._mark_queued_prompts_dropped = spy_mark

        await mgr._on_prompt_turn_closed("proj", "claude-code",
                                         session_id="sess", cause="error")
        await asyncio.gather(*mgr._rebuild_tasks)

        assert dead._transport.dispatched == []
        assert marked == [(["first", "second"],
                           "the sub-agent was no longer available before dispatch")]
        assert key not in mgr._prompt_queues
        assert key not in mgr._prompt_active
