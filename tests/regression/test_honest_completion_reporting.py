# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Defect B regression suite: honest sub-agent completion reporting.

Contract (TASK-honest-subagent-completion-reporting):
    No sub-agent ending is reported as ``on_completed`` unless a SUCCESSFUL
    completion signal was received. Successful = ResultMessage present AND
    ``is_error == False`` (SDK). Everything else routes to ``on_error``
    (genuine failure) or emits nothing (deliberate teardown). ``on_failed``
    must reach the management agent's session, not only the WebSocket.

Residual paths from INVESTIGATION-session-binding-resume-and-reap-trigger.md:
    (a) SDK stream exception        → was: on_completed with error as summary
    (b) ResultMessage(is_error=True)→ was: on_completed (the ``got_result``
        trap — a bare boolean ships a fix that still mislabels API errors)
    (c) external process death      → was: on_completed with last narration
    (e) _background_send exception  → was: WebSocket-only, session never told
    (f) pipe timeout / non-zero exit→ was: on_completed("Error: ...")
    (g) deliberate reap             → was: phantom "completed" during the
        SIGTERM grace window (the original production mislabel)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.adapters.cli_adapter import CLIAdapter
from agent_os.agent.transports.base import TransportEvent
from agent_os.agent.transports.sdk_transport import SDKTransport, HAS_SDK
from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver
from agent_os.daemon_v2.process_manager import ProcessManager
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result_message(*, is_error: bool, result=None, session_id="sdk-s-1"):
    from claude_agent_sdk.types import ResultMessage

    return ResultMessage(
        subtype="error" if is_error else "success",
        duration_ms=100, duration_api_ms=80,
        is_error=is_error, num_turns=1, session_id=session_id,
        total_cost_usd=0.01, usage={}, result=result, structured_output=None,
    )


async def _drain_queue(transport) -> list[TransportEvent]:
    events = []
    while True:
        try:
            events.append(transport._event_queue.get_nowait())
        except asyncio.QueueEmpty:
            return events


def _turn_completes(events):
    return [e for e in events if e.event_type == "turn_complete"]


class FakeSDKTransport:
    """Queue-backed transport stub (same pattern as
    test_sdk_completion_notification.py). Drains queued events after
    ``_alive`` goes False so pre-queued events are always delivered."""

    def __init__(self):
        self._event_queue: asyncio.Queue[TransportEvent] = asyncio.Queue()
        self._alive = True

    async def start(self, command, args, workspace, env=None):
        pass

    async def send(self, message):
        return None

    async def dispatch(self, message):
        pass

    async def read_stream(self):
        while True:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=0.2)
                yield event
            except asyncio.TimeoutError:
                if not self._alive:
                    break
                continue

    async def stop(self):
        self._alive = False

    def is_alive(self):
        return self._alive

    @property
    def session_id(self):
        return None


def _wire_consumer(transport):
    """Real ProcessManager + mock observer wired to a CLIAdapter over
    ``transport``. Returns (process_manager, observer, adapter, transcript)."""
    adapter = CLIAdapter(handle="test-agent", display_name="Test Agent",
                         transport=transport)
    observer = MagicMock()
    observer.on_completed = AsyncMock()
    observer.on_error = AsyncMock()
    ws = MagicMock()
    activity = MagicMock()
    pm = ProcessManager(ws_manager=ws, activity_translator=activity,
                        lifecycle_observer=observer)
    transcript = MagicMock()
    transcript.filepath = "/tmp/fake_transcript.jsonl"
    transcript.append = MagicMock()
    return pm, observer, adapter, transcript


async def _run_consumer(pm, adapter, transcript):
    await pm.start("proj_x", "test-agent", adapter, transcript=transcript,
                   session_id="sess_x")
    await asyncio.wait_for(pm._tasks["proj_x:sess_x:test-agent"], timeout=5.0)  # session-scoped key (Part E)


# ---------------------------------------------------------------------------
# Transport level: turn_complete must carry an honest cause
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")
class TestSDKTransportCause:

    async def _dispatch_and_drain(self, receive_factory):
        transport = SDKTransport()
        mock_client = AsyncMock()
        mock_client.receive_response = receive_factory
        transport._client = mock_client
        transport._alive = True
        await transport.dispatch("go")
        await asyncio.gather(transport._bg_task, return_exceptions=True)
        return await _drain_queue(transport)

    async def test_success_turn_complete_carries_cause_success(self):
        """Control: a clean ResultMessage(is_error=False) → cause=success."""
        ok = _make_result_message(is_error=False)

        async def receive():
            yield ok

        events = await self._dispatch_and_drain(lambda: receive())
        tcs = _turn_completes(events)
        assert len(tcs) == 1
        assert tcs[0].data.get("cause") == "success"

    async def test_is_error_result_message_yields_cause_error(self):
        """Path (b) — THE trap: got_result is True but is_error is True.

        A naive ``got_result`` boolean fix ships a Defect B fix that still
        mislabels API errors (context-window-exceeded etc.) as completions.
        """
        bad = _make_result_message(is_error=True, result="context window exceeded")

        async def receive():
            yield bad

        events = await self._dispatch_and_drain(lambda: receive())
        tcs = _turn_completes(events)
        assert len(tcs) == 1
        assert tcs[0].data.get("cause") == "error", (
            "ResultMessage(is_error=True) must NOT yield a success cause"
        )

    async def test_stream_exception_yields_cause_error(self):
        """Path (a): receive_response() raises mid-stream."""

        async def receive():
            raise RuntimeError("network dropped")
            yield  # pragma: no cover

        events = await self._dispatch_and_drain(lambda: receive())
        assert any(e.event_type == "error" for e in events)
        tcs = _turn_completes(events)
        assert len(tcs) == 1
        assert tcs[0].data.get("cause") == "error"

    async def test_stream_end_without_result_yields_cause_error(self):
        """Path (c): stream ends with no ResultMessage (process died)."""

        async def receive():
            return
            yield  # pragma: no cover

        events = await self._dispatch_and_drain(lambda: receive())
        tcs = _turn_completes(events)
        assert len(tcs) == 1
        assert tcs[0].data.get("cause") == "error"

    async def test_cancelled_bg_task_yields_cause_stopped(self):
        """Path (g), transport half: a deliberate stop (bg-task cancel) must
        tag its turn_complete as stopped so the consumer does not report it.
        """
        transport = SDKTransport()
        mock_client = AsyncMock()

        async def hang():
            await asyncio.sleep(30)
            yield  # pragma: no cover

        mock_client.receive_response = lambda: hang()
        transport._client = mock_client
        transport._alive = True
        await transport.dispatch("go")
        await asyncio.sleep(0.05)
        transport._bg_task.cancel()
        await asyncio.gather(transport._bg_task, return_exceptions=True)

        events = await _drain_queue(transport)
        tcs = _turn_completes(events)
        assert len(tcs) == 1
        assert tcs[0].data.get("cause") == "stopped"


# ---------------------------------------------------------------------------
# Consumer level: ProcessManager routes by cause
# ---------------------------------------------------------------------------


class TestConsumerRouting:

    async def test_success_cause_routes_to_on_completed_exactly_once(self):
        """Success completes once — and the stream-end boundary after a closed
        turn must NOT fire a second (phantom) on_completed."""
        transport = FakeSDKTransport()
        pm, observer, adapter, transcript = _wire_consumer(transport)
        await transport._event_queue.put(
            TransportEvent("message", data={"text": "hello"}, raw_text="hello"))
        await transport._event_queue.put(
            TransportEvent("turn_complete", data={"cause": "success"}))
        await transport.stop()
        await _run_consumer(pm, adapter, transcript)

        assert observer.on_completed.await_count == 1, (
            f"expected exactly one on_completed, got "
            f"{observer.on_completed.await_count} (stream-end double-fire?)"
        )
        assert "hello" in observer.on_completed.await_args.kwargs.get(
            "summary", observer.on_completed.await_args.args[2]
            if len(observer.on_completed.await_args.args) > 2 else "")
        observer.on_error.assert_not_awaited()

    async def test_error_cause_routes_to_on_error(self):
        """Paths (a)/(b)/(c) consumer half: error cause → on_error, never
        on_completed — and the error text must not become a 'summary'."""
        transport = FakeSDKTransport()
        pm, observer, adapter, transcript = _wire_consumer(transport)
        await transport._event_queue.put(
            TransportEvent("error", data={"error": "context window exceeded"},
                           raw_text="Error: context window exceeded"))
        await transport._event_queue.put(
            TransportEvent("turn_complete", data={"cause": "error"}))
        await transport.stop()
        await _run_consumer(pm, adapter, transcript)

        observer.on_completed.assert_not_awaited()
        assert observer.on_error.await_count == 1
        err_text = str(observer.on_error.await_args)
        assert "context window exceeded" in err_text

    async def test_stopped_cause_emits_nothing(self):
        """Path (g): deliberate reap → no lifecycle event at all. This is the
        phantom 'completed' from the original production incident."""
        transport = FakeSDKTransport()
        pm, observer, adapter, transcript = _wire_consumer(transport)
        await transport._event_queue.put(
            TransportEvent("message", data={"text": "working on it"},
                           raw_text="working on it"))
        await transport._event_queue.put(
            TransportEvent("turn_complete", data={"cause": "stopped"}))
        await transport.stop()
        await _run_consumer(pm, adapter, transcript)

        observer.on_completed.assert_not_awaited()
        observer.on_error.assert_not_awaited()

    async def test_bare_turn_complete_is_not_reported_completed(self):
        """A turn_complete with no cause is not a verified success — the
        contract forbids reporting it as on_completed."""
        transport = FakeSDKTransport()
        pm, observer, adapter, transcript = _wire_consumer(transport)
        await transport._event_queue.put(TransportEvent("turn_complete", data={}))
        await transport.stop()
        await _run_consumer(pm, adapter, transcript)

        observer.on_completed.assert_not_awaited()

    async def test_stream_end_mid_turn_routes_to_on_error(self):
        """Path (c) consumer half: activity then stream death with no
        turn_complete at all → honest on_error, not on_completed."""
        transport = FakeSDKTransport()
        pm, observer, adapter, transcript = _wire_consumer(transport)
        await transport._event_queue.put(
            TransportEvent("message", data={"text": "let me try"},
                           raw_text="let me try"))
        await transport.stop()
        await _run_consumer(pm, adapter, transcript)

        observer.on_completed.assert_not_awaited()
        assert observer.on_error.await_count == 1

    async def test_stream_end_with_no_activity_emits_nothing(self):
        """Teardown of a never-dispatched / fully-idle adapter is silent.
        (Pre-fix: pipe adapters injected '[Sub-agent] completed. Summary:
        (no output)' at startup because their read_stream is empty.)"""
        transport = FakeSDKTransport()
        pm, observer, adapter, transcript = _wire_consumer(transport)
        await transport.stop()
        await _run_consumer(pm, adapter, transcript)

        observer.on_completed.assert_not_awaited()
        observer.on_error.assert_not_awaited()


# ---------------------------------------------------------------------------
# Path (f): pipe/ACP background-send error strings are not summaries
# ---------------------------------------------------------------------------


def _make_sub_agent_manager(observer):
    pm = MagicMock()
    pm._ws = MagicMock()
    mgr = SubAgentManager(process_manager=pm, lifecycle_observer=observer)
    return mgr


class _BlockingAdapter:
    """Adapter without a dispatch-capable transport → _background_send path."""

    def __init__(self, response):
        self._transport = None
        self._response = response
        self._last_response = None
        self._background_send_task = None
        self._broken = False

    async def send(self, message):
        self._last_response = self._response


class TestBackgroundSendRouting:

    async def _run(self, response):
        observer = MagicMock()
        observer.on_completed = AsyncMock()
        observer.on_error = AsyncMock()
        mgr = _make_sub_agent_manager(observer)
        transcript = MagicMock()
        transcript.filepath = "/tmp/fake_transcript.jsonl"
        transcript.append = MagicMock()
        mgr._transcripts[("proj_x", "sess_x", "pipe-agent")] = transcript  # session-scoped (Part E)
        adapter = _BlockingAdapter(response)
        await mgr._dispatch_async(adapter, "proj_x", "pipe-agent", "go",
                                  session_id="sess_x")
        await asyncio.wait_for(adapter._background_send_task, timeout=5.0)
        return observer

    async def test_pipe_error_string_routes_to_on_error(self):
        """Path (f): 'Error: sub-agent timed out after 5 minutes' must route
        to on_error — it is a failure, not a completion summary."""
        observer = await self._run("Error: sub-agent timed out after 5 minutes")
        observer.on_completed.assert_not_awaited()
        assert observer.on_error.await_count == 1
        assert "timed out" in str(observer.on_error.await_args)

    async def test_pipe_success_response_routes_to_on_completed(self):
        """Control: a normal pipe response still completes."""
        observer = await self._run("All done. Wrote radar/weekly/week-0.md")
        observer.on_error.assert_not_awaited()
        assert observer.on_completed.await_count == 1


# ---------------------------------------------------------------------------
# Path (e) / fix 5: on_failed reaches the management session
# ---------------------------------------------------------------------------


class TestOnFailedInjection:

    async def test_background_send_failure_reaches_management_session(self):
        """Integration (path e, full chain): a real SubAgentManager whose
        _background_send raises, wired to a REAL LifecycleObserver — the
        management session must receive the failure injection. Mock only at
        the leaves (agent_manager, ws)."""
        mock_agent_mgr = MagicMock()
        mock_agent_mgr.inject_system_message = AsyncMock(return_value="delivered")
        ws = MagicMock()
        observer = LifecycleObserver(agent_manager=mock_agent_mgr, ws_manager=ws)
        mgr = _make_sub_agent_manager(observer)
        transcript = MagicMock()
        transcript.filepath = "/tmp/fake_transcript.jsonl"
        mgr._transcripts[("proj_x", "claude-code")] = transcript

        class _ExplodingAdapter:
            _transport = None
            _last_response = None
            _background_send_task = None
            _broken = False

            async def send(self, message):
                raise RuntimeError("synthetic transport failure")

        adapter = _ExplodingAdapter()
        await mgr._dispatch_async(adapter, "proj_x", "claude-code", "go",
                                  session_id="sess_x")
        await asyncio.wait_for(adapter._background_send_task, timeout=5.0)

        assert mock_agent_mgr.inject_system_message.await_count == 1
        call = mock_agent_mgr.inject_system_message.await_args
        assert "failed" in call.args[1].lower()
        assert call.kwargs.get("session_id") == "sess_x"
        assert adapter._broken is True

    async def test_on_failed_injects_into_management_session(self):
        mock_agent_mgr = MagicMock()
        mock_agent_mgr.inject_system_message = AsyncMock(return_value="delivered")
        ws = MagicMock()
        observer = LifecycleObserver(agent_manager=mock_agent_mgr, ws_manager=ws)

        result = observer.on_failed("proj_x", "claude-code",
                                    reason="background_send_exception",
                                    session_id="sess_x")
        if asyncio.iscoroutine(result):
            await result

        assert mock_agent_mgr.inject_system_message.await_count == 1, (
            "on_failed must inject a system message into the management "
            "session — WebSocket-only means the management agent is never "
            "told the dispatch failed (path e)."
        )
        call = mock_agent_mgr.inject_system_message.await_args
        content = call.args[1]
        assert "claude-code" in content
        assert "fail" in content.lower()
        assert call.kwargs.get("session_id") == "sess_x"
        # WS broadcast must be preserved.
        assert any(c.args[1].get("type") == "sub_agent.failed"
                   for c in ws.broadcast.call_args_list)
