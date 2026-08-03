from pathlib import Path
from types import SimpleNamespace
import json
import time
from unittest.mock import MagicMock

import pytest

from agent_os.agent.adapters.base import OutputChunk
from agent_os.daemon_v2.models import make_session_key
from agent_os.daemon_v2.process_manager import ProcessManager
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager


class _ProcessManager:
    def __init__(self):
        self.turn_closed = None
        self.permission = None
        self.dispatch_ids = []
        self.cleared = []

    def set_turn_closed_callback(self, callback):
        self.turn_closed = callback

    def set_permission_request_callback(self, callback):
        self.permission = callback

    def set_active_dispatch(self, project_id, handle, dispatch_id, **kwargs):
        self.dispatch_ids.append(dispatch_id)

    def clear_dispatch(self, project_id, handle, dispatch_id, **kwargs):
        self.cleared.append(dispatch_id)

    async def stop(self, *args, **kwargs):
        return None


class _Transport:
    def __init__(self):
        self.messages = []

    async def dispatch(self, message):
        self.messages.append(message)


class _Lifecycle:
    def __init__(self):
        self.routed = []
        self.failed = []
        # backlog #35a: dropped queued prompts no longer borrow on_failed —
        # they have their own amber shape, and keeping the two lists apart is
        # the point (a drop must not be counted as a failure).
        self.dropped = []

    async def on_message_routed(self, *args, **kwargs):
        self.routed.append((args, kwargs))

    async def on_failed(self, project_id, handle, reason, **kwargs):
        self.failed.append((project_id, handle, reason, kwargs))

    async def on_queue_dropped(self, project_id, handle, *, why, **kwargs):
        self.dropped.append((project_id, handle, why, kwargs))


@pytest.mark.asyncio
async def test_sub_agent_prompt_queue_drains_fifo_only_at_turn_boundary():
    pm = _ProcessManager()
    lifecycle = _Lifecycle()
    manager = SubAgentManager(pm, lifecycle_observer=lifecycle)
    transport = _Transport()
    adapter = SimpleNamespace(_transport=transport, _broken=False)
    manager._adapters[make_session_key("p1", "s1")] = {"cursor": adapter}

    first = await manager.send(
        "p1", "cursor", "first", session_id="s1", dispatch_id="d1")
    second = await manager.send(
        "p1", "cursor", "second", session_id="s1", dispatch_id="d2")
    third = await manager.send(
        "p1", "cursor", "third", session_id="s1", dispatch_id="d3")

    assert "Message sent" in first
    assert "position 1" in second
    assert "position 2" in third
    assert transport.messages == ["first"]
    assert pm.dispatch_ids == ["d1"]

    await pm.turn_closed("p1", "cursor", session_id="s1", cause="success")
    assert transport.messages == ["first", "second"]
    assert pm.dispatch_ids == ["d1", "d2"]

    await pm.turn_closed("p1", "cursor", session_id="s1", cause="success")
    assert transport.messages == ["first", "second", "third"]
    assert pm.dispatch_ids == ["d1", "d2", "d3"]
    assert len(lifecycle.routed) == 3


@pytest.mark.asyncio
async def test_mock_adapter_without_explicit_broken_flag_is_dispatchable():
    pm = _ProcessManager()
    manager = SubAgentManager(pm)
    adapter = MagicMock()
    adapter._transport = _Transport()
    manager._adapters[make_session_key("p1", "s1")] = {"cursor": adapter}

    result = await manager.send("p1", "cursor", "work", session_id="s1")

    assert result.startswith("Message sent")
    assert adapter._transport.messages == ["work"]


@pytest.mark.asyncio
async def test_queued_followup_does_not_clear_temporary_permission_bypass():
    pm = _ProcessManager()
    manager = SubAgentManager(pm)
    transport = _Transport()
    manager._adapters[make_session_key("p1", "s1")] = {
        "cursor": SimpleNamespace(_transport=transport, _broken=False)}
    key = ("p1", "s1", "cursor")

    await manager.send("p1", "cursor", "first", session_id="s1")
    expiry = time.monotonic() + 600
    manager._permission_bypass_until[key] = expiry
    await manager.send("p1", "cursor", "queued", session_id="s1")

    assert manager._permission_bypass_until[key] == expiry


@pytest.mark.asyncio
async def test_abnormal_terminal_drops_queue_and_marks_adapter_broken():
    pm = _ProcessManager()
    manager = SubAgentManager(pm)
    adapter = SimpleNamespace(_transport=_Transport(), _broken=False)
    manager._adapters[make_session_key("p1", "s1")] = {"cursor": adapter}
    key = ("p1", "s1", "cursor")
    manager._prompt_active.add(key)
    manager._prompt_queues[key] = __import__("collections").deque([
        SimpleNamespace(message="must-not-run")])

    await manager._on_prompt_turn_closed(
        "p1", "cursor", session_id="s1", cause="stream_ended")

    assert adapter._broken is True
    assert key not in manager._prompt_active
    assert key not in manager._prompt_queues
    assert adapter._transport.messages == []
    result = await manager.send("p1", "cursor", "later", session_id="s1")
    assert result.startswith("Error: agent 'cursor' transport is broken")
    assert adapter._transport.messages == []


@pytest.mark.asyncio
async def test_stop_clears_active_and_waiting_prompts(monkeypatch):
    pm = _ProcessManager()
    manager = SubAgentManager(pm)
    key = ("p1", "s1", "cursor")
    manager._prompt_active.add(key)
    manager._prompt_queues[key] = __import__("collections").deque([
        SimpleNamespace(message="later")])

    async def _stopped(*args, **kwargs):
        return "stopped"

    monkeypatch.setattr(manager, "_kill_confirm_and_release", _stopped)
    result = await manager.stop("p1", "cursor", session_id="s1")

    assert result == "Stopped cursor"
    assert key not in manager._prompt_active
    assert key not in manager._prompt_queues


class TestDroppedQueuedPromptsLeaveMarkers:
    """backlog #26d — a queued-then-dropped @mention left zero trace.

    The mention's own user message is persisted up front (by
    ``persist_mention_message``), but its dispatch marker is only written when
    the prompt actually fires. Every drop path therefore stranded a "You ->
    handle" bubble with nothing after it. Each drop must now leave a durable
    marker row explaining what became of the message.

    backlog #35a moved these rows off ``on_failed`` onto their own
    ``on_queue_dropped`` shape, so the assertions below check ``dropped`` and
    deliberately check that ``failed`` stays empty: nothing was dispatched and
    nothing malfunctioned, and a drop reported as a failure is the exact lie
    the borrowed shape told.
    """

    @staticmethod
    def _queue(manager, key, *messages):
        from collections import deque
        manager._prompt_queues[key] = deque(
            SimpleNamespace(message=m) for m in messages)

    @pytest.mark.asyncio
    async def test_stop_marks_each_dropped_queued_prompt(self, monkeypatch):
        pm = _ProcessManager()
        lifecycle = _Lifecycle()
        manager = SubAgentManager(pm, lifecycle_observer=lifecycle)
        key = ("p1", "s1", "cursor")
        manager._prompt_active.add(key)
        self._queue(manager, key, "later", "later-still")

        async def _stopped(*args, **kwargs):
            return "stopped"

        monkeypatch.setattr(manager, "_kill_confirm_and_release", _stopped)
        await manager.stop("p1", "cursor", session_id="s1")

        # One marker per dropped message — each has its own orphaned user
        # bubble upstream, so an aggregate row would leave one unexplained.
        assert len(lifecycle.dropped) == 2
        for project_id, handle, why, kwargs in lifecycle.dropped:
            assert (project_id, handle) == ("p1", "cursor")
            assert kwargs["session_id"] == "s1"
            # A user-initiated stop is not a malfunction: the text has to read
            # as an explanation, not an error.
            assert why == "agent stopped before dispatch"
        assert lifecycle.failed == []

    @pytest.mark.asyncio
    async def test_stop_with_empty_queue_marks_nothing(self, monkeypatch):
        """Stopping an idle handle must not manufacture a phantom failure."""
        pm = _ProcessManager()
        lifecycle = _Lifecycle()
        manager = SubAgentManager(pm, lifecycle_observer=lifecycle)

        async def _stopped(*args, **kwargs):
            return "stopped"

        monkeypatch.setattr(manager, "_kill_confirm_and_release", _stopped)
        await manager.stop("p1", "cursor", session_id="s1")

        assert lifecycle.dropped == []
        assert lifecycle.failed == []

    @pytest.mark.asyncio
    async def test_abnormal_terminal_marks_dropped_queue(self):
        pm = _ProcessManager()
        lifecycle = _Lifecycle()
        manager = SubAgentManager(pm, lifecycle_observer=lifecycle)
        adapter = SimpleNamespace(_transport=_Transport(), _broken=False)
        manager._adapters[make_session_key("p1", "s1")] = {"cursor": adapter}
        key = ("p1", "s1", "cursor")
        manager._prompt_active.add(key)
        self._queue(manager, key, "must-not-run")

        await manager._on_prompt_turn_closed(
            "p1", "cursor", session_id="s1", cause="stream_ended")

        assert adapter._broken is True
        assert key not in manager._prompt_queues
        assert len(lifecycle.dropped) == 1
        assert lifecycle.dropped[0][2] == "transport ended before dispatch"
        assert lifecycle.failed == []

    @pytest.mark.asyncio
    async def test_broken_adapter_at_boundary_marks_dropped_queue(self):
        """The whole waiting FIFO is discarded when the adapter is gone or
        already broken — every entry gets its own row."""
        pm = _ProcessManager()
        lifecycle = _Lifecycle()
        manager = SubAgentManager(pm, lifecycle_observer=lifecycle)
        adapter = SimpleNamespace(_transport=_Transport(), _broken=True)
        manager._adapters[make_session_key("p1", "s1")] = {"cursor": adapter}
        key = ("p1", "s1", "cursor")
        manager._prompt_active.add(key)
        self._queue(manager, key, "one", "two")

        await manager._on_prompt_turn_closed(
            "p1", "cursor", session_id="s1", cause="success")

        assert key not in manager._prompt_queues
        assert len(lifecycle.dropped) == 2
        assert lifecycle.dropped[0][2] == (
            "the sub-agent was no longer available before dispatch")
        assert lifecycle.failed == []

    @pytest.mark.asyncio
    async def test_background_send_exception_marks_dropped_queue(self):
        """A blocking-transport send that raises pops the waiting FIFO — the
        send itself already gets on_failed (it really did fail), but the
        prompts queued behind it never dispatched, so they get the dropped
        shape instead. Both markers, each honest about its own event."""
        pm = _ProcessManager()
        lifecycle = _Lifecycle()
        manager = SubAgentManager(pm, lifecycle_observer=lifecycle)

        async def _boom(message):
            raise RuntimeError("send exploded")

        # No _transport with dispatch() -> send() takes the blocking
        # _background_send path.
        adapter = SimpleNamespace(_broken=False, send=_boom)
        manager._adapters[make_session_key("p1", "s1")] = {"cursor": adapter}

        await manager.send("p1", "cursor", "first", session_id="s1",
                           dispatch_id="d1")
        await manager.send("p1", "cursor", "waiting", session_id="s1",
                           dispatch_id="d2")
        await adapter._background_send_task

        key = ("p1", "s1", "cursor")
        assert key not in manager._prompt_queues
        assert "background_send_exception" in [e[2] for e in lifecycle.failed]
        assert ("a prior send failed before dispatch"
                in [e[2] for e in lifecycle.dropped])

    @pytest.mark.asyncio
    async def test_normal_drain_marks_nothing(self):
        """A prompt that actually dispatches gets a routed marker and must
        NOT also be reported as dropped."""
        pm = _ProcessManager()
        lifecycle = _Lifecycle()
        manager = SubAgentManager(pm, lifecycle_observer=lifecycle)
        transport = _Transport()
        adapter = SimpleNamespace(_transport=transport, _broken=False)
        manager._adapters[make_session_key("p1", "s1")] = {"cursor": adapter}

        await manager.send("p1", "cursor", "first", session_id="s1",
                           dispatch_id="d1")
        await manager.send("p1", "cursor", "second", session_id="s1",
                           dispatch_id="d2")
        await pm.turn_closed("p1", "cursor", session_id="s1", cause="success")

        assert transport.messages == ["first", "second"]
        assert lifecycle.dropped == []
        assert lifecycle.failed == []
        assert len(lifecycle.routed) == 2


@pytest.mark.asyncio
async def test_dropped_queued_mention_lands_a_durable_row_in_session_jsonl(tmp_path):
    """End-to-end (backlog #26d): the user-visible bug is an empty timeline,
    so assert the marker survives all the way to the session JSONL on disk —
    through the REAL LifecycleObserver, not just an observer spy.

    The agent-manager double reproduces ``inject_system_message``'s
    live-handle branch verbatim (role/content/source/_meta appended to the
    session); everything upstream of it is production code.
    """
    from agent_os.agent.session import Session
    from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver

    session = Session.new("proj_abcd1234", str(tmp_path), "p1",
                          session_id="s1")

    class _InjectingAgentManager:
        async def inject_system_message(self, project_id, content, **kwargs):
            record = {"role": "system", "content": content, "source": "daemon"}
            if kwargs.get("meta"):
                record["_meta"] = kwargs["meta"]
            session.append(record)

    class _WS:
        def broadcast(self, project_id, payload):
            pass

    observer = LifecycleObserver(_InjectingAgentManager(), _WS())
    manager = SubAgentManager(_ProcessManager(), lifecycle_observer=observer)
    transport = _Transport()
    manager._adapters[make_session_key("p1", "s1")] = {
        "cursor": SimpleNamespace(_transport=transport, _broken=False)}

    # An @mention that dispatches, then a second one that queues behind it.
    await manager.send("p1", "cursor", "first", session_id="s1",
                       dispatch_id="d1", initiator="user_mention")
    queued = await manager.send("p1", "cursor", "the dropped one",
                                session_id="s1", dispatch_id="d2",
                                initiator="user_mention")
    assert "position 1" in queued

    # Transport dies with no honest boundary — the queued mention is dropped.
    await manager._on_prompt_turn_closed(
        "p1", "cursor", session_id="s1", cause="stream_ended")

    rows = [
        json.loads(line)
        for line in Path(session._filepath).read_text().splitlines()
        if line.strip()
    ]
    dropped_rows = [
        r for r in rows
        if "queued message dropped" in str(r.get("content", ""))
    ]
    assert len(dropped_rows) == 1, (
        f"expected exactly one dropped-mention marker, got rows: {rows}")
    row = dropped_rows[0]
    assert row["role"] == "system"
    # backlog #35a: its own shape now, matched by chatTransform's
    # SUB_AGENT_QUEUE_DROPPED_RE. No "failed:" and no second colon, because
    # nothing was dispatched and nothing malfunctioned.
    assert row["content"] == (
        "[Sub-agent] cursor queued message dropped: transport ended before "
        "dispatch.")
    assert "failed" not in row["content"]
    assert "The dispatched task did not complete" not in row["content"]
    # The wake tag survives the reshape: a manager awaiting this very message
    # has to hear that it is never coming (backlog #23 D1).
    assert row["_meta"]["event"] == "sub_agent_terminal"
    assert row["_meta"]["kind"] == "queue_dropped"
    assert "transport ended before dispatch" in row["content"]


def test_acp_sdk_manifest_resolves_streaming_transport():
    manager = SubAgentManager(_ProcessManager())
    manifest = SimpleNamespace(
        slug="cursor",
        runtime=SimpleNamespace(
            transport="acp-sdk", mode="pipe", command="agent"),
    )

    transport = manager._resolve_transport(
        manifest,
        {"args": ["acp", "--orbital-permission-mode", "ask"]},
        resume_record={"session_id": "resume-1"},
    )

    assert type(transport).__name__ == "ACPSDKTransport"
    assert transport._resume_session_id == "resume-1"


def test_acp_resume_candidate_reaches_transport_and_provider_outcome_wins():
    pm = _ProcessManager()
    manifest = SimpleNamespace(
        slug="cursor",
        runtime=SimpleNamespace(
            transport="acp-sdk", mode="pipe", command="agent"),
    )
    manager = SubAgentManager(
        pm, registry=SimpleNamespace(get=lambda handle: manifest))
    record = {"session_id": "cursor-session-7"}
    session = SimpleNamespace(get_sub_agent_thread=lambda handle: record)
    manager._session_resolver = lambda project_id, session_id: session

    candidate, status, reason = manager._determine_resume(
        "/workspace", "p1", "cursor", "s1")
    transport = manager._resolve_transport(
        manifest, {"args": ["acp"]}, resume_record=candidate)

    assert candidate == record
    assert transport._resume_session_id == "cursor-session-7"

    transport._resume_outcome = ("fresh", "resume_failed")
    assert manager._provider_confirmed_resume_outcome(
        candidate, status, reason, transport) == ("fresh", "resume_failed")
    transport._resume_outcome = ("resumed", None)
    assert manager._provider_confirmed_resume_outcome(
        candidate, status, reason, transport) == ("resumed", None)


class _WS:
    def broadcast(self, *args, **kwargs):
        pass


class _Activity:
    def on_message(self, *args, **kwargs):
        pass


class _StreamEndsWithoutBoundary:
    _transport = None

    async def read_stream(self):
        yield OutputChunk(text="partial", chunk_type="response")


class _ConsumerRaises:
    _transport = None

    async def read_stream(self):
        if False:
            yield None
        raise RuntimeError("consumer exploded")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("adapter", "cause"),
    [(_StreamEndsWithoutBoundary(), "stream_ended"),
     (_ConsumerRaises(), "consumer_exception")],
)
async def test_process_manager_reports_abnormal_terminal_to_queue_owner(
    adapter, cause,
):
    pm = ProcessManager(_WS(), _Activity())
    terminal = []

    async def on_closed(project_id, handle, **kwargs):
        terminal.append(kwargs["cause"])

    pm.set_turn_closed_callback(on_closed)
    await pm.start("p1", "cursor", adapter, session_id="s1")
    await pm._tasks[pm._key("p1", "s1", "cursor")]

    assert terminal == [cause]
