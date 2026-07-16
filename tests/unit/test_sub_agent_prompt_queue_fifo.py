from types import SimpleNamespace
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

    async def on_message_routed(self, *args, **kwargs):
        self.routed.append((args, kwargs))


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
