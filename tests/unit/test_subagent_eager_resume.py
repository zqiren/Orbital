# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for eager resume-record persistence — BACKLOG 005 §4a (SDK only).

The resume record used to land only when a terminal chunk carried
``meta["session_id"]`` (i.e. AFTER the turn completed). For SDKTransport the
upstream session id is available at turn START via the CLI's ``init`` system
message. §4a surfaces it eagerly through a ``thread_started`` event that the
ProcessManager turns into an early ``on_thread_update`` — so a concurrent
@-mention/dispatch in the same chat session finds a resume record mid-turn,
not only once the dispatch returns.

Three layers, each tested honestly:
  1. SDKTransport._message_to_events: init SystemMessage → thread_started event.
  2. transport_event_to_chunk: thread_started keeps its own chunk_type.
  3. ProcessManager.consume: a thread_started chunk fires on_thread_update
     eagerly and is NOT written to the transcript / broadcast.
"""

from __future__ import annotations

import asyncio

from agent_os.agent.transports.sdk_transport import SDKTransport
from agent_os.agent.transports.base import (
    TransportEvent,
    transport_event_to_chunk,
)
from agent_os.daemon_v2.process_manager import ProcessManager


# ---------------------------------------------------------------------------
# Layer 1 — SDKTransport surfaces the init session id eagerly
# ---------------------------------------------------------------------------

def test_init_system_message_emits_thread_started_with_session_id():
    from claude_agent_sdk.types import SystemMessage
    transport = SDKTransport()
    msg = SystemMessage(subtype="init",
                        data={"session_id": "claude-sess-1", "model": "x"})
    events = transport._message_to_events(msg)
    # The session id is captured at turn start (not only at ResultMessage).
    assert transport._session_id == "claude-sess-1"
    started = [e for e in events if e.event_type == "thread_started"]
    assert len(started) == 1
    assert started[0].data["session_id"] == "claude-sess-1"


def test_system_message_without_session_id_emits_nothing():
    from claude_agent_sdk.types import SystemMessage
    transport = SDKTransport()
    events = transport._message_to_events(
        SystemMessage(subtype="compact_boundary", data={}))
    assert events == []
    assert transport._session_id is None


def test_thread_started_emitted_once_per_session_id():
    from claude_agent_sdk.types import SystemMessage
    transport = SDKTransport()
    e1 = transport._message_to_events(
        SystemMessage(subtype="init", data={"session_id": "s1"}))
    e2 = transport._message_to_events(
        SystemMessage(subtype="init", data={"session_id": "s1"}))
    assert sum(ev.event_type == "thread_started" for ev in e1) == 1
    # Same id again → no duplicate thread_started (the guard dedups).
    assert sum(ev.event_type == "thread_started" for ev in e2) == 0


def test_non_init_system_subtype_does_not_emit_thread_started():
    """A SystemMessage that is NOT the init frame (e.g. a compact boundary) must
    not establish the resume identity even if it somehow carried a session_id —
    only subtype == 'init' counts (NIT 1)."""
    from claude_agent_sdk.types import SystemMessage
    transport = SDKTransport()
    events = transport._message_to_events(
        SystemMessage(subtype="compact_boundary",
                      data={"session_id": "leaked"}))
    assert all(e.event_type != "thread_started" for e in events)
    assert transport._session_id is None


def test_task_subclass_message_does_not_emit_thread_started():
    """TaskStarted/TaskProgress/TaskNotification subclass SystemMessage AND
    carry data['session_id'], so a bare isinstance match would let a mid-turn
    task frame masquerade as a thread start. The subtype=='init' gate (NIT 1)
    excludes them."""
    from claude_agent_sdk.types import TaskStartedMessage
    transport = SDKTransport()
    msg = TaskStartedMessage(
        subtype="task_started", data={"session_id": "should-be-ignored"},
        task_id="t1", description="bg work", uuid="u1",
        session_id="should-be-ignored", tool_use_id=None, task_type="agent",
    )
    events = transport._message_to_events(msg)
    assert all(e.event_type != "thread_started" for e in events)
    # The task frame must not hijack the transport's resume identity.
    assert transport._session_id is None


# ---------------------------------------------------------------------------
# Layer 2 — event → chunk mapping keeps thread_started distinct
# ---------------------------------------------------------------------------

def test_thread_started_event_maps_to_thread_started_chunk():
    chunk = transport_event_to_chunk(TransportEvent(
        event_type="thread_started",
        data={"session_id": "s1", "model": "m"}))
    # Must NOT fall through to the "response" default.
    assert chunk.chunk_type == "thread_started"
    assert chunk.metadata["session_id"] == "s1"
    assert chunk.metadata["model"] == "m"


# ---------------------------------------------------------------------------
# Layer 3 — ProcessManager records eagerly, doesn't render the control event
# ---------------------------------------------------------------------------

class _Chunk:
    def __init__(self, text, chunk_type, metadata=None):
        self.text = text
        self.chunk_type = chunk_type
        self.timestamp = "2026-07-01T10:00:00+00:00"
        self.metadata = metadata or {}


class _FakeProc:
    pid = 4242

    def create_time(self):
        return 1000.0


class _Transport:
    def __init__(self, proc=None):
        self._proc = proc


class _Adapter:
    def __init__(self, script, transport=None):
        self._script = script
        self._transport = transport

    async def read_stream(self):
        for c in self._script:
            yield c


class _Transcript:
    def __init__(self):
        self.entries = []
        self.filepath = "/tmp/_eager_resume_test.jsonl"

    def append(self, entry):
        self.entries.append(entry)


class _Lifecycle:
    def __init__(self):
        self.thread_updates = []
        self.completed = []

    async def on_thread_update(self, project_id, handle, *, claude_session_id,
                               model=None, session_id=None, proc_pid=None,
                               proc_create_time=None, rollout_path=None):
        self.thread_updates.append({
            "claude_session_id": claude_session_id, "model": model,
            "session_id": session_id, "proc_pid": proc_pid,
            "proc_create_time": proc_create_time,
        })

    async def on_completed(self, p, h, summary, transcript_path, *, session_id=None):
        self.completed.append(summary)

    async def on_error(self, *a, **k):
        pass


class _WS:
    def __init__(self):
        self.broadcasts = []

    def broadcast(self, project_id, payload):
        self.broadcasts.append(payload)


class _Activity:
    def on_message(self, *a, **k):
        pass


async def _run(script, transport=None):
    ws, lc = _WS(), _Lifecycle()
    pm = ProcessManager(ws, _Activity(), lc)
    tx = _Transcript()
    await pm.start("proj", "claude-code",
                   _Adapter(script, transport=transport),
                   transcript=tx, session_id="sess")
    key = pm._key("proj", "sess", "claude-code")
    await pm._tasks[key]
    return pm, tx, lc, ws


def test_thread_started_records_resume_eagerly_without_turn_complete():
    # A turn still in flight (no turn_complete yet) must already have its
    # resume record persisted — that is the whole point of §4a.
    script = [_Chunk("", "thread_started",
                     {"session_id": "s-eager", "model": "m1"})]
    _pm, tx, lc, ws = asyncio.run(_run(script, transport=_Transport(_FakeProc())))
    assert lc.thread_updates == [{
        "claude_session_id": "s-eager", "model": "m1", "session_id": "sess",
        "proc_pid": 4242, "proc_create_time": 1000.0,
    }]
    # Control event: never written to the transcript, never broadcast.
    assert tx.entries == []
    assert ws.broadcasts == []


def test_thread_started_without_proc_records_with_null_anchor():
    script = [_Chunk("", "thread_started", {"session_id": "s-eager"})]
    _pm, _tx, lc, _ws = asyncio.run(_run(script, transport=_Transport(None)))
    assert len(lc.thread_updates) == 1
    assert lc.thread_updates[0]["proc_pid"] is None
    assert lc.thread_updates[0]["proc_create_time"] is None


def test_eager_then_terminal_both_record_and_turn_renders_normally():
    script = [
        _Chunk("", "thread_started", {"session_id": "s1", "model": "m1"}),
        _Chunk("the answer", "response"),
        _Chunk("", "turn_complete", {"cause": "success", "session_id": "s1",
                                     "model": "m1"}),
    ]
    _pm, tx, lc, ws = asyncio.run(_run(script, transport=_Transport(_FakeProc())))
    # Two records: the eager one (turn start) + the terminal one (turn end).
    assert len(lc.thread_updates) == 2
    assert all(u["claude_session_id"] == "s1" for u in lc.thread_updates)
    # The visible turn is unchanged: response written + boundary appended, but
    # NO thread_started row in the transcript.
    types = [e.get("chunk_type") for e in tx.entries]
    assert types == ["response", "turn_complete"]
    assert lc.completed == ["the answer"]
