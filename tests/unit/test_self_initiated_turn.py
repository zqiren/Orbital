# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 129 §5B/§5C: a worker turn nobody dispatched is its own event.

The claude-code CLI runs a turn on its own when a ``run_in_background``
shell exits. The transport now reads it live and ends it with
``turn_complete(origin="self")``. The consume loop must treat that end as
the worker *continuing*: no dispatch id popped (none originated it — popping
would desync the FIFO), no ``on_completed``, a ``continued`` row in the
management session, and the usual bookkeeping (resume identity, transcript
boundary, turn-closed hook) so transcript slices and eviction stay sane.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.adapters.base import OutputChunk
from agent_os.daemon_v2.background_work import BackgroundWorkRegistry
from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver
from agent_os.daemon_v2.models import make_session_key
from agent_os.daemon_v2.process_manager import ProcessManager
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

PID = "proj_self"
SID = "sess_self"
HANDLE = "claude-code"
KEY = f"{PID}:{SID}:{HANDLE}"


def _chunk(chunk_type: str, text: str = "", **meta) -> OutputChunk:
    """``_pending`` (test-only) is what the transport's ``dispatch_pending``
    reads while this chunk is processed: True for a dispatched turn's
    chunks, False (the default) for a self-initiated turn's."""
    return OutputChunk(text=text, chunk_type=chunk_type, metadata=meta)


def _turn_complete(origin: str | None, cause: str = "success") -> OutputChunk:
    meta = {"cause": cause, "session_id": "cli-sid-1", "model": "claude-x"}
    if origin is not None:
        meta["origin"] = origin
    return _chunk("turn_complete", **meta)


class _Adapter:
    """A CLIAdapter stand-in: idle until a dispatch, idle again at the
    boundary (read_stream flips ``_idle`` back exactly as CLIAdapter does)."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self._transport = MagicMock()
        self._transport._proc = None
        self._transport.dispatch_pending = False
        self._idle = True
        self.idle_seen: list[bool] = []

    async def read_stream(self):
        for chunk in self._chunks:
            pending = bool(chunk.metadata.pop("_pending", False))
            if isinstance(self._transport, MagicMock):
                self._transport.dispatch_pending = pending
            if chunk.chunk_type == "turn_complete":
                self._idle = True
                if isinstance(self._transport, MagicMock):
                    self._transport.dispatch_pending = False
            yield chunk
            await asyncio.sleep(0)

    def is_idle(self):
        return self._idle

    def is_alive(self):
        return True


def _wire(chunks):
    ws = MagicMock()
    ws.broadcast = MagicMock()
    activity = MagicMock()
    adapter = _Adapter(chunks)
    # Record the adapter's idle flag as each content chunk is processed —
    # the status endpoint reads is_idle(), so a self turn must flip it.
    activity.on_message = MagicMock(
        side_effect=lambda *a, **k: adapter.idle_seen.append(adapter._idle))
    activity.on_worker_chunk = MagicMock()
    activity.on_worker_turn_closed = MagicMock()
    observer = MagicMock()
    for name in ("on_completed", "on_error", "on_continued", "on_thread_update",
                 "on_turn_interrupted", "on_interaction_required"):
        setattr(observer, name, AsyncMock())
    pm = ProcessManager(ws_manager=ws, activity_translator=activity,
                        lifecycle_observer=observer)
    closed: list[tuple] = []

    async def on_turn_closed(project_id, handle, *, session_id=None, cause=None):
        closed.append((project_id, handle, session_id, cause))

    pm.set_turn_closed_callback(on_turn_closed)
    transcript = MagicMock()
    transcript.filepath = "/tmp/claude-code.jsonl"
    transcript.append = MagicMock()
    return pm, observer, adapter, transcript, closed


async def _run(pm, adapter, transcript):
    await pm.start(PID, HANDLE, adapter, transcript=transcript, session_id=SID)
    await asyncio.wait_for(pm._tasks[KEY], timeout=5.0)


def _rows(transcript) -> list[dict]:
    return [c.args[0] for c in transcript.append.call_args_list]


# ---------------------------------------------------------------------------
# §6.7 — the consume loop on a self-initiated turn
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_self_turn_is_recorded_as_continued_and_pops_no_dispatch_id():
    pm, observer, adapter, transcript, closed = _wire([
        _chunk("response", "frames look fine, v2 is out"),
        _turn_complete("self"),
    ])
    # A dispatch id enqueued for the NEXT dispatch (set_active_dispatch runs
    # before the recap build that precedes the query) must survive a self
    # turn closing in between.
    pm.set_active_dispatch(PID, HANDLE, "d-next", session_id=SID)

    await _run(pm, adapter, transcript)

    observer.on_completed.assert_not_awaited()
    observer.on_error.assert_not_awaited()
    observer.on_continued.assert_awaited_once()
    kwargs = observer.on_continued.await_args.kwargs
    assert observer.on_continued.await_args.args[:2] == (PID, HANDLE)
    assert kwargs["summary"] == "frames look fine, v2 is out"
    assert kwargs["transcript_path"] == transcript.filepath
    assert kwargs["session_id"] == SID
    assert kwargs["cause"] == "success"
    # Resume identity is still recorded, like a completed turn's.
    observer.on_thread_update.assert_awaited_once()
    assert observer.on_thread_update.await_args.kwargs["claude_session_id"] == "cli-sid-1"
    # The boundary row is written so transcript slices stay well-formed, but
    # it carries no dispatch id and the queued one is untouched.
    rows = _rows(transcript)
    assert rows[-1]["chunk_type"] == "turn_complete"
    assert "dispatch_id" not in rows[-1]
    assert "dispatch_id" not in rows[0]
    assert list(pm._active_dispatch_id[KEY]) == ["d-next"]
    assert pm._turn_open[KEY] is False
    assert closed == [(PID, HANDLE, SID, "continued")]
    # Status: running while the self turn streams (the adapter's idle flag
    # is cleared by its first chunk), idle again at the boundary.
    assert adapter.idle_seen == [False]
    assert adapter._idle is True


@pytest.mark.asyncio
async def test_self_turn_error_is_continued_with_cause_error():
    pm, observer, adapter, transcript, closed = _wire([
        _chunk("error", "Error: the reader died"),
        _turn_complete("self", cause="error"),
    ])
    await _run(pm, adapter, transcript)
    observer.on_error.assert_not_awaited()
    observer.on_continued.assert_awaited_once()
    assert observer.on_continued.await_args.kwargs["cause"] == "error"
    assert observer.on_continued.await_args.kwargs["summary"] == "Error: the reader died"
    assert closed == [(PID, HANDLE, SID, "continued")]


@pytest.mark.asyncio
async def test_dispatch_turn_after_a_self_turn_pops_its_own_id():
    pm, observer, adapter, transcript, closed = _wire([
        _chunk("response", "self text"),
        _turn_complete("self"),
        _chunk("response", "the reply", _pending=True),
        _turn_complete("dispatch"),
    ])
    pm.set_active_dispatch(PID, HANDLE, "d-1", session_id=SID)

    await _run(pm, adapter, transcript)

    observer.on_continued.assert_awaited_once()
    observer.on_completed.assert_awaited_once()
    assert observer.on_completed.await_args.kwargs["summary"] == "the reply"
    rows = _rows(transcript)
    # The self turn's row borrows nothing; the dispatched turn's row and
    # boundary carry the id that was queued for it.
    assert [r.get("dispatch_id") for r in rows if r["chunk_type"] == "response"] == [None, "d-1"]
    boundaries = [r for r in rows if r["chunk_type"] == "turn_complete"]
    assert [b.get("dispatch_id") for b in boundaries] == [None, "d-1"]
    assert KEY not in pm._active_dispatch_id
    assert [c[3] for c in closed] == ["continued", "success"]


@pytest.mark.asyncio
async def test_missing_origin_is_a_dispatch_boundary():
    """Codex / ACP / Pi transports do not stamp origin: unchanged behaviour."""
    pm, observer, adapter, transcript, closed = _wire([
        _chunk("response", "done"),
        _turn_complete(None),
    ])
    class _NoFlagTransport:  # codex / ACP / Pi: no dispatch_pending at all
        _proc = None

    adapter._transport = _NoFlagTransport()
    pm.set_active_dispatch(PID, HANDLE, "d-1", session_id=SID)
    await _run(pm, adapter, transcript)
    observer.on_completed.assert_awaited_once()
    observer.on_continued.assert_not_awaited()
    assert [r.get("dispatch_id") for r in _rows(transcript)] == ["d-1", "d-1"]


@pytest.mark.asyncio
async def test_self_turn_background_shell_registers_and_status_follows_the_registry():
    """A run_in_background Bash call inside a self turn registers like one in
    a dispatched turn, and after the boundary the status reads
    background-running while the shell lives, idle once it is gone."""
    pm, observer, adapter, transcript, closed = _wire([
        _chunk("tool_activity", "[Using tool: Bash]", tool_name="Bash",
               tool_id="t1", tool_input={"command": "sleep 25", "run_in_background": True}),
        _turn_complete("self"),
    ])
    registered = []
    pm.background_work.register = MagicMock(
        side_effect=lambda *a, **k: registered.append((a, k)))
    await _run(pm, adapter, transcript)
    assert registered and registered[0][0] == (PID, SID, HANDLE)
    assert registered[0][1]["command"] == "sleep 25"

    sam = SubAgentManager(process_manager=pm)
    registry = MagicMock(spec=BackgroundWorkRegistry)
    registry.three_state_supported = True
    adapter._transport.supports_background_status = True
    pm.background_work = registry
    registry.has_live.return_value = True
    assert sam._adapter_status(adapter, PID, SID, HANDLE) == "background-running"
    registry.has_live.return_value = False
    assert sam._adapter_status(adapter, PID, SID, HANDLE) == "idle"


@pytest.mark.asyncio
async def test_a_powershell_tool_background_shell_registers_too():
    """On Windows the CLI may launch the same background shell through its
    PowerShell tool."""
    pm, observer, adapter, transcript, closed = _wire([
        _chunk("tool_activity", "[Using tool: PowerShell]", tool_name="PowerShell",
               tool_id="t1", tool_input={"command": "ping -n 60 127.0.0.1",
                                         "run_in_background": True}),
        _turn_complete("self"),
    ])
    pm.background_work.register = MagicMock()
    await _run(pm, adapter, transcript)
    pm.background_work.register.assert_called_once()
    assert pm.background_work.register.call_args.kwargs["command"] == "ping -n 60 127.0.0.1"


# ---------------------------------------------------------------------------
# §6.8 — the lifecycle row and broadcast
# ---------------------------------------------------------------------------

def _observer():
    agent_manager = MagicMock()
    agent_manager.inject_system_message = AsyncMock()
    ws = MagicMock()
    ws.broadcast = MagicMock()
    return LifecycleObserver(agent_manager, ws), agent_manager, ws


@pytest.mark.asyncio
async def test_on_continued_writes_a_continued_row_and_broadcasts():
    observer, agent_manager, ws = _observer()
    await observer.on_continued(PID, HANDLE, summary="v2 is out at out/v2.mp4",
                                transcript_path="/t/claude-code.jsonl",
                                session_id=SID, cause="success")

    agent_manager.inject_system_message.assert_awaited_once()
    args, kwargs = agent_manager.inject_system_message.await_args
    assert args[0] == PID
    assert args[1].startswith("[Sub-agent] claude-code continued. Summary: v2 is out at out/v2.mp4.")
    assert "/t/claude-code.jsonl" in args[1]
    assert kwargs["session_id"] == SID
    meta = kwargs["meta"]
    assert meta["event"] == "sub_agent_terminal"
    assert meta["kind"] == "continued"
    assert meta["handle"] == HANDLE
    assert meta["reply"] == "v2 is out at out/v2.mp4"
    assert meta["display_content"].startswith("[Sub-agent] claude-code continued.")
    # Never a reply to a prompt, never a dispatch boundary: the row lands
    # for a later manager wake to read, it does not wake one itself.
    assert meta["suppress_wake"] is True
    ws.broadcast.assert_called_once()
    event = ws.broadcast.call_args.args[1]
    assert event["type"] == "sub_agent.continued"
    assert event["handle"] == HANDLE and event["session_id"] == SID
    assert event["summary"] == "v2 is out at out/v2.mp4"


@pytest.mark.asyncio
async def test_on_continued_with_no_text_and_an_error_cause():
    observer, agent_manager, ws = _observer()
    await observer.on_continued(PID, HANDLE, summary="", transcript_path="/t/x.jsonl",
                                session_id=SID, cause="error")
    meta = agent_manager.inject_system_message.await_args.kwargs["meta"]
    assert meta["kind"] == "continued" and meta["cause"] == "error"
    assert meta["reply"] == "(no output)"


@pytest.mark.asyncio
async def test_on_continued_kicks_the_pinned_quiescence_hook():
    observer, agent_manager, ws = _observer()
    kicked = []
    observer.pinned_terminal_hook = lambda pid, sid: kicked.append((pid, sid))
    observer.set_dispatch_initiator(PID, HANDLE, "user_pinned", session_id=SID)
    await observer.on_continued(PID, HANDLE, summary="x", transcript_path="/t/x.jsonl",
                                session_id=SID, cause="success")
    assert kicked == [(PID, SID)]
