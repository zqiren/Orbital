# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 112 A6 (7d): a pinned direct-chat worker gets the record turn too.

The record turn goes to whoever ended the exchange with the user. A worker
the manager dispatched is not asked (the manager records on its wake run);
a worker the user talks to directly (the composer pin, spec 074) is — as its
next turn on its resumed session, right after its completion row lands,
once per turn, never for the reminder turn itself, and only on transports
that report tool names and paths (claude-code SDK stream, codex app-server,
ACP, Pi RPC — not pipe/PTY, whose output is parsed text).
"""

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.loop import MEMORY_RECORD_KIND
from agent_os.agent.session import Session, persist_user_row
from agent_os.daemon_v2.agent_manager import (
    AgentManager, ProjectHandle, worker_turn_changed_files,
)
from agent_os.daemon_v2.models import make_session_key


PROJECT = "proj_pin"
SID = "sess-pin-0001"
HANDLE = "codex"


class CodexTransport:  # eligibility is by transport class name
    pass


class SDKTransport:
    pass


class ACPSDKTransport:
    pass


class PipeTransport:
    pass


def _write_transcript(path: str, *, tool_rows: list[dict], dispatch_id="d1",
                      reply="Done, routes fixed.") -> None:
    rows = []
    for idx, tr in enumerate(tool_rows):
        rows.append({
            "source": HANDLE, "content": f"[Tool: {tr['tool_name']}]",
            "timestamp": f"2026-10-20T10:00:0{idx}+00:00", "chunk_type": "tool_activity",
            "metadata": {"tool_call_id": f"t{idx}", **tr}, "dispatch_id": dispatch_id,
        })
    rows.append({"source": HANDLE, "content": reply,
                 "timestamp": "2026-10-20T10:00:09+00:00", "chunk_type": "response",
                 "dispatch_id": dispatch_id})
    rows.append({"source": HANDLE, "content": "", "timestamp": "2026-10-20T10:00:10+00:00",
                 "chunk_type": "turn_complete", "dispatch_id": dispatch_id})
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _manager(tmp_path, *, transport_cls=CodexTransport, tool_rows=None):
    ws = MagicMock()
    ws.broadcast = MagicMock()
    sub = MagicMock()
    sub.send = AsyncMock(return_value=f"Message sent to {HANDLE}. Transcript: t.jsonl")
    sub.list_active = MagicMock(return_value=[])
    adapter = MagicMock()
    adapter._transport = transport_cls() if transport_cls else None
    sk = make_session_key(PROJECT, SID)
    sub._adapters = {sk: {HANDLE: adapter}}
    transcript_path = os.path.join(str(tmp_path), "codex.jsonl")
    _write_transcript(transcript_path, tool_rows=tool_rows if tool_rows is not None else [
        {"tool_name": "Edit", "arguments": {"file_path": os.path.join(str(tmp_path), "src/api/routes.py")}},
        {"tool_name": "Bash", "arguments": {"command": "pytest"}},
        {"tool_name": "Write", "arguments": {"file_path": "tests/test_routes.py"}},
    ])
    transcript = MagicMock()
    transcript.filepath = transcript_path
    sub._transcripts = {(PROJECT, SID, HANDLE): transcript}
    project_store = MagicMock()
    project_store.get_project = MagicMock(return_value={"workspace": str(tmp_path)})
    mgr = AgentManager(
        project_store=project_store, ws_manager=ws, sub_agent_manager=sub,
        activity_translator=MagicMock(), process_manager=MagicMock(),
    )
    return mgr, sub


def _plant_session(mgr, tmp_path, *, pinned_target=HANDLE):
    session = Session.new(SID, str(tmp_path))
    session.pinned_target = pinned_target
    persist_user_row(session, "fix the route")
    handle = ProjectHandle(
        session=session, loop=MagicMock(), provider=MagicMock(), registry=MagicMock(),
        context_manager=MagicMock(), interceptor=None, task=None,
    )
    mgr._handles[make_session_key(PROJECT, SID)] = handle
    return session


def _completed_meta(handle=HANDLE, **extra):
    display = f"[Sub-agent] {handle} completed. Summary: Done. Transcript: t.jsonl."
    return {"event": "sub_agent_terminal", "kind": "completed",
            "display_content": display, "reply": "Done, routes fixed.",
            "suppress_wake": True, **extra}


async def _settle():
    for _ in range(10):
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Changed-file extraction from a worker turn
# ---------------------------------------------------------------------------


def test_worker_turn_changed_files_collects_edit_like_tools_relative_to_the_workspace(tmp_path):
    rows = [
        {"name": "Edit", "arguments": {"file_path": os.path.join(str(tmp_path), "src/a.py")}},
        {"name": "Write", "arguments": {"file_path": "docs/b.md"}},
        {"name": "MultiEdit", "arguments": {"file_path": "src/a.py"}},   # duplicate path
        {"name": "fileChange", "arguments": {"changes": [{"path": "src/c.py"}, {"path": "src/d.py"}]}},
        {"name": "Bash", "arguments": {"command": "echo hi > e.txt"}},    # not a file tool
        {"name": "Read", "arguments": {"file_path": "src/z.py"}},
        {"name": "edit", "arguments": {"path": "lib/f.py"}},               # pi / acp shape
    ]
    changed, wrote_memory = worker_turn_changed_files(rows, str(tmp_path))
    assert changed == ["src/a.py", "docs/b.md", "src/c.py", "src/d.py", "lib/f.py"]
    assert wrote_memory is False


def test_worker_turn_changed_files_flags_memory_writes(tmp_path):
    rows = [{"name": "Edit", "arguments": {"file_path": "orbital/PROJECT_STATE.md"}},
            {"name": "Write", "arguments": {"file_path": "src/a.py"}}]
    changed, wrote_memory = worker_turn_changed_files(rows, str(tmp_path))
    assert changed == ["src/a.py"] and wrote_memory is True


# ---------------------------------------------------------------------------
# The reminder dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pinned_worker_completion_sends_the_worker_variant_as_its_next_turn(tmp_path):
    mgr, sub = _manager(tmp_path)
    session = _plant_session(mgr, tmp_path)
    content = "[Sub-agent] codex completed. Summary: Done. Transcript: t.jsonl. (guidance)"

    status = await mgr.inject_system_message(PROJECT, content, session_id=SID,
                                             meta=_completed_meta())
    assert status == "suppressed"
    await _settle()

    sub.send.assert_awaited_once()
    args, kwargs = sub.send.call_args
    assert args[:2] == (PROJECT, HANDLE)
    text = args[2]
    assert text.startswith("[Orbital memory] Your last turn is ending.")
    assert "Files this turn changed: src/api/routes.py, tests/test_routes.py\n" in text
    assert text.endswith("If nothing durable happened, reply exactly: nothing to record")
    assert kwargs["session_id"] == SID
    assert kwargs["initiator"] == "user_pinned"
    assert kwargs["dispatch_id"]

    rows = session.get_messages()
    reminder = rows[-1]
    assert reminder["role"] == "user" and reminder["target"] == HANDLE
    assert reminder["source"] == "daemon"
    assert reminder["_kind"] == MEMORY_RECORD_KIND
    assert reminder["content"] == text
    # The worker's completion row before it stays untagged — it IS the reply.
    assert rows[-2]["content"] == content and rows[-2].get("_kind") is None


@pytest.mark.asyncio
async def test_the_reminders_marker_and_completion_are_tagged_and_do_not_re_trigger(tmp_path):
    mgr, sub = _manager(tmp_path)
    session = _plant_session(mgr, tmp_path)
    await mgr.inject_system_message(PROJECT, "[Sub-agent] codex completed. Summary: Done.",
                                    session_id=SID, meta=_completed_meta())
    await _settle()
    dispatch_id = sub.send.call_args.kwargs["dispatch_id"]

    # send() fires on_message_routed → the dispatch marker lands with the id.
    await mgr.inject_system_message(
        PROJECT, f'[Sub-agent] Message sent to {HANDLE}: "[Orbital memory] …". Transcript: t.jsonl',
        session_id=SID,
        meta={"dispatch_id": dispatch_id, "handle": HANDLE, "transcript_path": "t.jsonl",
              "suppress_wake": True})
    # …and its completion closes the reminder turn.
    await mgr.inject_system_message(
        PROJECT, "[Sub-agent] codex completed. Summary: nothing to record. Transcript: t.jsonl.",
        session_id=SID,
        meta=_completed_meta(reply="nothing to record"))
    await _settle()

    rows = session.get_messages()
    assert rows[-2]["_kind"] == MEMORY_RECORD_KIND          # the marker
    assert rows[-1]["_kind"] == MEMORY_RECORD_KIND          # the reminder's completion
    assert sub.send.await_count == 1                        # no second ask

    # A later ordinary completion is untagged and asks again.
    await mgr.inject_system_message(PROJECT, "[Sub-agent] codex completed. Summary: more work.",
                                    session_id=SID, meta=_completed_meta())
    await _settle()
    assert session.get_messages()[-2].get("_kind") is None
    assert sub.send.await_count == 2


@pytest.mark.asyncio
async def test_worker_dispatched_by_the_manager_is_not_asked(tmp_path):
    """The session is not pinned to the worker: the manager is waiting for
    the result and records on its own wake run."""
    mgr, sub = _manager(tmp_path)
    _plant_session(mgr, tmp_path, pinned_target=None)
    meta = _completed_meta()
    meta.pop("suppress_wake")
    mgr._start_loop = AsyncMock()
    await mgr.inject_system_message(PROJECT, "[Sub-agent] codex completed. Summary: Done.",
                                    session_id=SID, meta=meta)
    await _settle()
    sub.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_session_pinned_to_another_worker_is_not_asked_for_this_one(tmp_path):
    mgr, sub = _manager(tmp_path)
    _plant_session(mgr, tmp_path, pinned_target="claude-code")
    await mgr.inject_system_message(PROJECT, "[Sub-agent] codex completed. Summary: Done.",
                                    session_id=SID, meta=_completed_meta())
    await _settle()
    sub.send.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport_cls", [PipeTransport, None])
async def test_pipe_and_pty_transports_are_not_asked(tmp_path, transport_cls):
    mgr, sub = _manager(tmp_path, transport_cls=transport_cls)
    _plant_session(mgr, tmp_path)
    await mgr.inject_system_message(PROJECT, "[Sub-agent] codex completed. Summary: Done.",
                                    session_id=SID, meta=_completed_meta())
    await _settle()
    sub.send.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport_cls", [SDKTransport, ACPSDKTransport])
async def test_sdk_and_acp_transports_are_asked(tmp_path, transport_cls):
    mgr, sub = _manager(tmp_path, transport_cls=transport_cls)
    _plant_session(mgr, tmp_path)
    await mgr.inject_system_message(PROJECT, "[Sub-agent] codex completed. Summary: Done.",
                                    session_id=SID, meta=_completed_meta())
    await _settle()
    sub.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_worker_turn_that_wrote_a_memory_file_is_not_asked(tmp_path):
    # A project with goals (spec 167: a goal-less one is still asked, to draft them).
    goals = os.path.join(str(tmp_path), "orbital", "instructions", "project_goals.md")
    os.makedirs(os.path.dirname(goals), exist_ok=True)
    with open(goals, "w", encoding="utf-8") as fh:
        fh.write("# Goals\n- ship\n")
    mgr, sub = _manager(tmp_path, tool_rows=[
        {"tool_name": "Edit", "arguments": {"file_path": "orbital/PROJECT_STATE.md"}},
        {"tool_name": "Write", "arguments": {"file_path": "src/x.py"}},
    ])
    _plant_session(mgr, tmp_path)
    await mgr.inject_system_message(PROJECT, "[Sub-agent] codex completed. Summary: Done.",
                                    session_id=SID, meta=_completed_meta())
    await _settle()
    sub.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_worker_turn_with_no_file_changes_is_still_asked_with_none(tmp_path):
    mgr, sub = _manager(tmp_path, tool_rows=[
        {"tool_name": "Bash", "arguments": {"command": "ls"}},
    ])
    _plant_session(mgr, tmp_path)
    await mgr.inject_system_message(PROJECT, "[Sub-agent] codex completed. Summary: Done.",
                                    session_id=SID, meta=_completed_meta())
    await _settle()
    sub.send.assert_awaited_once()
    assert "Files this turn changed: none\n" in sub.send.call_args.args[2]


@pytest.mark.asyncio
async def test_error_and_stopped_terminals_are_not_asked(tmp_path):
    mgr, sub = _manager(tmp_path)
    _plant_session(mgr, tmp_path)
    for kind in ("error", "stopped", "interrupted", "failed"):
        await mgr.inject_system_message(PROJECT, f"[Sub-agent] codex {kind}", session_id=SID,
                                        meta={"event": "sub_agent_terminal", "kind": kind,
                                              "suppress_wake": True})
    await _settle()
    sub.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_send_leaves_no_pending_tag(tmp_path):
    mgr, sub = _manager(tmp_path)
    sub.send = AsyncMock(return_value="Error: agent 'codex' not running")
    session = _plant_session(mgr, tmp_path)
    await mgr.inject_system_message(PROJECT, "[Sub-agent] codex completed. Summary: Done.",
                                    session_id=SID, meta=_completed_meta())
    await _settle()
    # The next real completion is untagged (nothing is left pending).
    await mgr.inject_system_message(PROJECT, "[Sub-agent] codex completed. Summary: Again.",
                                    session_id=SID, meta=_completed_meta())
    await _settle()
    completions = [m for m in session.get_messages()
                   if (m.get("_meta") or {}).get("event") == "sub_agent_terminal"]
    assert all(m.get("_kind") is None for m in completions)


@pytest.mark.asyncio
async def test_the_reminder_task_is_strongly_referenced_until_it_finishes(tmp_path):
    """The event loop holds tasks weakly; a fire-and-forget task with no
    reference can be collected mid-flight ("Task was destroyed but it is
    pending"). The manager keeps it in a set that empties itself."""
    mgr, sub = _manager(tmp_path)
    gate = asyncio.Event()

    async def slow_send(*args, **kwargs):
        await gate.wait()
        return "Message sent to codex. Transcript: t.jsonl"

    sub.send = AsyncMock(side_effect=slow_send)
    _plant_session(mgr, tmp_path)
    await mgr.inject_system_message(PROJECT, "[Sub-agent] codex completed. Summary: Done.",
                                    session_id=SID, meta=_completed_meta())
    await _settle()
    assert len(mgr._background_tasks) == 1
    task = next(iter(mgr._background_tasks))
    assert isinstance(task, asyncio.Task) and not task.done()
    gate.set()
    await _settle()
    assert mgr._background_tasks == set()


# ---------------------------------------------------------------------------
# Spec 129 §5D: the reminder fires on quiescence, once per run
# ---------------------------------------------------------------------------

from agent_os.daemon_v2.background_work import BackgroundWorkRegistry  # noqa: E402


def _registry(mgr, live: list[str]):
    registry = MagicMock(spec=BackgroundWorkRegistry)
    registry.live_commands = MagicMock(return_value=list(live))
    mgr._process_manager.background_work = registry
    return registry


def _continued_meta(handle=HANDLE, reply="v2 is out", **extra):
    display = f"[Sub-agent] {handle} continued. Summary: {reply}. Transcript: t.jsonl."
    return {"event": "sub_agent_terminal", "kind": "continued", "handle": handle,
            "display_content": display, "reply": reply, "suppress_wake": True,
            "cause": "success", **extra}


@pytest.mark.asyncio
async def test_completed_terminal_with_a_live_background_shell_sends_no_reminder(tmp_path):
    """The worker ended its turn to wait for a shell: the run is not over.
    Asking now would get its answer from the shell's self-wake turn."""
    mgr, sub = _manager(tmp_path, transport_cls=SDKTransport)
    _plant_session(mgr, tmp_path)
    registry = _registry(mgr, ["sleep 25; echo done"])
    await mgr.inject_system_message(PROJECT, "[Sub-agent] codex completed. Summary: waiting.",
                                    session_id=SID, meta=_completed_meta())
    await _settle()
    sub.send.assert_not_awaited()
    registry.live_commands.assert_called_with(PROJECT, SID, HANDLE)


@pytest.mark.asyncio
async def test_continued_terminal_with_an_empty_registry_sends_one_reminder_spanning_the_run(tmp_path):
    """The shell exited, the worker ran its own turn and went quiet: now ask,
    and list every file the run touched since the last record row — the
    dispatched turn's edits AND the self turn's — not just the last turn's."""
    mgr, sub = _manager(tmp_path, transport_cls=SDKTransport)
    session = _plant_session(mgr, tmp_path)
    _registry(mgr, [])
    transcript_path = sub._transcripts[(PROJECT, SID, HANDLE)].filepath

    def tool(name, path, dispatch_id, ts):
        return {"source": HANDLE, "content": f"[Tool: {name}]", "timestamp": ts,
                "chunk_type": "tool_activity",
                "metadata": {"tool_call_id": f"t-{ts}", "tool_name": name,
                             "arguments": {"file_path": path}},
                **({"dispatch_id": dispatch_id} if dispatch_id else {})}

    def boundary(dispatch_id, ts):
        return {"source": HANDLE, "content": "", "timestamp": ts, "chunk_type": "turn_complete",
                **({"dispatch_id": dispatch_id} if dispatch_id else {})}

    rows = [
        # An earlier run, already recorded (its reminder turn is dR).
        tool("Edit", "old/a.py", "d0", "2026-10-22T10:00:00+00:00"), boundary("d0", "2026-10-22T10:00:01+00:00"),
        tool("Edit", "orbital/PROJECT_STATE.md", "dR", "2026-10-22T10:00:02+00:00"), boundary("dR", "2026-10-22T10:00:03+00:00"),
        # This run: the dispatched turn starts a shell and ends...
        tool("Edit", "src/b.py", "d1", "2026-10-22T10:01:00+00:00"), boundary("d1", "2026-10-22T10:01:01+00:00"),
        # ...then the worker's own turn when the shell exits (no dispatch id).
        tool("Write", "src/c.py", None, "2026-10-22T10:02:00+00:00"),
        {"source": HANDLE, "content": "v2 is out", "timestamp": "2026-10-22T10:02:01+00:00",
         "chunk_type": "response"},
        boundary(None, "2026-10-22T10:02:02+00:00"),
    ]
    with open(transcript_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    # The management session: the earlier run's reminder marker (tagged,
    # carrying its dispatch id) is the record row the files list starts after.
    session.append({"role": "system", "source": "daemon",
                    "content": '[Sub-agent] Message sent to codex: "[Orbital memory] …"',
                    "_kind": MEMORY_RECORD_KIND,
                    "_meta": {"dispatch_id": "dR", "handle": HANDLE, "transcript_path": transcript_path,
                              "suppress_wake": True}})
    session.append({"role": "system", "source": "daemon",
                    "content": "[Sub-agent] codex completed. Summary: nothing to record.",
                    "_kind": MEMORY_RECORD_KIND, "_meta": _completed_meta(reply="nothing to record")})
    session.append({"role": "system", "source": "daemon",
                    "content": '[Sub-agent] Message sent to codex: "run it"',
                    "_meta": {"dispatch_id": "d1", "handle": HANDLE, "transcript_path": transcript_path,
                              "suppress_wake": True}})

    # The dispatched turn's completion arrived while the shell was live → no ask.
    _registry(mgr, ["sleep 25"])
    await mgr.inject_system_message(PROJECT, "[Sub-agent] codex completed. Summary: waiting.",
                                    session_id=SID, meta=_completed_meta())
    await _settle()
    sub.send.assert_not_awaited()

    # The self turn ends with nothing live underneath → exactly one ask.
    _registry(mgr, [])
    await mgr.inject_system_message(PROJECT, "[Sub-agent] codex continued. Summary: v2 is out.",
                                    session_id=SID, meta=_continued_meta())
    await _settle()
    sub.send.assert_awaited_once()
    text = sub.send.call_args.args[2]
    assert "Files this turn changed: src/b.py, src/c.py\n" in text
    assert "old/a.py" not in text


@pytest.mark.asyncio
async def test_a_shared_transcript_never_lends_another_sessions_turn_to_the_reminder(tmp_path):
    """Two sessions of one worker write one transcript file. When this
    session's turn has no boundary row yet (the terminal landed first), the
    reminder must still list ITS files, not the other session's last turn;
    and when nothing of this session is in the file, it must say none."""
    mgr, sub = _manager(tmp_path, transport_cls=SDKTransport)
    session = _plant_session(mgr, tmp_path)
    transcript_path = sub._transcripts[(PROJECT, SID, HANDLE)].filepath

    def tool(name, path, dispatch_id, ts):
        return {"source": HANDLE, "content": f"[Tool: {name}]", "timestamp": ts,
                "chunk_type": "tool_activity",
                "metadata": {"tool_call_id": f"t-{ts}", "tool_name": name,
                             "arguments": {"file_path": path}},
                "dispatch_id": dispatch_id}

    def boundary(dispatch_id, ts):
        return {"source": HANDLE, "content": "", "timestamp": ts, "chunk_type": "turn_complete",
                "dispatch_id": dispatch_id}

    def write(rows):
        with open(transcript_path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    # Another session's completed turn, then this session's turn still
    # without its boundary row.
    write([
        tool("Edit", "other/x.py", "dX", "2026-10-22T10:00:00+00:00"),
        boundary("dX", "2026-10-22T10:00:01+00:00"),
        tool("Edit", "calc.py", "d1", "2026-10-22T10:01:00+00:00"),
        {"source": HANDLE, "content": "fixed", "timestamp": "2026-10-22T10:01:01+00:00",
         "chunk_type": "response", "dispatch_id": "d1"},
    ])
    session.append({"role": "system", "source": "daemon",
                    "content": '[Sub-agent] Message sent to codex: "fix calc.py"',
                    "_meta": {"dispatch_id": "d1", "handle": HANDLE, "transcript_path": transcript_path,
                              "suppress_wake": True}})
    await mgr.inject_system_message(PROJECT, "[Sub-agent] codex completed. Summary: fixed.",
                                    session_id=SID, meta=_completed_meta())
    await _settle()
    sub.send.assert_awaited_once()
    text = sub.send.call_args.args[2]
    assert "Files this turn changed: calc.py\n" in text
    assert "other/x.py" not in text

    # Nothing of this session in the file at all: none, never the other turn.
    sub.send.reset_mock()
    mgr._worker_memory_records.clear()
    write([
        tool("Edit", "other/x.py", "dX", "2026-10-22T10:00:00+00:00"),
        boundary("dX", "2026-10-22T10:00:01+00:00"),
    ])
    session.append({"role": "system", "source": "daemon",
                    "content": '[Sub-agent] Message sent to codex: "again"',
                    "_meta": {"dispatch_id": "d2", "handle": HANDLE, "transcript_path": transcript_path,
                              "suppress_wake": True}})
    await mgr.inject_system_message(PROJECT, "[Sub-agent] codex completed. Summary: ok.",
                                    session_id=SID, meta=_completed_meta())
    await _settle()
    sub.send.assert_awaited_once()
    text = sub.send.call_args.args[2]
    assert "Files this turn changed: none\n" in text
    assert "other/x.py" not in text


@pytest.mark.asyncio
async def test_continued_terminal_with_a_queued_prompt_waits_for_the_next_terminal(tmp_path):
    """A user prompt is queued behind the turn that just ended: the reminder
    would ask about the wrong turn. The next terminal re-evaluates."""
    mgr, sub = _manager(tmp_path, transport_cls=SDKTransport)
    _plant_session(mgr, tmp_path)
    _registry(mgr, [])
    sub._prompt_queues = {(PROJECT, SID, HANDLE): ["a queued user prompt"]}
    await mgr.inject_system_message(PROJECT, "[Sub-agent] codex continued. Summary: v2 is out.",
                                    session_id=SID, meta=_continued_meta())
    await _settle()
    sub.send.assert_not_awaited()

    sub._prompt_queues = {}
    await mgr.inject_system_message(PROJECT, "[Sub-agent] codex completed. Summary: done.",
                                    session_id=SID, meta=_completed_meta())
    await _settle()
    sub.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_continued_terminal_with_cause_error_sends_no_reminder(tmp_path):
    mgr, sub = _manager(tmp_path, transport_cls=SDKTransport)
    _plant_session(mgr, tmp_path)
    _registry(mgr, [])
    await mgr.inject_system_message(PROJECT, "[Sub-agent] codex continued. Summary: (no output).",
                                    session_id=SID, meta=_continued_meta(cause="error"))
    await _settle()
    sub.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_reminder_lock_exhaustion_is_dropped_with_a_warning_naming_the_session(tmp_path, caplog, monkeypatch):
    """§7.2: on retry exhaustion the reminder is dropped (no re-queue); the
    warning names the path so the log reader does not need a traceback."""
    import logging
    from agent_os.utils import file_lock
    from agent_os.utils.file_lock import FileLockError
    mgr, sub = _manager(tmp_path, transport_cls=SDKTransport)
    _plant_session(mgr, tmp_path)
    _registry(mgr, [])
    monkeypatch.setattr(file_lock, "LOCK_RETRY_DELAYS", (0.001, 0.002))
    locked_path = os.path.join(str(tmp_path), "orbital", "sessions", f"{SID}.jsonl")

    async def contended(*a, **k):
        raise FileLockError(f"Session file is locked by another process: {locked_path}")

    mgr.persist_mention_message = contended
    with caplog.at_level(logging.WARNING, logger="agent_os.daemon_v2.agent_manager"):
        await mgr.inject_system_message(PROJECT, "[Sub-agent] codex completed. Summary: done.",
                                        session_id=SID, meta=_completed_meta())
        await _settle()
    sub.send.assert_not_awaited()
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("memory reminder" in w and locked_path in w and SID in w for w in warnings), warnings
    assert not mgr._worker_memory_records
