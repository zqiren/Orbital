# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 155 §5.1: sessions of one project run in parallel, so their round-end
record turns take one per-project lock BEFORE the ``[MEMORY]`` row is built.

The second session's row is then the file as the first session's record turn
left it, its exact-match edit lands on the first try, and neither entry is
lost. The wait is the loop's in-flight task, so a user message to the waiting
session (spec 142) or Stop ends it, and the lock is released whenever the
acquire completed. A record-turn ``write`` on an existing memory file is
refused (every record-turn write is an exact match against the live file),
and the row's PROJECT_STATE ids join the seen-set so a deliberate overwrite
of another session's line is not put back by conservation.

Real ``AgentLoop.run()`` with scripted LLMs over a real workspace, the
harness of ``test_memory_record_turn.py``.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from agent_os.agent import memory_editor
from agent_os.agent.context import ContextManager
from agent_os.agent.loop import AgentLoop
from agent_os.agent.session import Session, persist_user_row
from agent_os.agent.tools.edit import EditTool
from tests.unit.test_memory_record_turn import (
    _PromptBuilder,
    _ScriptedLLM,
    _Tools,
    _memory_rows,
    _prompt_context,
    _reply,
    _seed_memory,
    _tool_calls,
)


def _state(workspace: str) -> str:
    with open(os.path.join(workspace, "orbital", "PROJECT_STATE.md"), encoding="utf-8") as fh:
        return fh.read()


def _loop(workspace: str, llm, sid: str, tools=None):
    session = Session.new(sid, workspace)
    session.origin = "chat"
    tools = tools or _Tools(workspace)
    ctx = ContextManager(session, _PromptBuilder(), _prompt_context(workspace),
                         model_context_limit=200_000)
    loop = AgentLoop(session, llm, tools, ctx, project_dir=workspace,
                     max_iterations=20, memory_record_enabled=True)
    diags: list[dict] = []
    real_diag = loop._diag

    def record_diag(event, **fields):
        if event == "memory_record":
            diags.append(fields)
        return real_diag(event, **fields)

    loop._diag = record_diag
    persist_user_row(session, f"task for {sid}")
    return session, loop, diags


def _add_line(call_id: str, anchor: str, line: str):
    return _tool_calls((call_id, "edit", {
        "path": "orbital/PROJECT_STATE.md",
        "old_text": anchor,
        "new_text": f"{anchor}\n{line}",
    }))


async def _until(pred, timeout=2.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not pred():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition never held")
        await asyncio.sleep(0.005)


# ---------------------------------------------------------------------------
# 1. Two record turns: the second waits, then writes from the first's file.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_sessions_record_in_turn_and_both_entries_survive(tmp_path):
    workspace = str(tmp_path)
    _seed_memory(workspace)
    # A's record call (call 1) is held at a barrier while it holds the lock.
    llm_a = _ScriptedLLM([
        _reply("A's answer"),
        _add_line("ta1", "- next: write copy", "- launch date: Friday"),
        _reply("recorded"),
    ], gate_call=1)
    # B anchors on A's line: only possible if B's row shows A's edit.
    llm_b = _ScriptedLLM([
        _reply("B's answer"),
        _add_line("tb1", "- launch date: Friday", "- pricing: 9 USD a month"),
        _reply("recorded"),
    ])
    sess_a, loop_a, diag_a = _loop(workspace, llm_a, "sess-a")
    sess_b, loop_b, diag_b = _loop(workspace, llm_b, "sess-b")
    lock = memory_editor.record_lock(workspace)

    task_a = asyncio.create_task(loop_a.run())
    await asyncio.wait_for(llm_a.in_flight.wait(), timeout=2)
    assert lock.locked()                      # A holds it during its record call
    task_b = asyncio.create_task(loop_b.run())
    # B answered and is now waiting on the lock: no [MEMORY] row yet.
    await _until(lambda: llm_b.calls == 1 and loop_b._memory_record_active)
    await asyncio.sleep(0.03)
    assert _memory_rows(sess_b) == []
    llm_a.release.set()
    await asyncio.wait_for(asyncio.gather(task_a, task_b), timeout=5)

    state = _state(workspace)
    assert "- launch date: Friday" in state
    assert "- pricing: 9 USD a month" in state
    row_b = _memory_rows(sess_b)[0]["content"]
    assert "- launch date: Friday" in row_b      # B's row is A's file
    assert diag_a[-1]["outcome"] == "edited"
    assert diag_b[-1]["outcome"] == "edited"
    assert diag_b[-1]["lock_wait_ms"] > 0
    assert "lock_wait_ms" in diag_a[-1]
    assert not lock.locked()
    # No "old_text not found" in either session.
    for sess in (sess_a, sess_b):
        assert not any("not found" in str(m.get("content")) for m in sess.get_messages()
                       if m.get("role") == "tool")


@pytest.mark.asyncio
async def test_two_workspaces_do_not_share_a_lock(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    assert memory_editor.record_lock(str(a)) is not memory_editor.record_lock(str(b))
    # Same folder by another path: same lock (realpath, like claim()).
    link = tmp_path / "a-link"
    os.symlink(a, link)
    assert memory_editor.record_lock(str(link)) is memory_editor.record_lock(str(a))


# ---------------------------------------------------------------------------
# 2. The wait yields to a user message; a cancel after acquisition releases.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_message_to_the_waiting_session_ends_the_wait(tmp_path):
    workspace = str(tmp_path)
    _seed_memory(workspace)
    lock = memory_editor.record_lock(workspace)
    await lock.acquire()                         # another session's record turn
    llm = _ScriptedLLM([_reply("answer"), _reply("should not run")])
    sess, loop, diags = _loop(workspace, llm, "sess-wait")
    task = asyncio.create_task(loop.run())
    await _until(lambda: llm.calls == 1 and loop._memory_record_active)
    sess.queue_message("one more thing", nonce="n1")
    assert loop.yield_memory_record() is True
    await asyncio.wait_for(task, timeout=2)      # did not wait for the holder
    assert lock.locked()                         # still the other turn's
    assert diags[-1]["outcome"] == "yielded"
    assert _memory_rows(sess) == []              # the row was never built
    assert sess.list_queued() == [("one more thing", "n1")]
    assert llm.calls == 1
    lock.release()
    assert not lock.locked()


@pytest.mark.asyncio
async def test_stop_while_waiting_leaves_the_lock_to_its_holder(tmp_path):
    workspace = str(tmp_path)
    _seed_memory(workspace)
    lock = memory_editor.record_lock(workspace)
    await lock.acquire()
    llm = _ScriptedLLM([_reply("answer")])
    sess, loop, _ = _loop(workspace, llm, "sess-stop")
    task = asyncio.create_task(loop.run())
    await _until(lambda: llm.calls == 1 and loop._memory_record_active)
    await loop.terminate()
    try:
        await asyncio.wait_for(task, timeout=2)
    except asyncio.CancelledError:
        pass
    lock.release()
    # The lock is free and works: nothing of the stopped wait is left in it.
    assert not lock.locked()
    await asyncio.wait_for(lock.acquire(), timeout=1)
    lock.release()


@pytest.mark.asyncio
async def test_a_cancel_right_after_acquisition_releases_the_lock(tmp_path):
    workspace = str(tmp_path)
    _seed_memory(workspace)
    lock = memory_editor.record_lock(workspace)
    llm = _ScriptedLLM([_reply("answer"), _reply("nothing to record")], gate_call=1)
    sess, loop, _ = _loop(workspace, llm, "sess-cancel")
    task = asyncio.create_task(loop.run())
    await asyncio.wait_for(llm.in_flight.wait(), timeout=2)
    assert lock.locked()
    task.cancel()                                # e.g. daemon quit
    try:
        await asyncio.wait_for(task, timeout=2)
    except asyncio.CancelledError:
        pass
    assert not lock.locked()


@pytest.mark.asyncio
async def test_the_acquire_completing_as_the_cancel_lands_counts_as_held(tmp_path):
    """The invariant behind R4: the acquire task finished (the lock is ours)
    but the waiting turn is cancelled before it resumes. The helper must
    report the lock as held so the turn's ``finally`` releases it."""
    workspace = str(tmp_path)
    _seed_memory(workspace)
    _sess, loop, _ = _loop(workspace, _ScriptedLLM([_reply("x")]), "sess-race")
    lock = asyncio.Lock()
    task = asyncio.create_task(loop._acquire_record_lock(lock))
    await asyncio.sleep(0)                       # helper created the acquire task
    waiter = loop._inflight_stream
    await asyncio.sleep(0)                       # the acquire ran and finished
    assert waiter.done() and lock.locked() and not task.done()
    task.cancel()                                # lands before the helper resumes
    with pytest.raises(asyncio.CancelledError):
        await task
    assert loop._record_lock_held is True
    assert loop._inflight_stream is None


# ---------------------------------------------------------------------------
# 3. write on an existing memory file is refused; on a missing one it runs.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_turn_write_on_an_existing_memory_file_is_refused(tmp_path):
    workspace = str(tmp_path)
    _seed_memory(workspace)
    before = _state(workspace)
    llm = _ScriptedLLM([
        _reply("answer"),
        _tool_calls(("tw1", "write", {"path": "orbital/PROJECT_STATE.md",
                                       "content": "# Project State\n- clobbered\n"})),
        _reply("ok"),
    ])
    sess, loop, diags = _loop(workspace, llm, "sess-write")
    await loop.run()
    assert _state(workspace) == before
    results = [m for m in sess.get_messages() if m.get("role") == "tool"]
    assert results[-1]["content"] == (
        "Error: the memory record turn edits the files with the exact text "
        "shown above. Use edit.")


@pytest.mark.asyncio
async def test_record_turn_write_on_a_missing_memory_file_runs(tmp_path):
    workspace = str(tmp_path)
    _seed_memory(workspace)
    os.remove(os.path.join(workspace, "orbital", "LESSONS.md"))
    llm = _ScriptedLLM([
        _reply("answer"),
        _tool_calls(("tw1", "write", {"path": "orbital/LESSONS.md",
                                       "content": "# Lessons\n- check the port first\n"})),
        _reply("ok"),
    ])
    sess, loop, diags = _loop(workspace, llm, "sess-write2")
    await loop.run()
    with open(os.path.join(workspace, "orbital", "LESSONS.md"), encoding="utf-8") as fh:
        assert "check the port first" in fh.read()
    assert diags[-1]["outcome"] == "edited"


# ---------------------------------------------------------------------------
# 4. The row's ids join the seen-set: overwriting another session's line sticks.
# ---------------------------------------------------------------------------


class _RealEditTools(_Tools):
    """``edit`` through the real tool, so the write path's id conservation
    (``memory_entries.process_on_write``) runs."""

    def execute(self, name, arguments):
        if name == "edit":
            self.executed.append((name, dict(arguments)))
            return EditTool(self._ws).execute(**arguments)
        return super().execute(name, arguments)


@pytest.mark.asyncio
async def test_an_entry_added_after_run_start_can_be_overwritten_by_the_record_turn(tmp_path):
    workspace = str(tmp_path)
    _seed_memory(workspace)
    state_path = os.path.join(workspace, "orbital", "PROJECT_STATE.md")
    entry = ("- blocker: waiting on the design review\n"
             "  <!--mem id:x9 created:2026-01-02 touched:2026-01-02-->\n")
    gate = asyncio.Event()
    started = asyncio.Event()

    class _Gated(_ScriptedLLM):
        async def stream(self, messages, tools=None):
            if self.calls == 0:
                started.set()
                await gate.wait()
            async for chunk in super().stream(messages, tools):
                yield chunk

    # A's record edit replaces the blocker line B added after A's run start.
    llm = _Gated([
        _reply("A's answer"),
        _tool_calls(("te1", "edit", {
            "path": "orbital/PROJECT_STATE.md",
            "old_text": entry,
            "new_text": "- blocker: none, the review passed\n",
        })),
        _reply("recorded"),
    ])
    sess, loop, diags = _loop(workspace, llm, "sess-a", tools=_RealEditTools(workspace))
    task = asyncio.create_task(loop.run())
    await asyncio.wait_for(started.wait(), timeout=2)   # A's run has started
    with open(state_path, "a", encoding="utf-8") as fh:  # session B records X
        fh.write(entry)
    gate.set()
    await asyncio.wait_for(task, timeout=5)

    state = _state(workspace)
    assert "the review passed" in state
    assert "waiting on the design review" not in state
    assert "id:x9" not in state
    assert diags[-1]["outcome"] == "edited"
