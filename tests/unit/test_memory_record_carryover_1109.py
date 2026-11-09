# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""1109: a run that went unrecorded is recorded by the next one.

Spec 142 (accepted 2026-11-02): a user message to the session goes ahead of
the run-end record turn, and "the agent decides what to record at the end of
the following round". Live 2026-11-09 the following round's ask named only
its own files and said "write only what you know from this run", so a fact
said in the skipped run (the next step) was lost. The loop now carries the
skipped run's changed files into the next ask, with one line saying that run
was not recorded. Only that session's own message ever skips a record turn,
and that session's next run has the whole conversation, so nothing else is
needed.

Also 1109: PROJECT_STATE.md must be true after the run, so the ask lets the
model fix a stale state line instead of "do not tidy anything".
"""

from __future__ import annotations

import asyncio

import pytest

from agent_os.agent.loop import build_memory_record_text
from tests.unit.test_memory_record_turn import (
    _ScriptedLLM,
    _memory_rows,
    _reply,
    _rig,
    _seed_memory,
    _tool_calls,
)

CARRY_LINE = "The previous run was not recorded"


def _write(call_id, path):
    return _tool_calls((call_id, "write", {"path": path, "content": "x"}))


# ---------------------------------------------------------------------------
# The ask's wording
# ---------------------------------------------------------------------------


def test_the_ask_lets_the_model_fix_a_stale_state_line(tmp_path):
    _seed_memory(str(tmp_path))
    text = " ".join(build_memory_record_text(str(tmp_path), ["a.md"], []).split())
    assert "fix or remove any line in it that is no longer true" in text
    assert "Do not tidy or trim anything" not in text
    # The other three files are still added to, never tidied, by a record turn.
    assert "Do not tidy DECISIONS.md, LESSONS.md or INDEX.md" in text


def test_the_ask_names_an_unrecorded_previous_run_and_its_files(tmp_path):
    _seed_memory(str(tmp_path))
    text = build_memory_record_text(str(tmp_path), ["b.md"], [],
                                    unrecorded_files=["a.md"])
    assert CARRY_LINE in text
    assert "Files changed since the last record: a.md, b.md\n" in text
    assert "Files this run changed" not in text


def test_an_unrecorded_run_that_changed_no_files_is_still_named(tmp_path):
    _seed_memory(str(tmp_path))
    text = build_memory_record_text(str(tmp_path), [], [], unrecorded_files=[])
    assert CARRY_LINE in text
    assert "Files changed since the last record" not in text


def test_no_carry_means_the_ask_is_unchanged(tmp_path):
    _seed_memory(str(tmp_path))
    text = build_memory_record_text(str(tmp_path), ["b.md"], [])
    assert CARRY_LINE not in text
    assert "Files this run changed: b.md\n" in text


# ---------------------------------------------------------------------------
# The loop carries a skipped run into the next ask
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_run_skipped_by_a_held_message_is_named_in_the_next_ask(tmp_path):
    llm = _ScriptedLLM([
        _write("t1", "docs/a.md"), _reply("first reply"),       # run 1 (gated)
        _write("t2", "docs/b.md"), _reply("second reply"),      # run 2
        _reply("nothing to record"),                            # run 2's record
    ], gate_call=1)
    session, tools, loop = _rig(tmp_path, llm)
    task = asyncio.create_task(loop.run())
    await asyncio.wait_for(llm.in_flight.wait(), timeout=2)
    session.queue_message("and the next step is the HTTP API", nonce="n1")
    llm.release.set()
    await asyncio.wait_for(task, timeout=5)
    assert _memory_rows(session) == []  # spec 142: run 1 stays unrecorded

    await asyncio.wait_for(loop.run(), timeout=5)
    [row] = _memory_rows(session)
    assert CARRY_LINE in row["content"]
    assert "Files changed since the last record: docs/a.md, docs/b.md\n" in row["content"]


@pytest.mark.asyncio
async def test_a_record_call_cancelled_by_a_message_is_named_in_the_next_ask(tmp_path):
    llm = _ScriptedLLM([
        _write("t1", "docs/a.md"), _reply("first reply"),
        _reply("nothing to record"),                            # run 1's record (gated)
        _reply("second reply"),                                 # run 2
        _reply("nothing to record"),                            # run 2's record
    ], gate_call=2)
    session, tools, loop = _rig(tmp_path, llm)
    task = asyncio.create_task(loop.run())
    await asyncio.wait_for(llm.in_flight.wait(), timeout=2)
    session.queue_message("one more thing", nonce="n1")
    assert loop.yield_memory_record() is True
    await asyncio.wait_for(task, timeout=5)

    await asyncio.wait_for(loop.run(), timeout=5)
    rows = _memory_rows(session)
    assert len(rows) == 2
    assert CARRY_LINE not in rows[0]["content"]
    assert CARRY_LINE in rows[1]["content"]
    assert "Files changed since the last record: docs/a.md\n" in rows[1]["content"]


@pytest.mark.asyncio
async def test_the_carry_is_spent_once_the_next_ask_is_made(tmp_path):
    llm = _ScriptedLLM([
        _reply("first reply"),                                  # run 1 (gated)
        _reply("second reply"), _reply("nothing to record"),    # run 2 + record
        _reply("third reply"), _reply("nothing to record"),     # run 3 + record
    ], gate_call=0)
    session, tools, loop = _rig(tmp_path, llm)
    task = asyncio.create_task(loop.run())
    await asyncio.wait_for(llm.in_flight.wait(), timeout=2)
    session.queue_message("second", nonce="n1")
    llm.release.set()
    await asyncio.wait_for(task, timeout=5)
    await asyncio.wait_for(loop.run(), timeout=5)
    session.queue_message("third", nonce="n2")
    await asyncio.wait_for(loop.run(), timeout=5)
    rows = _memory_rows(session)
    assert len(rows) == 2
    assert CARRY_LINE in rows[0]["content"]
    assert CARRY_LINE not in rows[1]["content"]


@pytest.mark.asyncio
async def test_a_run_that_is_never_asked_carries_nothing(tmp_path):
    """Only a message-skipped record turn carries: a scratch / memory-session
    / disabled loop is never asked, so it never names a previous run."""
    llm = _ScriptedLLM([_reply("first reply"), _reply("second reply")], gate_call=0)
    session, tools, loop = _rig(tmp_path, llm, enabled=False)
    task = asyncio.create_task(loop.run())
    await asyncio.wait_for(llm.in_flight.wait(), timeout=2)
    session.queue_message("second", nonce="n1")
    llm.release.set()
    await asyncio.wait_for(task, timeout=5)
    assert loop._unrecorded_files is None
