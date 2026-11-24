# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 195 decision 8: Orbital stays quiet while the user talks to a worker.

While a session is pinned to a worker, a row that would wake Orbital (a
worker's result, error, question, a drained dispatch marker, a job ending)
lands in the session as usual but starts no Orbital turn. Its ``_meta`` is
left alone, so it stays wake-eligible. The switch back to Orbital
(``wake_if_unread``, called by the sessions PATCH on unpin) starts ONE turn
when such a row is unread, and that one turn reads every held row.

Three parts, as the critique specified:
  H1  one gate in ``inject_system_message`` (hydrated, idle, deferred);
  H2  ``_on_loop_done`` skips the wake while pinned;
  H3  ``wake_if_unread`` on unpin (hydrating an evicted session).

The consumer tests run a REAL ``AgentLoop.run()`` task with a scripted LLM
(the harness of ``test_wake_on_deferred_terminal.py``): a result is never
lost, never delivered twice, and three held results are read in one round.
"""

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.session import Session
from agent_os.daemon_v2.models import make_session_key
from tests.unit.test_wake_on_deferred_terminal import (
    PROJECT, SID, _ScriptedLLM, _SlowTool, _reply, _run_turn, _spy_wake,
    _start_real_turn, _terminal_meta, _turn_ended, manager,  # noqa: F401 (fixture)
)


def _held_rows(session) -> list[dict]:
    return [m for m in session.get_messages()
            if (m.get("_meta") or {}).get("event") in (
                "sub_agent_terminal", "interaction_required")]


async def _settle(n: int = 5) -> None:
    for _ in range(n):
        await asyncio.sleep(0)


def _wire_real_resume(manager) -> None:
    """Let ``_start_loop`` run for real (it starts ``loop.run()`` with the
    real done-callback); only the project-config rebuild is stubbed away."""
    manager._abuild_agent_config_from_project = AsyncMock(
        side_effect=RuntimeError("no project in this test"))
    manager._note_goals_state_at_run_start = MagicMock()


# ---------------------------------------------------------------------------
# H1 — idle branch: held, meta untouched
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("meta", [
    _terminal_meta("completed", reply="done"),
    _terminal_meta("error"),
    {"event": "interaction_required", "handle": "dsh", "interaction_id": "i1",
     "kind": "question"},
    # The marker of an Orbital dispatch that drained from the worker's FIFO.
    {"dispatch_id": "d1", "handle": "codex", "transcript_path": "/t",
     "initiator": "management_agent"},
    {"event": "job_terminal"},
])
async def test_idle_pinned_session_holds_the_row(manager, tmp_path, meta):
    wake = _spy_wake(manager)
    llm = _ScriptedLLM([_reply("first turn")])
    handle, session = _start_real_turn(manager, tmp_path, llm)
    await _turn_ended(handle)
    session.set_pinned_target("codex")

    status = await manager.inject_system_message(
        PROJECT, "[Sub-agent] row", session_id=SID, meta=dict(meta))

    assert status == "held"
    (row,) = [m for m in session.get_messages() if m.get("content") == "[Sub-agent] row"]
    assert row["_meta"] == meta  # no suppress_wake stamped: still wake-eligible
    wake.assert_not_called()


@pytest.mark.asyncio
async def test_unpinned_session_still_wakes(manager, tmp_path):
    """The other direction: nothing changes for a session nobody pinned."""
    wake = _spy_wake(manager)
    llm = _ScriptedLLM([_reply("first turn")])
    handle, session = _start_real_turn(manager, tmp_path, llm)
    await _turn_ended(handle)

    status = await manager.inject_system_message(
        PROJECT, "[Sub-agent] codex completed.", session_id=SID,
        meta=_terminal_meta("completed"))

    assert status == "delivered"
    wake.assert_called_once_with(PROJECT, session_id=SID)


# ---------------------------------------------------------------------------
# H1 deferred branch + H2: a turn running while pinned does not wake again
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_result_deferred_while_pinned_does_not_wake_after_the_turn(
        manager, tmp_path):
    wake = _spy_wake(manager)
    llm = _ScriptedLLM([_reply("the reply")], gate_call=0)
    handle, session = _start_real_turn(manager, tmp_path, llm)
    await asyncio.wait_for(llm.in_flight.wait(), timeout=2)
    session.set_pinned_target("codex")

    status = await manager.inject_system_message(
        PROJECT, "[Sub-agent] codex completed.", session_id=SID,
        meta=_terminal_meta("completed"))
    assert status == "deferred"
    assert handle.wake_after_turn is False

    llm.release.set()
    await _turn_ended(handle)
    assert len(_held_rows(session)) == 1
    assert manager._terminal_unread(session) is True  # held, not lost
    wake.assert_not_called()


@pytest.mark.asyncio
async def test_pin_landing_mid_turn_cancels_the_wake_decided_earlier(
        manager, tmp_path):
    """H2: the row was deferred before the pin (wake decided at defer time);
    the pin lands while the turn still runs. No wake when it ends."""
    wake = _spy_wake(manager)
    llm = _ScriptedLLM([_reply("the reply")], gate_call=0)
    handle, session = _start_real_turn(manager, tmp_path, llm)
    await asyncio.wait_for(llm.in_flight.wait(), timeout=2)

    await manager.inject_system_message(
        PROJECT, "[Sub-agent] codex completed.", session_id=SID,
        meta=_terminal_meta("completed"))
    assert handle.wake_after_turn is True
    session.set_pinned_target("codex")

    llm.release.set()
    await _turn_ended(handle)
    wake.assert_not_called()
    assert manager._terminal_unread(session) is True


# ---------------------------------------------------------------------------
# H3 — the switch back: one round reads every held row
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_three_held_results_are_read_in_one_round_after_the_switch_back(
        manager, tmp_path):
    llm = _ScriptedLLM([_reply("dispatched all three"),
                        _reply("all three are in")])
    handle, session = _start_real_turn(manager, tmp_path, llm)
    await _turn_ended(handle)
    session.set_pinned_target("codex")

    for agent in ("claude-code", "codex", "dsh"):
        assert await manager.inject_system_message(
            PROJECT, f"[Sub-agent] {agent} completed. Summary: ok-{agent}.",
            session_id=SID, meta=_terminal_meta("completed", handle=agent),
        ) == "held"
    # The user's direct chat with codex meanwhile: user rows only.
    session.append({"role": "user", "content": "hi codex", "target": "codex",
                    "source": "user"})
    assert llm.calls == 1

    _wire_real_resume(manager)
    session.set_pinned_target(None)
    assert await manager.wake_if_unread(PROJECT, SID) == "delivered"
    await _turn_ended(handle)

    assert llm.calls == 2  # ONE Orbital round
    seen = json.dumps(llm.sent[1], default=str)
    for agent in ("claude-code", "codex", "dsh"):
        assert f"ok-{agent}" in seen
    # Read once: the round wrote its reply, nothing is unread, a second unpin
    # (re-pin, unpin) starts nothing.
    assert manager._terminal_unread(session) is False
    session.set_pinned_target("codex")
    session.set_pinned_target(None)
    assert await manager.wake_if_unread(PROJECT, SID) == "nothing"
    await _settle()
    assert llm.calls == 2


@pytest.mark.asyncio
async def test_retarget_to_another_worker_keeps_holding(manager, tmp_path):
    wake = _spy_wake(manager)
    llm = _ScriptedLLM([_reply("first")])
    handle, session = _start_real_turn(manager, tmp_path, llm)
    await _turn_ended(handle)
    session.set_pinned_target("codex")
    await manager.inject_system_message(
        PROJECT, "[Sub-agent] dsh completed.", session_id=SID,
        meta=_terminal_meta("completed"))

    session.set_pinned_target("claude-code")
    assert await manager.wake_if_unread(PROJECT, SID) == "pinned"
    wake.assert_not_called()


@pytest.mark.asyncio
async def test_unpin_with_nothing_held_starts_nothing(manager, tmp_path):
    wake = _spy_wake(manager)
    llm = _ScriptedLLM([_reply("first")])
    handle, session = _start_real_turn(manager, tmp_path, llm)
    await _turn_ended(handle)
    session.set_pinned_target("codex")
    # The pinned exchange's own rows are suppress_wake: not Orbital's.
    await manager.inject_system_message(
        PROJECT, "[Sub-agent] codex completed.", session_id=SID,
        meta=_terminal_meta("completed", suppress_wake=True))
    session.set_pinned_target(None)
    assert await manager.wake_if_unread(PROJECT, SID) == "nothing"
    wake.assert_not_called()


@pytest.mark.asyncio
async def test_unpin_while_orbitals_turn_still_runs_wakes_after_it(
        manager, tmp_path):
    """Pinned mid-turn, result deferred (held), unpinned before the turn
    ends, and the turn's own finally drained the row after its last reply:
    the turn never read it, so one wake follows."""
    wake = _spy_wake(manager)
    llm = _ScriptedLLM([_reply("the reply")], gate_call=0)
    handle, session = _start_real_turn(manager, tmp_path, llm)
    await asyncio.wait_for(llm.in_flight.wait(), timeout=2)
    session.set_pinned_target("codex")
    await manager.inject_system_message(
        PROJECT, "[Sub-agent] dsh completed.", session_id=SID,
        meta=_terminal_meta("completed"))
    session.set_pinned_target(None)
    assert await manager.wake_if_unread(PROJECT, SID) == "running"

    llm.release.set()
    await _turn_ended(handle)
    wake.assert_called_once_with(PROJECT, session_id=SID)


@pytest.mark.asyncio
async def test_unpin_while_running_does_not_wake_for_a_row_the_turn_read(
        manager, tmp_path):
    """Same unpin, but the row was drained after a tool batch and read by
    the turn's next LLM call: no second turn for it."""
    wake = _spy_wake(manager)
    tools = _SlowTool()
    from tests.unit.test_wake_on_deferred_terminal import _tool_call
    llm = _ScriptedLLM([_tool_call("tc1", "slow_tool"), _reply("read it")])
    handle, session = _start_real_turn(manager, tmp_path, llm, tools)
    await asyncio.wait_for(tools.running.wait(), timeout=2)
    session.set_pinned_target("codex")
    await manager.inject_system_message(
        PROJECT, "[Sub-agent] dsh completed.", session_id=SID,
        meta=_terminal_meta("completed"))
    session.set_pinned_target(None)
    assert await manager.wake_if_unread(PROJECT, SID) == "running"

    tools.release.set()
    await _turn_ended(handle)
    assert llm.calls == 2
    wake.assert_not_called()


# ---------------------------------------------------------------------------
# H1 hydrated branch + H3 on an evicted session
# ---------------------------------------------------------------------------


def _disk_session(tmp_path, stem="sess_195_evicted"):
    s = Session.new(stem, str(tmp_path))
    s.append({"role": "user", "content": "do three things", "source": "user"})
    s.append({"role": "assistant", "content": "dispatched", "source": "management"})
    s.set_pinned_target("codex")
    return stem


def _rows_on_disk(tmp_path, stem):
    p = os.path.join(str(tmp_path), "orbital", "sessions", f"{stem}.jsonl")
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.mark.asyncio
async def test_evicted_pinned_session_holds_on_disk_then_wakes_on_unpin(
        manager, tmp_path):
    stem = _disk_session(tmp_path)
    manager._project_store.get_project = MagicMock(
        return_value={"workspace": str(tmp_path)})
    manager.start_agent = AsyncMock()
    manager._abuild_agent_config_from_project = AsyncMock(return_value="cfg")

    status = await manager.inject_system_message(
        PROJECT, "[Sub-agent] codex completed.", session_id=stem,
        meta=_terminal_meta("completed"))
    assert status == "held"
    manager.start_agent.assert_not_awaited()
    (row,) = [r for r in _rows_on_disk(tmp_path, stem)
              if r.get("content") == "[Sub-agent] codex completed."]
    assert "suppress_wake" not in row["_meta"]  # unchanged on disk

    # The unpin (the PATCH writes the meta first, then wakes).
    manager.patch_session(PROJECT, stem, pinned_target=None)
    assert await manager.wake_if_unread(PROJECT, stem) == "delivered"
    manager.start_agent.assert_awaited_once()
    assert manager.start_agent.await_args.kwargs["session"].session_uuid == stem


@pytest.mark.asyncio
async def test_evicted_session_unpinned_with_nothing_held_is_not_started(
        manager, tmp_path):
    stem = _disk_session(tmp_path)
    manager._project_store.get_project = MagicMock(
        return_value={"workspace": str(tmp_path)})
    manager.start_agent = AsyncMock()
    manager.patch_session(PROJECT, stem, pinned_target=None)
    assert await manager.wake_if_unread(PROJECT, stem) == "nothing"
    manager.start_agent.assert_not_awaited()
