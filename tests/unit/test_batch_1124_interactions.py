# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Batch 1124: interactions between specs 195, 196 and 197 that no single
builder could see.

196's hidden ``files_restored`` row and 195's hold-while-pinned gate both go
through ``inject_system_message``. A restore made while the session is pinned
to a worker must neither wake Orbital nor be lost, and it must not count as
an unread terminal that starts an Orbital turn when the user switches back.
A result held while pinned must still wake ONE round on unpin when a restore
row sits after it, and that round reads both.

The rows are written by the real route helper (``history._tell_session``),
so the meta under test is the one the daemon stamps.
"""

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.session import Session
from agent_os.api.routes import history as history_routes
from tests.unit.test_wake_on_deferred_terminal import (
    PROJECT, SID, _ScriptedLLM, _reply, _spy_wake, _start_real_turn,
    _terminal_meta, _turn_ended, manager,  # noqa: F401 (fixture)
)

_OK = [{"path": "notes.md", "status": "ok"}, {"path": "logo.png", "status": "ok"}]


@pytest.fixture
def routed(manager, monkeypatch):
    monkeypatch.setattr(history_routes, "_agent_manager", manager)
    return manager


def _restored_rows(session) -> list[dict]:
    return [m for m in session.get_messages()
            if (m.get("_meta") or {}).get("event") == "files_restored"]


def _wire_real_resume(manager) -> None:
    manager._abuild_agent_config_from_project = AsyncMock(
        side_effect=RuntimeError("no project in this test"))
    manager._note_goals_state_at_run_start = MagicMock()


@pytest.mark.asyncio
async def test_restore_while_pinned_and_idle_lands_without_waking(routed, tmp_path):
    wake = _spy_wake(routed)
    llm = _ScriptedLLM([_reply("first turn")])
    handle, session = _start_real_turn(routed, tmp_path, llm)
    await _turn_ended(handle)
    session.set_pinned_target("codex")

    await history_routes._tell_session(PROJECT, SID, None, _OK)

    (row,) = _restored_rows(session)  # not lost
    assert row["_meta"]["paths"] == ["notes.md", "logo.png"]
    wake.assert_not_called()
    # Not an unread terminal: the switch back starts nothing.
    assert routed._terminal_unread(session) is False
    session.set_pinned_target(None)
    assert await routed.wake_if_unread(PROJECT, SID) == "nothing"
    wake.assert_not_called()
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_restore_during_a_running_turn_while_pinned_is_kept_and_not_woken(
        routed, tmp_path):
    wake = _spy_wake(routed)
    llm = _ScriptedLLM([_reply("the reply")], gate_call=0)
    handle, session = _start_real_turn(routed, tmp_path, llm)
    await asyncio.wait_for(llm.in_flight.wait(), timeout=2)
    session.set_pinned_target("codex")

    await history_routes._tell_session(PROJECT, SID, None, _OK)
    assert handle.wake_after_turn is False

    llm.release.set()
    await _turn_ended(handle)
    assert len(_restored_rows(session)) == 1  # drained, not lost
    wake.assert_not_called()
    session.set_pinned_target(None)
    assert await routed.wake_if_unread(PROJECT, SID) == "nothing"
    wake.assert_not_called()


@pytest.mark.asyncio
async def test_held_result_then_restore_wakes_one_round_that_reads_both(
        routed, tmp_path):
    llm = _ScriptedLLM([_reply("dispatched"), _reply("read both")])
    handle, session = _start_real_turn(routed, tmp_path, llm)
    await _turn_ended(handle)
    session.set_pinned_target("codex")

    assert await routed.inject_system_message(
        PROJECT, "[Sub-agent] dsh completed. Summary: ok-dsh.", session_id=SID,
        meta=_terminal_meta("completed", handle="dsh")) == "held"
    await history_routes._tell_session(PROJECT, SID, None, _OK)
    # The restore row after the held result does not hide it.
    assert routed._terminal_unread(session) is True

    _wire_real_resume(routed)
    session.set_pinned_target(None)
    assert await routed.wake_if_unread(PROJECT, SID) == "delivered"
    await _turn_ended(handle)
    assert llm.calls == 2  # ONE round
    seen = json.dumps(llm.sent[1], default=str)
    assert "ok-dsh" in seen
    assert "logo.png" in seen


@pytest.mark.asyncio
async def test_restore_on_an_evicted_pinned_session_is_persisted_without_a_start(
        routed, tmp_path):
    stem = "sess_1124_evicted"
    s = Session.new(stem, str(tmp_path))
    s.append({"role": "user", "content": "edit things", "source": "user"})
    s.append({"role": "assistant", "content": "done", "source": "management"})
    s.set_pinned_target("codex")
    routed._project_store.get_project = MagicMock(
        return_value={"workspace": str(tmp_path)})
    routed.start_agent = AsyncMock()

    await history_routes._tell_session(PROJECT, stem, None, _OK)

    p = os.path.join(str(tmp_path), "orbital", "sessions", f"{stem}.jsonl")
    with open(p, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert [r for r in rows if (r.get("_meta") or {}).get("event") == "files_restored"]
    routed.start_agent.assert_not_awaited()
    routed.patch_session(PROJECT, stem, pinned_target=None)
    assert await routed.wake_if_unread(PROJECT, stem) == "nothing"
    routed.start_agent.assert_not_awaited()
