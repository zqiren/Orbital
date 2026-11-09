# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 167: project goals come from the first run, not from an interview.

When a run ends on a project with no (or an empty)
``orbital/instructions/project_goals.md``, the end-of-run record turn that
already writes PROJECT_STATE / DECISIONS / LESSONS / INDEX drafts the goals
first, from the folder and the conversation. When the goals file comes into
existence that way, the daemon posts the spec 160 confirm card (这样理解对吗？
对 | 需要改) — once, through ``post_goals_confirm``.

Manager: the [MEMORY] row gains the goals bullet; its tool scope admits the
goals path (write while missing, read + exact edit after); missing goals
override the already-wrote-memory skip. Worker: the "[Orbital memory]" text
gains the same bullet, and the memory turn's terminal row is the hook.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock

import pytest

from agent_os.agent.loop import (
    MEMORY_SHAPE_RULES,
    build_memory_record_text,
    build_worker_memory_record_text,
    project_goals_missing,
)
from tests.unit.test_memory_record_turn import (
    _ScriptedLLM, _edit_state_call, _memory_rows, _reply, _rig, _tool_calls,
)
from tests.unit import test_worker_memory_record as wmr


GOALS_REL = "orbital/instructions/project_goals.md"


def _goals_path(ws) -> str:
    return os.path.join(str(ws), "orbital", "instructions", "project_goals.md")


def _drop_goals(ws) -> None:
    try:
        os.remove(_goals_path(ws))
    except FileNotFoundError:
        pass


def _write_goals(ws, text="# Goals\nShip the launch page.\n") -> None:
    os.makedirs(os.path.dirname(_goals_path(ws)), exist_ok=True)
    with open(_goals_path(ws), "w", encoding="utf-8") as fh:
        fh.write(text)


def _write_goals_call(call_id="tc_goals", content="First draft.\nA launch page for the shop.\n- works in Chinese\n"):
    return _tool_calls((call_id, "write", {"path": GOALS_REL, "content": content}))


# ---------------------------------------------------------------------------
# The condition: missing or empty
# ---------------------------------------------------------------------------


def test_goals_missing_means_missing_or_whitespace(tmp_path):
    assert project_goals_missing(str(tmp_path)) is True
    _write_goals(tmp_path, "   \n\t\n")
    assert project_goals_missing(str(tmp_path)) is True
    _write_goals(tmp_path, "# Goals\n- x\n")
    assert project_goals_missing(str(tmp_path)) is False
    assert project_goals_missing("") is False  # no workspace: nothing to draft


# ---------------------------------------------------------------------------
# The texts
# ---------------------------------------------------------------------------


def test_manager_row_leads_with_the_goals_bullet_only_while_goals_are_missing(tmp_path):
    ws = str(tmp_path)
    text = build_memory_record_text(ws, [], [])
    assert "no goals yet" in text and GOALS_REL in text
    flat = " ".join(text.split())
    assert "draft" in flat.lower() and "this conversation" in flat
    # First: before the state instruction.
    assert text.index("no goals yet") < text.index("update orbital/PROJECT_STATE.md")
    _write_goals(tmp_path)
    assert "no goals yet" not in build_memory_record_text(ws, [], [])


def test_worker_text_carries_the_same_bullet_only_when_asked():
    asked = build_worker_memory_record_text(["src/a.py"], goals_missing=True)
    plain = build_worker_memory_record_text(["src/a.py"])
    assert "no goals yet" in asked and "no goals yet" not in plain
    assert asked.index("no goals yet") < asked.index("update orbital/PROJECT_STATE.md")
    assert asked.endswith("If nothing durable happened, reply exactly: nothing to record")
    assert "INDEX" not in asked  # workers never touch INDEX


def test_shape_rules_carry_the_one_supersedes_line():
    assert "supersedes: " in MEMORY_SHAPE_RULES
    assert "project_goals.md" in MEMORY_SHAPE_RULES


# ---------------------------------------------------------------------------
# Manager record turn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_goal_less_run_that_wrote_state_mid_run_still_gets_its_record_turn(tmp_path):
    llm = _ScriptedLLM([_edit_state_call("tc_main"), _reply("Updated the state."),
                        _write_goals_call(), _reply("Recorded.")])
    session, tools, loop = _rig(tmp_path, llm)
    _drop_goals(tmp_path)
    await loop.run()
    assert len(_memory_rows(session)) == 1
    assert "no goals yet" in _memory_rows(session)[0]["content"]
    assert not project_goals_missing(str(tmp_path))


@pytest.mark.asyncio
async def test_run_with_goals_that_wrote_state_is_still_not_asked(tmp_path):
    llm = _ScriptedLLM([_edit_state_call("tc_main"), _reply("Updated the state.")])
    session, tools, loop = _rig(tmp_path, llm)
    await loop.run()
    assert _memory_rows(session) == []


@pytest.mark.asyncio
async def test_record_turn_writes_goals_while_missing_and_the_card_hook_fires_once(tmp_path):
    llm = _ScriptedLLM([_reply("Renamed the file."), _write_goals_call(), _reply("Recorded.")])
    session, tools, loop = _rig(tmp_path, llm)
    _drop_goals(tmp_path)
    hook = AsyncMock()
    loop.on_goals_created = hook
    await loop.run()
    assert ("write", {"path": GOALS_REL, "content": "First draft.\nA launch page for the shop.\n- works in Chinese\n"}) in tools.executed
    hook.assert_awaited_once()


@pytest.mark.asyncio
async def test_empty_goals_file_is_drafted_again(tmp_path):
    llm = _ScriptedLLM([_reply("ok"), _edit_goals_whole_call(), _reply("Recorded.")])
    session, tools, loop = _rig(tmp_path, llm)
    _write_goals(tmp_path, "  \n")
    hook = AsyncMock()
    loop.on_goals_created = hook
    await loop.run()
    assert "no goals yet" in _memory_rows(session)[0]["content"]
    assert not project_goals_missing(str(tmp_path))
    hook.assert_awaited_once()


def _edit_goals_whole_call():
    # An existing (empty) file is edited, never overwritten (spec 155 §5.1.4).
    return _tool_calls(("tc_ge", "edit", {"path": GOALS_REL, "old_text": "  \n",
                                          "new_text": "First draft.\nA shop page.\n"}))


@pytest.mark.asyncio
async def test_existing_goals_take_read_and_exact_edit_but_never_write(tmp_path):
    llm = _ScriptedLLM([
        _reply("ok"),
        _tool_calls(("tc_w", "write", {"path": GOALS_REL, "content": "clobbered"}),
                    ("tc_r", "read", {"path": GOALS_REL}),
                    ("tc_e", "edit", {"path": GOALS_REL, "old_text": "Ship the launch page.",
                                      "new_text": "Ship the launch page and the blog.\nsupersedes: Ship the launch page."})),
        _reply("Recorded."),
    ])
    session, tools, loop = _rig(tmp_path, llm)
    hook = AsyncMock()
    loop.on_goals_created = hook
    await loop.run()
    results = {m["tool_call_id"]: m["content"] for m in session.get_messages() if m.get("role") == "tool"}
    assert results["tc_w"].startswith("Error")
    assert results["tc_r"].startswith("# Goals")
    assert results["tc_e"].startswith("Edited")
    with open(_goals_path(tmp_path), encoding="utf-8") as fh:
        assert "and the blog" in fh.read()
    # Goals existed when the run started: no card.
    hook.assert_not_awaited()


@pytest.mark.asyncio
async def test_other_orbital_files_stay_out_of_the_record_turn(tmp_path):
    llm = _ScriptedLLM([
        _reply("ok"),
        _tool_calls(("tc_x", "write", {"path": "orbital/instructions/user_directives.md", "content": "x"})),
        _reply("Recorded."),
    ])
    session, tools, loop = _rig(tmp_path, llm)
    await loop.run()
    results = {m["tool_call_id"]: m["content"] for m in session.get_messages() if m.get("role") == "tool"}
    assert results["tc_x"].startswith("Error")
    assert "project_goals.md" in results["tc_x"]


@pytest.mark.asyncio
async def test_record_turn_that_skips_the_goals_draft_posts_no_card(tmp_path):
    llm = _ScriptedLLM([_reply("ok"), _reply("nothing to record")])
    session, tools, loop = _rig(tmp_path, llm)
    _drop_goals(tmp_path)
    hook = AsyncMock()
    loop.on_goals_created = hook
    await loop.run()
    hook.assert_not_awaited()
    assert project_goals_missing(str(tmp_path))


@pytest.mark.asyncio
async def test_goals_written_by_the_run_itself_post_the_card_at_its_record_turn(tmp_path):
    """The run (not the record turn) created the file: it did not exist when
    the run started and exists at the record turn's end — the card follows."""
    llm = _ScriptedLLM([_write_goals_call("tc_main"), _reply("Wrote the goals."),
                        _reply("nothing to record")])
    session, tools, loop = _rig(tmp_path, llm)
    _drop_goals(tmp_path)
    hook = AsyncMock()
    loop.on_goals_created = hook
    await loop.run()
    assert len(_memory_rows(session)) == 1
    assert "no goals yet" not in _memory_rows(session)[0]["content"]
    hook.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_failing_card_hook_never_fails_the_run(tmp_path):
    llm = _ScriptedLLM([_reply("ok"), _write_goals_call(), _reply("Recorded.")])
    session, tools, loop = _rig(tmp_path, llm)
    _drop_goals(tmp_path)
    loop.on_goals_created = AsyncMock(side_effect=RuntimeError("boom"))
    await loop.run()
    assert loop.get_completion_state()[0] == "text"


@pytest.mark.asyncio
async def test_stopped_run_leaves_goals_missing_for_the_next_run(tmp_path):
    llm = _ScriptedLLM([_reply("unreached")], gate_call=0)
    session, tools, loop = _rig(tmp_path, llm)
    _drop_goals(tmp_path)
    task = asyncio.create_task(loop.run())
    await asyncio.wait_for(llm.in_flight.wait(), timeout=2)
    await loop.cancel_turn()
    llm.release.set()
    await task
    assert _memory_rows(session) == []
    assert project_goals_missing(str(tmp_path))


# ---------------------------------------------------------------------------
# Pinned worker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_goal_less_worker_turn_is_asked_for_goals_even_after_writing_state(tmp_path):
    mgr, sub = wmr._manager(tmp_path, tool_rows=[
        {"tool_name": "Edit", "arguments": {"file_path": "orbital/PROJECT_STATE.md"}},
        {"tool_name": "Write", "arguments": {"file_path": "src/x.py"}},
    ])
    wmr._plant_session(mgr, tmp_path)
    await mgr.inject_system_message(PROJECT_W, "[Sub-agent] codex completed. Summary: Done.",
                                    session_id=wmr.SID, meta=wmr._completed_meta())
    await wmr._settle()
    sub.send.assert_awaited_once()
    assert "no goals yet" in sub.send.call_args.args[2]


@pytest.mark.asyncio
async def test_worker_with_goals_gets_the_plain_text(tmp_path):
    _write_goals(tmp_path)
    mgr, sub = wmr._manager(tmp_path)
    wmr._plant_session(mgr, tmp_path)
    await mgr.inject_system_message(PROJECT_W, "[Sub-agent] codex completed. Summary: Done.",
                                    session_id=wmr.SID, meta=wmr._completed_meta())
    await wmr._settle()
    text = sub.send.call_args.args[2]
    assert "no goals yet" not in text


PROJECT_W = wmr.PROJECT


async def _finish_memory_turn(mgr, sub):
    """The memory turn's dispatch marker and its terminal row land."""
    dispatch_id = sub.send.call_args.kwargs["dispatch_id"]
    await mgr.inject_system_message(
        PROJECT_W, f'[Sub-agent] Message sent to {wmr.HANDLE}: "[Orbital memory] …". Transcript: t.jsonl',
        session_id=wmr.SID, meta={"handle": wmr.HANDLE, "dispatch_id": dispatch_id,
                                  "suppress_wake": True})
    await mgr.inject_system_message(
        PROJECT_W, "[Sub-agent] codex completed. Summary: nothing to record.",
        session_id=wmr.SID, meta=wmr._completed_meta(dispatch_id=dispatch_id))
    await wmr._settle()


@pytest.mark.asyncio
async def test_worker_memory_turn_that_created_goals_posts_the_card(tmp_path):
    mgr, sub = wmr._manager(tmp_path)
    mgr._confirm_goals = AsyncMock()
    wmr._plant_session(mgr, tmp_path)
    await mgr.inject_system_message(PROJECT_W, "[Sub-agent] codex completed. Summary: Done.",
                                    session_id=wmr.SID, meta=wmr._completed_meta())
    await wmr._settle()
    _write_goals(tmp_path)  # the worker wrote them in its memory turn
    await _finish_memory_turn(mgr, sub)
    mgr._confirm_goals.assert_awaited_once_with(PROJECT_W, wmr.SID)


@pytest.mark.asyncio
async def test_worker_memory_turn_that_skipped_the_draft_posts_nothing(tmp_path):
    mgr, sub = wmr._manager(tmp_path)
    mgr._confirm_goals = AsyncMock()
    wmr._plant_session(mgr, tmp_path)
    await mgr.inject_system_message(PROJECT_W, "[Sub-agent] codex completed. Summary: Done.",
                                    session_id=wmr.SID, meta=wmr._completed_meta())
    await wmr._settle()
    await _finish_memory_turn(mgr, sub)
    mgr._confirm_goals.assert_not_awaited()


@pytest.mark.asyncio
async def test_worker_memory_turn_on_a_project_with_goals_posts_nothing(tmp_path):
    _write_goals(tmp_path)
    mgr, sub = wmr._manager(tmp_path)
    mgr._confirm_goals = AsyncMock()
    wmr._plant_session(mgr, tmp_path)
    await mgr.inject_system_message(PROJECT_W, "[Sub-agent] codex completed. Summary: Done.",
                                    session_id=wmr.SID, meta=wmr._completed_meta())
    await wmr._settle()
    await _finish_memory_turn(mgr, sub)
    mgr._confirm_goals.assert_not_awaited()


# ---------------------------------------------------------------------------
# The card's language on a project that was not imported
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ui,expect", [("zh-CN", "这样理解对吗？"), ("en-US", "Did I get this right?"),
                                       ("ja", None)])
def test_a_new_projects_card_follows_the_ui_language(tmp_path, monkeypatch, ui, expect):
    """Spec 160's card took the import's UI language. A project created by
    hand has none, so its card follows the language the app reports
    (``PUT /settings/ui-language``), else the system's."""
    import asyncio as _asyncio

    from agent_os.agent import asks
    from agent_os.api.routes import agents_v2
    from agent_os.agent.session import Session
    from agent_os.onboarding import deferred_seed
    from tests.unit.keyless_harness import build_client, create_project

    monkeypatch.setattr(deferred_seed, "default_lang", lambda: "en")
    tc, deps = build_client(tmp_path, keyed=False)
    ws = tmp_path / "ws"; ws.mkdir()
    pid = create_project(tc, ws)["project_id"]
    deps["settings_store"].get().ui_language = ui  # what PUT /settings/ui-language stores
    _write_goals(ws)
    Session.new("s1", str(ws), session_id="s1").append({"role": "assistant", "content": "ok"})
    ask_id = _asyncio.run(agents_v2.post_goals_confirm(pid, "s1"))
    (ask,) = [a for a in asks.read_asks(str(ws / "orbital")) if a.id == ask_id]
    assert ask.text == (expect or "Did I get this right?")
