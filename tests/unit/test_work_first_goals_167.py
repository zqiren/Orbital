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
    # Live (§8): deepseek-v4.1-flash took "nothing to record" over the draft
    # in 2 of 7 runs until the bullet said the draft is not optional (7/7).
    assert "even if nothing else in this run is worth recording" in flat
    # Live (§8): Codex copied its seat rules (orbital/bin/bg, MCP dispatch)
    # into the first goals; the bullet rules that out.
    assert ("never Orbital's or your own operating rules (background jobs, "
            "dispatch, seat rules)") in flat
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
    assert "never Orbital's or your own operating rules" in " ".join(asked.split())


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


def test_a_second_post_while_the_first_card_is_open_posts_nothing(tmp_path):
    """§5.2 race (found live, §8 case 9): the scan and a first conversation's
    record turn both see the goals file appear and both post. The card text
    carries its options, which ``open_ask``'s same-text check does not see, so
    the guard is the project's own open card: while it is open, no second."""
    import asyncio as _asyncio

    from agent_os.agent import asks
    from agent_os.api.routes import agents_v2
    from agent_os.agent.session import Session
    from tests.unit.keyless_harness import build_client, create_project

    tc, deps = build_client(tmp_path, keyed=False)
    ws = tmp_path / "ws"; ws.mkdir()
    pid = create_project(tc, ws)["project_id"]
    _write_goals(ws)
    for sid in ("scan", "task"):
        Session.new(sid, str(ws), session_id=sid).append({"role": "assistant", "content": "ok"})
    first = _asyncio.run(agents_v2.post_goals_confirm(pid, "scan"))
    second = _asyncio.run(agents_v2.post_goals_confirm(pid, "task"))
    assert first and second is None
    assert [a.id for a in asks.read_asks(str(ws / "orbital")) if a.is_open] == [first]
    assert deps["project_store"].get_project(pid)["import_scan_ask"]["session_id"] == "scan"
    # Once answered, a later draft (goals emptied and redrafted) asks again.
    asks.append_event(str(ws / "orbital"), "done", first, "Yes")
    assert _asyncio.run(agents_v2.post_goals_confirm(pid, "task"))


# ---------------------------------------------------------------------------
# Batch 1110 item 8: the card is never lost
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_goals_written_by_the_run_with_no_record_turn_still_post_at_run_end(tmp_path):
    """The main run wrote the goals but its record turn never ran (here:
    recording disabled, as with a queued user message): the run end posts."""
    llm = _ScriptedLLM([_write_goals_call(), _reply("Done, goals written.")])
    session, tools, loop = _rig(tmp_path, llm)
    _drop_goals(tmp_path)
    loop._memory_record_enabled = False
    hook = AsyncMock()
    loop.on_goals_created = hook
    await loop.run()
    assert not project_goals_missing(str(tmp_path))
    hook.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_pending_draft_stamp_makes_a_run_with_goals_post_the_card(tmp_path):
    """The goals already exist as this run starts (an earlier run drafted them
    and was cut off before its card): the AgentManager's stamp tells this
    run's end to post, once."""
    _write_goals(tmp_path)
    llm = _ScriptedLLM([_reply("ok")])
    session, tools, loop = _rig(tmp_path, llm)
    loop._memory_record_enabled = False
    hook = AsyncMock()
    loop.on_goals_created = hook
    loop.goals_draft_pending = True
    await loop.run()
    hook.assert_awaited_once()
    assert loop.goals_draft_pending is False
    # And without the stamp, a project with goals never asks again.
    hook.reset_mock()
    await loop.run()
    hook.assert_not_awaited()


def test_manager_stamps_a_goal_less_project_at_run_start_and_tells_the_loop(tmp_path):
    from unittest.mock import MagicMock as _MM
    mgr, _sub = wmr._manager(tmp_path)
    store = mgr._project_store
    state = {"workspace": str(tmp_path)}
    store.get_project = _MM(side_effect=lambda pid: dict(state))
    def _update(pid, fields):
        state.update(fields)
    store.update_project = _MM(side_effect=_update)
    loop = _MM()
    loop.goals_draft_pending = False
    mgr._note_goals_state_at_run_start(PROJECT_W, loop)
    assert state["goals_draft_pending"] is True
    assert loop.goals_draft_pending is True
    # With goals present and no stamp, nothing is stamped and the loop is told so.
    _write_goals(tmp_path)
    state.pop("goals_draft_pending")
    loop.goals_draft_pending = True
    mgr._note_goals_state_at_run_start(PROJECT_W, loop)
    assert "goals_draft_pending" not in state
    assert loop.goals_draft_pending is False


@pytest.mark.asyncio
async def test_worker_terminal_after_a_restart_posts_from_the_stamp(tmp_path):
    """The in-memory set of drafting reminders is gone (daemon restart), but
    the project store still says a draft is owed: the memory-record terminal
    posts when the goals exist."""
    from unittest.mock import MagicMock as _MM
    from agent_os.agent.loop import MEMORY_RECORD_KIND
    mgr, sub = wmr._manager(tmp_path)
    mgr._confirm_goals = AsyncMock()
    state = {"workspace": str(tmp_path), "goals_draft_pending": True}
    mgr._project_store.get_project = _MM(side_effect=lambda pid: dict(state))
    wmr._plant_session(mgr, tmp_path)
    _write_goals(tmp_path)
    assert mgr._worker_goals_drafts == set()
    await mgr.inject_system_message(
        PROJECT_W, "[Sub-agent] codex completed. Summary: nothing to record.",
        session_id=wmr.SID, meta={**wmr._completed_meta(), "_kind": MEMORY_RECORD_KIND})
    # inject_system_message tags the row's _kind through meta? If not, send the
    # row the way the reminder's terminal lands.
    await wmr._settle()
    if not mgr._confirm_goals.await_count:
        row = {"role": "system", "source": "daemon", "_kind": MEMORY_RECORD_KIND,
               "content": "[Sub-agent] codex completed. Summary: nothing to record.",
               "_meta": wmr._completed_meta()}
        mgr._maybe_confirm_worker_goals(PROJECT_W, wmr.SID, row)
        await wmr._settle()
    mgr._confirm_goals.assert_awaited_once_with(PROJECT_W, wmr.SID)


def test_confirm_goals_clears_the_stamp(tmp_path):
    from unittest.mock import MagicMock as _MM
    mgr, _sub = wmr._manager(tmp_path)
    state = {"workspace": str(tmp_path), "goals_draft_pending": True}
    mgr._project_store.get_project = _MM(side_effect=lambda pid: dict(state))
    mgr._project_store.update_project = _MM(side_effect=lambda pid, f: state.update(f))
    import asyncio as _aio
    from unittest.mock import patch as _patch
    with _patch("agent_os.api.routes.agents_v2.post_goals_confirm", AsyncMock(return_value="ask1")):
        _aio.run(mgr._confirm_goals(PROJECT_W, wmr.SID))
    assert state["goals_draft_pending"] is False
