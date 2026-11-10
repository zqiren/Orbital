# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 169 A + D (backend).

A. Quick Tasks (the scratch project, ``is_scratch: true``) never records
memory, never drafts goals and never gets the goals confirm card, whichever
agent is pinned. The worker record turn is gated at its scheduler, the
goals-owed stamp is not set at a manager run's start, and the card's single
owner (``post_goals_confirm``) refuses the project, which also covers the
manager's run-end ``finally`` route with a stale goals file.

D. The goals confirm card carries the goals file as a markdown link at the
top of its detail, and its language is the draft's: zh when CJK is a real
share of the file's letters, else the existing ``_card_lang`` rule.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent import asks
from agent_os.agent.loop import MEMORY_RECORD_KIND
from tests.unit import test_worker_memory_record as wmr
from tests.unit.test_memory_record_turn import _ScriptedLLM, _reply, _rig


GOALS_REL = "orbital/instructions/project_goals.md"
GOALS_LINK = f"[project_goals.md]({GOALS_REL})"

ZH_GOALS = ("# First draft\n\n这是一个随手处理小任务的地方。用户用中文交流，"
            "希望回答简短直接。\n\n- 当前任务的完成标准：回复已展示给用户\n")
EN_GOALS = ("# First draft\n\nA launch page for the shop, written in plain words.\n\n"
            "- done when the page is live\n- the user prefers short replies\n")
# An English draft that quotes a couple of Chinese file names: still English.
MIXED_EN_GOALS = ("# First draft\n\nTranslate the onboarding copy kept in 欢迎.md and "
                  "定价.md into English for the launch, and keep the tone friendly.\n\n"
                  "- done when every screen has English copy\n")


def _write_goals(ws, text=EN_GOALS) -> None:
    path = os.path.join(str(ws), "orbital", "instructions", "project_goals.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _write_next_steps(ws, steps) -> None:
    path = os.path.join(str(ws), "orbital", "PROJECT_STATE.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# State\n\n## Next steps\n" + "".join(f"- {s}\n" for s in steps))


def _client(tmp_path, monkeypatch, *, ui="en-US"):
    from agent_os.onboarding import deferred_seed
    from tests.unit.keyless_harness import build_client
    monkeypatch.setattr(deferred_seed, "default_lang", lambda: "en")
    tc, deps = build_client(tmp_path, keyed=False)
    deps["settings_store"].get().ui_language = ui
    return tc, deps


def _project(tc, ws, **extra) -> str:
    from tests.unit.keyless_harness import create_project
    return create_project(tc, ws, **extra)["project_id"]


def _session(ws, sid="s1"):
    from agent_os.agent.session import Session
    s = Session.new(sid, str(ws), session_id=sid)
    s.append({"role": "assistant", "content": "ok"})
    return s


def _open_asks(ws):
    return [a for a in asks.read_asks(os.path.join(str(ws), "orbital")) if a.is_open]


def _scratch_store(tmp_path, **extra):
    """A project store whose project is Quick Tasks, recording updates."""
    state = {"workspace": str(tmp_path), "is_scratch": True, **extra}
    store = MagicMock()
    store.get_project = MagicMock(side_effect=lambda pid: dict(state))
    store.update_project = MagicMock(side_effect=lambda pid, f: state.update(f))
    return store, state


# ---------------------------------------------------------------------------
# A1. The worker record turn: gated in Quick Tasks, intact elsewhere
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_quick_tasks_pinned_worker_completion_gets_no_record_turn(tmp_path):
    mgr, sub = wmr._manager(tmp_path)
    store, state = _scratch_store(tmp_path)
    mgr._project_store = store
    mgr._send_worker_daemon_turn = AsyncMock()
    session = wmr._plant_session(mgr, tmp_path)
    await mgr.inject_system_message(wmr.PROJECT, "[Sub-agent] codex completed. Summary: Done.",
                                    session_id=wmr.SID, meta=wmr._completed_meta())
    await wmr._settle()
    mgr._send_worker_daemon_turn.assert_not_awaited()
    sub.send.assert_not_awaited()
    assert not [m for m in session.get_messages() if m.get("_kind") == MEMORY_RECORD_KIND]
    # No goals-owed stamp either (the goals file is missing here).
    assert "goals_draft_pending" not in state


@pytest.mark.asyncio
async def test_normal_project_pinned_worker_completion_still_records(tmp_path):
    """Spec 112 A6 protection: the gate is Quick Tasks only."""
    mgr, sub = wmr._manager(tmp_path)
    wmr._plant_session(mgr, tmp_path)
    await mgr.inject_system_message(wmr.PROJECT, "[Sub-agent] codex completed. Summary: Done.",
                                    session_id=wmr.SID, meta=wmr._completed_meta())
    await wmr._settle()
    sub.send.assert_awaited_once()
    assert "[Orbital memory]" in sub.send.call_args.args[2]


@pytest.mark.asyncio
async def test_quick_tasks_worker_terminal_still_notes_ask_changes(tmp_path):
    """``_note_worker_ask_changes`` stays above the gate: a worker's own ask
    still gets its card row in Quick Tasks."""
    mgr, _sub = wmr._manager(tmp_path)
    store, _state = _scratch_store(tmp_path)
    mgr._project_store = store
    session = wmr._plant_session(mgr, tmp_path)
    mgr._worker_asks_seen[(wmr.PROJECT, wmr.SID)] = set()
    asks.open_ask(os.path.join(str(tmp_path), "orbital"), "Which colour for the logo?")
    await mgr.inject_system_message(wmr.PROJECT, "[Sub-agent] codex completed. Summary: Done.",
                                    session_id=wmr.SID, meta=wmr._completed_meta())
    await wmr._settle()
    assert [m for m in session.get_messages() if m.get("_kind") == "ask_card"]


# ---------------------------------------------------------------------------
# A2. No goals-owed stamp at a Quick Tasks manager run's start
# ---------------------------------------------------------------------------


def test_quick_tasks_manager_run_start_stamps_no_goals_draft(tmp_path):
    mgr, _sub = wmr._manager(tmp_path)
    store, state = _scratch_store(tmp_path)
    mgr._project_store = store
    loop = MagicMock()
    loop.goals_draft_pending = False
    mgr._note_goals_state_at_run_start(wmr.PROJECT, loop)
    assert "goals_draft_pending" not in state
    assert loop.goals_draft_pending is False


# ---------------------------------------------------------------------------
# A3. The card's owner refuses Quick Tasks
# ---------------------------------------------------------------------------


def test_post_goals_confirm_refuses_quick_tasks(tmp_path, monkeypatch):
    from agent_os.api.routes import agents_v2
    from agent_os.onboarding.import_scan import ASK_FIELD
    tc, deps = _client(tmp_path, monkeypatch)
    ws = tmp_path / "scratch"; ws.mkdir()
    pid = _project(tc, ws, name="Quick Tasks", is_scratch=True)
    assert deps["project_store"].get_project(pid)["is_scratch"] is True
    _write_goals(ws, ZH_GOALS)
    _session(ws)
    assert asyncio.run(agents_v2.post_goals_confirm(pid, "s1")) is None
    assert _open_asks(ws) == []
    assert ASK_FIELD not in deps["project_store"].get_project(pid)


@pytest.mark.asyncio
async def test_quick_tasks_manager_run_end_with_a_stale_goals_file_posts_no_card(
        tmp_path, monkeypatch):
    """The run's ``finally`` route: a goals file left over from an earlier
    build and a stale ``goals_draft_pending`` stamp. The loop calls the
    AgentManager's hook (the real one); the card's owner posts nothing."""
    tc, deps = _client(tmp_path, monkeypatch)
    llm = _ScriptedLLM([_reply("ok")])
    session, _tools, loop = _rig(tmp_path, llm)
    pid = _project(tc, tmp_path, name="Quick Tasks", is_scratch=True)
    deps["project_store"].update_project(pid, {"goals_draft_pending": True})
    _write_goals(tmp_path, ZH_GOALS)
    mgr = deps["agent_manager"]
    loop._memory_record_enabled = False
    loop.on_goals_created = mgr._goals_confirm_hook(pid, session.session_id)
    loop.goals_draft_pending = True
    await loop.run()
    await asyncio.sleep(0)
    assert _open_asks(tmp_path) == []
    assert not [m for m in session.get_messages() if m.get("_kind") == "ask_card"]


@pytest.mark.asyncio
async def test_normal_project_manager_run_end_still_opens_the_card(tmp_path, monkeypatch):
    """Control for the test above: the same route on a normal project asks."""
    tc, deps = _client(tmp_path, monkeypatch)
    llm = _ScriptedLLM([_reply("ok")])
    session, _tools, loop = _rig(tmp_path, llm)
    pid = _project(tc, tmp_path, name="shop")
    _write_goals(tmp_path, EN_GOALS)
    mgr = deps["agent_manager"]
    loop._memory_record_enabled = False
    loop.on_goals_created = mgr._goals_confirm_hook(pid, session.session_id)
    loop.goals_draft_pending = True
    await loop.run()
    await asyncio.sleep(0)
    assert [a.text for a in _open_asks(tmp_path)] == ["Did I get this right?"]


# ---------------------------------------------------------------------------
# D1. The goals file rides on the card
# ---------------------------------------------------------------------------


def test_card_detail_is_the_goals_link_when_there_are_no_steps(tmp_path, monkeypatch):
    from agent_os.api.routes import agents_v2
    tc, _deps = _client(tmp_path, monkeypatch)
    ws = tmp_path / "ws"; ws.mkdir()
    pid = _project(tc, ws)
    _write_goals(ws, EN_GOALS)
    _session(ws)
    ask_id = asyncio.run(agents_v2.post_goals_confirm(pid, "s1"))
    (ask,) = [a for a in _open_asks(ws) if a.id == ask_id]
    assert ask.detail.startswith(GOALS_LINK)
    assert ask.detail.strip() == GOALS_LINK


def test_card_detail_puts_the_goals_link_above_the_steps(tmp_path, monkeypatch):
    from agent_os.api.routes import agents_v2
    tc, deps = _client(tmp_path, monkeypatch)
    ws = tmp_path / "ws"; ws.mkdir()
    pid = _project(tc, ws)
    _write_goals(ws, EN_GOALS)
    _write_next_steps(ws, ["Write the hero copy", "Pick the colours"])
    _session(ws)
    ask_id = asyncio.run(agents_v2.post_goals_confirm(pid, "s1"))
    (ask,) = [a for a in _open_asks(ws) if a.id == ask_id]
    first, rest = ask.detail.split("\n", 1)
    assert first == GOALS_LINK
    assert rest.strip() == "1. Write the hero copy\n2. Pick the colours"
    # The steps themselves are unchanged on the project record.
    rec = deps["project_store"].get_project(pid)["import_scan_ask"]
    assert rec["steps"] == ["Write the hero copy", "Pick the colours"]


# ---------------------------------------------------------------------------
# D2. The card speaks the draft's language
# ---------------------------------------------------------------------------


def test_chinese_goals_get_a_chinese_card_under_an_english_ui(tmp_path, monkeypatch):
    from agent_os.api.routes import agents_v2
    tc, deps = _client(tmp_path, monkeypatch, ui="en-US")
    ws = tmp_path / "ws"; ws.mkdir()
    pid = _project(tc, ws)
    _write_goals(ws, ZH_GOALS)
    _session(ws)
    ask_id = asyncio.run(agents_v2.post_goals_confirm(pid, "s1"))
    (ask,) = [a for a in _open_asks(ws) if a.id == ask_id]
    assert ask.text == "这样理解对吗？"
    assert ask.options == ("对", "需要改")
    rec = deps["project_store"].get_project(pid)["import_scan_ask"]
    assert (rec["yes"], rec["change"]) == ("对", "需要改")


def test_chinese_goals_with_steps_get_the_chinese_step_options(tmp_path, monkeypatch):
    from agent_os.api.routes import agents_v2
    from agent_os.onboarding.import_scan import classify_answer
    tc, deps = _client(tmp_path, monkeypatch, ui="en-US")
    ws = tmp_path / "ws"; ws.mkdir()
    pid = _project(tc, ws)
    _write_goals(ws, ZH_GOALS)
    _write_next_steps(ws, ["写首页文案"])
    _session(ws)
    ask_id = asyncio.run(agents_v2.post_goals_confirm(pid, "s1"))
    (ask,) = [a for a in _open_asks(ws) if a.id == ask_id]
    assert ask.options == ("对，做第 1 步", "对，先不做", "需要改")
    # The answer matcher reads the posted options from the record, so it
    # classifies the zh answers even though the project's own lang is en.
    rec = deps["project_store"].get_project(pid)["import_scan_ask"]
    assert classify_answer(rec, "对，做第 1 步", "en") == "go"
    assert classify_answer(rec, "对，先不做", "en") == "yes"
    assert classify_answer(rec, "需要改", "en") == "change"
    assert classify_answer(rec, "需要改：不是这个项目", "en") == "change"


@pytest.mark.parametrize("goals,ui,expect", [
    (EN_GOALS, "en-US", "Did I get this right?"),
    (EN_GOALS, "zh-CN", "这样理解对吗？"),          # the existing rule: UI language
    (MIXED_EN_GOALS, "en-US", "Did I get this right?"),  # a few CJK names: still English
])
def test_non_chinese_goals_keep_the_existing_language_rule(tmp_path, monkeypatch,
                                                            goals, ui, expect):
    from agent_os.api.routes import agents_v2
    tc, _deps = _client(tmp_path, monkeypatch, ui=ui)
    ws = tmp_path / "ws"; ws.mkdir()
    pid = _project(tc, ws)
    _write_goals(ws, goals)
    _session(ws)
    ask_id = asyncio.run(agents_v2.post_goals_confirm(pid, "s1"))
    (ask,) = [a for a in _open_asks(ws) if a.id == ask_id]
    assert ask.text == expect


def test_english_card_answers_still_classify(tmp_path, monkeypatch):
    from agent_os.api.routes import agents_v2
    from agent_os.onboarding.import_scan import classify_answer
    tc, deps = _client(tmp_path, monkeypatch, ui="en-US")
    ws = tmp_path / "ws"; ws.mkdir()
    pid = _project(tc, ws)
    _write_goals(ws, EN_GOALS)
    _session(ws)
    asyncio.run(agents_v2.post_goals_confirm(pid, "s1"))
    rec = deps["project_store"].get_project(pid)["import_scan_ask"]
    assert classify_answer(rec, "Yes", "en") == "yes"
    assert classify_answer(rec, "Needs changes", "en") == "change"


@pytest.mark.parametrize("text,expect", [
    (ZH_GOALS, "zh"),
    (EN_GOALS, None),
    (MIXED_EN_GOALS, None),
    ("", None),
    ("# 目标\n\n- 上线\n", "zh"),
])
def test_text_lang_counts_cjk_against_letters(text, expect):
    from agent_os.onboarding.import_scan import text_lang
    assert text_lang(text) == expect
