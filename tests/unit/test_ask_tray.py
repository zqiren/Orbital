# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 146: an ask the agent opens reaches the composer of the conversation
that opened it, however the agent wrote the file. The row that records it never
reaches the model; answering closes it in the user's own words."""

import asyncio
import json
import os

import pytest

from agent_os.agent import asks
from agent_os.agent.context import ASK_CARD_KIND
from agent_os.api.routes import workbench
from tests.unit.test_memory_record_turn import (
    _ScriptedLLM, _reply, _rig, _tool_calls,
)


def _orbital(tmp_path):
    return os.path.join(str(tmp_path), "orbital")


def _ask_rows(session):
    return [m for m in session.get_messages() if m.get("_kind") == ASK_CARD_KIND]


def _run_with_ask(tmp_path, *, text="Keep the 9,800 creator?", write=None):
    llm = _ScriptedLLM([
        _tool_calls(("t1", "read", {"path": "x"})),
        _reply("Two things need your call."),
        _reply("nothing to record"),
    ])
    session, tools, loop = _rig(tmp_path, llm)
    real = tools.execute

    def execute(name, args):
        (write or (lambda: asks.open_ask(_orbital(tmp_path), text)))()
        tools.execute = real
        return real(name, args)

    tools.execute = execute
    return session, loop, llm


@pytest.mark.asyncio
async def test_an_ask_written_in_a_tool_batch_gets_one_row(tmp_path):
    session, loop, _ = _run_with_ask(tmp_path)
    await loop.run()
    rows = _ask_rows(session)
    assert len(rows) == 1
    assert rows[0]["_meta"]["event"] == "ask_opened"
    assert rows[0]["content"] == "Keep the 9,800 creator?"


@pytest.mark.asyncio
async def test_a_shell_append_with_no_stamp_is_caught_too(tmp_path):
    def shell_append():
        with open(asks.asks_path(_orbital(tmp_path)), "a", encoding="utf-8") as f:
            f.write("\n- open Pick the launch date\n")

    session, loop, _ = _run_with_ask(tmp_path, write=shell_append)
    await loop.run()
    assert [r["content"] for r in _ask_rows(session)] == ["Pick the launch date"]


@pytest.mark.asyncio
async def test_an_ask_already_open_before_the_run_gets_no_row(tmp_path):
    os.makedirs(_orbital(tmp_path), exist_ok=True)
    asks.open_ask(_orbital(tmp_path), "From an earlier session")
    llm = _ScriptedLLM([_reply("hi"), _reply("nothing to record")])
    session, tools, loop = _rig(tmp_path, llm)
    await loop.run()
    assert _ask_rows(session) == []


@pytest.mark.asyncio
async def test_the_row_never_reaches_the_model(tmp_path):
    session, loop, llm = _run_with_ask(tmp_path)
    await loop.run()
    await asyncio.sleep(0)
    llm_followup = _ScriptedLLM([_reply("ok"), _reply("nothing")])
    loop._llm = llm_followup
    sent = json.dumps(loop._context_manager.prepare(), ensure_ascii=False)
    assert "Keep the 9,800 creator?" not in sent


@pytest.mark.asyncio
async def test_the_session_hears_about_any_change_to_open_asks(tmp_path):
    session, loop, _ = _run_with_ask(tmp_path)
    heard = []
    session.on_asks_changed = lambda: heard.append(1)
    await loop.run()
    assert heard


# ---- the session's list ----

class _Store:
    def __init__(self, project):
        self.project = project

    def get_project(self, pid):
        return self.project

    def list_projects(self):
        return [self.project]


@pytest.fixture
def project(tmp_path):
    p = {"project_id": "p1", "name": "n", "workspace": str(tmp_path)}
    workbench.configure(_Store(p), None, None)
    return p


@pytest.mark.asyncio
async def test_the_list_is_this_sessions_open_asks_in_order(tmp_path, project):
    session, loop, _ = _run_with_ask(tmp_path)
    await loop.run()
    other = asks.open_ask(_orbital(tmp_path), "Opened in another session")
    body = await workbench.session_asks("p1", session.session_id)
    assert [a["text"] for a in body["asks"]] == ["Keep the 9,800 creator?"]
    assert other.id not in [a["id"] for a in body["asks"]]


@pytest.mark.asyncio
async def test_a_closed_ask_leaves_the_list(tmp_path, project):
    session, loop, _ = _run_with_ask(tmp_path)
    await loop.run()
    ask_id = (await workbench.session_asks("p1", session.session_id))["asks"][0]["id"]
    workbench.close_answered_ask(project, ask_id, "Keep them")
    assert (await workbench.session_asks("p1", session.session_id))["asks"] == []
    closed = [a for a in asks.read_asks(_orbital(tmp_path)) if a.id == ask_id][0]
    assert closed.closed_by == "user" and "Keep them" in closed.note


@pytest.mark.asyncio
async def test_an_unknown_session_has_no_asks(project):
    assert (await workbench.session_asks("p1", "nope"))["asks"] == []
    assert (await workbench.session_asks("p1", "../../etc/passwd"))["asks"] == []


def test_answering_an_unknown_or_closed_ask_never_blocks_the_message(tmp_path, project):
    os.makedirs(_orbital(tmp_path), exist_ok=True)
    workbench.close_answered_ask(project, "abcdef", "words")  # no raise
    ask_id = asks.open_ask(_orbital(tmp_path), "Q").id
    workbench.close_answered_ask(project, ask_id, "first")
    workbench.close_answered_ask(project, ask_id, "second")
    a = [x for x in asks.read_asks(_orbital(tmp_path)) if x.id == ask_id][0]
    assert "first" in a.note


def test_a_session_started_by_answering_an_ask_is_named_by_the_answer():
    from agent_os.agent.session import _derive_name

    wire = '[Answer to ask bdb0a2, already closed: "Send the letter"]\nIt is in the folder now'
    assert _derive_name(wire) == "It is in the folder now"
