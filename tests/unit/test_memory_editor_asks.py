# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 089 v2 §6: the memory editor may close an ask only with the user's words.

The editor sees the open asks and the sessions since its last run. It may
close an ask it can quote the user answering; the daemon checks that the quote
is verbatim in a USER message of the session it cites before appending
``done <id> … by:editor "<quote>"`` through the asks module. Anything else is
rejected and the ask stays open.
"""

from __future__ import annotations

import json
import os

import pytest

from agent_os.agent import asks
from agent_os.agent import memory_editor as E
from agent_os.agent.project_paths import ProjectPaths
from agent_os.agent.workspace_files import WorkspaceFileManager

TODAY = "2026-09-19"
ANSWER = "I talked to my cofounder: the webinar will be in English, final."


@pytest.fixture(autouse=True)
def _clear_single_flight():
    E._RUNNING.clear()
    yield
    E._RUNNING.clear()


@pytest.fixture
def ws(tmp_path):
    workspace = str(tmp_path)
    wf = WorkspaceFileManager(workspace)
    wf.write("state", "- Webinar agenda drafted.\n")
    orbital = wf.dir
    ask = asks.open_ask(orbital, "Decide the webinar language (EN or ZH)", today="2026-09-18")
    sessions = ProjectPaths(workspace).sessions_dir
    os.makedirs(sessions, exist_ok=True)
    rows = [
        {"role": "meta", "type": "session_start"},
        {"role": "user", "content": ANSWER, "source": "user"},
        {"role": "assistant", "content": "Noted — English it is, and I will ship it today.",
         "source": "management"},
    ]
    with open(os.path.join(sessions, "proj_aaaa1111.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(sessions, "proj_bbbb2222.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"role": "user", "content": "unrelated", "source": "user"}) + "\n")
    return workspace, orbital, ask.id


def _state(orbital, ask_id):
    return {a.id: a for a in asks.read_asks(orbital)}[ask_id]


def test_close_with_a_verbatim_user_quote(ws):
    workspace, orbital, ask_id = ws
    applied, rejected = E.apply_ask_closes(workspace, orbital, [
        {"id": ask_id, "session": "proj_aaaa1111",
         "quote": "the webinar will be in English, final"},
    ], today=TODAY)
    assert rejected == []
    assert applied == [{"kind": "close_ask", "id": ask_id, "session": "proj_aaaa1111"}]
    ask = _state(orbital, ask_id)
    assert ask.state == "done" and ask.closed_by == "editor"
    assert '"the webinar will be in English, final"' in ask.note


def test_quote_matching_ignores_whitespace_differences(ws):
    workspace, orbital, ask_id = ws
    applied, _ = E.apply_ask_closes(workspace, orbital, [
        {"id": ask_id, "session": "proj_aaaa1111",
         "quote": "the webinar  will be\nin English, final"},
    ], today=TODAY)
    assert len(applied) == 1


@pytest.mark.parametrize("item, why", [
    ({"session": "proj_aaaa1111", "quote": "English it is"}, "assistant words"),
    ({"session": "proj_bbbb2222", "quote": "the webinar will be in English"}, "other session"),
    ({"session": "proj_missing", "quote": "the webinar will be in English"}, "missing session"),
    ({"session": "../proj_aaaa1111", "quote": "the webinar will be in English"}, "path escape"),
    ({"session": "proj_aaaa1111", "quote": ""}, "empty quote"),
    ({"session": "proj_aaaa1111", "quote": "x" * 600}, "quote too long"),
])
def test_rejected_closes_leave_the_ask_open(ws, item, why):
    workspace, orbital, ask_id = ws
    applied, rejected = E.apply_ask_closes(workspace, orbital, [{"id": ask_id, **item}],
                                           today=TODAY)
    assert applied == [], why
    assert len(rejected) == 1, why
    assert _state(orbital, ask_id).state == "open", why


def test_unknown_and_already_closed_ids_are_rejected(ws):
    workspace, orbital, ask_id = ws
    asks.append_event(orbital, "done", ask_id, "by user", "user", today=TODAY)
    applied, rejected = E.apply_ask_closes(workspace, orbital, [
        {"id": ask_id, "session": "proj_aaaa1111", "quote": "in English, final"},
        {"id": "ffffff", "session": "proj_aaaa1111", "quote": "in English, final"},
        "not an object",
    ], today=TODAY)
    assert applied == []
    assert len(rejected) == 3
    assert _state(orbital, ask_id).closed_by == "user"


def test_no_close_asks_key_is_a_no_op(ws):
    workspace, orbital, ask_id = ws
    assert E.apply_ask_closes(workspace, orbital, None, today=TODAY) == ([], [])
    assert _state(orbital, ask_id).state == "open"


def test_prompt_lists_open_asks_only_when_there_are_some(ws):
    workspace, orbital, ask_id = ws
    contents = {k: "" for k in E.LIVE_KEYS}
    open_asks = [a for a in asks.read_asks(orbital) if a.is_open]
    with_asks = E.build_prompt(contents, today=TODAY, since=None, sessions=[],
                               open_asks=open_asks)
    assert f"[{ask_id}] Decide the webinar language" in with_asks
    assert '"close_asks"' in with_asks
    without = E.build_prompt(contents, today=TODAY, since=None, sessions=[])
    assert "OPEN ASKS" not in without


@pytest.mark.asyncio
async def test_run_editor_applies_a_quote_backed_close(ws):
    from types import SimpleNamespace

    workspace, orbital, ask_id = ws
    reply = json.dumps({"close_asks": [
        {"id": ask_id, "session": "proj_aaaa1111", "quote": "the webinar will be in English, final"},
    ]})

    class _LLM:
        model = "fake"

        def __init__(self):
            self.prompts = []

        async def complete(self, messages, tools=None, **_kw):
            self.prompts.append(messages[-1]["content"])
            return SimpleNamespace(text=reply, raw_message={"role": "assistant", "content": reply},
                                   tool_calls=[], has_tool_calls=False)

    llm = _LLM()
    result = await E.run_editor(WorkspaceFileManager(workspace), llm, today=TODAY)
    assert f"[{ask_id}]" in llm.prompts[0]
    assert result.outcome == "edited"
    assert {"kind": "close_ask", "id": ask_id, "session": "proj_aaaa1111"} in result.applied
    assert _state(orbital, ask_id).closed_by == "editor"
