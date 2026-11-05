# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Item 160 follow-up (2026-11-05): an open ask may offer the user a few
choices, the way Claude Code's question tool does. The options ride at the END
of the open line — ``- open <id> <date> [due:…] <text> options: A | B | C`` —
and every line without them parses exactly as before."""

from __future__ import annotations

import os

import httpx
import pytest
from fastapi import FastAPI

from agent_os.agent import asks
from agent_os.api.routes import workbench as wb
from agent_os.daemon_v2.checked_store import CheckedStore
from tests.unit.test_workbench_feed import (
    NOW, FakeManager, FakeStore, FakeSubAgents, FakeWS, _asks_file, _project, _row,
)

TODAY = "2026-11-05"


class TestParse:
    def test_options_at_the_end_of_an_open_line(self):
        a = asks.fold("- open a1b2c3 2026-11-05 Which venue? options: Rooftop | Hall | Garden\n")[0]
        assert a.text == "Which venue?"
        assert list(a.options) == ["Rooftop", "Hall", "Garden"]

    def test_options_with_a_due_date(self):
        a = asks.fold("- open a1b2c3 2026-11-05 due:2026-11-20 Ship it? options:Yes|No\n")[0]
        assert (a.text, a.due, list(a.options)) == ("Ship it?", "2026-11-20", ["Yes", "No"])

    def test_items_are_trimmed_and_chinese_is_fine(self):
        a = asks.fold("- open a1b2c3 2026-11-05 报税方式？ options:  自己报 |  找会计  \n")[0]
        assert a.text == "报税方式？"
        assert list(a.options) == ["自己报", "找会计"]

    def test_an_idless_agent_line_with_options(self):
        a = asks.fold("- open Pick the font options: Serif | Sans\n")[0]
        assert a.text == "Pick the font"
        assert list(a.options) == ["Serif", "Sans"]

    @pytest.mark.parametrize("line", [
        "- open a1b2c3 2026-09-01 Choose the venue",
        "- open d4e5f6 2026-09-02 due:2026-10-01 Send the deck",
        "- open [a1b2c3] 2026-09-01 Call the vendor due:2026-09-30T15:00",
        "- open Decide the booth budget",
        "- open a1b2c3 2026-09-01 Pick a | b layout",
    ])
    def test_lines_without_options_parse_as_before(self, line):
        a = asks.fold(line + "\n")[0]
        assert a.options == ()
        assert "options" not in a.text

    @pytest.mark.parametrize("tail", [
        "options: only one",                    # one item is not a choice
        "options: A | B | C | D | E",           # more than four
        "options:",                             # nothing after the keyword
        "options: A | ",                        # an empty item
    ])
    def test_a_malformed_options_tail_stays_text(self, tail):
        line = f"- open a1b2c3 2026-11-05 Which one? {tail}"
        a = asks.fold(line + "\n")[0]
        assert a.options == ()
        assert a.text == asks._one_line(f"Which one? {tail}")

    def test_everything_after_the_keyword_is_options(self):
        a = asks.fold("- open a1b2c3 2026-11-05 Read options: A | B then decide later\n")[0]
        # Everything after the keyword is options, by the grammar: the line
        # ends with them. "then decide later" is part of the last option.
        assert list(a.options) == ["A", "B then decide later"]

    def test_render_round_trips_options(self):
        e = asks.Event("open", "a1b2c3", TODAY, None, "Which venue?", "2026-11-20",
                       options=("Rooftop", "Hall"))
        line = asks.render_event(e)
        assert line == "- open a1b2c3 2026-11-05 due:2026-11-20 Which venue? options: Rooftop | Hall"
        a = asks.fold(line)[0]
        assert (a.text, a.due, list(a.options)) == ("Which venue?", "2026-11-20", ["Rooftop", "Hall"])

    def test_render_without_options_is_unchanged(self):
        e = asks.Event("open", "a1b2c3", "2026-09-01", None, "Send the deck", "2026-10-01")
        assert asks.render_event(e) == "- open a1b2c3 2026-09-01 due:2026-10-01 Send the deck"


class TestWritePath:
    def test_the_agent_append_keeps_its_options_when_stamped(self):
        prev = asks.FORMAT_HEADER + "\n"
        new = prev + "- open Which venue? options: Rooftop | Hall\n"
        out, _ = asks.process_write(prev, new, today=TODAY)
        last = [ln for ln in out.split("\n") if ln.strip()][-1]
        assert last.endswith("Which venue? options: Rooftop | Hall")
        a = asks.fold(out)[0]
        assert a.text == "Which venue?" and list(a.options) == ["Rooftop", "Hall"]

    def test_the_same_question_with_options_is_a_duplicate(self):
        prev = asks.FORMAT_HEADER + "\n- open a1b2c3 2026-11-01 Which venue?\n"
        new = prev + "- open Which venue? options: Rooftop | Hall\n"
        out, warnings = asks.process_write(prev, new, today=TODAY)
        assert len(asks.fold(out)) == 1
        assert any("already open" in w for w in warnings)

    def test_an_idless_line_with_options_is_stamped_in_place(self, tmp_path):
        od = tmp_path / "orbital"
        od.mkdir()
        (od / "ASKS.md").write_text(asks.FORMAT_HEADER + "\n- open Pick the font options: Serif | Sans\n",
                                    encoding="utf-8")
        before = asks.read_asks(str(od))[0]
        asks.open_ask(str(od), "Something else", today=TODAY)
        content = (od / "ASKS.md").read_text(encoding="utf-8")
        assert f"- open {before.id} {TODAY} Pick the font options: Serif | Sans" in content
        after = {a.id: a for a in asks.read_asks(str(od))}[before.id]
        assert list(after.options) == ["Serif", "Sans"]


# ---- the wire: session-asks + the feed ----

class _Store:
    def __init__(self, project):
        self.project = project

    def get_project(self, pid):
        return self.project

    def list_projects(self):
        return [self.project]


def _write_session_with_ask(workspace, sid, ask_id):
    from agent_os.agent.context import ASK_CARD_KIND
    import json
    path = os.path.join(workspace, "orbital", "sessions", f"{sid}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"role": "meta", "event": "session_start", "session_id": sid}) + "\n")
        f.write(json.dumps({"role": "system", "_kind": ASK_CARD_KIND, "content": "q",
                            "_meta": {"event": "ask_opened", "ask_id": ask_id}}) + "\n")


@pytest.mark.asyncio
async def test_session_asks_carry_options(tmp_path):
    ws = tmp_path / "ws"
    (ws / "orbital").mkdir(parents=True)
    (ws / "orbital" / "ASKS.md").write_text("\n".join([
        asks.FORMAT_HEADER,
        "- open a1b2c3 2026-11-05 Which venue? options: Rooftop | Hall",
        "- open d4e5f6 2026-11-05 Send the deck",
        "",
    ]), encoding="utf-8")
    sid = "sess_1"
    _write_session_with_ask(str(ws), sid, "a1b2c3")
    _write_session_with_ask_append(str(ws), sid, "d4e5f6")
    wb.configure(_Store({"project_id": "p1", "name": "n", "workspace": str(ws)}), None, None)
    body = await wb.session_asks("p1", sid)
    by_id = {a["id"]: a for a in body["asks"]}
    assert by_id["a1b2c3"]["options"] == ["Rooftop", "Hall"]
    assert by_id["a1b2c3"]["text"] == "Which venue?"
    assert by_id["d4e5f6"]["options"] == []


def _write_session_with_ask_append(workspace, sid, ask_id):
    from agent_os.agent.context import ASK_CARD_KIND
    import json
    path = os.path.join(workspace, "orbital", "sessions", f"{sid}.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"role": "system", "_kind": ASK_CARD_KIND, "content": "q",
                            "_meta": {"event": "ask_opened", "ask_id": ask_id}}) + "\n")


@pytest.mark.asyncio
async def test_the_feed_and_the_card_list_carry_options(tmp_path):
    a = _project(tmp_path, "proj_a")
    _asks_file(a, [
        "- open a1b2c3 2026-11-04 Which venue? options: Rooftop | Hall | Garden",
        "- open e5e5e5 2026-11-01 Renew the domain.",
    ])
    sessions = {"proj_a": [_row("s_ask", ask_ids=["a1b2c3"])]}
    store = CheckedStore.in_dir(str(tmp_path / "data"), now_fn=lambda: NOW)
    wb.configure(FakeStore([a]), FakeManager(sessions), None, now_fn=lambda: NOW,
                 ws_manager=FakeWS(), checked_store=store,
                 sub_agent_manager=FakeSubAgents({}))
    app = FastAPI()
    app.include_router(wb.router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
        r = await client.get("/api/v2/workbench/feed")
        assert r.status_code == 200
        rows = {(x["project_id"], x["session_id"]): x for x in r.json()["rows"]}
        assert rows[("proj_a", "s_ask")]["ask"]["options"] == ["Rooftop", "Hall", "Garden"]
        assert rows[("proj_a", "s_ask")]["ask"]["text"] == "Which venue?"
        assert rows[("proj_a", None)]["ask"]["options"] == []
        r = await client.get("/api/v2/workbench")
        assert r.status_code == 200
        cards = {c["id"]: c for c in r.json()["entries"]}
        assert cards["a1b2c3"]["options"] == ["Rooftop", "Hall", "Garden"]
        assert cards["e5e5e5"]["options"] == []


def test_the_prompt_tells_the_agent_how_to_offer_options():
    from agent_os.agent.prompt_builder import Autonomy, PromptBuilder, PromptContext
    ctx = PromptContext(
        workspace="/tmp/ws", model="m", autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[], tool_names=["read", "write", "edit"],
        os_type="macos", datetime_now="2026-11-05T00:00:00",
    )
    _, semi, _ = PromptBuilder().build(ctx)
    flat = " ".join(semi.split())
    assert "options: A | B" in flat


# ---- detail: indented continuation lines under an open line (2026-11-05) ----

DETAIL_ASK = "\n".join([
    asks.FORMAT_HEADER,
    "- open a1b2c3 2026-11-05 due:2026-11-19 Which account first?",
    "  The **Chase** statement covers `Jul–Sep`; Amex only has Aug.",
    "",
    "  Starting with Chase unblocks the $1,280 question.",
    "  options: Chase | Amex | Both",
    "- open d4e5f6 2026-11-05 Send the deck",
    "",
])


class TestDetail:
    def test_continuation_lines_are_the_detail(self):
        a = {x.id: x for x in asks.fold(DETAIL_ASK)}["a1b2c3"]
        assert a.text == "Which account first?"
        assert a.due == "2026-11-19"
        assert a.detail == ("The **Chase** statement covers `Jul–Sep`; Amex only has Aug.\n"
                            "\n"
                            "Starting with Chase unblocks the $1,280 question.")
        assert list(a.options) == ["Chase", "Amex", "Both"]

    def test_the_next_event_after_the_detail_still_parses(self):
        by_id = {x.id: x for x in asks.fold(DETAIL_ASK)}
        assert by_id["d4e5f6"].text == "Send the deck"
        assert by_id["d4e5f6"].detail == ""

    def test_an_indented_event_line_ends_the_detail(self):
        content = "\n".join([
            "- open a1b2c3 2026-11-05 Pick one",
            "  some context",
            "  - done a1b2c3 2026-11-06 by:user \"A\"",
        ])
        a = asks.fold(content)[0]
        assert a.detail == "some context"
        assert a.state == "done"

    def test_options_on_the_last_continuation_line_win_over_the_open_line(self):
        content = "- open a1b2c3 2026-11-05 Pick one options: X | Y\n  why it matters\n  options: A | B\n"
        a = asks.fold(content)[0]
        assert a.text == "Pick one"
        assert list(a.options) == ["A", "B"]
        assert a.detail == "why it matters"

    def test_nested_markdown_keeps_its_relative_indent(self):
        content = "- open a1b2c3 2026-11-05 Pick one\n  Choices:\n    - fast\n    - cheap\n"
        assert asks.fold(content)[0].detail == "Choices:\n  - fast\n  - cheap"

    def test_unindented_prose_and_a_blank_line_end_the_detail(self):
        content = "- open a1b2c3 2026-11-05 Pick one\n  context\n\nSome prose a human typed.\n  stray indented\n"
        a = asks.fold(content)[0]
        assert a.detail == "context"

    @pytest.mark.parametrize("content", [
        "- open a1b2c3 2026-09-01 Choose the venue\n- open d4e5f6 2026-09-02 due:2026-10-01 Send the deck\n",
        "# Asks\nSome prose.\n- a plain bullet\n- open a1b2c3 2026-09-01 Real ask\n",
        '- open a1b2c3 2026-09-01 Choose\n- done a1b2c3 2026-09-03 by:agent "rooftop"\n',
    ])
    def test_files_without_continuations_fold_as_before(self, content):
        for a in asks.fold(content):
            assert a.detail == ""

    def test_render_round_trips_detail_and_options(self):
        a = {x.id: x for x in asks.fold(DETAIL_ASK)}["a1b2c3"]
        e = asks.Event("open", a.id, "2026-11-05", None, a.text, a.due, options=a.options, detail=a.detail,
                       opts_in_detail=True)
        line = asks.render_event(e)
        assert line.split("\n")[0] == "- open a1b2c3 2026-11-05 due:2026-11-19 Which account first?"
        assert line.split("\n")[-1] == "  options: Chase | Amex | Both"
        b = asks.fold(line)[0]
        assert (b.text, b.detail, list(b.options)) == (a.text, a.detail, list(a.options))
        # Options kept on the open line round-trip too.
        head = asks.render_event(asks.Event("open", a.id, "2026-11-05", None, a.text, None,
                                            options=a.options, detail=a.detail))
        assert head.split("\n")[0].endswith("options: Chase | Amex | Both")
        assert asks.fold(head)[0].detail == a.detail

    def test_the_agent_append_keeps_the_detail_when_stamped(self):
        prev = asks.FORMAT_HEADER + "\n- open d4e5f6 2026-11-01 Send the deck\n"
        new = prev + "\n".join([
            "- open Which account first?",
            "  The **Chase** statement covers Jul–Sep.",
            "",
            "  Second paragraph.",
            "  options: Chase | Amex",
            "",
        ])
        out, warnings = asks.process_write(prev, new, today=TODAY)
        assert not any("not an ask event" in w for w in warnings), warnings
        a = [x for x in asks.fold(out) if x.text == "Which account first?"][0]
        assert a.detail == "The **Chase** statement covers Jul–Sep.\n\nSecond paragraph."
        assert list(a.options) == ["Chase", "Amex"]
        assert len(a.id) == 6 and a.opened == TODAY

    def test_an_existing_asks_detail_survives_an_agent_append(self):
        prev = DETAIL_ASK
        new = prev + "- done d4e5f6 \"sent it\"\n"
        out, warnings = asks.process_write(prev, new, today=TODAY)
        assert not any("restored" in w for w in warnings), warnings
        a = {x.id: x for x in asks.fold(out)}["a1b2c3"]
        assert a.detail.startswith("The **Chase** statement")
        assert list(a.options) == ["Chase", "Amex", "Both"]

    def test_an_idless_open_with_detail_is_stamped_in_place_once(self, tmp_path):
        od = tmp_path / "orbital"
        od.mkdir()
        (od / "ASKS.md").write_text(
            asks.FORMAT_HEADER + "\n- open Pick the font\n  Body copy is long-form.\n  options: Serif | Sans\n",
            encoding="utf-8")
        before = asks.read_asks(str(od))[0]
        asks.open_ask(str(od), "Something else", today=TODAY)
        content = (od / "ASKS.md").read_text(encoding="utf-8")
        assert content.count("Body copy is long-form.") == 1
        assert content.count("options: Serif | Sans") == 1
        after = {a.id: a for a in asks.read_asks(str(od))}[before.id]
        assert after.detail == "Body copy is long-form."
        assert list(after.options) == ["Serif", "Sans"]


@pytest.mark.asyncio
async def test_session_asks_and_the_feed_carry_detail(tmp_path):
    ws = tmp_path / "ws"
    (ws / "orbital").mkdir(parents=True)
    (ws / "orbital" / "ASKS.md").write_text(DETAIL_ASK, encoding="utf-8")
    sid = "sess_1"
    _write_session_with_ask(str(ws), sid, "a1b2c3")
    _write_session_with_ask_append(str(ws), sid, "d4e5f6")
    wb.configure(_Store({"project_id": "p1", "name": "n", "workspace": str(ws)}), None, None)
    by_id = {a["id"]: a for a in (await wb.session_asks("p1", sid))["asks"]}
    assert by_id["a1b2c3"]["detail"].startswith("The **Chase** statement")
    assert by_id["a1b2c3"]["text"] == "Which account first?"
    assert by_id["d4e5f6"]["detail"] == ""
    a = {x.id: x for x in asks.fold(DETAIL_ASK)}["a1b2c3"]
    from datetime import datetime, timezone
    now = datetime(2026, 11, 5, tzinfo=timezone.utc)
    assert wb._feed_ask("p1", a, "UTC", now)["detail"] == a.detail
    assert wb._ask_row("p1", a, "UTC", now)["detail"] == a.detail


def test_the_prompt_says_question_first_then_indented_context():
    from agent_os.agent.prompt_builder import Autonomy, PromptBuilder, PromptContext
    ctx = PromptContext(
        workspace="/tmp/ws", model="m", autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[], tool_names=["read", "write", "edit"],
        os_type="macos", datetime_now="2026-11-05T00:00:00",
    )
    _, semi, _ = PromptBuilder().build(ctx)
    flat = " ".join(semi.split())
    assert "question, in one sentence" in flat
    assert "indented by two spaces" in flat
    assert "options: A | B" in flat


def test_the_file_header_teaches_the_detail_and_options_form():
    """ASKS.md's own line 1 is what an agent Orbital does not prompt reads."""
    h = asks.FORMAT_HEADER
    assert "\n" not in h and h.startswith("<!--format") and h.endswith("-->")
    assert "indented two spaces" in h
    assert "`options: A | B`" in h
    assert "question in one sentence" in h
    # The header is never mistaken for an event or an ask's detail.
    assert asks.parse_line(h) is None
    content = h + "\n- open a1b2c3 2026-11-05 Pick one\n  why\n  options: A | B\n"
    a = asks.fold(content)[0]
    assert (a.text, a.detail, list(a.options)) == ("Pick one", "why", ["A", "B"])
