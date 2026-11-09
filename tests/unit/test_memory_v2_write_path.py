# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 124 §6 item 9 (+ §3.8): the agent's write path normalises into the
one shape and says so in the tool result; the prompt and format headers
carry the new contract."""

from __future__ import annotations

import os

from agent_os.agent import memory_entries as M
from agent_os.agent import state_blocks
from agent_os.agent.tools.write import WriteTool
from agent_os.agent.workspace_files import WorkspaceFileManager


def test_heading_lessons_land_as_bullets_with_ids_and_the_note_quotes_them(tmp_path):
    ws = WorkspaceFileManager(str(tmp_path))
    ws.ensure_dir()
    out = WriteTool(str(tmp_path)).execute(
        path="orbital/LESSONS.md",
        content="# LESSONS\n\n### foo\nWhat happened: it broke.\nDo instead: fix it.\n\n### bar\nAlways grep first.\n",
    )
    assert '"status": "success"' in out.content
    live = ws.read("lessons")
    blocks = state_blocks.parse(live)[1]
    assert [b.content_lines[0] for b in blocks] == ["- foo", "- bar"]
    assert blocks[0].id == "foo" and blocks[1].id == "bar"
    assert blocks[0].content_lines[1:] == ["  What happened: it broke.", "  Do instead: fix it."]
    assert "### foo" not in live
    note = out.content.split("Note: ", 1)[1]
    assert "LESSONS.md was normalised" in note
    assert '"- foo"' in note and '"- bar"' in note
    assert "re-read before the next edit" in note


def test_pure_prose_lessons_are_stored_as_is_with_the_expected_shape(tmp_path):
    ws = WorkspaceFileManager(str(tmp_path))
    ws.ensure_dir()
    prose = "# LESSONS\nJust some notes about the day.\nAnd another line.\n"
    out = WriteTool(str(tmp_path)).execute(path="orbital/LESSONS.md", content=prose)
    assert ws.read("lessons") == M.FORMAT_HEADERS["lessons"] + "\n" + prose
    assert "one `- ` bullet line per entry" in out.content


def test_conforming_write_is_silent(tmp_path):
    ws = WorkspaceFileManager(str(tmp_path))
    ws.ensure_dir()
    ws.write("decisions", "# D\n- Charge per seat.\n")
    live = ws.read("decisions")
    out = WriteTool(str(tmp_path)).execute(path="orbital/DECISIONS.md", content=live)
    assert "Note:" not in out.content
    assert ws.read("decisions") == live


def test_format_headers_state_the_one_shape_and_index_granularity():
    for key in ("state", "decisions", "lessons"):
        h = M.FORMAT_HEADERS[key]
        assert "one `- ` bullet line that stands on its own" in h
        assert "indented" in h and "`## ` headings" in h
        assert "mem-comment on the line under the bullet" in h
    assert "numbered" not in M.FORMAT_HEADERS["lessons"]
    assert "'## <slug>'" not in M.FORMAT_HEADERS["decisions"]
    idx = M.FORMAT_HEADERS["index"]
    assert "folders and topics, not individual dated files" in idx
    assert "source of truth" in idx and "*_ARCHIVE.md" in idx


def test_prompt_memory_section_carries_the_new_contract():
    from agent_os.agent.prompt_builder import Autonomy, PromptBuilder, PromptContext
    _, semi, _ = PromptBuilder().build(PromptContext(
        workspace="/tmp/ws", model="m", autonomy=Autonomy.HANDS_OFF, enabled_agents=[],
        tool_names=["read", "write", "edit"], os_type="macos", datetime_now="2026-10-20T00:00:00",
    ))
    assert "Do not update the memory files during a run." in semi
    assert "the daemon asks you what to record and hands" in semi
    # 1109 R4: the entry-shape rules travel in the record-turn rows, where
    # the files are written (test_memory_record_turn), not on every call.
    assert "stands on its own" not in semi and "<!--format-->" not in semi
    assert "tidied by a Memory session" in " ".join(semi.split())
    assert "nothing is lost" in semi
    assert "contents table at the top" in semi and "grep it for `id:<id>`" in semi
    for gone in ("leaving a pointer line", "[archived DATE id:X]", "Update them proactively",
                 "Sources and pointers"):
        assert gone not in semi, gone
