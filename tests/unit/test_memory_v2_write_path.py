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


# ---------------------------------------------------------------------------
# Spec 182: a `supersedes: <old bullet line>` detail line moves the old entry
# to the archive in the same write, under `## [replaced DATE by id:<new>]`.
# ---------------------------------------------------------------------------

from agent_os.agent.tools.edit import EditTool  # noqa: E402

TODAY = "2026-11-16"


def _entry(text, eid, created="2026-09-01", tag="", detail=""):
    tagtxt = f" tag:{tag}" if tag else ""
    out = f"- {text}\n  <!--mem id:{eid} created:{created} touched:{created}{tagtxt}-->"
    if detail:
        out += "\n" + detail
    return out


def _seed(tmp_path, key, body):
    ws = WorkspaceFileManager(str(tmp_path))
    ws.ensure_dir()
    ws.write(key, body)
    return ws


def _process(tmp_path, filename, content):
    target = os.path.join(str(tmp_path), "orbital", filename)
    return M.process_on_write(str(tmp_path), target, content, today=TODAY)


def _ids(text):
    return [b.id for b in state_blocks.parse(text)[1]]


def test_supersedes_moves_the_quoted_entry_to_the_archive_and_the_guard_keeps_it_out(tmp_path):
    old = _entry("Ship the API on Flask, the team knows it.", "flask", detail="  Rejected: Django.")
    ws = _seed(tmp_path, "decisions", "# D\n\n" + old + "\n" + _entry("Free tier = 3 projects.", "free") + "\n")
    before = ws.read("decisions")
    old_raw = [b for b in state_blocks.parse(before)[1] if b.id == "flask"][0].raw
    out = EditTool(str(tmp_path)).execute(
        path="orbital/DECISIONS.md",
        old_text="- Free tier = 3 projects.",
        new_text=("- Free tier = 3 projects.\n- Ship the API on FastAPI; Flask could not stream.\n"
                  "  supersedes: Ship the API on Flask, the team knows it."),
    )
    assert '"status": "success"' in out.content, out.content
    live = ws.read("decisions")
    assert "flask" not in _ids(live) and "Ship the API on Flask" not in live
    assert "  supersedes:" not in live
    assert "put back" not in out.content
    new_id = [b.id for b in state_blocks.parse(live)[1] if b.text.startswith("Ship the API on FastAPI")][0]
    archive = ws.read("decisions_archive")
    assert f"## [replaced {M._today()} by id:{new_id}]" in archive
    assert old_raw in archive                       # moved intact: bullet, comment, detail
    assert f"- id:flask · replaced {M._today()} → id:{new_id} · Ship the API on Flask" in archive
    assert 'id:flask "Ship the API on Flask, the team knows it." moved to DECISIONS_ARCHIVE.md' in out.content


def test_supersedes_archive_is_written_before_the_live_text_is_returned(tmp_path):
    _seed(tmp_path, "decisions", "# D\n" + _entry("Use tabs.", "tabs") + "\n")
    content = "# D\n" + _entry("Use tabs.", "tabs") + "\n- Use spaces.\n  supersedes: Use tabs.\n"
    text, warns = _process(tmp_path, "DECISIONS.md", content)
    archive = open(os.path.join(str(tmp_path), "orbital", "DECISIONS_ARCHIVE.md"), encoding="utf-8").read()
    assert "id:tabs" in archive and "## [replaced 2026-11-16 by id:use-spaces]" in archive
    assert _ids(text) == ["use-spaces"]
    assert "  supersedes:" not in text


def test_supersedes_quote_ignores_list_marker_bold_and_whitespace(tmp_path):
    _seed(tmp_path, "decisions", "# D\n" + _entry("**Pricing:** charge per seat.", "seat") + "\n")
    content = ("# D\n" + _entry("**Pricing:** charge per seat.", "seat")
               + "\n- Charge per project.\n  supersedes: -  Pricing:   charge per seat.\n")
    text, warns = _process(tmp_path, "DECISIONS.md", content)
    assert "seat" not in _ids(text)


def test_supersedes_written_unindented_under_the_bullet_still_resolves(tmp_path):
    _seed(tmp_path, "decisions", "# D\n" + _entry("Use tabs.", "tabs") + "\n")
    content = "# D\n" + _entry("Use tabs.", "tabs") + "\n- Use spaces.\nsupersedes: Use tabs.\n"
    text, _ = _process(tmp_path, "DECISIONS.md", content)
    assert _ids(text) == ["use-spaces"] and "  supersedes:" not in text


def _unchanged_case(tmp_path, live_body, content, *, archive=None):
    ws = _seed(tmp_path, "decisions", live_body)
    if archive is not None:
        from agent_os.agent import workspace_files as WF
        WF._append_archive_section(ws, "decisions_archive", "## [archived 2026-10-01]", archive)
    arch_before = ws.read("decisions_archive")
    text, warns = _process(tmp_path, "DECISIONS.md", content)
    assert ws.read("decisions_archive") == arch_before
    return text, " ".join(warns)


def test_supersedes_matching_nothing_moves_nothing_and_says_so(tmp_path):
    body = "# D\n" + _entry("Use tabs.", "tabs") + "\n"
    text, note = _unchanged_case(tmp_path, body, body + "- Use spaces.\n  supersedes: Use tab.\n")
    assert _ids(text) == ["tabs", "use-spaces"]
    assert "  supersedes: Use tab." in text
    assert 'supersedes: "Use tab." under "Use spaces." matched no entry in DECISIONS.md' in note
    assert "word for word" in note


def test_supersedes_matching_two_entries_moves_nothing(tmp_path):
    body = "# D\n## A\n" + _entry("Use tabs.", "tabs") + "\n## B\n" + _entry("Use tabs.", "tabs-2") + "\n"
    text, note = _unchanged_case(tmp_path, body, body + "- Use spaces.\n  supersedes: Use tabs.\n")
    assert _ids(text) == ["tabs", "tabs-2", "use-spaces"]
    assert "matched 2 entries" in note


def test_supersedes_matching_its_own_entry_moves_nothing(tmp_path):
    body = "# D\n" + _entry("Use tabs.", "tabs") + "\n"
    text, note = _unchanged_case(tmp_path, body, body + "- Use spaces.\n  supersedes: Use spaces.\n")
    assert _ids(text) == ["tabs", "use-spaces"] and "supersedes: Use spaces." in text
    assert "names this entry itself" in note


def test_supersedes_never_moves_a_pinned_entry(tmp_path):
    body = "# D\n" + _entry("Use tabs.", "tabs", tag="pinned") + "\n"
    text, note = _unchanged_case(tmp_path, body, body + "- Use spaces.\n  supersedes: Use tabs.\n")
    assert _ids(text) == ["tabs", "use-spaces"]
    assert "pinned" in note


def test_supersedes_never_moves_an_entry_whose_id_is_already_archived(tmp_path):
    body = "# D\n" + _entry("Use tabs.", "tabs") + "\n"
    text, note = _unchanged_case(tmp_path, body, body + "- Use spaces.\n  supersedes: Use tabs.\n",
                                 archive=_entry("Use tabs.", "tabs"))
    assert _ids(text) == ["tabs", "use-spaces"]
    assert "already in DECISIONS_ARCHIVE.md" in note


def test_two_supersedes_lines_under_one_entry_each_resolve(tmp_path):
    body = "# D\n" + _entry("Use tabs.", "tabs") + "\n" + _entry("Indent by 8.", "eight") + "\n"
    _seed(tmp_path, "decisions", body)
    content = body + "- Use 4 spaces.\n  supersedes: Use tabs.\n  supersedes: Indent by 8.\n  Reason: PEP 8.\n"
    text, warns = _process(tmp_path, "DECISIONS.md", content)
    assert _ids(text) == ["use-4-spaces"]
    assert "  Reason: PEP 8." in text and "  supersedes" not in text
    archive = open(os.path.join(str(tmp_path), "orbital", "DECISIONS_ARCHIVE.md"), encoding="utf-8").read()
    assert archive.count("## [replaced 2026-11-16 by id:use-4-spaces]") == 1
    assert "id:tabs" in archive and "id:eight" in archive
    assert len([w for w in warns if "moved to DECISIONS_ARCHIVE.md" in w]) == 2


def test_supersedes_works_for_lessons(tmp_path):
    body = "# L\n" + _entry("Retry the upload once.", "retry") + "\n"
    _seed(tmp_path, "lessons", body)
    text, warns = _process(tmp_path, "LESSONS.md",
                           body + "- Never retry the upload; resume it.\n  supersedes: Retry the upload once.\n")
    assert "retry" not in _ids(text)
    archive = open(os.path.join(str(tmp_path), "orbital", "LESSONS_ARCHIVE.md"), encoding="utf-8").read()
    assert "id:retry" in archive and "## [replaced 2026-11-16 by id:" in archive


def test_supersedes_in_chinese(tmp_path):
    body = "# 决策\n" + _entry("接口用 Flask，团队熟悉。", "flask-zh") + "\n"
    _seed(tmp_path, "decisions", body)
    text, warns = _process(tmp_path, "DECISIONS.md",
                           body + "- 接口改用 FastAPI，Flask 不能流式输出。\n  supersedes：接口用 Flask，团队熟悉。\n")
    assert "flask-zh" not in _ids(text) and "  supersedes" not in text
    assert "接口改用 FastAPI" in text


def test_state_supersedes_lines_are_left_alone(tmp_path):
    body = "## Now\n" + _entry("Focus: launch.", "focus") + "\n"
    _seed(tmp_path, "state", body)
    text, _ = _process(tmp_path, "PROJECT_STATE.md", body + "- Focus: docs.\n  supersedes: Focus: launch.\n")
    assert "focus" in _ids(text) and "supersedes: Focus: launch." in text
    assert not os.path.exists(os.path.join(str(tmp_path), "orbital", "PROJECT_STATE_ARCHIVE.md"))


def test_an_unresolved_supersedes_line_resolves_on_a_later_write(tmp_path):
    ws = _seed(tmp_path, "decisions", "# D\n- Use spaces.\n  supersedes: Use tabs.\n")
    first = ws.read("decisions")
    assert "supersedes: Use tabs." in first      # a system write is not processed
    # A later tool write adds a bullet with exactly the quoted text.
    out = WriteTool(str(tmp_path)).execute(path="orbital/DECISIONS.md", content=first + "- Use tabs.\n")
    assert '"status": "success"' in out.content
    live = ws.read("decisions")
    assert [b.text for b in state_blocks.parse(live)[1]] == ["Use spaces."]
    assert "id:use-tabs" in ws.read("decisions_archive")


def test_writer_that_deleted_the_old_entry_and_declared_it_still_gets_it_archived(tmp_path):
    """The old instruction said "supersede outright": a writer that deletes
    the old entry AND writes the line. The guard puts the entry back; the
    declared line then moves it to the archive in the same write, so it is
    neither lost nor left standing."""
    ws = _seed(tmp_path, "decisions", "# D\n" + _entry("Use tabs.", "tabs") + "\n")
    out = WriteTool(str(tmp_path)).execute(
        path="orbital/DECISIONS.md", content="# D\n- Use spaces.\n  supersedes: Use tabs.\n")
    assert '"status": "success"' in out.content
    live = ws.read("decisions")
    assert _ids(live) == ["use-spaces"] and "  supersedes" not in live
    assert "id:tabs" in ws.read("decisions_archive")
    assert "put back" not in out.content


def test_a_failed_archive_write_fails_the_tool_call_before_the_live_file(tmp_path, monkeypatch):
    ws = _seed(tmp_path, "decisions", "# D\n" + _entry("Use tabs.", "tabs") + "\n")
    live_before = ws.read("decisions")

    def boom(path, text):
        raise OSError("disk full")
    monkeypatch.setattr(M, "_atomic_write", boom)
    out = WriteTool(str(tmp_path)).execute(
        path="orbital/DECISIONS.md", content=live_before + "- Use spaces.\n  supersedes: Use tabs.\n")
    assert out.content.startswith("Error:")
    assert ws.read("decisions") == live_before
    assert ws.read("decisions_archive") is None


def test_decisions_and_lessons_headers_teach_the_supersedes_line():
    """Spec 182 §5.5: the header no longer says both "a Memory session
    archives it" and "never leave two side by side"; it says how to declare
    the replacement and that the old entry is moved, never deleted."""
    for key, noun in (("decisions", "decision"), ("lessons", "lesson")):
        h = M.FORMAT_HEADERS[key]
        assert f"When a {noun} changes" in h
        assert "supersedes: <the old bullet line, word for word>" in h
        assert "keep the old entry as it is" in h
        assert "moves to the archive" in h
        assert "\n" not in h
    assert "archived by a Memory session" not in M.FORMAT_HEADERS["decisions"]
    assert "side by side" not in M.FORMAT_HEADERS["decisions"]


def test_supersedes_after_anchoring_on_the_old_bullet_line_moves_it_with_its_id(tmp_path):
    """The record row anchors an append on the last line shown: when that is
    the old bullet itself, the new text lands between the bullet and its
    hidden comment. The old entry still leaves with its own id and comment."""
    ws = _seed(tmp_path, "decisions", "# D\n" + _entry("Free tier = 3 projects.", "free") + "\n"
               + _entry("Use tabs.", "tabs") + "\n")
    out = EditTool(str(tmp_path)).execute(
        path="orbital/DECISIONS.md", old_text="- Use tabs.",
        new_text="- Use tabs.\n- Use spaces.\n  supersedes: Use tabs.")
    assert '"status": "success"' in out.content, out.content
    live = ws.read("decisions")
    assert _ids(live) == ["free", "use-spaces"]
    archive = ws.read("decisions_archive")
    assert "- Use tabs.\n  <!--mem id:tabs created:2026-09-01 touched:2026-09-01-->" in archive
    assert f"## [replaced {M._today()} by id:use-spaces]" in archive


def test_a_live_write_that_fails_after_the_archive_loses_nothing_and_the_retry_moves_nothing(tmp_path, monkeypatch):
    """Spec 182 §9: the archive is appended, then the live write fails. The
    old entry is in both files (nothing lost); a retry does not archive it a
    second time and says why; removing the live copy later is allowed."""
    import builtins
    ws = _seed(tmp_path, "decisions", "# D\n" + _entry("Use tabs.", "tabs", detail="  Why: habit.") + "\n")
    before = ws.read("decisions")
    content = before + "- Use spaces.\n  supersedes: Use tabs.\n"
    real_open = builtins.open

    def cut(path, mode="r", *a, **k):
        if str(path).endswith(os.path.join("orbital", "DECISIONS.md")) and "w" in mode:
            raise OSError("power cut")
        return real_open(path, mode, *a, **k)
    monkeypatch.setattr(builtins, "open", cut)
    out = WriteTool(str(tmp_path)).execute(path="orbital/DECISIONS.md", content=content)
    monkeypatch.setattr(builtins, "open", real_open)
    assert out.content.startswith("Error:")
    assert ws.read("decisions") == before                       # live file untouched
    archive = ws.read("decisions_archive")
    assert "id:tabs" in archive and "  Why: habit." in archive  # and the old entry is safe

    out = WriteTool(str(tmp_path)).execute(path="orbital/DECISIONS.md", content=content)
    assert "already in DECISIONS_ARCHIVE.md; nothing moved" in out.content
    assert ws.read("decisions_archive") == archive              # no second copy
    live = ws.read("decisions")
    assert _ids(live) == ["tabs", "use-spaces"]

    # The guard lets the live copy go now: its id is in the archive.
    removed = "\n".join(ln for ln in live.split("\n")
                        if "Use tabs." not in ln and "id:tabs" not in ln and "Why: habit." not in ln
                        and "supersedes:" not in ln)
    out = WriteTool(str(tmp_path)).execute(path="orbital/DECISIONS.md", content=removed)
    assert "put back" not in out.content
    assert _ids(ws.read("decisions")) == ["use-spaces"]
