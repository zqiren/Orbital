# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 124 §6 items 6–8 and 11: retirement leaves nothing in the live file,
the floor never archives, id conservation (after a pass and at the write
path), archive contents tables, INDEX archive lines, and the deletions."""

from __future__ import annotations

import logging
import os
import re

from agent_os.agent import memory_editor as E
from agent_os.agent import memory_entries as M
from agent_os.agent import state_blocks
from agent_os.agent import workspace_files as WF
from agent_os.agent.tools.edit import EditTool
from agent_os.agent.tools.write import WriteTool
from agent_os.agent.workspace_files import WorkspaceFileManager

TODAY = "2026-10-20"
FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "memory_v2")


def _bullet(text, eid, created="2026-09-01", tag=""):
    tagtxt = f" tag:{tag}" if tag else ""
    return f"- {text}\n  <!--mem id:{eid} created:{created} touched:{created}{tagtxt}-->"


def _ws(tmp_path) -> WorkspaceFileManager:
    ws = WorkspaceFileManager(str(tmp_path))
    ws.ensure_dir()
    return ws


# ---------------------------------------------------------------------------
# 6. retirement through the agent's own tools; the floor moves nothing
# ---------------------------------------------------------------------------

def test_memory_agent_archives_with_the_write_tool_and_nothing_is_left_behind(tmp_path):
    ws = _ws(tmp_path)
    ws.write("decisions", "# D\n\n" + _bullet("Old pricing: per seat.", "per-seat") + "\n  Rejected: usage.\n"
             + _bullet("Free tier = 3 projects.", "free-tier") + "\n")
    live = ws.read("decisions")
    entry = state_blocks.parse(live)[1][0].raw
    tool = WriteTool(str(tmp_path))
    # archive first ...
    out = tool.execute(path="orbital/DECISIONS_ARCHIVE.md",
                       content=f"# DECISIONS_ARCHIVE.md\n\n## [archived {TODAY}]\n\n{entry}\n")
    assert '"status": "success"' in out.content
    # ... then the live file without it
    after = live.replace(entry + "\n", "")
    out = tool.execute(path="orbital/DECISIONS.md", content=after)
    assert '"status": "success"' in out.content and "put back" not in out.content
    live_now = ws.read("decisions")
    assert "per-seat" not in live_now and "[archived" not in live_now
    assert "free-tier" in live_now
    archive = ws.read("decisions_archive")
    assert entry in archive
    assert archive.startswith("# DECISIONS_ARCHIVE.md\n\n" + M.CONTENTS_OPEN + "\n## Contents\n")
    assert f"- id:per-seat · archived {TODAY} · Old pricing: per seat." in archive


def test_floor_never_archives_only_logs_and_reports(tmp_path, caplog, monkeypatch):
    monkeypatch.setitem(M.FILE_BUDGETS, "decisions", {"soft": 40, "hard": 60})
    monkeypatch.setitem(M.FILE_BUDGETS, "state", {"soft": 40, "hard": 60})
    ws = _ws(tmp_path)
    old = _bullet("old decision " + "o" * 600, "old777", "2024-01-01")
    ws.write("decisions", "# D\n" + old + "\n")
    ws.write("state", "## Now\n" + _bullet("old state " + "o" * 600, "st0001", "2024-01-01") + "\n")
    before = {k: ws.read(k) for k in ("decisions", "state")}
    with caplog.at_level(logging.INFO):
        over = WF._apply_hard_caps(ws)
    assert set(over) == {"state", "decisions"}
    assert ws.read("decisions") == before["decisions"] and ws.read("state") == before["state"]
    assert not ws.exists("decisions_archive") and not ws.exists("state_archive")
    assert any("over its hard cap" in r.getMessage() and "tidy pass is armed" in r.getMessage()
               for r in caplog.records)


# ---------------------------------------------------------------------------
# 7. id conservation
# ---------------------------------------------------------------------------

def test_postpass_restores_an_entry_that_left_without_reaching_the_archive(tmp_path, caplog):
    ws = _ws(tmp_path)
    ws.write("state", "## Now\n" + _bullet("keep me", "keep01") + "\n" + _bullet("deleted by the agent", "gone01") + "\n")
    snapshot = {"state": ws.read("state")}
    # a pass that simply dropped the entry (not archived)
    with open(ws._file_path("state"), "w", encoding="utf-8") as f:
        f.write(snapshot["state"].replace(_bullet("deleted by the agent", "gone01") + "\n", ""))
    assert "gone01" not in ws.read("state")
    with caplog.at_level(logging.WARNING):
        restored = WF.postpass_conservation(ws, snapshot, ["state"])
    assert restored == ["gone01"]
    assert "restored=['gone01']" in caplog.text
    assert _bullet("deleted by the agent", "gone01") in ws.read("state")


def test_postpass_leaves_archived_entries_alone(tmp_path):
    ws = _ws(tmp_path)
    ws.write("state", "## Now\n" + _bullet("keep me", "keep01") + "\n" + _bullet("done work", "done01") + "\n")
    snapshot = {"state": ws.read("state")}
    WF._append_archive_section(ws, "state_archive", f"## [archived {TODAY}]", _bullet("done work", "done01"))
    ws.write("state", "## Now\n" + _bullet("keep me", "keep01") + "\n")
    assert WF.postpass_conservation(ws, snapshot, ["state"]) == []
    assert "done01" not in ws.read("state")


def test_write_path_restores_a_bare_deletion_but_not_a_replacement(tmp_path):
    ws = _ws(tmp_path)
    ws.write("state", "## Now\n" + _bullet("alpha status", "aaa001") + "\n"
             + _bullet("beta status", "bbb001") + "\n" + _bullet("gamma status", "ccc001") + "\n")
    tool = WriteTool(str(tmp_path))
    # beta deleted outright, gamma reworded in place (its gap got a new line)
    out = tool.execute(path="orbital/PROJECT_STATE.md",
                       content="## Now\n- alpha status\n- gamma status is now done\n")
    assert "put back (id:bbb001)" in out.content
    live = ws.read("state")
    ids = [b.id for b in state_blocks.parse(live)[1]]
    assert "aaa001" in ids and "bbb001" in ids and "ccc001" not in ids
    assert "- gamma status is now done" in live and "- gamma status\n" not in live
    # beta came back where it was
    assert ids.index("bbb001") == 1


def test_write_path_does_not_restore_what_the_archive_holds(tmp_path):
    ws = _ws(tmp_path)
    ws.write("lessons", "# L\n" + _bullet("rule one", "rule-one") + "\n" + _bullet("rule two", "rule-two") + "\n")
    WF._append_archive_section(ws, "lessons_archive", f"## [archived {TODAY}]", _bullet("rule two", "rule-two"))
    out = WriteTool(str(tmp_path)).execute(path="orbital/LESSONS.md", content="# L\n- rule one\n")
    assert "put back" not in out.content
    assert "rule-two" not in ws.read("lessons")


def test_edit_tool_matches_the_comment_stripped_view_of_decisions(tmp_path):
    ws = _ws(tmp_path)
    ws.write("decisions", "# D\n" + _bullet("Charge per seat.", "per-seat") + "\n  Rejected: usage.\n")
    out = EditTool(str(tmp_path)).execute(
        path="orbital/DECISIONS.md",
        old_text="- Charge per seat.\n  Rejected: usage.",
        new_text="- Charge per seat.\n  Rejected: usage-based billing.",
    )
    assert '"status": "success"' in out.content, out.content
    live = ws.read("decisions")
    b = state_blocks.parse(live)[1][0]
    assert b.id == "per-seat" and b.fields["touched"] == M._today()
    assert "Rejected: usage-based billing." in live


# ---------------------------------------------------------------------------
# 8. INDEX archive lines + contents tables
# ---------------------------------------------------------------------------

def test_index_archive_line_counts_and_dates_are_computed_subject_kept():
    archive = (
        "# DECISIONS_ARCHIVE.md\n\n## [archived 2026-08-12]\n\n"
        + _bullet("a", "a1", "2026-08-01") + "\n\n## [merged 2026-10-19 into id:zz]\n\n"
        + _bullet("b", "b1", "2026-09-01") + "\n" + _bullet("c", "c1", "2026-09-02") + "\n"
    )
    line = M.index_archive_line("DECISIONS_ARCHIVE.md", archive)
    assert line == ("- orbital/DECISIONS_ARCHIVE.md — 3 archived entries, 2026-08-12 to 2026-10-19: "
                    "to be described. Contents table at the top; grep id:<id>.")
    described = line.replace("to be described", "pricing history, KOL channels")
    refreshed = M.index_archive_line("DECISIONS_ARCHIVE.md", archive + _bullet("d", "d1") + "\n", described)
    assert refreshed.startswith("- orbital/DECISIONS_ARCHIVE.md — 4 archived entries, 2026-08-12 to 2026-10-19: pricing history, KOL channels.")
    # a legacy daemon line is replaced; an agent-written description survives
    legacy = "- orbital/DECISIONS_ARCHIVE.md — older entries demoted from the live file (read on demand)."
    assert "to be described" in M.index_archive_line("DECISIONS_ARCHIVE.md", archive, legacy)
    agent = "- DECISIONS_ARCHIVE.md — v0.6 signing decisions (read on demand)"
    assert ": v0.6 signing decisions (read on demand)." in M.index_archive_line("DECISIONS_ARCHIVE.md", archive, agent)


def test_index_lines_refresh_after_an_archive_write_and_survive_a_rewrite(tmp_path):
    ws = _ws(tmp_path)
    ws.write("index", "# INDEX\n\n## Code\n- src/ — code\n")
    WF._append_archive_section(ws, "lessons_archive", f"## [archived {TODAY}]", _bullet("old rule", "old-rule"))
    WF.refresh_index_archive_lines(ws)
    index = ws.read("index")
    assert f"- orbital/LESSONS_ARCHIVE.md — 1 archived entry, {TODAY} to {TODAY}: to be described." in index
    assert "- src/ — code" in index
    # idempotent
    assert WF.refresh_index_archive_lines(ws) is False
    # an INDEX rewrite that drops the line gets it back
    kept = E._keep_archive_lines("# INDEX\n- docs/ — docs\n", index)
    assert "LESSONS_ARCHIVE.md" in kept
    # the tail trim never drops it
    trimmed = M.trim_volatile(index + "".join(f"- f{i}.md — x\n" for i in range(300)), 30)
    assert "LESSONS_ARCHIVE.md" in trimmed


def test_contents_table_rows_match_mem_ids_on_the_real_archives():
    for name in ("DECISIONS_ARCHIVE.md", "LESSONS_ARCHIVE.md", "PROJECT_STATE_ARCHIVE.md"):
        with open(os.path.join(FIXTURES, "orbital-marketing", name), encoding="utf-8") as f:
            raw = f.read()
        tabled = M.with_contents_table(raw)
        head, _, rest = tabled.partition(M.CONTENTS_CLOSE)
        rows = [ln for ln in head.split("\n") if ln.startswith("- id:")]
        assert len(rows) == len(re.findall(r"<!--mem id:", raw)), name
        assert len(rows) == M.archive_summary(raw)[0]
        # regenerating is idempotent and the body is untouched
        assert M.with_contents_table(tabled) == tabled
        assert rest.strip("\n") in raw
        # newest section first
        dates = [re.search(r"(archived|merged|trimmed|folded) (\d{4}-\d{2}-\d{2})", r) for r in rows]
        seen = [m.group(2) for m in dates if m]
        assert seen == sorted(seen, reverse=True)


def test_replaced_section_parses_and_its_contents_row_names_the_replacement():
    """Spec 182 §5.3: a fifth section kind, ``## [replaced DATE by id:<new>]``,
    rendered like a merge with the arrow to the entry that replaced it."""
    archive = M.append_archive_section(
        M.append_archive_section(None, "DECISIONS_ARCHIVE.md", "## [archived 2026-10-01]",
                                 _bullet("Old thing.", "old-thing")),
        "DECISIONS_ARCHIVE.md", f"## [replaced {TODAY} by id:fastapi]",
        _bullet("Ship the API on Flask.", "flask"),
    )
    rows = M.archive_rows(archive)
    assert (rows[0].id, rows[0].kind, rows[0].date, rows[0].into) == ("flask", "replaced", TODAY, "fastapi")
    assert (rows[1].kind, rows[1].into) == ("archived", "")
    assert f"- id:flask · replaced {TODAY} → id:fastapi · Ship the API on Flask." in archive
    assert "- id:old-thing · archived 2026-10-01 · Old thing." in archive
    assert M.archive_summary(archive) == (2, "2026-10-01", TODAY)
    assert M.archive_ids(archive) == {"flask", "old-thing"}
    assert M.with_contents_table(archive) == archive


def test_every_archive_write_path_regenerates_the_table(tmp_path):
    ws = _ws(tmp_path)
    # daemon append
    WF._append_archive_section(ws, "state_archive", f"## [archived {TODAY}]", _bullet("one", "one01"))
    assert "- id:one01 · archived " in ws.read("state_archive")
    # agent write through process_on_write
    out = WriteTool(str(tmp_path)).execute(
        path="orbital/PROJECT_STATE_ARCHIVE.md",
        content=ws.read("state_archive") + f"\n## [archived {TODAY}]\n\n" + _bullet("two", "two02") + "\n")
    assert '"status": "success"' in out.content
    table = ws.read("state_archive").split(M.CONTENTS_CLOSE)[0]
    assert "id:one01" in table and "id:two02" in table and table.count("<!--contents") == 1
    # N5 folding
    ws.write("state", "- live\n[archived 2026-08-03] legacy stub → PROJECT_STATE_ARCHIVE.md\n")
    archive = ws.read("state_archive")
    assert "## [pointers folded " in archive and archive.count(M.CONTENTS_OPEN) == 1
    # the Memory agent's edit (archive first, then the live file) + the
    # pass's postpass refresh of the INDEX archive line
    ws.write("decisions", "# D\n" + _bullet("retire me", "ret001") + "\n" + _bullet("stay", "stay01") + "\n")
    WF._append_archive_section(ws, "decisions_archive", f"## [archived {TODAY}]", _bullet("retire me", "ret001"))
    out = WriteTool(str(tmp_path)).execute(
        path="orbital/DECISIONS.md", content="# D\n" + _bullet("stay", "stay01") + "\n")
    assert '"status": "success"' in out.content and "put back" not in out.content
    WF.refresh_index_archive_lines(ws)
    assert "- id:ret001 · archived 2026-10-20 · retire me" in ws.read("decisions_archive")
    assert "ret001" not in ws.read("decisions") and "[archived" not in ws.read("decisions")
    assert "- orbital/DECISIONS_ARCHIVE.md — 1 archived entry" in ws.read("index")


def test_merge_leaves_one_bullet_with_a_fresh_id_and_archives_the_originals(tmp_path):
    """The merge the task message asks for: the merged entry written in the
    live file (the daemon stamps it), then the originals moved under a
    ``[merged DATE into id:<new>]`` section; the write guard lets the two
    originals go only because the archive has them."""
    ws = _ws(tmp_path)
    ws.write("state", "## Now\n" + _bullet("radar 9/18: none", "r18") + "\n" + _bullet("radar 9/17: none", "r17") + "\n")
    snapshot = {k: ws.read(k) or "" for k in E.LIVE_KEYS}
    originals = "\n".join(b.raw for b in state_blocks.parse(ws.read("state"))[1])
    tool = WriteTool(str(tmp_path))
    out = tool.execute(path="orbital/PROJECT_STATE.md",
                       content="## Now\n- radar 9/17–18: none\n" + originals + "\n")
    assert '"status": "success"' in out.content
    new_id = [b.id for b in state_blocks.parse(ws.read("state"))[1] if b.id not in ("r18", "r17")][0]
    WF._append_archive_section(ws, "state_archive", f"## [merged {TODAY} into id:{new_id}]", originals)
    merged_only = [b for b in state_blocks.parse(ws.read("state"))[1] if b.id == new_id][0]
    out = tool.execute(path="orbital/PROJECT_STATE.md", content="## Now\n" + merged_only.raw + "\n")
    assert '"status": "success"' in out.content and "put back" not in out.content
    assert WF.postpass_conservation(ws, snapshot, ["state"]) == []
    live = ws.read("state")
    blocks = state_blocks.parse(live)[1]
    assert len(blocks) == 1 and blocks[0].text == "radar 9/17–18: none"
    new_id = blocks[0].id
    assert new_id not in ("r18", "r17")
    archive = ws.read("state_archive")
    assert f"## [merged {TODAY} into id:{new_id}]" in archive
    assert "id:r18" in archive and "id:r17" in archive
    assert f"· merged {TODAY} → id:{new_id} ·" in archive


# ---------------------------------------------------------------------------
# 11. deletions
# ---------------------------------------------------------------------------

def test_the_pointer_and_age_machinery_is_gone():
    for name in ("pointer_line", "pointer_summary", "demote_with_pointers", "floor_state",
                 "split_out_pointer_lines", "PROTECT_OLDEST", "ENTRY_MARKERS", "STATE_PROBATION_DAYS"):
        assert not hasattr(M, name), name
    for name in ("_clean_pointer", "plan_durable", "plan_state", "normalise_ids", "_pointer_summary",
                 "build_prompt", "_SYSTEM", "apply_choices", "plan_entries"):
        assert not hasattr(E, name), name
    import inspect
    src = inspect.getsource(E.build_task_message)
    assert "pointer_line" not in src.lower()
