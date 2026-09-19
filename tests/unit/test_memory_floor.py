# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The deterministic floor and pointer format (spec 089 §4.3, §4.4).

PROJECT_STATE's floor is AGE-based: it never cuts a line whose ``created``
stamp is inside the 14-day probation window, it moves the oldest first, and
every move leaves ``[archived DATE id:X] … → PROJECT_STATE_ARCHIVE.md``. On
2026-09-18 the old position-based trim archived the GOAI booth prep four days
before the event; the replay test below pins that it can no longer happen.
"""

import re

from agent_os.agent import memory_entries as M
from agent_os.agent import state_blocks
from agent_os.agent.workspace_files import WorkspaceFileManager, _apply_hard_caps


TODAY = "2026-09-18"
ARCH = "PROJECT_STATE_ARCHIVE.md"


def _bullet(text, eid, created):
    return f"- {text}\n  <!--mem id:{eid} created:{created} touched:{created}-->"


def _state(*bullets, heading="## Now"):
    return heading + "\n" + "\n".join(bullets) + "\n"


# ---------------------------------------------------------------------------
# pointer format
# ---------------------------------------------------------------------------

def test_pointer_line_format():
    line = M.pointer_line("2026-09-19", ["c9bf60"], "GOAI booth prep", ARCH)
    assert line == "[archived 2026-09-19 id:c9bf60] GOAI booth prep → PROJECT_STATE_ARCHIVE.md"
    assert state_blocks.pointer_date(line) == "2026-09-19"


def test_pointer_line_with_several_ids_and_multiline_summary():
    line = M.pointer_line("2026-09-19", ["a1", "b2"], "two\nlines → here", ARCH)
    assert line.startswith("[archived 2026-09-19 id:a1 id:b2] two lines -> here → ")
    assert "\n" not in line


def test_legacy_idless_stub_is_a_pointer_too():
    assert state_blocks.pointer_date(
        "[archived 2026-07-28] Old stance → DECISIONS_ARCHIVE.md") == "2026-07-28"


# ---------------------------------------------------------------------------
# floor_state
# ---------------------------------------------------------------------------

def test_floor_leaves_a_fitting_file_alone():
    content = _state(_bullet("short", "aaaaaa", "2026-01-01"))
    r = M.floor_state(content, 1000, TODAY)
    assert r.content is content and r.archived == "" and r.moved_ids == []


def test_floor_never_cuts_inside_probation_even_far_over_target(caplog):
    young = [_bullet(f"recent line {i} " + "x" * 200, f"a0000{i}", "2026-09-10") for i in range(5)]
    content = _state(*young)
    r = M.floor_state(content, 10, TODAY)
    assert r.content is content
    assert r.moved_ids == []
    assert r.over_by > 0


def test_floor_moves_oldest_first_and_leaves_id_pointers():
    content = _state(
        _bullet("newest " + "n" * 200, "new001", "2026-09-17"),
        _bullet("middle " + "m" * 200, "mid001", "2026-08-01"),
        _bullet("oldest " + "o" * 200, "old001", "2026-07-01"),
    )
    target = int(M.est_tokens(M._budget_text(content, "state"))) - 20
    r = M.floor_state(content, target, TODAY)
    assert r.moved_ids == ["old001"]
    assert "oldest " + "o" * 200 not in r.content
    assert "[archived 2026-09-18 id:old001] oldest" in r.content
    assert r.content.rstrip().endswith(ARCH)
    # verbatim, with its id, in the archive text
    assert r.archived == _bullet("oldest " + "o" * 200, "old001", "2026-07-01")
    assert "middle" in r.content and "newest" in r.content


def test_floor_keeps_the_pointer_where_the_bullet_was():
    content = _state(
        _bullet("first " + "f" * 300, "fst001", "2026-06-01"),
        _bullet("second", "snd001", "2026-09-17"),
    )
    r = M.floor_state(content, 20, TODAY)
    lines = r.content.split("\n")
    assert lines[1].startswith("[archived 2026-09-18 id:fst001]")
    assert lines[2] == "- second"


def test_floor_removes_aged_pointers_without_re_archiving():
    old_ptr = "[archived 2026-08-01 id:zzz999] ancient thing → PROJECT_STATE_ARCHIVE.md"
    content = "## Now\n" + old_ptr + "\n" + "Prose " + "p" * 100 + "\n"
    r = M.floor_state(content, 20, TODAY)
    assert old_ptr not in r.content
    assert r.dropped_pointers == 1
    assert r.archived == ""


def test_floor_never_touches_prose_headings_or_young_pointers():
    young_ptr = "[archived 2026-09-15 id:yyy111] recent move → PROJECT_STATE_ARCHIVE.md"
    content = "## Heading\nProse " + "p" * 400 + "\n" + young_ptr + "\n"
    r = M.floor_state(content, 5, TODAY)
    assert r.content is content


def test_floor_never_grows_the_file_with_a_longer_pointer():
    """A bullet shorter than its own pointer stays: moving it would make the
    file bigger, the opposite of what the floor is for."""
    content = _state(*[_bullet(f"tiny {i}", f"t0000{i}", "2026-01-01") for i in range(9)])
    r = M.floor_state(content, 5, TODAY)
    assert r.content is content
    assert r.moved_ids == []


def test_floor_ignores_unstamped_bullets():
    content = "## Now\n- no comment at all " + "u" * 400 + "\n"
    assert M.floor_state(content, 5, TODAY).content is content


def test_floor_moves_continuation_lines_with_their_bullet():
    block = (
        "- old parent line\n"
        "  <!--mem id:par001 created:2026-06-01 touched:2026-06-01-->\n"
        "  - nested detail " + "d" * 300
    )
    content = "## Now\n" + block + "\n- keep me\n  <!--mem id:kep001 created:2026-09-17 touched:2026-09-17-->\n"
    r = M.floor_state(content, 20, TODAY)
    assert "d" * 300 not in r.content
    assert "  - nested detail " + "d" * 300 in r.archived
    assert "- keep me" in r.content


# ---------------------------------------------------------------------------
# The 2026-09-18 replay: the booth prep is safe now
# ---------------------------------------------------------------------------

def test_replay_recent_booth_prep_survives_the_floor():
    """Two old lines and the GOAI booth prep (created 4 days earlier). The
    old trim cut from the bottom — where the booth prep sat. The floor may
    only take the old lines."""
    booth = "GOAI 开源周展位准备（9/22–23 云谷中心）：单页400张、贴纸300枚" + "。" * 80
    content = (
        M.FORMAT_HEADERS["state"] + "\n"
        + _state(
            _bullet("Adjacent-repo run #13 posted " + "a" * 300, "adj013", "2026-08-02"),
            _bullet("Cold-DM automation live " + "c" * 300, "cdm001", "2026-08-03"),
            _bullet(booth, "goai01", "2026-09-14"),
        )
    )
    r = M.floor_state(content, 60, TODAY)
    assert booth in r.content
    assert set(r.moved_ids) == {"adj013", "cdm001"}
    for eid in r.moved_ids:
        assert f"id:{eid}" in r.content          # pointer left
        assert f"id:{eid}" in r.archived         # body archived with its id


# ---------------------------------------------------------------------------
# demote_with_pointers (DECISIONS / LESSONS)
# ---------------------------------------------------------------------------

def _decision(title, eid, touched, body_len=400):
    return (
        f"## {title} <!--mem id:{eid} created:2026-06-01 touched:{touched}-->\n"
        f"**Chose:** {'c' * body_len}\n\n"
    )


def test_demotion_leaves_id_pointers_and_keeps_existing_pointers():
    old_ptr = "[archived 2026-07-28] Old stance → DECISIONS_ARCHIVE.md"
    content = (
        "# DECISIONS\n\n"
        + _decision("Foundation 1", "f1", "2026-01-01")
        + _decision("Foundation 2", "f2", "2026-01-01")
        + _decision("Foundation 3", "f3", "2026-01-01")
        + _decision("Cold one", "cold", "2026-02-01")
        + old_ptr + "\n\n"
        + _decision("Warm one", "warm", "2026-09-01")
    )
    target = int(M.est_tokens(M._budget_text(content, "decisions"))) - 50
    kept, demoted, ids = M.demote_with_pointers(
        content, "decisions", target, TODAY, "DECISIONS_ARCHIVE.md")
    assert ids == ["cold"]
    assert "## Cold one" not in kept
    assert "[archived 2026-09-18 id:cold] Cold one → DECISIONS_ARCHIVE.md" in kept
    assert old_ptr in kept and old_ptr not in demoted
    assert "## Cold one <!--mem id:cold" in demoted
    assert "## Warm one" in kept and "Foundation 1" in kept


def test_demotion_never_moves_an_entry_without_an_id():
    content = (
        "# D\n\n"
        + _decision("A", "a", "2026-01-01") + _decision("B", "b", "2026-01-01")
        + _decision("C", "c", "2026-01-01")
        + "## No id at all\n" + "x" * 800 + "\n\n"
    )
    kept, demoted, ids = M.demote_with_pointers(content, "decisions", 10, TODAY, "DECISIONS_ARCHIVE.md")
    assert "## No id at all" in kept
    assert ids == []


# ---------------------------------------------------------------------------
# inject_view: over hard budget, the OLDEST lines leave the view first
# ---------------------------------------------------------------------------

def test_state_view_hides_oldest_lines_first_and_strips_comments():
    content = _state(
        _bullet("old one " + "o" * 150, "o00001", "2026-06-01"),
        _bullet("NEWEST LINE", "n00001", "2026-09-17"),
        _bullet("mid one " + "m" * 150, "m00001", "2026-08-01"),
    )
    view = M.inject_view(content, "state", 60)
    assert "NEWEST LINE" in view
    assert "old one" not in view
    assert "<!--mem" not in view
    assert "omitted from this view" in view
    assert M.inject_view(content, "state", 60) == view    # deterministic


def test_state_view_everything_fits_shows_everything_without_comments():
    content = _state(_bullet("a", "a00001", "2026-06-01"), _bullet("b", "b00001", "2026-09-01"))
    view = M.inject_view(content, "state", 1000)
    assert view == "## Now\n- a\n- b\n"


# ---------------------------------------------------------------------------
# trim_volatile is INDEX-only now: no flag protection, pointers pinned
# ---------------------------------------------------------------------------

def test_trim_volatile_pins_archive_pointer_lines():
    content = "\n".join(f"- path/file{i}.py — does thing {i}" for i in range(20))
    content += "\n- LESSONS_ARCHIVE.md — older lessons (read on demand)"
    out = M.trim_volatile(content, 60)
    assert "LESSONS_ARCHIVE.md" in out
    assert "file19.py" not in out


def test_trim_volatile_without_pointers_is_the_legacy_head_trim():
    content = "\n".join(f"- path/to/file{i}.py — does thing {i}" for i in range(1, 15))
    assert M.trim_volatile(content, 20) == M._head_within(content, 20)


# ---------------------------------------------------------------------------
# _apply_hard_caps wiring
# ---------------------------------------------------------------------------

def test_hard_caps_state_floor_archives_with_ids_and_logs_when_protected(tmp_path, caplog, monkeypatch):
    import logging
    wf = WorkspaceFileManager(str(tmp_path))
    old = _bullet("old stuff " + "o" * 3000, "old777", "2026-01-01")
    young = _bullet("young stuff " + "y" * 9000, "yng777", M._today())
    wf.write("state", _state(old, young))
    with caplog.at_level(logging.INFO):
        _apply_hard_caps(wf)
    live = wf.read("state")
    archive = wf.read("state_archive")
    assert "o" * 3000 not in live and "[archived " in live and "id:old777" in live
    assert "young stuff" in live
    assert "old stuff " + "o" * 3000 in archive and "id:old777" in archive
    assert re.search(r"## \[trimmed \d{4}-\d{2}-\d{2}\]", archive)
    assert any("younger than" in r.getMessage() for r in caplog.records)


def test_hard_caps_stamps_idless_decisions_before_demoting(tmp_path):
    wf = WorkspaceFileManager(str(tmp_path))
    entries = "".join(
        f"## 2026-0{m}-01: Decision {m}\n**Chose:** {'z' * 9000}\n\n" for m in range(1, 6)
    )
    wf.write("decisions", "# DECISIONS\n\n" + entries)
    _apply_hard_caps(wf)
    live = wf.read("decisions")
    archive = wf.read("decisions_archive") or ""
    # every demoted entry is addressable: a pointer in live, the id in the archive
    for eid in re.findall(r"\[archived \d{4}-\d{2}-\d{2} id:(\S+)\]", live):
        assert f"id:{eid}" in archive
    assert "[archived" in live
