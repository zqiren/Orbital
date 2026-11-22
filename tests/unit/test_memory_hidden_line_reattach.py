# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 192: a hidden ``<!--mem …-->`` line an agent left below a detail line
is put back under its bullet (``memory_entries.reattach_hidden_lines``) before
the write rules run, on DECISIONS and LESSONS, on both the new and the
previous text. Every other shape, and PROJECT_STATE, exactly as before.

Ported from the Orbital plugin's ``tests/test_hidden_lines.py``; the plugin's
record step and dashboard are replaced by the app's own write paths: the
``write``/``edit`` tools (``process_on_write``), the memory pass's
``prepass`` and ``WorkspaceFileManager.write`` (``prepare``). Invented files
only."""

from __future__ import annotations

import os
import shutil

import pytest

from agent_os.agent import loop
from agent_os.agent import memory_entries as me
from agent_os.agent import state_blocks
from agent_os.agent import workspace_files as wf
from agent_os.agent.tools.edit import EditTool

BASE_DAY = "2026-11-01"
DAY1, DAY2, DAY3, DAY4 = "2026-11-19", "2026-11-20", "2026-11-21", "2026-11-22"

B1 = "- Use SQLite for the index because it ships with Python."
B2 = "- Keep the folder format exactly the app's."
B1_REWORDED = "- Use SQLite (with FTS5) for the index because it ships with Python."
D1 = "  Chosen after FTS5 was checked."
D2 = "  Checked against the app's reader."
OLD_D2 = "  The app opens it with no import."
THIRD = "- Third entry stays untouched."
PLAIN = f"# {{name}}\n\n## Storage\n{B1}\n{B2}\n{OLD_D2}\n{THIRD}\n"
BY, SESSION = "codex", "codex:abc"
# The app's own id for each bullet ("matches"), and an id minted for older
# wording ("differs", about one real entry in five).
IDS = {"matches": ("use-sqlite-for-the-index-because", "keep-the-folder-format-exactly-t"),
       "differs": ("sqlite-search-index", "folder-format-kept")}
KEYS = ("decisions", "lessons")


# --------------------------------------------------------------------------
# The function itself
# --------------------------------------------------------------------------

C = "  <!--mem id:x created:2026-11-01 touched:2026-11-01-->"


def test_a_stray_hidden_line_moves_under_its_bullet_and_nothing_else_moves():
    text = f"# D\n\n- One.\n  detail a\n  detail b\n{C}\n- Two.\n  <!--mem id:y-->\n  more\n"
    out = me.reattach_hidden_lines(text)
    assert out == f"# D\n\n- One.\n{C}\n  detail a\n  detail b\n- Two.\n  <!--mem id:y-->\n  more\n"
    assert me.reattach_hidden_lines(out) is out          # twice changes nothing


@pytest.mark.parametrize("text", [
    "",
    None,
    "# D\n\n- One.\n  detail\n",                                       # no hidden line
    f"# D\n\n- One.\n{C}\n  detail\n",                                 # already in place
    f"# D\n\n- One.\n  detail\n{C}\n  <!--mem id:z-->\n",              # two hidden lines
    f"# D\n\n- One.\n{C}\n  detail\n  <!--mem id:z-->\n",              # two, one in place
    "# D\n\n- One.\n  detail <!--mem id:x-->\n",                       # shares its line
    "# D\n\n- One. <!--mem id:x-->\n  detail\n",                       # on the bullet line
    "# D\n\n- One.\n  detail\n  <!--mem id:x\n  created:2026-11-01-->\n",   # spans two lines
    f"# D\n\n- One.\n  detail\n\n{C.strip()}\n",                       # outside the block
])
def test_other_shapes_are_left_byte_for_byte(text):
    assert me.reattach_hidden_lines(text) == text


def test_crlf_and_odd_indent_are_kept_on_the_moved_line():
    text = "- One.\r\n    detail\r\n\t<!--mem id:x-->\r\n"
    assert me.reattach_hidden_lines(text) == "- One.\r\n\t<!--mem id:x-->\r\n    detail\r\n"


def test_the_format_header_and_other_blocks_keep_their_lines():
    header = me.FORMAT_HEADERS["decisions"]
    text = (f"{header}\n# DECISIONS.md\n\n## A\n- One.\n  detail\n{C}\n\n## B\n"
            f"- Two.\n  <!--mem id:y-->\n")
    out = me.reattach_hidden_lines(text)
    assert out.split("\n")[0] == header
    assert out == text.replace(f"- One.\n  detail\n{C}", f"- One.\n{C}\n  detail")
    assert len(out.split("\n")) == len(text.split("\n"))


# --------------------------------------------------------------------------
# Invented projects
# --------------------------------------------------------------------------


def _path(ws, key):
    return os.path.join(ws, "orbital", me.MEMORY_FILENAME[key])


def _read(ws, key):
    with open(_path(ws, key), encoding="utf-8") as fh:
        return fh.read()


def _write(ws, key, text):
    with open(_path(ws, key), "w", encoding="utf-8") as fh:
        fh.write(text)


def _project(tmp_path, key, ids, name="ws"):
    """A project whose ``key`` file the app wrote, with entries 1 and 2
    carrying ``by`` and ``session`` marks (fields the app keeps as given)."""
    ws = str(tmp_path / name)
    os.makedirs(os.path.join(ws, "orbital"))
    text, _w = me.process_on_write(ws, _path(ws, key), PLAIN.format(name=me.MEMORY_FILENAME[key]),
                                   today=BASE_DAY)
    for app_id, our_id in zip(IDS["matches"], IDS[ids]):
        text = text.replace(f"id:{app_id} ", f"id:{our_id} ")
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        if ln in (B1, B2):
            lines[i + 1] = lines[i + 1].replace("-->", f" by:{BY} session:{SESSION}-->")
    text = "\n".join(lines)
    _write(ws, key, text)
    assert me.process_on_write(ws, _path(ws, key), text, today=BASE_DAY)[0] == text  # the app's own
    return ws


# The agent's edits, on the raw text it reads.
def edit_a(t):              # detail between the bullet and its hidden line
    return t.replace(B1 + "\n", f"{B1}\n{D1}\n", 1)


def edit_a2(t):             # the same on an entry that already had detail
    return t.replace(B2 + "\n", f"{B2}\n{D2}\n", 1)


def edit_ad(t):             # the bullet reworded and a detail line above the hidden line
    return t.replace(B1 + "\n", f"{B1_REWORDED}\n{D1}\n", 1)


def edit_b(t):              # detail after the hidden line
    lines = t.split("\n")
    lines.insert(lines.index(B1) + 2, D1)
    return "\n".join(lines)


def edit_c(t):              # a new bullet with detail, no hidden line
    return t.rstrip("\n") + "\n- Load trims to 9,500 characters.\n  Its hook cap is 10,000.\n"


def edit_d(t):              # the bullet reworded in place
    return t.replace(B1, B1_REWORDED, 1)


def edit_e(t):              # the hidden line deleted
    lines = t.split("\n")
    del lines[lines.index(B1) + 1]
    return "\n".join(lines)


def edit_f(t):              # the hidden line copied onto a new bullet
    lines = t.split("\n")
    hidden = lines[lines.index(B1) + 1]
    return t.rstrip("\n") + f"\n- Load trims to 9,500 characters.\n{hidden}\n"


# (edit, the entry's bullet after it, its detail after it, which entry: 0 or 1)
STRAY = {
    "a": (edit_a, B1, [D1], 0),
    "a2": (edit_a2, B2, [D2, OLD_D2], 1),
    "a+d": (edit_ad, B1_REWORDED, [D1], 0),
}
CONTROLS = {"b": edit_b, "c": edit_c, "d": edit_d, "e": edit_e, "f": edit_f}


def _blocks(text):
    return state_blocks.parse(text)[1]


def _assert_one_entry_kept(text, case, ids):
    """One copy of the entry, its id, created date, by and session kept, the
    hidden line directly under the bullet and the detail under that."""
    _edit, bullet, detail, n = STRAY[case]
    want_id = IDS[ids][n]
    stem = ("- Use SQLite", "- Keep the folder")[n]
    copies = [b for b in _blocks(text) if b.content_lines[0].startswith(stem)]
    assert len(copies) == 1, text
    b = copies[0]
    assert b.content_lines[0] == bullet
    assert b.lines[1] == b.comment_line                  # directly under the bullet
    assert b.content_lines[1:] == detail
    assert b.id == want_id
    assert b.created == BASE_DAY
    assert b.fields.get("by") == BY and b.fields.get("session") == SESSION
    assert text.count(D1) + text.count(D2) == len([d for d in detail if d != OLD_D2])


def _tool_write(ws, key, change, today):
    """The write/edit tools' chokepoint: the disk is the previous text."""
    out, _w = me.process_on_write(ws, _path(ws, key), change(_read(ws, key)), today=today)
    _write(ws, key, out)
    return out


def _add_fact(n):
    return lambda t: t.rstrip("\n") + f"\n- Later fact number {n}.\n"


def _no_growth_after(ws, key, case, ids, *, now_n):
    """Two more tool writes, each adding one fact: the file grows by that
    fact only, the entry stays as it was, and the result is the app's own."""
    entry = None
    for n, day in ((2, DAY2), (3, DAY3)):
        text = _tool_write(ws, key, _add_fact(n), day)
        _assert_one_entry_kept(text, case, ids)
        assert len(_blocks(text)) == now_n + n - 1
        kept = next(b.raw for b in _blocks(text) if b.id == IDS[ids][STRAY[case][3]])
        assert entry is None or kept == entry
        entry = kept
        assert me.process_on_write(ws, _path(ws, key), text, today=DAY4)[0] == text


# --------------------------------------------------------------------------
# (a), (a2), (a+d): the write/edit tools' path
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ids", IDS)
@pytest.mark.parametrize("case", STRAY)
@pytest.mark.parametrize("key", KEYS)
def test_a_tool_write_keeps_one_entry_when_the_agent_wrote_above_the_hidden_line(
        tmp_path, key, case, ids):
    ws = _project(tmp_path, key, ids)
    text = _tool_write(ws, key, STRAY[case][0], DAY1)
    _assert_one_entry_kept(text, case, ids)
    assert len(_blocks(text)) == 3
    assert me.process_on_write(ws, _path(ws, key), text, today=DAY2)[0] == text
    _no_growth_after(ws, key, case, ids, now_n=3)


@pytest.mark.parametrize("ids", IDS)
@pytest.mark.parametrize("case", STRAY)
@pytest.mark.parametrize("key", KEYS)
def test_a_later_tool_write_repairs_the_shape_left_on_disk(tmp_path, key, case, ids):
    """No rule processed the edit (an external worker's own tools, a hand
    edit): the odd shape is on disk and is the next write's previous text."""
    ws = _project(tmp_path, key, ids)
    _write(ws, key, STRAY[case][0](_read(ws, key)))
    text = _tool_write(ws, key, _add_fact(1), DAY1)
    _assert_one_entry_kept(text, case, ids)
    assert len(_blocks(text)) == 4
    _no_growth_after(ws, key, case, ids, now_n=4)


@pytest.mark.parametrize("key", KEYS)
def test_where_the_app_kept_two_entries_one_write_now_keeps_one(tmp_path, monkeypatch, key):
    """The new rule (spec 192): a bullet reworded with a detail line added
    above its hidden line, in one edit, is ONE entry that keeps its id. The
    old write path (no pre-step) kept the old block beside the reworded
    bullet as a second entry."""
    ws = _project(tmp_path, key, "differs")
    prev = _read(ws, key)
    new = edit_ad(prev)
    ours, _w = me.process_on_write(ws, _path(ws, key), new, today=DAY1)
    assert len(_blocks(ours)) == 3
    _assert_one_entry_kept(ours, "a+d", "differs")
    b = next(b for b in _blocks(ours) if b.id == IDS["differs"][0])
    assert b.fields["touched"] == DAY1                    # the wording changed
    monkeypatch.setattr(me, "reattach_hidden_lines", lambda t: t)   # the old code
    old, _w = me.process_on_write(ws, _path(ws, key), new, today=DAY1)
    assert len(_blocks(old)) == 4


@pytest.mark.parametrize("key", KEYS)
def test_the_edit_tool_on_the_odd_shape_keeps_one_entry(tmp_path, key):
    """The real edit tool, anchored on a later line, on a file an external
    worker left in the odd shape (``differs``: the case that grew a copy
    per write)."""
    ws = _project(tmp_path, key, "differs")
    _write(ws, key, edit_a(_read(ws, key)))
    tool = EditTool(ws)
    for n in (1, 2, 3):
        last = [ln for ln in _read(ws, key).split("\n") if ln.strip()][-1]
        out = tool.execute(path=f"orbital/{me.MEMORY_FILENAME[key]}", old_text=last,
                           new_text=f"{last}\n- Edited fact number {n}.")
        assert '"status": "success"' in out.content, out.content
        text = _read(ws, key)
        b = next(b for b in _blocks(text) if b.content_lines[0] == B1)
        assert [x.content_lines[0] for x in _blocks(text)].count(B1) == 1
        assert b.id == IDS["differs"][0] and b.created == BASE_DAY
        assert b.fields.get("by") == BY and b.lines[1] == b.comment_line
        assert len(_blocks(text)) == 3 + n


# --------------------------------------------------------------------------
# (a), (a2), (a+d): the memory pass's prepass and the daemon's own writes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ids", IDS)
@pytest.mark.parametrize("case", STRAY)
@pytest.mark.parametrize("key", KEYS)
def test_the_prepass_and_daemon_writes_keep_one_entry(tmp_path, key, case, ids):
    ws = _project(tmp_path, key, ids)
    _write(ws, key, STRAY[case][0](_read(ws, key)))
    mgr = wf.WorkspaceFileManager(ws)
    res = wf.prepass(mgr, today=DAY1)
    assert key in res.touched
    text = _read(ws, key)
    _assert_one_entry_kept(text, case, ids)
    assert len(_blocks(text)) == 3
    assert key not in wf.prepass(mgr, today=DAY2).touched       # idempotent now
    entry = next(b.raw for b in _blocks(text) if b.id == IDS[ids][STRAY[case][3]])
    for n in (1, 2):
        mgr.write(key, _add_fact(n)(mgr.read(key)))
        text = _read(ws, key)
        _assert_one_entry_kept(text, case, ids)
        assert len(_blocks(text)) == 3 + n
        assert next(b.raw for b in _blocks(text) if b.id == IDS[ids][STRAY[case][3]]) == entry


@pytest.mark.parametrize("key", KEYS)
def test_a_daemon_write_of_an_odd_shape_keeps_one_entry(tmp_path, key):
    """``WorkspaceFileManager.write`` (a history restore, any daemon write)
    handed the odd shape: the stray line is re-attached in the new text."""
    ws = _project(tmp_path, key, "differs")
    mgr = wf.WorkspaceFileManager(ws)
    mgr.write(key, edit_ad(_read(ws, key)))
    _assert_one_entry_kept(_read(ws, key), "a+d", "differs")
    assert len(_blocks(_read(ws, key))) == 3


def test_a_new_entry_that_kept_an_old_hidden_line_inherits_its_id(tmp_path):
    """Spec 192 §7 risk: a bullet replaced by a different entry with the old
    hidden line left under its new detail takes the old id (as rewording in
    place does today); ``touched`` advances, nothing is restored."""
    ws = _project(tmp_path, "decisions", "differs")
    prev = _read(ws, "decisions")
    new = prev.replace(B1 + "\n", "- Use Postgres for the index.\n  Needed for concurrency.\n", 1)
    out, _w = me.process_on_write(ws, _path(ws, "decisions"), new, today=DAY1)
    b = next(b for b in _blocks(out) if b.id == IDS["differs"][0])
    assert b.content_lines == ["- Use Postgres for the index.", "  Needed for concurrency."]
    assert b.lines[1] == b.comment_line
    assert b.created == BASE_DAY and b.fields["touched"] == DAY1
    assert B1 not in out and len(_blocks(out)) == 3


# --------------------------------------------------------------------------
# (b) to (f): exactly as before
# --------------------------------------------------------------------------


def _flow(tmp_path, key, edit, name):
    """Every text the tool writes, the prepass and a daemon write leave, from
    one start."""
    out = []
    ws = _project(tmp_path, key, "differs", name=name)
    _tool_write(ws, key, edit, DAY1)
    out.append(_read(ws, key))
    _tool_write(ws, key, _add_fact(1), DAY2)
    out.append(_read(ws, key))
    # The edit straight on disk, no rule ran, then each later writer.
    for writer in ("tool", "prepass", "daemon"):
        ws2 = _project(tmp_path, key, "differs", name=f"{name}-raw-{writer}")
        _write(ws2, key, edit(_read(ws2, key)))
        mgr = wf.WorkspaceFileManager(ws2)
        if writer == "tool":
            _tool_write(ws2, key, _add_fact(1), DAY3)
        elif writer == "prepass":
            wf.prepass(mgr, today=DAY3)
        else:
            text, _w = mgr.prepare(key, _add_fact(1)(mgr.read(key)), today=DAY3)
            _write(ws2, key, text)
        out.append(_read(ws2, key))
        arch = os.path.join(ws2, "orbital", me.ARCHIVE_FILENAME[me.ARCHIVE_OF[key]])
        out.append(open(arch, encoding="utf-8").read() if os.path.exists(arch) else None)
    shutil.rmtree(ws)
    return out


@pytest.mark.parametrize("case", CONTROLS)
@pytest.mark.parametrize("key", KEYS)
def test_the_other_shapes_come_out_exactly_as_before(tmp_path, monkeypatch, key, case):
    edit = CONTROLS[case]
    now = _flow(tmp_path, key, edit, "now")
    monkeypatch.setattr(me, "reattach_hidden_lines", lambda t: t)   # the old code
    before = _flow(tmp_path, key, edit, "before")
    assert now == before


@pytest.mark.parametrize("case", ["a", "a2", "a+d", "b", "e"])
def test_project_state_never_goes_through_the_repair(tmp_path, monkeypatch, case):
    calls = []
    real = me.reattach_hidden_lines
    monkeypatch.setattr(me, "reattach_hidden_lines",
                        lambda t: calls.append(t) or real(t))
    ws = str(tmp_path / "ws")
    os.makedirs(os.path.join(ws, "orbital"))
    state = PLAIN.format(name="PROJECT_STATE.md")
    _write(ws, "state", me.process_on_write(ws, _path(ws, "state"), state, today=BASE_DAY)[0])
    edit = {**{k: v[0] for k, v in STRAY.items()}, **CONTROLS}[case]
    _tool_write(ws, "state", edit, DAY1)
    mgr = wf.WorkspaceFileManager(ws)
    _write(ws, "state", edit(_read(ws, "state")))
    wf.prepass(mgr, today=DAY2)
    mgr.write("state", _add_fact(1)(mgr.read("state")))
    assert calls == []


# --------------------------------------------------------------------------
# The wording that invited the shape (spec 192 §5.2)
# --------------------------------------------------------------------------


def _flat(text):
    return " ".join(text.split())


def test_the_record_rules_say_detail_goes_below_the_hidden_line():
    flat = _flat(loop.MEMORY_SHAPE_RULES)
    assert "below its hidden <!--mem--> line when it has one" in flat
    assert "Detail goes on indented lines under the bullet, only when needed." not in flat
    for text in (loop._MEMORY_RECORD_BODY,
                 loop.build_worker_memory_record_text(["a.py"])):
        assert "below its hidden <!--mem--> line when it has one" in _flat(text)


@pytest.mark.parametrize("key", ("state", "decisions", "lessons"))
def test_the_format_headers_say_detail_goes_below_the_mem_comment(key):
    assert "below its mem-comment line when it has one" in me.FORMAT_HEADERS[key]
