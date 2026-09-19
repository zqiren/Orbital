# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The asks list (spec 089 §3): ``orbital/ASKS.md``, an append-only event log.

Covers the grammar + fold, the append-only write path, the append API the
Workbench (and later the editor) closes asks through, the legacy ``[user]``
conversion, the per-call runtime block, and the one-time migration with its
backup / marker / resume / never-delete guarantees.
"""

from __future__ import annotations

import json
import os

import pytest

from agent_os.agent import asks
from agent_os.agent.retractions import Retraction, add_retraction

TODAY = "2026-09-19"


def _orbital(tmp_path):
    d = tmp_path / "orbital"
    d.mkdir(exist_ok=True)
    return d


def _asks_text(orbital_dir):
    return (orbital_dir / "ASKS.md").read_text(encoding="utf-8")


def _body_lines(content):
    return [
        ln for ln in content.split("\n")
        if ln.strip() and not ln.startswith("<!--format")
    ]


# ---------------------------------------------------------------------------
# Grammar + fold
# ---------------------------------------------------------------------------

class TestFold:
    def test_state_is_the_last_event(self):
        content = "\n".join([
            asks.FORMAT_HEADER,
            "- open a1b2c3 2026-09-01 Choose the venue",
            "- open d4e5f6 2026-09-02 due:2026-10-01 Send the deck",
            '- done a1b2c3 2026-09-03 by:agent "go with the rooftop"',
            "- dropped d4e5f6 2026-09-04 by:user",
            "- reopen d4e5f6 2026-09-05 by:user",
            "",
        ])
        by_id = {a.id: a for a in asks.fold(content)}
        assert by_id["a1b2c3"].state == "done"
        assert by_id["a1b2c3"].closed_by == "agent"
        assert by_id["a1b2c3"].note == '"go with the rooftop"'
        assert by_id["a1b2c3"].opened == "2026-09-01"
        assert by_id["d4e5f6"].state == "open"
        assert by_id["d4e5f6"].due == "2026-10-01"
        assert by_id["d4e5f6"].text == "Send the deck"
        assert by_id["d4e5f6"].updated == "2026-09-05"

    def test_prose_and_unknown_lines_are_ignored(self):
        content = "\n".join([
            "# Asks",
            "Some prose a human typed.",
            "- a plain bullet",
            "- done ffffff 2026-09-03 by:user",   # close for an unknown id
            "- open a1b2c3 2026-09-01 Real ask",
        ])
        folded = asks.fold(content)
        assert [a.id for a in folded] == ["a1b2c3"]

    def test_idless_open_gets_a_stable_text_derived_id(self):
        content = "- open Decide the booth budget\n"
        first = asks.fold(content)
        second = asks.fold(content)
        assert len(first) == 1
        assert first[0].id == second[0].id
        assert len(first[0].id) == 6
        assert first[0].state == "open"
        assert first[0].text == "Decide the booth budget"

    def test_idless_open_id_avoids_explicit_ids(self, monkeypatch):
        # Force the text hash to collide with an explicitly stamped ask.
        content_probe = "- open Something\n"
        synthetic = asks.fold(content_probe)[0].id
        content = f"- open {synthetic} 2026-09-01 Another ask\n- open Something\n"
        ids = [a.id for a in asks.fold(content)]
        assert len(ids) == 2 and len(set(ids)) == 2

    def test_due_token_anywhere_and_bracketed_ids_are_tolerated(self):
        content = "\n".join([
            "- open [a1b2c3] 2026-09-01 Call the vendor due:2026-09-30T15:00",
            "- done [a1b2c3] by:agent \"called\"",
        ])
        a = asks.fold(content)[0]
        assert a.due == "2026-09-30T15:00"
        assert a.text == "Call the vendor"
        assert a.state == "done"

    def test_render_round_trips(self):
        line = asks.render_event(asks.Event(
            kind="open", id="a1b2c3", date="2026-09-01", by=None,
            text="Send the deck", due="2026-10-01",
        ))
        assert line == "- open a1b2c3 2026-09-01 due:2026-10-01 Send the deck"
        a = asks.fold(line)[0]
        assert (a.id, a.text, a.due, a.opened) == (
            "a1b2c3", "Send the deck", "2026-10-01", "2026-09-01")


# ---------------------------------------------------------------------------
# Append-only write path
# ---------------------------------------------------------------------------

PREV = "\n".join([
    asks.FORMAT_HEADER,
    "- open a1b2c3 2026-09-01 Choose the venue",
    "- open d4e5f6 2026-09-02 Send the deck",
    "",
])


class TestProcessWrite:
    def test_new_open_is_stamped_and_appended(self):
        new = PREV + "- open due:2026-10-02 Book the flights\n"
        out, warns = asks.process_write(PREV, new, today=TODAY)
        assert out.startswith(PREV)
        last = _body_lines(out)[-1]
        assert last.startswith("- open ")
        a = [x for x in asks.fold(out) if x.text == "Book the flights"][0]
        assert a.opened == TODAY and a.due == "2026-10-02" and a.state == "open"
        assert a.id not in ("a1b2c3", "d4e5f6")
        assert any(a.id in w for w in warns)

    def test_note_quotes_the_stamped_last_line_for_the_next_edit(self):
        # The agent's next edit anchors on the text it wrote; after stamping
        # that text is gone, so the note must hand back what is on disk.
        new = PREV + "- open Book the flights\n"
        out, warns = asks.process_write(PREV, new, today=TODAY)
        last = _body_lines(out)[-1]
        assert any(f"ends with: {last}" in w for w in warns)

    def test_deleted_and_altered_lines_are_restored_with_a_warning(self):
        # The agent rewrote the file: dropped one line, reworded the other,
        # and added a close.
        new = "\n".join([
            asks.FORMAT_HEADER,
            "- open a1b2c3 2026-09-01 Choose the venue (rooftop?)",
            '- done a1b2c3 "rooftop, final"',
            "",
        ])
        out, warns = asks.process_write(PREV, new, today=TODAY)
        assert out.startswith(PREV)                      # originals intact, in order
        assert "Send the deck" in out
        by_id = {a.id: a for a in asks.fold(out)}
        assert by_id["a1b2c3"].state == "done"
        assert by_id["a1b2c3"].closed_by == "agent"
        assert by_id["d4e5f6"].state == "open"
        assert any("restored" in w for w in warns)

    def test_close_is_forced_to_by_agent_and_dated_today(self):
        new = PREV + '- done d4e5f6 2020-01-01 by:user "sent it this morning"\n'
        out, _ = asks.process_write(PREV, new, today=TODAY)
        last = _body_lines(out)[-1]
        assert last == f'- done d4e5f6 {TODAY} by:agent "sent it this morning"'

    def test_close_without_quote_is_kept_but_warned(self):
        new = PREV + "- done d4e5f6 sent\n"
        out, warns = asks.process_write(PREV, new, today=TODAY)
        assert {a.id: a for a in asks.fold(out)}["d4e5f6"].state == "done"
        assert any("quote" in w.lower() for w in warns)

    def test_close_of_unknown_id_is_rejected(self):
        new = PREV + '- done 999999 "yes"\n'
        out, warns = asks.process_write(PREV, new, today=TODAY)
        assert out == PREV
        assert any("999999" in w for w in warns)

    def test_duplicate_open_text_is_skipped(self):
        new = PREV + "- open Choose the venue\n"
        out, warns = asks.process_write(PREV, new, today=TODAY)
        assert out == PREV
        assert any("a1b2c3" in w for w in warns)

    def test_reopening_a_dropped_text_is_refused(self):
        prev = PREV + "- dropped d4e5f6 2026-09-03 by:user\n"
        new = prev + "- open Send the deck\n"
        out, warns = asks.process_write(prev, new, today=TODAY)
        assert out == prev
        assert any("dropped" in w for w in warns)

    def test_legacy_user_bullet_becomes_an_open_ask(self):
        new = PREV + "- [user due:2026-10-03] Approve the budget\n"
        out, _ = asks.process_write(PREV, new, today=TODAY)
        a = [x for x in asks.fold(out) if x.text == "Approve the budget"][0]
        assert a.state == "open" and a.due == "2026-10-03"

    def test_prose_is_dropped_with_a_warning(self):
        new = PREV + "Note to self: the venue is tricky\n"
        out, warns = asks.process_write(PREV, new, today=TODAY)
        assert out == PREV
        assert warns

    def test_whole_file_overwrite_with_one_line_appends_it(self):
        out, warns = asks.process_write(PREV, "- open Pick a date\n", today=TODAY)
        assert out.startswith(PREV)
        assert "Pick a date" in _body_lines(out)[-1]
        assert any("restored" in w for w in warns)

    def test_new_file_gets_the_header(self):
        out, _ = asks.process_write(None, "- open Pick a date\n", today=TODAY)
        assert out.startswith(asks.FORMAT_HEADER + "\n")
        assert [a.text for a in asks.fold(out)] == ["Pick a date"]

    def test_moved_line_is_not_reprocessed(self):
        # Same lines, reordered: nothing new, nothing lost, no bogus events.
        new = "\n".join([
            asks.FORMAT_HEADER,
            "- open d4e5f6 2026-09-02 Send the deck",
            "- open a1b2c3 2026-09-01 Choose the venue",
            "",
        ])
        out, _ = asks.process_write(PREV, new, today=TODAY)
        assert out == PREV

    def test_idless_lines_on_disk_are_stamped_in_place(self):
        prev = PREV + "- open Written by an external agent\n"
        synthetic = [a for a in asks.fold(prev) if a.text.startswith("Written")][0].id
        out, _ = asks.process_write(prev, prev + "- open Another\n", today=TODAY)
        assert f"- open {synthetic} {TODAY} Written by an external agent" in out
        assert "- open Written by an external agent" not in out


# ---------------------------------------------------------------------------
# Append API (Workbench, editor)
# ---------------------------------------------------------------------------

class TestAppendApi:
    def test_open_then_close_then_reopen(self, tmp_path):
        orbital = _orbital(tmp_path)
        a = asks.open_ask(str(orbital), "Choose the venue", due="2026-10-01", today=TODAY)
        assert a is not None and a.state == "open"
        line = asks.append_event(str(orbital), "done", a.id, '"rooftop"', "editor", today=TODAY)
        assert line == f'- done {a.id} {TODAY} by:editor "rooftop"'
        assert asks.read_asks(str(orbital))[0].state == "done"
        # Closing an already-closed ask is a no-op, not an error.
        assert asks.append_event(str(orbital), "dropped", a.id, "", "user", today=TODAY) is None
        asks.append_event(str(orbital), "reopen", a.id, "", "user", today=TODAY)
        assert asks.read_asks(str(orbital))[0].state == "open"
        assert _asks_text(orbital).startswith(asks.FORMAT_HEADER + "\n")

    def test_unknown_id_raises(self, tmp_path):
        orbital = _orbital(tmp_path)
        with pytest.raises(KeyError):
            asks.append_event(str(orbital), "done", "abcdef", "", "user", today=TODAY)

    def test_bad_kind_or_writer_raises(self, tmp_path):
        orbital = _orbital(tmp_path)
        a = asks.open_ask(str(orbital), "x", today=TODAY)
        with pytest.raises(ValueError):
            asks.append_event(str(orbital), "open", a.id, "", "user")
        with pytest.raises(ValueError):
            asks.append_event(str(orbital), "done", a.id, "", "robot")

    def test_append_normalizes_idless_lines_so_ids_line_up(self, tmp_path):
        orbital = _orbital(tmp_path)
        (orbital / "ASKS.md").write_text("- open External ask\n", encoding="utf-8")
        ext = asks.read_asks(str(orbital))[0]
        asks.append_event(str(orbital), "done", ext.id, "", "user", today=TODAY)
        text = _asks_text(orbital)
        assert f"- open {ext.id} {TODAY} External ask" in text
        assert asks.read_asks(str(orbital))[0].state == "done"

    def test_newlines_in_notes_never_break_the_one_line_grammar(self, tmp_path):
        orbital = _orbital(tmp_path)
        a = asks.open_ask(str(orbital), "multi\nline\ntext", today=TODAY)
        asks.append_event(str(orbital), "done", a.id, '"a\nb"', "agent", today=TODAY)
        got = asks.read_asks(str(orbital))[0]
        assert got.text == "multi line text"
        assert got.state == "done"


# ---------------------------------------------------------------------------
# Legacy [user] conversion (pure)
# ---------------------------------------------------------------------------

LEGACY_STATE = "\n".join([
    "<!--format old header-->",
    "## Focus",
    "- Plain fact stays.",
    "- [user] Pick option A or B for the landing page.",
    "  <!--mem id:a1b2c3 created:2026-09-01 touched:2026-09-02-->",
    "- [user due:2026-10-01] Send the deck to Acme.",
    "3. [user] Numbered decision to make",
    "- [user] Decide the venue:",
    "  - rooftop",
    "  - basement",
    "- [user] Already settled question.",
    "  <!--mem id:0e0e0e created:2026-08-01 resolved:2026-08-05-->",
    "- [due:2026-09-30] Dated fact stays.",
    "- [ ] a checkbox stays",
    "  - [user] nested flags are left alone",
    "",
])


class TestExtractLegacyFlags:
    def test_open_flags_become_asks_and_leave_the_state(self):
        new_state, events = asks.extract_legacy_flags(LEGACY_STATE)
        texts = [e.text for e in events]
        assert texts == [
            "Pick option A or B for the landing page.",
            "Send the deck to Acme.",
            "Numbered decision to make",
            "Decide the venue: rooftop; basement",
        ]
        first = events[0]
        assert first.id == "a1b2c3" and first.date == "2026-09-01"
        assert events[1].due == "2026-10-01"
        assert "Pick option A" not in new_state
        assert "id:a1b2c3" not in new_state
        assert "rooftop" not in new_state
        assert "- Plain fact stays." in new_state
        assert "- [due:2026-09-30] Dated fact stays." in new_state
        assert "- [ ] a checkbox stays" in new_state
        assert "  - [user] nested flags are left alone" in new_state

    def test_resolved_flag_is_a_settled_fact_not_an_ask(self):
        new_state, events = asks.extract_legacy_flags(LEGACY_STATE)
        assert "Already settled question." not in [e.text for e in events]
        assert "- Already settled question." in new_state
        assert "resolved:2026-08-05" in new_state        # comment untouched

    def test_state_without_flags_is_returned_unchanged(self):
        s = "## Focus\n- nothing flagged here\n"
        assert asks.extract_legacy_flags(s) == (s, [])
        assert asks.extract_legacy_flags("") == ("", [])


# ---------------------------------------------------------------------------
# Runtime block
# ---------------------------------------------------------------------------

class TestRuntimeBlock:
    def _write(self, orbital, lines):
        (orbital / "ASKS.md").write_text(
            asks.FORMAT_HEADER + "\n" + "\n".join(lines) + "\n", encoding="utf-8")

    def test_empty_when_nothing(self, tmp_path):
        orbital = _orbital(tmp_path)
        assert asks.render_runtime_block(str(orbital), today=TODAY) == ""
        self._write(orbital, ["- open a1b2c3 2026-09-01 x", "- done a1b2c3 2026-09-02 by:user"])
        assert asks.render_runtime_block(str(orbital), today=TODAY) == ""

    def test_none_cue_when_asked_for_and_nothing_is_open(self, tmp_path):
        orbital = _orbital(tmp_path)
        cue = "Open asks: none (orbital/ASKS.md)"
        assert asks.render_runtime_block(str(orbital), today=TODAY, empty_cue=True) == cue
        self._write(orbital, [
            "- open a1b2c3 2026-09-01 x", "- dropped a1b2c3 2026-09-02 by:user",
        ])
        block = asks.render_runtime_block(str(orbital), today=TODAY, empty_cue=True)
        assert block == cue + "\n\nDropped by the user — never re-propose:\n- x"
        self._write(orbital, ["- open a1b2c3 2026-09-01 x"])
        block = asks.render_runtime_block(str(orbital), today=TODAY, empty_cue=True)
        assert block == "Open asks (1):\n[a1b2c3] x"

    def test_order_overdue_and_due_soonest_first_then_newest(self, tmp_path):
        orbital = _orbital(tmp_path)
        self._write(orbital, [
            "- open 000001 2026-09-01 old undated",
            "- open 000002 2026-09-10 new undated",
            "- open 000003 2026-09-05 due:2026-09-25 due later",
            "- open 000004 2026-09-06 due:2026-09-15 overdue one",
        ])
        block = asks.render_runtime_block(str(orbital), today=TODAY)
        lines = block.split("\n")
        assert lines[0] == "Open asks (4):"
        assert lines[1] == "[000004] overdue one (due 2026-09-15, overdue)"
        assert lines[2] == "[000003] due later (due 2026-09-25)"
        assert lines[3] == "[000002] new undated"
        assert lines[4] == "[000001] old undated"

    def test_open_list_is_capped_at_ten(self, tmp_path):
        orbital = _orbital(tmp_path)
        self._write(orbital, [f"- open {i:06x} 2026-09-01 ask {i}" for i in range(13)])
        block = asks.render_runtime_block(str(orbital), today=TODAY)
        assert block.startswith("Open asks (13):")
        assert sum(1 for ln in block.split("\n") if ln.startswith("[")) == 10
        assert "…and 3 more in orbital/ASKS.md" in block

    def test_dropped_last_60_days_newest_first_capped_at_15(self, tmp_path):
        orbital = _orbital(tmp_path)
        lines = []
        for i in range(17):
            lines.append(f"- open {i:06x} 2026-08-01 dropped thing {i}")
            lines.append(f"- dropped {i:06x} 2026-09-{i + 1:02d} by:user")
        lines.append("- open 0000ff 2026-06-01 ancient")
        lines.append("- dropped 0000ff 2026-07-01 by:user")      # > 60 days ago
        self._write(orbital, lines)
        block = asks.render_runtime_block(str(orbital), today=TODAY)
        assert "Open asks" not in block
        dropped = block.split("\n")
        assert dropped[0] == "Dropped by the user — never re-propose:"
        assert dropped[1] == "- dropped thing 16"
        assert len(dropped) == 16
        assert "ancient" not in block

    def test_never_raises_on_unreadable_dir(self, tmp_path):
        assert asks.render_runtime_block(str(tmp_path / "nope"), today=TODAY) == ""


# ---------------------------------------------------------------------------
# One-time migration
# ---------------------------------------------------------------------------

def _seed_workspace(tmp_path, state=LEGACY_STATE, retractions=True):
    ws = tmp_path / "ws"
    orbital = ws / "orbital"
    orbital.mkdir(parents=True)
    (orbital / "PROJECT_STATE.md").write_text(state, encoding="utf-8")
    if retractions:
        add_retraction(str(orbital), Retraction(
            id="5e55c9", title="Decide if Multica DM precedes AionUi refresh.",
            reason="", date="2026-07-26"))
        add_retraction(str(orbital), Retraction(
            id="r2", title='**选方案 A / B / C**（默认 A）？',
            reason="changed my mind", date="2026-09-10"))
    return ws, orbital


class TestMigration:
    def test_first_run_backs_up_converts_and_marks(self, tmp_path):
        ws, orbital = _seed_workspace(tmp_path)
        original_state = (orbital / "PROJECT_STATE.md").read_text(encoding="utf-8")
        original_retr = (orbital / "retractions.md").read_text(encoding="utf-8")

        result = asks.migrate_project(str(ws), today=TODAY)
        assert result["status"] == "migrated"
        assert result["opened"] == 4 and result["dropped"] == 2

        # Backup holds the originals, byte-exact.
        backups = list((orbital / "backups").iterdir())
        assert len(backups) == 1 and backups[0].name.startswith("asks-migration-")
        assert (backups[0] / "PROJECT_STATE.md").read_text(encoding="utf-8") == original_state
        assert (backups[0] / "retractions.md").read_text(encoding="utf-8") == original_retr

        folded = {a.text: a for a in asks.read_asks(str(orbital))}
        assert folded["Pick option A or B for the landing page."].id == "a1b2c3"
        assert folded["Pick option A or B for the landing page."].opened == "2026-09-01"
        assert folded["Send the deck to Acme."].due == "2026-10-01"
        dm = folded["Decide if Multica DM precedes AionUi refresh."]
        assert dm.state == "dropped" and dm.closed_by == "user" and dm.id == "5e55c9"
        assert dm.updated == "2026-07-26"
        assert folded["**选方案 A / B / C**（默认 A）？"].note == "changed my mind"

        state = (orbital / "PROJECT_STATE.md").read_text(encoding="utf-8")
        assert [ln for ln in state.split("\n")[1:] if "[user" in ln] == [
            "  - [user] nested flags are left alone"]
        assert "Pick option A" not in state
        assert "- Already settled question." in state
        assert "- Plain fact stays." in state
        # The state header is refreshed to the current (asks-aware) contract.
        from agent_os.agent import memory_entries
        assert state.startswith(memory_entries.FORMAT_HEADERS["state"])

        # retractions.md is left on disk for older versions.
        assert (orbital / "retractions.md").read_text(encoding="utf-8") == original_retr
        marker = json.loads((orbital / asks.MIGRATION_MARKER).read_text(encoding="utf-8"))
        assert marker["version"] == 1

    def test_second_run_changes_nothing(self, tmp_path):
        ws, orbital = _seed_workspace(tmp_path)
        asks.migrate_project(str(ws), today=TODAY)
        snapshot = {p.name: p.read_bytes() for p in orbital.iterdir() if p.is_file()}
        result = asks.migrate_project(str(ws), today=TODAY)
        assert result["status"] == "unchanged"
        assert {p.name: p.read_bytes() for p in orbital.iterdir() if p.is_file()} == snapshot
        assert len(list((orbital / "backups").iterdir())) == 1

    def test_interrupted_migration_resumes_without_duplicates(self, tmp_path, monkeypatch):
        ws, orbital = _seed_workspace(tmp_path)
        real = asks._atomic_write

        def boom(path, content):
            if os.path.basename(path) == "PROJECT_STATE.md":
                raise OSError("disk yanked mid-migration")
            real(path, content)

        monkeypatch.setattr(asks, "_atomic_write", boom)
        with pytest.raises(OSError):
            asks.migrate_project(str(ws), today=TODAY)
        assert not (orbital / asks.MIGRATION_MARKER).exists()
        # ASKS.md was written first; PROJECT_STATE is untouched.
        assert "Pick option A" in (orbital / "PROJECT_STATE.md").read_text(encoding="utf-8")
        assert len(asks.read_asks(str(orbital))) == 6

        monkeypatch.setattr(asks, "_atomic_write", real)
        result = asks.migrate_project(str(ws), today=TODAY)
        assert result["status"] == "migrated"
        folded = asks.read_asks(str(orbital))
        assert len(folded) == 6                                    # no duplicates
        assert len({a.text for a in folded}) == 6
        assert "Pick option A" not in (orbital / "PROJECT_STATE.md").read_text(encoding="utf-8")
        assert (orbital / asks.MIGRATION_MARKER).exists()

    def test_after_migration_retractions_are_not_read_again(self, tmp_path):
        ws, orbital = _seed_workspace(tmp_path)
        asks.migrate_project(str(ws), today=TODAY)
        add_retraction(str(orbital), Retraction(
            id="r3", title="Added by an older version", reason="", date="2026-09-18"))
        asks.migrate_project(str(ws), today=TODAY)
        assert "Added by an older version" not in [a.text for a in asks.read_asks(str(orbital))]

    def test_flags_written_after_migration_are_swept_at_next_start(self, tmp_path):
        ws, orbital = _seed_workspace(tmp_path)
        asks.migrate_project(str(ws), today=TODAY)
        state_path = orbital / "PROJECT_STATE.md"
        state_path.write_text(
            state_path.read_text(encoding="utf-8") + "- [user] Written by v0.13 after a downgrade\n",
            encoding="utf-8")
        result = asks.migrate_project(str(ws), today=TODAY)
        assert result["status"] == "swept"
        assert "Written by v0.13 after a downgrade" in [a.text for a in asks.read_asks(str(orbital))]
        assert "Written by v0.13" not in state_path.read_text(encoding="utf-8")
        assert len(list((orbital / "backups").iterdir())) == 2

    def test_project_with_nothing_to_convert_only_gets_a_marker(self, tmp_path):
        from agent_os.agent import memory_entries
        ws = tmp_path / "ws"
        orbital = ws / "orbital"
        orbital.mkdir(parents=True)
        state = memory_entries.FORMAT_HEADERS["state"] + "\n## Focus\n- all quiet\n"
        (orbital / "PROJECT_STATE.md").write_text(state, encoding="utf-8")
        result = asks.migrate_project(str(ws), today=TODAY)
        assert result["status"] == "migrated"
        assert (orbital / "PROJECT_STATE.md").read_text(encoding="utf-8") == state
        assert not (orbital / "backups").exists()
        assert not (orbital / "ASKS.md").exists()

    def test_undecodable_state_is_never_rewritten(self, tmp_path):
        ws, orbital = _seed_workspace(tmp_path)
        raw = b"\xff\xfe legacy bytes\n- [user] Pick a venue\n"
        (orbital / "PROJECT_STATE.md").write_bytes(raw)
        result = asks.migrate_project(str(ws), today=TODAY)
        assert result["status"] == "migrated"
        assert (orbital / "PROJECT_STATE.md").read_bytes() == raw     # untouched
        assert result["dropped"] == 2                                  # the rest still ran

    def test_append_to_undecodable_asks_file_keeps_its_bytes(self, tmp_path):
        orbital = _orbital(tmp_path)
        raw = b"- open a1b2c3 2026-09-01 Pick a venue\n\xff\xfe stray bytes\n"
        (orbital / "ASKS.md").write_bytes(raw)
        asks.append_event(str(orbital), "done", "a1b2c3", "", "user", today=TODAY)
        data = (orbital / "ASKS.md").read_bytes()
        assert data.startswith(raw)
        assert data.endswith(f"- done a1b2c3 {TODAY} by:user\n".encode())

    def test_workspace_without_orbital_dir_is_skipped(self, tmp_path):
        assert asks.migrate_project(str(tmp_path / "empty"))["status"] == "no_orbital"

    def test_migrate_all_isolates_failures(self, tmp_path, monkeypatch):
        ws_ok, orbital_ok = _seed_workspace(tmp_path)
        broken = tmp_path / "broken"

        class Store:
            def list_projects(self):
                return [
                    {"project_id": "bad", "workspace": str(broken)},
                    {"project_id": "none", "workspace": ""},
                    {"project_id": "ok", "workspace": str(ws_ok)},
                ]

        real = asks.migrate_project

        def flaky(workspace, **kw):
            if workspace == str(broken):
                raise RuntimeError("boom")
            return real(workspace, **kw)

        monkeypatch.setattr(asks, "migrate_project", flaky)
        broken.mkdir()
        results = asks.migrate_all(Store())
        assert results["ok"]["status"] == "migrated"
        assert results["bad"]["status"] == "error"
        assert (orbital_ok / asks.MIGRATION_MARKER).exists()


# ---------------------------------------------------------------------------
# Wiring: the agent's write/edit tools reach the asks rules
# ---------------------------------------------------------------------------

class TestToolWiring:
    def test_edit_tool_cannot_delete_an_ask_line(self, tmp_path):
        from agent_os.agent.tools.edit import EditTool
        orbital = _orbital(tmp_path)
        (orbital / "ASKS.md").write_text(PREV, encoding="utf-8")
        tool = EditTool(str(tmp_path))
        result = tool.execute(
            path="orbital/ASKS.md",
            old_text="- open d4e5f6 2026-09-02 Send the deck\n",
            new_text="",
        )
        assert "restored" in result.content
        assert _asks_text(orbital) == PREV

    def test_write_tool_appends_and_stamps(self, tmp_path):
        from agent_os.agent.tools.write import WriteTool
        orbital = _orbital(tmp_path)
        (orbital / "ASKS.md").write_text(PREV, encoding="utf-8")
        tool = WriteTool(str(tmp_path))
        result = tool.execute(path="orbital/ASKS.md", content=PREV + '- done a1b2c3 "rooftop"\n')
        assert "done [a1b2c3]" in result.content
        assert {a.id: a for a in asks.read_asks(str(orbital))}["a1b2c3"].state == "done"

    def test_user_line_written_into_state_becomes_an_ask(self, tmp_path):
        from agent_os.agent import memory_entries
        orbital = _orbital(tmp_path)
        state_path = orbital / "PROJECT_STATE.md"
        state_path.write_text("## Focus\n- A fact.\n", encoding="utf-8")
        new = "## Focus\n- A fact.\n- [user due:2026-10-01] Approve the budget\n"
        out, warns = memory_entries.process_on_write(
            str(tmp_path), str(state_path), new, today=TODAY)
        assert "Approve the budget" not in out
        assert "- A fact." in out
        got = asks.read_asks(str(orbital))
        assert [(a.text, a.due, a.state) for a in got] == [
            ("Approve the budget", "2026-10-01", "open")]
        assert any("ASKS.md" in w for w in warns)
        # The same stale line again is not a second ask.
        memory_entries.process_on_write(str(tmp_path), str(state_path), new, today=TODAY)
        assert len(asks.read_asks(str(orbital))) == 1

    def test_state_write_never_advises_adding_user_flags(self, tmp_path):
        # A [user] line now becomes an ask, so "consider a [user] flag"
        # advice would turn ordinary prose into asks.
        from agent_os.agent import memory_entries
        orbital = _orbital(tmp_path)
        state_path = orbital / "PROJECT_STATE.md"
        new = "## Blockers\n- You must renew the domain.\n- None active.\n"
        _out, warns = memory_entries.process_on_write(
            str(tmp_path), str(state_path), new, today=TODAY)
        assert not any("[user] flag" in w for w in warns)

    def test_asks_path_is_a_memory_key(self, tmp_path):
        from agent_os.agent import memory_entries
        (tmp_path / "orbital").mkdir()
        assert memory_entries.memory_key_for_path(
            str(tmp_path / "orbital" / "ASKS.md"), str(tmp_path)) == "asks"


# ---------------------------------------------------------------------------
# What the agent sees: asks ride the per-call runtime block
# ---------------------------------------------------------------------------

class TestRuntimeInjection:
    def _prepare(self, tmp_path, user_text="hello", is_scratch=False):
        from agent_os.agent.context import ContextManager
        from agent_os.agent.prompt_builder import Autonomy, PromptContext
        from agent_os.agent.session import Session
        from agent_os.agent.workspace_files import WorkspaceFileManager

        class Builder:
            def build(self, context):
                return ("cached-system-prefix", "semi-stable-suffix", "dynamic-runtime")

        ctx = PromptContext(
            workspace=str(tmp_path), model="test-model", autonomy=Autonomy.HANDS_OFF,
            enabled_agents=[], tool_names=["read", "write"], os_type="linux",
            datetime_now="2026-01-01T00:00:00", context_usage_pct=0.0,
            is_scratch=is_scratch,
        )
        wfm = WorkspaceFileManager(str(tmp_path))
        wfm.ensure_dir()
        session = Session.new("asks-ctx", str(tmp_path))
        session.append({"role": "user", "content": user_text, "source": "user"})
        return ContextManager(session, Builder(), ctx, workspace_files=wfm).prepare()

    def test_open_and_dropped_asks_ride_the_last_user_turn(self, tmp_path):
        orbital = _orbital(tmp_path)
        (orbital / "ASKS.md").write_text("\n".join([
            asks.FORMAT_HEADER,
            "- open a1b2c3 2026-09-01 Choose the venue",
            "- open d4e5f6 2026-09-02 Sponsor the GOAI booth",
            "- dropped d4e5f6 2026-09-03 by:user",
            "",
        ]), encoding="utf-8")
        prepared = self._prepare(tmp_path)
        tail = prepared[-1]
        assert tail["role"] == "user"
        assert "[a1b2c3] Choose the venue" in tail["content"]
        assert "Dropped by the user — never re-propose:" in tail["content"]
        assert "Sponsor the GOAI booth" in tail["content"]
        for m in prepared[:-1]:
            assert "Choose the venue" not in str(m.get("content"))

    def test_retractions_file_is_no_longer_injected(self, tmp_path):
        orbital = _orbital(tmp_path)
        add_retraction(str(orbital), Retraction(
            id="x7f3a2", title="Send DM drafts", reason="no", date="2026-07-24"))
        prepared = self._prepare(tmp_path)
        assert not any("Retracted by user" in str(m.get("content")) for m in prepared)
        assert not any("Send DM drafts" in str(m.get("content")) for m in prepared)

    def test_no_open_asks_still_carries_the_none_cue(self, tmp_path):
        prepared = self._prepare(tmp_path)
        assert prepared[-1]["role"] == "user"
        assert prepared[-1]["content"].endswith("Open asks: none (orbital/ASKS.md)")
        for m in prepared[:-1]:
            assert "Open asks" not in str(m.get("content"))

    def test_scratch_projects_get_no_cue(self, tmp_path):
        prepared = self._prepare(tmp_path, is_scratch=True)
        assert not any("Open asks" in str(m.get("content")) for m in prepared)


# ---------------------------------------------------------------------------
# Prompt rules (semi-stable section, right after the memory section)
# ---------------------------------------------------------------------------

# Approved 2026-09-19 after live test (b): the agent-side "a decision you are
# parking" never matched a user saying "I'll decide after …", and a TBD note
# in the deliverable passed the "would something be lost?" test.
TRIGGER = (
    "or a decision the user put off for later (\"I'll decide after …\") while you "
    "carry on. A TBD note in a file or in chat is not an ask — only ASKS.md "
    "reaches the user's Workbench."
)


class TestPromptRules:
    def _semi(self, **over):
        from agent_os.agent.prompt_builder import Autonomy, PromptBuilder, PromptContext
        base = dict(
            workspace="/tmp/ws", model="m", autonomy=Autonomy.HANDS_OFF,
            enabled_agents=[], tool_names=["read", "write", "edit"],
            os_type="macos", datetime_now="2026-09-19T00:00:00",
        )
        base.update(over)
        _, semi, _ = PromptBuilder().build(PromptContext(**base))
        return semi

    def test_rules_say_when_to_open_and_how_to_close(self):
        semi = " ".join(self._semi().split())
        assert "ASKS.md" in semi
        assert "Questions go in chat" in semi
        assert "if this session were closed now and never reopened" in semi
        assert "`- open <text>`" in semi
        assert "open it in that same turn, before you reply" in semi
        assert "If ASKS.md does not exist yet, create it" in semi
        assert "user's own words" in semi
        assert "if unsure, leave it open" in semi
        assert TRIGGER in semi
        assert "you are parking" not in semi
        assert "Never re-propose a dropped ask" in semi
        assert "[user]" not in semi

    def test_rules_follow_the_memory_section(self):
        semi = self._semi()
        assert semi.index("PROJECT_STATE.md: your current-state scratchpad") < semi.index("## Asks")

    def test_scratch_projects_get_no_asks_rules(self):
        assert "ASKS.md" not in self._semi(is_scratch=True)


# ---------------------------------------------------------------------------
# Call site: the migration runs once at daemon start, before routes serve
# ---------------------------------------------------------------------------

class TestDaemonStart:
    def test_create_app_migrates_every_project(self, tmp_path):
        from unittest.mock import patch

        from agent_os.api.app import create_app
        from agent_os.daemon_v2.project_store import ProjectStore

        ws, orbital = _seed_workspace(tmp_path)
        data = tmp_path / "data"
        store = ProjectStore(data_dir=str(data))
        store.create_project({"name": "P", "workspace": str(ws), "agent_name": "P"})
        if hasattr(store, "flush"):
            store.flush()
        with patch("agent_os.api.app.acquire_pid_file"):
            create_app(data_dir=str(data))
        assert (orbital / asks.MIGRATION_MARKER).exists()
        assert "Pick option A or B for the landing page." in [
            a.text for a in asks.read_asks(str(orbital))]


class TestTriggerWordsEverywhere:
    """The same trigger sentence reaches every reader: the management agent's
    prompt, ASKS.md's own header, AGENTS.md for external agents."""

    def test_same_words_in_prompt_header_and_agents_md(self):
        from agent_os.agent.prompt_builder import Autonomy, PromptBuilder, PromptContext
        from agent_os.daemon_v2.agent_md_seeder import AGENT_MD_TEMPLATE

        def flat(text):
            return " ".join(text.split())

        _, semi, _ = PromptBuilder().build(PromptContext(
            workspace="/tmp/ws", model="m", autonomy=Autonomy.HANDS_OFF,
            enabled_agents=[], tool_names=["read"], os_type="macos",
            datetime_now="2026-09-19T00:00:00"))
        agents_md = AGENT_MD_TEMPLATE.format(project_name="P", agent_name="A")
        for text in (semi, asks.FORMAT_HEADER, agents_md):
            assert TRIGGER in flat(text)


class TestWindowsLineEndings:
    """CRLF files (Windows) must not be rewritten or get doubled line breaks.

    `_read_exact` used to keep `\\r\\n` while `_atomic_write` writes in text
    mode, which on Windows turns every `\\n` into `\\r\\n` again — so each
    append doubled the file's line breaks and the first-start migration
    rewrote every Windows user's PROJECT_STATE even with nothing to convert.
    """

    def test_read_exact_normalises_line_endings(self, tmp_path):
        p = tmp_path / "x.md"
        p.write_bytes(b"a\r\nb\r\n")
        text, ok = asks._read_exact(str(p))
        assert ok and text == "a\nb\n"

    def test_crlf_state_with_nothing_to_convert_is_not_rewritten(self, tmp_path):
        from agent_os.agent import memory_entries
        ws = tmp_path / "ws"
        orbital = ws / "orbital"
        orbital.mkdir(parents=True)
        state = memory_entries.FORMAT_HEADERS["state"] + "\n## Focus\n- all quiet\n"
        raw = state.replace("\n", "\r\n").encode("utf-8")
        (orbital / "PROJECT_STATE.md").write_bytes(raw)
        result = asks.migrate_project(str(ws), today=TODAY)
        assert result["status"] == "migrated"
        assert result["backup"] is None                      # nothing touched
        assert (orbital / "PROJECT_STATE.md").read_bytes() == raw

    def test_append_to_a_crlf_log_keeps_one_line_per_event(self, tmp_path):
        orbital = tmp_path / "orbital"
        orbital.mkdir()
        (orbital / "ASKS.md").write_bytes(
            (asks.FORMAT_HEADER + "\n- open a1b2c3 2026-07-20 Pick a venue.\n")
            .replace("\n", "\r\n").encode("utf-8"))
        asks.append_event(str(orbital), "done", "a1b2c3", "", "user", today=TODAY)
        text = (orbital / "ASKS.md").read_text(encoding="utf-8")
        assert "\n\n" not in text
        assert [a.state for a in asks.read_asks(str(orbital))] == ["done"]
