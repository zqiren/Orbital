# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Layer-1 format contracts: in-file <!--format--> headers + report-only shape lint.

Design (user-approved plan): the format skeleton lives IN the file (self-healing
at both write chokepoints), the shape lint NEVER fires a hygiene flag or extra
checkpoint — it only rides along inside the session-end merge prompt.
"""
import os

import pytest

from agent_os.agent import memory_entries as mem
from agent_os.agent.workspace_files import WorkspaceFileManager


LAYER1_KEYS = ("state", "decisions", "lessons", "index")


class TestFormatHeaders:
    def test_all_four_keys_have_single_line_comment_headers(self):
        assert set(mem.FORMAT_HEADERS) == set(LAYER1_KEYS)
        for key, header in mem.FORMAT_HEADERS.items():
            assert header.startswith("<!--format"), key
            assert header.rstrip().endswith("-->"), key
            assert "\n" not in header.strip(), key

    def test_ensure_prepends_when_missing(self):
        out = mem.ensure_format_header("# INDEX\n- a.py — thing\n", "index")
        assert out.startswith(mem.FORMAT_HEADERS["index"])
        assert "# INDEX\n- a.py — thing\n" in out

    def test_ensure_idempotent(self):
        once = mem.ensure_format_header("# INDEX\n", "index")
        twice = mem.ensure_format_header(once, "index")
        assert twice == once
        assert twice.count("<!--format") == 1

    def test_ensure_on_empty_content(self):
        out = mem.ensure_format_header("", "state")
        assert out.startswith(mem.FORMAT_HEADERS["state"])


class TestWriteChokepoints:
    def test_manager_write_adds_header_to_layer1(self, tmp_path):
        wf = WorkspaceFileManager(str(tmp_path))
        wf.write("index", "# INDEX\n- a.py — thing\n")
        on_disk = wf.read("index")
        assert on_disk.startswith(mem.FORMAT_HEADERS["index"])
        assert "- a.py — thing" in on_disk

    def test_manager_write_does_not_duplicate_header(self, tmp_path):
        wf = WorkspaceFileManager(str(tmp_path))
        wf.write("index", mem.FORMAT_HEADERS["index"] + "\n# INDEX\n")
        assert wf.read("index").count("<!--format") == 1

    def test_manager_write_leaves_archives_alone(self, tmp_path):
        wf = WorkspaceFileManager(str(tmp_path))
        wf.write("decisions_archive", "# archive\n")
        assert "<!--format" not in wf.read("decisions_archive")

    def test_process_on_write_adds_header_for_volatile_files(self, tmp_path):
        # Previously state/index passed through untouched (no ENTRY_MARKERS).
        os.makedirs(tmp_path / "orbital", exist_ok=True)
        target = str(tmp_path / "orbital" / "INDEX.md")
        out, warns = mem.process_on_write(str(tmp_path), target, "# INDEX\n- a.py — x\n")
        assert out.startswith(mem.FORMAT_HEADERS["index"])

    def test_process_on_write_headers_and_stamps_durable_files(self, tmp_path):
        os.makedirs(tmp_path / "orbital", exist_ok=True)
        target = str(tmp_path / "orbital" / "DECISIONS.md")
        content = "# DECISIONS\n\n## 2026-07-09: Pick A\n**Chose:** A\n"
        out, warns = mem.process_on_write(str(tmp_path), target, content, today="2026-07-09")
        assert out.startswith(mem.FORMAT_HEADERS["decisions"])
        assert "<!--mem" in out  # stamping still applies

    def test_process_on_write_ignores_non_memory_paths(self, tmp_path):
        out, warns = mem.process_on_write(str(tmp_path), str(tmp_path / "notes.md"), "hello")
        assert out == "hello"
        assert warns == []


_CLEAN_INDEX = """# INDEX

## code
- src/main.py — entry point
- src/util.py — helpers

## docs
- docs/plan.md — the plan
- DECISIONS_ARCHIVE.md — older entries demoted from the live file (read on demand).
"""

# Modeled on the real drifted orbital-marketing INDEX.md.
_DIRTY_INDEX = """# Active Context

## 🚨 CRITICAL CONTEXT
- **v1 SHIPPED (2026-06-30)** — all gates unblocked.
- **KOL targets locked (2026-06-30)** — DM drafts ready.
- Watch trigger: Tutti VM launch coming soon and we should be very careful.

## Triggers
- **Daily Competitor Watch** — cron 0 14 * * *. 9 repos (07-08: added tutti).
"""

_CLEAN_STATE = """Current focus: shipping v1 installer.
Blockers: none.
Next: verify DMG on a clean machine.
"""

_DIRTY_STATE = """# STATE

## 2026-07-08 update
did things

## 2026-07-07 update
did other things
"""


class TestShapeReport:
    def test_clean_index_reports_none(self):
        assert mem.shape_report(_CLEAN_INDEX, "index") is None

    def test_format_header_line_is_not_a_violation(self):
        content = mem.ensure_format_header(_CLEAN_INDEX, "index")
        assert mem.shape_report(content, "index") is None

    def test_dirty_index_reports_dates_emoji_and_shape(self):
        report = mem.shape_report(_DIRTY_INDEX, "index")
        assert report is not None
        assert "INDEX" in report
        assert "dated" in report
        assert "emoji" in report

    def test_clean_state_reports_none(self):
        assert mem.shape_report(_CLEAN_STATE, "state") is None

    def test_dirty_state_reports_changelog_headers(self):
        report = mem.shape_report(_DIRTY_STATE, "state")
        assert report is not None
        assert "dated" in report

    def test_durable_files_not_linted_in_v1(self):
        assert mem.shape_report("anything ## 2026-01-01", "decisions") is None
        assert mem.shape_report("anything", "lessons") is None

    def test_empty_content_reports_none(self):
        assert mem.shape_report("", "index") is None
        assert mem.shape_report(None, "index") is None


class TestMergePromptRideAlong:
    def _summary(self):
        return {"recent_messages": [], "files_modified": [], "message_count": 0,
                "tool_calls_count": 0}

    def test_prompt_contains_formatting_section_when_dirty(self, tmp_path):
        wf = WorkspaceFileManager(str(tmp_path))
        os.makedirs(tmp_path / "orbital", exist_ok=True)
        (tmp_path / "orbital" / "INDEX.md").write_text(_DIRTY_INDEX, encoding="utf-8")
        prompt = wf.build_session_end_prompt(self._summary())
        assert "FORMATTING TO FIX" in prompt
        assert mem.FORMAT_HEADERS["index"] in prompt

    def test_prompt_omits_formatting_section_when_clean(self, tmp_path):
        wf = WorkspaceFileManager(str(tmp_path))
        os.makedirs(tmp_path / "orbital", exist_ok=True)
        (tmp_path / "orbital" / "INDEX.md").write_text(_CLEAN_INDEX, encoding="utf-8")
        prompt = wf.build_session_end_prompt(self._summary())
        assert "FORMATTING TO FIX" not in prompt

    def test_prompt_always_contains_cross_file_consistency_rule(self, tmp_path):
        wf = WorkspaceFileManager(str(tmp_path))
        prompt = wf.build_session_end_prompt(self._summary())
        assert "stale copies" in prompt


class TestShapeReportFilenameDates:
    """Dates embedded in artifact FILENAMES (agent_output/YYYY-MM-DD-*.md) are
    navigation, not drift — they must not count as dated lines."""

    def test_dated_filenames_are_clean(self):
        content = """# INDEX

## agent_output
- agent_output/2026-07-08-competitor-watch.md — daily watch report
- agent_output/2026-06-25-pilotdeck-technical-audit.md — PilotDeck audit
- ACTIVE-strategy-reframe-2026-04.md — date ends the stem, then extension
"""
        assert mem.shape_report(content, "index") is None

    def test_sentence_ending_date_still_flagged(self):
        content = """# INDEX

## stuff
- the thing shipped 2026-06-30.
- another thing landed 2026-07-01.
- and one more happened on 2026-07-02.
"""
        report = mem.shape_report(content, "index")
        assert report is not None and "dated" in report

    def test_prose_dates_still_flagged(self):
        content = """# INDEX

## stuff
- **v1 SHIPPED (2026-06-30)** — all gates unblocked
- plan launched 2026-06-30 and going well
- another dated thing happened (2026-07-01)
"""
        report = mem.shape_report(content, "index")
        assert report is not None and "dated" in report
