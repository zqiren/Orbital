# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the shared `[user]` flag grammar parser (user_flags.py).

Covers: parsing the spec §4 canonical entry (flagged, due, evidence w/ CJK +
spaces, unconfirmed, multi-line wrapped comment — legacy attributes the
parser still tolerates even though the receipt machinery that wrote them is
retired); dated-but-unflagged facts; plain bullets (including markdown
checkboxes) not surfaced; comment stripping/round-trip; id generation; and
lint warnings (malformed due only — the receipt-completeness check was
retired, spec amendment 2026-07-24).
"""
import re

from agent_os.agent import user_flags as uf


# ---------------------------------------------------------------------------
# The spec §4 canonical example, verbatim (multi-line wrapped mem-comment,
# CJK + spaces inside the evidence quote).
# ---------------------------------------------------------------------------
CANONICAL_ENTRY = (
    "- [user due:2026-07-28] Send 宝玉 + Simon DM drafts — only you can send "
    "from your accounts.\n"
    "  <!--mem id:x7f3a2 from:orbital-marketing_7c045c40\n"
    '      evidence:"EN 这边宝玉和 Simon 的 draft 写好就准备发" confidence:unconfirmed\n'
    "      created:2026-07-19 touched:2026-07-23-->\n"
)

# Modeled on the real orbital-marketing-style PROJECT_STATE — no tags at all.
REAL_PROSE = """<!--format: current-state scratchpad; OVERWRITE to reflect reality; NOT a changelog-->
# PROJECT_STATE

## Current Focus
Bug #29 investigating: duplicate messages + missing intermediate thinking when user enters queue messages.

## Active Work
- [x] Recorded bug in BACKLOG.md as item 29
- [x] Dispatched claude-code for deep codebase investigation
- [ ] Awaiting claude-code findings on root cause
- [ ] Will write spec/plan to `plans/` directory when investigation completes

## Blockers
None — awaiting investigation results.

## What's True Right Now
- Bug #29 has two defects: (D1) duplicate messages on queue entry, (D2) missing intermediate thinking on queue entry.
- Both likely related to the same queue-message insertion path.
"""


class TestParseCanonicalEntry:
    def test_parses_flagged_due_evidence_confidence(self):
        entries = uf.parse_entries(CANONICAL_ENTRY)
        assert len(entries) == 1
        e = entries[0]
        assert e.flagged is True
        assert e.due == "2026-07-28"
        assert e.text == (
            "Send 宝玉 + Simon DM drafts — only you can send from your accounts."
        )
        assert e.id == "x7f3a2"
        assert e.from_session == "orbital-marketing_7c045c40"
        assert e.evidence == "EN 这边宝玉和 Simon 的 draft 写好就准备发"
        assert e.confidence == "unconfirmed"
        assert e.created == "2026-07-19"
        assert e.touched == "2026-07-23"
        assert e.resolved is None

    def test_line_span_includes_comment_lines_0_based_inclusive(self):
        entries = uf.parse_entries(CANONICAL_ENTRY)
        e = entries[0]
        lines = CANONICAL_ENTRY.splitlines()
        assert e.line_start == 0
        # Comment spans lines 1-3 (0-based); line_end must include the last
        # comment line (the one holding the closing "-->").
        assert e.line_end == 3
        assert "-->" in lines[e.line_end]

    def test_evidence_preserves_internal_spaces_and_cjk(self):
        e = uf.parse_entries(CANONICAL_ENTRY)[0]
        assert " " in e.evidence
        assert "宝玉" in e.evidence and "Simon" in e.evidence


class TestDatedUnflaggedFact:
    def test_due_without_user_is_unflagged(self):
        content = "- [due:2026-07-23] Renew the domain before it lapses.\n"
        entries = uf.parse_entries(content)
        assert len(entries) == 1
        e = entries[0]
        assert e.flagged is False
        assert e.due == "2026-07-23"
        assert e.text == "Renew the domain before it lapses."
        assert e.id is None
        assert e.evidence is None


class TestPlainBulletsNotSurfaced:
    def test_plain_bullet_no_brackets_not_returned(self):
        content = "- Just a plain fact with no tag at all.\n"
        assert uf.parse_entries(content) == []

    def test_markdown_checkbox_bullets_not_treated_as_tags(self):
        # These are common in plans/backlogs (e.g. "- [ ] Write failing tests")
        # and must NOT be misparsed as a `[user]`/`[due:...]` bracket tag.
        content = (
            "- [ ] Awaiting claude-code findings on root cause\n"
            "- [x] Recorded bug in BACKLOG.md as item 29\n"
        )
        assert uf.parse_entries(content) == []

    def test_real_prose_parses_to_zero_entries(self):
        assert uf.parse_entries(REAL_PROSE) == []


class TestStripMemComments:
    def test_no_tags_round_trips_unchanged(self):
        assert uf.strip_mem_comments(REAL_PROSE) == REAL_PROSE

    def test_strips_multiline_comment_leaving_bullet_text_intact(self):
        stripped = uf.strip_mem_comments(CANONICAL_ENTRY)
        assert "<!--mem" not in stripped
        assert "-->" not in stripped
        assert "Send 宝玉 + Simon DM drafts" in stripped

    def test_strip_is_idempotent(self):
        once = uf.strip_mem_comments(CANONICAL_ENTRY)
        twice = uf.strip_mem_comments(once)
        assert once == twice

    def test_empty_and_none_pass_through(self):
        assert uf.strip_mem_comments("") == ""
        assert uf.strip_mem_comments(None) is None


class TestRenderCommentRoundTrip:
    def test_render_then_parse_recovers_fields(self):
        fields = {
            "id": "ab12cd",
            "from": "myproj_deadbeef",
            "evidence": "the user said 明天 do it — quotes \"inside\" too",
            "confidence": "unconfirmed",
            "created": "2026-07-20",
            "touched": "2026-07-23",
        }
        comment = uf.render_comment(fields)
        assert comment.startswith("<!--mem")
        assert comment.endswith("-->")
        content = f"- [user] Do the thing.\n  {comment}\n"
        entries = uf.parse_entries(content)
        assert len(entries) == 1
        e = entries[0]
        assert e.id == "ab12cd"
        assert e.from_session == "myproj_deadbeef"
        assert e.evidence == fields["evidence"]
        assert e.confidence == "unconfirmed"
        assert e.created == "2026-07-20"
        assert e.touched == "2026-07-23"

    def test_render_includes_resolved_when_present(self):
        comment = uf.render_comment({"id": "aaaaaa", "resolved": "2026-07-23"})
        assert "resolved:2026-07-23" in comment

    def test_render_omits_missing_fields(self):
        comment = uf.render_comment({"id": "aaaaaa"})
        assert "evidence" not in comment
        assert "from:" not in comment


class TestTagLessCommentCarryingBullets:
    """A fulfilled Workbench exit rewrites a retired entry as a tag-less
    bullet + adjacent mem-comment (id + resolved:<date>, spec §5.3). The
    entry must keep surfacing through parse_entries so the chokepoint can
    keep re-associating/preserving that comment across agent rewrites —
    the resolved stamp is the anti-resurrection trace."""

    def test_tagless_bullet_with_comment_is_returned(self):
        content = (
            "- Send the drafts.\n"
            "  <!--mem id:x7f3a2 resolved:2026-07-24 created:2026-07-19-->\n"
        )
        entries = uf.parse_entries(content)
        assert len(entries) == 1
        e = entries[0]
        assert e.flagged is False
        assert e.due is None
        assert e.text == "Send the drafts."
        assert e.id == "x7f3a2"
        assert e.resolved == "2026-07-24"
        assert e.created == "2026-07-19"
        assert e.line_start == 0
        assert e.line_end == 1

    def test_tagless_bullet_without_comment_still_absent(self):
        content = "- Send the drafts.\n"
        assert uf.parse_entries(content) == []


class TestNewEntryId:
    def test_six_hex_chars(self):
        eid = uf.new_entry_id()
        assert re.fullmatch(r"[0-9a-f]{6}", eid)

    def test_ids_are_not_all_identical(self):
        ids = {uf.new_entry_id() for _ in range(20)}
        assert len(ids) > 1


class TestPrefixCapture:
    def test_bullet_prefix_is_dash_space(self):
        e = uf.parse_entries(CANONICAL_ENTRY)[0]
        assert e.prefix == "- "

    def test_numbered_dot_prefix_captured(self):
        content = (
            "3. [user] Approve the vendor contract.\n"
            "  <!--mem id:abc123 from:s1 evidence:\"approve it\" "
            "created:2026-07-19-->\n"
        )
        entries = uf.parse_entries(content)
        assert len(entries) == 1
        e = entries[0]
        assert e.prefix == "3. "
        assert e.flagged is True
        assert e.text == "Approve the vendor contract."
        assert e.id == "abc123"

    def test_numbered_paren_prefix_captured(self):
        content = "12) [user] Do the thing.\n"
        e = uf.parse_entries(content)[0]
        assert e.prefix == "12) "
        assert e.flagged is True
        assert e.text == "Do the thing."

    def test_numbered_dated_unflagged_fact(self):
        content = "7. [due:2026-07-28] Renew the domain.\n"
        e = uf.parse_entries(content)[0]
        assert e.prefix == "7. "
        assert e.flagged is False
        assert e.due == "2026-07-28"

    def test_plain_numbered_item_no_tag_no_comment_not_returned(self):
        content = "1. Just a plain numbered note, no grammar at all.\n"
        assert uf.parse_entries(content) == []

    def test_numbered_tagless_bullet_with_comment_is_returned(self):
        # Mirrors the fulfilled-exit anti-resurrection trace, but for a
        # numbered item retiring to a plain numbered fact (spec §5.3).
        content = (
            "3. Send the drafts.\n"
            "  <!--mem id:x7f3a2 resolved:2026-07-24 created:2026-07-19-->\n"
        )
        entries = uf.parse_entries(content)
        assert len(entries) == 1
        e = entries[0]
        assert e.prefix == "3. "
        assert e.flagged is False
        assert e.text == "Send the drafts."
        assert e.id == "x7f3a2"
        assert e.resolved == "2026-07-24"


class TestSectionDetection:
    def test_entry_above_any_heading_has_none_section(self):
        content = "- [user] Do the thing before any heading.\n"
        e = uf.parse_entries(content)[0]
        assert e.section is None

    def test_entry_under_heading_captures_section_text(self):
        content = (
            "## Blockers\n"
            "- [user] Approve the vendor contract.\n"
        )
        e = uf.parse_entries(content)[0]
        assert e.section == "Blockers"

    def test_section_with_parenthetical_preserved(self):
        content = (
            "## Next Steps (priority)\n"
            "- [user] Ship the release notes.\n"
        )
        e = uf.parse_entries(content)[0]
        assert e.section == "Next Steps (priority)"

    def test_nearest_preceding_heading_wins_across_multiple_sections(self):
        content = (
            "## Focus\n"
            "- [user] First entry.\n"
            "## Blockers\n"
            "- [user] Second entry.\n"
        )
        entries = uf.parse_entries(content)
        by_text = {e.text: e for e in entries}
        assert by_text["First entry."].section == "Focus"
        assert by_text["Second entry."].section == "Blockers"

    def test_h1_and_h3_headings_do_not_set_section(self):
        content = (
            "# PROJECT_STATE\n"
            "### Sub-detail\n"
            "- [user] An entry under non-## headings.\n"
        )
        e = uf.parse_entries(content)[0]
        assert e.section is None

    def test_numbered_entry_also_gets_section(self):
        content = (
            "## Blockers\n"
            "3. [user] Approve the vendor contract.\n"
        )
        e = uf.parse_entries(content)[0]
        assert e.section == "Blockers"


class TestExistingBulletRoundTripUnchanged:
    """Every existing `- [user]` parsing behavior must be byte-identical
    after the numbered-item grammar extension."""

    def test_canonical_entry_fields_unchanged(self):
        entries = uf.parse_entries(CANONICAL_ENTRY)
        assert len(entries) == 1
        e = entries[0]
        assert e.flagged is True
        assert e.due == "2026-07-28"
        assert e.prefix == "- "
        assert e.text == (
            "Send 宝玉 + Simon DM drafts — only you can send from your accounts."
        )

    def test_real_prose_still_parses_to_zero_entries(self):
        assert uf.parse_entries(REAL_PROSE) == []

    def test_markdown_checkbox_bullets_still_not_tags(self):
        content = (
            "- [ ] Awaiting claude-code findings on root cause\n"
            "- [x] Recorded bug in BACKLOG.md as item 29\n"
        )
        assert uf.parse_entries(content) == []


class TestLint:
    def test_malformed_due_warns(self):
        content = "- [user due:tomorrow] Do the thing.\n"
        warnings = uf.lint(content)
        assert any("malformed due" in w for w in warnings)

    def test_malformed_due_entry_still_parses_with_due_none(self):
        content = "- [user due:tomorrow] Do the thing.\n"
        entries = uf.parse_entries(content)
        assert len(entries) == 1
        assert entries[0].flagged is True
        assert entries[0].due is None

    def test_well_formed_due_no_warning(self):
        content = "- [user due:2026-07-28] Do the thing.\n"
        warnings = uf.lint(content)
        assert not any("malformed due" in w for w in warnings)

    def test_clean_prose_has_no_warnings(self):
        assert uf.lint(REAL_PROSE) == []

    def test_flagged_entry_with_no_comment_at_all_warns_nothing(self):
        # The receipt-completeness check (missing evidence/confidence) was
        # retired with the receipt machinery (spec amendment, 2026-07-24) —
        # a flagged entry with no mem-comment at all must lint clean now.
        content = "- [user] Do the thing with no comment at all.\n"
        assert uf.lint(content) == []

    def test_lint_tolerates_legacy_evidence_from_confidence_attrs(self):
        # Old files that already adopted evidence/from/confidence must not
        # warn or break — those attributes are still parseable, just no
        # longer instructed or linted for.
        assert uf.lint(CANONICAL_ENTRY) == []
