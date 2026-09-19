# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Write-chokepoint reconciliation (spec 089 §4.1).

Every PROJECT_STATE bullet carries a daemon-managed ``<!--mem-->`` comment
with ``id`` / ``created`` / ``touched``. Agents never see those comments (they
are stripped from the injected view), so every state write is diffed against
the previous on-disk content and the comments are re-attached — but ONLY on an
exact (whitespace-normalised) text match. A reworded line is a new line: new
id, today's dates. That errs toward keeping it (younger lines are protected
from the size floor).

The fuzzy title matcher, the ``resolved`` lock, the retraction drop and the
omission lint are gone (089 v2 retired them); these tests pin their absence.
"""

import re

from agent_os.agent import state_blocks, user_flags
from agent_os.agent.flag_chokepoint import reconcile_flags


TODAY = "2026-07-23"


def _blocks(content):
    return state_blocks.parse(content)[1]


def _block(content, idx=0):
    return _blocks(content)[idx]


# ---------------------------------------------------------------------------
# every bullet is stamped
# ---------------------------------------------------------------------------

class TestEveryBulletStamped:
    def test_plain_bullet_gets_id_created_touched_today(self):
        merged, _ = reconcile_flags(None, "# State\n\n- Shipping the installer.\n", TODAY)
        b = _block(merged)
        assert re.fullmatch(r"[0-9a-f]{6}", b.id)
        assert b.fields["created"] == TODAY
        assert b.fields["touched"] == TODAY

    def test_comment_sits_directly_under_the_bullet(self):
        merged, _ = reconcile_flags(None, "- Shipping the installer.\n", TODAY)
        lines = merged.split("\n")
        assert lines[0] == "- Shipping the installer."
        assert lines[1].startswith("  <!--mem id:")

    def test_numbered_item_keeps_its_marker(self):
        merged, _ = reconcile_flags(None, "## Next\n3. Verify the DMG.\n", TODAY)
        assert "3. Verify the DMG." in merged
        assert _block(merged).prefix == "3. "
        assert _block(merged).id

    def test_flagged_and_dated_bullets_are_stamped_too(self):
        new = "- [user] Approve the budget.\n- [due:2026-08-01] Renew the domain.\n"
        merged, _ = reconcile_flags(None, new, TODAY)
        ids = [b.id for b in _blocks(merged)]
        assert len(ids) == 2 and all(ids) and ids[0] != ids[1]
        # the user_flags grammar still reads them
        entries = user_flags.parse_entries(merged)
        assert entries[0].flagged and entries[1].due == "2026-08-01"

    def test_continuation_and_nested_lines_are_kept_with_their_bullet(self):
        new = (
            "- GOAI booth prep:\n"
            "  flyers ordered, posters printed\n"
            "  - sub: sticker count 300\n"
            "- Next thing.\n"
        )
        merged, _ = reconcile_flags(None, new, TODAY)
        # nothing lost, and the nested bullet is part of the parent block
        for text in ("flyers ordered, posters printed", "  - sub: sticker count 300"):
            assert text in merged
        blocks = _blocks(merged)
        assert len(blocks) == 2
        assert "sticker count" in blocks[0].raw

    def test_each_block_gets_exactly_one_comment(self):
        merged, _ = reconcile_flags(None, "- a\n- b\n- c\n", TODAY)
        assert merged.count("<!--mem") == 3

    def test_pointer_lines_and_prose_are_not_stamped(self):
        new = (
            "Current focus: the release.\n"
            "[archived 2026-07-01 id:abc123] old launch plan → PROJECT_STATE_ARCHIVE.md\n"
        )
        merged, _ = reconcile_flags(None, new, TODAY)
        assert merged == new


# ---------------------------------------------------------------------------
# exact-text carry-forward
# ---------------------------------------------------------------------------

PREV = (
    "# State\n\n"
    "- Send drafts to the client.\n"
    "  <!--mem id:abc123 from:sess_1 evidence:\"send the drafts\" "
    "created:2026-07-19 touched:2026-07-20-->\n"
    "- Ship the release.\n"
    "  <!--mem id:ship01 created:2026-07-01 touched:2026-07-01-->\n"
)


class TestExactTextCarryForward:
    def test_comment_stripped_rewrite_keeps_id_and_dates(self):
        new = user_flags.strip_mem_comments(PREV)
        merged, _ = reconcile_flags(PREV, new, TODAY)
        assert merged == PREV

    def test_legacy_fields_are_carried_verbatim(self):
        new = "# State\n\n- Send drafts to the client.\n"
        merged, _ = reconcile_flags(PREV, new, TODAY)
        b = _block(merged)
        assert b.id == "abc123"
        assert b.fields["from"] == "sess_1"
        assert b.fields["evidence"] == "send the drafts"
        assert b.fields["created"] == "2026-07-19"
        assert b.fields["touched"] == "2026-07-20"

    def test_whitespace_only_difference_keeps_id(self):
        new = "# State\n\n-   Send   drafts to the   client. \n"
        merged, _ = reconcile_flags(PREV, new, TODAY)
        assert _block(merged).id == "abc123"

    def test_reworded_line_gets_new_id_and_today(self):
        new = "# State\n\n- Send the drafts to the client today.\n"
        merged, _ = reconcile_flags(PREV, new, TODAY)
        b = _block(merged)
        assert b.id not in ("abc123", "ship01")
        assert b.fields["created"] == TODAY and b.fields["touched"] == TODAY
        assert "from" not in b.fields and "evidence" not in b.fields

    def test_reworded_line_with_its_old_comment_still_attached_gets_new_id(self):
        # The edit tool rewrites the bullet line on disk and leaves the
        # comment line under it: a new sentence is still a new line.
        new = PREV.replace("- Send drafts to the client.", "- Drafts are sent.")
        merged, _ = reconcile_flags(PREV, new, TODAY)
        b = _block(merged)
        assert b.id != "abc123"
        assert b.fields["created"] == TODAY

    def test_tag_toggle_is_not_a_rewording(self):
        prev = "- [user] Approve the budget.\n  <!--mem id:bud001 created:2026-07-01 touched:2026-07-01-->\n"
        merged, _ = reconcile_flags(prev, "- Approve the budget.\n", TODAY)
        assert _block(merged).id == "bud001"
        assert _block(merged).fields["created"] == "2026-07-01"

    def test_renumbered_item_keeps_id(self):
        prev = "1. Verify the DMG.\n  <!--mem id:dmg001 created:2026-07-01 touched:2026-07-01-->\n"
        merged, _ = reconcile_flags(prev, "1. New first step.\n2. Verify the DMG.\n", TODAY)
        assert _block(merged, 1).id == "dmg001"
        assert _block(merged, 0).id != "dmg001"

    def test_duplicated_line_second_copy_gets_new_id(self):
        new = "# State\n\n- Ship the release.\n- Ship the release.\n"
        merged, _ = reconcile_flags(PREV, new, TODAY)
        a, b = _blocks(merged)
        assert a.id == "ship01"
        assert b.id != "ship01" and b.fields["created"] == TODAY

    def test_new_comment_fields_win_over_prev_for_the_same_line(self):
        # A daemon writer (e.g. the Workbench) adds a field to a kept line.
        new = PREV.replace(
            "created:2026-07-01 touched:2026-07-01-->",
            "created:2026-07-01 touched:2026-07-01 resolved:2026-07-22-->",
        )
        merged, _ = reconcile_flags(PREV, new, TODAY)
        b = [x for x in _blocks(merged) if x.id == "ship01"][0]
        assert b.fields["resolved"] == "2026-07-22"

    def test_unknown_id_in_new_content_is_kept(self):
        # Restoring from a backup / the editor re-inserting a stamped block.
        new = "- Restored line.\n  <!--mem id:fff000 created:2026-06-01 touched:2026-06-02-->\n"
        merged, _ = reconcile_flags(PREV, new, TODAY)
        b = _block(merged)
        assert b.id == "fff000"
        assert b.fields["created"] == "2026-06-01"

    def test_duplicate_ids_in_new_content_are_split(self):
        new = (
            "- One.\n  <!--mem id:fff000 created:2026-06-01 touched:2026-06-01-->\n"
            "- Two.\n  <!--mem id:fff000 created:2026-06-01 touched:2026-06-01-->\n"
        )
        merged, _ = reconcile_flags(None, new, TODAY)
        a, b = _blocks(merged)
        assert a.id == "fff000" and b.id != "fff000"

    def test_comment_moved_below_continuation_is_canonicalised(self):
        new = (
            "- Long line\n"
            "  wrapped here\n"
            "  <!--mem id:abc999 created:2026-07-01 touched:2026-07-01-->\n"
        )
        merged, _ = reconcile_flags(None, new, TODAY)
        assert merged == (
            "- Long line\n"
            "  <!--mem id:abc999 created:2026-07-01 touched:2026-07-01-->\n"
            "  wrapped here\n"
        )

    def test_wrapped_legacy_comment_is_read(self):
        prev = (
            "- [user due:2026-07-28] Send DM drafts.\n"
            "  <!--mem id:x7f3a2 from:orbital-marketing_7c045c40\n"
            "      evidence:\"draft 写好就准备发\" confidence:unconfirmed\n"
            "      created:2026-07-19 touched:2026-07-23-->\n"
        )
        merged, _ = reconcile_flags(prev, prev, TODAY)
        b = _block(merged)
        assert b.id == "x7f3a2"
        assert b.fields["created"] == "2026-07-19"
        assert merged.count("<!--mem") == 1


# ---------------------------------------------------------------------------
# retired machinery stays retired
# ---------------------------------------------------------------------------

class TestRetiredMachinery:
    def test_no_resolved_lock(self):
        prev = (
            "- Pick option A, B or C?\n"
            "  <!--mem id:4148d8 created:2026-07-20 touched:2026-07-20 resolved:2026-07-22-->\n"
        )
        merged, warns = reconcile_flags(prev, "- [user] Pick option A, B or C?\n", TODAY)
        assert "- [user] Pick option A, B or C?" in merged
        assert not any("resolved" in w.lower() for w in warns)

    def test_retraction_titles_are_ignored(self):
        new = "- Launch on Product Hunt.\n"
        merged, warns = reconcile_flags(None, new, TODAY, ["Launch on Product Hunt"])
        assert "- Launch on Product Hunt." in merged
        assert warns == []

    def test_no_omission_lint(self):
        new = "## Blockers\n- You must sign the release form.\n- 用户需要确认\n"
        _merged, warns = reconcile_flags(None, new, TODAY)
        assert warns == []

    def test_malformed_due_still_warns(self):
        _merged, warns = reconcile_flags(None, "- [due:tomorrow] Renew.\n", TODAY)
        assert any("malformed due" in w for w in warns)


# ---------------------------------------------------------------------------
# round-trip / idempotency
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_file_without_bullets_is_byte_identical(self):
        plain = (
            "# State\n\n"
            "Current focus: shipping the installer.\n"
            "Blockers: none.\n\n\n"
            "Next: verify the DMG on a clean machine.\n"
        )
        merged, warns = reconcile_flags(None, plain, TODAY)
        assert merged == plain
        assert warns == []

    def test_reconcile_is_idempotent(self):
        merged1, _ = reconcile_flags(None, "- a\n- b\n\n## X\n1. c\n", TODAY)
        merged2, _ = reconcile_flags(merged1, merged1, TODAY)
        assert merged2 == merged1

    def test_stripped_view_round_trips(self):
        merged1, _ = reconcile_flags(None, "# S\n\n- a\n  wrapped\n- b\n", TODAY)
        merged2, _ = reconcile_flags(merged1, user_flags.strip_mem_comments(merged1), "2026-09-01")
        assert merged2 == merged1

    def test_unterminated_comment_does_not_swallow_the_file(self):
        new = "- a <!--mem id:abc\n## Heading\n- b\n"
        merged, _ = reconcile_flags(None, new, TODAY)
        assert "## Heading" in merged and "- b" in merged
        assert len(_blocks(merged)) == 2

    def test_empty_content_passes_through(self):
        assert reconcile_flags("- a\n", "", TODAY) == ("", [])


# ---------------------------------------------------------------------------
# wiring: both write paths run reconcile for PROJECT_STATE.md
# ---------------------------------------------------------------------------

class TestWiring:
    def test_process_on_write_reconciles_state(self, tmp_path):
        from agent_os.agent import memory_entries
        orbital = tmp_path / "orbital"
        orbital.mkdir()
        state_path = orbital / "PROJECT_STATE.md"
        state_path.write_text(
            "- Send drafts to the client.\n"
            "  <!--mem id:abc123 created:2026-07-19 touched:2026-07-19-->\n",
            encoding="utf-8",
        )
        out, _warns = memory_entries.process_on_write(
            str(tmp_path), str(state_path), "- Send drafts to the client.\n", today=TODAY
        )
        assert _block(out).id == "abc123"

    def test_process_on_write_index_unchanged_behavior(self, tmp_path):
        from agent_os.agent import memory_entries
        orbital = tmp_path / "orbital"
        orbital.mkdir()
        target = orbital / "INDEX.md"
        out, _warns = memory_entries.process_on_write(
            str(tmp_path), str(target), "# INDEX\n- a.py — thing\n"
        )
        assert out.startswith(memory_entries.FORMAT_HEADERS["index"])
        assert "<!--mem" not in out

    def test_manager_write_reconciles_state(self, tmp_path):
        from agent_os.agent.workspace_files import WorkspaceFileManager
        wf = WorkspaceFileManager(str(tmp_path))
        wf.write(
            "state",
            "- Ship the release.\n"
            "  <!--mem id:ship01 created:2026-07-01 touched:2026-07-01-->\n",
        )
        assert _block(wf.read("state")).id == "ship01"
        wf.write("state", "# State\n\n- Ship the release.\n")
        assert _block(wf.read("state")).id == "ship01"

    def test_manager_write_prose_state_round_trips(self, tmp_path):
        from agent_os.agent import memory_entries as mem
        from agent_os.agent.workspace_files import WorkspaceFileManager
        wf = WorkspaceFileManager(str(tmp_path))
        content = "# State\nDoing well.\n"
        wf.write("state", content)
        assert wf.read("state") == mem.FORMAT_HEADERS["state"] + "\n" + content
