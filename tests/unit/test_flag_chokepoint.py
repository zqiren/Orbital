# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Write-chokepoint reconciliation (spec §5, §5.2, §5.5, §8).

The chokepoint keeps machine identity (ids, created/from/evidence) and user
lifecycle decisions (resolved, confidence:stated, retractions) stable while
agents freely rewrite PROJECT_STATE.md — agents never see the ``<!--mem-->``
comments, so every write is diffed against the previous on-disk content.
"""
import os
import re

import pytest

from agent_os.agent import user_flags
from agent_os.agent.flag_chokepoint import reconcile_flags


TODAY = "2026-07-23"


def _entry(content, idx=0):
    entries = user_flags.parse_entries(content)
    return entries[idx]


# ---------------------------------------------------------------------------
# id-preserving merge
# ---------------------------------------------------------------------------

class TestIdPreservation:
    def test_exact_text_rewrite_without_comments_keeps_id(self):
        prev = (
            "# State\n\n"
            "- [user] Send drafts to the client.\n"
            "  <!--mem id:abc123 from:sess_1 evidence:\"send the drafts\" "
            "confidence:unconfirmed created:2026-07-19 touched:2026-07-19-->\n"
        )
        # Agent rewrite: same sentence, comment stripped (never seen by agent).
        new = "# State\n\n- [user] Send drafts to the client.\n"
        merged, warns = reconcile_flags(prev, new, TODAY)
        e = _entry(merged)
        assert e.id == "abc123"
        assert e.created == "2026-07-19"
        # text unchanged → touched preserved, not bumped to today
        assert e.touched == "2026-07-19"
        assert e.evidence == "send the drafts"
        assert e.from_session == "sess_1"
        assert e.confidence == "unconfirmed"

    def test_fuzzy_reassociation_of_rephrased_bullet(self):
        prev = (
            "- [user] Send the DM drafts to 宝玉 and Simon.\n"
            "  <!--mem id:x7f3a2 from:s1 evidence:\"发 draft\" "
            "created:2026-07-19 touched:2026-07-19-->\n"
        )
        # Rephrased (normalized ratio >= 0.75) and comment-less.
        new = "- [user] Send DM drafts to 宝玉 and Simon.\n"
        merged, warns = reconcile_flags(prev, new, TODAY)
        e = _entry(merged)
        assert e.id == "x7f3a2"                 # re-associated by fuzzy title
        assert e.created == "2026-07-19"        # created preserved
        assert e.touched == TODAY               # body changed → touched bumped

    def test_unrelated_new_bullet_does_not_steal_id(self):
        prev = (
            "- [user] Book the venue for the offsite.\n"
            "  <!--mem id:aaa000 from:s1 evidence:\"book it\" "
            "created:2026-07-10 touched:2026-07-10-->\n"
        )
        new = "- [user] Buy a birthday cake for the team.\n"
        merged, warns = reconcile_flags(prev, new, TODAY)
        e = _entry(merged)
        assert e.id != "aaa000"                 # < 0.75 ratio → fresh id
        assert re.fullmatch(r"[0-9a-f]{6}", e.id)

    def test_numbered_item_id_survives_comment_stripped_rewrite(self):
        # A `3. [user] ...` item's id must survive the agent's comment-less
        # rewrite exactly like a `- [user] ...` bullet does — the chokepoint
        # doesn't special-case the marker, it just diffs by (id else fuzzy
        # title) match, so this exercises the numbered grammar end-to-end.
        prev = (
            "## Blockers\n\n"
            "3. [user] Approve the vendor contract.\n"
            "  <!--mem id:num900 from:sess_1 evidence:\"approve it\" "
            "created:2026-07-19 touched:2026-07-19-->\n"
        )
        new = "## Blockers\n\n3. [user] Approve the vendor contract.\n"
        merged, warns = reconcile_flags(prev, new, TODAY)
        e = _entry(merged)
        assert e.id == "num900"
        assert e.created == "2026-07-19"
        assert e.prefix == "3. "
        assert "3. [user] Approve the vendor contract." in merged


class TestNewEntries:
    def test_new_flagged_bullet_gets_id_and_created_today(self):
        new = "# State\n\n- [user] Review the vendor contract.\n"
        merged, warns = reconcile_flags(None, new, TODAY)
        e = _entry(merged)
        assert re.fullmatch(r"[0-9a-f]{6}", e.id)
        assert e.created == TODAY
        assert e.touched == TODAY

    def test_new_flagged_bullet_with_explicit_id_is_preserved(self):
        # A writer that already stamped an id (e.g. a system write) keeps it.
        new = (
            "- [user] Ship the release.\n"
            "  <!--mem id:keep99 from:s1 evidence:\"ship it\" "
            "created:2026-07-01 touched:2026-07-01-->\n"
        )
        merged, warns = reconcile_flags(None, new, TODAY)
        e = _entry(merged)
        assert e.id == "keep99"
        assert e.created == "2026-07-01"

    def test_dated_fact_passes_through_unstamped(self):
        new = "- [due:2026-07-28] Quarterly report auto-generates.\n"
        merged, warns = reconcile_flags(None, new, TODAY)
        e = _entry(merged)
        assert e.id is None                     # facts stay unstamped
        assert e.due == "2026-07-28"
        assert not e.flagged
        assert "<!--mem" not in merged

    def test_new_numbered_flagged_item_gets_id_and_keeps_its_marker(self):
        # Tag-in-place grammar: a `3. [user] ...` item goes through the SAME
        # id-stamping merge as a `- [user] ...` bullet — no source change to
        # the chokepoint itself was needed, since it operates on whatever
        # line text/marker parse_entries handed it.
        new = "## Blockers\n\n3. [user] Approve the numbered blocker.\n"
        merged, warns = reconcile_flags(None, new, TODAY)
        e = _entry(merged)
        assert re.fullmatch(r"[0-9a-f]{6}", e.id)
        assert e.created == TODAY
        assert e.touched == TODAY
        assert e.prefix == "3. "
        assert "3. [user] Approve the numbered blocker." in merged


class TestTouchedStamp:
    def test_text_change_stamps_touched_today(self):
        prev = (
            "- [user] Approve Q3 budget.\n"
            "  <!--mem id:def456 from:s1 evidence:\"approve\" "
            "created:2026-07-01 touched:2026-07-01-->\n"
        )
        new = "- [user] Approve the Q3 budget now.\n"
        merged, warns = reconcile_flags(prev, new, TODAY)
        assert _entry(merged).touched == TODAY

    def test_no_text_change_keeps_touched(self):
        prev = (
            "- [user] Approve Q3 budget.\n"
            "  <!--mem id:def456 from:s1 evidence:\"approve\" "
            "created:2026-07-01 touched:2026-07-01-->\n"
        )
        new = "- [user] Approve Q3 budget.\n"
        merged, warns = reconcile_flags(prev, new, TODAY)
        assert _entry(merged).touched == "2026-07-01"


# ---------------------------------------------------------------------------
# lifecycle-fields-win (spec §5.5)
# ---------------------------------------------------------------------------

class TestLifecycleFieldsWin:
    def test_resolved_survives_agent_rewrite(self):
        prev = (
            "- [user] Confirm the flight booking.\n"
            "  <!--mem id:res111 from:s1 evidence:\"confirm\" "
            "created:2026-07-10 touched:2026-07-20 resolved:2026-07-20-->\n"
        )
        # Agent rewrites the bullet and drops the resolved stamp.
        new = "- [user] Confirm the flight booking.\n"
        merged, warns = reconcile_flags(prev, new, TODAY)
        assert _entry(merged).resolved == "2026-07-20"

    def test_best_ratio_bullet_wins_id_when_two_clear_threshold(self):
        # Two new bullets both clear 0.75 against ONE previous entry. The
        # continuation (ratio 1.0) must win the id + lifecycle fields even
        # though a weaker decoy (ratio ~0.86) appears FIRST in the file.
        prev = (
            "- [user] Send the DM drafts to 宝玉 and Simon today.\n"
            "  <!--mem id:old111 from:s1 evidence:\"发 draft\" "
            "confidence:stated created:2026-07-10 touched:2026-07-10 "
            "resolved:2026-07-20-->\n"
        )
        new = (
            "- [user] Send the DM drafts to 宝玉 today.\n"          # decoy, first
            "- [user] Send the DM drafts to 宝玉 and Simon today.\n"  # verbatim, second
        )
        merged, warns = reconcile_flags(prev, new, TODAY)
        entries = user_flags.parse_entries(merged)
        by_text = {e.text: e for e in entries}
        verbatim = by_text["Send the DM drafts to 宝玉 and Simon today."]
        decoy = by_text["Send the DM drafts to 宝玉 today."]
        assert verbatim.id == "old111"
        assert verbatim.resolved == "2026-07-20"
        assert verbatim.confidence == "stated"
        # The decoy is a genuinely new bullet — fresh id, no inherited lifecycle.
        assert decoy.id != "old111"
        assert re.fullmatch(r"[0-9a-f]{6}", decoy.id)
        assert decoy.resolved is None

    def test_confidence_stated_wins_over_reverted_unconfirmed(self):
        prev = (
            "- [user] Cancel the old subscription.\n"
            "  <!--mem id:con222 from:s1 evidence:\"cancel it\" "
            "confidence:stated created:2026-07-10 touched:2026-07-10-->\n"
        )
        # Agent's stale copy re-asserts unconfirmed; user's 'stated' must win.
        new = (
            "- [user] Cancel the old subscription.\n"
            "  <!--mem confidence:unconfirmed-->\n"
        )
        merged, warns = reconcile_flags(prev, new, TODAY)
        assert _entry(merged).confidence == "stated"


# ---------------------------------------------------------------------------
# resolved-trace re-attachment (F1) — a fulfilled entry's tag-less trace
# (id + resolved in a comment) must survive an agent rewrite that strips the
# comment and re-emits the sentence as a plain bullet.
# ---------------------------------------------------------------------------

class TestResolvedTraceReattach:
    def _trace(self, eid, resolved="2026-07-20"):
        return (
            f"- Send the DM drafts to 宝玉 and Simon.\n"
            f"  <!--mem id:{eid} from:s1 evidence:\"发 draft\" "
            f"created:2026-07-19 touched:2026-07-20 resolved:{resolved}-->\n"
        )

    def test_verbatim_plain_bullet_reattaches_resolved_trace(self):
        # (1) prev = tag-less bullet + comment (id+resolved); new = the SAME
        # sentence as a plain bullet with no comment (the comment-stripped
        # agent view, re-emitted). The comment must re-attach, resolved intact.
        prev = self._trace("trc001")
        new = "- Send the DM drafts to 宝玉 and Simon.\n"
        merged, warns = reconcile_flags(prev, new, TODAY)
        e = _entry(merged)
        assert e.id == "trc001"
        assert e.resolved == "2026-07-20"
        assert e.created == "2026-07-19"
        assert e.flagged is False               # stays a retired, tag-less trace

    def test_rephrased_plain_bullet_reattaches_resolved_trace(self):
        # (2) rephrased-but->=0.75 plain bullet re-associates the same way.
        prev = self._trace("trc002")
        new = "- Send DM drafts to 宝玉 and Simon.\n"     # dropped "the", >=0.75
        merged, warns = reconcile_flags(prev, new, TODAY)
        e = _entry(merged)
        assert e.id == "trc002"
        assert e.resolved == "2026-07-20"

    def test_deleted_sentence_lets_trace_die_without_warning(self):
        # (3) the sentence is absent from new (agent deleted it) → the trace
        # legitimately dies; no id/resolved lingers and nothing warns loudly.
        prev = self._trace("trc003")
        new = "- Book the venue for the offsite.\n"      # unrelated, < 0.75
        merged, warns = reconcile_flags(prev, new, TODAY)
        assert "trc003" not in merged
        assert "resolved" not in merged
        assert user_flags.parse_entries(merged) == []    # the plain bullet is untouched
        assert not any("trc003" in w for w in warns)

    def test_flagged_bullet_wins_trace_over_plain_duplicate(self):
        # (4) both a flagged bullet AND a plain bullet match the same prev
        # trace. Flagged matching takes precedence (existing semantics): the
        # flagged entry inherits the id + resolved; the plain duplicate does
        # NOT also steal it (one-to-one), so it stays a plain, untracked bullet.
        prev = self._trace("trc004")
        new = (
            "- [user] Send the DM drafts to 宝玉 and Simon.\n"   # flagged
            "- Send the DM drafts to 宝玉 and Simon.\n"           # plain duplicate
        )
        merged, warns = reconcile_flags(prev, new, TODAY)
        entries = user_flags.parse_entries(merged)
        assert [e.id for e in entries if e.id == "trc004"] == ["trc004"]  # exactly one
        flagged = [e for e in entries if e.flagged]
        assert len(flagged) == 1
        assert flagged[0].id == "trc004"
        assert flagged[0].resolved == "2026-07-20"

    def test_round_trip_fulfilled_exit_survives_agent_rewrite(self):
        # (5) end-to-end: a fulfilled exit's on-disk output → the agent's
        # comment-stripped view → an identical re-emit → reconcile. The entry
        # is still resolved, still carries its original id.
        fulfilled_exit = (
            "# State\n\n"
            "## Done\n"
            "- Send the DM drafts to 宝玉 and Simon.\n"
            "  <!--mem id:trc005 from:s1 evidence:\"发 draft\" "
            "created:2026-07-19 touched:2026-07-20 resolved:2026-07-20-->\n"
        )
        agent_view = user_flags.strip_mem_comments(fulfilled_exit)
        assert "<!--mem" not in agent_view          # the agent never sees the comment
        merged, warns = reconcile_flags(fulfilled_exit, agent_view, TODAY)
        e = _entry(merged)
        assert e.id == "trc005"
        assert e.resolved == "2026-07-20"
        assert e.flagged is False

    def test_new_flagged_bullet_matching_resolved_trace_inherits_lifecycle(self):
        # m17 baseline (pins CURRENT behavior for the backlogged m17 decision):
        # a NEW [user]-tagged bullet whose sentence matches a resolved trace
        # inherits the trace's id + resolved via lifecycle-wins. If m17 later
        # changes this, THIS assertion is the deliberate line to revisit.
        prev = self._trace("trc017")
        new = "- [user] Send the DM drafts to 宝玉 and Simon.\n"
        merged, warns = reconcile_flags(prev, new, TODAY)
        e = _entry(merged)
        assert e.id == "trc017"
        assert e.resolved == "2026-07-20"
        assert e.flagged is True

    def test_verbatim_numbered_item_reattaches_resolved_trace(self):
        # Same as test_verbatim_plain_bullet_reattaches_resolved_trace, but
        # the retired entry was found as a numbered item ("3. ..."), not a
        # dash bullet. A later freeform agent rewrite drops the mem-comment
        # and re-emits the same numbered line — the id + resolved stamp must
        # still re-attach (review finding 1).
        prev = (
            "3. Approve the vendor contract.\n"
            "  <!--mem id:trc010 from:s1 evidence:\"approve it\" "
            "created:2026-07-19 touched:2026-07-20 resolved:2026-07-20-->\n"
        )
        new = "3. Approve the vendor contract.\n"
        merged, warns = reconcile_flags(prev, new, TODAY)
        e = _entry(merged)
        assert e.id == "trc010"
        assert e.resolved == "2026-07-20"
        assert e.created == "2026-07-19"
        assert e.flagged is False
        assert e.prefix == "3. "

    def test_rephrased_paren_numbered_item_reattaches_resolved_trace(self):
        # A "12) ..." marker (the other numbered-list style) rephrased just
        # enough to stay >= 0.75 ratio must also re-associate.
        prev = (
            "12) Send the DM drafts to 宝玉 and Simon.\n"
            "  <!--mem id:trc011 from:s1 evidence:\"发 draft\" "
            "created:2026-07-19 touched:2026-07-20 resolved:2026-07-20-->\n"
        )
        new = "12) Send DM drafts to 宝玉 and Simon.\n"   # dropped "the", >=0.75
        merged, warns = reconcile_flags(prev, new, TODAY)
        e = _entry(merged)
        assert e.id == "trc011"
        assert e.resolved == "2026-07-20"
        assert e.prefix == "12) "


# ---------------------------------------------------------------------------
# retraction resurrection guard (spec §5.2)
# ---------------------------------------------------------------------------

class TestRetractionGuard:
    def test_retracted_title_match_keeps_entry_out_and_warns_loudly(self):
        new = (
            "# State\n\n"
            "- [user] Send 宝玉 + Simon the DM drafts.\n"
            "- [user] Draft the Q3 report.\n"
        )
        retractions = ["Send 宝玉 and Simon the DM drafts"]
        merged, warns = reconcile_flags(None, new, TODAY, retraction_titles=retractions)
        titles = [e.text for e in user_flags.parse_entries(merged)]
        assert not any("宝玉" in t for t in titles)       # retracted one dropped
        assert any("Q3 report" in t for t in titles)     # the other survives
        assert any("RETRACT" in w.upper() for w in warns)  # loud warning

    def test_no_retraction_titles_keeps_everything(self):
        new = "- [user] Send 宝玉 + Simon the DM drafts.\n"
        merged, warns = reconcile_flags(None, new, TODAY, retraction_titles=None)
        assert len(user_flags.parse_entries(merged)) == 1


# ---------------------------------------------------------------------------
# omission heuristic lint (spec §8) — warns, never blocks
# ---------------------------------------------------------------------------

class TestOmissionHeuristic:
    def test_unflagged_bullet_under_blocker_heading_warns(self):
        new = (
            "# State\n\n"
            "## Blockers\n"
            "- Waiting on the client to approve the vendor.\n"
        )
        merged, warns = reconcile_flags(None, new, TODAY)
        assert any("user-facing" in w.lower() or "flag" in w.lower() for w in warns)
        # Never blocks: the line is still present in the output.
        assert "Waiting on the client" in merged

    def test_you_must_phrasing_warns(self):
        new = "## Notes\n- You must sign the release form before Friday.\n"
        merged, warns = reconcile_flags(None, new, TODAY)
        assert any("user-facing" in w.lower() or "flag" in w.lower() for w in warns)

    def test_cjk_user_phrasing_warns(self):
        new = "## 进度\n- 用户需要确认预算。\n"
        merged, warns = reconcile_flags(None, new, TODAY)
        assert any("user-facing" in w.lower() or "flag" in w.lower() for w in warns)

    def test_flagged_bullet_under_blocker_heading_does_not_warn(self):
        new = (
            "## Blockers\n"
            "- [user] Approve the vendor contract.\n"
            "  <!--mem id:zzz999 from:s1 evidence:\"approve\" "
            "created:2026-07-01 touched:2026-07-01-->\n"
        )
        merged, warns = reconcile_flags(new, new, TODAY)
        assert not any("user-facing" in w.lower() for w in warns)

    def test_plain_bullet_without_triggers_does_not_warn(self):
        new = "## Progress\n- Refactored the exporter module.\n"
        merged, warns = reconcile_flags(None, new, TODAY)
        assert warns == []

    def test_unflagged_numbered_item_under_blocker_heading_warns(self):
        # Same heuristic as test_unflagged_bullet_under_blocker_heading_warns,
        # but the unflagged line is a numbered item ("1. ..."), not a dash
        # bullet (review finding 2).
        new = (
            "# State\n\n"
            "## Blockers\n"
            "1. Waiting on the client to approve the vendor.\n"
        )
        merged, warns = reconcile_flags(None, new, TODAY)
        assert any("user-facing" in w.lower() or "flag" in w.lower() for w in warns)
        # Never blocks: the line is still present in the output.
        assert "Waiting on the client" in merged

    def test_you_must_phrasing_numbered_item_warns(self):
        new = "## Notes\n2) You must sign the release form before Friday.\n"
        merged, warns = reconcile_flags(None, new, TODAY)
        assert any("user-facing" in w.lower() or "flag" in w.lower() for w in warns)


# ---------------------------------------------------------------------------
# round-trip / idempotency / non-adopting files
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_plain_state_file_is_byte_identical(self):
        plain = (
            "# State\n\n"
            "Current focus: shipping the installer.\n"
            "Blockers: none.\n\n\n"          # 3 blank lines must NOT collapse
            "Next: verify the DMG on a clean machine.\n"
        )
        merged, warns = reconcile_flags(None, plain, TODAY)
        assert merged == plain
        assert warns == []

    def test_reconcile_is_idempotent(self):
        prev = (
            "- [user] Send drafts to the client.\n"
            "  <!--mem id:abc123 from:sess_1 evidence:\"send the drafts\" "
            "confidence:unconfirmed created:2026-07-19 touched:2026-07-19-->\n"
        )
        new = "- [user] Send drafts to the client.\n"
        merged1, _ = reconcile_flags(prev, new, TODAY)
        merged2, _ = reconcile_flags(merged1, merged1, TODAY)
        assert merged2 == merged1

    def test_no_prev_no_flags_returns_new(self):
        new = "just some freeform prose\nwith no bullets at all\n"
        merged, warns = reconcile_flags(None, new, TODAY)
        assert merged == new
        assert warns == []


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
            "- [user] Send drafts to the client.\n"
            "  <!--mem id:abc123 from:s1 evidence:\"send it\" "
            "created:2026-07-19 touched:2026-07-19-->\n",
            encoding="utf-8",
        )
        new = "- [user] Send drafts to the client.\n"
        out, warns = memory_entries.process_on_write(
            str(tmp_path), str(state_path), new, today=TODAY
        )
        assert _entry(out).id == "abc123"

    def test_process_on_write_index_unchanged_behavior(self, tmp_path):
        # Non-state volatile file keeps header-only behavior (no reconcile).
        from agent_os.agent import memory_entries
        orbital = tmp_path / "orbital"
        orbital.mkdir()
        target = orbital / "INDEX.md"
        out, warns = memory_entries.process_on_write(
            str(tmp_path), str(target), "# INDEX\n- a.py — thing\n"
        )
        assert out.startswith(memory_entries.FORMAT_HEADERS["index"])
        assert "<!--mem" not in out

    def test_manager_write_reconciles_state(self, tmp_path):
        from agent_os.agent.workspace_files import WorkspaceFileManager
        wf = WorkspaceFileManager(str(tmp_path))
        wf.write(
            "state",
            "- [user] Ship the release.\n"
            "  <!--mem id:ship01 from:s1 evidence:\"ship it\" "
            "created:2026-07-01 touched:2026-07-01-->\n",
        )
        assert _entry(wf.read("state")).id == "ship01"
        # Agent rewrite with the comment stripped: id must survive.
        wf.write("state", "# State\n\n- [user] Ship the release.\n")
        assert _entry(wf.read("state")).id == "ship01"

    def test_manager_write_plain_state_byte_identical(self, tmp_path):
        # A non-adopting state file still round-trips to header + content.
        from agent_os.agent import memory_entries as mem
        from agent_os.agent.workspace_files import WorkspaceFileManager
        wf = WorkspaceFileManager(str(tmp_path))
        content = "# State\nDoing well.\n"
        wf.write("state", content)
        assert wf.read("state") == mem.FORMAT_HEADERS["state"] + "\n" + content
