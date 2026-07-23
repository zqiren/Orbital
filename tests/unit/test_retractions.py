# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The retractions store: the permanent "user said no" record (spec §3, §5.2).

``orbital/retractions.md`` is never trimmed — every retraction is appended
and stays forever, injected into every session as a hard constraint so an
agent can never re-propose, re-infer, or re-add something the user has
already declined.
"""
from __future__ import annotations

import pytest

from agent_os.agent.retractions import (
    Retraction,
    add_retraction,
    list_retractions,
    normalized_title_match,
    render_constraints,
)


# ---------------------------------------------------------------------------
# Store round-trip
# ---------------------------------------------------------------------------

class TestStoreRoundTrip:
    def test_append_then_list_round_trips(self, tmp_path):
        r = Retraction(
            id="x7f3a2",
            title="Send 宝玉 + Simon DM drafts",
            reason="changed my mind",
            date="2026-07-24",
        )
        add_retraction(tmp_path, r)

        got = list_retractions(tmp_path)

        assert got == [r]

    def test_round_trip_preserves_cjk_title(self, tmp_path):
        r = Retraction(
            id="a1b2c3",
            title="给宝玉和Simon发送私信草稿",
            reason="not needed",
            date="2026-07-24",
        )
        add_retraction(tmp_path, r)

        got = list_retractions(tmp_path)

        assert got[0].title == "给宝玉和Simon发送私信草稿"

    def test_round_trip_preserves_colon_in_reason(self, tmp_path):
        r = Retraction(
            id="deadbe",
            title="Schedule the kickoff",
            reason="conflict: I already booked it myself at 3:00pm",
            date="2026-07-24",
        )
        add_retraction(tmp_path, r)

        got = list_retractions(tmp_path)

        assert got[0].reason == "conflict: I already booked it myself at 3:00pm"

    def test_multiple_appends_accumulate_in_order(self, tmp_path):
        r1 = Retraction(id="111111", title="First thing", reason="no", date="2026-07-20")
        r2 = Retraction(id="222222", title="Second thing", reason="no thanks", date="2026-07-21")

        add_retraction(tmp_path, r1)
        add_retraction(tmp_path, r2)

        got = list_retractions(tmp_path)

        assert [r.id for r in got] == ["111111", "222222"]

    def test_list_retractions_empty_when_no_file(self, tmp_path):
        assert list_retractions(tmp_path) == []

    def test_file_format_matches_spec_example(self, tmp_path):
        r = Retraction(
            id="x7f3a2",
            title="Send 宝玉 + Simon DM drafts",
            reason="changed my mind",
            date="2026-07-24",
        )
        add_retraction(tmp_path, r)

        content = (tmp_path / "retractions.md").read_text(encoding="utf-8")

        assert (
            '- [x7f3a2] "Send 宝玉 + Simon DM drafts" — retracted by user '
            "2026-07-24: changed my mind" in content
        )

    def test_never_trimmed_append_only(self, tmp_path):
        """Nothing in this module ever removes or rewrites a prior line."""
        for i in range(5):
            add_retraction(
                tmp_path,
                Retraction(id=f"{i:06x}", title=f"Thing {i}", reason="no", date="2026-07-20"),
            )
        before = (tmp_path / "retractions.md").read_text(encoding="utf-8")

        # A later read-only call must not mutate the file.
        list_retractions(tmp_path)

        after = (tmp_path / "retractions.md").read_text(encoding="utf-8")
        assert before == after
        assert len(list_retractions(tmp_path)) == 5


# ---------------------------------------------------------------------------
# render_constraints
# ---------------------------------------------------------------------------

class TestRenderConstraints:
    def test_empty_list_renders_empty_string(self):
        assert render_constraints([]) == ""

    def test_nonempty_starts_with_heading(self):
        rs = [Retraction(id="x7f3a2", title="Send DM drafts", reason="changed my mind", date="2026-07-24")]

        block = render_constraints(rs)

        assert block.startswith("## Retracted by user — hard constraints")

    def test_nonempty_instructs_no_repropose_reinfer_readd(self):
        rs = [Retraction(id="x7f3a2", title="Send DM drafts", reason="changed my mind", date="2026-07-24")]

        block = render_constraints(rs)

        assert "re-propose" in block
        assert "re-infer" in block
        assert "re-add" in block

    def test_nonempty_instructs_explicit_user_request_only(self):
        rs = [Retraction(id="x7f3a2", title="Send DM drafts", reason="changed my mind", date="2026-07-24")]

        block = render_constraints(rs)

        assert "explicit user request" in block

    def test_nonempty_lists_each_retraction(self):
        rs = [
            Retraction(id="111111", title="First thing", reason="no", date="2026-07-20"),
            Retraction(id="222222", title="Second thing", reason="no thanks", date="2026-07-21"),
        ]

        block = render_constraints(rs)

        assert "First thing" in block
        assert "Second thing" in block


# ---------------------------------------------------------------------------
# normalized_title_match — matching tiers
# ---------------------------------------------------------------------------

class TestNormalizedTitleMatch:
    def test_exact_id_match(self):
        rs = [Retraction(id="x7f3a2", title="Send DM drafts to the client", reason="no", date="2026-07-24")]

        got = normalized_title_match("x7f3a2", rs)

        assert got is rs[0]

    def test_fuzzy_title_match_above_threshold(self):
        rs = [Retraction(id="x7f3a2", title="Send DM drafts to 宝玉 and Simon", reason="no", date="2026-07-24")]

        # Rephrased, punctuation/case/whitespace differ but same substance.
        got = normalized_title_match("send dm drafts to 宝玉 and simon!!", rs)

        assert got is rs[0]

    def test_below_threshold_returns_none(self):
        rs = [Retraction(id="x7f3a2", title="Send DM drafts to the client", reason="no", date="2026-07-24")]

        got = normalized_title_match("Completely unrelated sentence about lunch", rs)

        assert got is None

    def test_empty_list_returns_none(self):
        assert normalized_title_match("anything", []) is None

    def test_empty_title_returns_none(self):
        rs = [Retraction(id="x7f3a2", title="Send DM drafts", reason="no", date="2026-07-24")]
        assert normalized_title_match("", rs) is None


# ---------------------------------------------------------------------------
# Injection into session context (same seam as test_cold_resume.py)
# ---------------------------------------------------------------------------

class TestContextInjection:
    def _make_context_manager(self, tmp_path, workspace_files=None):
        from agent_os.agent.context import ContextManager
        from agent_os.agent.prompt_builder import PromptContext, Autonomy
        from agent_os.agent.session import Session

        class MockPromptBuilder:
            def build(self, context):
                return ("cached-system-prefix", "semi-stable-suffix", "dynamic-runtime")

        ctx = PromptContext(
            workspace=str(tmp_path),
            model="test-model",
            autonomy=Autonomy.HANDS_OFF,
            enabled_agents=[],
            tool_names=["read", "write", "shell"],
            os_type="linux",
            datetime_now="2026-01-01T00:00:00",
            context_usage_pct=0.0,
        )
        session = Session.new("retraction-ctx", str(tmp_path))
        return ContextManager(session, MockPromptBuilder(), ctx, workspace_files=workspace_files)

    def test_constraint_block_injected_when_retractions_exist(self, tmp_path):
        from agent_os.agent.workspace_files import WorkspaceFileManager

        wfm = WorkspaceFileManager(str(tmp_path))
        wfm.ensure_dir()
        add_retraction(
            tmp_path / "orbital",
            Retraction(id="x7f3a2", title="Send DM drafts", reason="changed my mind", date="2026-07-24"),
        )

        cm = self._make_context_manager(tmp_path, workspace_files=wfm)
        messages = cm.prepare()

        constraint_msgs = [
            m for m in messages
            if "Retracted by user" in m.get("content", "")
        ]
        assert len(constraint_msgs) == 1
        assert "Send DM drafts" in constraint_msgs[0]["content"]

    def test_no_constraint_block_when_no_retractions(self, tmp_path):
        from agent_os.agent.workspace_files import WorkspaceFileManager

        wfm = WorkspaceFileManager(str(tmp_path))
        wfm.ensure_dir()

        cm = self._make_context_manager(tmp_path, workspace_files=wfm)
        messages = cm.prepare()

        constraint_msgs = [
            m for m in messages
            if "Retracted by user" in m.get("content", "")
        ]
        assert len(constraint_msgs) == 0
