# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""RED-before-GREEN gates for the Layer-1 memory redesign (TASK-mem-1).

Covers the 10 deliverable behaviors: injection cap + oldest-3, metadata survival,
demotion-to-archive, LESSONS not word-trimmed, soft-cap placement, no-delta gate,
contradiction dedup (fake provider), SESSION_LOG removal, registry-derived budget,
roster injection. No live LLM; the session-end pass uses a fake provider.
"""
from __future__ import annotations

import json
import os

import pytest

from agent_os.agent import memory_entries as M
from agent_os.agent import workspace_files as WF
from agent_os.agent.workspace_files import WorkspaceFileManager


def _wf(tmp_path) -> WorkspaceFileManager:
    wf = WorkspaceFileManager(str(tmp_path))
    wf.ensure_dir()
    return wf


def _decisions(n: int, *, touched=None, body="x" * 120) -> str:
    """n stamped DECISIONS entries, oldest-first, with explicit touched dates
    (the one shape of spec 124: bullet, comment under it, indented detail)."""
    out = []
    for i in range(1, n + 1):
        t = touched(i) if touched else f"2026-01-{i:02d}"
        out.append(
            f"- 2026-01-{i:02d}: Decision {i} — chose option {i}\n"
            f"  <!--mem id:d{i} created:2026-01-{i:02d} touched:{t}-->\n"
            f"  **Reason:** {body}\n  **Rejected:** alt {i}\n\n"
        )
    return "".join(out)


# ---------------------------------------------------------------------------
# Gate 1 — injection: every bullet line, detail for the hottest, fold by section
# ---------------------------------------------------------------------------

def test_injection_keeps_every_bullet_and_only_hot_detail(monkeypatch):
    monkeypatch.setitem(M.FILE_BUDGETS, "decisions", {"soft": 200, "hard": 400})
    content = _decisions(20, body="x" * 20)
    view = M.inject_view(content, "decisions", 400)
    for i in range(1, 21):
        assert f"Decision {i} " in view
    assert "<!--mem" not in view
    # detail only for the most recently touched entries, within budget
    assert "**Reason:**" in view
    assert view.count("**Reason:**") < 20
    assert "**Rejected:** alt 20" in view and "**Rejected:** alt 1\n" not in view
    assert M.est_tokens(view) <= 400


def test_injection_folds_by_section_when_bullets_overflow(monkeypatch):
    monkeypatch.setitem(M.FILE_BUDGETS, "decisions", {"soft": 60, "hard": 100})
    content = "## Pricing\n" + _decisions(10, body="x") + "## Channels\n" + _decisions(10, body="x").replace("id:d", "id:c")
    view = M.inject_view(content, "decisions", 100)
    assert "## Pricing" in view and "## Channels" in view
    assert view.count("more entries in this section not shown") == 2
    assert M.est_tokens(view) <= 100 + 40


def test_injection_unchanged_when_fits(monkeypatch):
    monkeypatch.setitem(M.FILE_BUDGETS, "decisions", {"soft": 7000, "hard": 9000})
    content = "# D\n- a decision\n  detail\n- another\n"
    view = M.inject_view(content, "decisions", 9000)
    assert view is content  # same object — prefix-cache friendly


# ---------------------------------------------------------------------------
# Gate 2 — metadata survival through a strip-on-overwrite
# ---------------------------------------------------------------------------

def test_metadata_survives_overwrite_with_headers_stripped():
    old = (
        "- 2026-06-15: Keep API base url\n"
        "  <!--mem id:keep-api created:2026-06-15 touched:2026-06-15-->\n"
        "  **Chose:** moonshot\n\n"
    )
    # agent overwrites WITHOUT the metadata comment, body unchanged
    new = "- 2026-06-15: Keep API base url\n  **Chose:** moonshot\n\n"
    stamped, warns = M.stamp(new, old, "decisions", today="2026-06-18")
    assert "id:keep-api" in stamped          # id preserved
    assert "created:2026-06-15" in stamped    # created preserved
    assert "touched:2026-06-15" in stamped    # body unchanged → touched preserved
    # body changed → touched advances
    new2 = "- 2026-06-15: Keep API base url\n  **Chose:** minimax\n\n"
    stamped2, _ = M.stamp(new2, old, "decisions", today="2026-06-18")
    assert "touched:2026-06-18" in stamped2
    assert "created:2026-06-15" in stamped2


def test_legacy_numbered_lessons_become_bullets_not_renumbered():
    broken = "1. **A.** a.\n\n4. **B.** b.\n\n7. **C.** c.\n"
    text, _ = M.normalise_entries(broken, "lessons")
    stamped, _ = M.stamp(text, None, "lessons", today="2026-06-18")
    bullets = [ln for ln in stamped.splitlines() if ln.startswith("- ")]
    assert bullets == ["- A. a.", "- B. b.", "- C. c."]
    assert not any(ln[:1].isdigit() for ln in stamped.splitlines())


# ---------------------------------------------------------------------------
# Gate 3 — the floor never demotes (spec 124); pinned is never moved by anyone
# ---------------------------------------------------------------------------

def test_hard_cap_never_demotes(tmp_path, monkeypatch):
    monkeypatch.setitem(M.FILE_BUDGETS, "decisions", {"soft": 120, "hard": 180})
    wf = _wf(tmp_path)
    def touched(i):
        return "2025-01-01" if i in (4, 5, 6, 7) else "2026-06-18"
    wf.write("decisions", _decisions(10, touched=touched, body="x" * 20))
    before = wf.read("decisions")
    assert WF._apply_hard_caps(wf) == ["decisions"]
    assert wf.read("decisions") == before
    assert not wf.exists("decisions_archive")


def test_the_task_message_forbids_touching_pinned_entries():
    from agent_os.agent import memory_editor
    text = memory_editor.build_task_message(
        sizes={k: (1, 10, 5) for k in memory_editor.LIVE_KEYS}, folded=[], over=["decisions"],
        forced=False, pointer_paths=[], last_pass=None, sessions=[], today="2026-06-18", tool_cap=12,
    )
    assert "Never touch entries tagged tag:pinned." in text
    assert "never because it is\n  old" in text


# ---------------------------------------------------------------------------
# Gate 4 — LESSONS never word-trimmed
# ---------------------------------------------------------------------------

def test_lessons_long_playbook_survives_persist(tmp_path):
    long_body = " ".join(["word"] * 243)
    lesson = f"- Slate.js playbook. {long_body}\n"
    stamped, warns = M.stamp(lesson, None, "lessons", today="2026-06-18")
    assert long_body in stamped  # byte-intact, not shortened
    assert not warns


# ---------------------------------------------------------------------------
# Gate 5 — soft-cap flag (logic + only fires over threshold)
# ---------------------------------------------------------------------------

def test_soft_flag_only_over_threshold(monkeypatch):
    monkeypatch.setitem(M.FILE_BUDGETS, "decisions", {"soft": 50, "hard": 80})
    small = _decisions(1, body="x" * 4)
    big = _decisions(20)
    assert M.soft_flag(small, "decisions") is None
    flag = M.soft_flag(big, "decisions")
    assert flag and "memory pass" in flag and "automatically" in flag and "tok" in flag and "entries" in flag


# ---------------------------------------------------------------------------
# Gate 6 — no-delta gate (no write since last cleanup → no-op marker logic)
# ---------------------------------------------------------------------------

def test_no_delta_gate(tmp_path):
    wf = _wf(tmp_path)
    wf.write("decisions", _decisions(2))
    # fresh project, no marker → delta present
    assert WF._has_cleanup_delta(wf) is True
    WF._write_cleanup_marker(wf)
    # nothing changed → no delta
    assert WF._has_cleanup_delta(wf) is False
    # a write flips it back to delta
    wf.write("decisions", _decisions(3))
    assert WF._has_cleanup_delta(wf) is True


# ---------------------------------------------------------------------------
# Gate 7 — contradiction dedup via the session-end pass (fake provider)
# ---------------------------------------------------------------------------

class _FakeTurn:
    """A Memory turn runner (spec 124 §3.7) that applies ``moves(wf)`` the
    way the agent's edits would, and counts its calls."""

    def __init__(self, moves=None):
        self._moves = moves
        self.calls = 0

    async def __call__(self, task_text):
        from agent_os.agent import memory_editor
        self.calls += 1
        if self._moves is not None:
            self._moves()
        return memory_editor.TurnResult(mode="disk", llm_calls=1, tool_calls=2, ended="text_complete")


class _FakeSession:
    def __init__(self):
        self.session_id = "s1"

    def get_messages(self):
        return [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_session_end_merges_supersedes(tmp_path):
    """Spec 124 §3.7: superseding is a MERGE the Memory session writes — the
    contradiction pair leaves the live file (both originals archived
    verbatim under a [merged …] section) and one condensed entry takes its
    place."""
    from agent_os.agent import state_blocks
    wf = _wf(tmp_path)
    # the real contradiction pair from the reports
    wf.write("decisions",
        "- 2026-06-15: QClaw PM Identity for Outreach\n  <!--mem id:qclaw created:2026-06-15 touched:2026-06-15-->\n"
        "  **Chose:** Use QClaw PM identity.\n\n"
        "- 2026-06-15: Use Orbital Name (Not QClaw) for Outreach\n  <!--mem id:orbital-name created:2026-06-15 touched:2026-06-15-->\n"
        "  **Chose:** Orbital indie builder.\n\n")
    merged = (
        "- 2026-06-15: Use Orbital Name (Not QClaw) for Outreach\n"
        "  **Chose:** Orbital indie builder.\n  **Rejected:** QClaw PM identity (superseded).\n"
    )

    def moves():
        # the merged entry in the live file first, then the originals under
        # a [merged …] section (what the task message asks for)
        from agent_os.agent import state_blocks
        live = wf.read("decisions")
        blocks = {b.id: b for b in state_blocks.parse(live)[1] if b.id}
        new_live = live.replace(blocks["qclaw"].raw + "\n\n", "").replace(
            blocks["orbital-name"].raw + "\n", merged)
        wf.write("decisions", new_live)
        # the daemon stamps the merged entry: a title it already knows keeps
        # its id (orbital-name here), a new title gets a fresh one
        new_id = state_blocks.parse(wf.read("decisions"))[1][0].id
        WF._append_archive_section(
            wf, "decisions_archive", f"## [merged 2026-06-18 into id:{new_id}]",
            blocks["qclaw"].raw + "\n" + blocks["orbital-name"].raw)

    prov = _FakeTurn(moves)
    await WF.run_session_end_routine(
        _FakeSession(), None, wf, utility_provider=None,
        session_uuid="u1", bypass_idempotency=True, project_id="p1", force=True, run_turn=prov,
    )
    out = wf.read("decisions")
    assert "Use Orbital Name" in out
    assert "QClaw PM Identity for Outreach" not in out   # superseded entry left
    assert "id:qclaw" not in out and "[archived" not in out   # nothing left behind
    assert len(state_blocks.parse(out)[1]) == 1
    archive = wf.read("decisions_archive")
    assert "- 2026-06-15: QClaw PM Identity for Outreach\n  <!--mem id:qclaw" in archive
    assert "## [merged " in archive and "into id:" in archive
    assert prov.calls == 1


@pytest.mark.asyncio
async def test_session_end_no_delta_skips_llm(tmp_path):
    wf = _wf(tmp_path)
    wf.write("decisions", _decisions(2))
    WF._write_cleanup_marker(wf)  # mark as already cleaned
    prov = _FakeTurn()
    await WF.run_session_end_routine(
        _FakeSession(), None, wf, session_uuid="u2",
        bypass_idempotency=True, project_id="p1", run_turn=prov,
    )
    assert prov.calls == 0  # no-delta → no LLM call


@pytest.mark.asyncio
async def test_session_end_survives_llm_timeout_and_leaves_the_file_whole(tmp_path, monkeypatch):
    """LLM times out → no propagation; the floor runs but moves nothing."""
    from agent_os.agent import memory_editor
    monkeypatch.setitem(M.FILE_BUDGETS, "decisions", {"soft": 120, "hard": 180})
    monkeypatch.setattr(memory_editor, "EDITOR_TOTAL_TIMEOUT_S", 0.01)
    wf = _wf(tmp_path)
    wf.write("decisions", _decisions(12, body="x" * 20))
    before = wf.read("decisions")

    async def _hangs(task_text):
        import asyncio
        await asyncio.sleep(999)

    out = await WF.run_session_end_routine(
        _FakeSession(), None, wf, session_uuid="u3",
        bypass_idempotency=True, project_id="p1", run_turn=_hangs,
    )
    assert out == "backstop_only"
    # the floor never moves an entry: the file is whole, there is no archive
    assert wf.read("decisions") == before
    assert not wf.exists("decisions_archive")


# ---------------------------------------------------------------------------
# Gate 8 — SESSION_LOG removed
# ---------------------------------------------------------------------------

def test_session_log_removed_from_roster():
    assert "session_log" not in WF.FILE_NAMES
    assert "index" in WF.FILE_NAMES
    assert "decisions_archive" in WF.FILE_NAMES
    assert not hasattr(WF, "_RESUME_ORDER")  # cold resume removed 2026-10-20
    assert not hasattr(WF, "_truncate_session_log")
    assert not hasattr(WF, "_get_session_log_lock")


def test_cold_resume_builder_is_gone(tmp_path):
    assert not hasattr(_wf(tmp_path), "build_cold_resume_context")


def test_sub_agent_prompt_omits_session_log(tmp_path):
    from agent_os.agent.sub_agent_prompt import render_sub_agent_prompt
    out = render_sub_agent_prompt(str(tmp_path), None, "claude-code", ["claude-code"])
    assert "SESSION_LOG.md" not in out
    assert "INDEX.md" in out


# ---------------------------------------------------------------------------
# Gate 9 — registry-derived budget
# ---------------------------------------------------------------------------

def test_budget_derives_from_window():
    big = M.budgets_for_window(1_000_000)
    assert big["decisions"]["hard"] == 9000   # floor for a large window
    tiny = M.budgets_for_window(40_000)
    assert tiny["decisions"]["hard"] < 9000   # tiny window scales down
    missing = M.budgets_for_window(None)
    assert missing["decisions"]["hard"] == 9000  # missing → floor


def test_active_window_comes_from_registry():
    from agent_os.config.provider_registry import ProviderRegistry
    reg = ProviderRegistry()
    assert reg.get_context_window("minimax", "MiniMax-M3") == 1_000_000


# ---------------------------------------------------------------------------
# Gate 10 — roster injection (INDEX not CONTEXT; SESSION_LOG never)
# ---------------------------------------------------------------------------

def test_layer1_roster():
    from agent_os.agent.context import _LAYER1_FILES
    keys = [k for k, _ in _LAYER1_FILES]
    names = [n for _, n in _LAYER1_FILES]
    assert keys == ["decisions", "lessons", "index", "state"]
    assert "INDEX.md" in names and "CONTEXT.md" not in names
    assert "session_log" not in keys
