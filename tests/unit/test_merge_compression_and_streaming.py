# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests: the memory pass must actually COMPRESS.

Incident (orbital-marketing, 2026-07-27): a pass that SUCCEEDED returned every
file bigger than it started and archived nothing — the prompt stated the
target but never the file's CURRENT size, so the model always concluded it was
fine. (The second defect of that incident — a 15-minute non-streaming
whole-file merge dropping its connection — went away with the whole-file
merge: the memory editor's replies are short id lists.)

Invariants:
  - consolidation_target(key) sits a real margin BELOW the soft budget, so a
    pass that lands on target does not re-trip the flag on the next append.
  - The editor prompt carries MEASURED sizes and an explicit amount to cut.
  - The deterministic floor enforces the target when the editor declines.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.testutils import streamable

from agent_os.agent import memory_editor
from agent_os.agent import memory_entries as _mem
from agent_os.agent import workspace_files as wsf_module
from agent_os.agent.workspace_files import (
    WorkspaceFileManager,
    run_session_end_routine,
)


def _mock_session(session_id="sess_compress"):
    session = MagicMock()
    session.session_id = session_id
    session.session_uuid = session_id
    session.get_messages.return_value = []
    return session


def _declining_editor():
    """An editor model that looks and chooses nothing."""
    provider = streamable(AsyncMock())
    resp = MagicMock()
    resp.text = "{}"
    resp.raw_message = {"role": "assistant", "content": "{}"}
    resp.tool_calls = []
    provider.complete.return_value = resp
    return provider


@pytest.fixture(autouse=True)
def _reset_completion_set():
    wsf_module._completed_session_ends.clear()
    memory_editor._RUNNING.clear()
    yield
    wsf_module._completed_session_ends.clear()


def _over_soft_workspace(tmp_path) -> WorkspaceFileManager:
    """decisions and lessons both over their soft budgets, like the incident."""
    ws = WorkspaceFileManager(str(tmp_path))
    ws.ensure_dir()
    ws.write("decisions", "".join(
        f"## 2026-07-{d:02d}: Decision {d}\n**Chose:** {'x' * 1100}\n\n"
        for d in range(1, 27)
    ))
    ws.write("lessons", "".join(
        f"{i}. **Lesson {i}.** {'y' * 1150}\n" for i in range(1, 19)
    ))
    return ws


def _prompt(ws) -> str:
    return memory_editor.build_prompt(
        {k: ws.read(k) or "" for k in memory_editor.LIVE_KEYS},
        today="2026-09-19", since=None, sessions=[],
    )


# ---------------------------------------------------------------------------
# 1. Consolidation target carries headroom below the soft budget
# ---------------------------------------------------------------------------

def test_target_sits_a_full_headroom_below_soft_for_durable_files():
    """Landing exactly on the soft budget means the very next appended entry
    re-trips the flag. The pass has to buy room."""
    for key in ("decisions", "lessons"):
        soft = _mem.FILE_BUDGETS[key]["soft"]
        target = _mem.consolidation_target(key)
        assert soft - target >= _mem.CONSOLIDATION_HEADROOM_TOKENS


def test_target_never_demands_an_absurd_cut_from_a_small_file():
    """A flat 1000-token headroom would gut PROJECT_STATE (soft 1800) and
    INDEX (soft 1500), so the target is floored proportionally."""
    for key in ("state", "index"):
        soft = _mem.FILE_BUDGETS[key]["soft"]
        target = _mem.consolidation_target(key)
        assert 0 < target < soft
        assert target >= soft * 0.5


def test_target_is_below_soft_for_every_layer1_file():
    for key in ("state", "decisions", "lessons", "index"):
        assert _mem.consolidation_target(key) < _mem.FILE_BUDGETS[key]["soft"]


# ---------------------------------------------------------------------------
# 2. The editor prompt states MEASURED sizes and an explicit amount to cut
# ---------------------------------------------------------------------------

def test_prompt_states_measured_size_and_quota_for_over_budget_files(tmp_path):
    ws = _over_soft_workspace(tmp_path)
    prompt = _prompt(ws)

    dec_now = _mem.est_tokens(_mem._budget_text(ws.read("decisions"), "decisions"))
    dec_target = _mem.consolidation_target("decisions")

    assert "exact counts" in prompt
    assert f"{int(dec_now)} tokens" in prompt
    assert f"target {dec_target}" in prompt
    assert f"cut ~{int(dec_now) - dec_target}" in prompt


def test_prompt_does_not_ask_the_model_to_estimate_its_own_size(tmp_path):
    """The '(~4 chars/token)' self-estimate is what made archiving never fire."""
    assert "4 chars/token" not in _prompt(_over_soft_workspace(tmp_path))


def test_prompt_marks_within_target_files_as_needing_no_action(tmp_path):
    ws = WorkspaceFileManager(str(tmp_path))
    ws.ensure_dir()
    ws.write("decisions", "## 2026-07-01: Tiny\n**Chose:** A\n")
    assert "within target" in _prompt(ws).lower()


# ---------------------------------------------------------------------------
# 4. Deterministic floor enforces the same target when the LLM declines
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_backstop_demotes_to_target_when_the_merge_declines(tmp_path):
    """An editor that archives nothing (the incident's exact behaviour) must
    NOT leave it over budget — the deterministic pass is the floor."""
    ws = _over_soft_workspace(tmp_path)
    before = _mem.est_tokens(_mem._budget_text(ws.read("decisions"), "decisions"))
    assert before > _mem.FILE_BUDGETS["decisions"]["soft"]  # guard

    # The editor runs but chooses nothing.
    await run_session_end_routine(
        _mock_session("s_floor"), _declining_editor(), ws, session_uuid="s_floor"
    )

    after = _mem.est_tokens(_mem._budget_text(ws.read("decisions"), "decisions"))
    assert after <= _mem.consolidation_target("decisions")


@pytest.mark.asyncio
async def test_demoted_entries_are_moved_to_archive_not_deleted(tmp_path):
    ws = _over_soft_workspace(tmp_path)
    await run_session_end_routine(
        _mock_session("s_arch"), _declining_editor(), ws, session_uuid="s_arch"
    )

    archive = ws.read("decisions_archive") or ""
    assert archive.strip(), "over-budget entries must land in the archive"
    # Each demoted entry left a pointer carrying its id, and kept it.
    import re
    for eid in re.findall(r"\[archived \S+ id:(\S+)\]", ws.read("decisions")):
        assert f"id:{eid}" in archive
    # And INDEX gains a pointer so the archive is discoverable.
    assert "DECISIONS_ARCHIVE.md" in (ws.read("index") or "")


def test_backstop_leaves_a_file_already_under_target_alone(tmp_path):
    ws = WorkspaceFileManager(str(tmp_path))
    ws.ensure_dir()
    ws.write("decisions", "## 2026-07-01: Small\n**Chose:** A\n")
    before = ws.read("decisions")          # as persisted (write adds the format header)
    wsf_module._apply_hard_caps(ws)
    assert ws.read("decisions") == before
    assert not (ws.read("decisions_archive") or "").strip()
