# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for the state-aware [MEMORY HYGIENE] soft flag.

Incident (orbital-marketing_33eed65d, 2026-07-09): the flag re-fired
identically every turn while a background consolidation pass was in flight,
and its text suggested "edit the file directly" — so the agent concluded the
async checkpoint_state tool had failed and hand-trimmed PROJECT_STATE.md
mid-pass. The flag is a state machine driven by scheduler state
(RefreshView), not a repeating alarm.

Spec 089: over-budget files are tidied AUTOMATICALLY by the memory editor, so
no state asks the agent to call checkpoint_state or to trim a file by hand:

  idle           → "tidied automatically in the background, no action"
  in-flight      → "the memory editor is tidying it — no action needed"
  backstop_only  → the editor could not run; it retries on its own
  edited / ...   → tidied; what remains is recent or live
  near hard cap  → escalation note appended in ANY state
"""

import pytest

from agent_os.agent import memory_entries as M
from agent_os.agent.memory_entries import RefreshView


BUDGET = {"soft": 100, "hard": 200}


@pytest.fixture(autouse=True)
def _small_state_budget(monkeypatch):
    """Small budgets so tests use short strings (est_tokens = len/4)."""
    monkeypatch.setitem(M.FILE_BUDGETS, "state", dict(BUDGET))


def _over_soft() -> str:
    # ~150 tokens: over soft (100), under the near-hard band (>=180).
    return "x" * 600


def _near_hard() -> str:
    # ~190 tokens: >= 90% of hard (180).
    return "x" * 760


# ---------------------------------------------------------------------------
# Baseline states
# ---------------------------------------------------------------------------

def test_under_soft_returns_none_in_every_state():
    small = "x" * 100  # 25 tokens
    assert M.soft_flag(small, "state") is None
    assert M.soft_flag(small, "state", refresh=RefreshView(in_flight=True)) is None
    assert M.soft_flag(
        small, "state", refresh=RefreshView(last_outcome="backstop_only")
    ) is None


def test_idle_flag_says_the_tidy_is_automatic():
    flag = M.soft_flag(_over_soft(), "state")
    assert flag is not None
    assert "automatically" in flag
    assert "no action needed" in flag.lower()
    assert "call the checkpoint_state tool" not in flag
    assert "edit the file directly" not in flag


# ---------------------------------------------------------------------------
# In-flight: the state that would have stopped the incident
# ---------------------------------------------------------------------------

def test_in_flight_flag_says_no_action_needed():
    flag = M.soft_flag(
        _over_soft(), "state",
        refresh=RefreshView(in_flight=True, in_flight_since_turn=14),
    )
    assert flag is not None
    assert "memory editor" in flag
    assert "turn 14" in flag
    assert "no action needed" in flag.lower()
    assert "do not trim it by hand" in flag.lower()
    assert "call the checkpoint_state tool" not in flag


def test_in_flight_without_turn_number_still_renders():
    flag = M.soft_flag(
        _over_soft(), "state", refresh=RefreshView(in_flight=True),
    )
    assert flag is not None
    assert "in the background" in flag


# ---------------------------------------------------------------------------
# Last-pass outcomes (not in flight)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("outcome", ["backstop_only", "failed"])
def test_failed_pass_says_it_retries_by_itself(outcome):
    flag = M.soft_flag(
        _over_soft(), "state",
        refresh=RefreshView(last_outcome=outcome, last_turn=14),
    )
    assert flag is not None
    assert "could not run" in flag
    assert "retries on its own" in flag
    assert "call the checkpoint_state tool" not in flag


@pytest.mark.parametrize("outcome", ["edited", "no_change", "llm_merged"])
def test_successful_pass_but_still_over_says_content_is_current(outcome):
    flag = M.soft_flag(
        _over_soft(), "state",
        refresh=RefreshView(last_outcome=outcome, last_turn=9),
    )
    assert flag is not None
    assert "turn 9" in flag
    assert "recent or still live" in flag
    assert "call the checkpoint_state tool" not in flag


def test_no_delta_outcome_falls_back_to_idle_text():
    flag = M.soft_flag(
        _over_soft(), "state",
        refresh=RefreshView(last_outcome="no_delta", last_turn=3),
    )
    assert flag is not None
    assert "automatically" in flag


# ---------------------------------------------------------------------------
# Near-hard-cap escalation tier (appended in any state)
# ---------------------------------------------------------------------------

def test_near_hard_cap_escalates_idle():
    flag = M.soft_flag(_near_hard(), "state")
    assert flag is not None
    assert "hard cap" in flag
    assert "pointer" in flag


def test_near_hard_cap_escalates_even_in_flight():
    flag = M.soft_flag(
        _near_hard(), "state", refresh=RefreshView(in_flight=True),
    )
    assert flag is not None
    assert "hard cap" in flag


def test_over_soft_but_below_band_has_no_escalation():
    flag = M.soft_flag(_over_soft(), "state")
    assert flag is not None
    assert "hard cap" not in flag
