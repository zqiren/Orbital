# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests: the "State checkpoint:" dynamic status line renders scheduler
state — in-flight passes and the last pass's outcome — so the agent can
reason about the ASYNC consolidation instead of guessing (orbital-marketing
incident, 2026-07-09).
"""

from agent_os.agent.prompt_builder import Autonomy, PromptBuilder, PromptContext


def _ctx(**overrides) -> PromptContext:
    base = dict(
        workspace="/tmp/ws",
        model="test-model",
        autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[],
        tool_names=["read", "write"],
        os_type="macos",
        datetime_now="2026-07-09T12:00:00",
    )
    base.update(overrides)
    return PromptContext(**base)


def test_status_line_no_checkpoint_yet():
    line = PromptBuilder()._state_checkpoint_status(_ctx())
    assert "no consolidation yet" in line


def test_status_line_in_flight_overrides_last_update():
    line = PromptBuilder()._state_checkpoint_status(_ctx(
        last_state_update_turn=9,
        last_state_update_ts="2026-07-09T11:00:00+00:00",
        turns_since_last_update=5,
        refresh_in_flight=True,
        refresh_in_flight_since_turn=14,
    ))
    assert "in flight" in line
    assert "turn 14" in line
    # Must warn against the two incident failure modes:
    assert "re-trigger" in line.lower()
    assert "persist" in line.lower()


def test_status_line_reports_backstop_only_outcome():
    line = PromptBuilder()._state_checkpoint_status(_ctx(
        last_state_update_turn=14,
        last_state_update_ts="2026-07-09T11:21:01+00:00",
        turns_since_last_update=2,
        last_state_update_outcome="backstop_only",
    ))
    assert "turn 14" in line
    assert "backstop" in line.lower()
    assert "edit the file directly" in line


def test_status_line_reports_failed_outcome():
    line = PromptBuilder()._state_checkpoint_status(_ctx(
        last_state_update_turn=14,
        last_state_update_ts="2026-07-09T11:21:01+00:00",
        turns_since_last_update=2,
        last_state_update_outcome="failed",
    ))
    assert "edit the file directly" in line


def test_status_line_llm_merged_stays_plain():
    """A successful merge needs no caveats — keep the metadata line as-is."""
    line = PromptBuilder()._state_checkpoint_status(_ctx(
        last_state_update_turn=14,
        last_state_update_ts="2026-07-09T11:21:01+00:00",
        turns_since_last_update=2,
        last_state_update_outcome="llm_merged",
    ))
    assert "turn 14" in line
    assert "backstop" not in line.lower()
    assert "edit the file directly" not in line
