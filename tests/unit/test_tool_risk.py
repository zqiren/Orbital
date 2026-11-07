# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 161 §5.5: Orbital's own dispatch tool is approved by name under
check-in; Claude Code's native ``Agent`` (the DELEGATE class) keeps prompting."""

from agent_os.agent.prompt_builder import Autonomy
from agent_os.agent.transports.tool_risk import should_auto_approve

DISPATCH = "mcp__orbital__dispatch"


def test_dispatch_is_auto_approved_under_check_in():
    assert should_auto_approve(DISPATCH, Autonomy.CHECK_IN) is True


def test_dispatch_is_auto_approved_under_hands_off():
    assert should_auto_approve(DISPATCH, Autonomy.HANDS_OFF) is True


def test_supervised_still_asks_for_dispatch():
    assert should_auto_approve(DISPATCH, Autonomy.SUPERVISED) is False


def test_native_agent_tool_still_prompts_under_check_in():
    assert should_auto_approve("Agent", Autonomy.CHECK_IN) is False


def test_other_mcp_tools_still_prompt_under_check_in():
    assert should_auto_approve("mcp__orbital__other", Autonomy.CHECK_IN) is False
    assert should_auto_approve("mcp__someone__dispatch", Autonomy.CHECK_IN) is False
