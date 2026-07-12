# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The management-agent system prompt must teach request_network_access
routing whenever the tool is registered, and omit it otherwise."""

from agent_os.agent.prompt_builder import (
    _NETWORK_ROUTING_GUIDANCE,
    Autonomy,
    PromptBuilder,
    PromptContext,
)


def _ctx(tool_names: list) -> PromptContext:
    return PromptContext(
        workspace="/ws", model="test-model", autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[], tool_names=tool_names, os_type="linux",
        datetime_now="2026-07-12T00:00:00",
    )


def test_guidance_present_when_tool_registered():
    section = PromptBuilder()._network_access_section(_ctx(["request_network_access"]))
    assert section == _NETWORK_ROUTING_GUIDANCE
    assert "request_network_access" in section
    assert "browser" in section


def test_guidance_omitted_when_tool_not_registered():
    section = PromptBuilder()._network_access_section(_ctx(["read", "write", "shell"]))
    assert section is None


def test_section_included_in_built_semi_stable_prompt():
    context = _ctx(["shell", "request_network_access"])
    _, semi_stable, _ = PromptBuilder().build(context)
    assert "request_network_access" in semi_stable
