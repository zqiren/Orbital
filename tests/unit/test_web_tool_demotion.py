# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for web tool consolidation — search/fetch are now browser actions.

Validates:
- web_search and web_fetch tools are never registered (removed)
- Browser tool handles search/fetch natively
- Prompt includes correct web access instructions when browser is available
"""

from unittest.mock import MagicMock

import pytest

from agent_os.agent.prompt_builder import PromptBuilder, PromptContext, Autonomy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(tmp_path, **overrides) -> PromptContext:
    defaults = dict(
        workspace=str(tmp_path),
        model="test-model",
        autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[],
        tool_names=["read", "write", "shell"],
        os_type="linux",
        datetime_now="2026-02-25T00:00:00",
        project_name="TestProject",
        project_instructions="",
    )
    defaults.update(overrides)
    return PromptContext(**defaults)


def _make_config(**overrides):
    """Create a minimal AgentConfig-like object for _register_tools."""
    from agent_os.daemon_v2.models import AgentConfig
    defaults = dict(
        workspace="/tmp/test",
        model="test-model",
        api_key="sk-test",
        search_api_key=None,
    )
    defaults.update(overrides)
    return AgentConfig(**defaults)


# ---------------------------------------------------------------------------
# Tool Registration Tests
# ---------------------------------------------------------------------------

class TestWebToolsRemoved:
    """web_search and web_fetch tools are completely removed from the system."""

    def _register_and_get_names(self, config):
        """Run _register_tools and return the set of registered tool names."""
        from agent_os.agent.tools.registry import ToolRegistry
        from agent_os.daemon_v2.agent_manager import AgentManager

        registry = ToolRegistry()
        mgr = AgentManager.__new__(AgentManager)
        mgr._platform_provider = MagicMock()
        mgr._sub_agent_manager = MagicMock()
        mgr._user_credential_store = None
        mgr._browser_manager = None
        mgr._project_store = None
        mgr._trigger_manager = None
        mgr._register_tools(registry, config)
        return set(registry.tool_names())

    def test_web_tools_never_registered(self):
        config = _make_config(search_api_key=None)
        names = self._register_and_get_names(config)
        assert "web_search" not in names
        assert "web_fetch" not in names

    def test_web_tools_not_registered_even_with_key(self):
        """Even with a search API key, legacy web tools should not be registered."""
        config = _make_config(search_api_key="tvly-test-key")
        names = self._register_and_get_names(config)
        assert "web_search" not in names
        assert "web_fetch" not in names

    def test_core_tools_always_registered(self):
        config = _make_config(search_api_key=None)
        names = self._register_and_get_names(config)
        assert "read" in names
        assert "write" in names
        assert "edit" in names
        assert "shell" in names


# ---------------------------------------------------------------------------
# Prompt Instructions Tests
# ---------------------------------------------------------------------------

class TestWebAccessInstructions:
    """Prompt includes correct web access guidance based on available tools."""

    def test_browser_shows_search_fetch_actions(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, tool_names=["read", "write", "browser"])
        cached, _ = builder.build(ctx)
        assert "### Web Access" in cached
        assert 'action="search"' in cached
        assert 'action="fetch"' in cached

    def test_no_browser_no_web_section(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, tool_names=["read", "write", "shell"])
        cached, _ = builder.build(ctx)
        assert "### Web Access" not in cached
