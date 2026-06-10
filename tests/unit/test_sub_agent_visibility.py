# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for the shared sub-agent visibility resolver.

A project's ``disabled_sub_agents`` denylist must remove a sub-agent from
every place a management/peer agent is TOLD about it — including the
auto-detect (no explicit allowlist) branch, which previously ignored the
denylist on the /agents/start path.
"""

from __future__ import annotations

from agent_os.daemon_v2.sub_agent_visibility import resolve_visible_sub_agent_slugs


class _Status:
    def __init__(self, slug: str, installed: bool = True):
        self.slug = slug
        self.installed = installed


class _Engine:
    def __init__(self, statuses):
        self._statuses = statuses

    def check_all(self):
        return self._statuses


def test_autodetect_excludes_disabled():
    engine = _Engine([_Status("claude-code"), _Status("gemini-cli"), _Status("built-in")])
    slugs = resolve_visible_sub_agent_slugs(
        enabled_sub_agents=[],
        disabled_sub_agents=["gemini-cli"],
        setup_engine=engine,
    )
    assert "gemini-cli" not in slugs
    assert "claude-code" in slugs
    assert "built-in" not in slugs  # the orchestrator itself is never a peer


def test_autodetect_without_denylist_lists_installed():
    engine = _Engine([_Status("claude-code"), _Status("gemini-cli")])
    slugs = resolve_visible_sub_agent_slugs(
        enabled_sub_agents=[],
        disabled_sub_agents=[],
        setup_engine=engine,
    )
    assert set(slugs) == {"claude-code", "gemini-cli"}


def test_explicit_allowlist_still_filtered_by_denylist():
    slugs = resolve_visible_sub_agent_slugs(
        enabled_sub_agents=["claude-code", "gemini-cli"],
        disabled_sub_agents=["gemini-cli"],
        setup_engine=None,
    )
    assert slugs == ["claude-code"]


def test_uninstalled_excluded():
    engine = _Engine([_Status("claude-code", installed=True), _Status("gemini-cli", installed=False)])
    slugs = resolve_visible_sub_agent_slugs(
        enabled_sub_agents=[],
        disabled_sub_agents=[],
        setup_engine=engine,
    )
    assert slugs == ["claude-code"]


def test_legacy_enabled_agents_used_when_no_enabled_sub_agents():
    slugs = resolve_visible_sub_agent_slugs(
        enabled_sub_agents=[],
        disabled_sub_agents=["gemini-cli"],
        setup_engine=None,
        enabled_agents_legacy=["claude-code", "gemini-cli"],
    )
    assert slugs == ["claude-code"]


def test_no_engine_no_lists_returns_empty():
    assert resolve_visible_sub_agent_slugs(
        enabled_sub_agents=[],
        disabled_sub_agents=[],
        setup_engine=None,
    ) == []
