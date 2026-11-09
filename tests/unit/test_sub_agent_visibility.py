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


# ---------------------------------------------------------------------------
# 2026-11-09: one readiness test for "who can this project's work go to"
# ---------------------------------------------------------------------------

from agent_os.daemon_v2.sub_agent_visibility import worker_ready  # noqa: E402


class _Full:
    def __init__(self, slug, *, installed=True, dependencies_met=True,
                 credentials_configured=True, update_needed=False, handshake=None):
        self.slug = slug
        self.installed = installed
        self.dependencies_met = dependencies_met
        self.credentials_configured = credentials_configured
        self.update_needed = update_needed
        self.handshake = handshake


def test_worker_ready_is_the_dispatch_gate_test():
    assert worker_ready(_Full("claude-code"))
    assert not worker_ready(_Full("built-in"))
    assert not worker_ready(_Full("codebuddy", credentials_configured=False))
    assert not worker_ready(_Full("gemini-cli", installed=False))
    assert not worker_ready(_Full("kimi", dependencies_met=False))
    assert not worker_ready(_Full("codex", update_needed=True))
    assert not worker_ready(_Full("dsh", handshake={"state": "failed", "detail": "x"}))
    assert worker_ready(_Full("dsh", handshake={"state": "unknown"}))


def _ready_engine():
    return _Engine([_Full("claude-code"), _Full("codex"),
                    _Full("codebuddy", credentials_configured=False),
                    _Full("gemini-cli", credentials_configured=False),
                    _Full("pi"), _Full("built-in")])


def test_ready_only_drops_unready_workers_from_autodetect():
    slugs = resolve_visible_sub_agent_slugs(
        enabled_sub_agents=None, disabled_sub_agents=["pi"],
        setup_engine=_ready_engine(), ready_only=True)
    assert slugs == ["claude-code", "codex"]


def test_ready_only_filters_an_explicit_allowlist_too():
    slugs = resolve_visible_sub_agent_slugs(
        enabled_sub_agents=["codex", "codebuddy"], disabled_sub_agents=[],
        setup_engine=_ready_engine(), ready_only=True)
    assert slugs == ["codex"]


def test_ready_only_without_an_engine_keeps_the_list():
    slugs = resolve_visible_sub_agent_slugs(
        enabled_sub_agents=["codex", "codebuddy"], disabled_sub_agents=[],
        setup_engine=None, ready_only=True)
    assert slugs == ["codex", "codebuddy"]


def test_default_still_lists_installed_but_unready():
    # The orchestrator-prompt path keeps its old behaviour (not ready_only).
    slugs = resolve_visible_sub_agent_slugs(
        enabled_sub_agents=None, disabled_sub_agents=[], setup_engine=_ready_engine())
    assert "codebuddy" in slugs and "gemini-cli" in slugs
