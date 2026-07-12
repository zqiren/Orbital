# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""_apply_network_rules pushes DEFAULT + approved (wildcarded) to the provider."""

from unittest.mock import MagicMock

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager
from agent_os.platform.types import DEFAULT_ALLOWLIST_DOMAINS


@pytest.fixture
def agent_manager_fixture():
    """AgentManager with a mocked project store + platform provider, following
    test_build_worker_deps.py's ``_make_manager`` convention (a MagicMock for
    every collaborator AgentManager doesn't touch in this test)."""
    project_store = MagicMock()
    platform_provider = MagicMock()
    mgr = AgentManager(
        project_store=project_store,
        ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(),
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
        platform_provider=platform_provider,
    )
    return mgr


def test_apply_network_rules_pushes_grants(agent_manager_fixture):
    mgr = agent_manager_fixture              # provider + store mocked per file convention
    mgr._project_store.get_project = MagicMock(
        return_value={"project_id": "p1", "approved_domains": ["x.com"]}
    )
    mgr._apply_network_rules("p1")
    rules = mgr._platform_provider.configure_network.call_args[0][1]
    assert rules.mode == "allowlist"
    assert "x.com" in rules.domains and "*.x.com" in rules.domains
    assert set(DEFAULT_ALLOWLIST_DOMAINS) <= set(rules.domains)


def test_apply_network_rules_no_provider_is_noop(agent_manager_fixture):
    mgr = agent_manager_fixture
    mgr._platform_provider = None
    mgr._apply_network_rules("p1")           # must not raise
