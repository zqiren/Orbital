# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Issue-0 regression (2026-07-05): the inject-Case-3 auto-start path built
AgentConfig without is_scratch/agent_name, silently disabling the whole
scratch scope plane (prompt section, multi-root tools, portals) for every
chat session that auto-starts an agent. Both AgentConfig construction sites
(``_build_agent_config_from_project`` here, and the ``/agents/start`` route
in agents_v2.py) must agree on these fields.
"""

from unittest.mock import MagicMock

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager


def _manager_with(projects: dict) -> AgentManager:
    """Mirror the construction idiom in tests/unit/test_build_worker_deps.py
    (``_make_manager``): a real AgentManager with a MagicMock project store
    plus MagicMock settings/credential stores whose lookups return None so
    the fallback chains in ``_build_agent_config_from_project`` don't blow up
    on missing attributes.
    """
    ws = MagicMock()
    project_store = MagicMock()
    project_store.get_project = MagicMock(side_effect=lambda pid: projects.get(pid))
    project_store.list_projects = MagicMock(return_value=list(projects.values()))
    sub_agent_manager = MagicMock()
    activity_translator = MagicMock()
    process_manager = MagicMock()
    provider_registry = MagicMock()
    provider_registry.get_model_info.return_value = MagicMock(
        max_output=16384, capabilities=None, reasoning=None,
    )
    settings_store = MagicMock()
    settings_store.get = MagicMock(return_value=None)
    credential_store = MagicMock()
    credential_store.get_api_key = MagicMock(return_value=None)
    mgr = AgentManager(
        project_store=project_store,
        ws_manager=ws,
        sub_agent_manager=sub_agent_manager,
        activity_translator=activity_translator,
        process_manager=process_manager,
        provider_registry=provider_registry,
        settings_store=settings_store,
        credential_store=credential_store,
    )
    return mgr


SCRATCH = {"project_id": "p_s", "name": "Quick Tasks", "agent_name": "Assistant",
           "workspace": "/tmp/s", "is_scratch": True}
NORMAL = {"project_id": "p_n", "name": "Hn-daily", "workspace": "/tmp/n",
          "is_scratch": False}


def test_auto_start_config_carries_is_scratch_and_agent_name():
    mgr = _manager_with({"p_s": SCRATCH})
    cfg = mgr._build_agent_config_from_project("p_s")
    assert cfg.is_scratch is True
    assert cfg.agent_name == "Assistant"


def test_auto_start_config_non_scratch_stays_false():
    mgr = _manager_with({"p_n": NORMAL})
    cfg = mgr._build_agent_config_from_project("p_n")
    assert cfg.is_scratch is False
    # agent_name falls back to the project name, matching the /agents/start route
    assert cfg.agent_name == "Hn-daily"
