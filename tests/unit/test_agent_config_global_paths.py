# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 073 §2.1 regression: ``_build_agent_config_from_project`` must
populate BOTH user-scoped file paths from global settings.

``AgentConfig.global_preferences_path`` was declared and read into
``PromptContext``, but the (only) construction site never assigned it — so the
"## Global User Preferences" section silently never rendered in production.
These tests pin the assignment, the ~/orbital defaults when Global Settings
leaves the paths unset, and the ``user_memory_enabled`` gate (off → empty
``user_memory_path`` → tool unregistered and prompt section omitted
together)."""

import os
from unittest.mock import MagicMock

from agent_os.daemon_v2.agent_manager import AgentManager
from agent_os.daemon_v2.settings_store import GlobalSettings

PROJECT = {"project_id": "p1", "name": "Proj", "workspace": "/tmp/p1"}

DEFAULT_PREFS = os.path.join(os.path.expanduser("~"), "orbital", "user_preferences.md")
DEFAULT_MEMORY = os.path.join(os.path.expanduser("~"), "orbital", "user_memory.md")


def _manager(global_settings) -> AgentManager:
    """The construction idiom from tests/unit/test_agent_config_parity.py,
    with a REAL GlobalSettings (not a MagicMock) so the path/toggle fields
    behave like production values instead of truthy mocks."""
    project_store = MagicMock()
    project_store.get_project = MagicMock(
        side_effect=lambda pid: {"p1": PROJECT}.get(pid))
    settings_store = MagicMock()
    settings_store.get = MagicMock(return_value=global_settings)
    credential_store = MagicMock()
    credential_store.get_api_key = MagicMock(return_value=None)
    return AgentManager(
        project_store=project_store,
        ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(),
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
        provider_registry=MagicMock(),
        settings_store=settings_store,
        credential_store=credential_store,
    )


def test_defaults_when_global_settings_leave_paths_unset():
    cfg = _manager(GlobalSettings())._build_agent_config_from_project("p1")
    assert cfg.global_preferences_path == DEFAULT_PREFS
    assert cfg.user_memory_path == DEFAULT_MEMORY


def test_configured_paths_propagate():
    gs = GlobalSettings(user_preferences_path="/custom/prefs.md",
                        user_memory_path="/custom/memory.md")
    cfg = _manager(gs)._build_agent_config_from_project("p1")
    assert cfg.global_preferences_path == "/custom/prefs.md"
    assert cfg.user_memory_path == "/custom/memory.md"


def test_toggle_off_empties_memory_path_but_keeps_prefs():
    gs = GlobalSettings(user_memory_enabled=False,
                        user_memory_path="/custom/memory.md")
    cfg = _manager(gs)._build_agent_config_from_project("p1")
    assert cfg.user_memory_path == ""
    assert cfg.global_preferences_path == DEFAULT_PREFS


def test_defaults_when_settings_store_returns_none():
    """No settings on disk at all: both paths still resolve to the ~/orbital
    siblings, and user memory defaults ON."""
    cfg = _manager(None)._build_agent_config_from_project("p1")
    assert cfg.global_preferences_path == DEFAULT_PREFS
    assert cfg.user_memory_path == DEFAULT_MEMORY
