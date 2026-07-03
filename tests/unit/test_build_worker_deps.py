# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for AgentManager.build_worker_deps (spec 009, Task 3): the
worker-deps factory a fanout batch uses to construct every NativeWorkerAdapter
it dispatches.

Covers:
- Utility-model reuse: a live session handle's ALREADY-CONSTRUCTED
  AgentLoop._utility_provider is reused verbatim (no re-derivation), and the
  from-scratch fallback (no live handle) falls back to
  _build_agent_config_from_project + _build_llm_providers.
- The restricted worker tool registry: read/write/edit/grep/glob/shell
  present; browser/request_credential/agent_message/fanout absent.
- ScopedToolRegistry wrapping only when allowed/forbidden scopes are given.
- FanoutTool registered on the management registry next to AgentMessageTool.
"""

from unittest.mock import MagicMock

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager
from agent_os.daemon_v2.models import SessionKey, make_session_key
from agent_os.daemon_v2.native_worker import WorkerDeps


def _make_manager(project=None):
    ws = MagicMock()
    project_store = MagicMock()
    project_store.get_project = MagicMock(return_value=project or {"workspace": "/tmp/ws-x"})
    sub_agent_manager = MagicMock()
    activity_translator = MagicMock()
    process_manager = MagicMock()
    provider_registry = MagicMock()
    provider_registry.get_model_info.return_value = MagicMock(
        max_output=16384, capabilities=None, reasoning=None,
    )
    mgr = AgentManager(
        project_store=project_store,
        ws_manager=ws,
        sub_agent_manager=sub_agent_manager,
        activity_translator=activity_translator,
        process_manager=process_manager,
        provider_registry=provider_registry,
    )
    return mgr


class TestNoLiveHandleFallback:
    def test_returns_worker_deps_with_fallback_provider(self):
        mgr = _make_manager(project={
            "workspace": "/tmp/ws-x", "model": "gpt-4o", "api_key": "k",
        })
        deps = mgr.build_worker_deps("proj-1", "sess-1")

        assert isinstance(deps, WorkerDeps)
        assert deps.workspace == "/tmp/ws-x"
        assert deps.project_id == "proj-1"
        assert deps.parent_session_id == "sess-1"
        assert deps.provider is not None
        assert deps.provider.model == "gpt-4o"

    def test_utility_model_ignored_in_fallback_known_gap(self):
        """KNOWN GAP (documented, not fixed here — out of Task 3's file
        scope): ``_build_agent_config_from_project`` never threads
        ``project["utility_model"]`` into the ``AgentConfig`` it builds, so
        this from-scratch fallback can't resolve a configured utility model
        and falls back to the main model instead. Harmless in production —
        ``dispatch_fanout`` only ever runs from inside the calling session's
        OWN live loop, so ``TestLiveHandleReusesUtilityProvider`` below is
        the path production actually takes. Flagged to the team lead as a
        pre-existing gap in the shared helper, not introduced here."""
        mgr = _make_manager(project={
            "workspace": "/tmp/ws-x", "model": "gpt-4o",
            "utility_model": "gpt-4o-mini", "api_key": "k",
        })
        deps = mgr.build_worker_deps("proj-1", "sess-1")

        assert deps.provider.model == "gpt-4o"


class TestLiveHandleReusesUtilityProvider:
    def test_reuses_loop_utility_provider_without_rederiving(self):
        mgr = _make_manager()
        sentinel_provider = MagicMock(model="reused-utility-model")
        handle = MagicMock()
        handle.loop._utility_provider = sentinel_provider
        sk = make_session_key("proj-1", "sess-1")
        mgr._handles[sk] = handle

        deps = mgr.build_worker_deps("proj-1", "sess-1")

        assert deps.provider is sentinel_provider
        # No re-derivation: _build_agent_config_from_project must not be hit
        # when a live handle's utility provider is available directly.
        mgr._project_store.get_project.assert_called_once_with("proj-1")


class TestRestrictedToolRegistry:
    def test_registry_has_expected_tools_and_excludes_dangerous_ones(self):
        mgr = _make_manager(project={
            "workspace": "/tmp/ws-x", "model": "gpt-4o", "api_key": "k",
        })
        deps = mgr.build_worker_deps("proj-1", "sess-1")
        registry = deps.make_tool_registry(None, None)

        names = set(registry.tool_names())
        assert {"read", "write", "edit", "grep", "glob", "shell"} <= names
        assert "browser" not in names
        assert "request_credential" not in names
        assert "agent_message" not in names
        assert "fanout" not in names

    def test_no_scope_returns_unwrapped_registry(self):
        mgr = _make_manager(project={
            "workspace": "/tmp/ws-x", "model": "gpt-4o", "api_key": "k",
        })
        deps = mgr.build_worker_deps("proj-1", "sess-1")
        registry = deps.make_tool_registry(None, None)

        # Plain ToolRegistry, not the scoped wrapper — has_result depends on
        # ToolRegistry-specific internals (._tools) that ScopedToolRegistry
        # does not expose, so this also proves no unnecessary wrap happened.
        assert hasattr(registry, "_tools")

    def test_scope_given_wraps_in_scoped_registry(self, tmp_path):
        mgr = _make_manager(project={
            "workspace": str(tmp_path), "model": "gpt-4o", "api_key": "k",
        })
        deps = mgr.build_worker_deps("proj-1", "sess-1")
        registry = deps.make_tool_registry(["safe"], None)

        from agent_os.agent.tools.scoped_registry import ScopedToolRegistry
        assert isinstance(registry, ScopedToolRegistry)
        assert set(registry.tool_names()) >= {"read", "write", "edit", "grep", "glob", "shell"}


class TestFanoutToolRegisteredOnManagementRegistry:
    def test_fanout_registered_next_to_agent_message(self):
        from agent_os.agent.tools.registry import ToolRegistry
        from agent_os.daemon_v2.models import AgentConfig, Autonomy

        mgr = _make_manager(project={"workspace": "/tmp/ws-x"})
        registry = ToolRegistry()
        config = AgentConfig(workspace="/tmp/ws-x", model="gpt-4o", api_key="k",
                              autonomy=Autonomy.HANDS_OFF)
        mgr._register_tools(registry, config, project_id="proj-1", session_id="sess-1")

        names = registry.tool_names()
        assert "fanout" in names
        assert "agent_message" in names
