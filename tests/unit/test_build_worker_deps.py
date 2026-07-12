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


def _make_manager(project=None, projects=None, browser_manager=None):
    """``projects`` (dict of project_id -> project dict) drives a live,
    multi-project ``get_project``/``list_projects`` double — needed for
    ``_compute_scope_roots`` (Spec 12 §2a) to actually walk the store, as
    opposed to the single fixed-project stub used by the rest of this file.

    ``browser_manager`` defaults to ``None`` (no BrowserTool registered on
    worker registries) — pass a double to exercise the worker BrowserTool
    wiring (see ``TestWorkerRegistryBrowserTool`` below).
    """
    ws = MagicMock()
    project_store = MagicMock()
    if projects is not None:
        project_store.get_project = MagicMock(side_effect=lambda pid: projects.get(pid))
        project_store.list_projects = MagicMock(return_value=list(projects.values()))
    else:
        project_store.get_project = MagicMock(return_value=project or {"workspace": "/tmp/ws-x"})
        project_store.list_projects = MagicMock(return_value=[])
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
        browser_manager=browser_manager,
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
        registry = deps.make_tool_registry(None, None, "worker:test-0")

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
        registry = deps.make_tool_registry(None, None, "worker:test-0")

        # Plain ToolRegistry, not the scoped wrapper — has_result depends on
        # ToolRegistry-specific internals (._tools) that ScopedToolRegistry
        # does not expose, so this also proves no unnecessary wrap happened.
        assert hasattr(registry, "_tools")

    def test_scope_given_wraps_in_scoped_registry(self, tmp_path):
        mgr = _make_manager(project={
            "workspace": str(tmp_path), "model": "gpt-4o", "api_key": "k",
        })
        deps = mgr.build_worker_deps("proj-1", "sess-1")
        registry = deps.make_tool_registry(["safe"], None, "worker:test-0")

        from agent_os.agent.tools.scoped_registry import ScopedToolRegistry
        assert isinstance(registry, ScopedToolRegistry)
        assert set(registry.tool_names()) >= {"read", "write", "edit", "grep", "glob", "shell"}


class TestWorkerRegistryBrowserTool:
    """Plan 3 Task 2: a fanout worker gets its own anonymous ``BrowserTool``
    keyed to its ``worker:<fanout_id>-<i>`` handle — a distinct scope from
    the management agent's browser (Task 1's ``BrowserManager`` worker-scope
    routing), never shared credentials, never vision."""

    def test_worker_registry_includes_browser_tool_keyed_to_handle(self):
        mgr = _make_manager(
            project={"workspace": "/tmp/ws-x", "model": "gpt-4o", "api_key": "k"},
            browser_manager=MagicMock(),
        )
        deps = mgr.build_worker_deps("proj-1", "sess-1")
        registry = deps.make_tool_registry(None, None, "worker:f1-0")
        assert "browser" in list(registry.tool_names())

    def test_worker_registry_without_browser_manager_has_no_browser(self):
        mgr = _make_manager(project={
            "workspace": "/tmp/ws-x", "model": "gpt-4o", "api_key": "k",
        })
        deps = mgr.build_worker_deps("proj-1", "sess-1")
        registry = deps.make_tool_registry(None, None, "worker:f1-0")
        assert "browser" not in list(registry.tool_names())

    def test_worker_registry_with_scope_still_includes_browser_tool(self, tmp_path):
        """The browser tool must survive the ``ScopedToolRegistry`` wrap —
        that wrapper only gates path-bearing write tools and delegates
        ``tool_names()`` straight to the inner registry (scoped_registry.py),
        but workers always set a files_scope in practice, so this is the
        realistic path."""
        mgr = _make_manager(
            project={"workspace": str(tmp_path), "model": "gpt-4o", "api_key": "k"},
            browser_manager=MagicMock(),
        )
        deps = mgr.build_worker_deps("proj-1", "sess-1")
        registry = deps.make_tool_registry(["safe"], None, "worker:f1-0")

        from agent_os.agent.tools.scoped_registry import ScopedToolRegistry
        assert isinstance(registry, ScopedToolRegistry)
        assert "browser" in list(registry.tool_names())

    def test_worker_browser_tool_screenshot_namespace_is_windows_safe(self):
        """Round-4 review finding: ``screenshot_namespace`` must NOT carry
        the raw ``worker:<fanout>-<i>`` handle verbatim — ``:`` is invalid in
        a Windows path component, and ``capture_screenshot`` mkdirs
        ``screenshots_dir / namespace`` on every BrowserTool action
        (unconditionally, even with vision off), so an unsanitized handle
        would fail every worker browser action on Windows (WinError 123).
        ``project_id`` (the BrowserManager routing key) must keep the raw
        ``worker:`` prefix though — only the screenshot namespace changes."""
        mgr = _make_manager(
            project={"workspace": "/tmp/ws-x", "model": "gpt-4o", "api_key": "k"},
            browser_manager=MagicMock(),
        )
        deps = mgr.build_worker_deps("proj-1", "sess-1")
        registry = deps.make_tool_registry(None, None, "worker:f1-0")

        browser_tool = registry._tools["browser"]
        assert ":" not in browser_tool._screenshot_namespace
        assert browser_tool._project_id == "worker:f1-0"


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


class TestWorkerReadRootsInheritScratchScope:
    """Spec 12 §2a: a fanout worker dispatched from a scratch (Quick Tasks)
    session inherits the PARENT session's cross-project read scope, computed
    live via ``_compute_scope_roots`` (real project-store entries — this is
    the integration point, not a mock of the method under test). Normal
    projects keep single-root (``_read_roots is None``), byte-identical to
    before this change.
    """

    def test_worker_registry_inherits_scratch_read_roots(self, tmp_path):
        scratch_ws = tmp_path / "scratch"
        other_ws = tmp_path / "other"
        scratch_ws.mkdir()
        other_ws.mkdir()
        projects = {
            "p_scratch": {
                "project_id": "p_scratch", "workspace": str(scratch_ws),
                "model": "gpt-4o", "api_key": "k", "is_scratch": True,
            },
            "p_other": {
                "project_id": "p_other", "workspace": str(other_ws),
                "model": "gpt-4o", "api_key": "k", "is_scratch": False,
            },
        }
        mgr = _make_manager(projects=projects)
        deps = mgr.build_worker_deps("p_scratch", "quick_tasks_11112222")
        registry = deps.make_tool_registry(None, None, "worker:test-0")
        read_tool = registry._tools["read"]
        assert read_tool._read_roots is not None
        roots = read_tool._read_roots()
        assert roots[0] == str(tmp_path / "scratch")
        assert str(tmp_path / "other") in roots

    def test_worker_registry_single_root_for_normal_project(self, tmp_path):
        scratch_ws = tmp_path / "scratch"
        other_ws = tmp_path / "other"
        scratch_ws.mkdir()
        other_ws.mkdir()
        projects = {
            "p_scratch": {
                "project_id": "p_scratch", "workspace": str(scratch_ws),
                "model": "gpt-4o", "api_key": "k", "is_scratch": True,
            },
            "p_other": {
                "project_id": "p_other", "workspace": str(other_ws),
                "model": "gpt-4o", "api_key": "k", "is_scratch": False,
            },
        }
        mgr = _make_manager(projects=projects)
        deps = mgr.build_worker_deps("p_other", "sess_1")
        registry = deps.make_tool_registry(None, None, "worker:test-0")
        read_tool = registry._tools["read"]
        assert read_tool._read_roots is None
