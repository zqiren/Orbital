# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""One rule for a connector write, whoever calls it: the project's autonomy.

- hands-off: the write runs without asking (what the user chose);
- check-in: it asks first (a connector write is outward-facing);
- supervised: it asks first.
A read never asks because of this rule. Workers read the setting at call time
(``SubAgentManager.project_autonomy``); the manager's interceptor gets the
names of its write-classified connector tools at start.

And the manager actually receives connectors now: its config carries the
project's ``enabled_connectors`` (never assigned since spec 011 phase A,
02b9473e, so no release ever gave the manager a connector tool), filtered by
the connector's agent access for ``orbital``. With nothing enabled, its tool
list and prompt are what they were.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.prompt_builder import Autonomy
from agent_os.agent.tools.registry import ToolRegistry
from agent_os.connectors import load_catalog
from agent_os.connectors.manager import ConnectorManager
from agent_os.daemon_v2.autonomy import AutonomyInterceptor
from agent_os.daemon_v2.connector_sharing import WorkerConnectorTools
from agent_os.daemon_v2.dispatch_tools import DispatchContext

from tests.unit._mock_mcp_server import build_mock_server, in_memory_opener

CTX = DispatchContext("p1", "s1", "codex", can_dispatch=False, connectors=("custom-mock",))


async def _cm(tmp_path):
    cm = ConnectorManager(catalog=load_catalog(), credential_store=None,
                          data_dir=str(tmp_path),
                          session_opener=in_memory_opener(build_mock_server()))
    await cm.add_custom("Mock", "https://mock/mcp", "none")
    return cm


def _sam(cm, autonomy):
    sam = MagicMock()
    sam._connector_manager = cm
    sam.live_connector_ids = lambda pid, handle: ("custom-mock",)
    sam.request_connector_approval = AsyncMock(return_value=True)
    sam.autonomy = autonomy
    sam.project_autonomy = lambda pid: sam.autonomy
    return sam


# ---------------------------------------------------------------- workers

@pytest.mark.parametrize("autonomy,asks", [
    (Autonomy.HANDS_OFF, False),
    (Autonomy.CHECK_IN, True),
    (Autonomy.SUPERVISED, True),
])
async def test_a_worker_write_follows_the_project_autonomy(tmp_path, autonomy, asks):
    sam = _sam(await _cm(tmp_path), autonomy)
    tools = WorkerConnectorTools(sam)
    result = await tools.call(CTX, "custom-mock__echo_write", {"text": "x"})
    assert result.isError is False
    assert result.content[0].text == "write:x"
    assert sam.request_connector_approval.await_count == (1 if asks else 0)


@pytest.mark.parametrize("autonomy", list(Autonomy))
async def test_a_worker_read_never_asks(tmp_path, autonomy):
    sam = _sam(await _cm(tmp_path), autonomy)
    result = await WorkerConnectorTools(sam).call(CTX, "custom-mock__echo_read", {"text": "x"})
    assert result.content[0].text == "read:x"
    sam.request_connector_approval.assert_not_awaited()


async def test_the_setting_is_read_at_call_time_not_at_mount(tmp_path):
    sam = _sam(await _cm(tmp_path), Autonomy.HANDS_OFF)
    tools = WorkerConnectorTools(sam)
    await tools.call(CTX, "custom-mock__echo_write", {"text": "a"})
    sam.autonomy = Autonomy.CHECK_IN
    await tools.call(CTX, "custom-mock__echo_write", {"text": "b"})
    assert sam.request_connector_approval.await_count == 1


def _sub_agent_manager(project):
    from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
    store = MagicMock()
    store.get_project.return_value = project
    return SubAgentManager(process_manager=MagicMock(), registry=None,
                           setup_engine=None, project_store=store)


@pytest.mark.parametrize("stored,expected", [
    ({"autonomy": "check_in"}, Autonomy.CHECK_IN),
    ({"autonomy": "supervised"}, Autonomy.SUPERVISED),
    ({"autonomy": "hands_off"}, Autonomy.HANDS_OFF),
    # The manager's own default when the field is missing or unreadable
    # (agent_manager: project.get("autonomy", "hands_off")).
    ({}, Autonomy.HANDS_OFF),
    ({"autonomy": "weird"}, Autonomy.HANDS_OFF),
])
def test_project_autonomy_reads_the_project_like_the_manager_does(stored, expected):
    assert _sub_agent_manager(stored).project_autonomy("p1") is expected


# ---------------------------------------------------------------- manager interceptor

def _interceptor(preset, writes=frozenset()):
    return AutonomyInterceptor(preset, MagicMock(), "p1", connector_writes=writes)


WRITES = frozenset({"custom-mock__echo_write"})


@pytest.mark.parametrize("preset,asks", [
    (Autonomy.HANDS_OFF, False),
    (Autonomy.CHECK_IN, True),
    (Autonomy.SUPERVISED, True),
])
def test_a_manager_connector_write_follows_the_same_rule(preset, asks):
    tc = {"name": "custom-mock__echo_write", "arguments": {"text": "x"}}
    assert _interceptor(preset, WRITES).should_intercept(tc) is asks


def test_this_rule_never_makes_a_manager_read_ask():
    tc = {"name": "custom-mock__echo_read", "arguments": {}}
    assert _interceptor(Autonomy.HANDS_OFF, WRITES).should_intercept(tc) is False
    assert _interceptor(Autonomy.CHECK_IN, WRITES).should_intercept(tc) is False


@pytest.mark.parametrize("preset", list(Autonomy))
@pytest.mark.parametrize("name", ["read", "write", "shell", "browser", "custom-mock__echo_write"])
def test_without_connector_writes_the_interceptor_is_unchanged(preset, name):
    tc = {"name": name, "arguments": {"action": "click"} if name == "browser" else {}}
    old = AutonomyInterceptor(preset, MagicMock(), "p1")
    new = _interceptor(preset)
    assert new.should_intercept(tc) == old.should_intercept(tc)


def test_approve_all_still_covers_a_manager_connector_write():
    ic = _interceptor(Autonomy.CHECK_IN, WRITES)
    import time
    ic._bypass_all_until = time.time() + 60
    assert ic.should_intercept({"name": "custom-mock__echo_write", "arguments": {}}) is False


# ---------------------------------------------------------------- manager wiring

def _agent_manager(projects, cm=None):
    from agent_os.daemon_v2.agent_manager import AgentManager
    from tests.card_doubles import FakeCardStore
    store = MagicMock()
    store.get_project = MagicMock(side_effect=lambda pid: projects.get(pid))
    store.list_projects = MagicMock(return_value=list(projects.values()))
    sam = MagicMock()
    sam.list_active = MagicMock(return_value=[])
    sam.stop = AsyncMock()
    sam.stop_all = AsyncMock()
    credential_store = MagicMock()
    credential_store.get_api_key = MagicMock(return_value=None)
    return AgentManager(
        project_store=store, ws_manager=MagicMock(), sub_agent_manager=sam,
        activity_translator=MagicMock(), process_manager=MagicMock(),
        settings_store=FakeCardStore.with_default(
            provider="anthropic", model="claude-x", key="sk-test"),
        credential_store=credential_store, connector_manager=cm,
    )


def test_the_manager_config_carries_the_projects_connectors():
    mgr = _agent_manager({"p1": {"project_id": "p1", "name": "n", "workspace": "/tmp/w",
                                 "enabled_connectors": ["custom-mock", "gmail"]},
                          "p2": {"project_id": "p2", "name": "m", "workspace": "/tmp/w"}})
    assert mgr._build_agent_config_from_project("p1").enabled_connectors == [
        "custom-mock", "gmail"]
    assert mgr._build_agent_config_from_project("p2").enabled_connectors == []


async def _start(mgr, tmp_path, enabled):
    """Run start_agent with a REAL ToolRegistry and capture what the loop and
    the interceptor were given."""
    from agent_os.daemon_v2.models import AgentConfig
    ws = str(tmp_path / "ws")
    os.makedirs(ws, exist_ok=True)
    config = AgentConfig(workspace=ws, model="claude-x", api_key="sk-test",
                         enabled_connectors=list(enabled))
    with patch("agent_os.daemon_v2.agent_manager.LLMProvider"), \
         patch("agent_os.daemon_v2.agent_manager.Session") as MockSession, \
         patch("agent_os.daemon_v2.agent_manager.ContextManager"), \
         patch("agent_os.daemon_v2.agent_manager.AgentLoop") as MockLoop, \
         patch("agent_os.daemon_v2.agent_manager.AutonomyInterceptor") as MockIc:
        MockSession.new.return_value = MagicMock()
        loop = MagicMock()
        loop.run = AsyncMock()
        MockLoop.return_value = loop
        await mgr.start_agent("p1", config, session_id="s1")
    registry = MockLoop.call_args.args[2]
    return registry, MockIc.call_args.kwargs


@pytest.mark.parametrize("case", ["enabled", "project_off", "not_allowed", "disconnected"])
async def test_the_manager_gets_a_connector_only_when_it_may(tmp_path, case):
    cm = await _cm(tmp_path)
    enabled = ["custom-mock"]
    if case == "project_off":
        enabled = []
    elif case == "not_allowed":
        cm.set_access("custom-mock", "selected", ["codex"])
    elif case == "disconnected":
        await cm.remove_custom("custom-mock")
    mgr = _agent_manager({}, cm)
    registry, ic_kwargs = await _start(mgr, tmp_path, enabled)
    names = set(registry.tool_names())
    if case == "enabled":
        assert {"custom-mock__echo_read", "custom-mock__echo_write"} <= names
        assert ic_kwargs["connector_writes"] == frozenset({"custom-mock__echo_write"})
    else:
        assert not any(n.startswith("custom-mock") for n in names)
        assert ic_kwargs["connector_writes"] == frozenset()


async def test_with_nothing_enabled_the_manager_tools_and_prompt_are_as_before(tmp_path):
    """Same tool schemas, same prompt context with and without a connector
    manager, when the project enables no connector."""
    seen = []
    for cm in (None, await _cm(tmp_path)):
        mgr = _agent_manager({}, cm)
        with patch("agent_os.daemon_v2.agent_manager.PromptBuilder") as PB:
            registry, _ = await _start(mgr, tmp_path, [])
        built = [c.args for c in PB.return_value.build.call_args_list]
        seen.append((registry.schemas(), repr(built)))
    assert seen[0][0] == seen[1][0]
    assert seen[0][1] == seen[1][1]


def test_a_manager_connector_result_still_gets_the_json_prefilter():
    """tool_result_filters recognised connector shims by the dot; the new
    names keep that JSON pre-filter, and no built-in tool is caught by it."""
    from agent_os.agent.tool_result_filters import _is_json_api_tool
    assert _is_json_api_tool("gmail__search_threads")
    assert _is_json_api_tool("gmail.search_threads")        # older sessions' rows
    for builtin in ("read", "write", "shell", "agent_message", "request_credential"):
        assert not _is_json_api_tool(builtin)
