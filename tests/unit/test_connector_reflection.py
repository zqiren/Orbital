# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for reflecting remote MCP tools into the agent's ToolRegistry (Task B3).

Fail-closed classification (Spec 011 §0.9): manifest override wins, then the
MCP ``readOnlyHint``, then default to write."""

from dataclasses import replace

import pytest

from agent_os.agent.tools.registry import ToolRegistry
from agent_os.connectors import load_catalog
from agent_os.connectors.manager import ConnectorManager
from agent_os.connectors.reflection import (
    build_connector_tools,
    classify_is_write,
    register_connector_tools,
)

from tests.unit._mock_mcp_server import build_mock_server, in_memory_opener


async def _manager_with_custom(tmp_path):
    server = build_mock_server()
    mgr = ConnectorManager(
        catalog=load_catalog(),
        credential_store=None,
        data_dir=str(tmp_path),
        session_opener=in_memory_opener(server),
    )
    await mgr.add_custom("Mock Server", "https://mock/mcp", "none")
    return mgr


class _FakeAnnotations:
    def __init__(self, read_only):
        self.readOnlyHint = read_only


class _FakeTool:
    def __init__(self, name, read_only=None):
        self.name = name
        self.annotations = _FakeAnnotations(read_only) if read_only is not None else None


def test_classify_is_write_manifest_override_wins():
    cat = {m.id: m for m in load_catalog()}
    cal = cat["google-calendar"]
    # tool_overrides classify these regardless of any hint.
    assert classify_is_write(cal, _FakeTool("create_event", read_only=True)) is True
    assert classify_is_write(cal, _FakeTool("list_events", read_only=False)) is False


def test_classify_is_write_falls_back_to_readonly_hint():
    cat = {m.id: m for m in load_catalog()}
    cal = cat["google-calendar"]
    unknown_ro = _FakeTool("some_unlisted_read", read_only=True)
    unknown_rw = _FakeTool("some_unlisted_write", read_only=False)
    assert classify_is_write(cal, unknown_ro) is False
    assert classify_is_write(cal, unknown_rw) is True


def test_classify_is_write_fails_closed_when_unknown():
    cat = {m.id: m for m in load_catalog()}
    cal = cat["google-calendar"]
    no_hint = _FakeTool("totally_unknown")  # no override, no annotation
    assert classify_is_write(cal, no_hint) is True


async def test_build_connector_tools_namespaces_and_classifies(tmp_path):
    mgr = await _manager_with_custom(tmp_path)
    tools = await build_connector_tools(mgr, ["custom-mock-server"])
    by_name = {t.name for t in tools}
    assert by_name == {"custom-mock-server.echo_read", "custom-mock-server.echo_write"}

    read_shim = next(t for t in tools if t.name.endswith("echo_read"))
    write_shim = next(t for t in tools if t.name.endswith("echo_write"))
    # readOnlyHint True -> read; missing annotation -> fail closed to write.
    assert read_shim.is_write is False
    assert write_shim.is_write is True
    # inputSchema passed through unchanged.
    assert read_shim.parameters.get("type") == "object"
    assert "text" in read_shim.parameters.get("properties", {})


async def test_register_into_registry_and_dispatch(tmp_path):
    mgr = await _manager_with_custom(tmp_path)
    registry = ToolRegistry()
    await register_connector_tools(registry, mgr, ["custom-mock-server"])

    names = set(registry.tool_names())
    assert "custom-mock-server.echo_read" in names
    assert "custom-mock-server.echo_write" in names
    # Reflected shims are async in the registry.
    assert registry.is_async("custom-mock-server.echo_read") is True

    result = await registry.execute_async(
        "custom-mock-server.echo_read", {"text": "hi"}
    )
    assert "read:hi" in (result.content if isinstance(result.content, str) else "")


async def test_disconnected_or_unknown_connectors_are_skipped(tmp_path):
    # google-calendar is oauth2 + not connected; unknown id doesn't exist.
    mgr = ConnectorManager(
        catalog=load_catalog(), credential_store=None, data_dir=str(tmp_path),
        session_opener=in_memory_opener(build_mock_server()),
    )
    registry = ToolRegistry()
    # Must not raise even though nothing can be reflected.
    await register_connector_tools(
        registry, mgr, ["google-calendar", "does-not-exist"]
    )
    assert registry.tool_names() == []
