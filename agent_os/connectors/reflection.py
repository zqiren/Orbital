# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Reflect remote MCP tools into the agent's ToolRegistry (Spec 011 §2; Task B3).

For each connector enabled on a project (and globally connected), the manager's
remote tools are listed and wrapped in namespaced ``<connector_id>.<tool_name>``
shim ``Tool`` s that forward call → MCP → tool-result. Each shim carries an
``is_write`` flag the autonomy/approval layer keys off (Spec 011 §0.9):

    manifest tool_overrides  →  MCP readOnlyHint  →  default True (fail closed)

``register_connector_tools`` is the single entry point the coordinator wires
into ``AgentManager.start_agent`` right after ``_register_tools`` (an async
seam). It never raises for one bad connector — it logs and moves on.
"""

import logging

from agent_os.agent.tools.base import Tool, ToolResult

from .manifest import ConnectorManifest

logger = logging.getLogger(__name__)


def classify_is_write(manifest: ConnectorManifest, tool) -> bool:
    """Fail-closed write classification for a reflected tool.

    Precedence: manifest ``tool_overrides`` (``"write"``/``"read"``) → the MCP
    ``readOnlyHint`` annotation → default ``True`` (unknown tools require
    approval). ``tool`` is an ``mcp.types.Tool`` (or any object exposing
    ``name`` + optional ``annotations.readOnlyHint``)."""
    override = manifest.tool_overrides.get(tool.name)
    if override == "write":
        return True
    if override == "read":
        return False
    annotations = getattr(tool, "annotations", None)
    read_only = getattr(annotations, "readOnlyHint", None) if annotations is not None else None
    if read_only is True:
        return False
    if read_only is False:
        return True
    return True  # fail closed


class ReflectedConnectorTool(Tool):
    """A ``Tool`` shim that forwards execution to a remote MCP tool call."""

    is_async = True

    def __init__(self, manager, manifest: ConnectorManifest, mcp_tool):
        self._manager = manager
        self._connector_id = manifest.id
        self._tool_name = mcp_tool.name
        self.name = f"{manifest.id}.{mcp_tool.name}"
        self.description = mcp_tool.description or f"{manifest.name}: {mcp_tool.name}"
        # Pass the MCP inputSchema through unchanged; default to an open object.
        self.parameters = mcp_tool.inputSchema or {"type": "object", "properties": {}}
        self.is_write = classify_is_write(manifest, mcp_tool)

    async def execute(self, **arguments) -> ToolResult:
        try:
            result = await self._manager.call_tool(
                self._connector_id, self._tool_name, arguments
            )
        except Exception as exc:  # tools NEVER raise — surface as content
            return ToolResult(content=f"Error calling {self.name}: {exc}")
        return _to_tool_result(result)


def _to_tool_result(call_result) -> ToolResult:
    """Render an MCP ``CallToolResult`` into a ``ToolResult`` (text v1)."""
    parts: list[str] = []
    for block in getattr(call_result, "content", None) or []:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
        else:
            parts.append(f"[{getattr(block, 'type', 'content')} block]")
    text = "\n".join(parts)
    if getattr(call_result, "isError", False):
        return ToolResult(content=f"Error: {text}" if text else "Error: tool call failed")
    return ToolResult(content=text)


async def build_connector_tools(manager, enabled_ids) -> list[ReflectedConnectorTool]:
    """List every enabled+connected connector's tools and build shims.

    Skips connectors that are unknown, not connected, or whose ``list_tools``
    fails — one bad connector never blocks the others (Spec 011 §0.9)."""
    connected: dict[str, bool] = {
        e["manifest"].id: e["connected"] for e in manager.list_connectors()
    }
    shims: list[ReflectedConnectorTool] = []
    for cid in enabled_ids:
        manifest = manager.get_manifest(cid)
        if manifest is None:
            logger.info("connector %s enabled on project but not in catalog; skipping", cid)
            continue
        if not connected.get(cid, False):
            logger.info("connector %s enabled but not connected; skipping reflection", cid)
            continue
        try:
            tools = await manager.list_tools(cid)
        except Exception:
            logger.warning("failed to list tools for connector %s; skipping", cid, exc_info=True)
            continue
        for mcp_tool in tools:
            shims.append(ReflectedConnectorTool(manager, manifest, mcp_tool))
    return shims


async def register_connector_tools(registry, manager, enabled_ids) -> None:
    """Reflect + register every enabled connector's tools into ``registry``.

    The single call site the coordinator wires into ``start_agent`` (guarded by
    a non-empty ``config.enabled_connectors``)."""
    shims = await build_connector_tools(manager, enabled_ids)
    for shim in shims:
        try:
            registry.register(shim)
        except ValueError:
            # Duplicate name (already registered) — skip rather than crash start.
            logger.warning("connector tool %s already registered; skipping", shim.name)
