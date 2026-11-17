# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Connectors shared with workers, over the daemon's one MCP endpoint.

A connector (a remote MCP server the user signed in to once) used to reach
only Orbital's own loop. A worker (Claude Code, Codex, ACP agents, Pi) now
gets the connectors that are connected, enabled for the project it runs in,
and allowed for that agent (``ConnectorManager.shared_ids``), served as more
tools of the ``orbital`` MCP server it already mounts for spec 161. The
bearer token the worker holds resolves its ``DispatchContext``, which names
the connectors fixed at spawn; the connector's own secret stays in the
daemon, and nothing is written into the worker's config files.

- Names: ``<connector_id>__<tool>``, cut to ``[A-Za-z0-9_-]`` and 50
  characters (Claude Code prefixes ``mcp__orbital__``; both Anthropic and
  OpenAI cap a function name at 64), with a short hash when cut so the name
  stays stable and unique.
- A read runs. A write follows the project's autonomy, read at call time,
  the same rule as the manager's: hands-off runs it; check-in and supervised
  ask the user first in the session's sub-agent approval card
  (``SubAgentManager.request_connector_approval``); a denial is a tool error.
- Access is re-read on every list and call: a connector the user took away
  stops working at once, although the worker's tool list only changes when
  it is rebuilt at its next turn.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time

from mcp.types import CallToolResult, TextContent, Tool as MCPTool

from agent_os.connectors.reflection import (  # noqa: F401 — re-exported
    MAX_TOOL_NAME, TOOL_NAME_SEP, classify_is_write, exposed_tool_name)

logger = logging.getLogger(__name__)

#: The manager's agent id (the pin switcher and the Workbench call it this;
#: ``agents_v2`` reserves the slug).
MANAGER_AGENT_ID = "orbital"
MANAGER_NAME = "Orbital"

_SEP = TOOL_NAME_SEP
_INVALID = re.compile(r"[^A-Za-z0-9_-]")

#: How long a connector's tool list is reused across workers' lists and calls.
TOOL_LIST_TTL_S = 300.0
#: How often a write waiting on the user checks that the worker is still
#: waiting for it (its MCP request still open).
DISCONNECT_POLL_S = 1.0

#: Transports that cannot mount an HTTP MCP server: they never receive
#: connectors (``GET /api/v2/connectors/agents`` reports the reason).
UNSUPPORTED_TRANSPORTS = {"pty": "no_mcp_mount", "pipe": "no_mcp_mount"}


def transport_kind(manifest) -> str:
    """The transport ``SubAgentManager._resolve_transport`` picks for a
    manifest, without building it ("auto" pipe means the SDK)."""
    runtime = getattr(manifest, "runtime", None)
    hint = getattr(runtime, "transport", "auto") or "auto"
    if hint != "auto":
        return hint
    return "sdk" if getattr(runtime, "mode", None) == "pipe" else "pty"


def support_of(manifest) -> tuple[bool, str | None]:
    reason = UNSUPPORTED_TRANSPORTS.get(transport_kind(manifest))
    return reason is None, reason


def connectors_note(names: list[str]) -> str | None:
    """The first-turn line for a worker that got connector tools. Codex keeps
    MCP tools out of its visible list (code mode) and Claude Code may defer
    them, so the tool list alone is not enough for the model to reach for
    them."""
    if not names:
        return None
    return (f"Connected services you can use through the `orbital` MCP server: "
            f"{', '.join(names)}. Anything that changes something there "
            f"(sending, creating, deleting) waits for the user's approval in "
            f"Orbital; if they decline, do not retry it.")


def _text(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], isError=True)


class WorkerConnectorTools:
    """The connector half of the ``orbital`` MCP server: list and call the
    connectors a worker's token carries. ``sam`` is the SubAgentManager (its
    ``_connector_manager``, ``live_connector_ids`` and
    ``request_connector_approval``)."""

    def __init__(self, sub_agent_manager):
        self._sam = sub_agent_manager
        self._cache: dict[str, tuple[float, list]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    @property
    def _cm(self):
        return getattr(self._sam, "_connector_manager", None)

    async def _tools_of(self, cid: str) -> list:
        now = time.monotonic()
        hit = self._cache.get(cid)
        if hit is not None and hit[0] > now:
            return hit[1]
        lock = self._locks.setdefault(cid, asyncio.Lock())
        async with lock:
            hit = self._cache.get(cid)
            if hit is not None and hit[0] > time.monotonic():
                return hit[1]
            tools = list(await self._cm.list_tools(cid))
            self._cache[cid] = (time.monotonic() + TOOL_LIST_TTL_S, tools)
            return tools

    async def _live(self, ctx) -> list[str]:
        if not ctx.connectors or self._cm is None:
            return []
        live = await asyncio.to_thread(
            self._sam.live_connector_ids, ctx.project_id, ctx.handle)
        return [c for c in ctx.connectors if c in live]

    async def _entries(self, ctx) -> list[tuple]:
        """``(exposed_name, connector_id, manifest, mcp_tool)`` for every
        tool of the connectors this context may use right now. One connector
        that cannot list costs only its own tools."""
        ids = await self._live(ctx)
        if not ids:
            return []
        listed = await asyncio.gather(*(self._tools_of(c) for c in ids),
                                      return_exceptions=True)
        out = []
        for cid, tools in zip(ids, listed):
            if isinstance(tools, BaseException):
                logger.warning("connector %s: tool list failed for worker %s (%s)",
                               cid, ctx.handle, type(tools).__name__)
                continue
            manifest = self._cm.get_manifest(cid)
            if manifest is None:
                continue
            for tool in tools:
                out.append((exposed_tool_name(cid, tool.name), cid, manifest, tool))
        return out

    async def list(self, ctx) -> list[MCPTool]:
        if ctx is None or not ctx.connectors:
            return []
        tools = []
        for name, _cid, manifest, tool in await self._entries(ctx):
            tools.append(MCPTool(
                name=name,
                description=f"{manifest.name}: {tool.description or tool.name}",
                inputSchema=tool.inputSchema or {"type": "object", "properties": {}},
                annotations=getattr(tool, "annotations", None),
            ))
        return tools

    def _spawned_with(self, ctx, name: str) -> str | None:
        for cid in ctx.connectors:
            if name.startswith(_INVALID.sub("_", cid) + _SEP):
                return cid
        return None

    async def _approved(self, ctx, label: str, args: dict, request) -> bool | None:
        """The user's answer, or None when the worker gave up on the call
        first (its client timed out or it was stopped): then the card comes
        down and the write never runs, since nobody would see its result."""
        ask = asyncio.ensure_future(
            self._sam.request_connector_approval(ctx, label, args))
        watch = None
        is_disconnected = getattr(request, "is_disconnected", None)
        if callable(is_disconnected):
            async def _gone():
                while True:
                    await asyncio.sleep(DISCONNECT_POLL_S)
                    try:
                        if await is_disconnected():
                            return
                    except Exception:  # noqa: BLE001 — no signal: keep waiting
                        pass
            watch = asyncio.ensure_future(_gone())
        try:
            await asyncio.wait({ask} | ({watch} if watch else set()),
                               return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            ask.cancel()
            raise
        finally:
            if watch is not None:
                watch.cancel()
        if ask.done():
            return bool(ask.result())
        ask.cancel()
        try:
            await ask
        except asyncio.CancelledError:
            pass
        return None

    async def _asks(self, ctx) -> bool:
        """One rule with the manager's: the project's autonomy, read now.
        Hands-off runs the write (the user chose that); check-in and
        supervised ask first."""
        from agent_os.agent.prompt_builder import Autonomy
        autonomy = await asyncio.to_thread(self._sam.project_autonomy, ctx.project_id)
        return autonomy is not Autonomy.HANDS_OFF

    async def call(self, ctx, name: str, arguments: dict | None,
                   request=None) -> CallToolResult | None:
        """The result of a connector tool call, or None when ``name`` is not
        a connector tool of this context (the caller falls through).
        ``request`` is the worker's HTTP request, watched while a write waits
        on the user."""
        if ctx is None or not ctx.connectors or self._cm is None:
            return None
        entry = next((e for e in await self._entries(ctx) if e[0] == name), None)
        if entry is None:
            cid = self._spawned_with(ctx, name)
            if cid is None:
                return None
            return _text(f"Error: {cid} is no longer available to you (the user "
                         f"changed who can use it, turned it off for this project, "
                         f"or disconnected it). Tell the user; do not retry.")
        _name, cid, manifest, tool = entry
        label = f"{cid}.{tool.name}"
        args = dict(arguments or {})
        if classify_is_write(manifest, tool) and await self._asks(ctx):
            approved = await self._approved(ctx, label, args, request)
            logger.info("connector write %s from %s/%s: %s", label, ctx.project_id,
                        ctx.handle, {True: "approved", False: "declined"}.get(
                            approved, "abandoned by the worker; not run"))
            if approved is None:
                return _text(f"Error: {label} was not run (the call ended before "
                             f"the user answered).")
            if not approved:
                return _text(f"Error: the user declined this action ({label}). Do "
                             f"not retry it unless the user asks you to.")
        try:
            result = await self._cm.call_tool(cid, tool.name, args)
        except Exception as exc:  # noqa: BLE001 — the model gets text, never a trace
            logger.warning("connector call %s from %s failed: %s", label, ctx.handle,
                           type(exc).__name__)
            return _text(f"Error calling {label}: {exc}")
        return CallToolResult(content=list(getattr(result, "content", None) or []),
                              isError=bool(getattr(result, "isError", False)))
