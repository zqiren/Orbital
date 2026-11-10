# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 161: the worker at the top of a session can dispatch other workers.

The manager seat is a position. Whoever holds it — the Orbital loop, a CLI
worker the user pinned the chat to, or a worker an automation is assigned
to — can hand work to any other installed worker, and the result comes back
to whoever asked as its next turn. The Orbital loop does it with
``agent_message``; a CLI worker does it with the one tool here, served over
one streamable-HTTP MCP endpoint on the daemon (``/mcp``, loopback).

Only the top of a session gets the tool (``TOP_INITIATORS``): workers below
the seat pay no context for it, so nothing deeper than one level can exist.
Each such spawn mints a bearer token that resolves to its
``DispatchContext``; the token lives in memory on the sub-agent manager and
dies with the adapter. The tool is a thin wrapper over
``SubAgentManager.send`` — the only dispatch verb — with
``initiator="worker:<handle>"``, which is what routes the child's dispatch
marker and terminal back to this worker (``LifecycleObserver``).
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Annotated

from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field
from mcp.server.fastmcp.server import StreamableHTTPASGIApp
from starlette.routing import Route

logger = logging.getLogger(__name__)

#: The MCP server name both CLIs see; claude-code prefixes tools with it.
SERVER_NAME = "orbital"
TOOL_NAME = "dispatch"
#: The env var codex reads the bearer token from (``bearer_token_env_var``).
TOKEN_ENV = "ORBITAL_DISPATCH_TOKEN"
#: Initiators that put a worker at the top of its session (spec 161 §5.2).
TOP_INITIATORS = ("user_pinned", "queue_item")
#: ``initiator`` prefix for a dispatch a worker made (the parent's handle follows).
WORKER_INITIATOR_PREFIX = "worker:"

_DESCRIPTION = (
    "Hand a task to another agent in this project. Use it whenever the user "
    "asks for another agent, by name or by role. Never run another agent's "
    "CLI from your shell instead: its reply would never reach the user. The "
    "agent works in the background and its result arrives as your next "
    "message: after calling this, end your turn and wait."
)
# Live 2026-11-09 (v0.21.0): with only "what it does" in the description,
# pinned Codex listed the tool and then ran ``claude -p`` in its shell. Codex
# gets no Orbital system prompt, so the trigger has to live here.

_AGENT_PARAM = "The agent's id, exactly as listed in this tool's description."
# Live 2026-11-10 (Windows smoke, 168-2): Claude Code first passed the text as
# `message`, got a validation error, and then recorded the schema detail in the
# project's LESSONS.md. The argument's own words now say where the text goes.
_TASK_PARAM = ("The task text itself goes in this `task` argument (there is no "
               "`message` argument). Write the whole task so the agent can act "
               "on it without seeing this conversation.")
_FRESH_PARAM = ("true starts the agent over without its earlier conversation. "
                "Leave false to continue it.")


@dataclass(frozen=True)
class DispatchContext:
    project_id: str
    session_id: str
    handle: str          # the seat-holder's handle (slug)
    # The project's enabled slugs at spawn time, named in the tool
    # description so the model needs no lookup.
    agents: tuple[str, ...] = ()


def dispatch_url() -> str | None:
    """``http://127.0.0.1:{AGENT_OS_PORT}/mcp``, or None when the port is
    unknown (the packaged app and the dev scripts export it)."""
    port = (os.environ.get("AGENT_OS_PORT") or "").strip()
    if not port.isdigit():
        return None
    return f"http://127.0.0.1:{port}/mcp"


def tool_description(ctx: DispatchContext | None) -> str:
    others = [a for a in (ctx.agents if ctx else ()) if a != (ctx.handle if ctx else None)]
    if not others:
        return _DESCRIPTION
    return f"{_DESCRIPTION} Agents: {', '.join(others)}."


def seat_note(ctx: DispatchContext) -> str | None:
    """The first-turn line for every seat-holder with dispatch available,
    whatever its transport. Codex's code mode lists MCP tools only inside
    ``ALL_TOOLS`` and Claude Code may defer them behind ToolSearch, so a
    tool's own description is not enough for the model to reach for it.
    None when there is no one to hand work to."""
    others = [a for a in ctx.agents if a != ctx.handle]
    if not others:
        return None
    return (f"Other agents in this project: {', '.join(others)}. When the user "
            f"asks for one of them, hand it the task with the `{TOOL_NAME}` tool "
            f"of the `{SERVER_NAME}` MCP server. Never run that agent's CLI from "
            f"your shell: its reply would never reach the user.")


async def dispatch(sub_agent_manager, ctx: DispatchContext, agent: str, task: str,
                   fresh: bool = False) -> str:
    """Send ``task`` to ``agent`` on behalf of the seat-holder ``ctx``."""
    agent = (agent or "").strip()
    if agent == ctx.handle:
        return f"You are {agent}; do this task yourself."
    ack = await sub_agent_manager.send(
        ctx.project_id, agent, task, session_id=ctx.session_id, depth=1,
        fresh=fresh, initiator=f"{WORKER_INITIATOR_PREFIX}{ctx.handle}")
    if isinstance(ack, str) and ack.startswith("Error"):
        return ack
    return (f"Dispatched to {agent}. {ack} The result will arrive as your next "
            f"message. End your turn now and wait for it.")


def _bearer(request) -> str | None:
    if request is None:
        return None
    header = request.headers.get("authorization") or ""
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


class DispatchServer:
    """The one MCP endpoint: a FastMCP app whose single tool resolves the
    caller's context from its bearer token. Mounted as a plain route on the
    daemon's FastAPI app; its session manager runs for the daemon's life."""

    def __init__(self, sub_agent_manager):
        self._sam = sub_agent_manager
        server_self = self

        class _Server(FastMCP):
            async def list_tools(self):
                tools = await super().list_tools()
                ctx = server_self._context_of(self.get_context())
                for tool in tools:
                    if tool.name == TOOL_NAME:
                        tool.description = tool_description(ctx)
                return tools

        mcp = _Server(SERVER_NAME, stateless_http=True, json_response=True,
                      streamable_http_path="/mcp")

        @mcp.tool(name=TOOL_NAME, description=_DESCRIPTION, structured_output=False)
        async def _dispatch(
            agent: Annotated[str, Field(description=_AGENT_PARAM)],
            task: Annotated[str, Field(description=_TASK_PARAM)],
            ctx: Context,
            fresh: Annotated[bool, Field(description=_FRESH_PARAM)] = False,
        ) -> str:
            dctx = server_self._context_of(ctx)
            if dctx is None:
                return ("Error: this dispatch link is no longer valid (the agent "
                        "was restarted). Tell the user the task was not handed on.")
            try:
                return await dispatch(server_self._sam, dctx, agent, task, fresh=fresh)
            except Exception as exc:  # noqa: BLE001 — the model gets text, never a trace
                logger.warning("dispatch from %s to %s failed", dctx.handle, agent,
                               exc_info=True)
                return f"Error: the dispatch failed ({type(exc).__name__})."

        mcp.streamable_http_app()  # builds the session manager
        self._mcp = mcp
        self.route = Route("/mcp", endpoint=StreamableHTTPASGIApp(mcp.session_manager))
        self._task: asyncio.Task | None = None
        self._stop: asyncio.Event | None = None

    def _context_of(self, ctx) -> DispatchContext | None:
        try:
            request = ctx.request_context.request
        except Exception:  # noqa: BLE001 — no request in scope
            return None
        token = _bearer(request)
        if token is None:
            return None
        return self._sam.dispatch_context(token)

    async def start(self) -> None:
        if self._task is not None:
            return
        ready = asyncio.Event()
        self._stop = asyncio.Event()

        async def _run():
            async with self._mcp.session_manager.run():
                ready.set()
                await self._stop.wait()

        self._task = asyncio.create_task(_run(), name="orbital-dispatch-mcp")
        await ready.wait()

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop.set()
        try:
            await self._task
        except Exception:  # noqa: BLE001 — shutdown must not raise
            logger.warning("dispatch MCP server stopped with an error", exc_info=True)
        self._task = None


class BoundPortProbe:
    """ASGI middleware: the first request tells the daemon which port it is
    bound to, so the dispatch URL never has to be guessed.

    ``AGENT_OS_PORT`` is what the packaged app and the scripts export; a
    daemon started any other way (``python -m uvicorn … --port N``) used to
    attach no dispatch endpoint at all, and a pinned worker then quietly ran
    the other agent's CLI from its shell (batch 1110, item 7). uvicorn puts
    the bound ``(host, port)`` in every request scope, and no worker is ever
    spawned before an HTTP request has arrived, so the first request pins
    the URL exactly. A wrong default (another daemon on 8000) would be worse
    than none, so there is no default."""

    def __init__(self, app, sub_agent_manager) -> None:
        self._app = app
        self._sam = sub_agent_manager

    async def __call__(self, scope, receive, send):
        if (self._sam._dispatch_url is None
                and scope.get("type") in ("http", "websocket")):
            server = scope.get("server")
            port = server[1] if isinstance(server, (tuple, list)) and len(server) > 1 else None
            if isinstance(port, int) and port > 0:
                self._sam._dispatch_url = f"http://127.0.0.1:{port}/mcp"
                logger.info("dispatch MCP endpoint resolved from the bound port: %s",
                            self._sam._dispatch_url)
        await self._app(scope, receive, send)


def attach_dispatch_server(app, sub_agent_manager) -> DispatchServer:
    """Mount ``/mcp`` on ``app`` and tell the sub-agent manager its URL.

    The URL comes from ``AGENT_OS_PORT`` when set; otherwise from the port
    the first request shows the daemon is bound to (``BoundPortProbe``)."""
    server = DispatchServer(sub_agent_manager)
    app.router.routes.append(server.route)
    app.on_event("startup")(server.start)
    app.on_event("shutdown")(server.stop)
    url = dispatch_url()
    if url is not None:
        sub_agent_manager._dispatch_url = url
    else:
        sub_agent_manager._dispatch_url = None
        app.add_middleware(BoundPortProbe, sub_agent_manager=sub_agent_manager)
        logger.info("AGENT_OS_PORT is not set: the dispatch MCP URL is taken from "
                    "the bound port on the first request")
    return server
