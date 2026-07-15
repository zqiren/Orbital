# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression (Root A / seam 3): the @mention inject route must forward the
chat session id to the sub-agent manager's send()/start(), and stamp the
RESOLVED session (never None) on the ack broadcast and the lifecycle marker.

Before the fix the route called send()/start() WITHOUT session_id, so send()'s
sub-agent resolver hard-raised on None and POST /agents/{pid}/inject returned
404 "No active session for project" even for a running sub-agent and even when
the client forwarded session_id. Fix is caller-side: forward the session id the
route already has (req.session_id, or the persisted chat session as a
session-less fallback) — NOT a loosening of the sub-agent resolver.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from agent_os.api.routes import agents_v2
from agent_os.api.routes.agents_v2 import InjectRequest


def _wire(monkeypatch, *, send_result="delivered"):
    project_store = MagicMock()
    project_store.get_project.return_value = {
        "workspace": "/tmp/ws", "project_id": "proj_x",
    }
    sub_agent_manager = MagicMock()
    sub_agent_manager.send = AsyncMock(return_value=send_result)
    sub_agent_manager.start = AsyncMock(return_value="started")
    sub_agent_manager.get_transcript.return_value = None
    ws_manager = MagicMock()
    ws_manager.broadcast = MagicMock()
    agent_manager = MagicMock()
    # The @mention path now persists via the canonical resolver
    # AgentManager.persist_mention_message(project_id, session_id, user_msg),
    # which returns the ONE concrete session id threaded to dispatch + ack +
    # lifecycle. Mirror its resolution: passthrough a client-supplied id, else
    # the project's persistent chat session (here a fixed funnel id).
    agent_manager.persist_mention_message = MagicMock(
        side_effect=lambda pid, sid, msg: sid or "proj_x_chatfunnel"
    )
    lifecycle_observer = MagicMock()
    lifecycle_observer.on_message_routed = AsyncMock()
    agents_v2.configure(
        project_store, agent_manager, ws_manager, sub_agent_manager,
        lifecycle_observer=lifecycle_observer,
    )
    return sub_agent_manager, ws_manager, lifecycle_observer


@pytest.mark.asyncio
async def test_mention_forwards_client_session_id_to_send(monkeypatch):
    sam, ws, lifecycle = _wire(monkeypatch)
    req = InjectRequest(content="hi", target="researcher", session_id="proj_x_sessA")

    result = await agents_v2.inject_message("proj_x", req)

    assert result == {"status": "delivered"}
    # send() receives the forwarded chat session id (not dropped to None)
    sam.send.assert_awaited_once()
    assert sam.send.await_args.kwargs.get("session_id") == "proj_x_sessA"
    # ack broadcast carries the resolved session, not None
    ack_payload = ws.broadcast.call_args.args[1]
    assert ack_payload["session_id"] == "proj_x_sessA"
    # lifecycle transcript marker carries the resolved session
    assert lifecycle.on_message_routed.await_args.kwargs.get("session_id") == "proj_x_sessA"
    # arg-threading guard (persist_mention_message is mocked here): the route must
    # hand the resolver (project_id, req.session_id, authored user_msg) so an
    # arg-threading regression is caught even though the resolver is stubbed.
    pm = agents_v2._agent_manager.persist_mention_message
    pm.assert_called_once()
    call_pid, call_sid, call_msg = pm.call_args.args
    assert (call_pid, call_sid) == ("proj_x", "proj_x_sessA")
    assert call_msg["role"] == "user" and call_msg["target"] == "researcher"
    assert call_msg["content"] == "hi"


@pytest.mark.asyncio
async def test_mention_spawn_on_demand_send_forwards_session_id(monkeypatch):
    """send() spawns-on-demand inside the manager now
    (TASK-collapse-dispatch-to-send): the route makes ONE send call — no
    manual auto-start/retry — and that call must carry the session id. A
    spawn failure surfaces as an HTTP error, never a silent drop."""
    from fastapi import HTTPException

    sam, ws, lifecycle = _wire(monkeypatch)
    req = InjectRequest(content="go", target="researcher", session_id="proj_x_sessB")

    result = await agents_v2.inject_message("proj_x", req)

    assert result == {"status": "delivered"}
    sam.start.assert_not_awaited()  # spawn is internal to the manager now
    sam.send.assert_awaited_once()
    assert sam.send.await_args.kwargs.get("session_id") == "proj_x_sessB"

    # Spawn/dispatch failure path: honest HTTP error, no retry dance.
    sam.send = AsyncMock(return_value="Error: unknown agent 'researcher'")
    with pytest.raises(HTTPException):
        await agents_v2.inject_message("proj_x", req)


@pytest.mark.asyncio
async def test_mention_threads_same_dispatch_id_to_send_and_lifecycle_marker(monkeypatch):
    """TASK-dispatch-id-pairing: the @mention route calls send() (which
    stamps its own 'management_agent'-flavored marker internally) AND fires
    its own 'user_mention'-flavored marker for the same physical dispatch.
    Both describe the SAME turn, so both must carry the identical
    dispatch_id — otherwise the two markers would incorrectly join to
    different (or no) transcript turns."""
    sam, ws, lifecycle = _wire(monkeypatch)
    req = InjectRequest(content="hi", target="researcher", session_id="proj_x_sessA")

    await agents_v2.inject_message("proj_x", req)

    # send() must receive a dispatch_id (the route mints one up front so it
    # can reuse it for its own marker below).
    send_kwargs = sam.send.await_args.kwargs
    assert send_kwargs.get("dispatch_id"), "route must pass a dispatch_id into send()"

    lifecycle.on_message_routed.assert_awaited_once()
    routed_kwargs = lifecycle.on_message_routed.await_args.kwargs
    assert routed_kwargs.get("dispatch_id") == send_kwargs.get("dispatch_id")


@pytest.mark.asyncio
async def test_mention_session_less_resolves_to_chat_session_not_raise(monkeypatch):
    # No session_id from the client: must resolve to the persisted chat session
    # (a concrete uuid), never forward None into the hard-raising resolver.
    sam, ws, lifecycle = _wire(monkeypatch)
    req = InjectRequest(content="hi", target="researcher")  # session_id omitted

    result = await agents_v2.inject_message("proj_x", req)

    assert result == {"status": "delivered"}
    assert sam.send.await_args.kwargs.get("session_id") == "proj_x_chatfunnel"
    assert ws.broadcast.call_args.args[1]["session_id"] == "proj_x_chatfunnel"
