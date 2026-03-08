# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: approval events must be scoped per project, not global.

Bug: When a user switches from project A's chat to project B's chat,
approval cards from project A remained visible in project B's panel.

Root cause (frontend): React reused the same ChatView component instance
across project switches because there was no `key` prop. The `approvals`
state Map was never cleared, so stale approvals from the previous project
rendered in the new project's chat.

Root cause (frontend subscription): ChatView called subscribe([projectId])
which overwrote App.tsx's broader subscribe(allProjectIds), narrowing the
WebSocket subscription on the backend to a single project.

Fix:
1. Added key={project_id} to <ChatView> so React remounts on project switch
2. Removed ChatView's redundant subscribe() call

This test verifies the backend WebSocket manager correctly scopes approval
events per project subscription — ensuring the backend was never the source
of cross-project event bleed.
"""

from unittest.mock import MagicMock, patch

import pytest

from agent_os.api.ws import WebSocketManager


@pytest.fixture
def ws_manager():
    """WebSocketManager with async queue disabled for sync testing."""
    mgr = WebSocketManager()
    # Prevent async queue creation so broadcast uses the sync fallback path
    mgr._ensure_drain = lambda: None
    return mgr


def _make_ws():
    """Create a mock WebSocket that records sent payloads synchronously."""
    ws = MagicMock()
    ws._sent = []
    ws.send_json = MagicMock(side_effect=lambda payload: ws._sent.append(payload))
    return ws


def test_approval_event_only_reaches_subscribed_project(ws_manager):
    """Approval events must only be delivered to clients subscribed to that project."""
    ws_a = _make_ws()
    ws_b = _make_ws()

    ws_manager.connect(ws_a)
    ws_manager.connect(ws_b)

    ws_manager.subscribe(ws_a, ["proj_alpha"])
    ws_manager.subscribe(ws_b, ["proj_beta"])

    approval_event = {
        "type": "approval.request",
        "project_id": "proj_alpha",
        "tool_name": "write_file",
        "tool_call_id": "tc_001",
        "tool_args": {"path": "/tmp/test"},
        "what": "Agent wants to run: write_file",
    }

    ws_manager.broadcast("proj_alpha", approval_event)

    # Client subscribed to proj_alpha receives the event
    assert len(ws_a._sent) == 1
    assert ws_a._sent[0]["type"] == "approval.request"
    assert ws_a._sent[0]["project_id"] == "proj_alpha"

    # Client subscribed to proj_beta does NOT receive it
    assert len(ws_b._sent) == 0


def test_approval_event_not_delivered_to_unsubscribed_client(ws_manager):
    """A client with no subscription must not receive approval events."""
    ws_connected = _make_ws()
    ws_manager.connect(ws_connected)
    # No subscribe call — empty subscription set

    ws_manager.broadcast("proj_alpha", {
        "type": "approval.request",
        "project_id": "proj_alpha",
        "tool_name": "bash",
        "tool_call_id": "tc_002",
        "tool_args": {},
        "what": "Agent wants to run: bash",
    })

    assert len(ws_connected._sent) == 0


def test_multi_project_subscriber_receives_correct_events(ws_manager):
    """A client subscribed to multiple projects receives events for each."""
    ws_both = _make_ws()
    ws_manager.connect(ws_both)
    ws_manager.subscribe(ws_both, ["proj_alpha", "proj_beta"])

    ws_manager.broadcast("proj_alpha", {
        "type": "approval.request",
        "project_id": "proj_alpha",
        "tool_call_id": "tc_a",
        "tool_name": "write_file",
        "tool_args": {},
        "what": "write_file",
    })
    ws_manager.broadcast("proj_beta", {
        "type": "approval.request",
        "project_id": "proj_beta",
        "tool_call_id": "tc_b",
        "tool_name": "bash",
        "tool_args": {},
        "what": "bash",
    })

    assert len(ws_both._sent) == 2
    assert ws_both._sent[0]["project_id"] == "proj_alpha"
    assert ws_both._sent[1]["project_id"] == "proj_beta"


def test_subscribe_replaces_previous_subscription(ws_manager):
    """Calling subscribe replaces the previous subscription set."""
    ws = _make_ws()
    ws_manager.connect(ws)

    # Initially subscribed to both
    ws_manager.subscribe(ws, ["proj_alpha", "proj_beta"])

    # Re-subscribe to only proj_beta
    ws_manager.subscribe(ws, ["proj_beta"])

    ws_manager.broadcast("proj_alpha", {
        "type": "approval.request",
        "project_id": "proj_alpha",
        "tool_call_id": "tc_x",
        "tool_name": "write_file",
        "tool_args": {},
        "what": "write_file",
    })

    # Should NOT receive proj_alpha event after re-subscribing to only proj_beta
    assert len(ws._sent) == 0

    ws_manager.broadcast("proj_beta", {
        "type": "approval.resolved",
        "project_id": "proj_beta",
        "tool_call_id": "tc_y",
        "resolution": "approved",
    })

    assert len(ws._sent) == 1
    assert ws._sent[0]["project_id"] == "proj_beta"
