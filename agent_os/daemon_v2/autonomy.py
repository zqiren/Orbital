# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Autonomy interceptor — implements ToolInterceptor protocol from Component A.

Preset-based tool interception with approval bypass window.
"""

import hashlib
import json
import time

from agent_os.agent.prompt_builder import Autonomy
from agent_os.daemon_v2.activity_translator import _describe_tool

BROWSER_WRITE_ACTIONS = frozenset({
    "click", "type", "fill", "press", "hover", "select",
    "drag", "upload_file", "evaluate",
})


class AutonomyInterceptor:
    """Implements ToolInterceptor protocol from Component A."""

    def __init__(self, preset: Autonomy, ws_manager, project_id: str,
                 user_credential_store=None):
        self._preset = preset
        self._ws = ws_manager
        self._project_id = project_id
        self._user_credential_store = user_credential_store
        self._recent_approvals: dict[str, float] = {}  # hash(tool+args) -> timestamp
        self._bypass_window = 60  # seconds
        self._pending_approvals: dict[str, dict] = {}   # tool_call_id -> {tool_name, tool_args}
        self._bypass_all_until: float | None = None  # epoch timestamp, None = inactive

    def _hash_tool(self, tool_name: str, tool_args: dict) -> str:
        raw = tool_name + json.dumps(tool_args, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _is_bypassed(self, tool_name: str, tool_args: dict) -> bool:
        h = self._hash_tool(tool_name, tool_args)
        ts = self._recent_approvals.get(h)
        if ts is None:
            return False
        return (time.time() - ts) < self._bypass_window

    def should_intercept(self, tool_call: dict) -> bool:
        """Check if tool call should be intercepted based on autonomy preset.

        Autonomy.HANDS_OFF:  intercept only request_access
        Autonomy.CHECK_IN:   intercept shell, write (non-workspace paths), request_access
        Autonomy.SUPERVISED: intercept all except read

        Skip if tool+args hash was approved within bypass_window.
        If an internal error occurs, let it propagate -- the loop treats
        any exception from should_intercept() as DENY (fail-closed).
        """
        name = tool_call.get("name", "")
        args = tool_call.get("arguments", {})

        # Credentials always require human interaction, regardless of autonomy preset
        if name == "request_credential":
            return True

        # Check approve-all bypass (time-bounded session-level override)
        if self._bypass_all_until is not None and time.time() < self._bypass_all_until:
            return False

        # Check per-action bypass window
        if self._is_bypassed(name, args):
            return False

        # Browser tool: action-level interception
        if name == "browser":
            action = args.get("action", "")
            if self._preset == Autonomy.HANDS_OFF:
                return False
            if self._preset == Autonomy.CHECK_IN:
                return action in BROWSER_WRITE_ACTIONS
            if self._preset == Autonomy.SUPERVISED:
                return action not in ("snapshot", "screenshot")
            return False

        if self._preset == Autonomy.HANDS_OFF:
            return name == "request_access"

        if self._preset == Autonomy.CHECK_IN:
            return name in ("shell", "request_access", "write")

        if self._preset == Autonomy.SUPERVISED:
            return name != "read"

        return False

    def on_intercept(self, tool_call: dict, recent_context: list[dict], reasoning: str | None = None) -> None:
        """Broadcast approval.request via WebSocket and store pending."""
        tool_call_id = tool_call.get("id", "")
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("arguments", {})

        payload = {
            "type": "approval.request",
            "project_id": self._project_id,
            "what": _describe_tool(tool_name, tool_args),
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "tool_args": tool_args,
            "recent_activity": recent_context,
        }
        if reasoning and reasoning.strip():
            payload["reasoning"] = reasoning.strip()

        # Store full payload so the REST recovery endpoint can return it
        self._pending_approvals[tool_call_id] = {
            "tool_name": tool_name,
            "tool_args": tool_args,
            "what": payload["what"],
            "tool_call_id": tool_call_id,
            "recent_activity": recent_context,
        }
        if "reasoning" in payload:
            self._pending_approvals[tool_call_id]["reasoning"] = payload["reasoning"]

        self._ws.broadcast(self._project_id, payload)

    def get_pending(self, tool_call_id: str) -> dict | None:
        """Return the pending approval info for a tool_call_id, or None."""
        return self._pending_approvals.get(tool_call_id)

    def remove_pending(self, tool_call_id: str) -> None:
        """Remove a tool_call_id from pending approvals."""
        self._pending_approvals.pop(tool_call_id, None)

    def activate_bypass_all(self, duration: float = 600) -> None:
        """Activate approve-all bypass for the given duration (default 10 min).

        All subsequent tool calls will be auto-approved until the time expires,
        the user sends a new message, or deactivate_bypass_all() is called.
        """
        self._bypass_all_until = time.time() + duration

    def deactivate_bypass_all(self) -> None:
        """Deactivate approve-all bypass (called on new user message)."""
        self._bypass_all_until = None

    def record_approval(self, tool_name: str, tool_args: dict) -> None:
        """Record hash for bypass window. Called by agent_manager.approve()."""
        h = self._hash_tool(tool_name, tool_args)
        self._recent_approvals[h] = time.time()
