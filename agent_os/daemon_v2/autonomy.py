# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Autonomy interceptor — implements ToolInterceptor protocol from Component A.

Preset-based tool interception with approval bypass window.
"""

import hashlib
import json
import time
from typing import Callable
from uuid import uuid4

from agent_os.agent.prompt_builder import Autonomy
from agent_os.daemon_v2.activity_translator import _describe_tool

# Spec 117 D4/D5 — the pending-approval kind raised when the sandbox blocked a
# shell command in safe mode. It rides the SAME pending registry, WS event
# (``approval.request``) and routes (``/approve`` with ``response_payload``
# ``once`` | ``always`` | ``disable_safe_mode``, ``/deny``) as every other
# approval; ``kind`` is what tells the card apart. The pending's
# ``tool_call_id`` is SYNTHETIC (``unsandboxed_<hex>``): the shell call that
# triggered it already has its tool result (the ``[sandbox]`` text), so the
# answer is injected as a terminal system row, never as a second tool result.
UNSANDBOXED_RUN_KIND = "unsandboxed_run"
UNSANDBOXED_RUN_DECISIONS = ("once", "always", "disable_safe_mode")

BROWSER_WRITE_ACTIONS = frozenset({
    "click", "type", "fill", "press", "hover", "select",
    "drag", "upload_file",
})

# Hands-off runs must never hang on a human. A network ask that goes
# unanswered this long auto-denies FOR THE RUN; the request itself
# survives in the project's pending_domain_requests (Settings → Network).
NETWORK_APPROVAL_TIMEOUT_S = 180


class AutonomyInterceptor:
    """Implements ToolInterceptor protocol from Component A."""

    def __init__(self, preset: Autonomy, ws_manager, project_id: str,
                 user_credential_store=None, *, session_id: str | None = None,
                 safe_mode_cb: Callable[[], bool] | None = None,
                 connector_writes: frozenset[str] = frozenset()):
        self._preset = preset
        # Connectors shared across agents: the names of this loop's
        # write-classified connector tools (``ReflectedConnectorTool.is_write``).
        # One rule with the workers': ask under check-in and supervised, run
        # under hands-off. Reads are never asked about because of it.
        self._connector_writes = frozenset(connector_writes or ())
        self._ws = ws_manager
        self._project_id = project_id
        # Spec 117 D3: read per call. On an open project the network ask is
        # not intercepted — the tool itself answers "unrestricted" (no card).
        self._safe_mode_cb = safe_mode_cb
        # Seam 3 / Phase 2: the canonical session id this project handle runs
        # under, stamped on the approval.request broadcast so the frontend can
        # route it by session_id (management approvals previously carried no
        # session_id — the interceptor only knew the project).
        self._session_id = session_id
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

        # Credentials and network-domain grants always require human
        # interaction, regardless of autonomy preset
        if name == "request_credential":
            return True
        if name == "request_network_access":
            # Spec 117 D3: nothing to grant on an open project.
            return self._safe_mode_on()

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
            # glob and grep are strictly read-only, treat them like read.
            # A connector write (send a mail, create an event) is outward-facing.
            return (name in ("shell", "request_access", "write")
                    or name in self._connector_writes)

        if self._preset == Autonomy.SUPERVISED:
            # glob and grep are read-only; auto-approve alongside read.
            return name not in ("read", "glob", "grep")

        return False

    def on_intercept(self, tool_call: dict, recent_context: list[dict], reasoning: str | None = None) -> None:
        """Broadcast approval.request via WebSocket and store pending."""
        tool_call_id = tool_call.get("id", "")
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("arguments", {})

        payload = {
            "type": "approval.request",
            "project_id": self._project_id,
            "session_id": self._session_id,
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

    def _safe_mode_on(self) -> bool:
        if self._safe_mode_cb is None:
            return True
        try:
            return bool(self._safe_mode_cb())
        except Exception:
            return True  # an unreadable switch fails closed

    def on_sandbox_denial(self, tool_call: dict, denial: dict,
                          recent_context: list[dict]) -> str | None:
        """Raise an ``unsandboxed_run`` approval for a blocked shell result.

        Spec 117 D4: called by the loop's tool-result path when a safe-mode
        shell result carries ``meta["sandbox_denial"]`` (the daemon's own
        denial detection). Registers ONE pending under a synthetic id,
        broadcasts the usual ``approval.request`` with ``kind``, ``command``
        and ``reason`` added, and returns the pending id (the loop then
        pauses on it exactly as on an intercepted call). Returns ``None``
        when the project is open — belt and braces; the open shell path
        never emits the meta.
        """
        if not self._safe_mode_on():
            return None
        command = str((denial or {}).get("command") or
                      (tool_call.get("arguments") or {}).get("command") or "")
        reason = str((denial or {}).get("reason") or "The sandbox blocked this command.")
        pending_id = f"unsandboxed_{uuid4().hex[:12]}"
        tool_args = {"command": command}
        # The program "always allow" would record ("" = not allowlistable),
        # so the card never offers to allowlist ``cmd`` itself.
        from agent_os.agent.tools.shell import ShellTool
        executable = ShellTool.executable_of(command)
        pending = {
            "tool_name": "shell",
            "tool_args": tool_args,
            "what": f"Run outside the sandbox: {command}",
            "tool_call_id": pending_id,
            "recent_activity": recent_context,
            "kind": UNSANDBOXED_RUN_KIND,
            "command": command,
            "reason": reason,
            "executable": executable,
            "session_id": self._session_id,
            "origin_tool_call_id": tool_call.get("id", ""),
        }
        self._pending_approvals[pending_id] = pending
        self._ws.broadcast(self._project_id, {
            "type": "approval.request",
            "project_id": self._project_id,
            "session_id": self._session_id,
            "what": pending["what"],
            "tool_name": "shell",
            "tool_call_id": pending_id,
            "tool_args": tool_args,
            "recent_activity": recent_context,
            "kind": UNSANDBOXED_RUN_KIND,
            "command": command,
            "reason": reason,
            "executable": executable,
        })
        return pending_id

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

    def update_preset(self, preset: Autonomy) -> None:
        """Change the autonomy preset at runtime (live settings update)."""
        self._preset = preset

    def record_approval(self, tool_name: str, tool_args: dict) -> None:
        """Record hash for bypass window. Called by agent_manager.approve()."""
        h = self._hash_tool(tool_name, tool_args)
        self._recent_approvals[h] = time.time()
