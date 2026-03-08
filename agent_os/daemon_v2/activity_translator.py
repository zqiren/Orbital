# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Session events -> WebSocket events translator.

Translates session messages to agent.activity and chat.stream_delta WS events.
"""

import json
import re
from datetime import datetime, timezone
from uuid import uuid4

_STATUS_RE = re.compile(r'\[STATUS:\s*(.+?)\]')
_SENSITIVE_PATH_RE = re.compile(
    r'[A-Za-z]:\\Users\\|/home/|/Users/|%USERPROFILE%|%APPDATA%|%LOCALAPPDATA%|\$HOME|\$USERPROFILE',
    re.IGNORECASE,
)


_TOOL_CATEGORY_MAP = {
    "read": "file_read",
    "write": "file_write",
    "edit": "file_edit",
    "shell": "command_exec",
    "request_access": "request_access",
    "agent_message": "agent_message",
    "browser": "browser_automation",
}

_BROWSER_ACTIVITY_MAP = {
    "navigate": lambda args: f"Navigating to {args.get('url', 'unknown')}",
    "click": lambda args: f"Clicking element {args.get('ref', '?')}",
    "type": lambda args: f"Typing into element {args.get('ref', '?')}",
    "fill": lambda args: f"Filling {len(args.get('fields', []))} form fields",
    "press": lambda args: f"Pressing {args.get('key', '?')}",
    "hover": lambda args: f"Hovering over element {args.get('ref', '?')}",
    "select": lambda args: f"Selecting '{args.get('value', '?')}' in element {args.get('ref', '?')}",
    "scroll": lambda args: f"Scrolling {args.get('direction', 'down')}",
    "drag": lambda args: "Dragging element",
    "upload_file": lambda args: "Uploading file",
    "snapshot": lambda args: "Reading page content",
    "screenshot": lambda args: "Taking screenshot",
    "extract": lambda args: f"Extracting: {args.get('text', '?')[:50]}",
    "search_page": lambda args: f"Searching page for '{args.get('text', '?')[:30]}'",
    "evaluate": lambda args: "Running script on page",
    "go_back": lambda args: "Going back",
    "go_forward": lambda args: "Going forward",
    "reload": lambda args: "Reloading page",
    "wait": lambda args: "Waiting for page",
    "pdf": lambda args: "Generating PDF",
    "tab_new": lambda args: "Opening new tab",
    "tab_switch": lambda args: "Switching tab",
    "tab_close": lambda args: "Closing tab",
    "done": lambda args: "Browser task complete",
    "search": lambda args: f"Searching web for '{args.get('query', '?')[:50]}'",
    "fetch": lambda args: f"Fetching {args.get('url', '?')[:60]}",
}

_TOOL_CATEGORY_MAP["request_credential"] = "credential_request"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _describe_tool(tool_name: str, args: dict) -> str:
    """Build human-readable description from tool name and arguments."""
    if tool_name == "read":
        return f"Reading {args.get('path', 'file')}"
    if tool_name == "write":
        return f"Writing {args.get('path', 'file')}"
    if tool_name == "edit":
        return f"Editing {args.get('path', 'file')}"
    if tool_name == "shell":
        cmd = args.get("command", "")
        if _SENSITIVE_PATH_RE.search(cmd):
            return "Running: shell command"
        return f"Running: {cmd[:80]}"
    if tool_name == "browser":
        action = args.get("action", "unknown")
        mapper = _BROWSER_ACTIVITY_MAP.get(action)
        if mapper:
            return mapper(args)
        return f"Browser: {action}"
    if tool_name == "request_credential":
        return f"Requesting credentials for {args.get('domain', 'website')}"
    return f"Using {tool_name}"


class ActivityTranslator:
    def __init__(self, ws_manager):
        self._ws = ws_manager
        self._last_status: dict[str, str] = {}  # project_id -> last status summary
        self._stream_seq: dict[str, int] = {}  # project_id -> monotonic seq counter

    def _extract_status(self, content: str) -> str | None:
        """Extract [STATUS: ...] from agent output."""
        match = _STATUS_RE.search(content)
        return match.group(1).strip() if match else None

    def get_last_status(self, project_id: str) -> str | None:
        """Return the last extracted status summary for a project."""
        return self._last_status.get(project_id)

    def on_message(self, message: dict, project_id: str) -> None:
        """Translate session messages to WS events."""
        role = message.get("role")
        source = message.get("source", "management")

        # Extract status summary from assistant messages
        if role == "assistant":
            content = message.get("content") or ""
            status = self._extract_status(content)
            if status:
                self._last_status[project_id] = status
                self._ws.broadcast(project_id, {
                    "type": "agent.status_summary",
                    "project_id": project_id,
                    "summary": status,
                    "timestamp": _now(),
                })

        if role == "assistant" and "tool_calls" in message:
            descriptions = {}
            for tc in message["tool_calls"]:
                # Handle both nested and flat formats
                if "function" in tc:
                    func = tc["function"]
                    tool_name = func.get("name", "unknown")
                    raw_args = func.get("arguments", "{}")
                else:
                    tool_name = tc.get("name", "unknown")
                    raw_args = tc.get("arguments", "{}")

                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except (json.JSONDecodeError, ValueError):
                        args = {}
                else:
                    args = raw_args

                tc_id = tc.get("id", "")
                description = _describe_tool(tool_name, args)

                if tc_id:
                    descriptions[tc_id] = description

                category = _TOOL_CATEGORY_MAP.get(tool_name, "tool_use")

                self._ws.broadcast(project_id, {
                    "type": "agent.activity",
                    "project_id": project_id,
                    "id": uuid4().hex,
                    "category": category,
                    "description": description,
                    "tool_name": tool_name,
                    "source": source,
                    "timestamp": _now(),
                })

            # Persist descriptions on the message dict (written to JSONL by session)
            if descriptions:
                message["_activity_descriptions"] = descriptions

        elif role == "tool":
            self._ws.broadcast(project_id, {
                "type": "agent.activity",
                "project_id": project_id,
                "id": uuid4().hex,
                "category": "tool_result",
                "description": "Tool result received",
                "tool_name": message.get("tool_call_id", "unknown"),
                "source": source,
                "timestamp": _now(),
            })

        elif role == "user":
            self._ws.broadcast(project_id, {
                "type": "chat.user_message",
                "project_id": project_id,
                "content": message.get("content", ""),
                "nonce": message.get("nonce", ""),
                "timestamp": message.get("timestamp") or _now(),
            })

        elif role == "agent":
            self._ws.broadcast(project_id, {
                "type": "agent.activity",
                "project_id": project_id,
                "id": uuid4().hex,
                "category": "agent_output",
                "description": (message.get("content", "") or "")[:100],
                "tool_name": "",
                "source": source,
                "timestamp": _now(),
            })

    def on_stream_chunk(self, chunk, project_id: str, source: str) -> None:
        """Broadcast chat.stream_delta with monotonic seq number."""
        is_final = getattr(chunk, "is_final", False)

        seq = self._stream_seq.get(project_id, 0) + 1
        self._stream_seq[project_id] = seq

        self._ws.broadcast(project_id, {
            "type": "chat.stream_delta",
            "project_id": project_id,
            "text": getattr(chunk, "text", ""),
            "source": source,
            "is_final": is_final,
            "seq": seq,
        })

        # Reset counter after final delta so next response starts at 1
        if is_final:
            self._stream_seq[project_id] = 0

    def on_network_blocked(self, project_id: str, domain: str, method: str) -> None:
        """Broadcast network_blocked event from platform provider."""
        self._ws.broadcast(project_id, {
            "type": "agent.activity",
            "project_id": project_id,
            "id": uuid4().hex,
            "category": "network_blocked",
            "description": f"Blocked {method} request to {domain}",
            "tool_name": "",
            "source": "platform",
            "timestamp": _now(),
        })
