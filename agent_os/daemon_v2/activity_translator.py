# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Session events -> WebSocket events translator.

Translates session messages to agent.activity and chat.stream_delta WS events.
"""

import asyncio
import json
import re
import time
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
    "glob": "file_search",
    "grep": "content_search",
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

# The chat capsule shows at most this much of a tool result (the frontend's
# `truncateResult` bounds in web/src/utils/chatTransform.ts). The live
# tool_result event carries only that part, so a huge result costs the relay
# and phones nothing extra.
_RESULT_CHAR_BOUND = 500
_RESULT_LINE_BOUND = 12


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _result_preview(content) -> dict:
    """The live-event slice of a tool result: exactly what the capsule shows.

    Mirrors `truncateResult`'s cut, so the frontend renders the preview the
    same way it renders the full result after a reload. When the result is
    cut, the full result's totals ride along for the capsule footer — chars
    counted in UTF-16 units, as the browser counts them. Non-text results
    (multimodal lists) render as empty after a reload, so they are empty here.
    """
    if not isinstance(content, str):
        return {"result_preview": ""}
    lines = content.split("\n")
    if len(content) <= _RESULT_CHAR_BOUND and len(lines) <= _RESULT_LINE_BOUND:
        return {"result_preview": content}
    first_lines = "\n".join(lines[:_RESULT_LINE_BOUND])
    char_bound_fires = len(content) > _RESULT_CHAR_BOUND
    line_bound_fires = len(lines) > _RESULT_LINE_BOUND
    if char_bound_fires and (not line_bound_fires or _RESULT_CHAR_BOUND <= len(first_lines)):
        preview = content[:_RESULT_CHAR_BOUND]
    else:
        preview = first_lines
    return {
        "result_preview": preview,
        "result_total_chars": len(content.encode("utf-16-le", "surrogatepass")) // 2,
        "result_total_lines": len(lines),
    }


# Spec 100: a worker's tool arguments ride the WS event and its transcript
# row. The capsule reads only a path/command/query out of them, so long strings
# (a Write's whole file, a diff) and long lists are cut.
_ARG_STRING_BOUND = 1000
_ARG_ITEMS_BOUND = 50
_ARG_DEPTH_BOUND = 4
# Worker thinking deltas are coalesced to at most one broadcast per interval
# per worker: each frame is also a relay forward to mobile.
_WORKER_THINKING_INTERVAL_S = 0.25


def _cap_value(value, depth: int = 0):
    if isinstance(value, str):
        return value if len(value) <= _ARG_STRING_BOUND else value[:_ARG_STRING_BOUND] + "…"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if depth >= _ARG_DEPTH_BOUND:
        return _cap_value(str(value), depth)
    if isinstance(value, dict):
        return {str(k): _cap_value(v, depth + 1)
                for k, v in list(value.items())[:_ARG_ITEMS_BOUND]}
    if isinstance(value, (list, tuple)):
        return [_cap_value(v, depth + 1) for v in list(value)[:_ARG_ITEMS_BOUND]]
    return _cap_value(str(value), depth)


def worker_display_meta(chunk_type: str, metadata: dict | None) -> dict:
    """Spec 100: the display slice of a worker chunk's transport metadata.

    One vocabulary over every transport's tool event shape, written to the
    worker transcript row and broadcast live, so the live capsule and the
    reloaded one read the same fields:

    - ``tool_activity`` / ``tool_result``: ``tool_call_id`` (SDK/codex/pi
      ``tool_id``, ACP ``tool_call_id``), ``tool_name``, capped ``arguments``,
      ``status``; plus, when the event carries a result (``tool_result``'s
      ``content``, or a completion's ``result``), the capsule's preview of it
      (``_result_preview``) and ``is_error``.
    - ``thinking``: ``delta`` when the text is a stream fragment.

    Display-only: never read by the manager's LLM context.
    """
    meta = metadata or {}
    out: dict = {}
    if chunk_type in ("tool_activity", "tool_result"):
        tool_call_id = meta.get("tool_call_id") or meta.get("tool_id")
        if tool_call_id:
            out["tool_call_id"] = str(tool_call_id)
        if meta.get("tool_name"):
            out["tool_name"] = str(meta["tool_name"])
        args = meta.get("tool_input")
        if args is not None:
            if not isinstance(args, dict):
                args = {"input": args}
            if args:
                out["arguments"] = _cap_value(args)
        status = meta.get("status")
        if isinstance(status, str) and status:
            out["status"] = status
        if chunk_type == "tool_result" or "result" in meta:
            result = meta.get("content") if chunk_type == "tool_result" else meta.get("result")
            if result is not None and not isinstance(result, str):
                result = str(result)
            out.update(_result_preview(result or ""))
            out["is_error"] = bool(meta.get("is_error")) or status in ("failed", "error")
        return out
    if chunk_type == "thinking":
        return {"delta": True} if meta.get("delta") else {}
    return out


class _WorkerLive:
    """Per-(project, session, handle) live state for a worker's turn."""

    def __init__(self, project_id: str, session_id: str | None, handle: str):
        self.project_id = project_id
        self.session_id = session_id
        self.handle = handle
        # tool_call_id -> (tool_name, arguments) last broadcast for the row.
        self.calls: dict[str, tuple] = {}
        self.thinking = ""
        self.last_flush = 0.0
        self.timer: "asyncio.TimerHandle | None" = None
        self.last_kind = ""
        # Spec 195: the dispatch this turn belongs to and who started it,
        # stamped on every live event (absent on a self-initiated turn).
        self.dispatch_id: str | None = None
        self.initiator: str | None = None

    def stamp(self, event: dict) -> dict:
        if self.dispatch_id:
            event["dispatch_id"] = self.dispatch_id
            if self.initiator:
                event["initiator"] = self.initiator
        return event


def _describe_tool(tool_name: str, args: dict) -> str:
    """Build human-readable description from tool name and arguments."""
    if tool_name == "read":
        return f"Reading {args.get('path', 'file')}"
    if tool_name == "write":
        return f"Writing {args.get('path', 'file')}"
    if tool_name == "edit":
        return f"Editing {args.get('path', 'file')}"
    if tool_name == "glob":
        pattern = args.get("pattern", "?")
        path = args.get("path", "")
        if path and path != ".":
            return f"Searching for files matching '{pattern}' in {path}"
        return f"Searching for files matching '{pattern}'"
    if tool_name == "grep":
        pattern = args.get("pattern", "?")
        path = args.get("path", "")
        if path and path != ".":
            return f"Searching for '{pattern}' in {path}"
        return f"Searching for '{pattern}'"
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
        # Spec 100: live state of each running worker turn.
        self._workers: dict[tuple, _WorkerLive] = {}

    def _extract_status(self, content: str) -> str | None:
        """Extract [STATUS: ...] from agent output."""
        match = _STATUS_RE.search(content)
        return match.group(1).strip() if match else None

    def get_last_status(self, project_id: str) -> str | None:
        """Return the last extracted status summary for a project."""
        return self._last_status.get(project_id)

    def on_message(self, message: dict, project_id: str,
                   *, session_id: str | None = None) -> None:
        """Translate session messages to WS events.

        ``session_id`` is included in every emitted event so the frontend
        can attribute each event to the correct session. Without it,
        multi-session projects route by holder heuristic and race during
        holder resolution.
        """
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
                    "session_id": session_id,
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
                    "session_id": session_id,
                    "id": uuid4().hex,
                    "category": category,
                    "description": description,
                    "tool_name": tool_name,
                    # Parsed args let the frontend render a localized
                    # description; `description` stays for old frontends.
                    "arguments": args,
                    # Lets the frontend pair this row with its tool_result
                    # event by id instead of by position.
                    "tool_call_id": tc_id,
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
                "session_id": session_id,
                "id": uuid4().hex,
                "category": "tool_result",
                "description": "Tool result received",
                # Old frontends read the tool_call_id from here; keep it.
                "tool_name": message.get("tool_call_id", "unknown"),
                "tool_call_id": message.get("tool_call_id", ""),
                # Additive: the capped result, so a row expanded mid-turn
                # shows content instead of waiting for a reload.
                **_result_preview(message.get("content")),
                "source": source,
                "timestamp": _now(),
            })

        elif role == "user":
            self._ws.broadcast(project_id, {
                "type": "chat.user_message",
                "project_id": project_id,
                "session_id": session_id,
                "content": message.get("content", ""),
                "nonce": message.get("nonce", ""),
                "timestamp": message.get("timestamp") or _now(),
            })

        elif role == "agent":
            self._ws.broadcast(project_id, {
                "type": "agent.activity",
                "project_id": project_id,
                "session_id": session_id,
                "id": uuid4().hex,
                "category": "agent_output",
                "description": (message.get("content", "") or "")[:100],
                "tool_name": "",
                "source": source,
                "timestamp": _now(),
            })

    def on_stream_chunk(self, chunk, project_id: str, source: str,
                        *, session_id: str | None = None) -> None:
        """Broadcast chat.stream_delta with monotonic seq number."""
        is_final = getattr(chunk, "is_final", False)

        seq = self._stream_seq.get(project_id, 0) + 1
        self._stream_seq[project_id] = seq

        self._ws.broadcast(project_id, {
            "type": "chat.stream_delta",
            "project_id": project_id,
            # Seam 3 / Phase 2: stamp the canonical session id so the frontend
            # can route deltas strictly by session_id (it previously had only
            # the viewingHolder heuristic — stream deltas carried no id).
            "session_id": session_id,
            "text": getattr(chunk, "text", ""),
            # Reasoning is carried on every delta — including reasoning-only
            # deltas during the <think> phase, which have empty text but
            # non-empty reasoning_content. The frontend needs both so it can
            # keep the thinking indicator alive while the model reasons.
            # No empty-text guard: dropping reasoning-only deltas was the
            # "thinking is off / message hidden" symptom.
            "reasoning_content": getattr(chunk, "reasoning_content", ""),
            "source": source,
            "is_final": is_final,
            "seq": seq,
        })

        # Reset counter after final delta so next response starts at 1
        if is_final:
            self._stream_seq[project_id] = 0

    # ── Spec 100: a worker's tool calls and thinking, live ────────────────

    def on_worker_chunk(self, chunk_type: str, display: dict, text: str,
                        project_id: str, *, session_id: str | None,
                        handle: str, dispatch_id: str | None = None,
                        initiator: str | None = None) -> None:
        """Broadcast one worker ``tool_activity`` / ``tool_result`` /
        ``thinking`` chunk (``display`` = ``worker_display_meta``).

        Tool events are ``agent.activity`` frames with ``category:
        "agent_output"`` plus ``worker_event`` ("tool_call" | "tool_result"):
        a frontend that predates spec 100 drops ``agent_output`` on arrival,
        so an old build ignores them instead of rendering a worker's calls
        into the manager's capsule. Thinking rides ``chat.stream_delta``
        (reasoning only, ``worker: true``), coalesced per worker. Never
        ``chat.sub_agent_message``: the status bar refetches on each of those.
        """
        key = (project_id, session_id or "", handle)
        state = self._workers.get(key)
        if state is None:
            state = self._workers[key] = _WorkerLive(project_id, session_id, handle)
        if dispatch_id != state.dispatch_id:
            # Pending thinking belongs to the dispatch it was produced under.
            self._flush_worker_thinking(key)
            state.dispatch_id, state.initiator = dispatch_id, initiator
        if chunk_type == "thinking":
            if text:
                # Whole blocks back to back are separate paragraphs (the
                # transcript reader joins them the same way).
                if not display.get("delta") and state.last_kind == "thinking":
                    text = "\n\n" + text
                state.thinking += text
                state.last_kind = "thinking"
                self._schedule_worker_thinking(key, state)
            return
        # Keep the order the worker produced: pending thinking goes first.
        self._flush_worker_thinking(key)
        state.last_kind = chunk_type
        tool_call_id = display.get("tool_call_id") or ""
        known = state.calls.get(tool_call_id) if tool_call_id else None
        name = display.get("tool_name") or (known[0] if known else "")
        args = display.get("arguments") or (known[1] if known else None)
        # A new row, or an in-place update to a known one (an ACP progress or
        # a codex completion adding a title or arguments). A bare result for
        # a call never seen opens its row only when it names the tool.
        opens_row = chunk_type == "tool_activity" or bool(name) or known is not None
        if opens_row and (known is None or (name, args) != known):
            if tool_call_id:
                state.calls[tool_call_id] = (name, args)
            self._broadcast_worker_tool(
                state, "tool_call", name or "tool", args, tool_call_id)
        if "result_preview" in display:
            self._broadcast_worker_tool(
                state, "tool_result", name or "tool", None, tool_call_id,
                extra={k: display[k] for k in (
                    "result_preview", "result_total_chars",
                    "result_total_lines", "is_error") if k in display})

    def on_worker_turn_closed(self, project_id: str, *,
                              session_id: str | None, handle: str) -> None:
        """Flush a worker's pending thinking and drop its turn state."""
        key = (project_id, session_id or "", handle)
        self._flush_worker_thinking(key)
        self._workers.pop(key, None)

    def _broadcast_worker_tool(self, state: _WorkerLive, worker_event: str,
                               name: str, args, tool_call_id: str,
                               extra: dict | None = None) -> None:
        event = {
            "type": "agent.activity",
            "project_id": state.project_id,
            "session_id": state.session_id,
            "id": uuid4().hex,
            "category": "agent_output",
            "worker_event": worker_event,
            "description": (f"Using {name}" if worker_event == "tool_call"
                            else "Tool result received"),
            "tool_name": name,
            "tool_call_id": tool_call_id,
            "source": state.handle,
            "timestamp": _now(),
        }
        if args is not None:
            event["arguments"] = args
        if extra:
            event.update(extra)
        self._ws.broadcast(state.project_id, state.stamp(event))

    def _schedule_worker_thinking(self, key: tuple, state: _WorkerLive) -> None:
        elapsed = time.monotonic() - state.last_flush
        if elapsed >= _WORKER_THINKING_INTERVAL_S:
            self._flush_worker_thinking(key)
            return
        if state.timer is not None:
            return  # a flush is already due; this text rides it
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._flush_worker_thinking(key)  # no loop to defer on
            return
        state.timer = loop.call_later(
            _WORKER_THINKING_INTERVAL_S - elapsed,
            self._flush_worker_thinking, key)

    def _flush_worker_thinking(self, key: tuple) -> None:
        state = self._workers.get(key)
        if state is None:
            return
        if state.timer is not None:
            state.timer.cancel()
            state.timer = None
        if not state.thinking:
            return
        text, state.thinking = state.thinking, ""
        state.last_flush = time.monotonic()
        self._ws.broadcast(state.project_id, state.stamp({
            "type": "chat.stream_delta",
            "project_id": state.project_id,
            "session_id": state.session_id,
            "text": "",
            "reasoning_content": text,
            "source": state.handle,
            "is_final": False,
            "worker": True,
        }))

    def on_network_blocked(self, project_id: str, domain: str, method: str,
                           *, session_id: str | None = None) -> None:
        """Broadcast network_blocked event from platform provider.

        The platform observer runs project-scoped (not session-scoped), so
        ``session_id`` is typically ``None`` here — the field is included
        for shape consistency with other ``agent.activity`` events. The
        frontend filter passes events with falsy session_id through, so a
        ``None`` value still surfaces in the viewed session.
        """
        self._ws.broadcast(project_id, {
            "type": "agent.activity",
            "project_id": project_id,
            "session_id": session_id,
            "id": uuid4().hex,
            "category": "network_blocked",
            "description": f"Blocked {method} request to {domain}",
            "tool_name": "",
            "source": "platform",
            "timestamp": _now(),
        })
