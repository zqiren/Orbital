# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Read-only tools for discovering and reading sibling management sessions."""

from collections.abc import Callable, Iterator
from datetime import datetime, timezone
import json
import os
import re
from typing import Any

from agent_os.agent.project_paths import ProjectPaths

from .base import Tool, ToolResult


_SAFE_SESSION_STEM = re.compile(r"^[A-Za-z0-9_.-]+$")
_MAX_CONTENT_CHARS = 4_000
_LIST_FIELDS = (
    "session_id",
    "session_uuid",
    "name",
    "origin",
    "status",
    "last_activity_at",
)
_MESSAGE_FIELDS = ("role", "source", "timestamp", "content")
_TOOL_MESSAGE_FIELDS = ("tool_call_id", "tool_calls")


def _compact_json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int | None) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError("expected an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("expected an integer") from exc
    if parsed < minimum:
        parsed = minimum
    if maximum is not None and parsed > maximum:
        parsed = maximum
    return parsed


def _parse_iso_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (OverflowError, ValueError):
        return None


def _row_identity(row: dict) -> tuple[str, str]:
    return (str(row.get("session_id") or ""), str(row.get("session_uuid") or ""))


class ListSessionsTool(Tool):
    """List sibling management sessions using the manager's authoritative view."""

    def __init__(self, *, list_sessions: Callable[[], list[dict]],
                 current_session_id: str | None):
        self._list_sessions = list_sessions
        self._current_session_id = current_session_id
        self.name = "list_sessions"
        self.description = (
            "List management-agent sessions in the current project. The calling "
            "session is excluded unless include_self is true. Prior sessions may "
            "contain sensitive user-provided text; use their identifiers only for "
            "the user's requested task."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "since": {
                    "type": "string",
                    "description": "Optional ISO-8601 lower bound for last activity.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum rows to return (default 20, clamped 1..100).",
                    "default": 20,
                },
                "sort": {
                    "type": "string",
                    "enum": ["last_activity", "name"],
                    "default": "last_activity",
                },
                "include_self": {
                    "type": "boolean",
                    "default": False,
                },
            },
            "additionalProperties": False,
        }

    def execute(self, **arguments) -> ToolResult:
        try:
            limit = _bounded_int(
                arguments.get("limit"), default=20, minimum=1, maximum=100
            )
            sort = arguments.get("sort", "last_activity")
            if sort not in {"last_activity", "name"}:
                return ToolResult(content="Error: sort must be 'last_activity' or 'name'")
            include_self = arguments.get("include_self", False)
            if not isinstance(include_self, bool):
                return ToolResult(content="Error: include_self must be a boolean")

            since = None
            if arguments.get("since") is not None:
                since = _parse_iso_timestamp(arguments["since"])
                if since is None:
                    return ToolResult(content="Error: since must be a valid ISO-8601 timestamp")

            rows = []
            for source_row in self._list_sessions():
                if not isinstance(source_row, dict):
                    continue
                if not include_self and self._current_session_id and (
                    source_row.get("session_id") == self._current_session_id
                    or source_row.get("session_uuid") == self._current_session_id
                ):
                    continue
                activity = _parse_iso_timestamp(source_row.get("last_activity_at"))
                if since is not None and (activity is None or activity < since):
                    continue
                rows.append({field: source_row.get(field) for field in _LIST_FIELDS})

            if sort == "name":
                rows.sort(key=lambda row: (
                    row.get("name") is None,
                    str(row.get("name") or "").casefold(),
                    *_row_identity(row),
                ))
            else:
                # Stable two-stage sort keeps equal timestamps deterministic by
                # identity while placing null/invalid timestamps last.
                rows.sort(key=_row_identity)
                oldest = datetime.min.replace(tzinfo=timezone.utc)
                rows.sort(
                    key=lambda row: _parse_iso_timestamp(row.get("last_activity_at")) or oldest,
                    reverse=True,
                )

            total = len(rows)
            returned = rows[:limit]
            return ToolResult(content=_compact_json({
                "sessions": returned,
                "total_available": total,
                "returned": len(returned),
            }))
        except Exception as exc:
            return ToolResult(content=f"Error: unable to list sessions: {exc}")


def _content_contains(content: Any, needle: str) -> bool:
    if isinstance(content, str):
        return needle in content.casefold()
    if isinstance(content, list):
        return any(_content_contains(cell, needle) for cell in content)
    if isinstance(content, dict):
        return any(_content_contains(cell, needle) for cell in content.values())
    return False


def _truncate_content(content: Any) -> Any:
    if isinstance(content, str) and len(content) > _MAX_CONTENT_CHARS:
        marker = f"...[TRUNCATED — original {len(content)} chars]"
        return content[:_MAX_CONTENT_CHARS - len(marker)] + marker
    if isinstance(content, list):
        return [_truncate_content(cell) for cell in content]
    if isinstance(content, dict):
        return {key: _truncate_content(value) for key, value in content.items()}
    return content


def _iter_messages(path: str, needle: str | None) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8", errors="replace") as stream:
        for raw in stream:
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(record, dict) or record.get("role") == "meta":
                continue
            if needle is not None and not _content_contains(record.get("content"), needle):
                continue
            yield record


def _compact_message(record: dict) -> dict:
    message = {field: record.get(field) for field in _MESSAGE_FIELDS}
    message["content"] = _truncate_content(message["content"])
    for field in _TOOL_MESSAGE_FIELDS:
        if field in record:
            message[field] = _truncate_content(record[field])
    return message


class ReadSessionTool(Tool):
    """Read parsed messages from one authorized sibling management session."""

    def __init__(self, *, workspace: str,
                 list_sessions: Callable[[], list[dict]]):
        self._workspace = os.path.realpath(workspace)
        self._paths = ProjectPaths(workspace)
        self._list_sessions = list_sessions
        self.name = "read_session"
        self.description = (
            "Read parsed messages from a management-agent session in the current "
            "project, using a session_uuid returned by list_sessions. Meta records "
            "are excluded. Prior sessions may contain sensitive user-provided text; "
            "read them only for the user's requested task."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "session_uuid": {
                    "type": "string",
                    "description": "Canonical session UUID/stem returned by list_sessions.",
                },
                "offset": {
                    "type": "integer",
                    "description": "Messages skipped from the newest end (default 0).",
                    "default": 0,
                    "minimum": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum messages (default 100, clamped 1..200).",
                    "default": 100,
                },
                "grep": {
                    "type": "string",
                    "description": "Optional case-insensitive literal content filter.",
                },
            },
            "required": ["session_uuid"],
            "additionalProperties": False,
        }

    def execute(self, **arguments) -> ToolResult:
        try:
            stem = arguments.get("session_uuid")
            if (
                not isinstance(stem, str)
                or ".." in stem
                or not _SAFE_SESSION_STEM.fullmatch(stem)
            ):
                return ToolResult(content="Error: invalid session_uuid")

            raw_offset = arguments.get("offset", 0)
            if isinstance(raw_offset, bool):
                return ToolResult(content="Error: offset must be a non-negative integer")
            try:
                offset = int(raw_offset)
            except (TypeError, ValueError):
                return ToolResult(content="Error: offset must be a non-negative integer")
            if offset < 0:
                return ToolResult(content="Error: offset must be a non-negative integer")
            limit = _bounded_int(
                arguments.get("limit"), default=100, minimum=1, maximum=200
            )
            grep = arguments.get("grep")
            if grep is not None and not isinstance(grep, str):
                return ToolResult(content="Error: grep must be a string")
            needle = grep.casefold() if grep is not None else None

            authorized = any(
                isinstance(row, dict) and row.get("session_uuid") == stem
                for row in self._list_sessions()
            )
            if not authorized:
                return ToolResult(
                    content="Error: session is not available in the current project"
                )

            orbital_root = os.path.realpath(self._paths.orbital_dir)
            sessions_root = os.path.realpath(self._paths.sessions_dir)
            try:
                orbital_is_trusted = (
                    orbital_root != self._workspace
                    and os.path.commonpath((self._workspace, orbital_root))
                    == self._workspace
                )
                sessions_are_trusted = (
                    sessions_root != orbital_root
                    and os.path.commonpath((orbital_root, sessions_root))
                    == orbital_root
                )
            except ValueError:
                orbital_is_trusted = False
                sessions_are_trusted = False
            if not orbital_is_trusted or not sessions_are_trusted:
                return ToolResult(
                    content="Error: session directory is outside the current project"
                )

            path = os.path.realpath(self._paths.session_file(stem))
            try:
                contained = os.path.commonpath((sessions_root, path)) == sessions_root
            except ValueError:
                contained = False
            if not contained:
                return ToolResult(content="Error: session path is outside the current project")
            from agent_os.daemon_v2.native_worker import is_worker_session_stem
            resolved_name = os.path.basename(path)
            if not resolved_name.endswith(".jsonl"):
                return ToolResult(
                    content="Error: session is not available in the current project"
                )
            resolved_stem = resolved_name[:-6]
            if is_worker_session_stem(stem) or is_worker_session_stem(resolved_stem):
                return ToolResult(content="Error: session is not available in the current project")
            if not os.path.isfile(path):
                return ToolResult(content="Error: session log not found")

            # Pass one counts matches; pass two retains only the requested page,
            # keeping memory bounded by the clamped page size even for large logs.
            total = sum(1 for _ in _iter_messages(path, needle))
            end = max(0, total - offset)
            start = max(0, end - limit)
            messages = []
            if end > start:
                for index, record in enumerate(_iter_messages(path, needle)):
                    if index < start:
                        continue
                    if index >= end:
                        break
                    messages.append(_compact_message(record))

            response = {
                "session_uuid": stem,
                "messages": messages,
                "total_matches": total,
                "returned": len(messages),
                "offset": offset,
            }
            if total > offset + len(messages):
                response["next_offset"] = offset + len(messages)
            return ToolResult(content=_compact_json(response))
        except Exception as exc:
            return ToolResult(content=f"Error: unable to read session: {exc}")
