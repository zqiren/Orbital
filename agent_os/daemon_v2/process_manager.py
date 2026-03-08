# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Sub-agent output -> session bridge.

Consumes adapter output streams and feeds into session as role='agent' messages.
"""

import asyncio
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProcessManager:
    """Consumes adapter output streams. Feeds into session as role='agent' messages."""

    def __init__(self, ws_manager, activity_translator):
        self._ws = ws_manager
        self._activity_translator = activity_translator
        self._tasks: dict[str, asyncio.Task] = {}      # "{project_id}:{handle}" -> consumer task
        self._sessions: dict[str, object] = {}           # project_id -> session

    def set_session(self, project_id: str, session) -> None:
        """Register session for a project."""
        self._sessions[project_id] = session

    async def start(self, project_id: str, handle: str, adapter) -> None:
        """Start background task consuming adapter.read_stream()."""
        key = f"{project_id}:{handle}"

        async def consume():
            session = self._sessions.get(project_id)
            if session is None:
                return
            try:
                async for chunk in adapter.read_stream():
                    msg = {
                        "role": "agent",
                        "source": handle,
                        "content": chunk.text,
                        "timestamp": _now(),
                    }
                    session.append(msg)

                    if chunk.chunk_type == "approval_request":
                        metadata = getattr(chunk, 'metadata', {}) or {}
                        self._ws.broadcast(project_id, {
                            "type": "approval.request",
                            "project_id": project_id,
                            "what": f"Sub-agent {handle} requests approval",
                            "tool_name": metadata.get("tool_name", ""),
                            "tool_call_id": metadata.get("request_id", ""),
                            "tool_args": metadata.get("tool_input", {}),
                            "source": handle,
                            "recent_activity": [],
                        })

                    self._activity_translator.on_message(msg, project_id)
            except asyncio.CancelledError:
                pass

        task = asyncio.create_task(consume())
        self._tasks[key] = task

    async def stop(self, project_id: str, handle: str) -> None:
        """Cancel consumer task."""
        key = f"{project_id}:{handle}"
        task = self._tasks.pop(key, None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
