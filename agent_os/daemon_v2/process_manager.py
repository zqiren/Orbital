# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Sub-agent output -> transcript bridge.

Consumes adapter output streams and writes to per-agent transcript files.
v5: ProcessManager never writes role=agent to the management session.
"""

import asyncio
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProcessManager:
    """Consumes adapter output streams. Writes to sub-agent transcripts (v5)."""

    def __init__(self, ws_manager, activity_translator, lifecycle_observer=None):
        self._ws = ws_manager
        self._activity_translator = activity_translator
        self._lifecycle = lifecycle_observer
        self._tasks: dict[str, asyncio.Task] = {}      # "{project_id}:{handle}" -> consumer task

    async def start(self, project_id: str, handle: str, adapter, transcript=None,
                    *, session_id: str | None = None) -> None:
        """Start background task consuming adapter.read_stream().

        ``session_id`` is the management session that owns this sub-agent. It is
        forwarded to every lifecycle event the consumer fires so the push lands
        under the correct SessionKey. Without it, ``on_completed`` /
        ``on_error`` would default to the single-loop sentinel and silently
        drop on non-default sessions (see investigation
        ``INVESTIGATION-2026-05-28-backend-still-broken.md`` — this is the SDK
        transport completion path the prior fix missed).
        """
        key = f"{project_id}:{handle}"

        async def consume():
            last_response_text = ""
            try:
                async for chunk in adapter.read_stream():
                    if chunk.chunk_type == "turn_complete":
                        if self._lifecycle and transcript is not None:
                            await self._lifecycle.on_completed(
                                project_id, handle,
                                summary=last_response_text or "(no output)",
                                transcript_path=transcript.filepath,
                                session_id=session_id,
                            )
                        last_response_text = ""  # reset for next turn
                        continue
                    entry = {
                        "source": handle,
                        "content": chunk.text,
                        "timestamp": _now(),
                        "chunk_type": chunk.chunk_type,
                    }
                    # Write to sub-agent transcript (v5: never to management session)
                    if transcript is not None:
                        transcript.append(entry)

                    # Track last response text for completion summary
                    if chunk.chunk_type in ("response", "message") or chunk.chunk_type is None:
                        last_response_text = chunk.text
                        self._ws.broadcast(project_id, {
                            "type": "chat.sub_agent_message",
                            "project_id": project_id,
                            "content": chunk.text,
                            "source": handle,
                            "timestamp": entry["timestamp"],
                        })

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

                    self._activity_translator.on_message(
                        {"role": "agent", "source": handle, "content": chunk.text, "timestamp": entry["timestamp"]},
                        project_id
                    )

                # Stream ended — fire lifecycle event
                if self._lifecycle and transcript is not None:
                    await self._lifecycle.on_completed(
                        project_id, handle,
                        summary=last_response_text or "(no output)",
                        transcript_path=transcript.filepath,
                        session_id=session_id,
                    )
            except asyncio.CancelledError:
                pass
            except Exception as e:
                if self._lifecycle and transcript is not None:
                    await self._lifecycle.on_error(
                        project_id, handle, str(e), transcript.filepath,
                        session_id=session_id,
                    )

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
