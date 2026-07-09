# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""CheckpointStateTool — agent-decided memory consolidation trigger.

When a [MEMORY HYGIENE] flag reports a Layer-1 file is over its soft budget,
the agent calls this tool to trigger a consolidation (dedup/cleanup) pass over
the project state files (PROJECT_STATE, DECISIONS, LESSONS, INDEX): merging
duplicates and superseding stale entries. The incremental write/edit calls
already persisted the content; this pass only relieves inflation. It runs via
run_session_end_routine with bypass_idempotency=True so it serializes like any
other write.

The actual refresh is performed asynchronously by the AgentLoop trigger
infrastructure; this tool is only a signal. The tool stores a callback
(set at registration time) that fires the refresh on the loop; the callback
returns a status string immediately, the pass runs in the background (spec 013).
"""

import asyncio

from .base import Tool, ToolResult


class CheckpointStateTool(Tool):
    """Signal the agent loop to run a state refresh immediately."""

    is_async = True

    def __init__(self, on_checkpoint):
        """Create the tool.

        Args:
            on_checkpoint: async callable() that fires the refresh. Called by
                           execute(); the callback returns a status string
                           immediately, the pass runs in the background (spec 013).
        """
        self._on_checkpoint = on_checkpoint
        self.name = "checkpoint_state"
        self.description = (
            "Consolidate the project state files "
            "(PROJECT_STATE, DECISIONS, LESSONS, INDEX): merge duplicates and "
            "supersede stale entries to relieve content inflation. "
            "Your incremental write/edit calls already SAVED the content — this "
            "tool does NOT persist anything new; it only cleans up. "
            "The pass runs in the background and can take a few MINUTES; this "
            "tool returns immediately, and the [MEMORY HYGIENE] flag may "
            "persist for several turns while the pass runs — that is normal, "
            "not a failure. Do not call this tool again while a pass is in "
            "flight (repeat calls just coalesce into it) and do not hand-edit "
            "the file mid-pass; the flag itself will tell you when a manual "
            "edit is the right move. "
            "Call it ONLY when a [MEMORY HYGIENE] flag reports a file is over its "
            "soft budget (i.e. consolidation is actually needed) — not on "
            "task completion or progress milestones."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Brief description of why a checkpoint is needed now.",
                },
            },
            "required": ["reason"],
        }

    async def execute(self, **arguments) -> ToolResult:
        reason = arguments.get("reason", "")
        try:
            status = await self._on_checkpoint()
            return ToolResult(content=f"{status} Reason noted: {reason}")
        except Exception as e:
            return ToolResult(content=f"State checkpoint failed: {e}")
