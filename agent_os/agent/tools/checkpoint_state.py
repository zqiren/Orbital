# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""CheckpointStateTool — agent-requested memory-editor pass.

Over-budget memory files (PROJECT_STATE, DECISIONS, LESSONS, INDEX) are
tidied automatically: ContextManager.prepare() notices the file and the loop
schedules the memory editor in the background (spec 089). This tool asks for
a pass now, even when nothing is over budget. The editor archives stale
entries by id and merges duplicates; it never records new facts — the agent's
own write/edit calls do that.

The actual pass is performed asynchronously by the AgentLoop scheduler; this
tool is only a signal. The callback returns a status string immediately, the
pass runs in the background (spec 013).
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
            "Ask the background memory editor to tidy the project memory files "
            "(PROJECT_STATE, DECISIONS, LESSONS, INDEX) now: it moves stale "
            "entries to their archive by id (leaving a pointer line) and merges "
            "duplicates. Files over their budget are tidied automatically, so "
            "this is rarely needed. It never records new facts — your write/"
            "edit calls do that. The pass runs in the background and can take a "
            "few MINUTES; this tool returns immediately, and a [MEMORY HYGIENE] "
            "flag may persist while it runs — that is normal. Do not call it "
            "again while a pass is in flight (repeat calls just coalesce into "
            "it) and do not trim the files by hand meanwhile."
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
