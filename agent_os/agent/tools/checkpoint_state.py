# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""CheckpointStateTool — agent-decided state refresh trigger.

When the agent judges that meaningful work has accumulated, it calls this
tool to trigger an immediate project-state refresh (PROJECT_STATE, DECISIONS,
LESSONS, CONTEXT, SESSION_LOG). The refresh runs via run_session_end_routine
with bypass_idempotency=True so it serializes like any other write.

The actual refresh is performed asynchronously by the AgentLoop trigger
infrastructure; this tool is only a signal. The tool stores a callback
(set at registration time) that fires the refresh on the loop.
"""

import asyncio

from .base import Tool, ToolResult


class CheckpointStateTool(Tool):
    """Signal the agent loop to run a state refresh immediately."""

    is_async = True

    def __init__(self, on_checkpoint):
        """Create the tool.

        Args:
            on_checkpoint: async callable() that fires the refresh.
                           Called by execute(); the loop awaits the result.
        """
        self._on_checkpoint = on_checkpoint
        self.name = "checkpoint_state"
        self.description = (
            "Trigger an immediate checkpoint of project state files "
            "(PROJECT_STATE, DECISIONS, LESSONS, CONTEXT, SESSION_LOG). "
            "Call this when you have completed a significant piece of work "
            "and want to ensure it is persisted before continuing. "
            "Do NOT call this more than once every 15 turns — a cooldown "
            "is enforced automatically."
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
            await self._on_checkpoint()
            return ToolResult(
                content=f"State checkpoint triggered successfully. Reason: {reason}"
            )
        except Exception as e:
            return ToolResult(
                content=f"State checkpoint failed: {e}"
            )
