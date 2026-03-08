# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""AgentMessageTool — IPC to sub-agents via SubAgentManager."""

import json

from .base import Tool, ToolResult

MAX_DEPTH = 3  # Max sub-agent nesting levels (0-indexed: 0, 1, 2 allowed)


class AgentMessageTool(Tool):
    """Send messages to sub-agents or manage their lifecycle."""

    is_async = True  # Signal to ToolRegistry that execute is async

    def __init__(self, sub_agent_manager=None, project_id: str = "",
                 max_sends_per_run: int = 10, depth: int = 0):
        self.sub_agent_manager = sub_agent_manager
        self.project_id = project_id
        self.name = "agent_message"
        self.description = "Communicate with sub-agents: start, send, stop, list, status."
        self._max_sends_per_run = max_sends_per_run
        self._send_count = 0
        self._depth = depth
        self.parameters = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action: start, send, stop, list, status",
                    "enum": ["start", "send", "stop", "list", "status"],
                },
                "agent": {"type": "string", "description": "Agent handle"},
                "message": {"type": "string", "description": "Message to send"},
            },
            "required": ["action", "agent"],
        }

    def on_run_start(self) -> None:
        """Reset per-run state. Called by ToolRegistry.reset_run_state()."""
        self._send_count = 0

    async def execute(self, action: str, agent: str = "", message: str = "", **kwargs) -> ToolResult:
        try:
            if self.sub_agent_manager is None:
                return ToolResult(content="Error: sub-agent support not yet available.")

            # Require agent for start/send/stop
            if action in ("start", "send", "stop") and not agent:
                return ToolResult(
                    content=f"Error: 'agent' parameter is required for action '{action}'"
                )

            if action == "list":
                agents = self.sub_agent_manager.list_active(self.project_id)
                return ToolResult(content=json.dumps(agents))

            if action == "status":
                status = self.sub_agent_manager.status(self.project_id, agent)
                return ToolResult(content=status)

            if action == "start":
                if self._depth >= MAX_DEPTH:
                    return ToolResult(
                        content=(
                            f"Error: sub-agent depth limit reached "
                            f"(max {MAX_DEPTH} levels). Cannot spawn deeper "
                            f"sub-agents. Complete this task directly or "
                            f"return results to your parent agent."
                        )
                    )
                result = await self.sub_agent_manager.start(
                    self.project_id, agent, depth=self._depth + 1,
                )
                return ToolResult(content=result)

            if action == "send":
                self._send_count += 1
                if self._send_count > self._max_sends_per_run:
                    return ToolResult(
                        content=(
                            f"Error: agent_message send limit reached "
                            f"({self._max_sends_per_run} sends per run). "
                            f"Summarize what you have so far and present results to the user."
                        )
                    )
                result = await self.sub_agent_manager.send(self.project_id, agent, message)
                return ToolResult(content=result)

            if action == "stop":
                result = await self.sub_agent_manager.stop(self.project_id, agent)
                return ToolResult(content=result)

            return ToolResult(content=f"Error: unknown action '{action}'")
        except Exception as e:
            return ToolResult(content=f"Error: {str(e)}")
