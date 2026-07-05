# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""FanoutTool — dispatch N independent sub-tasks to parallel native-worker
sessions via SubAgentManager.dispatch_fanout (spec 009, W2/W3/W4).

Mirrors AgentMessageTool's structure and error/yield_turn contract: a failed
dispatch (bad input, concurrency cap, unwired factory) surfaces as a plain
ToolResult so the LLM can react; a successful dispatch yields the turn — the
fanout is non-blocking, and results arrive together later via the join/
observer push (see sub_agent_manager.dispatch_fanout, fanout.py FanoutRegistry).
"""

from .agent_message import MAX_DEPTH
from .base import Tool, ToolResult

MIN_TASKS = 2
MAX_TASKS = 5


class FanoutTool(Tool):
    """Dispatch 2-5 independent sub-tasks to parallel worker sessions."""

    is_async = True

    def __init__(self, sub_agent_manager=None, project_id: str = "",
                 session_id: str | None = None, depth: int = 0):
        self.sub_agent_manager = sub_agent_manager
        self.project_id = project_id
        # Threaded to dispatch_fanout so the fanout's join group and worker
        # adapters are registered under the SAME SessionKey(project, session_id)
        # as the calling management session — see AgentMessageTool.session_id
        # for why this matters (non-default sessions losing sub-agents).
        self.session_id = session_id
        self._depth = depth
        self.name = "fanout"
        self.description = (
            "Dispatch 2-5 INDEPENDENT sub-tasks to parallel worker "
            "sessions. Use only when sub-tasks do not depend on each "
            "other's output. Give each task a disjoint files_scope when "
            "tasks write files. You will be woken ONCE with all results "
            "when every task finishes."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "minItems": MIN_TASKS,
                    "maxItems": MAX_TASKS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "brief": {
                                "type": "string",
                                "description": "Complete task brief for the worker",
                            },
                            "label": {
                                "type": "string",
                                "description": "Short display label",
                            },
                            "files_scope": {
                                "type": "object",
                                "properties": {
                                    "allowed": {"type": "array", "items": {"type": "string"}},
                                    "forbidden": {"type": "array", "items": {"type": "string"}},
                                },
                            },
                        },
                        "required": ["brief", "label"],
                    },
                },
                "max_runtime_s": {
                    "type": "integer",
                    "description": "Hard ceiling, default 3600",
                },
            },
            "required": ["tasks"],
        }

    async def execute(self, tasks=None, max_runtime_s: int = 3600, **kwargs) -> ToolResult:
        try:
            if self.sub_agent_manager is None:
                return ToolResult(content="Error: sub-agent support not yet available.")

            # Depth gate identical to AgentMessageTool's send gate. A worker's
            # own registry never includes fanout/agent_message (no recursive
            # fanout in v1), so this only ever fires for the management loop
            # itself — kept for parity and defense in depth.
            if self._depth >= MAX_DEPTH:
                return ToolResult(
                    content=(
                        f"Error: sub-agent depth limit reached "
                        f"(max {MAX_DEPTH} levels). Cannot spawn deeper "
                        f"sub-agents. Complete this task directly or "
                        f"return results to your parent agent."
                    )
                )

            if not isinstance(tasks, list) or not (MIN_TASKS <= len(tasks) <= MAX_TASKS):
                count = len(tasks) if isinstance(tasks, list) else 0
                return ToolResult(
                    content=(
                        f"Error: fanout requires {MIN_TASKS}-{MAX_TASKS} tasks "
                        f"(got {count}). A single task should use agent_message "
                        f"or be done inline."
                    )
                )
            for i, task in enumerate(tasks):
                if not isinstance(task, dict) or not task.get("brief") or not task.get("label"):
                    return ToolResult(
                        content=f"Error: task {i} is missing a required 'brief' and/or 'label'"
                    )

            result = await self.sub_agent_manager.dispatch_fanout(
                self.project_id, tasks,
                session_id=self.session_id,
                max_runtime_s=max_runtime_s,
                depth=self._depth + 1,
            )
            # A failed dispatch (bad input caught late, concurrency cap,
            # unwired factory) must NOT yield — surface the error so the LLM
            # can react, mirroring AgentMessageTool.send.
            if isinstance(result, str) and result.startswith("Error"):
                return ToolResult(content=result)
            # Successful dispatch ends the turn: the fanout is non-blocking,
            # and busy-polling while N workers run would trip the ping-pong
            # guard. Results are pushed back together once every task
            # finishes (or is stopped/stalled).
            return ToolResult(content=result, meta={"yield_turn": True})
        except Exception as e:
            return ToolResult(content=f"Error: {str(e)}")
