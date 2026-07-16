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
                 max_sends_per_run: int = 10, depth: int = 0,
                 session_id: str | None = None):
        self.sub_agent_manager = sub_agent_manager
        self.project_id = project_id
        # The management session this tool instance is registered under. Threaded
        # to every sub_agent_manager.* call so the sub-agent adapter is keyed by
        # SessionKey(project, session_id) — list_active, stop_all, eviction,
        # and the on_completed push all see the same bucket. Without this, a
        # non-default management session would route sub-agents under "default"
        # and lose them downstream (see Quick Tasks failure 2026-05-28).
        self.session_id = session_id
        self.name = "agent_message"
        self.description = (
            "Communicate with sub-agents: send (dispatches a task — spawns "
            "the agent automatically if it is not running), respond to a "
            "blocked interaction, stop, list, status. "
            "A send resumes the sub-agent's prior conversation in this chat "
            "session by default; pass fresh=true to start a clean thread."
        )
        self._max_sends_per_run = max_sends_per_run
        self._send_count = 0
        self._depth = depth
        self.parameters = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": (
                        "Action: send (dispatch a task; spawns on demand), "
                        "respond (answer a blocked in-flight request), stop, "
                        "list, status"
                    ),
                    "enum": ["send", "respond", "stop", "list", "status"],
                },
                "agent": {"type": "string", "description": "Agent handle"},
                "message": {"type": "string", "description": "The task to dispatch (required for send)"},
                "interaction_id": {
                    "type": "string",
                    "description": "Opaque id from an interaction_required event (required for respond)",
                },
                "selection": {
                    "description": "Optional provider-offered selection id or ids for respond",
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                        {"type": "object", "additionalProperties": True},
                    ],
                },
                "fresh": {
                    "type": "boolean",
                    "description": (
                        "Only for action=send. Default false: the send "
                        "continues the sub-agent's prior conversation in this "
                        "chat session (its earlier context is still there). "
                        "Set true ONLY when delegating a genuinely unrelated "
                        "task, or to reset a sub-agent thread that has grown "
                        "too long; this starts a clean session and drops the "
                        "prior context. Otherwise omit it."
                    ),
                },
            },
            "required": ["action", "agent"],
        }

    def on_run_start(self) -> None:
        """Reset per-run state. Called by ToolRegistry.reset_run_state()."""
        self._send_count = 0

    async def execute(self, action: str, agent: str = "", message: str = "",
                      fresh: bool = False, interaction_id: str = "",
                      selection=None, **kwargs) -> ToolResult:
        try:
            if self.sub_agent_manager is None:
                return ToolResult(content="Error: sub-agent support not yet available.")

            # Require agent for start/send/stop
            if action in ("send", "respond", "stop") and not agent:
                return ToolResult(
                    content=f"Error: 'agent' parameter is required for action '{action}'"
                )

            if action == "list":
                agents = self.sub_agent_manager.list_active(
                    self.project_id, session_id=self.session_id,
                )
                return ToolResult(content=json.dumps(agents))

            if action == "status":
                status = self.sub_agent_manager.status(
                    self.project_id, agent, session_id=self.session_id,
                )
                return ToolResult(content=status)

            if action == "send":
                # Depth gate: send spawns-on-demand, so the nesting limit
                # that previously guarded the (removed) start action lives
                # here. A max-depth agent can never have a deeper sub-agent
                # already running, so gating every send is equivalent.
                if self._depth >= MAX_DEPTH:
                    return ToolResult(
                        content=(
                            f"Error: sub-agent depth limit reached "
                            f"(max {MAX_DEPTH} levels). Cannot spawn deeper "
                            f"sub-agents. Complete this task directly or "
                            f"return results to your parent agent."
                        )
                    )
                self._send_count += 1
                if self._send_count > self._max_sends_per_run:
                    return ToolResult(
                        content=(
                            f"Error: agent_message send limit reached "
                            f"({self._max_sends_per_run} sends per run). "
                            f"Summarize what you have so far and present results to the user."
                        )
                    )
                result = await self.sub_agent_manager.send(
                    self.project_id, agent, message,
                    session_id=self.session_id,
                    depth=self._depth + 1,
                    fresh=bool(fresh),
                )
                # A failed dispatch (unknown agent, spawn failure, shutdown)
                # must NOT yield — surface the error so the LLM can react.
                if isinstance(result, str) and result.startswith("Error"):
                    return ToolResult(content=result)
                # Successful dispatch: delivering a task ends the management
                # turn (yield_turn). The send is non-blocking; the sub-agent's
                # result is pushed back later via on_completed and the loop
                # restarts. Yielding here prevents the LLM from busy-polling the
                # sub-agent (which trips the ping-pong guard). See
                # docs/investigations/REPORT-dispatch-yield-and-push.md.
                return ToolResult(
                    content=f"Dispatched to {agent}. Awaiting completion. {result}",
                    meta={"yield_turn": True},
                )

            if action == "respond":
                if not interaction_id:
                    return ToolResult(
                        content="Error: 'interaction_id' is required for action 'respond'"
                    )
                routed = await self.sub_agent_manager.respond_to_interaction(
                    self.project_id, agent, interaction_id,
                    session_id=self.session_id,
                    text=message or None,
                    selection=selection,
                )
                if not routed:
                    return ToolResult(
                        content=(
                            f"Error: no pending interaction '{interaction_id}' "
                            f"for agent '{agent}'"
                        )
                    )
                return ToolResult(
                    content=(
                        f"Response delivered to {agent}. "
                        "Awaiting the current task's completion."
                    ),
                    meta={"yield_turn": True},
                )

            if action == "stop":
                result = await self.sub_agent_manager.stop(
                    self.project_id, agent, session_id=self.session_id,
                )
                return ToolResult(content=result)

            return ToolResult(content=f"Error: unknown action '{action}'")
        except Exception as e:
            return ToolResult(content=f"Error: {str(e)}")
