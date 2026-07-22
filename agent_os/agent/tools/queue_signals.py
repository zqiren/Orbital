# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Queue signal tools: mark_task_complete and mark_task_blocked.

These are pure control-flow signals that exit the current agent loop run
and tell the QueueDispatcher how to advance. They produce no side effects
on the user's workspace; they only mutate session state and surface a
structured marker so the dispatcher and the UI can show what happened.

Detection happens earlier in the loop (loop.py) at response-parsing time,
which defers the signal until every co-emitted tool in the same response
has executed, then honors it and exits. These ToolResult-producing classes
exist so the LLM sees them in the schema and because the loop executes the
deferred signal through the registry to give its call id a real result row.
"""

from .base import Tool, ToolResult


class MarkTaskCompleteTool(Tool):
    """Signal that the current queue item is finished. Exits the loop."""

    is_async = False

    def __init__(self):
        self.name = "mark_task_complete"
        self.description = (
            "Call this tool to mark the current queued task as complete. "
            "After calling this, the loop will exit and the dispatcher will "
            "advance to the next queued item. Any other tools you emit in "
            "the same response are executed FIRST, then the loop exits — "
            "still, prefer finishing your work and calling this signal on "
            "its own as the final action."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": (
                        "One- to two-sentence summary of what was done. "
                        "Shown to the user in the completed-items list."
                    ),
                    "maxLength": 1000,
                },
            },
            "required": ["summary"],
        }

    def execute(self, **arguments) -> ToolResult:
        summary = (arguments.get("summary") or "").strip()
        if not summary:
            return ToolResult(
                content="mark_task_complete requires a non-empty summary.",
            )
        return ToolResult(
            content=f"Task marked complete: {summary}",
            meta={"queue_signal": "complete", "summary": summary},
        )


class MarkTaskBlockedTool(Tool):
    """Signal that the current queue item cannot proceed. Exits the loop."""

    is_async = False

    def __init__(self):
        self.name = "mark_task_blocked"
        self.description = (
            "Call this tool when the current queued task cannot be completed "
            "(missing credentials, ambiguous requirements, blocked by another "
            "task, etc.). The loop will exit and the dispatcher will bypass "
            "this item and move on. Any other tools you emit in the same "
            "response are executed first, then the loop exits — write the "
            "reason CLEARLY."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": (
                        "One- to two-sentence explanation of why the task "
                        "is blocked. Shown to the user so they can unblock."
                    ),
                    "maxLength": 1000,
                },
            },
            "required": ["reason"],
        }

    def execute(self, **arguments) -> ToolResult:
        reason = (arguments.get("reason") or "").strip()
        if not reason:
            return ToolResult(
                content="mark_task_blocked requires a non-empty reason.",
            )
        return ToolResult(
            content=f"Task marked blocked: {reason}",
            meta={"queue_signal": "blocked", "reason": reason},
        )
