# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tool registry — dispatches tool calls by name with a safety-net catch."""

import asyncio

from .base import Tool, ToolResult  # re-export ToolResult for downstream consumers


class ToolRegistry:
    """Registers tools and dispatches execute calls by name."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool. Raises ValueError on duplicate name."""
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def is_async(self, name: str) -> bool:
        """Return True if the named tool has an async execute method."""
        tool = self._tools.get(name)
        return tool is not None and getattr(tool, "is_async", False)

    def execute(self, name: str, arguments: dict) -> ToolResult:
        """Execute a sync tool by name. NEVER raises — returns ToolResult with error on failure."""
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(content=f"Error: unknown tool '{name}'")
        try:
            result = tool.execute(**arguments)
            if asyncio.iscoroutine(result):
                result.close()  # prevent "coroutine never awaited" warning
                return ToolResult(
                    content=f"Error: tool '{name}' is async — use execute_async()"
                )
            return result
        except Exception as e:
            # Safety net: individual tools should handle their own errors,
            # but this catch prevents loop crash if a tool violates the contract.
            return ToolResult(content=f"Error executing {name}: {str(e)}")

    async def execute_async(self, name: str, arguments: dict) -> ToolResult:
        """Execute an async tool by name. NEVER raises — returns ToolResult with error on failure."""
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(content=f"Error: unknown tool '{name}'")
        try:
            return await tool.execute(**arguments)
        except Exception as e:
            return ToolResult(content=f"Error executing {name}: {str(e)}")

    def schemas(self) -> list[dict]:
        """Return OpenAI-compatible schemas for all registered tools."""
        return [t.schema() for t in self._tools.values()]

    def reset_run_state(self) -> None:
        """Reset per-run state on all tools. Called at loop start."""
        for tool in self._tools.values():
            hook = getattr(tool, "on_run_start", None)
            if hook is not None:
                hook()

    def tool_names(self) -> list[str]:
        """Return list of registered tool names."""
        return list(self._tools.keys())
