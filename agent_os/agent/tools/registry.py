# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tool registry — dispatches tool calls by name with a safety-net catch.

Centralized secret substitution: arguments containing `<secret:name.field>`
tokens are resolved against the user credential store before being dispatched
to any tool. This means individual tools (Shell, Browser, …) never have to
implement their own substitution logic.
"""

import asyncio

from .base import Tool, ToolResult  # re-export ToolResult for downstream consumers
from .browser_safety import detect_secrets, substitute_secrets


class ToolRegistry:
    """Registers tools and dispatches execute calls by name."""

    def __init__(self, user_credential_store=None):
        self._tools: dict[str, Tool] = {}
        self._user_credential_store = user_credential_store
        self._resolver = self._make_resolver(user_credential_store)

    @staticmethod
    def _make_resolver(store):
        """Build a callable that resolves secret keys like 'ical.url' from the store."""
        if store is None:
            def _noop(key: str) -> str | None:
                return None
            return _noop

        def _resolve(key: str) -> str | None:
            if "." in key:
                name, field = key.split(".", 1)
                return store.get_value(name, field)
            return None
        return _resolve

    def _substitute_secrets_in_args(self, arguments: dict) -> dict:
        """Deep-walk arguments and replace `<secret:...>` tokens with resolved values.

        Returns a new structure; the original is not mutated. When no credential
        store is configured, returns ``arguments`` unchanged. Raises ``ValueError``
        from ``substitute_secrets`` if a key cannot be resolved — callers rely on
        the registry's safety-net catch to surface this as a tool error.
        """
        if self._user_credential_store is None:
            return arguments

        def _walk(value):
            if isinstance(value, str):
                if detect_secrets(value):
                    return substitute_secrets(value, self._resolver)
                return value
            if isinstance(value, dict):
                return {k: _walk(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_walk(v) for v in value]
            if isinstance(value, tuple):
                return tuple(_walk(v) for v in value)
            return value

        return _walk(arguments)

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
            arguments = self._substitute_secrets_in_args(arguments)
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
            arguments = self._substitute_secrets_in_args(arguments)
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
