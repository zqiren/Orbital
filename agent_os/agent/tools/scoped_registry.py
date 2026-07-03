# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""ScopedToolRegistry — wraps a ToolRegistry and enforces per-task file
scopes on the path-bearing WRITE tools (spec 009, W2/W3).

Only ``write``/``edit`` (the ``path`` argument on each — see write.py/edit.py)
are gated: a fanout task's ``files_scope`` restricts where a worker may
CREATE/MODIFY files, not where it may read from or what shell commands it may
run (that scope is prompt-level, enforced by the task brief itself). Reads and
shell always pass through untouched.

Containment mirrors ``_path_utils.resolve_safe`` (realpath BOTH the candidate
path and the allowed/forbidden prefixes, `+ os.sep` guard) rather than a
lexical string-prefix check — a symlink under an allowed directory that
resolves elsewhere must not slip through.
"""

from __future__ import annotations

import os

from .base import ToolResult

# Path-bearing WRITE tools this registry gates, and the argument on each that
# carries the workspace-relative (or absolute-inside-workspace) path.
_GATED_WRITE_TOOLS = {
    "write": "path",
    "edit": "path",
}

_SCOPE_ERROR = "Error: path outside this task's file scope"


class ScopedToolRegistry:
    """Wraps ``inner`` (a ``ToolRegistry``-shaped object); enforces write
    scopes. Delegates ``is_async``/``execute``/``execute_async``/``schemas``/
    ``reset_run_state``/``tool_names`` to ``inner`` — the full surface
    ``AgentLoop`` (loop.py:602,737,1087-1097) and ``NativeWorkerAdapter``'s
    ``PromptContext`` build consume."""

    def __init__(self, inner, allowed: list[str] | None,
                 forbidden: list[str] | None, workspace: str):
        self._inner = inner
        self._workspace = os.path.realpath(workspace)
        self._allowed = (
            [self._resolve_prefix(p) for p in allowed] if allowed is not None else None
        )
        self._forbidden = (
            [self._resolve_prefix(p) for p in forbidden] if forbidden is not None else None
        )

    def _resolve_prefix(self, prefix: str) -> str:
        if os.path.isabs(prefix):
            return os.path.realpath(prefix)
        return os.path.realpath(os.path.join(self._workspace, prefix))

    def _resolve_target(self, path: str) -> str:
        if os.path.isabs(path):
            return os.path.realpath(path)
        return os.path.realpath(os.path.join(self._workspace, path))

    @staticmethod
    def _is_under(target: str, prefix: str) -> bool:
        return target == prefix or target.startswith(prefix + os.sep)

    def _scope_violation(self, name: str, arguments: dict) -> ToolResult | None:
        path_arg = _GATED_WRITE_TOOLS.get(name)
        if path_arg is None:
            return None
        raw_path = arguments.get(path_arg)
        if not raw_path:
            return None  # let the inner tool report its own missing-arg error

        target = self._resolve_target(raw_path)

        if self._allowed is not None and not any(
            self._is_under(target, p) for p in self._allowed
        ):
            return ToolResult(content=_SCOPE_ERROR)
        if self._forbidden is not None and any(
            self._is_under(target, p) for p in self._forbidden
        ):
            return ToolResult(content=_SCOPE_ERROR)
        return None

    # ------------------------------------------------------------------
    # Gated dispatch
    # ------------------------------------------------------------------

    def execute(self, name: str, arguments: dict) -> ToolResult:
        violation = self._scope_violation(name, arguments)
        if violation is not None:
            return violation
        return self._inner.execute(name, arguments)

    async def execute_async(self, name: str, arguments: dict) -> ToolResult:
        violation = self._scope_violation(name, arguments)
        if violation is not None:
            return violation
        return await self._inner.execute_async(name, arguments)

    # ------------------------------------------------------------------
    # Pure delegation — unaffected by scoping
    # ------------------------------------------------------------------

    def is_async(self, name: str) -> bool:
        return self._inner.is_async(name)

    def schemas(self) -> list[dict]:
        return self._inner.schemas()

    def reset_run_state(self) -> None:
        return self._inner.reset_run_state()

    def tool_names(self) -> list[str]:
        return self._inner.tool_names()
