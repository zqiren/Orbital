# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for ScopedToolRegistry: write/edit path-scope enforcement
(allowed/forbidden prefixes, realpath-based symlink-escape resistance) and
full delegation of the registry surface AgentLoop/NativeWorkerAdapter consume
(schemas/is_async/execute/execute_async/reset_run_state/tool_names).

Spec 009 (subagent fanout), Task 3 brief.
"""

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.tools.base import ToolResult
from agent_os.agent.tools.scoped_registry import ScopedToolRegistry

SCOPE_ERROR = "Error: path outside this task's file scope"


def _mock_inner(tool_names=("read", "write", "edit", "grep", "glob", "shell")):
    inner = MagicMock()
    inner.execute.return_value = ToolResult(content="ok")
    inner.execute_async = AsyncMock(return_value=ToolResult(content="ok"))
    inner.is_async.return_value = False
    inner.schemas.return_value = [{"type": "function", "function": {"name": n}} for n in tool_names]
    inner.tool_names.return_value = list(tool_names)
    return inner


class TestWriteScopeGating:
    def test_write_inside_allowed_executes(self, tmp_path):
        inner = _mock_inner()
        (tmp_path / "safe").mkdir()
        registry = ScopedToolRegistry(inner, allowed=["safe"], forbidden=None, workspace=str(tmp_path))

        result = registry.execute("write", {"path": "safe/file.txt", "content": "x"})

        assert result.content == "ok"
        inner.execute.assert_called_once_with("write", {"path": "safe/file.txt", "content": "x"})

    def test_write_outside_allowed_blocked_inner_never_called(self, tmp_path):
        inner = _mock_inner()
        (tmp_path / "safe").mkdir()
        (tmp_path / "other").mkdir()
        registry = ScopedToolRegistry(inner, allowed=["safe"], forbidden=None, workspace=str(tmp_path))

        result = registry.execute("write", {"path": "other/file.txt", "content": "x"})

        assert result.content == SCOPE_ERROR
        inner.execute.assert_not_called()

    def test_write_under_forbidden_blocked(self, tmp_path):
        inner = _mock_inner()
        (tmp_path / "secrets").mkdir()
        registry = ScopedToolRegistry(inner, allowed=None, forbidden=["secrets"], workspace=str(tmp_path))

        result = registry.execute("write", {"path": "secrets/file.txt", "content": "x"})

        assert result.content == SCOPE_ERROR
        inner.execute.assert_not_called()

    def test_forbidden_takes_priority_even_if_also_allowed(self, tmp_path):
        inner = _mock_inner()
        (tmp_path / "safe").mkdir()
        (tmp_path / "safe" / "secrets").mkdir()
        registry = ScopedToolRegistry(
            inner, allowed=["safe"], forbidden=["safe/secrets"], workspace=str(tmp_path),
        )

        result = registry.execute("write", {"path": "safe/secrets/file.txt", "content": "x"})

        assert result.content == SCOPE_ERROR
        inner.execute.assert_not_called()

    def test_symlink_escaping_workspace_blocked(self, tmp_path):
        inner = _mock_inner()
        outside = tmp_path.parent / f"{tmp_path.name}_outside_target"
        outside.mkdir(exist_ok=True)
        safe_dir = tmp_path / "safe"
        safe_dir.mkdir()
        os.symlink(outside, safe_dir / "escape")
        registry = ScopedToolRegistry(inner, allowed=["safe"], forbidden=None, workspace=str(tmp_path))

        result = registry.execute("write", {"path": "safe/escape/file.txt", "content": "x"})

        assert result.content == SCOPE_ERROR
        inner.execute.assert_not_called()

    def test_edit_gated_same_as_write(self, tmp_path):
        inner = _mock_inner()
        (tmp_path / "safe").mkdir()
        (tmp_path / "other").mkdir()
        registry = ScopedToolRegistry(inner, allowed=["safe"], forbidden=None, workspace=str(tmp_path))

        result = registry.execute("edit", {"path": "other/file.txt", "old_text": "a", "new_text": "b"})

        assert result.content == SCOPE_ERROR
        inner.execute.assert_not_called()

    def test_edit_inside_allowed_executes(self, tmp_path):
        inner = _mock_inner()
        (tmp_path / "safe").mkdir()
        registry = ScopedToolRegistry(inner, allowed=["safe"], forbidden=None, workspace=str(tmp_path))

        result = registry.execute("edit", {"path": "safe/file.txt", "old_text": "a", "new_text": "b"})

        assert result.content == "ok"
        inner.execute.assert_called_once()

    def test_absolute_path_resolved_same_as_relative(self, tmp_path):
        inner = _mock_inner()
        (tmp_path / "other").mkdir()
        registry = ScopedToolRegistry(inner, allowed=["safe"], forbidden=None, workspace=str(tmp_path))

        abs_path = str(tmp_path / "other" / "file.txt")
        result = registry.execute("write", {"path": abs_path, "content": "x"})

        assert result.content == SCOPE_ERROR
        inner.execute.assert_not_called()


class TestReadAndShellNotGated:
    def test_read_never_gated(self, tmp_path):
        inner = _mock_inner()
        registry = ScopedToolRegistry(inner, allowed=["safe"], forbidden=None, workspace=str(tmp_path))

        result = registry.execute("read", {"path": "anywhere/file.txt"})

        assert result.content == "ok"
        inner.execute.assert_called_once()

    def test_shell_never_gated_even_with_path_like_arg(self, tmp_path):
        inner = _mock_inner()
        registry = ScopedToolRegistry(inner, allowed=["safe"], forbidden=None, workspace=str(tmp_path))

        result = registry.execute("shell", {"command": "cat /etc/passwd"})

        assert result.content == "ok"
        inner.execute.assert_called_once()

    def test_grep_never_gated(self, tmp_path):
        inner = _mock_inner()
        registry = ScopedToolRegistry(inner, allowed=["safe"], forbidden=None, workspace=str(tmp_path))

        result = registry.execute("grep", {"pattern": "x", "path": "other/dir"})

        assert result.content == "ok"
        inner.execute.assert_called_once()


class TestNoScopeIsUnrestricted:
    def test_no_allowed_no_forbidden_passthrough(self, tmp_path):
        inner = _mock_inner()
        registry = ScopedToolRegistry(inner, allowed=None, forbidden=None, workspace=str(tmp_path))

        result = registry.execute("write", {"path": "anywhere/file.txt", "content": "x"})

        assert result.content == "ok"
        inner.execute.assert_called_once()


class TestAsyncExecuteGating:
    @pytest.mark.asyncio
    async def test_execute_async_blocks_outside_allowed(self, tmp_path):
        inner = _mock_inner()
        (tmp_path / "safe").mkdir()
        (tmp_path / "other").mkdir()
        registry = ScopedToolRegistry(inner, allowed=["safe"], forbidden=None, workspace=str(tmp_path))

        result = await registry.execute_async("write", {"path": "other/file.txt", "content": "x"})

        assert result.content == SCOPE_ERROR
        inner.execute_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_async_allows_inside_allowed(self, tmp_path):
        inner = _mock_inner()
        (tmp_path / "safe").mkdir()
        registry = ScopedToolRegistry(inner, allowed=["safe"], forbidden=None, workspace=str(tmp_path))

        result = await registry.execute_async("write", {"path": "safe/file.txt", "content": "x"})

        assert result.content == "ok"
        inner.execute_async.assert_called_once()


class TestDelegatedSurface:
    """The full registry surface AgentLoop/NativeWorkerAdapter consume must
    pass through to the inner registry unchanged (loop.py:1087-1097 for
    is_async/execute/execute_async, loop.py:602/737 for reset_run_state/
    schemas, native_worker.py's PromptContext build for tool_names)."""

    def test_is_async_delegates(self, tmp_path):
        inner = _mock_inner()
        inner.is_async.return_value = True
        registry = ScopedToolRegistry(inner, allowed=None, forbidden=None, workspace=str(tmp_path))

        assert registry.is_async("shell") is True
        inner.is_async.assert_called_once_with("shell")

    def test_schemas_delegates(self, tmp_path):
        inner = _mock_inner()
        registry = ScopedToolRegistry(inner, allowed=None, forbidden=None, workspace=str(tmp_path))

        assert registry.schemas() == inner.schemas.return_value

    def test_tool_names_delegates(self, tmp_path):
        inner = _mock_inner(tool_names=("read", "write"))
        registry = ScopedToolRegistry(inner, allowed=None, forbidden=None, workspace=str(tmp_path))

        assert registry.tool_names() == ["read", "write"]

    def test_reset_run_state_delegates(self, tmp_path):
        inner = _mock_inner()
        registry = ScopedToolRegistry(inner, allowed=None, forbidden=None, workspace=str(tmp_path))

        registry.reset_run_state()

        inner.reset_run_state.assert_called_once()
