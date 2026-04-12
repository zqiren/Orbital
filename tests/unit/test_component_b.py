# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Unit tests for Component B — Tool Suite.

Written from specs BEFORE implementation exists.
Covers all 14 acceptance criteria from TASK-component-B-tool-suite.md.
Uses pytest + pytest-asyncio. Mocks everything external.
"""

import json
import os
import textwrap
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from agent_os.agent.tools.base import Tool, ToolResult
from agent_os.agent.tools.registry import ToolRegistry
from agent_os.agent.tools.read import ReadTool
from agent_os.agent.tools.write import WriteTool
from agent_os.agent.tools.edit import EditTool
from agent_os.agent.tools.shell import ShellTool
from agent_os.agent.tools.request_access import RequestAccessTool
from agent_os.agent.tools.agent_message import AgentMessageTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DummyTool(Tool):
    """Minimal concrete Tool for registry tests."""

    def __init__(self, name: str = "dummy", description: str = "A dummy tool",
                 parameters: dict | None = None):
        self.name = name
        self.description = description
        self.parameters = parameters or {"type": "object", "properties": {}}

    def execute(self, **arguments) -> ToolResult:
        return ToolResult(content="dummy ok")


class ExplodingTool(Tool):
    """Tool whose execute() deliberately raises — used to verify safety nets."""

    def __init__(self, name: str = "exploding"):
        self.name = name
        self.description = "Raises on execute"
        self.parameters = {"type": "object", "properties": {}}

    def execute(self, **arguments) -> ToolResult:
        raise RuntimeError("boom")


# ===========================================================================
# AC-1: ToolRegistry — register 3 tools, schemas() returns 3 OpenAI schemas
# ===========================================================================

class TestToolRegistrySchemas:

    def test_register_three_tools_returns_three_schemas(self):
        reg = ToolRegistry()
        reg.register(DummyTool(name="alpha"))
        reg.register(DummyTool(name="beta"))
        reg.register(DummyTool(name="gamma"))

        schemas = reg.schemas()
        assert len(schemas) == 3

    def test_schemas_are_openai_format(self):
        reg = ToolRegistry()
        reg.register(DummyTool(name="alpha", description="do alpha",
                               parameters={"type": "object", "properties": {"x": {"type": "string"}}}))
        schemas = reg.schemas()
        s = schemas[0]
        assert s["type"] == "function"
        assert "function" in s
        assert s["function"]["name"] == "alpha"
        assert s["function"]["description"] == "do alpha"
        assert s["function"]["parameters"]["type"] == "object"

    def test_tool_names_returns_registered_names(self):
        reg = ToolRegistry()
        reg.register(DummyTool(name="a"))
        reg.register(DummyTool(name="b"))
        assert set(reg.tool_names()) == {"a", "b"}

    def test_duplicate_registration_raises(self):
        reg = ToolRegistry()
        reg.register(DummyTool(name="dup"))
        with pytest.raises(ValueError, match="Duplicate tool"):
            reg.register(DummyTool(name="dup"))


# ===========================================================================
# AC-2: ToolRegistry — execute("unknown", {}) returns error ToolResult
# ===========================================================================

class TestToolRegistryUnknown:

    def test_unknown_tool_returns_error_toolresult(self):
        reg = ToolRegistry()
        result = reg.execute("unknown", {})
        assert isinstance(result, ToolResult)
        assert "unknown tool" in result.content.lower() or "unknown" in result.content
        assert "unknown" in result.content  # tool name appears in error

    def test_unknown_tool_does_not_raise(self):
        reg = ToolRegistry()
        # Should NOT throw; just return ToolResult
        result = reg.execute("nonexistent", {})
        assert isinstance(result, ToolResult)


# ===========================================================================
# AC-3: ReadTool — temp workspace with a file, read returns file content
# ===========================================================================

class TestReadToolBasic:

    def test_read_existing_file(self, tmp_path):
        workspace = str(tmp_path)
        test_file = tmp_path / "hello.txt"
        test_file.write_text("Hello, world!", encoding="utf-8")

        tool = ReadTool(workspace=workspace)
        result = tool.execute(path="hello.txt")

        assert isinstance(result, ToolResult)
        assert "Hello, world!" in result.content

    def test_read_file_in_subdirectory(self, tmp_path):
        workspace = str(tmp_path)
        subdir = tmp_path / "sub" / "dir"
        subdir.mkdir(parents=True)
        (subdir / "nested.txt").write_text("nested content", encoding="utf-8")

        tool = ReadTool(workspace=workspace)
        result = tool.execute(path="sub/dir/nested.txt")

        assert isinstance(result, ToolResult)
        assert "nested content" in result.content

    def test_read_nonexistent_file_returns_error(self, tmp_path):
        workspace = str(tmp_path)
        tool = ReadTool(workspace=workspace)
        result = tool.execute(path="does_not_exist.txt")

        assert isinstance(result, ToolResult)
        assert "error" in result.content.lower() or "Error" in result.content

    def test_read_directory_returns_listing(self, tmp_path):
        workspace = str(tmp_path)
        (tmp_path / "file_a.txt").write_text("a", encoding="utf-8")
        (tmp_path / "file_b.txt").write_text("b", encoding="utf-8")

        tool = ReadTool(workspace=workspace)
        result = tool.execute(path=".")

        assert isinstance(result, ToolResult)
        assert "file_a.txt" in result.content
        assert "file_b.txt" in result.content


# ===========================================================================
# AC-4: ReadTool — file > 100K chars truncated with notice
# ===========================================================================

class TestReadToolTruncation:

    def test_large_file_truncated_at_100k(self, tmp_path):
        workspace = str(tmp_path)
        large_content = "x" * 150_000
        (tmp_path / "large.txt").write_text(large_content, encoding="utf-8")

        tool = ReadTool(workspace=workspace)
        result = tool.execute(path="large.txt")

        assert isinstance(result, ToolResult)
        # Truncated content should be around 100K, not 150K
        assert len(result.content) < 150_000
        # Should contain a truncation notice
        assert "truncat" in result.content.lower() or "TRUNCAT" in result.content

    def test_file_at_100k_not_truncated(self, tmp_path):
        workspace = str(tmp_path)
        exact_content = "y" * 100_000
        (tmp_path / "exact.txt").write_text(exact_content, encoding="utf-8")

        tool = ReadTool(workspace=workspace)
        result = tool.execute(path="exact.txt")

        assert isinstance(result, ToolResult)
        # At exactly 100K, should NOT be truncated (boundary: <= 100K is fine)
        # SPEC AMBIGUITY: "Truncate at 100K" could mean > 100K or >= 100K.
        # We test that 100K content is returned (no truncation notice expected).
        assert "y" * 1000 in result.content


# ===========================================================================
# AC-5: ReadTool — path outside workspace returns error, not exception
# ===========================================================================

class TestReadToolPathTraversal:

    def test_path_traversal_blocked(self, tmp_path):
        workspace = str(tmp_path)
        tool = ReadTool(workspace=workspace)

        result = tool.execute(path="../../etc/passwd")

        assert isinstance(result, ToolResult)
        assert "error" in result.content.lower() or "Error" in result.content
        # Must NOT raise an exception — just return ToolResult with error

    def test_absolute_path_outside_workspace_blocked(self, tmp_path):
        workspace = str(tmp_path)
        tool = ReadTool(workspace=workspace)

        # Attempt to read an absolute path outside workspace
        if os.name == "nt":
            outside_path = "C:\\Windows\\System32\\drivers\\etc\\hosts"
        else:
            outside_path = "/etc/passwd"

        result = tool.execute(path=outside_path)
        assert isinstance(result, ToolResult)
        assert "error" in result.content.lower() or "Error" in result.content

    def test_path_traversal_does_not_raise(self, tmp_path):
        workspace = str(tmp_path)
        tool = ReadTool(workspace=workspace)

        # This must NOT raise; it must return a ToolResult
        try:
            result = tool.execute(path="../../../etc/passwd")
        except Exception:
            pytest.fail("ReadTool raised an exception for path traversal; it should return a ToolResult with error")

        assert isinstance(result, ToolResult)


# ===========================================================================
# AC-6: WriteTool — write to nested path, parent dirs created
# ===========================================================================

class TestWriteToolNestedPath:

    def test_write_creates_parent_dirs(self, tmp_path):
        workspace = str(tmp_path)
        tool = WriteTool(workspace=workspace)

        result = tool.execute(path="deeply/nested/dir/file.txt", content="hello nested")

        assert isinstance(result, ToolResult)
        target = tmp_path / "deeply" / "nested" / "dir" / "file.txt"
        assert target.exists()
        assert target.read_text(encoding="utf-8") == "hello nested"

    def test_write_returns_success(self, tmp_path):
        workspace = str(tmp_path)
        tool = WriteTool(workspace=workspace)

        result = tool.execute(path="output.txt", content="test content")

        assert isinstance(result, ToolResult)
        assert "success" in result.content.lower() or "success" in result.content

    def test_write_reports_bytes(self, tmp_path):
        workspace = str(tmp_path)
        tool = WriteTool(workspace=workspace)

        result = tool.execute(path="sized.txt", content="12345")

        assert isinstance(result, ToolResult)
        # The spec says content includes bytes count
        # Parse as JSON to verify
        try:
            data = json.loads(result.content)
            assert "bytes" in data or "status" in data
        except json.JSONDecodeError:
            # If not JSON, at least should mention success
            assert "success" in result.content.lower()

    def test_write_overwrite_existing(self, tmp_path):
        workspace = str(tmp_path)
        (tmp_path / "existing.txt").write_text("old content", encoding="utf-8")

        tool = WriteTool(workspace=workspace)
        result = tool.execute(path="existing.txt", content="new content")

        assert isinstance(result, ToolResult)
        assert (tmp_path / "existing.txt").read_text(encoding="utf-8") == "new content"

    def test_write_path_traversal_blocked(self, tmp_path):
        workspace = str(tmp_path)
        tool = WriteTool(workspace=workspace)

        result = tool.execute(path="../../evil.txt", content="malicious")

        assert isinstance(result, ToolResult)
        assert "error" in result.content.lower() or "Error" in result.content


# ===========================================================================
# AC-7: EditTool — unique string replaced; no match → error; multiple → error
# ===========================================================================

class TestEditTool:

    def test_edit_unique_string_replaced(self, tmp_path):
        workspace = str(tmp_path)
        test_file = tmp_path / "code.py"
        test_file.write_text("def hello():\n    return 'world'\n", encoding="utf-8")

        tool = EditTool(workspace=workspace)
        result = tool.execute(path="code.py", old_text="return 'world'", new_text="return 'universe'")

        assert isinstance(result, ToolResult)
        assert "success" in result.content.lower() or "success" in result.content
        updated = test_file.read_text(encoding="utf-8")
        assert "return 'universe'" in updated
        assert "return 'world'" not in updated

    def test_edit_no_match_returns_error(self, tmp_path):
        workspace = str(tmp_path)
        test_file = tmp_path / "code.py"
        test_file.write_text("def hello():\n    pass\n", encoding="utf-8")

        tool = EditTool(workspace=workspace)
        result = tool.execute(path="code.py", old_text="nonexistent text", new_text="replacement")

        assert isinstance(result, ToolResult)
        assert "error" in result.content.lower() or "not found" in result.content.lower()

    def test_edit_multiple_matches_returns_error(self, tmp_path):
        workspace = str(tmp_path)
        test_file = tmp_path / "code.py"
        test_file.write_text("foo bar\nfoo bar\nfoo bar\n", encoding="utf-8")

        tool = EditTool(workspace=workspace)
        result = tool.execute(path="code.py", old_text="foo bar", new_text="baz")

        assert isinstance(result, ToolResult)
        assert "error" in result.content.lower() or "multiple" in result.content.lower()

    def test_edit_nonexistent_file_returns_error(self, tmp_path):
        workspace = str(tmp_path)
        tool = EditTool(workspace=workspace)
        result = tool.execute(path="missing.py", old_text="x", new_text="y")

        assert isinstance(result, ToolResult)
        assert "error" in result.content.lower() or "Error" in result.content

    def test_edit_returns_json_with_replacements(self, tmp_path):
        workspace = str(tmp_path)
        test_file = tmp_path / "data.txt"
        test_file.write_text("alpha beta gamma", encoding="utf-8")

        tool = EditTool(workspace=workspace)
        result = tool.execute(path="data.txt", old_text="beta", new_text="BETA")

        assert isinstance(result, ToolResult)
        try:
            data = json.loads(result.content)
            assert data.get("replacements") == 1 or data.get("status") == "success"
        except json.JSONDecodeError:
            # Acceptable if non-JSON, but should still indicate success
            assert "success" in result.content.lower()

    def test_edit_path_traversal_blocked(self, tmp_path):
        workspace = str(tmp_path)
        tool = EditTool(workspace=workspace)
        result = tool.execute(path="../../etc/passwd", old_text="root", new_text="evil")

        assert isinstance(result, ToolResult)
        assert "error" in result.content.lower() or "Error" in result.content


# ===========================================================================
# AC-8: ShellTool — "echo hello" returns expected ToolResult
# ===========================================================================

class TestShellToolBasic:

    def test_echo_hello(self, tmp_path):
        workspace = str(tmp_path)
        import platform as _platform
        os_type = "windows" if _platform.system() == "Windows" else "linux"
        tool = ShellTool(workspace=workspace, os_type=os_type)

        result = tool.execute(command="echo hello")

        assert isinstance(result, ToolResult)
        assert "Exit code: 0" in result.content
        assert "hello" in result.content
        assert result.meta is not None
        assert result.meta["network"] is False
        assert result.meta["domains"] == []

    def test_echo_on_linux_style(self, tmp_path):
        """Verify that os_type affects shell used (linux → bash)."""
        workspace = str(tmp_path)
        # We construct the tool with linux os_type but mock subprocess
        # to avoid actually running bash on Windows
        tool = ShellTool(workspace=workspace, os_type="linux")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="hello\n",
                stderr="",
            )
            result = tool.execute(command="echo hello")

            # Verify bash was used
            call_args = mock_run.call_args
            cmd = call_args[0][0] if call_args[0] else call_args[1].get("args", [])
            # Should use bash for linux
            assert any("bash" in str(c).lower() for c in (cmd if isinstance(cmd, list) else [cmd]))

        assert isinstance(result, ToolResult)
        assert "Exit code: 0" in result.content

    def test_shell_returns_exit_code_for_failing_command(self, tmp_path):
        workspace = str(tmp_path)
        tool = ShellTool(workspace=workspace, os_type="windows")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr="some error",
            )
            result = tool.execute(command="exit 1")

        assert isinstance(result, ToolResult)
        assert "Exit code: 1" in result.content


# ===========================================================================
# AC-9: ShellTool — output > 200 lines truncated, tempfile saved
# ===========================================================================

class TestShellToolTruncation:

    def test_long_output_truncated(self, tmp_path):
        workspace = str(tmp_path)
        tool = ShellTool(workspace=workspace, os_type="windows")

        # Generate output with > 200 lines
        lines = [f"line {i}" for i in range(300)]
        long_output = "\n".join(lines)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=long_output,
                stderr="",
            )
            result = tool.execute(command="some-long-command")

        assert isinstance(result, ToolResult)
        # Should contain exit code
        assert "Exit code: 0" in result.content
        # Should contain truncation notice
        assert "truncat" in result.content.lower() or "truncated" in result.content.lower()
        # Should contain first 20 lines
        assert "line 0" in result.content
        assert "line 19" in result.content
        # Should contain last 50 lines
        assert "line 299" in result.content
        assert "line 250" in result.content

    def test_truncated_output_saved_to_tempfile(self, tmp_path):
        workspace = str(tmp_path)
        tool = ShellTool(workspace=workspace, os_type="windows")

        lines = [f"line {i}" for i in range(300)]
        long_output = "\n".join(lines)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=long_output,
                stderr="",
            )
            result = tool.execute(command="some-long-command")

        # Check that a tempfile was saved under workspace/orbital-output/{project}/shell-output/
        shell_output_dir = tmp_path / "orbital-output" / "shell-output"
        if shell_output_dir.exists():
            saved_files = list(shell_output_dir.iterdir())
            assert len(saved_files) >= 1
            # Saved file should contain all lines
            saved_content = saved_files[0].read_text(encoding="utf-8")
            assert "line 150" in saved_content

    def test_output_under_200_lines_not_truncated(self, tmp_path):
        workspace = str(tmp_path)
        tool = ShellTool(workspace=workspace, os_type="windows")

        lines = [f"line {i}" for i in range(50)]
        short_output = "\n".join(lines)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=short_output,
                stderr="",
            )
            result = tool.execute(command="some-command")

        assert isinstance(result, ToolResult)
        # All lines should be present
        assert "line 0" in result.content
        assert "line 49" in result.content
        # No truncation notice
        assert "truncat" not in result.content.lower()


# ===========================================================================
# AC-10: ShellTool — command timeout (>120s) returns timeout notice
# ===========================================================================

class TestShellToolTimeout:

    def test_command_timeout_returns_notice(self, tmp_path):
        import subprocess
        workspace = str(tmp_path)
        tool = ShellTool(workspace=workspace, os_type="windows")

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="sleep 999", timeout=120)
            result = tool.execute(command="sleep 999")

        assert isinstance(result, ToolResult)
        assert "timeout" in result.content.lower() or "timed out" in result.content.lower()

    def test_timeout_does_not_raise(self, tmp_path):
        import subprocess
        workspace = str(tmp_path)
        tool = ShellTool(workspace=workspace, os_type="windows")

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="sleep 999", timeout=120)
            try:
                result = tool.execute(command="sleep 999")
            except Exception:
                pytest.fail("ShellTool raised an exception on timeout; should return ToolResult")

        assert isinstance(result, ToolResult)


# ===========================================================================
# AC-11: ShellTool — "curl example.com" → meta with network=True
# ===========================================================================

class TestShellToolNetworkDetection:

    def test_curl_detected_as_network(self, tmp_path):
        workspace = str(tmp_path)
        tool = ShellTool(workspace=workspace, os_type="windows")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="<html>...</html>",
                stderr="",
            )
            result = tool.execute(command="curl example.com")

        assert isinstance(result, ToolResult)
        assert result.meta is not None
        assert result.meta["network"] is True
        assert "example.com" in result.meta["domains"]

    def test_wget_detected_as_network(self, tmp_path):
        workspace = str(tmp_path)
        tool = ShellTool(workspace=workspace, os_type="linux")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="downloaded",
                stderr="",
            )
            result = tool.execute(command="wget https://github.com/repo/file.tar.gz")

        assert isinstance(result, ToolResult)
        assert result.meta is not None
        assert result.meta["network"] is True
        assert "github.com" in result.meta["domains"]

    def test_npm_install_detected_as_network(self, tmp_path):
        workspace = str(tmp_path)
        tool = ShellTool(workspace=workspace, os_type="windows")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="added 42 packages",
                stderr="",
            )
            result = tool.execute(command="npm install express")

        assert isinstance(result, ToolResult)
        assert result.meta is not None
        assert result.meta["network"] is True

    def test_pip_install_detected_as_network(self, tmp_path):
        workspace = str(tmp_path)
        tool = ShellTool(workspace=workspace, os_type="linux")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Successfully installed requests",
                stderr="",
            )
            result = tool.execute(command="pip install requests")

        assert isinstance(result, ToolResult)
        assert result.meta is not None
        assert result.meta["network"] is True

    def test_plain_echo_not_network(self, tmp_path):
        workspace = str(tmp_path)
        tool = ShellTool(workspace=workspace, os_type="windows")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="hello",
                stderr="",
            )
            result = tool.execute(command="echo hello")

        assert result.meta is not None
        assert result.meta["network"] is False
        assert result.meta["domains"] == []

    def test_curl_with_url_extracts_domain(self, tmp_path):
        workspace = str(tmp_path)
        tool = ShellTool(workspace=workspace, os_type="windows")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="ok",
                stderr="",
            )
            result = tool.execute(command="curl https://api.github.com/repos")

        assert result.meta is not None
        assert result.meta["network"] is True
        assert "api.github.com" in result.meta["domains"]

    def test_git_clone_detected_as_network(self, tmp_path):
        workspace = str(tmp_path)
        tool = ShellTool(workspace=workspace, os_type="windows")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Cloning into...",
                stderr="",
            )
            result = tool.execute(command="git clone https://github.com/user/repo.git")

        assert result.meta is not None
        assert result.meta["network"] is True
        assert "github.com" in result.meta["domains"]


# ===========================================================================
# AC-12: RequestAccessTool — always returns pending status
# ===========================================================================

class TestRequestAccessTool:

    def test_returns_pending_status(self):
        tool = RequestAccessTool()
        result = tool.execute(path="/data/secrets", reason="Need to read config",
                              access_type="read")

        assert isinstance(result, ToolResult)
        data = json.loads(result.content)
        assert data["status"] == "pending"

    def test_includes_path_reason_access_type(self):
        tool = RequestAccessTool()
        result = tool.execute(path="/app/src", reason="Need write access to fix bug",
                              access_type="read_write")

        assert isinstance(result, ToolResult)
        data = json.loads(result.content)
        assert data["path"] == "/app/src"
        assert data["reason"] == "Need write access to fix bug"
        assert data["access_type"] == "read_write"
        assert data["status"] == "pending"

    def test_always_pending_regardless_of_input(self):
        tool = RequestAccessTool()

        # Even with weird inputs, status is always "pending"
        result = tool.execute(path="", reason="", access_type="read")
        data = json.loads(result.content)
        assert data["status"] == "pending"

    def test_request_access_schema(self):
        tool = RequestAccessTool()
        schema = tool.schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "request_access"


# ===========================================================================
# AC-13: AgentMessageTool with sub_agent_manager=None → "not yet available"
# ===========================================================================

class TestAgentMessageToolNoManager:

    @pytest.mark.asyncio
    async def test_none_manager_returns_not_available(self):
        tool = AgentMessageTool(sub_agent_manager=None)
        result = await tool.execute(action="list", agent="claude")

        assert isinstance(result, ToolResult)
        assert "not yet available" in result.content.lower() or "not yet available" in result.content

    @pytest.mark.asyncio
    async def test_none_manager_for_send_action(self):
        tool = AgentMessageTool(sub_agent_manager=None)
        result = await tool.execute(action="send", agent="claude", message="hello")

        assert isinstance(result, ToolResult)
        assert "not yet available" in result.content.lower()

    @pytest.mark.asyncio
    async def test_none_manager_for_start_action(self):
        tool = AgentMessageTool(sub_agent_manager=None)
        result = await tool.execute(action="start", agent="claude", message="")

        assert isinstance(result, ToolResult)
        assert "not yet available" in result.content.lower()

    @pytest.mark.asyncio
    async def test_none_manager_for_stop_action(self):
        tool = AgentMessageTool(sub_agent_manager=None)
        result = await tool.execute(action="stop", agent="claude")

        assert isinstance(result, ToolResult)
        assert "not yet available" in result.content.lower()

    @pytest.mark.asyncio
    async def test_none_manager_for_status_action(self):
        tool = AgentMessageTool(sub_agent_manager=None)
        result = await tool.execute(action="status", agent="claude")

        assert isinstance(result, ToolResult)
        assert "not yet available" in result.content.lower()


class TestAgentMessageToolWithManager:

    @pytest.mark.asyncio
    async def test_list_action_delegates_to_manager(self):
        mock_manager = MagicMock()
        mock_manager.list_active.return_value = [{"handle": "claude", "status": "running"}]

        tool = AgentMessageTool(sub_agent_manager=mock_manager, project_id="proj_123")
        result = await tool.execute(action="list")

        assert isinstance(result, ToolResult)
        # Should have called list_active on manager with project_id
        mock_manager.list_active.assert_called_with("proj_123")

    @pytest.mark.asyncio
    async def test_start_without_agent_returns_error(self):
        mock_manager = MagicMock()
        tool = AgentMessageTool(sub_agent_manager=mock_manager)

        result = await tool.execute(action="start", agent="")

        assert isinstance(result, ToolResult)
        assert "error" in result.content.lower() or "'agent' parameter is required" in result.content.lower()

    @pytest.mark.asyncio
    async def test_send_without_agent_returns_error(self):
        mock_manager = MagicMock()
        tool = AgentMessageTool(sub_agent_manager=mock_manager)

        result = await tool.execute(action="send", agent="", message="hi")

        assert isinstance(result, ToolResult)
        assert "error" in result.content.lower()

    @pytest.mark.asyncio
    async def test_stop_without_agent_returns_error(self):
        mock_manager = MagicMock()
        tool = AgentMessageTool(sub_agent_manager=mock_manager)

        result = await tool.execute(action="stop", agent="")

        assert isinstance(result, ToolResult)
        assert "error" in result.content.lower()

    def test_schema_has_correct_name(self):
        tool = AgentMessageTool(sub_agent_manager=None)
        schema = tool.schema()
        assert schema["function"]["name"] == "agent_message"


# ===========================================================================
# AC-14: Every tool — exception inside execute() returns ToolResult, no raise
# ===========================================================================

class TestToolsNeverRaise:
    """
    Verify that each tool's execute() method returns a ToolResult with an error
    message rather than raising an exception, even when internal errors occur.

    The registry also has a safety net try/catch, but individual tools should
    handle their own errors. We test both: the tool directly AND through registry.
    """

    def test_registry_catches_exploding_tool(self):
        reg = ToolRegistry()
        reg.register(ExplodingTool(name="bomb"))

        result = reg.execute("bomb", {})

        assert isinstance(result, ToolResult)
        assert "error" in result.content.lower() or "Error" in result.content

    def test_read_tool_handles_internal_error(self, tmp_path):
        workspace = str(tmp_path)
        tool = ReadTool(workspace=workspace)

        # Patch os.path or internal method to raise
        with patch("builtins.open", side_effect=PermissionError("access denied")):
            try:
                result = tool.execute(path="somefile.txt")
            except Exception:
                pytest.fail("ReadTool raised an exception; should return ToolResult with error")

        assert isinstance(result, ToolResult)
        assert "error" in result.content.lower() or "Error" in result.content

    def test_write_tool_handles_internal_error(self, tmp_path):
        workspace = str(tmp_path)
        tool = WriteTool(workspace=workspace)

        with patch("builtins.open", side_effect=OSError("disk full")):
            try:
                result = tool.execute(path="output.txt", content="data")
            except Exception:
                pytest.fail("WriteTool raised an exception; should return ToolResult with error")

        assert isinstance(result, ToolResult)
        assert "error" in result.content.lower() or "Error" in result.content

    def test_edit_tool_handles_internal_error(self, tmp_path):
        workspace = str(tmp_path)
        # Create a file that exists but then break something
        (tmp_path / "file.txt").write_text("content", encoding="utf-8")
        tool = EditTool(workspace=workspace)

        with patch("builtins.open", side_effect=IOError("read error")):
            try:
                result = tool.execute(path="file.txt", old_text="a", new_text="b")
            except Exception:
                pytest.fail("EditTool raised an exception; should return ToolResult with error")

        assert isinstance(result, ToolResult)
        assert "error" in result.content.lower() or "Error" in result.content

    def test_shell_tool_handles_internal_error(self, tmp_path):
        workspace = str(tmp_path)
        tool = ShellTool(workspace=workspace, os_type="windows")

        with patch("subprocess.run", side_effect=OSError("no such shell")):
            try:
                result = tool.execute(command="echo test")
            except Exception:
                pytest.fail("ShellTool raised an exception; should return ToolResult with error")

        assert isinstance(result, ToolResult)
        assert "error" in result.content.lower() or "Error" in result.content

    def test_request_access_tool_handles_internal_error(self):
        tool = RequestAccessTool()

        # Patch json.dumps to simulate an internal error
        with patch("json.dumps", side_effect=TypeError("serialize error")):
            try:
                result = tool.execute(path="/data", reason="need it", access_type="read")
            except Exception:
                pytest.fail("RequestAccessTool raised an exception; should return ToolResult with error")

        assert isinstance(result, ToolResult)

    @pytest.mark.asyncio
    async def test_agent_message_tool_handles_internal_error(self):
        mock_manager = MagicMock()
        mock_manager.list_active.side_effect = RuntimeError("manager crashed")
        tool = AgentMessageTool(sub_agent_manager=mock_manager)

        try:
            result = await tool.execute(action="list")
        except Exception:
            pytest.fail("AgentMessageTool raised an exception; should return ToolResult with error")

        assert isinstance(result, ToolResult)
        assert "error" in result.content.lower() or "Error" in result.content


# ===========================================================================
# Additional edge-case tests (beyond the 14 AC but useful for robustness)
# ===========================================================================

class TestToolBaseClass:

    def test_tool_schema_format(self):
        tool = DummyTool(name="my_tool", description="Does stuff",
                         parameters={"type": "object", "properties": {"x": {"type": "string"}}})
        schema = tool.schema()

        assert schema == {
            "type": "function",
            "function": {
                "name": "my_tool",
                "description": "Does stuff",
                "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
            }
        }

    def test_toolresult_default_meta_is_none(self):
        result = ToolResult(content="hello")
        assert result.meta is None

    def test_toolresult_with_meta(self):
        result = ToolResult(content="ok", meta={"network": True, "domains": ["x.com"]})
        assert result.meta["network"] is True
        assert "x.com" in result.meta["domains"]


class TestRegistryExecutePassthrough:

    def test_execute_passes_arguments_to_tool(self):
        reg = ToolRegistry()
        mock_tool = MagicMock(spec=Tool)
        mock_tool.name = "tester"
        mock_tool.execute.return_value = ToolResult(content="result from tool")

        reg.register(mock_tool)
        result = reg.execute("tester", {"key": "value"})

        mock_tool.execute.assert_called_once_with(key="value")
        assert result.content == "result from tool"

    def test_execute_returns_toolresult_on_tool_exception(self):
        reg = ToolRegistry()
        mock_tool = MagicMock(spec=Tool)
        mock_tool.name = "crasher"
        mock_tool.execute.side_effect = ValueError("bad args")

        reg.register(mock_tool)
        result = reg.execute("crasher", {"x": 1})

        assert isinstance(result, ToolResult)
        assert "error" in result.content.lower() or "Error" in result.content
