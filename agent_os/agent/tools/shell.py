# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""ShellTool — OS-aware command execution with output truncation and network meta."""

import asyncio
import os
import re
import subprocess
from uuid import uuid4

from .base import Tool, ToolResult

_TIMEOUT = 120
_MAX_LINES = 200
_HEAD_LINES = 20
_TAIL_LINES = 50
_HARD_CAP = 50_000

# Patterns for detecting network-related commands
_NETWORK_CMD_RE = re.compile(r'\b(curl|wget|npm|pip|git)\b')

# Pattern for extracting domains from URLs and bare domains
_DOMAIN_RE = re.compile(
    r'(?:https?://|@)([a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?)+)'
)
# Also match bare domain patterns like "curl example.com"
_BARE_DOMAIN_RE = re.compile(
    r'\b(?:curl|wget)\s+(?:-[^\s]*\s+)*([a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?(?:\.[a-zA-Z]{2,})(?:\.[a-zA-Z]{2,})?)\b'
)


class ShellTool(Tool):
    """Execute shell commands in the workspace with OS-aware shell selection."""

    # Patterns for detecting path-like strings in commands
    _WIN_ABS_RE = re.compile(r'[A-Za-z]:\\[^\s"\';&|>]+')
    _UNIX_ABS_RE = re.compile(r'/(?:home|Users|etc|var|tmp|root)/[^\s"\';&|>]*')
    _HOME_REL_RE = re.compile(r'~/[^\s"\';&|>]*')
    _ENV_VAR_RE = re.compile(r'(?:\$HOME|\$USERPROFILE|%USERPROFILE%|%APPDATA%|%LOCALAPPDATA%)')

    def __init__(self, workspace: str, os_type: str,
                 platform_provider=None, project_id: str | None = None,
                 project_dir_name: str = ""):
        self._workspace = workspace
        self._os_type = os_type
        self._platform_provider = platform_provider
        self._project_id = project_id
        self._project_dir_name = project_dir_name
        self.name = "shell"
        self.description = "Execute a shell command in the workspace."
        self.parameters = {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
            },
            "required": ["command"],
        }

    def _detect_external_paths(self, command: str) -> list[str]:
        """Extract path-like strings from command and return those outside the workspace."""
        candidates: list[str] = []
        candidates.extend(self._WIN_ABS_RE.findall(command))
        candidates.extend(self._UNIX_ABS_RE.findall(command))
        candidates.extend(self._HOME_REL_RE.findall(command))
        candidates.extend(self._ENV_VAR_RE.findall(command))

        workspace_norm = os.path.normcase(os.path.abspath(self._workspace))
        external: list[str] = []
        for path in candidates:
            # Environment variables and home-relative paths are always external
            if path.startswith('$') or path.startswith('%') or path.startswith('~'):
                if path not in external:
                    external.append(path)
                continue
            path_norm = os.path.normcase(os.path.abspath(path))
            if not path_norm.startswith(workspace_norm):
                if path not in external:
                    external.append(path)
        return external

    def _build_cmd(self, command: str) -> list[str]:
        """Build the shell command based on OS type."""
        if self._os_type == "windows":
            return ["powershell", "-NoProfile", "-Command", command]
        else:
            return ["bash", "-c", command]

    def _detect_network(self, command: str) -> dict:
        """Scan command for network-related tools and extract domains."""
        network = bool(_NETWORK_CMD_RE.search(command))
        domains: list[str] = []

        if network:
            # Extract domains from URLs (https://domain.com/...)
            for m in _DOMAIN_RE.finditer(command):
                domain = m.group(1)
                if domain not in domains:
                    domains.append(domain)

            # Extract bare domains (curl example.com)
            for m in _BARE_DOMAIN_RE.finditer(command):
                domain = m.group(1)
                if domain not in domains:
                    domains.append(domain)

        return {"network": network, "domains": domains}

    def _truncate_output(self, output: str, workspace: str) -> str:
        """Truncate output if it exceeds 200 lines. Save full output to tempfile."""
        lines = output.split("\n")
        if len(lines) <= _MAX_LINES:
            return output

        total = len(lines)

        # Save full output to tempfile
        if self._project_dir_name:
            output_dir = os.path.join(workspace, "orbital-output", self._project_dir_name, "shell-output")
        else:
            output_dir = os.path.join(workspace, "orbital-output", "shell-output")
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{uuid4().hex[:12]}.txt"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(output)

        # Build truncated output: first 20 + notice + last 50
        head = "\n".join(lines[:_HEAD_LINES])
        tail = "\n".join(lines[-_TAIL_LINES:])
        truncated = (
            f"{head}\n"
            f"... [truncated {total} lines, saved to {filepath}] ...\n"
            f"{tail}"
        )
        return truncated

    def _run_async(self, coro):
        """Run an async coroutine from sync context, handling both threaded and in-loop cases."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None:
            return asyncio.run(coro)
        else:
            # Already in an event loop (e.g., called from asyncio.to_thread or tests)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()

    def _execute_via_provider(self, command: str, meta: dict) -> ToolResult:
        """Execute command via platform provider (sandbox isolation)."""
        cmd = self._build_cmd(command)
        shell = cmd[0]
        args = cmd[1:]

        try:
            cmd_result = self._run_async(self._platform_provider.run_command(
                project_id=self._project_id,
                command=shell,
                args=args,
                working_dir=self._workspace,
                timeout_sec=_TIMEOUT,
            ))
        except RuntimeError as e:
            return ToolResult(content=f"Error: {e}", meta=meta)

        if cmd_result.timed_out:
            output = cmd_result.stdout
            if cmd_result.stderr:
                output = output + cmd_result.stderr if output else cmd_result.stderr
            output = self._truncate_output(output, self._workspace)
            return ToolResult(
                content=f"Error: command timed out after {_TIMEOUT} seconds\n{output}",
                meta=meta,
            )

        output = cmd_result.stdout
        if cmd_result.stderr:
            output = output + cmd_result.stderr if output else cmd_result.stderr

        output = self._truncate_output(output, self._workspace)

        if len(output) > _HARD_CAP:
            output = output[:_HARD_CAP] + "\n[OUTPUT TRUNCATED at 50,000 characters]"

        content = f"Exit code: {cmd_result.exit_code}\n{output}"
        return ToolResult(content=content, meta=meta)

    def _execute_via_subprocess(self, command: str, meta: dict) -> ToolResult:
        """Execute command via subprocess.run() (legacy/dev mode)."""
        cmd = self._build_cmd(command)

        try:
            proc = subprocess.run(
                cmd,
                cwd=self._workspace,
                capture_output=True,
                text=True,
                timeout=_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                content=f"Error: command timed out after {_TIMEOUT} seconds",
                meta=meta,
            )

        output = proc.stdout
        if proc.stderr:
            output = output + proc.stderr if output else proc.stderr

        output = self._truncate_output(output, self._workspace)

        if len(output) > _HARD_CAP:
            output = output[:_HARD_CAP] + "\n[OUTPUT TRUNCATED at 50,000 characters]"

        content = f"Exit code: {proc.returncode}\n{output}"
        return ToolResult(content=content, meta=meta)

    def execute(self, **arguments) -> ToolResult:
        try:
            command = arguments.get("command", "")
            meta = self._detect_network(command)
            external_paths = self._detect_external_paths(command)

            # Use provider path only when a real provider is set up (not NullProvider)
            if (self._platform_provider is not None
                    and self._platform_provider.get_capabilities().setup_complete):
                result = self._execute_via_provider(command, meta)
            else:
                result = self._execute_via_subprocess(command, meta)

            if external_paths:
                warning = (
                    f"[WARNING] Command references paths outside your workspace "
                    f"({self._workspace}): {', '.join(external_paths)}. "
                    f"Only workspace and portal paths are accessible.\n"
                )
                result = ToolResult(content=warning + result.content, meta=result.meta)

            return result
        except Exception as e:
            return ToolResult(
                content=f"Error: {str(e)}",
                meta={"network": False, "domains": []},
            )
