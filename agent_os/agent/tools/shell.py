# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""ShellTool — OS-aware command execution with output truncation and network meta."""

import asyncio
import os
import re
import subprocess
import sys
from uuid import uuid4

from agent_os.utils.subprocess_flags import win_no_window_flags

from .base import Tool, ToolResult

_TIMEOUT = 120
_MAX_LINES = 200
_HEAD_LINES = 20
_TAIL_LINES = 50
_HARD_CAP = 50_000

# Patterns for detecting network-related commands
_NETWORK_CMD_RE = re.compile(r'\b(curl|wget|npm|pip|git)\b')

# Markers proving the proxy rejected a domain on policy grounds.
_PROXY_BLOCK_MARKERS = (
    "received http code 403 from proxy",   # curl's CONNECT-refused message (older curl)
    "connect tunnel failed, response 403", # curl 8.x CONNECT-refused message
    "orbital network policy",              # our own 403 body (plain-HTTP path)
)

_NETWORK_POLICY_HINT = (
    "\n[network-policy] One or more domains were blocked by this project's "
    "network allowlist. A 403 from the proxy means policy, not a broken "
    "network — do not retry with workarounds. To read web content, use the "
    "browser tool; shell network is reserved for approved domains "
    "(package registries, LLM APIs, GitHub)."
    " If the task genuinely needs the domain, call request_network_access."
)

# --- Spec 077 §4.3 — say what the sandbox blocked --------------------------
#
# Each platform's denial string, lowercased. Matched against combined
# stdout+stderr of a command that already exited non-zero. String matching is
# imprecise on purpose (a tool that prints "Access is denied" for its own
# reasons gets the line too), which is why the hint says "may".
_SANDBOX_DENIAL_MARKERS = (
    "operation not permitted",   # macOS Seatbelt
    "access is denied",          # Windows ACLs
)

_SANDBOX_HINT = (
    "\n[sandbox] This command may have been blocked by the workspace sandbox. "
    "Writes are allowed in the workspace, read-write portals and temp; reads of "
    "credential stores are denied; network egress goes through the project's "
    "allowlist proxy. If you need a folder outside the workspace, ask the user "
    "to add it under Settings › Folder access, or call request_access."
)

# macOS violation lines look like:
#   Sandbox: ls(57603) deny(1) file-read-data /Users/me/.ssh
# parsed the way Codex's debug_sandbox/seatbelt.rs does. file-ioctl lines
# (path:… ioctl-command:…) are deliberately not matched — they are startup
# noise from bash touching /dev/dtracehelper, not the user's denial.
_SANDBOX_LOG_RE = re.compile(
    r"Sandbox:\s+\S+\(\d+\)\s+deny\(\d+\)\s+"
    r"(file-read\S*|file-write\S*|network\S*)\s+(\S.*?)\s*$"
)
_SANDBOX_LOG_TIMEOUT = 1.5

# The Sandbox log is machine-wide. Denials under these roots are background
# noise from other sandboxed processes (Bun, Spotlight, Chrome helpers writing
# scratch files) and are never a denial *this* profile would produce — temp,
# /dev and system reads are all allowed. Naming one of them would actively
# mislead, so a candidate under them is only used if nothing better was seen.
_SANDBOX_LOG_NOISE_ROOTS = (
    "/private/var/folders", "/var/folders", "/private/tmp", "/tmp",
    "/dev", "/System", "/Library", "/usr", "/opt", "/Applications",
)


def _parse_sandbox_denials(log_output: str) -> str | None:
    """Pick the denial most likely to be this command's, or ``None``.

    Prefers the most recent line whose path is not background noise; falls
    back to nothing rather than naming an unrelated process's scratch file.
    """
    best = None
    for line in log_output.splitlines():
        match = _SANDBOX_LOG_RE.search(line)
        if not match:
            continue
        path = match.group(2).strip()
        if any(path.startswith(root) for root in _SANDBOX_LOG_NOISE_ROOTS):
            continue
        best = f"{match.group(1)} {path}"
    return best


def _lookup_sandbox_denial() -> str | None:
    """Best-effort: name the path macOS just denied, or return ``None``.

    Time-bounded and failure-swallowing by contract — this runs only after a
    command has *already* failed, and must never turn a tool result into an
    error of its own. The log is machine-wide, so the line reported is the most
    recent denial seen, which is normally but not provably this command's.
    """
    if sys.platform != "darwin":
        return None
    try:
        proc = subprocess.run(
            [
                "/usr/bin/log", "show",
                "--last", "5s",
                "--style", "compact",
                "--predicate", 'sender == "Sandbox"',
            ],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=_SANDBOX_LOG_TIMEOUT,
            creationflags=win_no_window_flags(),
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return _parse_sandbox_denials(proc.stdout or "")


def _sandbox_hint(exit_code: int, output: str) -> str:
    """The ``[sandbox]`` suffix for a failed command, or an empty string."""
    if exit_code == 0:
        return ""
    lowered = output.lower()
    if not any(marker in lowered for marker in _SANDBOX_DENIAL_MARKERS):
        return ""
    hint = _SANDBOX_HINT
    denial = _lookup_sandbox_denial()
    if denial:
        hint += f" The most recent sandbox denial was: {denial}."
    return hint


# --- Spec 077 §4.7.3 — fewer false alarms in the focus nudge ---------------
#
# Paths under a system or toolchain root are a tool reading its own
# installation, not the agent exploring the machine. A warning that is usually
# wrong is ignored, so these never fire the nudge.
_TOOLCHAIN_ROOTS_POSIX = (
    "/usr", "/opt", "/bin", "/sbin", "/Library", "/System",
    "/Applications", "/dev", "/tmp", "/private",
)
# Home-relative toolchain installs — the same reasoning one level down
# (`~/.nvm/versions/node/v22/bin/node` is node reading node).
_TOOLCHAIN_ROOTS_HOME = (
    ".nvm", ".pyenv", ".rbenv", ".asdf", ".sdkman", ".volta",
    ".cargo", ".rustup", ".bun", ".deno", ".local/bin",
    ".npm-global", "go/bin", "Library/pnpm", "Library/Caches",
)
_TOOLCHAIN_ROOTS_WINDOWS = (
    r"c:\windows", r"c:\program files", r"c:\programdata",
)

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
                 platform_provider=None, project_id: str | None = None):
        self._workspace = workspace
        self._os_type = os_type
        self._platform_provider = platform_provider
        self._project_id = project_id
        self.name = "shell"
        self.description = "Execute a shell command in the workspace."
        self.parameters = {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
            },
            "required": ["command"],
        }

    def _is_toolchain_path(self, path: str) -> bool:
        """True when the path is a tool reading its own installation.

        Spec 077 §4.7.3 — the focus nudge is about *attention* (home, user
        folders, other projects), not about `/opt/homebrew/bin/node`.
        """
        lowered = os.path.normcase(path)
        if lowered.startswith("~/"):
            rest = path[2:]
            return any(
                rest == root or rest.startswith(root + "/")
                for root in _TOOLCHAIN_ROOTS_HOME
            )
        for root in _TOOLCHAIN_ROOTS_POSIX:
            if path == root or path.startswith(root + "/"):
                return True
        for root in _TOOLCHAIN_ROOTS_WINDOWS:
            if lowered.startswith(root):
                return True
        tmpdir = os.environ.get("TMPDIR")
        if tmpdir:
            tmp_norm = os.path.normcase(os.path.normpath(tmpdir))
            if lowered.startswith(tmp_norm):
                return True
        return False

    def _detect_external_paths(self, command: str) -> list[str]:
        """Extract path-like strings from command and return those outside the workspace."""
        candidates: list[str] = []
        candidates.extend(self._WIN_ABS_RE.findall(command))
        candidates.extend(self._UNIX_ABS_RE.findall(command))
        candidates.extend(self._HOME_REL_RE.findall(command))
        candidates.extend(self._ENV_VAR_RE.findall(command))

        workspace_norm = os.path.normcase(os.path.normpath(os.path.abspath(self._workspace)))
        external: list[str] = []
        for path in candidates:
            if self._is_toolchain_path(path):
                continue
            # Environment variables and home-relative paths are always external
            if path.startswith('$') or path.startswith('%') or path.startswith('~'):
                if path not in external:
                    external.append(path)
                continue
            path_norm = os.path.normcase(os.path.normpath(os.path.abspath(path)))
            # Trailing separator so a sibling directory that merely shares a
            # name prefix ("/ws-old" vs "/ws") is not read as inside.
            if path_norm != workspace_norm and not path_norm.startswith(
                workspace_norm + os.sep
            ):
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
        from agent_os.agent.project_paths import ProjectPaths
        output_dir = ProjectPaths(workspace).shell_output_dir
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
        lowered = output.lower()
        if meta.get("network") and any(m in lowered for m in _PROXY_BLOCK_MARKERS):
            content += _NETWORK_POLICY_HINT
        content += _sandbox_hint(cmd_result.exit_code, output)
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
                creationflags=win_no_window_flags(),
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
        content += _sandbox_hint(proc.returncode, output)
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
                # Spec 077 §4.7.2 — a focus rule, not a capability claim. Reads
                # outside the workspace now succeed at the OS level, so telling
                # the model they are "not accessible" would be a lie it can
                # disprove in one command, and then it stops trusting the rest.
                warning = (
                    f"[focus] This command references paths outside your "
                    f"workspace ({self._workspace}): {', '.join(external_paths)}. "
                    f"Files outside the workspace and granted portals are not "
                    f"project context — do not read or explore them. If you need "
                    f"that folder, use request_access.\n"
                )
                result = ToolResult(content=warning + result.content, meta=result.meta)

            return result
        except Exception as e:
            return ToolResult(
                content=f"Error: {str(e)}",
                meta={"network": False, "domains": []},
            )
