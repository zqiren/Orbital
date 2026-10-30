# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""ShellTool — OS-aware command execution with output truncation and network meta."""

import asyncio
import ntpath
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Callable
from uuid import uuid4

from agent_os.platform.proc_runner import run_foreground
from agent_os.utils.subprocess_flags import win_no_window_flags

from .base import Tool, ToolResult

_TIMEOUT = 120
_MAX_LINES = 200
_HEAD_LINES = 20
_TAIL_LINES = 50
_HARD_CAP = 50_000

# Spec 134 A1 — appended to "timed out after N seconds". A timeout now ends the
# whole process tree (it used to end only the shell and, on Windows, block on
# the grandchildren for as long as they ran), so say so, and point long work
# elsewhere than a foreground call.
_TIMEOUT_HINT = (
    " and was stopped, together with every process it started. If this is "
    "long-running work (a server, a watcher, a long build or script), run it "
    "with `background: true` instead of waiting for it here: the call returns "
    "at once and Orbital messages you when it ends."
)

# Spec 134 Part B — what the agent reads about background jobs.
_DESCRIPTION = "Execute a shell command in the workspace."
_BACKGROUND_DESCRIPTION = (
    "Execute a shell command in the workspace. A foreground call is stopped "
    f"after {_TIMEOUT} seconds. For anything that can take longer (a build, a "
    "test suite, training, a download, a server, a long script) pass "
    "`background: true`: the call returns at once with a job id and its output "
    "file, your turn ends after this message's tool calls, and Orbital sends you "
    "the exit code and the last output lines when it ends (and checks in every "
    "15 minutes while it runs). Do not sleep or poll to wait for it. Orbital "
    "tracks only the process this call starts. Anything it detaches (`nohup`, "
    "`&`, a daemon, work left running on a server after `ssh` returns) is not "
    "tracked and gets no message, so keep the work in the foreground of this "
    "command. A foreground call that detaches something isn't tracked either."
)
_BACKGROUND_PARAM = {
    "type": "boolean",
    "description": (
        "Run as a background job instead of waiting. Several background calls "
        "in one message form one group: you are woken when all of them have "
        "ended, or at once when one fails."
    ),
}
_STARTED_RE = re.compile(r"Started background job (job_[0-9a-f]+)")

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
    "拒绝访问",                   # Windows ACLs, zh-CN (spec 117 D4)
)
# Spec 117 D4 — per-kind markers, lowercased. The Windows set carries the
# zh-CN wording of ERROR_ACCESS_DENIED alongside the English one; other
# localizations are a follow-up (spec 117 §6).
_SEATBELT_DENIAL_MARKERS = ("operation not permitted",)
_WINDOWS_DENIAL_MARKERS = ("access is denied", "拒绝访问")
# PowerShell's cmdlet wording (bench 2026-10-19, issue 3): ``Access to the
# path 'X' is denied.`` / ``对路径“X”的访问被拒绝。``. Neither contains the
# cmd-style text above, so most Windows denials went unnoticed.
_WINDOWS_PATH_DENIAL_RE = re.compile(
    r"access to the path .{1,400}? is denied|的访问被拒绝", re.IGNORECASE)

#: Printed to stderr by the Windows script wrapper when any PowerShell error
#: record of the command is a permission denial. Language-independent, and
#: set even when a later statement succeeded (``exit 0``). Stripped before the
#: output reaches the agent.
ACCESS_DENIED_MARKER = "[orbital:access-denied]"

_SANDBOX_HINT = (
    "\n[sandbox] This command may have been blocked by the workspace sandbox "
    "(safe mode), not by the file itself: the path was not granted to the "
    "sandbox. It does not mean the file is secret or was deleted. Writes are "
    "allowed in the workspace, read-write portals and temp; network egress "
    "goes through the project's allowlist proxy. If you need a folder outside "
    "the workspace, ask the user to add it under Settings › Folder access, or "
    "call request_access."
)

# Spec 117 D4/D5 — appended in safe mode when the daemon raises an
# ``unsandboxed_run`` approval for this command: the user may answer it.
_SANDBOX_WAIT_HINT = (
    " You may wait for the user's decision on running this outside the sandbox."
)

# The open path (spec 117 D3) runs as the user with the user's network. A
# provider call made earlier in this process may have exported the allowlist
# proxy into the daemon's own environment; the child must never inherit it.
_PROXY_ENV_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY")


def _timeout_content(output: str) -> str:
    """The tool result for a timed-out command: the hint, then what it printed."""
    if len(output) > _HARD_CAP:
        output = output[:_HARD_CAP] + "\n[OUTPUT TRUNCATED at 50,000 characters]"
    content = f"Error: command timed out after {_TIMEOUT} seconds{_TIMEOUT_HINT}"
    if output:
        content += f"\nOutput before the timeout:\n{output}"
    return content


@dataclass(frozen=True)
class SandboxDenial:
    """A failed command the daemon attributes to the sandbox (spec 117 D4).

    ``kind`` is ``"seatbelt"`` (macOS Seatbelt denial), ``"windows_acl"``
    (the worker account was refused a path) or ``"proxy"`` (the allowlist
    proxy returned 403). ``reason`` is one sentence for the approval card.
    """
    kind: str
    reason: str


def detect_sandbox_denial(exit_code: int, output: str) -> SandboxDenial | None:
    """Decide whether a non-zero exit was the sandbox's doing, or ``None``.

    String matching is imprecise on purpose (a tool that prints "Access is
    denied" for its own reasons is flagged too), which is why the hint says
    "may" — the user, not the daemon, makes the call on the approval card.
    The Seatbelt log lookup is macOS-only and best-effort (``None`` off-mac).
    """
    if output and ACCESS_DENIED_MARKER in output:
        return SandboxDenial(
            kind="windows_acl",
            reason="Access was denied on a path the sandboxed worker was not granted.",
        )
    if exit_code == 0 or not output:
        return None
    lowered = output.lower()
    if any(marker in lowered for marker in _PROXY_BLOCK_MARKERS):
        return SandboxDenial(
            kind="proxy",
            reason="The project's network allowlist proxy refused a connection (403).",
        )
    if any(marker in lowered for marker in _SEATBELT_DENIAL_MARKERS):
        denial = _lookup_sandbox_denial()
        if denial:
            return SandboxDenial(
                kind="seatbelt", reason=f"The sandbox denied {denial}.")
        return SandboxDenial(
            kind="seatbelt",
            reason="The command reported 'Operation not permitted' inside the sandbox.",
        )
    if (any(marker in lowered for marker in _WINDOWS_DENIAL_MARKERS)
            or _WINDOWS_PATH_DENIAL_RE.search(output)):
        return SandboxDenial(
            kind="windows_acl",
            reason="Access was denied on a path the sandboxed worker was not granted.",
        )
    return None

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


_INTERPRETERS = frozenset({
    "cmd", "powershell", "pwsh", "bash", "sh", "zsh", "dash", "fish", "wsl",
})
_COMPOUND_CHARS = frozenset(";&|(){}\n\r")
#: Prefix assignments that change which program runs, or load code into it.
#: ``PATH=./evil ls`` runs the agent's ``ls``, not the one the user allowed.
_PROGRAM_CHANGING_VARS = frozenset({"PATH", "PATHEXT"})
_PROGRAM_CHANGING_PREFIXES = ("LD_", "DYLD_")


def _changes_program(name: str) -> bool:
    upper = name.upper()
    return upper in _PROGRAM_CHANGING_VARS or upper.startswith(_PROGRAM_CHANGING_PREFIXES)


def _same_file(a: str, b: str) -> bool:
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def _program_key(token: str) -> str:
    """The allowlist key for the program *token* names.

    A bare name keys as itself. A path keys as the bare name only when it is
    the program that name runs (``/usr/bin/git`` when PATH's ``git`` is it);
    otherwise as written, so the card names the program that runs:
    ``./evil/ls`` is not ``ls``.
    """
    name = os.path.basename(token.replace("\\", "/"))
    if name == token:
        return token
    if os.path.isabs(token) or ntpath.isabs(token):
        found = shutil.which(name)
        if found and _same_file(found, token):
            return name
    return token


def _is_compound(command: str) -> bool:
    """True when *command* chains, pipes or embeds another command.

    ``$(``, backticks and ``@(`` evaluate even inside double quotes (in bash
    and PowerShell alike), so they count anywhere; the other characters count
    outside quotes.
    """
    if "$(" in command or "`" in command or "@(" in command:
        return True
    quote = None
    for ch in command:
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch in _COMPOUND_CHARS:
            return True
    return False


def _split_command(command: str) -> list[str]:
    """Tokens of *command* with their quotes removed.

    Windows paths keep their backslashes (POSIX ``shlex`` would eat them).
    """
    posix = "\\" not in command
    try:
        tokens = shlex.split(command, posix=posix)
    except ValueError:
        tokens = command.split()
    if not posix:
        tokens = [t[1:-1] if len(t) >= 2 and t[0] == t[-1] and t[0] in "'\"" else t
                  for t in tokens]
    return tokens


def _interpreter_payload(stem: str, rest: list[str]) -> str | None:
    """The command an interpreter invocation runs, or ``None`` if not one."""
    lowered = [t.lower() for t in rest]
    if stem == "cmd":
        for flag in ("/c", "/k"):
            if flag in lowered:
                return " ".join(rest[lowered.index(flag) + 1:]) or None
        return None
    if stem in ("powershell", "pwsh"):
        for i, token in enumerate(lowered):
            if token in ("-command", "-c"):
                return " ".join(rest[i + 1:]) or None
            if token in ("-file", "-f", "-encodedcommand", "-e", "-ec"):
                return None
        return None
    if stem == "wsl":
        return None
    if "-c" in lowered:
        i = lowered.index("-c")
        return rest[i + 1] if i + 1 < len(rest) else None
    return None


def build_windows_script(command: str) -> str:
    """The ``.ps1`` text that runs *command* the way ``powershell -Command`` did.

    ``-File`` exits 0 unless the script throws or calls ``exit``, while
    ``-Command`` exited 1 when the last statement failed; the trailer keeps
    the ``-Command`` behaviour, except that a failed native program's own
    exit code is kept instead of 1, as on POSIX (batch 1026 Windows pass:
    ``SystemExit(7)`` was reported as exit 1). ``$?`` is read on the line
    straight after the command, before the trailer's own statements reset
    it; ``$LASTEXITCODE`` is cleared before the command so a value left
    over from before it is never reported. The marker is
    printed when any error record is a permission denial, independent of the
    UI language and of the exit code. The first line makes the redirected
    output UTF-8: PowerShell 5.1 otherwise encodes it in the OEM code page
    (437 on an en-US system locale), where CJK becomes ``?`` (bench
    2026-10-20). It also covers native programs the command runs.
    """
    return (
        "try { $OutputEncoding = [Console]::OutputEncoding = "
        "New-Object System.Text.UTF8Encoding $false } catch {}\n"
        "$global:LASTEXITCODE = 0\n"
        f"{command}\n"
        "$__orbital_ok = $?\n"
        "$__orbital_code = $LASTEXITCODE\n"
        "if ($Error | Where-Object { $_.CategoryInfo.Category -eq 'PermissionDenied' "
        "-or $_.Exception -is [System.UnauthorizedAccessException] }) {\n"
        f"    [Console]::Error.WriteLine('{ACCESS_DENIED_MARKER}')\n"
        "}\n"
        "if (-not $__orbital_ok) { if ($__orbital_code) { exit $__orbital_code } else { exit 1 } }\n"
    )


def _strip_marker(output: str) -> str:
    """*output* without the wrapper's marker line(s)."""
    if not output or ACCESS_DENIED_MARKER not in output:
        return output
    return "\n".join(
        line for line in output.split("\n") if line.strip() != ACCESS_DENIED_MARKER
    )


def _sandbox_hint(exit_code: int, output: str,
                  denial: SandboxDenial | None = None) -> str:
    """The ``[sandbox]`` suffix for a failed command, or an empty string.

    Proxy blocks get their own ``[network-policy]`` hint at the call site and
    no ``[sandbox]`` line — unchanged from before the detector existed.
    """
    if denial is None:
        denial = detect_sandbox_denial(exit_code, output)
    if denial is None or denial.kind == "proxy":
        return ""
    hint = _SANDBOX_HINT
    if denial is not None and denial.kind == "seatbelt" and denial.reason.startswith(
            "The sandbox denied "):
        hint += " The most recent sandbox denial was: " + denial.reason[
            len("The sandbox denied "):]
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
    # A whole quoted absolute path, spaces included: "C:\a b\c" or '/Users/a b'.
    _QUOTED_ABS_RE = re.compile(
        r'(["\'])((?:[A-Za-z]:\\|/(?:home|Users|etc|var|tmp|root)/)[^"\']*)\1')
    # What may follow the workspace path for it to be the workspace (or a path
    # under it) rather than a sibling sharing its name as a prefix.
    _WS_BOUNDARY = r'(?=$|[\\/\s"\';&|>)])'

    def _mask_workspace(self, command: str) -> str:
        """Blank out the workspace path wherever it appears, so its pieces are
        never scanned as separate paths. Case-insensitive on Windows."""
        workspace = os.path.normpath(self._workspace or "")
        if not workspace or workspace in (os.sep, "."):
            return command
        flags = re.IGNORECASE if self._os_type == "windows" else 0
        variants = {workspace, workspace.replace("\\", "/")}
        for variant in sorted(variants, key=len, reverse=True):
            command = re.sub(re.escape(variant) + self._WS_BOUNDARY, "", command,
                             flags=flags)
        return command

    def __init__(self, workspace: str, os_type: str,
                 platform_provider=None, project_id: str | None = None,
                 safe_mode_cb: Callable[[], bool] | None = None,
                 unsandboxed_cb: Callable[[], list[str]] | None = None,
                 jobs=None, session_id: str | None = None):
        self._workspace = workspace
        self._os_type = os_type
        self._platform_provider = platform_provider
        self._project_id = project_id
        # Spec 117 §3.2 — project config read PER CALL (like ``read_roots``):
        # ``safe_mode_cb()`` False → the open path; an executable named by
        # ``unsandboxed_cb()`` runs open even in safe mode (D5 "always
        # allow"). Both None → the legacy rule (provider when one is set up).
        self._safe_mode_cb = safe_mode_cb
        self._unsandboxed_cb = unsandboxed_cb
        # Spec 134 Part B: the daemon's JobRegistry and this tool's session.
        # Without them (fanout workers, tests) there is no ``background``.
        self._jobs = jobs
        self._session_id = session_id
        self.name = "shell"
        self.description = _BACKGROUND_DESCRIPTION if jobs is not None else _DESCRIPTION
        self.parameters = {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
            },
            "required": ["command"],
        }
        if jobs is not None:
            self.parameters["properties"]["background"] = dict(_BACKGROUND_PARAM)

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
        """Extract path-like strings from command and return those outside the workspace.

        The unquoted patterns stop at whitespace, so a path with a space was
        cut at it: for workspace ``...\\My Project`` the command
        ``dir "...\\My Project"`` read as ``...\\My``, outside, and the
        ``[focus]`` warning told the agent not to read its own project.
        Occurrences of the workspace itself are masked first (only at a path
        boundary, so ``My Project-old`` still counts), and quoted paths are
        then taken whole.
        """
        command = self._mask_workspace(command)
        candidates: list[str] = []
        for match in self._QUOTED_ABS_RE.finditer(command):
            candidates.append(match.group(2))
        command = self._QUOTED_ABS_RE.sub(" ", command)
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

    def _build_cmd(self, command: str, *,
                   as_user: bool = False) -> tuple[list[str], str | None]:
        """The argv for *command*, plus the script file to delete afterwards.

        Windows runs the command from a UTF-8 ``.ps1``: passed inline it
        crossed two layers of quoting (``cmd /c "… -Command <cmd> …"``) that
        dropped inner double quotes, and ``CreateProcessWithLogonW`` caps the
        command line at 1024 characters (bench 2026-10-19, issue 8).

        A sandboxed run's script goes in the workspace's ``orbital/.tmp``,
        where the worker can read it. A run *as the user* (``as_user``) uses
        the user's own temp folder instead: the worker can write the
        workspace, and a background worker process could swap the script
        between this write and PowerShell reading it, running its own code
        with the user's rights in place of the command the user approved.
        """
        if self._os_type != "windows":
            return ["bash", "-c", command], None
        if as_user:
            fd, script = tempfile.mkstemp(prefix="orbital_cmd_", suffix=".ps1")
            f = os.fdopen(fd, "w", encoding="utf-8-sig", newline="\r\n")
        else:
            from agent_os.agent.project_paths import ProjectPaths

            tmp_dir = ProjectPaths(self._workspace).tmp_dir
            os.makedirs(tmp_dir, exist_ok=True)
            script = os.path.join(tmp_dir, f"cmd_{uuid4().hex[:12]}.ps1")
            f = open(script, "w", encoding="utf-8-sig", newline="\r\n")
        with f:
            f.write(build_windows_script(command))
        return ["powershell", "-NoProfile", "-NonInteractive",
                "-ExecutionPolicy", "Bypass", "-File", script], script

    @staticmethod
    def _remove_script(script: str | None) -> None:
        if script:
            try:
                os.remove(script)
            except OSError:
                pass

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
        cmd, script = self._build_cmd(command)
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
        finally:
            self._remove_script(script)

        if cmd_result.timed_out:
            output = cmd_result.stdout
            if cmd_result.stderr:
                output = output + cmd_result.stderr if output else cmd_result.stderr
            output = _strip_marker(self._truncate_output(output, self._workspace))
            return ToolResult(content=_timeout_content(output), meta=meta)

        output = cmd_result.stdout
        if cmd_result.stderr:
            output = output + cmd_result.stderr if output else cmd_result.stderr

        # Detect on the raw output (the wrapper's marker), then hide the marker.
        denial = detect_sandbox_denial(cmd_result.exit_code, output)
        sandbox_hint = _sandbox_hint(cmd_result.exit_code, output, denial)
        output = _strip_marker(output)
        output = self._truncate_output(output, self._workspace)

        if len(output) > _HARD_CAP:
            output = output[:_HARD_CAP] + "\n[OUTPUT TRUNCATED at 50,000 characters]"

        content = f"Exit code: {cmd_result.exit_code}\n{output}"
        lowered = output.lower()
        if meta.get("network") and any(m in lowered for m in _PROXY_BLOCK_MARKERS):
            content += _NETWORK_POLICY_HINT
        content += sandbox_hint
        if denial is not None and self._safe_mode_cb is not None:
            # Spec 117 D4: the daemon raises an ``unsandboxed_run`` approval
            # off this meta (agent_manager / autonomy). Only when the
            # callbacks are wired — a bare tool (tests, dev daemon) has no
            # one to answer the card.
            meta = {**meta, "sandbox_denial": {
                "command": command, "reason": denial.reason, "kind": denial.kind,
            }}
            content += _SANDBOX_WAIT_HINT
        return ToolResult(content=content, meta=meta)

    def _execute_unsandboxed(self, command: str, meta: dict) -> ToolResult:
        """Run the command as the user, outside the sandbox (spec 117 D3).

        Inherited environment minus the allowlist-proxy variables, no
        Seatbelt, no worker account, no portals. Also the dev-daemon path
        (no platform provider wired), under its old name below.
        """
        cmd, script = self._build_cmd(command, as_user=True)
        env = {
            k: v for k, v in os.environ.items()
            if k.upper() not in _PROXY_ENV_VARS
        }

        # Spec 134 A1: output to a file in the user's temp folder (as-user,
        # same rule as the script above), own process group, tree kill on
        # timeout only, bounded read-back. A truncated log is kept under the
        # workspace's shell-output folder for the agent to ``read``.
        from agent_os.agent.project_paths import ProjectPaths

        try:
            run = run_foreground(
                cmd,
                cwd=self._workspace,
                timeout=_TIMEOUT,
                env=env,
                log_dir=None,
                keep_dir=ProjectPaths(self._workspace).shell_output_dir,
            )
        finally:
            self._remove_script(script)

        output = _strip_marker(run.output.render(run.log_path))
        if run.timed_out:
            return ToolResult(content=_timeout_content(output), meta=meta)

        if len(output) > _HARD_CAP:
            output = output[:_HARD_CAP] + "\n[OUTPUT TRUNCATED at 50,000 characters]"

        # No ``[sandbox]`` hint here: outside the sandbox an EPERM is the OS
        # talking, and naming the sandbox would send the agent (and the user)
        # after a wall that is not there.
        content = f"Exit code: {run.exit_code}\n{output}"
        return ToolResult(content=content, meta=meta)

    # Old name, kept for the dev-daemon path and existing callers.
    _execute_via_subprocess = _execute_unsandboxed

    # --- Spec 134 Part B — background jobs --------------------------------

    def _execute_background(self, command: str, meta: dict) -> ToolResult:
        """Start *command* as a background job and return at once.

        Open path (safe mode off, the default for new projects): Orbital spawns
        it detached through the A1 runner (own process group; on Windows a
        kill-on-close Job Object held for the app's lifetime). Sandboxed: the
        workspace launcher (``orbital/bin/bg``) is run through the provider,
        so the job inherits the sandbox and detaches inside it.
        """
        from agent_os.agent.loop import current_tool_batch

        refusal = self._jobs.admission(self._project_id or "", self._workspace)
        if refusal:
            return ToolResult(content=refusal, meta=meta)
        group = self._jobs.open_group(
            self._project_id or "", self._session_id or "", self._workspace,
            batch_key=current_tool_batch())
        sandboxed = not self._runs_unsandboxed(command) and (
            self._platform_provider is not None
            and self._platform_provider.get_capabilities().setup_complete)
        if sandboxed:
            job, failure = self._start_via_launcher(command, group, meta)
            if job is None:
                return failure
        else:
            cmd, _script = self._build_cmd(command, as_user=True)
            env = {
                k: v for k, v in os.environ.items()
                if k.upper() not in _PROXY_ENV_VARS
            }
            try:
                job = self._jobs.spawn_job(group, command, cmd, env=env)
            except OSError as e:
                self._remove_script(_script)
                return ToolResult(content=f"Error: the background job could not start: {e}",
                                  meta=meta)
            # PowerShell has read a ``-File`` script once it runs; the script
            # (Windows only) goes with the job's other scratch.
            if _script:
                job.scratch.append(_script)
        return ToolResult(content=self._jobs.start_text(job), meta={
            **meta,
            "yield_after_batch": True,
            "background_job": {
                "job_id": job.id, "group_id": job.group_id,
                "project_id": job.project_id, "session_id": job.session_id,
                "command": job.command, "pid": job.pid,
                "started_at_ms": int(job.started_at * 1000),
            },
        })

    def _start_via_launcher(self, command: str, group, meta: dict):
        """Run the workspace launcher through the sandboxed provider path."""
        from agent_os.agent.project_paths import ProjectPaths
        from agent_os.daemon_v2.job_launcher import ensure_launcher

        ensure_launcher(self._workspace)
        bin_dir = ProjectPaths(self._workspace).bin_dir
        script = None
        launched = command
        if self._os_type == "windows":
            # The launcher runs its command with cmd.exe; every other call of
            # this tool runs PowerShell, so hand it the same .ps1 wrapper
            # (batch 1026 Windows pass: `Write-Output` was "not recognized").
            argv, script = self._build_cmd(command)
            launched = subprocess.list2cmdline(argv)
        assignments = {
            "ORBITAL_SESSION_ID": group.session_id,
            "ORBITAL_PROJECT_ID": group.project_id,
            "ORBITAL_JOB_GROUP": group.id,
        }
        if self._os_type == "windows":
            def ps(v: str) -> str:
                return "'" + v.replace("'", "''") + "'"
            launch = "; ".join(f"$env:{k}={ps(v)}" for k, v in assignments.items())
            launch += f"; & {ps(os.path.join(bin_dir, 'bg.ps1'))} {ps(launched)}"
        else:
            launch = " ".join(f"{k}={shlex.quote(v)}" for k, v in assignments.items())
            launch += f" {shlex.quote(os.path.join(bin_dir, 'bg'))} {shlex.quote(command)}"
        result = self._execute_via_provider(launch, meta)
        match = _STARTED_RE.search(result.content if isinstance(result.content, str) else "")
        job = None
        if match:
            job = self._jobs.adopt(ProjectPaths(self._workspace).job_dir(match.group(1)),
                                   group=group, project_id=self._project_id,
                                   command=command)
        if job is not None and script:
            job.scratch.append(script)
        if job is None:
            self._remove_script(script)
            return None, ToolResult(
                content=f"Error: the background job could not start:\n{result.content}",
                meta=result.meta)
        return job, None

    @staticmethod
    def executable_of(command: str, _depth: int = 0) -> str:
        """The program "always allow" records for *command* (spec 117 D5).

        ``FOO=1 git push`` → ``git``; ``/usr/bin/git`` → ``git`` when that is
        PATH's ``git``. Interpreter wrappers are unwrapped to the program they
        run (``cmd /c curl.exe …`` → ``curl.exe``), because allowlisting
        ``cmd`` would allow everything (bench 2026-10-19, issue 7).

        The key names the program that actually runs, so the card never says
        ``ls`` for the agent's own ``./evil/ls`` (keyed as ``./evil/ls``), and
        a ``PATH=`` / ``LD_*`` / ``DYLD_*`` prefix gets no key at all.

        Empty — nothing to allowlist — when the command is not one program:
        chained, piped or carrying a sub-expression (``git status; X`` would
        otherwise let ``X`` ride out of the sandbox on ``git``), a bare
        interpreter, or a script file.
        """
        command = command.strip()
        if not command or _depth > 3 or _is_compound(command):
            return ""
        tokens = _split_command(command)
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if "=" in token and not token.startswith("="):
                name = token.split("=", 1)[0]
                if name and all(c.isalnum() or c == "_" for c in name):
                    if _changes_program(name):
                        return ""
                    index += 1
                    continue  # an environment assignment prefix
            break
        if index >= len(tokens):
            return ""
        program = _program_key(tokens[index])
        stem = program.lower().removesuffix(".exe")
        if stem not in _INTERPRETERS:
            return program
        inner = _interpreter_payload(stem, tokens[index + 1:])
        if inner is None:
            return ""
        return ShellTool.executable_of(inner, _depth + 1)

    def _safe_mode_on(self) -> bool:
        """The project's switch; on when unknown or unreadable (fails closed)."""
        if self._safe_mode_cb is None:
            return True
        try:
            return bool(self._safe_mode_cb())
        except Exception:
            return True

    def _runs_unsandboxed(self, command: str) -> bool:
        """Spec 117 §3.2 routing: open project, or an always-allowed executable."""
        if self._safe_mode_cb is None:
            return False
        try:
            if not self._safe_mode_cb():
                return True
        except Exception:
            return False  # an unreadable switch fails closed (sandboxed)
        if self._unsandboxed_cb is None:
            return False
        try:
            allowed = self._unsandboxed_cb() or []
        except Exception:
            return False
        exe = self.executable_of(command)
        return bool(exe) and exe in allowed

    def execute(self, **arguments) -> ToolResult:
        try:
            command = arguments.get("command", "")
            meta = self._detect_network(command)
            external_paths = self._detect_external_paths(command)

            # Spec 117: the project's switch decides first; then the legacy
            # rule — provider path only when a real provider is set up (not
            # NullProvider).
            if arguments.get("background") is True and self._jobs is not None:
                result = self._execute_background(command, meta)
            elif self._runs_unsandboxed(command):
                result = self._execute_unsandboxed(command, meta)
            elif (self._platform_provider is not None
                    and self._platform_provider.get_capabilities().setup_complete):
                result = self._execute_via_provider(command, meta)
            else:
                result = self._execute_unsandboxed(command, meta)

            if external_paths and self._safe_mode_on():
                # Spec 077 §4.7.2 — a focus rule, not a capability claim.
                # Safe mode only: an open project's agent runs with the
                # user's permissions and was asked to go there (bench
                # 2026-10-19, issue 6). Reads
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
