# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Seatbelt (sandbox-exec) profile generation for macOS process isolation."""

# TODO: Add violation monitoring via 'log stream --predicate ...'

from __future__ import annotations

import os

# Sensitive paths under the user's home directory that should be denied
# even though the base profile allows broad read access.
_SENSITIVE_RELATIVE_PATHS = (
    ".ssh",
    ".gnupg",
    ".aws",
    ".config",
    ".bashrc",
    ".zshrc",
    ".profile",
    ".git-credentials",
)

# Paths that must never be writable, regardless of portal configuration.
# Matched by subpath or literal within any allowed tree.
_MANDATORY_DENY_WRITE_PATTERNS = (
    ".bashrc",
    ".zshrc",
    ".profile",
    ".git/hooks",
)


def generate_profile(
    workspace_path: str,
    portal_paths: dict[str, str] | None = None,  # {path: "read_only" | "read_write"}
    network_proxy_port: int | None = None,
) -> str:
    """Generate a Seatbelt (SBPL) profile string for sandboxing a child process.

    Args:
        workspace_path: Absolute path to the agent workspace directory.
            The process will have full read/write access here.
        portal_paths: Optional mapping of absolute paths to access levels.
            ``"read_only"`` portals allow reads (already broadly permitted)
            but explicitly deny writes.  ``"read_write"`` portals additionally
            allow writes to the given path.
        network_proxy_port: If provided, all outbound network access is denied
            except to ``localhost:<proxy_port>`` and local Unix sockets.  When
            ``None``, no network restrictions are applied.

    Returns:
        A complete SBPL profile as a multi-line string suitable for passing
        to ``sandbox-exec -p``.
    """
    home = os.path.expanduser("~")
    lines: list[str] = []

    # ------------------------------------------------------------------
    # 1. Version & default-deny
    # ------------------------------------------------------------------
    lines.append("(version 1)")
    lines.append("(deny default)")

    # ------------------------------------------------------------------
    # 2. Base process permissions (required for process to start on
    #    macOS Sequoia without crashing)
    # ------------------------------------------------------------------
    lines.append("(allow process-exec*)")
    lines.append("(allow process-fork)")
    lines.append("(allow signal)")
    lines.append("(allow sysctl-read)")
    lines.append("(allow mach-lookup)")
    lines.append("(allow ipc-posix-shm-read-data)")
    lines.append("(allow ipc-posix-shm-write-data)")

    # ------------------------------------------------------------------
    # 3. Base filesystem read (system paths required for process startup)
    # ------------------------------------------------------------------
    lines.append(
        '(allow file-read* (subpath "/usr") (subpath "/Library") (subpath "/System")'
        ' (subpath "/private") (subpath "/dev") (subpath "/bin")'
        ' (subpath "/sbin") (subpath "/Applications") (literal "/"))'
    )

    # ------------------------------------------------------------------
    # 4. Base filesystem write
    # ------------------------------------------------------------------
    lines.append('(allow file-write* (subpath "/dev"))')

    # ------------------------------------------------------------------
    # 5. Workspace — full read/write
    # ------------------------------------------------------------------
    lines.append(f'(allow file-read* (subpath "{workspace_path}"))')
    lines.append(f'(allow file-write* (subpath "{workspace_path}"))')

    # ------------------------------------------------------------------
    # 6. Portal paths
    # ------------------------------------------------------------------
    if portal_paths:
        for path, access in portal_paths.items():
            if access == "read_only":
                # Read is already broadly allowed; explicitly deny write.
                lines.append(
                    f'(deny file-write* (subpath "{path}")'
                    ' (with message "orbital:portal-read-only"))'
                )
            elif access == "read_write":
                lines.append(f'(allow file-write* (subpath "{path}"))')

    # ------------------------------------------------------------------
    # 7. Sensitive path denies (deny read even though base allows it)
    # ------------------------------------------------------------------
    for rel in _SENSITIVE_RELATIVE_PATHS:
        abs_path = os.path.join(home, rel)
        lines.append(
            f'(deny file-read* (subpath "{abs_path}")'
            ' (with message "orbital:sensitive-path"))'
        )

    # ------------------------------------------------------------------
    # 8. Mandatory deny-write paths (always enforced, regardless of portals)
    # ------------------------------------------------------------------
    for pattern in _MANDATORY_DENY_WRITE_PATTERNS:
        abs_path = os.path.join(home, pattern)
        lines.append(
            f'(deny file-write* (subpath "{abs_path}")'
            ' (with message "orbital:mandatory-deny"))'
        )

    # ------------------------------------------------------------------
    # 9. Temp directories
    # ------------------------------------------------------------------
    lines.append('(allow file-read* (subpath "/tmp"))')
    lines.append('(allow file-write* (subpath "/tmp"))')
    lines.append('(allow file-read* (subpath "/private/tmp"))')
    lines.append('(allow file-write* (subpath "/private/tmp"))')
    lines.append('(allow file-read* (subpath "/var/folders/"))')
    lines.append('(allow file-write* (subpath "/var/folders/"))')

    tmpdir = os.environ.get("TMPDIR")
    if tmpdir and tmpdir not in ("/tmp", "/private/tmp") and not tmpdir.startswith("/var/folders/"):
        lines.append(f'(allow file-read* (subpath "{tmpdir}"))')
        lines.append(f'(allow file-write* (subpath "{tmpdir}"))')

    # ------------------------------------------------------------------
    # 10. Network rules
    # ------------------------------------------------------------------
    if network_proxy_port is not None:
        lines.append("(deny network*)")
        lines.append(
            f'(allow network-outbound (remote ip "localhost:{network_proxy_port}"))'
        )
        lines.append("(allow network-outbound (remote unix-socket))")

    return "\n".join(lines) + "\n"
