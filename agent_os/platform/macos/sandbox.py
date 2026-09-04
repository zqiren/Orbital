# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Seatbelt (sandbox-exec) profile generation for macOS process isolation.

Spec 077 — sandbox v2. The profile is a *security* boundary, not an attention
boundary: the agent can use every tool the user installed and everything in the
workspace; it cannot read the user's secrets; it cannot alter files that run
outside the sandbox; it cannot reach hosts the project did not allow. Keeping
the agent's attention on the workspace is the prompt and shell-tool layer's job
(``prompt_builder._safety``, ``ShellTool._detect_external_paths``).

Rule order is load-bearing. Seatbelt is **last-match-wins** (verified with a
live ``sandbox-exec`` probe), so the sections below are emitted in the order
they are numbered and each later section deliberately overrides the earlier
one. In particular the credential deny list precedes the writable roots, so a
workspace that lives inside a denied tree — the default scratch project sits
under ``~/Library/Application Support/Orbital/workspace`` — is still readable
by its own agent.
"""

from __future__ import annotations

import os

from agent_os.platform.types import (
    PROTECTED_CONTROL_FILES,
    credential_read_deny_paths,
)

# Package-manager caches the agent must be able to write (spec 077 M4). The one
# deliberate deviation from both reference sandboxes, taken because Orbital has
# no "retry outside the sandbox" prompt to catch a failing `npm install`.
# Home-relative, POSIX separators.
_WRITABLE_CACHE_ROOTS = (
    ".cache",
    "Library/Caches",
    ".npm",
    ".cargo/registry",
    ".cargo/git",
    "go/pkg/mod",
    "Library/pnpm",
)

# Shell rc files at the top of the user's home. Writes are already confined to
# the workspace, portals, temp and the cache roots, so this is defence in depth
# for the case where one of those roots *is* the home directory.
_HOME_CONTROL_FILES = (".bashrc", ".zshrc", ".profile", ".gitconfig")


def _protected_control_file_rules(root: str) -> list[str]:
    """Deny-write rules for the control files inside one writable root."""
    rules: list[str] = []
    for rel in PROTECTED_CONTROL_FILES:
        target = os.path.join(root, *rel.split("/"))
        rules.append(
            f'(deny file-write* (subpath "{target}")'
            ' (with message "orbital:protected-control-file"))'
        )
    return rules


def generate_profile(
    workspace_path: str,
    portal_paths: dict[str, str] | None = None,  # {path: "read" | "read_only" | "read_write"}
    network_proxy_port: int | None = None,
) -> str:
    """Generate a Seatbelt (SBPL) profile string for sandboxing a child process.

    Args:
        workspace_path: Absolute path to the agent workspace directory.
            The process will have full read/write access here.
        portal_paths: Optional mapping of absolute paths to access levels.
            ``"read"`` / ``"read_only"`` portals (synonyms) allow reads but
            explicitly deny writes.  ``"read_write"`` portals additionally
            allow writes to the given path.
        network_proxy_port: If provided, all outbound network access is denied
            except to ``localhost:<proxy_port>``, local Unix sockets, loopback
            and DNS.  When ``None``, no network restrictions are applied.

    Returns:
        A complete SBPL profile as a multi-line string suitable for passing
        to ``sandbox-exec -p``.
    """
    home = os.path.expanduser("~")
    # Resolve symlinks so Seatbelt path matching works correctly
    # (e.g. /var -> /private/var on macOS).
    workspace_path = os.path.realpath(workspace_path)
    lines: list[str] = []

    # ------------------------------------------------------------------
    # 1. Version & default-deny
    # ------------------------------------------------------------------
    lines.append("(version 1)")
    lines.append("(deny default)")

    # ------------------------------------------------------------------
    # 2. Base process permissions (required for the process to start on
    #    macOS Sequoia without crashing).
    #
    #    Orbital's broad forms of process-exec/signal/sysctl-read/mach-lookup
    #    are kept; tightening them to Codex-style allowlists is later
    #    hardening, not this spec. The IPC and tty operation names below are
    #    copied from Codex's shipped Seatbelt policy — they are what
    #    ProcessPoolExecutor (POSIX semaphores) and pty.openpty() need.
    # ------------------------------------------------------------------
    lines.append("(allow process-exec*)")
    lines.append("(allow process-fork)")
    lines.append("(allow signal)")
    lines.append("(allow sysctl-read)")
    lines.append("(allow mach-lookup)")
    lines.append("(allow ipc-posix-sem)")
    lines.append("(allow ipc-posix-shm*)")
    lines.append("(allow pseudo-tty)")
    lines.append(
        '(allow file-ioctl (literal "/dev/ptmx") (regex #"^/dev/ttys[0-9]*$"))'
    )

    # ------------------------------------------------------------------
    # 3. Reads: open to the whole disk (M1).
    #
    #    The v1 path allowlist (/usr /Library /System …) is retired, not kept
    #    alongside this rule: it blocked every toolchain that lives under the
    #    home directory (nvm, pyenv, cargo, /opt/homebrew), git's ownership
    #    walk over /Users and pytest's stat of the workspace parent.
    # ------------------------------------------------------------------
    lines.append("(allow file-read*)")

    # ------------------------------------------------------------------
    # 4. Credential deny list (M1) — the only reads taken back.
    #    Emitted BEFORE the writable roots so an explicitly granted workspace
    #    or portal inside one of these trees still resolves.
    # ------------------------------------------------------------------
    for deny_path in credential_read_deny_paths(home, "darwin"):
        real_deny = os.path.realpath(deny_path)
        lines.append(
            f'(deny file-read* (subpath "{real_deny}")'
            ' (with message "orbital:sensitive-path"))'
        )

    # ------------------------------------------------------------------
    # 5. Writable roots (M2, M4). Each one also re-grants read, which is what
    #    overrides a credential deny above it.
    # ------------------------------------------------------------------
    lines.append('(allow file-write* (subpath "/dev"))')

    # 5a. Workspace — full read/write.
    lines.append(f'(allow file-read* (subpath "{workspace_path}"))')
    lines.append(f'(allow file-write* (subpath "{workspace_path}"))')

    # 5b. Portals.
    #
    #     Read-only portals contribute a deny-write rule, but it is held back
    #     to section 6: a portal under /tmp or $TMPDIR would otherwise be
    #     re-opened by the temp write grants emitted just below, because the
    #     later rule wins. (v1 had this bug — its temp allows came after the
    #     portal denies.)
    read_write_roots: list[str] = []
    read_only_denies: list[str] = []
    if portal_paths:
        for path, access in portal_paths.items():
            real_path = os.path.realpath(path)
            # Explicit read grant: it is what lets a portal inside a
            # credential-denied tree be read at all, and it documents intent.
            lines.append(f'(allow file-read* (subpath "{real_path}"))')
            if access == "read_write":
                lines.append(f'(allow file-write* (subpath "{real_path}"))')
                read_write_roots.append(real_path)
            else:
                # "read" and "read_only" are synonyms: read-only. Emit an
                # explicit deny (defense-in-depth + a legible violation
                # message) — a "read" cross-project portal must never write.
                read_only_denies.append(
                    f'(deny file-write* (subpath "{real_path}")'
                    ' (with message "orbital:portal-read-only"))'
                )

    # 5c. Temp directories.
    lines.append('(allow file-write* (subpath "/tmp"))')
    lines.append('(allow file-write* (subpath "/private/tmp"))')
    lines.append('(allow file-write* (subpath "/private/var/folders"))')

    tmpdir = os.environ.get("TMPDIR")
    if tmpdir:
        real_tmpdir = os.path.realpath(tmpdir)
        if real_tmpdir not in ("/tmp", "/private/tmp") and not real_tmpdir.startswith(
            "/private/var/folders"
        ):
            lines.append(f'(allow file-read* (subpath "{real_tmpdir}"))')
            lines.append(f'(allow file-write* (subpath "{real_tmpdir}"))')

    # 5d. Package-manager caches (M4).
    for rel in _WRITABLE_CACHE_ROOTS:
        cache_path = os.path.join(home, *rel.split("/"))
        lines.append(f'(allow file-write* (subpath "{cache_path}"))')

    # ------------------------------------------------------------------
    # 6. Every deny-write rule, last, so nothing above can re-open it.
    # ------------------------------------------------------------------

    # 6a. Read-only portals (held back from 5b — see the note there).
    lines.extend(read_only_denies)

    # 6b. Protected control files (M3) — applied to each writable root that
    #    holds user code (the workspace and every read-write portal), using
    #    the root's realpath.
    #
    #    v1 joined this list onto $HOME, so it protected ``~/.git/hooks`` (a
    #    path that does not exist) and left the workspace's own hooks
    #    writable. Read-only portals are already deny-write wholesale, and
    #    temp/cache roots hold no repository worth protecting, so neither gets
    #    per-file noise.
    #
    #    ``.git/objects``, ``.git/index`` and ``.git/refs`` are deliberately
    #    absent: ``git commit`` keeps working.
    # ------------------------------------------------------------------
    for root in [workspace_path, *read_write_roots]:
        lines.extend(_protected_control_file_rules(root))

    # Home-level shell rc files stay write-protected regardless of which roots
    # are writable (the "as today" half of M3).
    for name in _HOME_CONTROL_FILES:
        lines.append(
            f'(deny file-write* (subpath "{os.path.join(home, name)}")'
            ' (with message "orbital:protected-control-file"))'
        )

    # ------------------------------------------------------------------
    # 7. Network rules (M5).
    #
    #    Egress policy is unchanged: everything is denied except the project's
    #    allowlist proxy and unix sockets. Loopback and DNS are added so a dev
    #    server, a test client and a pure-Go resolver work inside the sandbox.
    #
    #    Note: Seatbelt's ``(local ip "localhost:*")`` filter matches every
    #    local address, not only 127.0.0.1 — a literal IP is a parse error —
    #    so a LAN bind is allowed in practice. Recorded in spec 077 §8; the
    #    listener is still only reachable by hosts on the LAN, and outbound
    #    egress remains proxy-only.
    # ------------------------------------------------------------------
    if network_proxy_port is not None:
        lines.append("(deny network*)")
        lines.append(
            f'(allow network-outbound (remote ip "localhost:{network_proxy_port}"))'
        )
        lines.append("(allow network-outbound (remote unix-socket))")
        lines.append('(allow network-bind (local ip "localhost:*"))')
        lines.append('(allow network-inbound (local ip "localhost:*"))')
        lines.append('(allow network-outbound (remote ip "localhost:*"))')
        lines.append('(allow network-outbound (remote ip "*:53"))')

    return "\n".join(lines) + "\n"
