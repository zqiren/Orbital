# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for Seatbelt (SBPL) profile generation.

These are macOS-Seatbelt profile string assertions. The docstring once claimed
"pure string tests, runs on any platform" — empirically several of them embed
POSIX path expectations (forward slashes, quoted-space escapes, /tmp prefix
detection) that don't match Windows path representations. Skipped on win32
until either the assertions are platform-normalised or the suite is moved
behind macOS-only CI.
"""

import os
import sys
from unittest.mock import patch

import pytest

from agent_os.platform.macos.sandbox import generate_profile

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Seatbelt path-string assertions are POSIX-shaped (see module docstring)",
)


class TestProfileBasics:
    """Tests for top-level profile structure."""

    def test_profile_has_version_and_deny_default(self):
        """Output starts with (version 1) and contains (deny default)."""
        profile = generate_profile("/tmp/workspace")
        lines = profile.strip().splitlines()
        assert lines[0] == "(version 1)"
        assert "(deny default)" in profile

    def test_workspace_write_allowed(self):
        """Workspace path appears in (allow file-write* (subpath "..."))."""
        workspace = "/Users/testuser/workspace"
        profile = generate_profile(workspace)
        assert f'(allow file-write* (subpath "{workspace}"))' in profile

    def test_portal_readonly_denies_write(self):
        """Read-only portal has deny file-write rule."""
        portal = "/Users/testuser/Documents"
        profile = generate_profile(
            "/tmp/workspace",
            portal_paths={portal: "read_only"},
        )
        assert f'(deny file-write* (subpath "{portal}")' in profile

    def test_portal_readwrite_allows_write(self):
        """Read-write portal appears in write allow rules."""
        portal = "/Users/testuser/Projects"
        profile = generate_profile(
            "/tmp/workspace",
            portal_paths={portal: "read_write"},
        )
        assert f'(allow file-write* (subpath "{portal}"))' in profile

    def test_sensitive_paths_denied(self):
        """.ssh, .gnupg, .aws, .bashrc, .zshrc all appear in deny rules."""
        profile = generate_profile("/tmp/workspace")
        home = os.path.expanduser("~")
        for name in (".ssh", ".gnupg", ".aws", ".bashrc", ".zshrc"):
            abs_path = os.path.join(home, name)
            assert f'(deny file-read* (subpath "{abs_path}")' in profile, (
                f"Expected deny rule for {abs_path}"
            )

    def test_mandatory_deny_paths_always_present(self):
        """.bashrc, .zshrc, .profile, .git/hooks denied write even when workspace overlaps."""
        home = os.path.expanduser("~")
        # Use home as workspace to test overlap
        profile = generate_profile(home)
        for pattern in (".bashrc", ".zshrc", ".profile", ".git/hooks"):
            abs_path = os.path.join(home, pattern)
            assert f'(deny file-write* (subpath "{abs_path}")' in profile, (
                f"Expected mandatory deny-write for {abs_path}"
            )

    def test_tmpdir_detection(self):
        """/var/folders/ or TMPDIR parent added to write allowlist (mock os.environ for TMPDIR)."""
        profile = generate_profile("/tmp/workspace")
        # /private/var/folders/ is always included (macOS resolves symlinks)
        assert '(allow file-write* (subpath "/private/var/folders"))' in profile

        # Custom TMPDIR is also added
        custom_tmpdir = "/Users/testuser/custom_tmp"
        with patch.dict(os.environ, {"TMPDIR": custom_tmpdir}):
            profile = generate_profile("/tmp/workspace")
        assert f'(allow file-write* (subpath "{custom_tmpdir}"))' in profile
        assert f'(allow file-read* (subpath "{custom_tmpdir}"))' in profile


class TestNetworkRules:
    """Tests for network-related profile rules."""

    def test_network_deny_with_proxy(self):
        """When proxy port given: (deny network*) + allow localhost:PORT."""
        profile = generate_profile("/tmp/workspace", network_proxy_port=8080)
        assert "(deny network*)" in profile
        assert '(allow network-outbound (remote ip "localhost:8080"))' in profile

    def test_network_allow_without_proxy(self):
        """When no proxy port: no (deny network*) in output."""
        profile = generate_profile("/tmp/workspace", network_proxy_port=None)
        assert "(deny network*)" not in profile


class TestPortals:
    """Tests for multi-portal handling."""

    def test_multiple_portals(self):
        """Profile contains rules for all provided portal paths."""
        portals = {
            "/Users/testuser/docs": "read_only",
            "/Users/testuser/code": "read_write",
            "/Users/testuser/data": "read_only",
        }
        profile = generate_profile("/tmp/workspace", portal_paths=portals)
        assert '(deny file-write* (subpath "/Users/testuser/docs")' in profile
        assert '(allow file-write* (subpath "/Users/testuser/code"))' in profile
        assert '(deny file-write* (subpath "/Users/testuser/data")' in profile


class TestProfileValidity:
    """Tests for profile syntax and formatting."""

    def test_profile_is_valid_sbpl_syntax(self):
        """Balanced parens check."""
        profile = generate_profile(
            "/tmp/workspace",
            portal_paths={"/tmp/portal": "read_only"},
            network_proxy_port=9090,
        )
        assert profile.count("(") == profile.count(")"), (
            "Unbalanced parentheses in generated SBPL profile"
        )

    def test_paths_with_spaces_escaped(self):
        """Paths with spaces are properly in quotes."""
        workspace = "/Users/test user/my workspace"
        profile = generate_profile(workspace)
        # The path should appear inside quotes (subpath "...")
        assert f'(subpath "{workspace}")' in profile

    def test_violation_tags_present(self):
        """Deny rules contain (with message "orbital:...")."""
        profile = generate_profile(
            "/tmp/workspace",
            portal_paths={"/tmp/portal": "read_only"},
        )
        assert '(with message "orbital:' in profile


class TestBasePermissions:
    """Tests for base permission rules."""

    def test_base_permissions_present(self):
        """Profile includes all base permissions: process-exec, process-fork, signal, sysctl-read, mach-lookup, ipc-posix-shm-read-data, ipc-posix-shm-write-data."""
        profile = generate_profile("/tmp/workspace")
        required = [
            "process-exec",
            "process-fork",
            "signal",
            "sysctl-read",
            "mach-lookup",
            "ipc-posix-shm-read-data",
            "ipc-posix-shm-write-data",
        ]
        for perm in required:
            assert f"(allow {perm}" in profile, (
                f"Expected base permission (allow {perm}...) in profile"
            )

    def test_base_read_paths_present(self):
        """Profile allows read to /usr, /Library, /System, /private, /dev, /bin, /sbin, /Applications."""
        profile = generate_profile("/tmp/workspace")
        required_paths = ["/usr", "/Library", "/System", "/private", "/dev", "/bin", "/sbin", "/Applications"]
        for path in required_paths:
            assert f'(subpath "{path}")' in profile, (
                f"Expected read access to {path} in profile"
            )
