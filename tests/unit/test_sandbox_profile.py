# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 077 — Seatbelt profile v2: open reads, confined writes, protected
control files, loopback under the proxy.

Pure profile-string assertions (no ``sandbox-exec`` is spawned), so they run on
any platform. The live behaviour these strings buy is proven in
``tests/platform/test_sandbox_v2_macos.py``.

Ordering matters and is asserted: Seatbelt is last-match-wins (verified with a
live probe), so the credential deny list must be emitted BEFORE the writable
roots. Otherwise a default scratch workspace under
``~/Library/Application Support/Orbital/workspace`` would be unreadable by its
own agent.
"""

import os
import sys

import pytest

from agent_os.platform.macos.sandbox import generate_profile
from agent_os.platform.types import (
    CREDENTIAL_READ_DENY,
    CREDENTIAL_READ_DENY_MACOS,
    PROTECTED_CONTROL_FILES,
    credential_read_deny_paths,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Seatbelt path-string assertions are POSIX-shaped",
)

HOME = os.path.expanduser("~")


def _index(profile: str, needle: str) -> int:
    idx = profile.find(needle)
    assert idx >= 0, f"expected {needle!r} in profile"
    return idx


# ---------------------------------------------------------------------------
# §4.1.1 — base process permissions (M6)
# ---------------------------------------------------------------------------


class TestBaseProcessRules:
    def test_existing_base_rules_kept(self):
        profile = generate_profile("/tmp/workspace")
        for perm in (
            "process-exec",
            "process-fork",
            "signal",
            "sysctl-read",
            "mach-lookup",
        ):
            assert f"(allow {perm}" in profile

    def test_posix_semaphores_allowed(self):
        """M6 — Python's ProcessPoolExecutor needs POSIX semaphores."""
        assert "(allow ipc-posix-sem)" in generate_profile("/tmp/workspace")

    def test_shared_memory_create_allowed(self):
        """M6 — shm create/unlink, not only read/write data as v1 had."""
        assert "(allow ipc-posix-shm*)" in generate_profile("/tmp/workspace")

    def test_pty_allowed(self):
        """M6 — pseudo-tty plus the device nodes Codex's policy names."""
        profile = generate_profile("/tmp/workspace")
        assert "(allow pseudo-tty)" in profile
        assert '(literal "/dev/ptmx")' in profile
        assert "/dev/ttys" in profile


# ---------------------------------------------------------------------------
# §4.1.2 / M1 — reads open to the whole disk
# ---------------------------------------------------------------------------


class TestOpenReads:
    def test_whole_disk_read_allowed(self):
        profile = generate_profile("/tmp/workspace")
        assert "(allow file-read*)" in profile.splitlines()

    def test_no_path_scoped_base_read_allowlist(self):
        """The v1 allowlist of /usr /Library /System … is retired, not kept
        alongside the blanket rule — a reader must not think reads are scoped."""
        profile = generate_profile("/tmp/workspace")
        assert '(allow file-read* (subpath "/usr")' not in profile
        assert '(literal "/etc")' not in profile


# ---------------------------------------------------------------------------
# §4.1.3 / §4.4 — the credential deny list
# ---------------------------------------------------------------------------


class TestCredentialDenyList:
    def test_every_shared_entry_present(self):
        profile = generate_profile("/tmp/workspace")
        for entry in CREDENTIAL_READ_DENY:
            abs_path = os.path.realpath(os.path.join(HOME, *entry.split("/")))
            assert (
                f'(deny file-read* (subpath "{abs_path}")' in profile
            ), f"missing credential deny for {entry}"

    def test_macos_overlay_present(self):
        profile = generate_profile("/tmp/workspace")
        for entry in CREDENTIAL_READ_DENY_MACOS:
            abs_path = os.path.realpath(os.path.join(HOME, *entry.split("/")))
            assert f'(deny file-read* (subpath "{abs_path}")' in profile

    def test_deny_carries_the_sensitive_path_message(self):
        profile = generate_profile("/tmp/workspace")
        assert 'orbital:sensitive-path' in profile

    def test_dot_config_is_not_denied_wholesale(self):
        """M1 — the blanket ~/.config deny broke gh, git XDG config and more."""
        profile = generate_profile("/tmp/workspace")
        assert f'(deny file-read* (subpath "{os.path.join(HOME, ".config")}")' not in profile

    def test_shell_rc_files_are_readable(self):
        """Reads are open; shell rc files are a *write* concern now."""
        profile = generate_profile("/tmp/workspace")
        for name in (".zshrc", ".bashrc", ".profile", ".gitconfig", ".npmrc"):
            assert (
                f'(deny file-read* (subpath "{os.path.join(HOME, name)}")' not in profile
            ), f"{name} must be readable under v2"

    def test_helper_matches_emitted_paths(self):
        """§4.4 — one list, two mechanisms."""
        assert len(credential_read_deny_paths(HOME, "darwin")) == len(
            CREDENTIAL_READ_DENY
        ) + len(CREDENTIAL_READ_DENY_MACOS)
        assert len(credential_read_deny_paths(HOME, "win32")) == len(
            CREDENTIAL_READ_DENY
        )


# ---------------------------------------------------------------------------
# §4.1.4 / M2 + M4 — writable roots
# ---------------------------------------------------------------------------


class TestWritableRoots:
    def test_workspace_read_and_write(self):
        ws = "/Users/testuser/workspace"
        profile = generate_profile(ws)
        assert f'(allow file-read* (subpath "{ws}"))' in profile
        assert f'(allow file-write* (subpath "{ws}"))' in profile

    def test_workspace_allow_comes_after_credential_deny(self):
        """A workspace inside a denied tree (the default scratch project lives
        under ~/Library/Application Support/Orbital/workspace) must still be
        readable. Seatbelt is last-match-wins."""
        ws = os.path.join(
            HOME, "Library", "Application Support", "Orbital", "workspace", "scratch"
        )
        profile = generate_profile(ws)
        deny_at = _index(
            profile,
            f'(deny file-read* (subpath "{os.path.realpath(os.path.join(HOME, "Library/Application Support/Orbital"))}")',
        )
        allow_at = _index(profile, f'(allow file-read* (subpath "{os.path.realpath(ws)}"))')
        assert deny_at < allow_at, "credential deny must precede the workspace allow"

    def test_package_manager_caches_writable(self):
        """M4 — the one deliberate deviation from both reference products."""
        profile = generate_profile("/tmp/workspace")
        for rel in (
            ".cache",
            "Library/Caches",
            ".npm",
            ".cargo/registry",
            ".cargo/git",
            "go/pkg/mod",
            "Library/pnpm",
        ):
            abs_path = os.path.join(HOME, *rel.split("/"))
            assert (
                f'(allow file-write* (subpath "{abs_path}"))' in profile
            ), f"cache root {rel} should be writable"

    def test_temp_roots_still_writable(self):
        profile = generate_profile("/tmp/workspace")
        for path in ("/tmp", "/private/tmp", "/private/var/folders"):
            assert f'(allow file-write* (subpath "{path}"))' in profile

    def test_dev_writable(self):
        assert '(allow file-write* (subpath "/dev"))' in generate_profile("/tmp/ws")

    def test_home_itself_is_not_writable(self):
        """M2 — writes stay confined. Nothing grants file-write* on $HOME."""
        profile = generate_profile("/tmp/workspace")
        assert f'(allow file-write* (subpath "{HOME}"))' not in profile


# ---------------------------------------------------------------------------
# §4.1.5 / M3 — protected control files, per writable root
# ---------------------------------------------------------------------------


class TestProtectedControlFiles:
    def test_emitted_under_the_workspace(self):
        ws = "/Users/testuser/project"
        profile = generate_profile(ws)
        for rel in PROTECTED_CONTROL_FILES:
            target = os.path.join(ws, *rel.split("/"))
            assert (
                f'(deny file-write* (subpath "{target}")' in profile
            ), f"{rel} unprotected in the workspace"

    def test_emitted_under_each_read_write_portal(self):
        ws = "/Users/testuser/project"
        rw = "/Users/testuser/other-repo"
        ro = "/Users/testuser/reference"
        profile = generate_profile(ws, portal_paths={rw: "read_write", ro: "read"})
        assert f'(deny file-write* (subpath "{os.path.join(rw, ".git", "hooks")}")' in profile
        assert f'(deny file-write* (subpath "{os.path.join(rw, ".git", "config")}")' in profile
        # A read-only portal is already deny-write wholesale; no per-file noise.
        assert f'(deny file-write* (subpath "{os.path.join(ro, ".git", "hooks")}")' not in profile

    def test_not_joined_onto_home_when_home_is_not_a_root(self):
        """The v1 bug: the deny list was joined onto $HOME, so it protected
        ``~/.git/hooks`` — a path that does not exist — and left the
        workspace's own hooks writable."""
        profile = generate_profile("/Users/testuser/project")
        assert (
            f'(deny file-write* (subpath "{os.path.join(HOME, ".git", "hooks")}")'
            not in profile
        )

    def test_emitted_under_home_when_home_is_the_workspace(self):
        profile = generate_profile(HOME)
        assert f'(deny file-write* (subpath "{os.path.join(HOME, ".git", "hooks")}")' in profile

    def test_home_shell_rc_still_write_protected(self):
        """§4.1.5 — 'and the home-level shell rc files as today'."""
        profile = generate_profile("/Users/testuser/project")
        for name in (".bashrc", ".zshrc", ".profile", ".gitconfig"):
            assert f'(deny file-write* (subpath "{os.path.join(HOME, name)}")' in profile

    def test_git_object_store_is_not_protected(self):
        """git commit must keep working — Claude Code's rule, narrower than
        Codex's whole-.git exclusion."""
        ws = "/Users/testuser/project"
        profile = generate_profile(ws)
        for rel in ("objects", "index", "refs", "HEAD"):
            assert f'(deny file-write* (subpath "{os.path.join(ws, ".git", rel)}")' not in profile

    def test_read_only_portal_deny_survives_the_temp_write_grants(self):
        """Regression: a read-only portal that happens to live under /tmp or
        $TMPDIR was re-opened for writing by the temp grants, because v1
        emitted the portal deny first and Seatbelt takes the last match."""
        portal = "/tmp/reference-checkout"
        real_portal = os.path.realpath(portal)
        profile = generate_profile("/Users/testuser/project", portal_paths={portal: "read"})
        deny_at = _index(profile, f'(deny file-write* (subpath "{real_portal}")')
        temp_at = _index(profile, '(allow file-write* (subpath "/tmp"))')
        assert temp_at < deny_at, "the read-only portal deny must be emitted last"

    def test_protected_denies_come_after_the_write_allows(self):
        ws = "/Users/testuser/project"
        profile = generate_profile(ws)
        allow_at = _index(profile, f'(allow file-write* (subpath "{ws}"))')
        deny_at = _index(profile, f'(deny file-write* (subpath "{os.path.join(ws, ".git", "hooks")}")')
        assert allow_at < deny_at


# ---------------------------------------------------------------------------
# §4.1.6 / M5 — network
# ---------------------------------------------------------------------------


class TestNetworkRules:
    def test_proxy_rules_unchanged(self):
        profile = generate_profile("/tmp/workspace", network_proxy_port=8080)
        assert "(deny network*)" in profile
        assert '(allow network-outbound (remote ip "localhost:8080"))' in profile
        assert "(allow network-outbound (remote unix-socket))" in profile

    def test_loopback_and_dns_allowed_under_the_proxy(self):
        profile = generate_profile("/tmp/workspace", network_proxy_port=8080)
        assert '(allow network-bind (local ip "localhost:*"))' in profile
        assert '(allow network-inbound (local ip "localhost:*"))' in profile
        assert '(allow network-outbound (remote ip "localhost:*"))' in profile
        assert '(allow network-outbound (remote ip "*:53"))' in profile

    def test_no_network_section_without_a_proxy(self):
        """§7 — loopback and *:53 rules present only when a proxy port is set."""
        profile = generate_profile("/tmp/workspace", network_proxy_port=None)
        assert "(deny network*)" not in profile
        assert "network-bind" not in profile
        assert '"*:53"' not in profile


# ---------------------------------------------------------------------------
# §4.1.7 / M7 — no git env override
# ---------------------------------------------------------------------------


def test_no_git_config_global_anywhere():
    """M7 — v1's GIT_CONFIG_GLOBAL=/dev/null is dropped; the global config is
    readable now."""
    import agent_os.platform.macos.sandbox as sandbox_mod
    import agent_os.platform.macos.provider as provider_mod

    for mod in (sandbox_mod, provider_mod):
        src = open(mod.__file__, encoding="utf-8").read()
        assert "GIT_CONFIG_GLOBAL" not in src
        assert "GIT_CONFIG_NOSYSTEM" not in src


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


class TestProfileStructure:
    def test_balanced_parens(self):
        profile = generate_profile(
            "/tmp/workspace",
            portal_paths={"/tmp/portal": "read_only", "/tmp/rw": "read_write"},
            network_proxy_port=9090,
        )
        assert profile.count("(") == profile.count(")")

    def test_starts_with_version_and_deny_default(self):
        lines = generate_profile("/tmp/workspace").strip().splitlines()
        assert lines[0] == "(version 1)"
        assert lines[1] == "(deny default)"

    def test_paths_with_spaces_stay_quoted(self):
        ws = "/Users/test user/my workspace"
        assert f'(subpath "{ws}")' in generate_profile(ws)

    def test_symlinked_workspace_is_resolved(self, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real)
        profile = generate_profile(str(link))
        assert f'(allow file-write* (subpath "{os.path.realpath(str(real))}"))' in profile
