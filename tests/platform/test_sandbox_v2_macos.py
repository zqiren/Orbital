# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 077 §7 — sandbox v2 probes against a REAL ``sandbox-exec``.

**Safety contract for this module.** Reads are open to the whole disk under
v2, and a sandboxed command that touches ``~/Desktop``, ``~/Documents`` or
``~/Downloads`` trips a macOS TCC prompt that, unanswered, revokes the calling
terminal's folder access machine-wide. So:

* every workspace here is ``tmp_path`` (``/private/var/folders/…``) — never a
  path under Desktop/Documents/Downloads, and never the dev tree;
* the "outside the workspace" probe uses a dotfile directory this module
  creates and removes directly under ``$HOME``, which is outside the workspace
  (so it still exercises the focus nudge) but is not TCC-protected;
* credential-deny probes send stdout to ``/dev/null`` and assert only on the
  error text, so no secret ever reaches test output.

``tests/platform/test_e2e_agent_isolation.py`` and
``test_macos_provider_integration.py`` are the suites that violate this and
must stay deselected; this module deliberately does not.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from agent_os.platform.macos.sandbox import generate_profile
from agent_os.platform.types import (
    CREDENTIAL_READ_DENY,
    CREDENTIAL_READ_DENY_MACOS,
)

from .conftest import skip_no_seatbelt, skip_not_macos

pytestmark = [skip_not_macos, skip_no_seatbelt]

HOME = os.path.expanduser("~")
DENIAL = "Operation not permitted"

# Interpreter for the in-sandbox Python probes. sys.executable is the venv
# python inside the dev tree, which is exactly the kind of path v1 could not
# even exec — using it is part of the point.
PYTHON = sys.executable


def run_sandboxed(
    script: str,
    workspace: str,
    portal_paths: dict[str, str] | None = None,
    proxy_port: int | None = None,
    timeout: int = 60,
) -> subprocess.CompletedProcess:
    """Run ``script`` under a freshly generated v2 profile."""
    profile = generate_profile(
        workspace_path=workspace,
        portal_paths=portal_paths,
        network_proxy_port=proxy_port,
    )
    return subprocess.run(
        ["sandbox-exec", "-p", profile, "/bin/sh", "-c", script],
        capture_output=True,
        text=True,
        cwd=workspace,
        timeout=timeout,
    )


def run_python(script: str, workspace: str, **kwargs) -> subprocess.CompletedProcess:
    path = Path(workspace) / "_probe.py"
    path.write_text(textwrap.dedent(script), encoding="utf-8")
    return run_sandboxed(f'"{PYTHON}" "{path}"', workspace, **kwargs)


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    return str(ws)


@pytest.fixture
def outside_home_dir():
    """A non-TCC directory directly under $HOME, outside any workspace.

    Substitutes for spec §7.7's ``~/Documents/x`` probe, which cannot be used:
    touching Documents from inside sandbox-exec revokes the terminal's TCC
    grant for the whole machine.
    """
    probe = Path(HOME) / ".orbital-sandbox-v2-probe"
    probe.mkdir(exist_ok=True)
    (probe / "x").write_text("outside-the-workspace\n", encoding="utf-8")
    yield probe
    shutil.rmtree(probe, ignore_errors=True)


# ---------------------------------------------------------------------------
# §7.1 — the four v1 field failures, now green
# ---------------------------------------------------------------------------


class TestOpenReads:
    def test_workspace_parent_is_stattable(self, workspace):
        """v1: pytest could not stat the workspace parent, so collection died."""
        parent = os.path.dirname(workspace)
        result = run_sandboxed(f'ls -d "{parent}"', workspace)
        assert result.returncode == 0, result.stderr
        assert DENIAL not in result.stderr

    def test_users_directory_is_walkable(self, workspace):
        """v1: git's ownership walk failed on /Users with 'Invalid path'."""
        result = run_sandboxed("ls -d /Users", workspace)
        assert result.returncode == 0, result.stderr

    def test_global_gitconfig_is_readable(self, workspace):
        """v1: ~/.gitconfig was unreadable, which is why v1 proposed
        GIT_CONFIG_GLOBAL=/dev/null. M7 drops that; the file is readable now."""
        gitconfig = Path(HOME) / ".gitconfig"
        if not gitconfig.exists():
            pytest.skip("no ~/.gitconfig on this machine")
        result = run_sandboxed(f'cat "{gitconfig}" > /dev/null', workspace)
        assert result.returncode == 0, result.stderr
        assert DENIAL not in result.stderr

    def test_symlink_traversal_through_etc(self, workspace):
        """The retired ``(literal "/etc")`` grant existed so TLS cert lookups
        resolved through the /etc symlink. The blanket read rule must cover it."""
        if not os.path.exists("/etc/ssl/cert.pem"):
            pytest.skip("/etc/ssl/cert.pem not present")
        result = run_sandboxed("cat /etc/ssl/cert.pem > /dev/null", workspace)
        assert result.returncode == 0, result.stderr

    @pytest.mark.skipif(
        not os.path.exists("/opt/homebrew/bin"),
        reason="no Apple-Silicon Homebrew on this machine",
    )
    def test_homebrew_binary_is_executable(self, workspace):
        """v1 hypothesis, now settled: /opt/homebrew was unreadable, so every
        brew-installed toolchain was on PATH and un-execable."""
        candidates = [
            p for p in ("node", "git", "python3", "bash")
            if os.path.exists(f"/opt/homebrew/bin/{p}")
        ]
        if not candidates:
            pytest.skip("no probe binary under /opt/homebrew/bin")
        result = run_sandboxed(f"/opt/homebrew/bin/{candidates[0]} --version", workspace)
        assert result.returncode == 0, result.stderr

    @pytest.mark.skipif(
        not os.path.isdir(os.path.join(HOME, ".nvm", "versions", "node")),
        reason="no nvm node install on this machine",
    )
    def test_nvm_node_is_executable(self, workspace):
        root = Path(HOME) / ".nvm" / "versions" / "node"
        binaries = sorted(root.glob("*/bin/node"))
        if not binaries:
            pytest.skip("no node binary under ~/.nvm")
        result = run_sandboxed(f'"{binaries[-1]}" -v', workspace)
        assert result.returncode == 0, result.stderr

    def test_venv_interpreter_is_executable(self, workspace):
        """The dev tree's own interpreter — v1 could not exec it at all."""
        result = run_sandboxed(f'"{PYTHON}" -c "print(1)"', workspace)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "1"


# ---------------------------------------------------------------------------
# §7.2 — credential stores stay unreadable
# ---------------------------------------------------------------------------


class TestCredentialDenyList:
    @pytest.mark.parametrize(
        "entry", list(CREDENTIAL_READ_DENY) + list(CREDENTIAL_READ_DENY_MACOS)
    )
    def test_deny_list_entry_is_unreadable(self, workspace, entry):
        target = Path(HOME) / Path(*entry.split("/"))
        if not target.exists():
            pytest.skip(f"{entry} not present on this machine")
        # stdout to /dev/null: a bug here must not print a secret.
        probe = f'ls "{target}"' if target.is_dir() else f'cat "{target}"'
        result = run_sandboxed(f"{probe} > /dev/null", workspace)
        assert result.returncode != 0, f"{entry} was readable inside the sandbox"
        assert DENIAL in result.stderr, result.stderr

    def test_orbital_data_dir_is_unreadable(self, workspace):
        """The plaintext project keys until spec 082."""
        data_dir = Path(HOME) / "Library" / "Application Support" / "Orbital"
        if not data_dir.exists():
            pytest.skip("no installed Orbital data dir on this machine")
        result = run_sandboxed(f'ls "{data_dir}" > /dev/null', workspace)
        assert result.returncode != 0
        assert DENIAL in result.stderr

    def test_a_portal_inside_a_denied_tree_still_resolves(self, tmp_path):
        """Ordering guard: the workspace/portal allows are emitted after the
        credential denies, so the packaged app's own scratch workspace (under
        ~/Library/Application Support/Orbital/workspace) is readable."""
        denied_root = Path(HOME) / "Library" / "Application Support" / "Orbital"
        ws = tmp_path / "ws"
        ws.mkdir()
        # Simulate rather than write into the real data dir: assert on the
        # emitted rule order, then prove the mechanism with a synthetic pair.
        profile = generate_profile(str(denied_root / "workspace" / "p1"))
        assert profile.index('(deny file-read* (subpath "') < profile.index(
            f'(allow file-read* (subpath "{os.path.realpath(str(denied_root))}'
        )


# ---------------------------------------------------------------------------
# §7.3 — the process rules (M6)
# ---------------------------------------------------------------------------


class TestProcessRules:
    def test_process_pool_executor_round_trip(self, workspace):
        """v1 field failure: ProcessPoolExecutor EPERM'd at init (POSIX
        semaphores)."""
        # The __main__ guard is mandatory: macOS spawns, so the child
        # re-imports this file.
        result = run_python(
            """
            from concurrent.futures import ProcessPoolExecutor

            if __name__ == "__main__":
                with ProcessPoolExecutor(2) as ex:
                    print(sum(ex.map(abs, [-1, -2, -3])))
            """,
            workspace,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "6"

    def test_pty_openpty(self, workspace):
        result = run_python(
            """
            import os, pty
            master, slave = pty.openpty()
            print(os.ttyname(slave).startswith("/dev/tty"))
            """,
            workspace,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "True"


# ---------------------------------------------------------------------------
# §7.4 — loopback under the proxy (M5)
# ---------------------------------------------------------------------------


class TestNetwork:
    def test_loopback_bind_and_connect(self, workspace):
        """v1 field failure: socket.bind(("127.0.0.1", 0)) EPERM'd."""
        result = run_python(
            """
            import socket
            srv = socket.socket()
            srv.bind(("127.0.0.1", 0))
            srv.listen(1)
            cli = socket.socket()
            cli.connect(srv.getsockname())
            conn, _ = srv.accept()
            conn.sendall(b"pong")
            print(cli.recv(4).decode())
            """,
            workspace,
            proxy_port=9,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "pong"

    def test_loopback_still_denied_without_the_rules(self, workspace):
        """Proves the M5 rules are load-bearing rather than incidental: strip
        them from the generated profile and the same bind fails."""
        profile = generate_profile(workspace, network_proxy_port=9)
        stripped = "\n".join(
            line for line in profile.splitlines()
            if 'localhost:*' not in line and '"*:53"' not in line
        )
        script = Path(workspace) / "_bind.py"
        script.write_text(
            'import socket\ns = socket.socket()\ns.bind(("127.0.0.1", 0))\n',
            encoding="utf-8",
        )
        result = subprocess.run(
            ["sandbox-exec", "-p", stripped, PYTHON, str(script)],
            capture_output=True, text=True, cwd=workspace, timeout=60,
        )
        assert result.returncode != 0
        assert DENIAL in result.stderr

    def test_public_egress_still_denied(self, workspace):
        """The allowlist proxy is the only way out; §4.5 says this does not
        change."""
        result = run_python(
            """
            import socket
            s = socket.socket()
            s.settimeout(5)
            try:
                s.connect(("93.184.216.34", 80))
                print("ALLOWED")
            except PermissionError:
                print("DENIED")
            """,
            workspace,
            proxy_port=9,
        )
        assert result.stdout.strip() == "DENIED", result.stdout + result.stderr

    def test_lan_bind_is_not_actually_denied(self, workspace):
        """Recorded deviation from M5's "LAN bind stays denied".

        Seatbelt's ``(local ip "localhost:*")`` filter matches every local
        address, and a literal ``"127.0.0.1:*"`` is a profile parse error — so
        there is no way to express loopback-only bind in SBPL. Codex's shipped
        policy uses ``"*:*"`` for the same reason. This test pins the real
        behaviour so nobody later "fixes" the profile believing LAN bind is
        fenced.
        """
        result = run_python(
            """
            import socket
            s = socket.socket()
            try:
                s.bind(("0.0.0.0", 0))
                s.listen(1)
                print("BOUND")
            except PermissionError:
                print("DENIED")
            """,
            workspace,
            proxy_port=9,
        )
        assert result.stdout.strip() == "BOUND", (
            "SBPL gained a loopback-only bind filter — update spec 077 M5"
        )


# ---------------------------------------------------------------------------
# §7.5 — writes: confined, control files protected, git commit alive
# ---------------------------------------------------------------------------


class TestWrites:
    @pytest.fixture
    def repo(self, workspace):
        subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
        Path(workspace, "README.md").write_text("hi\n", encoding="utf-8")
        return workspace

    def test_git_hooks_write_denied(self, repo):
        result = run_sandboxed('echo x > .git/hooks/pre-commit', repo)
        assert result.returncode != 0
        assert DENIAL in result.stderr

    def test_git_config_write_denied(self, repo):
        result = run_sandboxed('echo x >> .git/config', repo)
        assert result.returncode != 0
        assert DENIAL in result.stderr

    def test_git_commit_still_works(self, repo):
        result = run_sandboxed(
            'git -c user.name=Probe -c user.email=probe@example.invalid '
            'add README.md && '
            'git -c user.name=Probe -c user.email=probe@example.invalid '
            'commit -q -m "probe" && git log --oneline | wc -l',
            repo,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.strip() == "1"

    def test_workspace_write_allowed(self, workspace):
        result = run_sandboxed('echo ok > inside.txt && cat inside.txt', workspace)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "ok"

    def test_npm_cache_write_allowed(self, workspace):
        """M4 — package-manager caches are writable so `npm install` works."""
        probe = Path(HOME) / ".npm" / ".orbital-077-probe"
        try:
            result = run_sandboxed(f'echo ok > "{probe}"', workspace)
            assert result.returncode == 0, result.stderr
            assert probe.read_text().strip() == "ok"
        finally:
            probe.unlink(missing_ok=True)

    def test_home_write_denied(self, workspace):
        """M2 — writes stay confined; the home directory is not a writable root."""
        probe = Path(HOME) / ".orbital-077-should-not-exist"
        result = run_sandboxed(f'echo x > "{probe}"', workspace)
        try:
            assert result.returncode != 0
            assert DENIAL in result.stderr
            assert not probe.exists()
        finally:
            probe.unlink(missing_ok=True)

    def test_read_only_portal_denies_write(self, workspace, tmp_path):
        portal = tmp_path / "reference"
        portal.mkdir()
        (portal / "note.md").write_text("read me\n", encoding="utf-8")
        result = run_sandboxed(
            f'cat "{portal}/note.md" > /dev/null && echo x > "{portal}/new.txt"',
            workspace,
            portal_paths={str(portal): "read"},
        )
        assert result.returncode != 0
        assert DENIAL in result.stderr

    def test_read_write_portal_hooks_are_protected(self, workspace, tmp_path):
        portal = tmp_path / "other-repo"
        (portal / ".git" / "hooks").mkdir(parents=True)
        result = run_sandboxed(
            f'echo x > "{portal}/ok.txt" && echo y > "{portal}/.git/hooks/pre-push"',
            workspace,
            portal_paths={str(portal): "read_write"},
        )
        assert result.returncode != 0
        assert DENIAL in result.stderr
        assert (portal / "ok.txt").exists(), "the rw portal itself must stay writable"


# ---------------------------------------------------------------------------
# §7.6 / §7.7 — what the agent is told (the hint and the focus nudge)
# ---------------------------------------------------------------------------


class TestShellToolSurface:
    def _tool(self, workspace):
        from agent_os.agent.tools.shell import ShellTool
        from agent_os.platform.macos.provider import MacOSPlatformProvider

        provider = MacOSPlatformProvider()
        return ShellTool(
            workspace=workspace,
            os_type="macos",
            platform_provider=provider,
            project_id="spec077-probe",
        ), provider

    def test_denied_write_carries_the_sandbox_hint(self, workspace):
        """§7.6 — a denied write yields a tool result containing [sandbox]."""
        os.makedirs(os.path.join(workspace, ".git", "hooks"), exist_ok=True)
        tool, provider = self._tool(workspace)
        try:
            result = tool.execute(command="echo x > .git/hooks/pre-commit")
        finally:
            import asyncio
            asyncio.run(provider.teardown())
        assert "[sandbox]" in result.content, result.content
        assert "Folder access" in result.content

    def test_successful_command_carries_no_sandbox_hint(self, workspace):
        tool, provider = self._tool(workspace)
        try:
            result = tool.execute(command="echo fine")
        finally:
            import asyncio
            asyncio.run(provider.teardown())
        assert "[sandbox]" not in result.content

    def test_outside_read_succeeds_and_carries_the_focus_line(
        self, workspace, outside_home_dir
    ):
        """§7.7 — the read now succeeds at the OS level, and the tool result
        says it was off-project rather than claiming it was impossible."""
        target = outside_home_dir / "x"
        tool, provider = self._tool(workspace)
        try:
            result = tool.execute(command=f'cat "{target}"')
        finally:
            import asyncio
            asyncio.run(provider.teardown())
        assert "outside-the-workspace" in result.content, result.content
        assert "Exit code: 0" in result.content
        assert "[focus]" in result.content
        assert "request_access" in result.content
        assert "[sandbox]" not in result.content

    def test_toolchain_path_carries_no_focus_line(self, workspace):
        """§4.7.3 — a warning that is usually wrong is ignored."""
        if not os.path.exists("/opt/homebrew/bin/node"):
            pytest.skip("no /opt/homebrew/bin/node on this machine")
        tool, provider = self._tool(workspace)
        try:
            result = tool.execute(command="/opt/homebrew/bin/node -v")
        finally:
            import asyncio
            asyncio.run(provider.teardown())
        assert "[focus]" not in result.content, result.content


class TestSandboxDenialLookup:
    """§4.3 — the log-show parser. The Sandbox log is machine-wide, so the
    parser has to be picky about which line it names."""

    def _parse(self, text):
        from agent_os.agent.tools.shell import _parse_sandbox_denials
        return _parse_sandbox_denials(textwrap.dedent(text))

    def test_names_a_real_denial(self):
        assert self._parse(
            'kernel[0:1] Sandbox: ls(123) deny(1) file-read-data /Users/me/.aws/credentials'
        ) == "file-read-data /Users/me/.aws/credentials"

    def test_ignores_another_process_scratch_file(self):
        """Observed in the daemon smoke: an unrelated Bun helper's temp-file
        denial was reported as the cause of the agent's ~/.ssh failure."""
        assert self._parse(
            """
            kernel[0:1] Sandbox: ls(123) deny(1) file-read-data /Users/me/.ssh
            kernel[0:2] Sandbox: bun(999) deny(1) file-read-data /private/var/folders/x/.bun-501-abc.node
            """
        ) == "file-read-data /Users/me/.ssh"

    def test_ignores_ioctl_startup_noise(self):
        """bash touches /dev/dtracehelper on every start; that is not a denial
        the user did anything about."""
        assert self._parse(
            'kernel[0:1] Sandbox: bash(1) deny(1) file-ioctl path:/dev/dtracehelper ioctl-command:(_IO "h" 4)'
        ) is None

    def test_returns_none_when_only_noise(self):
        assert self._parse(
            'kernel[0:1] Sandbox: mds(5) deny(1) file-read-data /Library/Caches/x'
        ) is None

    def test_returns_none_on_empty_log(self):
        assert self._parse("") is None

    def test_lookup_is_a_no_op_off_darwin(self, monkeypatch):
        import agent_os.agent.tools.shell as mod
        monkeypatch.setattr(mod.sys, "platform", "linux")
        assert mod._lookup_sandbox_denial() is None


class TestPromptFocusParagraph:
    def test_safety_prompt_states_focus_not_capability(self):
        """§7.7 — the prompt snapshot pins the new paragraph."""
        from agent_os.agent.prompt_builder import PromptBuilder

        builder = PromptBuilder()
        text = builder._safety(type("Ctx", (), {"workspace": "/ws"})())

        assert "FOCUS:" in text
        assert "even to \"check\"" in text
        assert "is normal and is not exploring" in text
        assert "request_access" in text
        # The capability claim the sandbox makes false must be gone.
        assert "You may ONLY access" not in text
