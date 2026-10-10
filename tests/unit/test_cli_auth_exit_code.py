# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Signed-in checks that work on Windows too.

Codex's check was ``codex login status >/dev/null 2>&1 && echo '{...true}' ||
echo '{...false}'`` — POSIX shell. ``SetupEngine`` runs it with
``shell=True``, which on Windows is cmd.exe: ``/dev/null`` does not exist
("The system cannot find the path specified"), so the ``||`` branch always
ran, and cmd's ``echo`` kept the single quotes, so even that was not JSON. A
signed-in Codex was reported signed out on every Windows machine, and its
usage was never read (the quota read only runs for a *ready* agent). Found on
the installed v0.16.0, 2026-10-11.

A credential with no ``check_field`` is now decided by the check command's
exit code alone, so Codex's check is a plain ``codex login status``.
"""

from __future__ import annotations

import glob
import os
import re
import sys

import pytest

from agent_os.agents.manifest import (
    AgentManifest,
    ManifestCredential,
    ManifestLoader,
    ManifestRuntime,
    ManifestSetup,
)
from agent_os.agents.registry import AgentRegistry
from agent_os.agents.setup_engine import SetupEngine

_MANIFESTS = os.path.join(os.path.dirname(__file__), "..", "..",
                          "agent_os", "agents", "manifests")


def _engine_with_exit_code_check(check_command: str):
    manifest = AgentManifest(
        manifest_version="1", name="Exit", slug="exit-agent", description="",
        author="t", version="1",
        runtime=ManifestRuntime(adapter="cli", command="exit-agent"),
        setup=ManifestSetup(credentials=[ManifestCredential(
            key="exit_auth", label="Exit", type="oauth_cli", required=True,
            check_command=check_command,
        )]),
    )
    registry = AgentRegistry()
    registry.register(manifest)
    return SetupEngine(registry=registry), manifest


@pytest.mark.parametrize("code, configured", [(0, True), (1, False)])
def test_no_check_field_means_the_exit_code_decides(code, configured):
    # A real process, on every platform: the point is the shell, not a mock.
    engine, manifest = _engine_with_exit_code_check(
        f'"{sys.executable}" -c "import sys; print(\'Logged in using ChatGPT\'); sys.exit({code})"')
    ok, missing = engine.check_credentials(manifest)
    assert ok is configured
    assert missing == ([] if configured else ["exit_auth"])


def test_codex_checks_login_status_by_exit_code():
    m = ManifestLoader.load(os.path.join(_MANIFESTS, "codex.yaml"))
    cred = next(c for c in m.setup.credentials if c.key == "codex_auth")
    assert cred.check_command == "codex login status"
    assert cred.check_field == "", "exit code decides; codex prints plain text"


# Constructs that only a POSIX shell understands. Manifest commands run
# through the platform shell, which is cmd.exe on Windows.
_POSIX_ONLY = [
    (re.compile(r"/dev/null"), "/dev/null"),
    (re.compile(r"&&|\|\|"), "&& / || chaining"),
    (re.compile(r"\becho\s+'"), "single-quoted echo (cmd keeps the quotes)"),
    (re.compile(r"\$\(|`"), "command substitution"),
]


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(_MANIFESTS, "*.yaml"))),
                         ids=lambda p: os.path.basename(p))
def test_manifest_commands_are_shell_neutral(path):
    m = ManifestLoader.load(path)
    commands = [m.setup.check_command, m.setup.install_command]
    for cred in m.setup.credentials:
        commands += [cred.check_command, cred.setup_command]
    for command in filter(None, commands):
        for pattern, name in _POSIX_ONLY:
            assert not pattern.search(command), (
                f"{os.path.basename(path)}: {command!r} uses {name}, which "
                f"cmd.exe does not understand")
