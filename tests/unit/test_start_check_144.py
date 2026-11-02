# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 144 "start check on the chosen copy only".

Detection walks the candidates in a fixed order (pasted path, vendor app,
PATH, manifest locations, check command) and stops at the first one that
exists AND starts. "Starts" is the manifest's own version check, run once
per binary identity (path + mtime) and cached; a candidate that fails it is
passed over. Sign-in and models never enter into it.

Every launch here goes through a fake ``subprocess.run``: nothing real runs.
"""

from __future__ import annotations

import os
import subprocess
from unittest.mock import patch

import pytest

from agent_os.agents.manifest import (
    AgentManifest,
    ManifestCapabilities,
    ManifestCredential,
    ManifestPermissions,
    ManifestRuntime,
    ManifestSetup,
)
from agent_os.agents.registry import AgentRegistry
from agent_os.agents.setup_engine import SetupEngine
from agent_os.daemon_v2.sub_agent_config_store import SubAgentConfigStore

pytestmark = pytest.mark.real_start_check

SLUG = "codex"


def _manifest(auto_detect=None, credentials=None) -> AgentManifest:
    return AgentManifest(
        manifest_version="1", name="Codex", slug=SLUG, description="d",
        author="t", version="1.0.0",
        runtime=ManifestRuntime(adapter="cli", command="codex", args=["app-server"]),
        setup=ManifestSetup(check_command="codex --version",
                            credentials=credentials or [],
                            auto_detect=auto_detect or {}),
        capabilities=ManifestCapabilities(),
        permissions=ManifestPermissions(),
    )


def _file(path, mtime=None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return str(path)


class FakeRun:
    """``subprocess.run`` stand-in: ``broken`` paths exit 1, ``slow`` paths
    time out, everything else prints a version. Records every call."""

    def __init__(self, broken=(), slow=()):
        self.broken = set(broken)
        self.slow = set(slow)
        self.calls: list[dict] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append({"cmd": cmd, **kwargs})
        text = cmd if isinstance(cmd, str) else " ".join(cmd)
        if any(p in text for p in self.slow):
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))
        if any(p in text for p in self.broken):
            return subprocess.CompletedProcess(cmd, 1, "", "dyld: Library not loaded")
        return subprocess.CompletedProcess(cmd, 0, "codex-cli 0.157.1\n", "")

    def launches_of(self, path: str) -> int:
        return sum(1 for c in self.calls
                   if path in (c["cmd"] if isinstance(c["cmd"], str) else " ".join(c["cmd"])))


def _engine(tmp_path, manifest=None) -> tuple[SetupEngine, SubAgentConfigStore]:
    reg = AgentRegistry()
    reg.register(manifest or _manifest())
    store = SubAgentConfigStore(str(tmp_path / "cfg" / "sub_agent_config.json"))
    return SetupEngine(reg, sub_agent_config_store=store,
                       data_dir=str(tmp_path / "data")), store


def _detect_os(name="macos"):
    return patch("agent_os.agents.setup_engine.detect_os", return_value=name)


def test_a_path_copy_that_cannot_start_steps_to_the_next_candidate(tmp_path):
    on_path = _file(tmp_path / "usr" / "codex")
    listed = _file(tmp_path / "npm" / "codex")
    engine, _ = _engine(tmp_path, _manifest({"macos": [listed]}))
    fake = FakeRun(broken=[on_path])
    with _detect_os(), patch("agent_os.agents.setup_engine.shutil.which",
                             return_value=on_path), \
            patch("agent_os.agents.setup_engine.subprocess.run", fake):
        status = engine.check_agent(SLUG)
    assert status.installed is True and status.binary_path == listed
    assert engine.resolved_source(SLUG) == {"kind": "auto_detect"}


def test_newest_glob_match_that_cannot_start_falls_to_the_next_newest(tmp_path):
    """Newest wins, but only if it starts: the just-updated hash folder can
    be half-written while the app installs."""
    newest = _file(tmp_path / "bin" / "0aaa" / "codex.exe", 2_000_000_000)
    older = _file(tmp_path / "bin" / "ffff" / "codex.exe", 1_900_000_000)
    pattern = str(tmp_path / "bin" / "*" / "codex.exe")
    engine, _ = _engine(tmp_path, _manifest({"macos": [pattern]}))
    fake = FakeRun(broken=[newest])
    with _detect_os(), patch("agent_os.agents.setup_engine.shutil.which", return_value=None), \
            patch("agent_os.agents.setup_engine.subprocess.run", fake):
        assert engine.resolve_binary(engine._registry.get(SLUG)) == older


def test_override_that_cannot_start_is_passed_over_and_not_reported_gone(tmp_path):
    chosen = _file(tmp_path / "chosen" / "codex")
    listed = _file(tmp_path / "npm" / "codex")
    engine, store = _engine(tmp_path, _manifest({"macos": [listed]}))
    store.set_override(SLUG, binary=chosen)
    fake = FakeRun(broken=[chosen])
    with _detect_os(), patch("agent_os.agents.setup_engine.shutil.which", return_value=None), \
            patch("agent_os.agents.setup_engine.subprocess.run", fake):
        status = engine.check_agent(SLUG)
    assert status.binary_path == listed
    assert status.override_missing is False and status.override_path is None


def test_nothing_starts_reads_not_found_as_before(tmp_path):
    on_path = _file(tmp_path / "usr" / "codex")
    listed = _file(tmp_path / "npm" / "codex")
    engine, _ = _engine(tmp_path, _manifest({"macos": [listed]}))
    fake = FakeRun(broken=[on_path, listed])
    with _detect_os(), patch("agent_os.agents.setup_engine.shutil.which",
                             return_value=on_path), \
            patch("agent_os.agents.setup_engine.subprocess.run", fake):
        status = engine.check_agent(SLUG)
    assert status.installed is False and status.binary_path is None
    assert status.override_missing is False
    assert engine.get_resolved_path(SLUG) is None


def test_happy_path_launches_the_binary_once_per_identity(tmp_path):
    """The start check is the version check: check_agent reuses its answer,
    and later resolutions of the same file reuse the verdict."""
    on_path = _file(tmp_path / "usr" / "codex")
    engine, _ = _engine(tmp_path)
    fake = FakeRun()
    with _detect_os(), patch("agent_os.agents.setup_engine.shutil.which",
                             return_value=on_path), \
            patch("agent_os.agents.setup_engine.subprocess.run", fake):
        status = engine.check_agent(SLUG)
        assert status.version and "0.157.1" in status.version
        assert fake.launches_of(on_path) == 1
        # A dropped pin (locate, install, Refresh of one agent) resolves again
        # without a second launch: the file did not change.
        engine.invalidate_resolved_path(SLUG)
        engine.check_agent(SLUG)
        assert fake.launches_of(on_path) == 1


def test_a_changed_file_is_checked_again(tmp_path):
    on_path_file = tmp_path / "usr" / "codex"
    on_path = _file(on_path_file, 1_900_000_000)
    engine, _ = _engine(tmp_path)
    fake = FakeRun(broken=[on_path])
    with _detect_os(), patch("agent_os.agents.setup_engine.shutil.which",
                             return_value=on_path), \
            patch("agent_os.agents.setup_engine.subprocess.run", fake):
        assert engine.resolve_binary(engine._registry.get(SLUG)) is None
        assert fake.launches_of(on_path) == 1
        engine.invalidate_resolved_path(SLUG)
        assert engine.resolve_binary(engine._registry.get(SLUG)) is None
        assert fake.launches_of(on_path) == 1  # the failed verdict is cached too
        # The CLI updated itself in place: a new mtime is a new identity.
        fake.broken.clear()
        os.utime(on_path_file, (2_000_000_000, 2_000_000_000))
        engine.invalidate_resolved_path(SLUG)
        assert engine.resolve_binary(engine._registry.get(SLUG)) == on_path
        assert fake.launches_of(on_path) == 2


def test_refresh_drops_the_cached_verdicts(tmp_path):
    on_path = _file(tmp_path / "usr" / "codex")
    engine, _ = _engine(tmp_path)
    fake = FakeRun(broken=[on_path])
    with _detect_os(), patch("agent_os.agents.setup_engine.shutil.which",
                             return_value=on_path), \
            patch("agent_os.agents.setup_engine.subprocess.run", fake):
        assert engine.resolve_binary(engine._registry.get(SLUG)) is None
        fake.broken.clear()  # e.g. the user fixed a missing library
        engine.forget_versions()  # Settings → Refresh
        engine.invalidate_resolved_path(SLUG)
        assert engine.resolve_binary(engine._registry.get(SLUG)) == on_path


def test_the_launch_has_no_stdin_and_a_short_timeout(tmp_path):
    on_path = _file(tmp_path / "usr" / "codex")
    engine, _ = _engine(tmp_path)
    fake = FakeRun()
    with _detect_os(), patch("agent_os.agents.setup_engine.shutil.which",
                             return_value=on_path), \
            patch("agent_os.agents.setup_engine.subprocess.run", fake):
        engine.resolve_binary(engine._registry.get(SLUG))
    (call,) = fake.calls
    assert call["cmd"] == f"{on_path} --version"
    assert call["stdin"] is subprocess.DEVNULL
    assert call["timeout"] == 5


def test_a_slow_start_is_still_a_start(tmp_path):
    """A copy that launched but did not answer in 5 s (a cold Windows start,
    an antivirus scan of a freshly updated codex.exe) has not failed to
    start; skipping it would trade the new build for an old one."""
    on_path = _file(tmp_path / "usr" / "codex")
    listed = _file(tmp_path / "npm" / "codex")
    engine, _ = _engine(tmp_path, _manifest({"macos": [listed]}))
    fake = FakeRun(slow=[on_path])
    with _detect_os(), patch("agent_os.agents.setup_engine.shutil.which",
                             return_value=on_path), \
            patch("agent_os.agents.setup_engine.subprocess.run", fake):
        assert engine.resolve_binary(engine._registry.get(SLUG)) == on_path


def test_sign_in_state_never_moves_the_pointer(tmp_path):
    """Signed out is card state, not a reason to pick another copy: only the
    version check runs during resolution, never the auth probe."""
    on_path = _file(tmp_path / "usr" / "codex")
    listed = _file(tmp_path / "npm" / "codex")
    manifest = _manifest({"macos": [listed]}, credentials=[ManifestCredential(
        key="codex_auth", label="Codex login", type="oauth_cli", required=True,
        check_command="codex login status")])
    engine, _ = _engine(tmp_path, manifest)
    fake = FakeRun(broken=["login status"])  # signed out
    with _detect_os(), patch("agent_os.agents.setup_engine.shutil.which",
                             return_value=on_path), \
            patch("agent_os.agents.setup_engine.subprocess.run", fake):
        assert engine.resolve_binary(engine._registry.get(SLUG)) == on_path
        assert all("login" not in str(c["cmd"]) for c in fake.calls)
        status = engine.check_agent(SLUG)
    assert status.installed is True and status.binary_path == on_path
    assert status.credentials_configured is False
