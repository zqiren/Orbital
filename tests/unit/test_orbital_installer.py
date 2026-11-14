# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for on-demand sub-agent installation (``agent_os/agents/installer.py``).

``SubAgentInstaller`` is Orbital's generic installer for agents that declare
``setup.orbital_install.platforms``: it stages the bundled assets from
``agent_os/agents/assets/<slug>/`` into ``<data_dir>/agents/<slug>/`` and runs
``npm ci --omit=dev`` there, never touching the ``.dmg``/``.exe``. dsh was its
first user until Oct 2026, when dsh moved to its vendor desktop app; no shipped
agent is Orbital-installed now, but the installer stays as infrastructure.

These tests therefore drive it with a fixture agent, ``npm-composed``
(``tests/unit/fixtures/orbital_install_agent/``): a manifest exercising every
field the installer and the status entry read, plus a tiny assets tree
(package.json, package-lock.json, a config template, a ``file:`` local
package). ``_resolve_assets_dir`` is pointed at that tree, and the route tests
register the manifest into the app's registry as if it had shipped. Nothing
here reads a real agent's manifest or assets except the bring-your-own
contrast cases (claude-code, aider). Covered:

- The route answers 202 immediately and never blocks on npm.
- ``npm ci --omit=dev`` — never ``npm install``. Unpinned resolution can
  silently install a different generation (dsh's upstream had inverted
  dist-tags), so the argv assertion is the guard.
- Progress lines and the two terminal events reach ``broadcast_global``.
- ``.install-complete`` is written only on success; its absence means partial,
  and a re-run repairs (assets re-copied, npm ci re-run).
- One in-flight job per slug (409 on a duplicate).
- The platform gate (``setup.orbital_install.platforms``).
- Node preflight: missing binary / too-old version produce an actionable
  failure instead of a confusing npm error.
- ``install`` + ``emits_tool_activity`` + ``credentials`` on the
  ``GET /settings/sub-agents`` entry, refresh-safe across the whole install
  lifecycle. The card renders its install button, its platform-support line,
  and its API-key form off these three, so their shapes are a contract.
- The SetupEngine caches (including the daemon-lifetime resolved path) are
  dropped on success, and the Windows ``.cmd`` shim is what gets detected.

No test here runs a real npm or node: ``asyncio.create_subprocess_exec`` is
monkeypatched and the fake stands in for both the node preflight probe and the
npm run.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import threading
import time

import pytest
from fastapi.testclient import TestClient

from agent_os.agents import installer as installer_mod
from agent_os.agents import setup_engine as setup_engine_mod
from agent_os.agents.installer import (
    InstallInProgress,
    InstallUnsupported,
    SubAgentInstaller,
)
from agent_os.agents.manifest import ManifestLoader
from agent_os.agents.registry import AgentRegistry
from agent_os.agents.setup_engine import SetupEngine

_HERE = os.path.dirname(os.path.abspath(__file__))

MANIFESTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(_HERE)), "agent_os", "agents", "manifests",
)

FIXTURE_DIR = os.path.join(_HERE, "fixtures", "orbital_install_agent")
FIXTURE_MANIFEST = os.path.join(FIXTURE_DIR, "npm-composed.yaml")
FIXTURE_ASSETS = os.path.join(FIXTURE_DIR, "assets")

SLUG = "npm-composed"
COMMAND = "npm-composed-acp"
TEMPLATE = "agent.template.yml"
ALL_PLATFORMS = ["macos", "linux", "windows"]

# The pinned generation in the fixture's package.json (every registry
# dependency pins it; the ``file:`` shim carries none). The marker records it
# so a later daemon can tell which composition is on disk.
PINNED_VERSION = "9.9.9-fixture.1"

# Env vars the fixture's credentials resolve through, plus the ones a
# developer may have exported that the route tests must not see.
_CREDENTIAL_ENV = ("NPM_COMPOSED_API_KEY", "NPM_COMPOSED_BASE_URL",
                   "OPENAI_API_KEY", "ANTHROPIC_API_KEY")


def load_fixture_manifest():
    return ManifestLoader.load(FIXTURE_MANIFEST)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeWS:
    """Collects broadcast_global payloads."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def broadcast_global(self, payload: dict) -> None:
        self.events.append(payload)

    def of_type(self, type_: str) -> list[dict]:
        return [e for e in self.events if e.get("type") == type_]


class FakeStream:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)

    async def readline(self) -> bytes:
        if self._lines:
            return self._lines.pop(0)
        return b""


class FakeProc:
    """Stands in for both ``node --version`` and ``npm ci``."""

    def __init__(self, lines=(), returncode=0, gate=None, on_wait=None) -> None:
        self.stdout = FakeStream(list(lines))
        self.returncode = returncode
        self.pid = 4242
        self._gate = gate
        self._on_wait = on_wait

    async def communicate(self, input=None):
        out = b"".join(self._drain())
        return out, b""

    def _drain(self) -> list[bytes]:
        drained = []
        while self.stdout._lines:
            drained.append(self.stdout._lines.pop(0))
        return drained

    async def wait(self) -> int:
        if self._gate is not None:
            # Cross-thread gate: the test releases it from the main thread
            # while the job runs on the app's event loop.
            await asyncio.get_running_loop().run_in_executor(None, self._gate.wait)
        if self._on_wait is not None:
            self._on_wait()
        return self.returncode


class FakeExec:
    """Monkeypatch target for ``asyncio.create_subprocess_exec``."""

    def __init__(self, *, node_version=b"v24.3.0\n", npm_lines=None,
                 npm_returncode=0, gate=None, on_npm_wait=None) -> None:
        self.node_version = node_version
        self.npm_lines = npm_lines if npm_lines is not None else [
            b"added 3 packages in 1s\n", b"\n",
        ]
        self.npm_returncode = npm_returncode
        self.gate = gate
        self.on_npm_wait = on_npm_wait
        self.calls: list[tuple[tuple, dict]] = []

    async def __call__(self, *argv, **kwargs):
        self.calls.append((argv, kwargs))
        if "--version" in argv:
            return FakeProc(lines=[self.node_version])
        return FakeProc(
            lines=self.npm_lines,
            returncode=self.npm_returncode,
            gate=self.gate,
            on_wait=self.on_npm_wait,
        )

    @property
    def npm_calls(self) -> list[tuple]:
        return [argv for argv, _ in self.calls if "--version" not in argv]


def patch_which(monkeypatch, *, node="/usr/local/bin/node", npm="/usr/local/bin/npm"):
    """Make node/npm resolvable regardless of the host, leaving every other
    lookup on the real implementation (the app resolves other binaries too)."""
    real_which = shutil.which

    def fake_which(name, *args, **kwargs):
        if name == "node":
            return node
        if name == "npm":
            return npm
        return real_which(name, *args, **kwargs)

    monkeypatch.setattr(shutil, "which", fake_which)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _fixture_assets(monkeypatch):
    """Resolve the fixture agent's bundled assets to the test tree. Every
    other slug keeps the real lookup. Tests that need a bespoke tree patch
    ``_resolve_assets_dir`` again; the later patch wins."""
    real_resolve = installer_mod._resolve_assets_dir

    def resolve(slug):
        if slug == SLUG:
            return FIXTURE_ASSETS
        return real_resolve(slug)

    monkeypatch.setattr(installer_mod, "_resolve_assets_dir", resolve)


@pytest.fixture
def agent_manifest():
    return load_fixture_manifest()


@pytest.fixture
def engine(tmp_path, agent_manifest):
    registry = AgentRegistry()
    registry.register(agent_manifest)
    return SetupEngine(registry, data_dir=str(tmp_path / "data"))


@pytest.fixture
def ws():
    return FakeWS()


@pytest.fixture
def installer(engine, ws, monkeypatch):
    # macOS is in the fixture's platform list; pinning it keeps every test
    # host-neutral.
    monkeypatch.setattr(installer_mod, "detect_os", lambda: "macos")
    return SubAgentInstaller(setup_engine=engine, ws_manager=ws)


def target_dir(engine, slug=SLUG) -> str:
    return os.path.join(engine.data_dir, "agents", slug)


async def run_install(installer, manifest, exec_fake, monkeypatch):
    """Start a job with the subprocess layer faked and await it."""
    monkeypatch.setattr(asyncio, "create_subprocess_exec", exec_fake)
    job_id = installer.start(manifest.slug, manifest)
    await installer.wait(manifest.slug)
    return job_id


# ---------------------------------------------------------------------------
# 0. The fixture itself
# ---------------------------------------------------------------------------

class TestFixtureAgent:
    """The fixture stands in for a shipped Orbital-installed agent; if it
    stops parsing as one, every test below silently tests less."""

    def test_manifest_declares_an_orbital_install(self, agent_manifest):
        assert agent_manifest.slug == SLUG
        assert agent_manifest.runtime.command == COMMAND
        assert agent_manifest.runtime.transport == "acp-sdk"
        assert agent_manifest.runtime.mode == "pipe"
        assert agent_manifest.runtime.config_template == TEMPLATE
        assert agent_manifest.setup.orbital_install.platforms == ALL_PLATFORMS
        assert agent_manifest.capabilities.emits_tool_activity is False

    def test_manifest_names_the_cmd_shim_for_windows(self, agent_manifest):
        prefix = f"${{ORBITAL_DATA_DIR}}/agents/{SLUG}/node_modules/.bin/"
        assert agent_manifest.setup.auto_detect["windows"] == [
            f"{prefix}{COMMAND}.cmd"]
        for other in ("macos", "linux"):
            assert agent_manifest.setup.auto_detect[other] == [
                f"{prefix}{COMMAND}"]

    def test_bundled_asset_path_follows_the_resolver(self):
        """The dispatch renderer falls back to ``bundled_asset_path`` when the
        install dir has no staged template; it resolves through the same
        ``_resolve_assets_dir`` the installer stages from."""
        assert installer_mod.bundled_asset_path(SLUG, TEMPLATE) == \
            os.path.join(FIXTURE_ASSETS, TEMPLATE)
        assert installer_mod.bundled_asset_path(SLUG, "absent.yml") is None
        assert installer_mod.bundled_asset_path(
            "no-such-agent-slug", TEMPLATE) is None


# ---------------------------------------------------------------------------
# 1. The install job
# ---------------------------------------------------------------------------

class TestInstallJob:

    async def test_copies_assets_and_writes_marker(self, installer, engine,
                                                   agent_manifest, monkeypatch):
        exec_fake = FakeExec()
        patch_which(monkeypatch)
        await run_install(installer, agent_manifest, exec_fake, monkeypatch)

        target = target_dir(engine)
        for name in ("package.json", "package-lock.json", TEMPLATE):
            assert os.path.isfile(os.path.join(target, name)), name

        marker = os.path.join(target, ".install-complete")
        assert os.path.isfile(marker)
        with open(marker, encoding="utf-8") as f:
            assert json.load(f)["version"] == PINNED_VERSION
        assert installer.get_state(SLUG)["state"] == "installed"

    async def test_runs_npm_ci_never_npm_install(self, installer, engine,
                                                 agent_manifest, monkeypatch):
        """`npm install` re-resolves ``^`` ranges against the registry's
        dist-tags and can install an entirely different generation (dsh's
        upstream ``latest`` pointed at an ancient rc.1). The lockfile is only
        a freeze when ``npm ci`` consumes it."""
        exec_fake = FakeExec()
        patch_which(monkeypatch)
        await run_install(installer, agent_manifest, exec_fake, monkeypatch)

        assert len(exec_fake.npm_calls) == 1
        argv = exec_fake.npm_calls[0]
        assert argv[0] == "/usr/local/bin/npm"
        assert list(argv[1:]) == ["ci", "--omit=dev"]
        assert "install" not in argv

        # …and in the agent's own directory, not the daemon's cwd.
        npm_kwargs = [kw for a, kw in exec_fake.calls if "--version" not in a][0]
        assert npm_kwargs["cwd"] == target_dir(engine)

    async def test_broadcasts_progress_then_done(self, installer, ws,
                                                 agent_manifest, monkeypatch):
        exec_fake = FakeExec(npm_lines=[b"npm warn deprecated x\n",
                                        b"\n",
                                        b"added 3 packages\n"])
        patch_which(monkeypatch)
        job_id = await run_install(installer, agent_manifest, exec_fake,
                                   monkeypatch)

        progress = ws.of_type("sub_agent_install_progress")
        assert [e["line"] for e in progress] == [
            "npm warn deprecated x", "added 3 packages"]  # blank lines dropped
        for event in progress:
            assert event["slug"] == SLUG
            assert event["job_id"] == job_id

        done = ws.of_type("sub_agent_install_done")
        assert len(done) == 1
        assert done[0] == {"type": "sub_agent_install_done", "slug": SLUG,
                           "job_id": job_id, "version": PINNED_VERSION}
        assert ws.of_type("sub_agent_install_failed") == []

    async def test_failure_broadcasts_terminal_event_and_no_marker(
            self, installer, engine, ws, agent_manifest, monkeypatch):
        exec_fake = FakeExec(npm_lines=[b"npm error code E404\n"],
                             npm_returncode=1)
        patch_which(monkeypatch)
        job_id = await run_install(installer, agent_manifest, exec_fake,
                                   monkeypatch)

        failed = ws.of_type("sub_agent_install_failed")
        assert len(failed) == 1
        assert failed[0]["slug"] == SLUG
        assert failed[0]["job_id"] == job_id
        assert "1" in failed[0]["error"]  # names the exit code
        assert ws.of_type("sub_agent_install_done") == []

        assert not os.path.exists(
            os.path.join(target_dir(engine), ".install-complete"))
        assert installer.get_state(SLUG)["state"] == "failed"

    async def test_partial_install_is_repaired_by_rerun(
            self, installer, engine, agent_manifest, monkeypatch):
        """No marker means partial. A re-run re-copies the assets (so a stale
        template from a half-finished install is refreshed) and re-runs npm."""
        target = target_dir(engine)
        os.makedirs(target, exist_ok=True)
        stale = os.path.join(target, TEMPLATE)
        with open(stale, "w", encoding="utf-8") as f:
            f.write("- id: stale\n")
        # A half-installed tree with no marker.
        os.makedirs(os.path.join(target, "node_modules"), exist_ok=True)

        exec_fake = FakeExec()
        patch_which(monkeypatch)
        await run_install(installer, agent_manifest, exec_fake, monkeypatch)

        with open(stale, encoding="utf-8") as f:
            refreshed = f.read()
        assert "id: stale" not in refreshed
        assert "npm-composed-agent" in refreshed
        assert os.path.isfile(os.path.join(target, ".install-complete"))
        assert len(exec_fake.npm_calls) == 1

    async def test_stale_marker_is_dropped_before_npm_runs(
            self, installer, engine, agent_manifest, monkeypatch):
        """A marker left by an earlier generation must not claim the tree is
        complete while npm is rebuilding it — and a failed rebuild leaves no
        marker at all."""
        target = target_dir(engine)
        os.makedirs(target, exist_ok=True)
        marker = os.path.join(target, ".install-complete")
        with open(marker, "w", encoding="utf-8") as f:
            json.dump({"version": "0.0.1-old"}, f)

        seen_during_npm = []
        exec_fake = FakeExec(
            npm_returncode=1,
            on_npm_wait=lambda: seen_during_npm.append(os.path.exists(marker)))
        patch_which(monkeypatch)
        await run_install(installer, agent_manifest, exec_fake, monkeypatch)

        assert seen_during_npm == [False]
        assert not os.path.exists(marker)

    async def test_stages_local_packages_beside_the_manifest(
            self, installer, engine, agent_manifest, monkeypatch, tmp_path):
        """A composition can carry local packages next to its manifest — a
        plugin the template loads by path, or a ``file:`` dependency npm
        resolves. Staging only the three named files leaves npm pointing at a
        directory that never arrived."""
        source = tmp_path / "assets"
        (source / "shims" / "activity").mkdir(parents=True)
        (source / "package.json").write_text(
            '{"dependencies": {"orbital-acp-activity": "file:./shims/activity"}}')
        (source / "package-lock.json").write_text("{}")
        (source / TEMPLATE).write_text("- id: npm-composed-agent\n")
        (source / "shims" / "activity" / "index.js").write_text("// shim\n")
        # A dev's stray npm run in the source tree must not be copied over the
        # install npm is about to build.
        (source / "node_modules" / "junk").mkdir(parents=True)
        monkeypatch.setattr(installer_mod, "_resolve_assets_dir",
                            lambda slug: str(source))

        patch_which(monkeypatch)
        await run_install(installer, agent_manifest, FakeExec(), monkeypatch)

        target = target_dir(engine)
        assert os.path.isfile(os.path.join(target, "shims", "activity",
                                           "index.js"))
        assert not os.path.exists(os.path.join(target, "node_modules", "junk"))
        assert os.path.isfile(os.path.join(target, ".install-complete"))

    async def test_bundled_fixture_tree_is_staged_whole(
            self, installer, engine, agent_manifest, monkeypatch):
        """The default fixture tree carries a ``file:`` shim too; it arrives
        intact and its unversioned spec does not leak into the marker."""
        patch_which(monkeypatch)
        await run_install(installer, agent_manifest, FakeExec(), monkeypatch)

        target = target_dir(engine)
        for rel in (("shims", "activity", "index.js"),
                    ("shims", "activity", "package.json")):
            assert os.path.isfile(os.path.join(target, *rel)), rel
        with open(os.path.join(target, ".install-complete"),
                  encoding="utf-8") as f:
            assert json.load(f) == {"version": PINNED_VERSION}

    async def test_rerun_replaces_a_stale_local_package(
            self, installer, engine, agent_manifest, monkeypatch, tmp_path):
        """The staged tree is rebuilt on every run, not merged into. A file
        the shim no longer ships must not survive a re-install, or a repaired
        install keeps loading code the composition dropped."""
        source = tmp_path / "assets"
        (source / "shims" / "activity").mkdir(parents=True)
        (source / "package.json").write_text('{"dependencies": {}}')
        (source / "package-lock.json").write_text("{}")
        (source / TEMPLATE).write_text("- id: npm-composed-agent\n")
        (source / "shims" / "activity" / "index.js").write_text("// current\n")
        monkeypatch.setattr(installer_mod, "_resolve_assets_dir",
                            lambda slug: str(source))
        patch_which(monkeypatch)

        target = target_dir(engine)
        stale = os.path.join(target, "shims", "activity", "dropped.js")
        os.makedirs(os.path.dirname(stale), exist_ok=True)
        with open(stale, "w", encoding="utf-8") as f:
            f.write("// from an older composition\n")

        await run_install(installer, agent_manifest, FakeExec(), monkeypatch)

        assert not os.path.exists(stale)
        with open(os.path.join(target, "shims", "activity", "index.js"),
                  encoding="utf-8") as f:
            assert f.read() == "// current\n"

    async def test_staging_is_fine_with_no_local_packages(
            self, installer, engine, ws, agent_manifest, monkeypatch, tmp_path):
        """A composition of three flat files — no subdirectories — installs
        exactly as before. The tree copy must not require one."""
        source = tmp_path / "assets"
        source.mkdir()
        (source / "package.json").write_text('{"dependencies": {}}')
        (source / "package-lock.json").write_text("{}")
        (source / TEMPLATE).write_text("- id: npm-composed-agent\n")
        monkeypatch.setattr(installer_mod, "_resolve_assets_dir",
                            lambda slug: str(source))

        patch_which(monkeypatch)
        await run_install(installer, agent_manifest, FakeExec(), monkeypatch)

        assert ws.of_type("sub_agent_install_failed") == []
        assert os.path.isfile(
            os.path.join(target_dir(engine), ".install-complete"))

    async def test_missing_bundled_asset_names_the_file(
            self, installer, agent_manifest, ws, monkeypatch, tmp_path):
        source = tmp_path / "assets"
        source.mkdir()
        (source / "package.json").write_text("{}")   # no lockfile
        monkeypatch.setattr(installer_mod, "_resolve_assets_dir",
                            lambda slug: str(source))

        patch_which(monkeypatch)
        await run_install(installer, agent_manifest, FakeExec(), monkeypatch)

        failed = ws.of_type("sub_agent_install_failed")
        assert len(failed) == 1
        assert "package-lock.json" in failed[0]["error"]

    async def test_missing_declared_template_names_the_file(
            self, installer, agent_manifest, ws, monkeypatch, tmp_path):
        """``runtime.config_template`` joins the required assets when the
        manifest declares one."""
        source = tmp_path / "assets"
        source.mkdir()
        (source / "package.json").write_text("{}")
        (source / "package-lock.json").write_text("{}")
        monkeypatch.setattr(installer_mod, "_resolve_assets_dir",
                            lambda slug: str(source))

        exec_fake = FakeExec()
        patch_which(monkeypatch)
        await run_install(installer, agent_manifest, exec_fake, monkeypatch)

        failed = ws.of_type("sub_agent_install_failed")
        assert len(failed) == 1
        assert TEMPLATE in failed[0]["error"]
        assert exec_fake.npm_calls == []

    async def test_missing_assets_dir_fails_actionably(
            self, installer, agent_manifest, ws, monkeypatch):
        monkeypatch.setattr(installer_mod, "_resolve_assets_dir",
                            lambda slug: None)

        exec_fake = FakeExec()
        patch_which(monkeypatch)
        await run_install(installer, agent_manifest, exec_fake, monkeypatch)

        failed = ws.of_type("sub_agent_install_failed")
        assert len(failed) == 1
        assert SLUG in failed[0]["error"]
        assert "reinstall Orbital" in failed[0]["error"]
        assert exec_fake.npm_calls == []

    async def test_success_reprobes_the_installed_agent_into_the_cache(
            self, installer, engine, agent_manifest, monkeypatch):
        """The resolved-path cache lives for the daemon's lifetime; without an
        explicit drop, a just-installed agent keeps reporting itself absent.
        Spec 136 A.5: the path is dropped and this one agent is re-probed into
        the cached sweep (same expiry) instead of every agent on next read."""
        from agent_os.agents.setup_types import AgentSetupStatus

        def absent(installed=False):
            return AgentSetupStatus(
                slug=SLUG, name=SLUG, installed=installed, binary_path=None,
                version=None, dependencies_met=True, missing_dependencies=[],
                credentials_configured=True, missing_credentials=[])

        expires = time.monotonic() + 999
        engine._check_all_cache = ([absent()], expires)
        stale = f"/stale/path/{COMMAND}"
        engine._resolved_paths[SLUG] = stale
        probes = []

        def check_agent(slug):
            probes.append((slug, engine._resolved_paths.get(slug)))
            return absent(installed=True)

        engine.check_agent = check_agent
        exec_fake = FakeExec()
        patch_which(monkeypatch)
        await run_install(installer, agent_manifest, exec_fake, monkeypatch)

        assert probes == [(SLUG, None)]  # the stale path was gone before the probe
        results, expires_after = engine._check_all_cache
        assert expires_after == expires
        assert [r.installed for r in results] == [True]


# ---------------------------------------------------------------------------
# 2. Gates: platform, concurrency, preflight
# ---------------------------------------------------------------------------

class TestGates:

    def test_platform_gate_rejects_unsupported_host(self, engine, ws,
                                                    agent_manifest,
                                                    monkeypatch):
        # The fixture lists every real OS value; a made-up host keeps the
        # gate logic itself under test.
        monkeypatch.setattr(installer_mod, "detect_os", lambda: "freebsd")
        inst = SubAgentInstaller(setup_engine=engine, ws_manager=ws)
        assert inst.supports(agent_manifest) is False
        with pytest.raises(InstallUnsupported) as exc:
            inst.start(SLUG, agent_manifest)
        assert "freebsd" in str(exc.value)
        assert inst.get_state(SLUG) is None

    def test_windows_is_a_supported_host(self, engine, ws, agent_manifest,
                                         monkeypatch):
        """An agent that lists windows is installable there: ``supports``
        says yes and the echoed platform list carries all three OS values."""
        monkeypatch.setattr(installer_mod, "detect_os", lambda: "windows")
        inst = SubAgentInstaller(setup_engine=engine, ws_manager=ws)
        assert inst.supports(agent_manifest) is True
        entry = inst.entry_for(agent_manifest, installed=False)
        assert entry["supported"] is True
        assert entry["platforms"] == ALL_PLATFORMS
        assert entry["state"] == "not_installed"

    def test_agent_without_orbital_install_is_rejected(self, installer):
        claude = ManifestLoader.load(
            os.path.join(MANIFESTS_DIR, "claude_code.yaml"))
        assert installer.supports(claude) is False
        with pytest.raises(InstallUnsupported):
            installer.start(claude.slug, claude)

    async def test_duplicate_job_rejected(self, installer, agent_manifest,
                                          monkeypatch):
        gate = threading.Event()
        exec_fake = FakeExec(gate=gate)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", exec_fake)
        patch_which(monkeypatch)

        job_id = installer.start(SLUG, agent_manifest)
        # Let the job reach the npm step before the second attempt.
        await asyncio.sleep(0)
        assert installer.get_state(SLUG)["state"] == "installing"
        with pytest.raises(InstallInProgress):
            installer.start(SLUG, agent_manifest)

        gate.set()
        await installer.wait(SLUG)
        assert installer.get_state(SLUG) == {"job_id": job_id,
                                             "state": "installed"}

    async def test_preflight_missing_node_fails_actionably(
            self, installer, ws, agent_manifest, monkeypatch):
        real_which = shutil.which
        monkeypatch.setattr(
            shutil, "which",
            lambda name, *a, **kw: None if name in ("node", "npm")
            else real_which(name, *a, **kw))
        exec_fake = FakeExec()
        await run_install(installer, agent_manifest, exec_fake, monkeypatch)

        failed = ws.of_type("sub_agent_install_failed")
        assert len(failed) == 1
        error = failed[0]["error"]
        assert "node" in error.lower()
        assert "20.12" in error
        assert agent_manifest.name in error
        # The packaged app merges the login-shell PATH; a bare GUI-context
        # uvicorn launch does not — the message has to say so.
        assert "PATH" in error
        assert exec_fake.calls == []  # never reached npm

    async def test_preflight_missing_npm_names_the_node_found(
            self, installer, ws, agent_manifest, monkeypatch):
        patch_which(monkeypatch, npm=None)
        exec_fake = FakeExec()
        await run_install(installer, agent_manifest, exec_fake, monkeypatch)

        failed = ws.of_type("sub_agent_install_failed")
        assert len(failed) == 1
        error = failed[0]["error"]
        assert "npm" in error
        assert "/usr/local/bin/node" in error
        assert "PATH" in error
        assert exec_fake.calls == []

    async def test_preflight_old_node_names_the_found_version(
            self, installer, ws, agent_manifest, monkeypatch):
        exec_fake = FakeExec(node_version=b"v18.20.4\n")
        patch_which(monkeypatch)
        await run_install(installer, agent_manifest, exec_fake, monkeypatch)

        failed = ws.of_type("sub_agent_install_failed")
        assert len(failed) == 1
        error = failed[0]["error"]
        assert "18.20.4" in error
        assert "20.12" in error
        assert "/usr/local/bin/node" in error
        assert exec_fake.npm_calls == []

    async def test_preflight_unparseable_node_version(
            self, installer, ws, agent_manifest, monkeypatch):
        exec_fake = FakeExec(node_version=b"command not understood\n")
        patch_which(monkeypatch)
        await run_install(installer, agent_manifest, exec_fake, monkeypatch)

        failed = ws.of_type("sub_agent_install_failed")
        assert len(failed) == 1
        assert "command not understood" in failed[0]["error"]
        assert exec_fake.npm_calls == []

    async def test_preflight_accepts_node_at_the_floor(
            self, installer, ws, agent_manifest, monkeypatch):
        exec_fake = FakeExec(node_version=b"v20.12.0\n")
        patch_which(monkeypatch)
        await run_install(installer, agent_manifest, exec_fake, monkeypatch)

        assert ws.of_type("sub_agent_install_failed") == []
        assert len(exec_fake.npm_calls) == 1


# ---------------------------------------------------------------------------
# 3. The status entry contract (what the settings card codes against)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_real_codex_probe(monkeypatch):
    """The settings routes probe the codex binary for its live model list.
    Never spawn the real one here — the route seam stays exercised, only the
    subprocess layer is stubbed."""
    import agent_os.agent.transports.codex_models as cm

    async def _unavailable(binary="codex", **_kw):
        return None

    monkeypatch.setattr(cm, "fetch_codex_models", _unavailable)
    cm.clear_codex_models_cache()
    yield
    cm.clear_codex_models_cache()


class DictStore:
    """Stands in for the sub-agent keychain store (``.get(key)``)."""

    def __init__(self, **values) -> None:
        self.values = dict(values)

    def get(self, key: str):
        return self.values.get(key)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    os.makedirs(str(tmp_path / "home"), exist_ok=True)
    monkeypatch.setattr(installer_mod, "detect_os", lambda: "macos")
    # Credential state resolves through the environment as a last step, so a
    # developer's exported keys would otherwise decide these assertions.
    for key in _CREDENTIAL_ENV:
        monkeypatch.delenv(key, raising=False)

    # The app loads its registry from the shipped manifests directory. Ship
    # the fixture alongside them for this app only: every registry that loads
    # a directory also registers it, so it is present from the first status
    # read exactly as a shipped Orbital-installed agent would be.
    real_load_directory = AgentRegistry.load_directory

    def load_directory_with_fixture(self, path):
        real_load_directory(self, path)
        self.register(load_fixture_manifest())

    monkeypatch.setattr(AgentRegistry, "load_directory",
                        load_directory_with_fixture)

    from agent_os.api.app import create_app
    app = create_app(data_dir=str(tmp_path / "data"))
    with TestClient(app) as c:
        yield c


def fixture_entry(client) -> dict:
    resp = client.get("/api/v2/settings/sub-agents")
    assert resp.status_code == 200
    return next(e for e in resp.json() if e["slug"] == SLUG)


def wait_for_job(slug: str = SLUG, timeout: float = 30) -> dict | None:
    """Poll the route module's in-memory job registry until the job leaves
    ``installing``.

    Not the status route: a cold status read costs seconds (check_all shells
    out to every agent's CLI), and one in flight while the finishing job
    calls invalidate_cache() lands its now-stale result afterwards — pinning
    "not installed" for the full 60 s TTL. That lost update is a real
    SetupEngine race, not these tests' subject; polling the registry keeps
    the window closed so they stay tests of the entry contract.
    """
    import agent_os.api.routes.settings as settings_routes
    deadline = time.time() + timeout
    state = None
    while time.time() < deadline:
        state = settings_routes._installer.get_state(slug)
        if state is not None and state["state"] != "installing":
            break
        time.sleep(0.02)
    return state


class TestStatusEntry:

    def test_every_entry_carries_install_and_activity_fields(self, client):
        resp = client.get("/api/v2/settings/sub-agents")
        assert resp.status_code == 200
        entries = resp.json()
        assert entries
        assert any(e["slug"] == SLUG for e in entries)
        for entry in entries:
            assert isinstance(entry["emits_tool_activity"], bool)
            install = entry["install"]
            assert isinstance(install["supported"], bool)
            assert isinstance(install["platforms"], list)
            assert install["state"] in (
                "installed", "installing", "failed", "not_installed")
            for cred in entry["credentials"]:
                assert set(cred) == {"key", "label", "type", "required",
                                     "has_setup_command", "configured",
                                     "stored_by"}
                assert isinstance(cred["configured"], bool)
                # Names the agent's own app when it holds the key; None when
                # Orbital's copy or the environment is in charge.
                assert cred["stored_by"] is None or isinstance(
                    cred["stored_by"], str)

    def test_orbital_installable_agent_is_offered_and_silent(self, client):
        entry = fixture_entry(client)
        assert entry["install"]["supported"] is True
        assert entry["install"]["state"] == "not_installed"
        assert entry["installed"] is False
        assert entry["emits_tool_activity"] is False

    def test_byo_agent_is_not_orbital_installable(self, client):
        resp = client.get("/api/v2/settings/sub-agents")
        claude = next(e for e in resp.json() if e["slug"] == "claude-code")
        # No orbital_install block -> the user brings their own binary.
        assert claude["install"]["supported"] is False
        assert claude["emits_tool_activity"] is True

    def test_unsupported_platform_is_distinguishable_from_byo(
            self, client, monkeypatch):
        """``supported: false`` alone conflates two different screens: an
        agent Orbital never installs (no button, no explanation) and one it
        installs elsewhere but not here (the "not yet supported on this
        platform" line). The echoed platform list separates them."""
        # A made-up host: the fixture lists every real OS.
        monkeypatch.setattr(installer_mod, "detect_os", lambda: "freebsd")

        entries = client.get("/api/v2/settings/sub-agents").json()
        managed = next(e for e in entries if e["slug"] == SLUG)
        claude = next(e for e in entries if e["slug"] == "claude-code")

        assert managed["install"]["supported"] is False
        assert managed["install"]["platforms"] == ALL_PLATFORMS
        assert claude["install"]["supported"] is False
        assert claude["install"]["platforms"] == []

    def test_supported_host_echoes_the_same_platform_list(self, client):
        assert fixture_entry(client)["install"]["platforms"] == ALL_PLATFORMS

    def test_windows_host_reports_the_agent_as_installable(self, client,
                                                           monkeypatch):
        """The Windows card gets the Install button, not the "not yet
        supported on this platform" line, when the manifest lists windows."""
        monkeypatch.setattr(installer_mod, "detect_os", lambda: "windows")
        entry = fixture_entry(client)
        assert entry["install"]["supported"] is True
        assert entry["install"]["platforms"] == ALL_PLATFORMS


class TestDeclaredCredentials:
    """``missing_credentials`` cannot drive a credential form on its own: it
    holds required-and-missing keys only. The declared list is what lets the
    card render an optional field and keep a manage affordance after the key
    is set."""

    def test_declares_both_credentials_with_state(self, client):
        creds = {c["key"]: c for c in fixture_entry(client)["credentials"]}
        assert set(creds) == {"NPM_COMPOSED_API_KEY", "NPM_COMPOSED_BASE_URL"}

        key = creds["NPM_COMPOSED_API_KEY"]
        assert key["label"] == "NPM Composed API Key"
        assert key["type"] == "secret"
        assert key["required"] is True
        assert key["has_setup_command"] is False   # no CLI login to delegate to
        assert key["configured"] is False
        assert creds["NPM_COMPOSED_BASE_URL"]["required"] is False
        assert creds["NPM_COMPOSED_BASE_URL"]["configured"] is False
        # No agent-owned store declared: Orbital's form is in charge.
        assert all(c["stored_by"] is None for c in creds.values())

    def test_unset_optional_credential_is_not_reported_configured(self, client):
        """The trap: ``check_credentials`` skips optional credentials, so they
        never appear in ``missing_credentials`` whether set or not. Deriving
        ``configured`` from that list marks every unset optional key as
        satisfied. (This used aider, the shipped agent with only optional
        keys; spec 171 B2 unlisted it, so the fixture's optional key carries
        the same check.)"""
        entry = fixture_entry(client)
        optional = [c for c in entry["credentials"] if not c["required"]]
        assert [c["key"] for c in optional] == ["NPM_COMPOSED_BASE_URL"]
        assert all(c["configured"] is False for c in optional)
        assert "NPM_COMPOSED_BASE_URL" not in (entry["missing_credentials"] or [])

    def test_configured_follows_the_store_then_the_environment(
            self, client, monkeypatch):
        import agent_os.api.routes.settings as settings_routes

        monkeypatch.setattr(settings_routes._setup_engine, "_credential_store",
                            DictStore(NPM_COMPOSED_API_KEY="sk-stored"))
        # The required key's state comes from the cached check_all pass.
        client.post("/api/v2/settings/sub-agents/refresh")
        creds = {c["key"]: c for c in fixture_entry(client)["credentials"]}
        assert creds["NPM_COMPOSED_API_KEY"]["configured"] is True
        assert creds["NPM_COMPOSED_BASE_URL"]["configured"] is False

        # The optional one resolves live, through the same env fallback the
        # spawn environment uses.
        monkeypatch.setenv("NPM_COMPOSED_BASE_URL", "https://api.example.test")
        creds = {c["key"]: c for c in fixture_entry(client)["credentials"]}
        assert creds["NPM_COMPOSED_BASE_URL"]["configured"] is True

    def test_oauth_credential_is_marked_as_cli_owned(self, client):
        """claude-code's key never comes through an Orbital form — the card
        has to send the user to ``claude login`` instead."""
        entries = client.get("/api/v2/settings/sub-agents").json()
        claude = next(e for e in entries if e["slug"] == "claude-code")
        cred = claude["credentials"][0]
        assert cred["type"] == "oauth_cli"
        assert cred["has_setup_command"] is True


class TestInstallRoute:

    def test_unknown_slug_404(self, client):
        resp = client.post("/api/v2/settings/sub-agents/nope/install")
        assert resp.status_code == 404

    def test_not_orbital_installable_400(self, client):
        resp = client.post("/api/v2/settings/sub-agents/claude-code/install")
        assert resp.status_code == 400

    def test_unsupported_platform_400(self, client, monkeypatch):
        # A made-up host: the fixture lists every real OS.
        monkeypatch.setattr(installer_mod, "detect_os", lambda: "freebsd")
        resp = client.post(f"/api/v2/settings/sub-agents/{SLUG}/install")
        assert resp.status_code == 400
        assert "freebsd" in resp.json()["detail"]

    def test_windows_host_gets_202(self, client, monkeypatch):
        """The install route does not 400 on Windows for an agent whose
        manifest lists it."""
        monkeypatch.setattr(installer_mod, "detect_os", lambda: "windows")
        exec_fake = FakeExec()
        monkeypatch.setattr(asyncio, "create_subprocess_exec", exec_fake)
        patch_which(monkeypatch)

        resp = client.post(f"/api/v2/settings/sub-agents/{SLUG}/install")
        assert resp.status_code == 202
        assert resp.json()["state"] == "installing"

        # Let the faked job settle so it never outlives this test.
        state = wait_for_job()
        assert state is not None and state["state"] == "installed"
        assert exec_fake.npm_calls, "npm ci must have been started"

    def test_failed_install_is_reported_on_the_entry(self, client, tmp_path,
                                                     monkeypatch):
        """A client that missed ``sub_agent_install_failed`` still lands on
        the failure screen from a plain refetch."""
        exec_fake = FakeExec(npm_lines=[b"npm error code E404\n"],
                             npm_returncode=1)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", exec_fake)
        patch_which(monkeypatch)

        resp = client.post(f"/api/v2/settings/sub-agents/{SLUG}/install")
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        state = wait_for_job()
        assert state is not None and state["state"] == "failed"

        entry = fixture_entry(client)
        assert entry["install"]["state"] == "failed"
        assert entry["install"]["job_id"] == job_id
        assert not os.path.exists(os.path.join(
            str(tmp_path / "data"), "agents", SLUG, ".install-complete"))

    def test_install_lifecycle_is_refresh_safe(self, client, tmp_path,
                                               monkeypatch):
        """202 immediately; a page refresh mid-install reads ``installing``
        from the entry (not from a WS event it may have missed); the finished
        install shows up as ``installed`` because the caches were dropped."""
        target = os.path.join(str(tmp_path / "data"), "agents", SLUG)
        gate = threading.Event()

        def plant_binary():
            bin_dir = os.path.join(target, "node_modules", ".bin")
            os.makedirs(bin_dir, exist_ok=True)
            # On Windows the manifest's auto_detect entry names npm's ``.cmd``
            # shim; elsewhere the ``#!/usr/bin/env node`` one.
            name = f"{COMMAND}.cmd" if sys.platform == "win32" else COMMAND
            path = os.path.join(bin_dir, name)
            with open(path, "w") as f:
                f.write("#!/bin/sh\n")
            os.chmod(path, 0o755)

        exec_fake = FakeExec(gate=gate, on_npm_wait=plant_binary)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", exec_fake)
        patch_which(monkeypatch)

        resp = client.post(f"/api/v2/settings/sub-agents/{SLUG}/install")
        assert resp.status_code == 202
        body = resp.json()
        assert body["slug"] == SLUG
        assert body["state"] == "installing"
        job_id = body["job_id"]

        # Mid-install refresh.
        entry = fixture_entry(client)
        assert entry["install"]["state"] == "installing"
        assert entry["install"]["job_id"] == job_id

        # A second install while the first is in flight.
        dup = client.post(f"/api/v2/settings/sub-agents/{SLUG}/install")
        assert dup.status_code == 409

        gate.set()
        state = wait_for_job()
        assert state is not None and state["state"] == "installed"

        # The client's own path: a plain refetch after the job finished.
        entry = fixture_entry(client)
        assert entry["install"]["state"] == "installed"
        assert entry["installed"] is True
        assert os.path.isfile(os.path.join(target, ".install-complete"))


# ---------------------------------------------------------------------------
# 4. Detection of the Orbital-installed tree (Windows ``.cmd`` shim)
# ---------------------------------------------------------------------------

class TestWindowsDetection:
    """On Windows the auto_detect entry names npm's ``.cmd`` shim — the only
    launcher CreateProcess can run. The extensionless file npm drops beside
    it is a POSIX sh script and must never be the one reported."""

    def _plant(self, engine, name):
        bin_dir = os.path.join(engine.data_dir, "agents", SLUG,
                               "node_modules", ".bin")
        os.makedirs(bin_dir, exist_ok=True)
        path = os.path.join(bin_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return path

    def test_resolves_the_cmd_shim_and_extends_the_daemon_path(
            self, engine, agent_manifest, monkeypatch):
        monkeypatch.setattr(setup_engine_mod, "detect_os", lambda: "windows")
        monkeypatch.setattr(setup_engine_mod.shutil, "which", lambda *_a, **_k: None)
        monkeypatch.setenv("PATH", os.pathsep.join(["/usr/bin", "/bin"]))
        posix = self._plant(engine, COMMAND)
        shim = self._plant(engine, f"{COMMAND}.cmd")

        resolved = engine.resolve_binary(agent_manifest)

        assert resolved == os.path.normpath(shim)
        assert resolved != posix
        assert os.path.dirname(shim) in os.environ["PATH"].split(os.pathsep)

    def test_posix_file_alone_is_not_detected_on_windows(
            self, engine, agent_manifest, monkeypatch):
        monkeypatch.setattr(setup_engine_mod, "detect_os", lambda: "windows")
        monkeypatch.setattr(setup_engine_mod.shutil, "which", lambda *_a, **_k: None)
        self._plant(engine, COMMAND)
        assert engine.resolve_binary(agent_manifest) is None
