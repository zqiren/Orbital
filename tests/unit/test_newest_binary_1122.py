# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Batch 1122: ``setup.binary_choice: newest`` — when an agent is found in
more than one place (a standalone CLI on PATH or in ``auto_detect``, the copy
inside a vendor desktop app), drive the copy whose ``--version`` is highest.

Codex is the case: Homebrew's codex-cli 0.157.1 lists no gpt-6.2 models,
while the codex inside ChatGPT.app (0.171.0-alpha.13) does, for the same
signed-in account. The standalone CLI used to win unconditionally.

Every launch goes through a fake ``subprocess.run``: nothing real runs.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from unittest.mock import patch

import pytest

from agent_os.agents.desktop_apps import DesktopApp, DesktopAppHit, DesktopAppPlatform
from agent_os.agents.manifest import (
    AgentManifest,
    ManifestCapabilities,
    ManifestPermissions,
    ManifestRuntime,
    ManifestSetup,
)
from agent_os.agents.registry import AgentRegistry
from agent_os.agents.setup_engine import SetupEngine, release_key
from agent_os.daemon_v2.sub_agent_config_store import SubAgentConfigStore

pytestmark = pytest.mark.real_start_check

SLUG = "codex"


def _manifest(*, choice="newest", auto_detect=None, desktop=True) -> AgentManifest:
    apps = []
    if desktop:
        apps = [DesktopApp(label="ChatGPT", order="after_cli", platforms={
            "macos": DesktopAppPlatform(executable="Contents/Resources/codex-cli/bin/codex"),
        })]
    return AgentManifest(
        manifest_version="1", name="Codex", slug=SLUG, description="d",
        author="t", version="1.0.0",
        runtime=ManifestRuntime(adapter="cli", command="codex", args=[]),
        setup=ManifestSetup(check_command="codex --version",
                            auto_detect=auto_detect or {},
                            desktop_apps=apps, binary_choice=choice),
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
    """``subprocess.run`` stand-in. ``versions`` maps a path fragment to what
    its ``--version`` prints; ``slow`` paths time out, ``broken`` exit 1."""

    def __init__(self, versions, slow=(), broken=()):
        self.versions = dict(versions)
        self.slow = set(slow)
        self.broken = set(broken)
        self.calls: list[str] = []

    def __call__(self, cmd, **kwargs):
        text = cmd if isinstance(cmd, str) else " ".join(cmd)
        self.calls.append(text)
        if any(p in text for p in self.slow):
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))
        if any(p in text for p in self.broken):
            return subprocess.CompletedProcess(cmd, 1, "", "dyld: Library not loaded")
        # Longest fragment first, so "a/codex" does not shadow "a/codex.exe".
        for fragment in sorted(self.versions, key=len, reverse=True):
            if fragment in text:
                return subprocess.CompletedProcess(cmd, 0, self.versions[fragment], "")
        return subprocess.CompletedProcess(cmd, 0, "codex-cli 0.1.0\n", "")

    def launches_of(self, path: str) -> int:
        return sum(1 for c in self.calls if path in c)


def _engine(tmp_path, manifest):
    reg = AgentRegistry()
    reg.register(manifest)
    store = SubAgentConfigStore(str(tmp_path / "cfg" / "sub_agent_config.json"))
    return SetupEngine(reg, sub_agent_config_store=store,
                       data_dir=str(tmp_path / "data")), store


def _desktop(executable, version="26.1118.0"):
    def _find(app, os_name, **_kw):
        if executable is None or os_name != "macos":
            return None
        return DesktopAppHit(label=app.label, app_path=os.path.dirname(executable),
                             executable=executable, version=version)
    return patch("agent_os.agents.desktop_apps.find_desktop_app", _find)


def _which(path):
    return patch("agent_os.agents.setup_engine.shutil.which", return_value=path)


def _os(name="macos"):
    return patch("agent_os.agents.setup_engine.detect_os", return_value=name)


def _run(fake):
    return patch("agent_os.agents.setup_engine.subprocess.run", fake)


@pytest.fixture
def copies(tmp_path):
    cli = _file(tmp_path / "homebrew" / "bin" / "codex")
    app = _file(tmp_path / "ChatGPT.app" / "Contents" / "Resources" / "codex-cli" / "bin" / "codex")
    return cli, app


# ---------------------------------------------------------------------------
# Version parsing
# ---------------------------------------------------------------------------

def test_release_key_orders_release_first_then_stable_over_prerelease():
    assert release_key("codex-cli 0.171.0-alpha.13") > release_key("codex-cli 0.157.1")
    assert release_key("codex-cli 0.171.0") > release_key("codex-cli 0.171.0-alpha.13")
    assert release_key("codex-cli 0.171.0-alpha.13") > release_key("codex-cli 0.171.0-alpha.2")
    assert release_key("codex-cli 0.171.0-beta.1") > release_key("codex-cli 0.171.0-alpha.13")
    assert release_key("0.172.0-alpha.1") > release_key("0.171.0")
    assert release_key("v1.10") > release_key("v1.9.9")


@pytest.mark.parametrize("text", [None, "", "codex-cli dev", "unknown", "v1"])
def test_release_key_is_none_for_text_without_a_version(text):
    assert release_key(text) is None


# ---------------------------------------------------------------------------
# Both copies present
# ---------------------------------------------------------------------------

def test_a_newer_desktop_copy_wins_over_the_standalone_cli(tmp_path, copies):
    cli, app = copies
    engine, _ = _engine(tmp_path, _manifest())
    fake = FakeRun({cli: "codex-cli 0.157.1\n", app: "codex-cli 0.171.0-alpha.13\n"})
    with _os(), _which(cli), _desktop(app), _run(fake):
        status = engine.check_agent(SLUG)
    assert status.binary_path == app
    assert status.version == "codex-cli 0.171.0-alpha.13"
    source = engine.resolved_source(SLUG)
    assert source["kind"] == "desktop_app" and source["label"] == "ChatGPT"


def test_a_newer_standalone_cli_wins_over_the_desktop_copy(tmp_path, copies):
    cli, app = copies
    engine, _ = _engine(tmp_path, _manifest())
    fake = FakeRun({cli: "codex-cli 0.172.0\n", app: "codex-cli 0.171.0-alpha.13\n"})
    with _os(), _which(cli), _desktop(app), _run(fake):
        status = engine.check_agent(SLUG)
    assert status.binary_path == cli
    assert status.version == "codex-cli 0.172.0"
    assert engine.resolved_source(SLUG) == {"kind": "path"}


def test_on_an_equal_release_the_stable_standalone_beats_an_alpha(tmp_path, copies):
    cli, app = copies
    engine, _ = _engine(tmp_path, _manifest())
    fake = FakeRun({cli: "codex-cli 0.171.0\n", app: "codex-cli 0.171.0-alpha.13\n"})
    with _os(), _which(cli), _desktop(app), _run(fake):
        assert engine.resolve_binary(engine._registry.get(SLUG)) == cli


def test_identical_versions_keep_the_standalone_cli(tmp_path, copies):
    cli, app = copies
    engine, _ = _engine(tmp_path, _manifest())
    fake = FakeRun({cli: "codex-cli 0.171.0\n", app: "codex-cli 0.171.0\n"})
    with _os(), _which(cli), _desktop(app), _run(fake):
        assert engine.resolve_binary(engine._registry.get(SLUG)) == cli


# ---------------------------------------------------------------------------
# Only one copy present: unchanged
# ---------------------------------------------------------------------------

def test_only_the_standalone_cli(tmp_path, copies):
    cli, _app = copies
    engine, _ = _engine(tmp_path, _manifest())
    fake = FakeRun({cli: "codex-cli 0.157.1\n"})
    with _os(), _which(cli), _desktop(None), _run(fake):
        assert engine.resolve_binary(engine._registry.get(SLUG)) == cli
    assert engine.resolved_source(SLUG) == {"kind": "path"}
    assert fake.launches_of(cli) == 1


def test_only_the_desktop_copy(tmp_path, copies):
    _cli, app = copies
    engine, _ = _engine(tmp_path, _manifest())
    fake = FakeRun({app: "codex-cli 0.171.0-alpha.13\n"})
    with _os(), _which(None), _desktop(app), _run(fake):
        assert engine.resolve_binary(engine._registry.get(SLUG)) == app
    assert engine.resolved_source(SLUG)["kind"] == "desktop_app"


def test_without_binary_choice_the_standalone_cli_still_wins(tmp_path, copies):
    """A manifest that does not opt in keeps ``after_cli``'s old meaning, and
    the desktop copy is never even launched."""
    cli, app = copies
    engine, _ = _engine(tmp_path, _manifest(choice="first"))
    fake = FakeRun({cli: "codex-cli 0.157.1\n", app: "codex-cli 0.171.0-alpha.13\n"})
    with _os(), _which(cli), _desktop(app), _run(fake):
        assert engine.resolve_binary(engine._registry.get(SLUG)) == cli
    assert fake.launches_of(app) == 0


# ---------------------------------------------------------------------------
# Versions that cannot be read
# ---------------------------------------------------------------------------

def test_a_copy_whose_version_cannot_be_parsed_never_wins(tmp_path, copies):
    cli, app = copies
    engine, _ = _engine(tmp_path, _manifest())
    fake = FakeRun({cli: "codex-cli 0.157.1\n", app: "codex-cli (dev build)\n"})
    with _os(), _which(cli), _desktop(app), _run(fake):
        assert engine.resolve_binary(engine._registry.get(SLUG)) == cli


def test_an_unparsable_standalone_loses_to_a_parsable_desktop_copy(tmp_path, copies):
    cli, app = copies
    engine, _ = _engine(tmp_path, _manifest())
    fake = FakeRun({cli: "codex-cli dev\n", app: "codex-cli 0.171.0-alpha.13\n"})
    with _os(), _which(cli), _desktop(app), _run(fake):
        assert engine.resolve_binary(engine._registry.get(SLUG)) == app


def test_a_version_check_that_times_out_never_wins(tmp_path, copies):
    """A timeout still counts as a start (spec 144), but it read no version."""
    cli, app = copies
    engine, _ = _engine(tmp_path, _manifest())
    fake = FakeRun({cli: "codex-cli 0.157.1\n"}, slow=[app])
    with _os(), _which(cli), _desktop(app), _run(fake):
        assert engine.resolve_binary(engine._registry.get(SLUG)) == cli


def test_a_copy_that_does_not_start_is_passed_over(tmp_path, copies):
    cli, app = copies
    engine, _ = _engine(tmp_path, _manifest())
    fake = FakeRun({cli: "codex-cli 0.157.1\n"}, broken=[app])
    with _os(), _which(cli), _desktop(app), _run(fake):
        assert engine.resolve_binary(engine._registry.get(SLUG)) == cli
    assert engine.resolved_source(SLUG) == {"kind": "path"}


def test_both_unparsable_keeps_the_first_candidate_and_never_raises(tmp_path, copies):
    cli, app = copies
    engine, _ = _engine(tmp_path, _manifest())
    fake = FakeRun({cli: "???\n", app: "!!!\n"})
    with _os(), _which(cli), _desktop(app), _run(fake):
        assert engine.resolve_binary(engine._registry.get(SLUG)) == cli


# ---------------------------------------------------------------------------
# The user's own choice
# ---------------------------------------------------------------------------

def test_an_explicit_user_path_wins_over_a_newer_desktop_copy(tmp_path, copies):
    cli, app = copies
    engine, store = _engine(tmp_path, _manifest())
    store.set_override(SLUG, binary=cli)
    fake = FakeRun({cli: "codex-cli 0.157.1\n", app: "codex-cli 0.171.0-alpha.13\n"})
    with _os(), _which(None), _desktop(app), _run(fake):
        assert engine.resolve_binary(engine._registry.get(SLUG)) == cli
    assert engine.resolved_source(SLUG) == {"kind": "override"}
    assert fake.launches_of(app) == 0


# ---------------------------------------------------------------------------
# Cost: one version probe per copy, cached per file identity
# ---------------------------------------------------------------------------

def test_versions_are_read_once_per_file_identity(tmp_path, copies):
    cli, app = copies
    engine, _ = _engine(tmp_path, _manifest())
    fake = FakeRun({cli: "codex-cli 0.157.1\n", app: "codex-cli 0.171.0-alpha.13\n"})
    with _os(), _which(cli), _desktop(app), _run(fake):
        engine.check_agent(SLUG)
        engine.invalidate_resolved_path(SLUG)  # the user's Refresh drops the pin
        assert engine.resolve_binary(engine._registry.get(SLUG)) == app
        engine.check_agent(SLUG)
    assert fake.launches_of(app + " --version") == 1
    assert fake.launches_of(cli + " --version") == 1


def test_a_cli_updated_past_the_desktop_copy_wins_after_refresh(tmp_path, copies):
    cli, app = copies
    engine, _ = _engine(tmp_path, _manifest())
    fake = FakeRun({cli: "codex-cli 0.157.1\n", app: "codex-cli 0.171.0-alpha.13\n"})
    with _os(), _which(cli), _desktop(app), _run(fake):
        assert engine.resolve_binary(engine._registry.get(SLUG)) == app
    # brew upgrade rewrites the file: a new identity, so it is asked again.
    _file(tmp_path / "homebrew" / "bin" / "codex", 2_000_000_000)
    engine.invalidate_resolved_path(SLUG)
    fake.versions[cli] = "codex-cli 0.172.0\n"
    with _os(), _which(cli), _desktop(app), _run(fake):
        assert engine.resolve_binary(engine._registry.get(SLUG)) == cli


def test_a_desktop_copy_that_wins_never_lands_on_the_daemon_path(tmp_path, copies, monkeypatch):
    cli, app = copies
    monkeypatch.setenv("PATH", "/usr/bin")
    listed = _file(tmp_path / "npm" / "codex")
    engine, _ = _engine(tmp_path, _manifest(auto_detect={"macos": [listed]}))
    fake = FakeRun({listed: "codex-cli 0.157.1\n", app: "codex-cli 0.171.0-alpha.13\n"})
    with _os(), _which(None), _desktop(app), _run(fake):
        assert engine.resolve_binary(engine._registry.get(SLUG)) == app
    assert os.environ["PATH"] == "/usr/bin"


def test_an_auto_detect_copy_that_wins_is_appended_to_the_daemon_path(tmp_path, copies, monkeypatch):
    _cli, app = copies
    monkeypatch.setenv("PATH", "/usr/bin")
    listed = _file(tmp_path / "npm" / "codex")
    engine, _ = _engine(tmp_path, _manifest(auto_detect={"macos": [listed]}))
    fake = FakeRun({listed: "codex-cli 0.172.0\n", app: "codex-cli 0.171.0-alpha.13\n"})
    with _os(), _which(None), _desktop(app), _run(fake):
        assert engine.resolve_binary(engine._registry.get(SLUG)) == listed
    assert os.environ["PATH"] == os.pathsep.join(["/usr/bin", os.path.dirname(listed)])


# ---------------------------------------------------------------------------
# Windows: the desktop copy is an auto_detect glob (not run on Windows here)
# ---------------------------------------------------------------------------

def _windows_layout(tmp_path):
    npm = _file(tmp_path / "AppData" / "Roaming" / "npm" / "codex.cmd")
    newest = _file(tmp_path / "Local" / "OpenAI" / "Codex" / "bin" / "0aaa" / "codex.exe",
                   2_000_000_000)
    older = _file(tmp_path / "Local" / "OpenAI" / "Codex" / "bin" / "ffff" / "codex.exe",
                  1_900_000_000)
    pattern = str(tmp_path / "Local" / "OpenAI" / "Codex" / "bin" / "*" / "codex.exe")
    return npm, newest, older, pattern


def test_windows_a_newer_desktop_app_cli_wins_over_npm(tmp_path):
    npm, newest, _older, pattern = _windows_layout(tmp_path)
    engine, _ = _engine(tmp_path, _manifest(
        desktop=False, auto_detect={"windows": [npm, pattern]}))
    fake = FakeRun({npm: "codex-cli 0.157.1\n", newest: "codex-cli 0.171.0-alpha.13\n"})
    with _os("windows"), _which(None), _run(fake):
        assert engine.resolve_binary(engine._registry.get(SLUG)) == newest
    assert engine.resolved_source(SLUG) == {"kind": "auto_detect"}


def test_windows_a_newer_npm_cli_wins_over_the_desktop_app_cli(tmp_path):
    npm, newest, _older, pattern = _windows_layout(tmp_path)
    engine, _ = _engine(tmp_path, _manifest(
        desktop=False, auto_detect={"windows": [npm, pattern]}))
    fake = FakeRun({npm: "codex-cli 0.172.0\n", newest: "codex-cli 0.171.0-alpha.13\n"})
    with _os("windows"), _which(None), _run(fake):
        assert engine.resolve_binary(engine._registry.get(SLUG)) == npm


def test_windows_npm_on_path_and_in_auto_detect_is_probed_once(tmp_path):
    npm, newest, _older, pattern = _windows_layout(tmp_path)
    engine, _ = _engine(tmp_path, _manifest(
        desktop=False, auto_detect={"windows": [npm, pattern]}))
    fake = FakeRun({npm: "codex-cli 0.157.1\n", newest: "codex-cli 0.171.0-alpha.13\n"})
    with _os("windows"), _which(npm), _run(fake):
        assert engine.resolve_binary(engine._registry.get(SLUG)) == newest
    assert fake.launches_of(npm) == 1


def test_windows_stale_hash_folder_still_falls_to_the_next_newest(tmp_path):
    """Spec 144 unchanged: the newest-modified hash folder that does not start
    is passed over for the next one, which then competes on version."""
    npm, newest, older, pattern = _windows_layout(tmp_path)
    engine, _ = _engine(tmp_path, _manifest(
        desktop=False, auto_detect={"windows": [npm, pattern]}))
    fake = FakeRun({npm: "codex-cli 0.157.1\n", older: "codex-cli 0.170.0\n"},
                   broken=[newest])
    with _os("windows"), _which(None), _run(fake):
        assert engine.resolve_binary(engine._registry.get(SLUG)) == older


def test_windows_only_npm(tmp_path):
    npm = _file(tmp_path / "AppData" / "Roaming" / "npm" / "codex.cmd")
    pattern = str(tmp_path / "Local" / "OpenAI" / "Codex" / "bin" / "*" / "codex.exe")
    engine, _ = _engine(tmp_path, _manifest(
        desktop=False, auto_detect={"windows": [npm, pattern]}))
    fake = FakeRun({npm: "codex-cli 0.157.1\n"})
    with _os("windows"), _which(None), _run(fake):
        assert engine.resolve_binary(engine._registry.get(SLUG)) == npm


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------

def test_codex_manifest_opts_in():
    from agent_os.agents.manifest import ManifestLoader
    path = os.path.join(os.path.dirname(__file__), "..", "..", "agent_os", "agents",
                        "manifests", "codex.yaml")
    manifest = ManifestLoader.load(path)
    assert manifest.setup.binary_choice == "newest"
    # The desktop copy is still listed after the CLI: the tie-break order.
    assert [a.order for a in manifest.setup.desktop_apps] == ["after_cli"]


def test_an_unknown_binary_choice_reads_as_first():
    from agent_os.agents.manifest import parse_binary_choice
    assert parse_binary_choice(None) == "first"
    assert parse_binary_choice("newest") == "newest"
    assert parse_binary_choice("latest") == "first"
    assert parse_binary_choice(3) == "first"


# ---------------------------------------------------------------------------
# Model list caches follow the binary
# ---------------------------------------------------------------------------

def test_codex_model_cache_is_per_binary(monkeypatch):
    from agent_os.agent.transports import codex_models
    codex_models.clear_codex_models_cache()
    lists = {"/old/codex": ["gpt-6-sol"], "/app/codex": ["gpt-6.2-sol", "gpt-6-sol"]}

    async def fake_fetch(binary, **_kw):
        return list(lists[binary])

    monkeypatch.setattr(codex_models, "fetch_codex_models", fake_fetch)
    try:
        assert asyncio.run(codex_models.get_codex_models_cached("/old/codex")) == ["gpt-6-sol"]
        assert asyncio.run(codex_models.get_codex_models_cached("/app/codex"))[0] == "gpt-6.2-sol"
    finally:
        codex_models.clear_codex_models_cache()


def test_a_slow_model_read_never_serves_another_binarys_list(monkeypatch):
    """While the live list is still probing, the settings read serves the list
    it produced last time — only when that list came from the same binary."""
    from agent_os.api.routes import settings as settings_routes
    monkeypatch.setattr(settings_routes, "_last_live_schemas", {})
    old = [{"slug": "codex", "binary_path": "/old/codex",
            "param_schema": {"model": {"allowed": None}}, "config": {}}]
    augmented = [{"slug": "codex", "binary_path": "/old/codex",
                  "param_schema": {"model": {"allowed": ["gpt-6-sol"]}}, "config": {}}]
    entries = [dict(e) for e in old]
    settings_routes._merge_live("codex", entries, old, augmented)

    same = [{"slug": "codex", "binary_path": "/old/codex",
             "param_schema": {"model": {"allowed": None}}, "config": {}}]
    settings_routes._serve_last_live("codex", same)
    assert same[0]["param_schema"]["model"]["allowed"] == ["gpt-6-sol"]

    switched = [{"slug": "codex", "binary_path": "/app/codex",
                 "param_schema": {"model": {"allowed": None}}, "config": {}}]
    settings_routes._serve_last_live("codex", switched)
    assert switched[0]["param_schema"]["model"]["allowed"] is None
