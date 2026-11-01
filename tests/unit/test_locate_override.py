# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The user-chosen agent location (the locate card): persisted beside the
sub-agent params, strategy 0 of ``resolve_binary``, carried into the spawn
env (PATH prepend + env names), never a silent fallback when the chosen file
is gone, and the four settings routes that drive the card.
"""

from __future__ import annotations

import os
import stat
import sys
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.agents.manifest import (
    AgentManifest,
    ManifestCapabilities,
    ManifestCredential,
    ManifestDependency,
    ManifestPermissions,
    ManifestRuntime,
    ManifestSetup,
)
from agent_os.agents.registry import AgentRegistry
from agent_os.agents.setup_engine import SetupEngine
from agent_os.api.routes import settings as settings_routes
from agent_os.daemon_v2.sub_agent_config_store import (
    SubAgentConfigError,
    SubAgentConfigStore,
)
from agent_os.telemetry.diagnostics import install_kind_of

# The fake CLIs below are sh scripts; Windows cannot execute them (WinError
# 193). The runner itself is argv-only on every platform, and the Windows
# shim refusal has its own test, so the executing tests skip there.
_runs_fake_cli = pytest.mark.skipif(sys.platform == "win32",
                                    reason="fake CLIs are sh scripts")

SLUG = "codex"


def _manifest(check_command="codex --version", credentials=None,
              dependencies=None) -> AgentManifest:
    return AgentManifest(
        manifest_version="1", name="Codex", slug=SLUG, description="d",
        author="t", version="1.0.0",
        runtime=ManifestRuntime(adapter="cli", command="codex", args=["exec"]),
        setup=ManifestSetup(check_command=check_command,
                            credentials=credentials or [],
                            dependencies=dependencies or []),
        capabilities=ManifestCapabilities(),
        permissions=ManifestPermissions(),
    )


def _fake_cli(tmp_path, name="codex", stdout="codex-cli 0.157.1", rc=0):
    """An executable that prints a version line (needs node on PATH when
    ``needs_node`` is set, mirroring an npm shim)."""
    path = tmp_path / name
    path.write_text(f"#!/bin/sh\necho '{stdout}'\nexit {rc}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def _node_shim_cli(tmp_path):
    """A shim that dies unless a ``node`` is on PATH — the codex.js shape."""
    path = tmp_path / "codex"
    path.write_text("#!/bin/sh\nnode --version >/dev/null 2>&1 || "
                    "{ echo 'env: node: No such file or directory' >&2; exit 127; }\n"
                    "echo 'codex-cli 0.157.1'\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    node_dir = tmp_path / "node@24" / "bin"
    node_dir.mkdir(parents=True)
    node = node_dir / "node"
    node.write_text("#!/bin/sh\necho v24.13.0\n")
    node.chmod(node.stat().st_mode | stat.S_IXUSR)
    return str(path), str(node_dir)


@pytest.fixture
def store(tmp_path):
    return SubAgentConfigStore(os.path.join(str(tmp_path), "cfg", "sub_agent_config.json"))


@pytest.fixture
def engine(store):
    reg = AgentRegistry()
    reg.register(_manifest())
    return SetupEngine(reg, sub_agent_config_store=store)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class TestStoreOverride:
    def test_round_trip_and_params_untouched(self, store, tmp_path):
        binary = _fake_cli(tmp_path)
        store.set(SLUG, {"model": "gpt-6-sol"})
        xbin = str(tmp_path / "xbin")
        store.set_override(SLUG, binary=binary, path_prepend=[xbin],
                           env={"CODEX_HOME": ""})
        assert store.get_override(SLUG) == {
            "binary": binary, "path_prepend": [xbin], "env": {"CODEX_HOME": ""}}
        # Params API never sees the override keys, and vice versa.
        assert store.get(SLUG) == {"model": "gpt-6-sol"}
        assert store.build_extra_args(SLUG) == ["-m", "gpt-6-sol"]
        # A later params write keeps the override.
        store.set(SLUG, {"model": "gpt-6-luna"})
        assert store.get_override(SLUG)["binary"] == binary
        # Survives a reload from disk with native types.
        again = SubAgentConfigStore(store.path)
        assert again.get_override(SLUG) == store.get_override(SLUG)
        assert again.get(SLUG) == {"model": "gpt-6-luna"}

    def test_clear(self, store, tmp_path):
        store.set_override(SLUG, binary=_fake_cli(tmp_path))
        store.clear_override(SLUG)
        assert store.get_override(SLUG) is None
        assert SubAgentConfigStore(store.path).get_override(SLUG) is None

    def test_override_keys_rejected_as_params(self, store):
        with pytest.raises(SubAgentConfigError):
            store.set(SLUG, {"binary": "/x"})

    def test_relative_binary_rejected(self, store):
        with pytest.raises(SubAgentConfigError, match="absolute"):
            store.set_override(SLUG, binary="codex")

    def test_missing_file_rejected(self, store, tmp_path):
        with pytest.raises(SubAgentConfigError, match="not a file"):
            store.set_override(SLUG, binary=str(tmp_path / "nope"))

    def test_windows_shim_rejected_with_the_real_executable_named(self, store, tmp_path):
        shim = tmp_path / "codex.cmd"
        shim.write_text("@echo off\n")
        with patch("agent_os.daemon_v2.sub_agent_config_store.detect_os", return_value="windows"):
            with pytest.raises(SubAgentConfigError) as exc:
                store.set_override(SLUG, binary=str(shim))
        assert "codex.cmd" in str(exc.value)
        assert "real executable" in str(exc.value)

    def test_windows_shim_allowed_for_an_agent_that_launches_via_a_shim(self, store, tmp_path):
        """dsh's only Windows launcher IS the npm .cmd (spec 127); the Save
        must accept the same file the card's Test accepted (2026-10-22)."""
        shim = tmp_path / "dsh-acp-demo.cmd"
        shim.write_text("@echo off\n")
        with patch("agent_os.daemon_v2.sub_agent_config_store.detect_os", return_value="windows"):
            cleaned = store.set_override(SLUG, binary=str(shim), allow_shim=True)
        assert cleaned["binary"] == str(shim)
        assert store.get_override(SLUG)["binary"] == str(shim)

    def test_manifest_launches_via_shim_reads_the_windows_auto_detect(self):
        from types import SimpleNamespace as NS
        from agent_os.daemon_v2.sub_agent_config_store import SubAgentConfigStore

        def manifest(windows):
            return NS(setup=NS(auto_detect={"windows": windows, "macos": ["/x/bin/dsh"]}))

        # dsh: the npm shim is the only launcher there is.
        assert SubAgentConfigStore.manifest_launches_via_shim(
            manifest(["${ORBITAL_DATA_DIR}/agents/dsh/node_modules/.bin/dsh-acp-demo.cmd"]))
        # codex/pi/kimi/minimax: shims only, in two npm locations.
        assert SubAgentConfigStore.manifest_launches_via_shim(
            manifest(["%APPDATA%\\npm\\codex.cmd", "%PROGRAMFILES%\\nodejs\\codex.cmd"]))
        # claude-code also ships claude.exe: a pasted shim is still refused
        # (spec 116), the real executable is the answer.
        assert not SubAgentConfigStore.manifest_launches_via_shim(
            manifest(["%APPDATA%\\npm\\claude.cmd", "%USERPROFILE%\\.local\\bin\\claude.exe"]))
        assert not SubAgentConfigStore.manifest_launches_via_shim(
            manifest(["C:/Program Files/x/x.exe"]))
        assert not SubAgentConfigStore.manifest_launches_via_shim(manifest([]))
        assert not SubAgentConfigStore.manifest_launches_via_shim(NS(setup=None))

    def test_bad_env_name_and_relative_path_entry_rejected(self, store, tmp_path):
        binary = _fake_cli(tmp_path)
        with pytest.raises(SubAgentConfigError, match="environment variable"):
            store.set_override(SLUG, binary=binary, env={"not a name": ""})
        with pytest.raises(SubAgentConfigError, match="absolute"):
            store.set_override(SLUG, binary=binary, path_prepend=["bin"])


# ---------------------------------------------------------------------------
# Engine: strategy 0, the gone state, spawn env
# ---------------------------------------------------------------------------

class TestResolveStrategyZero:
    @_runs_fake_cli
    @patch("agent_os.agents.setup_engine.shutil.which", return_value="/elsewhere/codex")
    def test_override_beats_which(self, _which, engine, store, tmp_path):
        binary = _fake_cli(tmp_path)
        store.set_override(SLUG, binary=binary)
        assert engine.resolve_binary(engine._registry.get(SLUG)) == binary
        status = engine.check_agent(SLUG)
        assert status.installed and status.binary_path == binary
        assert status.override_path == binary and status.override_missing is False
        assert "0.157.1" in (status.version or "")

    @patch("agent_os.agents.setup_engine.shutil.which", return_value=None)
    def test_gone_file_with_no_other_copy_reads_gone(self, _which, engine, store, tmp_path):
        binary = _fake_cli(tmp_path)
        store.set_override(SLUG, binary=binary)
        os.remove(binary)
        engine.invalidate_resolved_path(SLUG)
        assert engine.resolve_binary(engine._registry.get(SLUG)) is None
        status = engine.check_agent(SLUG)
        assert status.installed is False
        assert status.override_missing is True
        assert status.override_path == binary

    @patch("agent_os.agents.setup_engine.shutil.which")
    def test_gone_file_follows_the_agent_when_detection_finds_it(
            self, which, engine, store, tmp_path):
        """A vendor update moved the chosen file (the Codex desktop app keeps
        its CLI under bin/<hash>/). Detection finding the agent elsewhere must
        not leave it "not installed"; the stored override stays, and wins again
        if its file comes back."""
        binary = _fake_cli(tmp_path)
        elsewhere = _fake_cli(tmp_path, name="codex-newer")
        which.return_value = elsewhere
        store.set_override(SLUG, binary=binary)
        assert engine.resolve_binary(engine._registry.get(SLUG)) == binary

        os.remove(binary)  # the pin is re-checked, no restart or invalidate
        assert engine.resolve_binary(engine._registry.get(SLUG)) == elsewhere
        status = engine.check_agent(SLUG)
        assert status.installed is True and status.binary_path == elsewhere
        assert status.override_missing is False and status.override_path is None
        assert store.get_override(SLUG) is not None

    @_runs_fake_cli
    @patch("agent_os.agents.setup_engine.shutil.which", return_value=None)
    def test_path_prepend_reaches_the_version_check_and_daemon_path(
            self, _which, engine, store, tmp_path, monkeypatch):
        binary, node_dir = _node_shim_cli(tmp_path)
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        store.set_override(SLUG, binary=binary, path_prepend=[node_dir])
        status = engine.check_agent(SLUG)
        assert status.installed is True
        assert "0.157.1" in (status.version or "")
        # D3 rule: the directory joins the daemon PATH so dependency checks
        # (``node --version``) see it too.
        assert node_dir in os.environ["PATH"].split(os.pathsep)

    def test_invalidate_after_save_drops_a_which_hit(self, engine, store, tmp_path):
        with patch("agent_os.agents.setup_engine.shutil.which", return_value="/elsewhere/codex"):
            assert engine.resolve_binary(engine._registry.get(SLUG)) == "/elsewhere/codex"
        binary = _fake_cli(tmp_path)
        store.set_override(SLUG, binary=binary)
        engine.invalidate_resolved_path(SLUG)
        assert engine.resolve_binary(engine._registry.get(SLUG)) == binary


class TestSpawnEnv:
    @patch("agent_os.agents.setup_engine.shutil.which", return_value=None)
    def test_path_prepend_and_env_names_land_in_the_adapter_env(
            self, _which, store, tmp_path, monkeypatch):
        binary = _fake_cli(tmp_path)
        cred = ManifestCredential(key="CODEX_KEY", label="k", env_var="CODEX_KEY", required=False)
        reg = AgentRegistry()
        reg.register(_manifest(credentials=[cred]))
        engine = SetupEngine(reg, sub_agent_config_store=store)
        base_a, base_b = str(tmp_path / "base-a"), str(tmp_path / "base-b")
        monkeypatch.setenv("PATH", os.pathsep.join([base_a, base_b]))
        monkeypatch.setenv("CODEX_KEY", "from-manifest-cred")
        node_bin = str(tmp_path / "node@24" / "bin")
        store.set_override(SLUG, binary=binary, path_prepend=[node_bin],
                           env={"CODEX_KEY": "user-typed", "CODEX_HOME": "/tmp/codex-home"})
        cfg = engine.get_adapter_config(SLUG, "/ws")
        assert cfg["command"] == binary
        path = cfg["env"]["PATH"].split(os.pathsep)
        assert path[0] == node_bin
        assert base_a in path and base_b in path
        # User env never overrides a credential the manifest injects.
        assert cfg["env"]["CODEX_KEY"] == "from-manifest-cred"
        assert cfg["env"]["CODEX_HOME"] == "/tmp/codex-home"

    @patch("agent_os.agents.setup_engine.shutil.which", return_value="/elsewhere/codex")
    def test_no_override_means_no_path_key(self, _which, engine):
        cfg = engine.get_adapter_config(SLUG, "/ws")
        assert "PATH" not in cfg["env"]


class TestTestOverride:
    @_runs_fake_cli
    def test_runs_check_command_without_a_shell(self, engine, tmp_path):
        binary, node_dir = _node_shim_cli(tmp_path)
        with patch.dict(os.environ, {"PATH": "/usr/bin:/bin"}):
            bad = engine.test_override(SLUG, binary, path_prepend=[])
            good = engine.test_override(SLUG, binary, path_prepend=[node_dir])
        assert bad["ok"] is False and "node" in bad["error"]
        assert good["ok"] is True and "0.157.1" in good["version_output"]

    def test_missing_file_is_a_clear_error(self, engine, tmp_path):
        r = engine.test_override(SLUG, str(tmp_path / "gone"), path_prepend=[])
        assert r["ok"] is False and "gone" in r["error"]

    def test_no_check_command_passes_an_executable_without_spawning(
            self, tmp_path, store, monkeypatch):
        """Spec 127 D2: an agent with no ``check_command`` (dsh is an ACP
        stdio server that takes only ``--config``) is tested by existence
        and executability alone — never by running it, which would hang on
        stdin until the timeout."""
        binary = _fake_cli(tmp_path)
        reg = AgentRegistry()
        reg.register(_manifest(check_command=None))
        engine = SetupEngine(reg, sub_agent_config_store=store)

        def _never(*_a, **_k):
            raise AssertionError("test_override must not spawn without a check command")
        monkeypatch.setattr("agent_os.agents.setup_engine.subprocess.run", _never)

        r = engine.test_override(SLUG, binary, path_prepend=[])
        assert r == {"ok": True, "version_output": ""}

    def test_no_check_command_still_needs_the_file(self, tmp_path, store):
        reg = AgentRegistry()
        reg.register(_manifest(check_command=None))
        engine = SetupEngine(reg, sub_agent_config_store=store)
        r = engine.test_override(SLUG, str(tmp_path / "gone"), path_prepend=[])
        assert r["ok"] is False and "gone" in r["error"]

    @pytest.mark.skipif(sys.platform == "win32",
                        reason="Windows has no execute bit")
    def test_no_check_command_refuses_a_non_executable_file(
            self, tmp_path, store):
        path = tmp_path / "codex"
        path.write_text("#!/bin/sh\n")
        path.chmod(0o644)
        reg = AgentRegistry()
        reg.register(_manifest(check_command=None))
        engine = SetupEngine(reg, sub_agent_config_store=store)
        r = engine.test_override(SLUG, str(path), path_prepend=[])
        assert r["ok"] is False and "not executable" in r["error"]

    @_runs_fake_cli
    def test_space_in_path_survives(self, tmp_path, store):
        d = tmp_path / "Application Support"
        d.mkdir()
        binary = _fake_cli(d)
        reg = AgentRegistry()
        reg.register(_manifest())
        engine = SetupEngine(reg, sub_agent_config_store=store)
        r = engine.test_override(SLUG, binary, path_prepend=[])
        assert r["ok"] is True


def test_install_kind_user():
    assert install_kind_of("/opt/homebrew/bin/codex", user_override=True) == "user"
    assert install_kind_of("/opt/homebrew/bin/codex") == "unknown"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@pytest.fixture
def client(engine, store, monkeypatch):
    monkeypatch.setattr(settings_routes, "_setup_engine", engine)
    monkeypatch.setattr(settings_routes, "_sub_agent_config_store", store)

    async def _no_live(binary=None):
        return None
    # The listing augments codex/claude entries with a live model probe
    # against ``binary_path``; the fake CLI has no app-server.
    monkeypatch.setattr(settings_routes, "_codex_live_models", _no_live)
    monkeypatch.setattr(settings_routes, "_claude_live_models", _no_live)
    app = FastAPI()
    app.include_router(settings_routes.router)
    return TestClient(app)


class TestRoutes:
    def test_prompt(self, client):
        r = client.get(f"/api/v2/settings/sub-agents/{SLUG}/locate-prompt")
        assert r.status_code == 200
        assert r.json()["prompt"].startswith(
            "Orbital (a desktop app) could not find Codex on this machine.")
        assert client.get("/api/v2/settings/sub-agents/nope/locate-prompt").status_code == 404

    @_runs_fake_cli
    @patch("agent_os.agents.setup_engine.shutil.which", return_value=None)
    def test_test_then_save_then_restart_then_clear(self, _which, client, engine, store, tmp_path):
        binary, node_dir = _node_shim_cli(tmp_path)
        answer = (f"agent: Codex\ncommand: {binary}\nversion: 0.157.1\n"
                  f"env: {{}}\npath:\n  - {node_dir}\n")
        with patch.dict(os.environ, {"PATH": "/usr/bin:/bin"}):
            r = client.post(f"/api/v2/settings/sub-agents/{SLUG}/locate/test",
                            json={"answer": answer})
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["ok"] is True and body["binary"] == binary
            assert body["path_prepend"] == [node_dir] and body["env_names"] == []
            assert "0.157.1" in body["version_output"]

            wrong = client.post(f"/api/v2/settings/sub-agents/{SLUG}/locate/test",
                                json={"answer": str(tmp_path / "missing")})
            assert wrong.status_code == 200 and wrong.json()["ok"] is False
            assert "missing" in wrong.json()["error"]

            unparsable = client.post(f"/api/v2/settings/sub-agents/{SLUG}/locate/test",
                                     json={"answer": "command: codex"})
            assert unparsable.status_code == 200 and unparsable.json()["ok"] is False
            assert "absolute" in unparsable.json()["error"]

            saved = client.put(f"/api/v2/settings/sub-agents/{SLUG}/locate",
                               json={"answer": answer})
            assert saved.status_code == 200, saved.text
            entry = saved.json()
            assert entry["installed"] is True
            assert entry["binary_path"] == binary
            assert entry["override_path"] == binary
            assert entry["override_missing"] is False
            assert store.get_override(SLUG)["path_prepend"] == [node_dir]

            # A fresh engine over the same file (daemon restart) still honours it.
            reloaded = SubAgentConfigStore(store.path)
            reg = AgentRegistry()
            reg.register(_manifest())
            fresh = SetupEngine(reg, sub_agent_config_store=reloaded)
            assert fresh.check_agent(SLUG).binary_path == binary

            cleared = client.delete(f"/api/v2/settings/sub-agents/{SLUG}/locate")
            assert cleared.status_code == 200
            assert cleared.json()["installed"] is False
            assert cleared.json()["override_path"] is None
            assert store.get_override(SLUG) is None

    @_runs_fake_cli
    @patch("agent_os.agents.setup_engine.shutil.which", return_value=None)
    def test_put_refuses_a_failing_test(self, _which, client, store, tmp_path):
        binary, _node_dir = _node_shim_cli(tmp_path)
        with patch.dict(os.environ, {"PATH": "/usr/bin:/bin"}):
            r = client.put(f"/api/v2/settings/sub-agents/{SLUG}/locate",
                           json={"answer": binary})
        assert r.status_code == 400
        assert "node" in r.json()["detail"]
        assert store.get_override(SLUG) is None

    @patch("agent_os.agents.setup_engine.shutil.which", return_value=None)
    def test_gone_state_in_the_listing(self, _which, client, engine, store, tmp_path):
        binary = _fake_cli(tmp_path)
        store.set_override(SLUG, binary=binary)
        os.remove(binary)
        entry = client.get("/api/v2/settings/sub-agents").json()
        codex = next(e for e in entry if e["slug"] == SLUG)
        assert codex["installed"] is False
        assert codex["override_missing"] is True and codex["override_path"] == binary
