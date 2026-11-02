# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 115 D3 — every npm-distributed CLI lists the version-manager install
locations under ``auto_detect``, as a second line of defence behind the PATH
capture. The guard that keeps D3 complete when the next npm agent is added.
"""

import glob
import os

import pytest

from agent_os.agents.manifest import ManifestLoader

_MANIFESTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "agent_os", "agents", "manifests"))


def _npm_manifests():
    out = []
    for path in sorted(glob.glob(os.path.join(_MANIFESTS_DIR, "*.yaml"))):
        m = ManifestLoader.load(path)
        if (m.setup.install_command or "").startswith("npm"):
            out.append(m)
    return out


NPM = _npm_manifests()


def test_the_npm_set_is_not_vacuous():
    assert {m.slug for m in NPM} >= {"codex", "claude-code", "kimi-code",
                                     "minimax-code", "pi", "gemini-cli"}


@pytest.mark.parametrize("manifest", NPM, ids=lambda m: m.slug)
@pytest.mark.parametrize("os_name", ["macos", "linux"])
def test_npm_cli_lists_version_manager_locations(manifest, os_name):
    cmd = manifest.runtime.command
    entries = manifest.setup.auto_detect.get(os_name, [])
    for expected in (
        f"$HOME/.nvm/versions/node/*/bin/{cmd}",
        f"$HOME/.volta/bin/{cmd}",
        f"$HOME/.bun/bin/{cmd}",
        f"$HOME/.local/share/fnm/node-versions/*/installation/bin/{cmd}",
        f"$HOME/.local/bin/{cmd}",
    ):
        assert expected in entries, f"{manifest.slug}: auto_detect.{os_name} lacks {expected}"


@pytest.mark.parametrize("manifest", NPM, ids=lambda m: m.slug)
def test_npm_cli_lists_macos_only_locations(manifest):
    cmd = manifest.runtime.command
    entries = manifest.setup.auto_detect.get("macos", [])
    assert f"/opt/homebrew/opt/node@*/bin/{cmd}" in entries
    assert f"$HOME/Library/pnpm/{cmd}" in entries


def test_claude_code_windows_lists_the_native_installer_location():
    m = ManifestLoader.load(os.path.join(_MANIFESTS_DIR, "claude_code.yaml"))
    assert r"%USERPROFILE%\.local\bin\claude.exe" in m.setup.auto_detect["windows"]


def test_codex_windows_lists_the_desktop_apps_hash_versioned_cli():
    """A field report: %LOCALAPPDATA%\\OpenAI\\Codex\\bin\\<hash>\\codex.exe, with
    <hash> different after an app update, so no fixed path can name it."""
    codex = next(m for m in _npm_manifests() if m.slug == "codex")
    assert ("%LOCALAPPDATA%\\OpenAI\\Codex\\bin\\*\\codex.exe"
            in codex.setup.auto_detect["windows"])


# ---------------------------------------------------------------------------
# Spec 144 "newest wins": glob matches are ordered by modification time.
# Hash folder names (bin/<hash>/) say nothing about age.
# ---------------------------------------------------------------------------

def _hash_build(root, name, mtime):
    folder = root / "bin" / name
    folder.mkdir(parents=True)
    exe = folder / "codex.exe"
    exe.write_text("")
    os.utime(exe, (mtime, mtime))
    return str(exe)


def test_glob_newest_modification_time_wins_over_name(tmp_path):
    from agent_os.agents.setup_engine import _expand_auto_detect_entry

    # "0aaa" sorts below "ffff" by name, but it is the copy the app just wrote.
    newer = _hash_build(tmp_path, "0aaa1111", 2_000_000_000)
    older = _hash_build(tmp_path, "ffff2222", 1_900_000_000)
    pattern = str(tmp_path / "bin" / "*" / "codex.exe")
    assert _expand_auto_detect_entry(pattern) == [newer, older]


def test_glob_equal_modification_times_keep_the_name_order(tmp_path):
    """A tie keeps the old natural order, so nvm's highest version still
    wins when the folders share a timestamp."""
    from agent_os.agents.setup_engine import _expand_auto_detect_entry

    same = 1_950_000_000
    v9 = _hash_build(tmp_path, "v9.0.0", same)
    v22 = _hash_build(tmp_path, "v22.1.0", same)
    v20 = _hash_build(tmp_path, "v20.3.0", same)
    pattern = str(tmp_path / "bin" / "*" / "codex.exe")
    assert _expand_auto_detect_entry(pattern) == [v22, v20, v9]
    assert _expand_auto_detect_entry(pattern) == [v22, v20, v9]  # stable
