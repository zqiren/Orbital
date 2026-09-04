# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 077 W2 — a sandbox worker gets a home it actually owns.

Before this, a sandboxed Windows command ran with ``TEMP``/``TMP`` pointed at
the Windows public temp directory while ``APPDATA`` and ``LOCALAPPDATA`` were
*inherited from the daemon* — i.e. they pointed into the main user's profile,
which is owner-only by Windows default. Every tool that writes a cache there
(npm, pip, uv, pnpm) failed on write, and no read grant could have fixed it:
the worker is a different account.

The fix points the whole home at ``%ProgramData%\\Orbital\\worker``, a tree the
worker owns. ``PATH`` is deliberately left inherited — it is what makes the W1
toolchain grants discoverable.

These run on every platform. The path helpers are built with ``ntpath`` so a
macOS or Linux host computes real Windows paths, and the launcher import stubs
the Win32 layer, so the ACL/env contract is pinned by CI on ubuntu, macos-14
and windows-latest alike.
"""

from __future__ import annotations

import ctypes
from unittest.mock import MagicMock

import pytest

from agent_os.platform.types import (
    windows_worker_env,
    windows_worker_home,
)

ENV = {"ProgramData": r"C:\ProgramData", "APPDATA": r"C:\Users\real\AppData\Roaming"}


# ---------------------------------------------------------------------------
# The pure helpers
# ---------------------------------------------------------------------------


def test_worker_home_is_under_program_data():
    # ProgramData is readable by every account and writable only where
    # granted — that is what makes it the right root, rather than a second
    # folder inside the user's own profile.
    assert windows_worker_home(ENV) == r"C:\ProgramData\Orbital\worker"


def test_worker_home_falls_back_when_program_data_is_unset():
    home = windows_worker_home({})
    assert home.endswith(r"\Orbital\worker"), home
    assert home[1:3] == r":\\"[0:2] or home.startswith("C:"), home


def test_worker_env_redirects_every_cache_variable():
    env = windows_worker_env(environ=ENV)
    root = r"C:\ProgramData\Orbital\worker"
    assert env["USERPROFILE"] == root
    assert env["HOMEDRIVE"] == "C:"
    assert env["HOMEPATH"] == r"\ProgramData\Orbital\worker"
    # The two that were previously inherited from the daemon — the actual bug.
    assert env["APPDATA"] == root + r"\AppData\Roaming"
    assert env["LOCALAPPDATA"] == root + r"\AppData\Local"
    assert env["TEMP"] == root + r"\Temp"
    assert env["TMP"] == root + r"\Temp"


def test_worker_env_never_points_into_the_users_profile():
    # The regression this spec exists to prevent: no value may fall inside
    # C:\Users, which is owner-only and unreadable to the worker account.
    for key, value in windows_worker_env(environ=ENV).items():
        assert r"\Users" not in value, f"{key} leaks into the user profile: {value}"


def test_worker_env_does_not_override_path():
    # PATH stays inherited: it is what makes the W1 toolchain roots reachable.
    assert "PATH" not in windows_worker_env(environ=ENV)


# ---------------------------------------------------------------------------
# The launcher's env block
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def launcher_cls():
    """``ProcessLauncher`` with the Win32 layer stubbed.

    ``process.py`` resolves ``ctypes.windll`` at import time, so the module is
    unimportable off Windows. Stubbing lets the env-block contract — pure
    string assembly — be pinned on every runner instead of only the Windows
    one, where nothing else in this suite would catch a regression.
    """
    for name in ("windll", "GetLastError", "FormatError",
                 "get_last_error", "set_last_error"):
        if not hasattr(ctypes, name):
            setattr(ctypes, name, MagicMock())
    if not hasattr(ctypes, "WinError"):
        ctypes.WinError = OSError
    from agent_os.platform.windows.process import ProcessLauncher

    return ProcessLauncher


def _decode(buffer) -> dict[str, str]:
    """Turn the NUL-separated block back into a dict."""
    raw = buffer[:]
    if isinstance(raw, list):
        raw = "".join(raw)
    out = {}
    for part in raw.split("\0"):
        if "=" in part:
            k, _, v = part.partition("=")
            out[k] = v
    return out


def test_env_block_targets_the_worker_home(launcher_cls, monkeypatch):
    monkeypatch.setenv("ProgramData", r"C:\ProgramData")
    env = _decode(launcher_cls._build_env_block(None, inherit_env=True))
    root = r"C:\ProgramData\Orbital\worker"
    assert env["USERPROFILE"] == root
    assert env["APPDATA"] == root + r"\AppData\Roaming"
    assert env["LOCALAPPDATA"] == root + r"\AppData\Local"
    assert env["TEMP"] == env["TMP"] == root + r"\Temp"


def test_env_block_no_longer_uses_the_old_temp_scheme(launcher_cls, monkeypatch):
    monkeypatch.setenv("ProgramData", r"C:\ProgramData")
    monkeypatch.setenv("SystemRoot", r"C:\Windows")
    env = _decode(launcher_cls._build_env_block(None, inherit_env=True))
    assert env["TEMP"] != r"C:\Windows\Temp"
    assert env["USERPROFILE"] != r"C:\Temp"
    assert env["HOMEPATH"] != r"\Temp"


def test_explicit_env_vars_still_win(launcher_cls, monkeypatch):
    # An explicit override from the caller must not be clobbered by the
    # worker-home defaults — the defaults fill gaps, they do not dictate.
    monkeypatch.setenv("ProgramData", r"C:\ProgramData")
    env = _decode(
        launcher_cls._build_env_block({"TEMP": r"D:\scratch"}, inherit_env=True)
    )
    assert env["TEMP"] == r"D:\scratch"
    # …while the ones the caller did not set are still redirected.
    assert env["APPDATA"] == r"C:\ProgramData\Orbital\worker\AppData\Roaming"


def test_no_inherit_env_gets_no_worker_home(launcher_cls, monkeypatch):
    # inherit_env=False means the caller supplies the whole environment.
    monkeypatch.setenv("ProgramData", r"C:\ProgramData")
    env = _decode(launcher_cls._build_env_block({"FOO": "bar"}, inherit_env=False))
    assert env == {"FOO": "bar"}
