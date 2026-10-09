# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 109 — the Windows install path: no toolchain walk, bounded icacls,
a described and sign-in-hidden AgentOS-Worker account.

Issue #55: the installer's "Configuring agent sandbox" step pinned a CPU for
minutes (``icacls /T`` over every per-user toolchain root), uninstall hung the
same way in reverse, and the AgentOS-Worker account showed up on the sign-in
screen with no description and no prior disclosure.

Everything Win32 is stubbed: ``agent_os.platform.windows.sandbox`` resolves
``ctypes.windll`` at import time, so the module is unimportable off Windows
without the stub below (same pattern as ``test_windows_env_block``). These
tests pin the *decisions* — which calls are made with which arguments — and
never touch a real account, ACL or registry. The live behaviour is the
Windows verification checklist in the spec (§5).
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

_WIN32_NAMES = ("windll", "GetLastError", "FormatError", "get_last_error", "set_last_error")


def _install_win32_stubs() -> list[str]:
    """Stub the ctypes Win32 surface off Windows; return the names added."""
    added = []
    for name in _WIN32_NAMES:
        if not hasattr(ctypes, name):
            setattr(ctypes, name, MagicMock())
            added.append(name)
    if not hasattr(ctypes, "WinError"):
        ctypes.WinError = OSError
        added.append("WinError")
    return added


def _remove_win32_stubs(added: list[str]) -> None:
    for name in added:
        if isinstance(getattr(ctypes, name, None), MagicMock) or name == "WinError":
            delattr(ctypes, name)


# The stubs are needed for the imports below, then removed again: a fake
# ``ctypes.windll`` left on a macOS runner from collection time makes any
# Windows-only code path that probes for it run for real (the desktop icon
# thread did, and segfaulted the pytest process). The autouse fixture
# re-installs them for exactly this module's tests.
_IMPORT_STUBS = _install_win32_stubs()

from agent_os.platform.types import PermissionResult, SANDBOX_USERNAME  # noqa: E402
from agent_os.platform.windows import sandbox as sandbox_mod  # noqa: E402
from agent_os.platform.windows.permissions import PermissionManager  # noqa: E402
from agent_os.platform.windows.sandbox import SandboxAccountManager  # noqa: E402
from agent_os.platform.windows.setup import (  # noqa: E402
    INSTALLER_ICACLS_TIMEOUT,
    SetupOrchestrator,
)



@pytest.fixture(autouse=True, scope="module")
def _win32_stubs_for_this_module():
    added = _install_win32_stubs()
    try:
        yield
    finally:
        _remove_win32_stubs(added)

USER = SANDBOX_USERNAME


def _ok(path: str = "x") -> PermissionResult:
    return PermissionResult(success=True, path=path)


# ---------------------------------------------------------------------------
# D2 — run_setup no longer walks the toolchain roots (T3)
# ---------------------------------------------------------------------------


@pytest.fixture
def orchestrator(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    account = MagicMock()
    account.get_username.return_value = USER
    account.ensure_account_exists.return_value = MagicMock(exists=True, error=None)
    perms = MagicMock()
    perms.icacls_timeout = None
    perms.setup_workspace.return_value = _ok()
    perms.setup_worker_home.return_value = _ok()
    perms.protect_control_files.return_value = []
    return SetupOrchestrator(account, perms), account, perms


def test_run_setup_has_no_toolchain_step(orchestrator):
    orch, _, perms = orchestrator
    result = orch.run_setup()
    assert result.success
    perms.grant_toolchain_roots.assert_not_called()
    perms.setup_workspace.assert_called_once()
    perms.setup_worker_home.assert_called_once_with(USER)
    perms.protect_control_files.assert_called_once()


# ---------------------------------------------------------------------------
# D7 — the installer paths bound every icacls call (T7)
# ---------------------------------------------------------------------------


def test_run_setup_bounds_icacls_for_its_duration(orchestrator):
    orch, _, perms = orchestrator
    seen: list = []
    perms.setup_workspace.side_effect = lambda *a: (seen.append(perms.icacls_timeout), _ok())[1]
    orch.run_setup()
    assert seen == [INSTALLER_ICACLS_TIMEOUT]
    assert perms.icacls_timeout is None, "the bound is restored afterwards"


def test_installer_bound_is_two_minutes():
    assert INSTALLER_ICACLS_TIMEOUT == 120


def test_run_teardown_with_every_revoke_timing_out_still_deletes_the_account(tmp_path, monkeypatch):
    """The teardown rule: a timed-out revoke must never block uninstall."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    workspace = tmp_path / "AgentOS" / "workspace"
    workspace.mkdir(parents=True)
    root = tmp_path / "Programs"
    root.mkdir()
    monkeypatch.setattr(
        "agent_os.platform.windows.permissions.windows_toolchain_roots",
        lambda: [str(root)],
    )
    account = MagicMock()
    account.get_username.return_value = USER
    perms = PermissionManager()
    orch = SetupOrchestrator(account, perms)

    calls: list = []

    def always_slow(cmd, **kwargs):
        calls.append((cmd, kwargs.get("timeout")))
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))

    with patch("agent_os.platform.windows.permissions.subprocess.run", side_effect=always_slow):
        result = orch.run_teardown()

    assert result.success
    account.delete_account.assert_called_once()
    assert calls, "the revokes ran"
    assert all(t == INSTALLER_ICACLS_TIMEOUT for _, t in calls)
    assert perms.icacls_timeout is None


def test_run_teardown_still_revokes_toolchain_roots_without_recursion(tmp_path, monkeypatch):
    """Teardown keeps revoking the roots (the daemon may have granted them),
    with the root-only ``/remove`` shape of D1."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    root = tmp_path / "Programs"
    root.mkdir()
    monkeypatch.setattr(
        "agent_os.platform.windows.permissions.windows_toolchain_roots",
        lambda: [str(root)],
    )
    account = MagicMock()
    account.get_username.return_value = USER
    perms = PermissionManager()
    orch = SetupOrchestrator(account, perms)
    granted = subprocess.CompletedProcess(
        args=["icacls"], returncode=0, stderr="",
        stdout=f"{root} DESKTOP-ABC\\{USER}:(OI)(CI)(RX)\n")
    with patch.object(PermissionManager, "_run_icacls", return_value=granted) as run:
        assert orch.run_teardown().success
    shapes = [c[0][0] for c in run.call_args_list]
    assert any("/remove" in a and USER in a for a in shapes)
    assert all("/T" not in a for a in shapes)


# ---------------------------------------------------------------------------
# D3 + D4 — the account is described and hidden from the sign-in screen (T5)
# ---------------------------------------------------------------------------


class _FakeWinreg(types.ModuleType):
    """Just enough of ``winreg`` for the hide/unhide helpers."""

    HKEY_LOCAL_MACHINE = object()
    KEY_SET_VALUE = 0x0002
    KEY_WOW64_64KEY = 0x0100
    REG_DWORD = 4

    def __init__(self):
        super().__init__("winreg")
        self.CreateKeyEx = MagicMock(return_value="HANDLE")
        self.OpenKey = MagicMock(return_value="HANDLE")
        self.SetValueEx = MagicMock()
        self.DeleteValue = MagicMock()
        self.CloseKey = MagicMock()


@pytest.fixture
def winreg(monkeypatch):
    fake = _FakeWinreg()
    monkeypatch.setitem(sys.modules, "winreg", fake)
    return fake


@pytest.fixture
def netapi(monkeypatch):
    api = MagicMock()
    api.NetUserAdd.return_value = sandbox_mod.NERR_Success
    api.NetUserSetInfo.return_value = sandbox_mod.NERR_Success
    api.NetUserDel.return_value = sandbox_mod.NERR_Success
    monkeypatch.setattr(sandbox_mod, "netapi32", api)
    return api


@pytest.fixture
def manager(monkeypatch):
    store = MagicMock()
    store.retrieve.return_value = "pw"
    mgr = SandboxAccountManager(store)
    monkeypatch.setattr(SandboxAccountManager, "_is_admin", staticmethod(lambda: True))
    monkeypatch.setattr(SandboxAccountManager, "_test_logon", staticmethod(lambda u, p: True))
    monkeypatch.setattr(mgr, "_run_cmd", MagicMock(return_value=(0, "", "")))
    return mgr


def _user_list_writes(winreg) -> list:
    """(subkey, value name, value) for every SetValueEx on the UserList key."""
    out = []
    for c in winreg.CreateKeyEx.call_args_list:
        assert c[0][0] is winreg.HKEY_LOCAL_MACHINE
        assert c[0][1].endswith(r"Winlogon\SpecialAccounts\UserList")
    for c in winreg.SetValueEx.call_args_list:
        out.append((c[0][1], c[0][3], c[0][4]))
    return out


def test_create_path_sets_the_description_and_hides_the_account(manager, netapi, winreg, monkeypatch):
    monkeypatch.setattr(manager, "_account_exists", MagicMock(side_effect=[False, True]))
    status = manager.ensure_account_exists()
    assert status.exists

    info = netapi.NetUserAdd.call_args[0][2]._obj
    assert info.usri1_comment == sandbox_mod.SANDBOX_ACCOUNT_COMMENT
    assert "Orbital" in info.usri1_comment and "uninstaller" in info.usri1_comment

    assert _user_list_writes(winreg) == [(USER, winreg.REG_DWORD, 0)]
    # The key must be opened in the 64-bit view: Winlogon reads that one.
    assert winreg.CreateKeyEx.call_args[1]["access"] & winreg.KEY_WOW64_64KEY


def test_exists_path_with_admin_describes_and_hides_idempotently(manager, netapi, winreg, monkeypatch):
    monkeypatch.setattr(manager, "_account_exists", lambda: True)
    status = manager.ensure_account_exists()
    assert status.exists and status.password_valid

    levels = [c[0][2] for c in netapi.NetUserSetInfo.call_args_list]
    assert 1007 in levels, "NetUserSetInfo level 1007 = USER_INFO_1007 (comment only)"
    idx = levels.index(1007)
    info = netapi.NetUserSetInfo.call_args_list[idx][0][3]._obj
    assert info.usri1007_comment == sandbox_mod.SANDBOX_ACCOUNT_COMMENT
    assert _user_list_writes(winreg) == [(USER, winreg.REG_DWORD, 0)]


def test_exists_path_without_admin_touches_nothing(manager, netapi, winreg, monkeypatch):
    monkeypatch.setattr(SandboxAccountManager, "_is_admin", staticmethod(lambda: False))
    monkeypatch.setattr(manager, "_account_exists", lambda: True)
    status = manager.ensure_account_exists()
    assert status.exists
    netapi.NetUserSetInfo.assert_not_called()
    winreg.SetValueEx.assert_not_called()


def test_delete_account_unhides(manager, netapi, winreg, monkeypatch):
    manager.delete_account()
    netapi.NetUserDel.assert_called_once()
    assert winreg.DeleteValue.call_args[0][1] == USER


def test_registry_errors_are_logged_never_raised(manager, netapi, winreg, monkeypatch, caplog):
    winreg.SetValueEx.side_effect = PermissionError("Access is denied")
    winreg.DeleteValue.side_effect = PermissionError("Access is denied")
    monkeypatch.setattr(manager, "_account_exists", MagicMock(side_effect=[False, True]))
    with caplog.at_level("WARNING", logger="agent_os.platform.windows.sandbox"):
        status = manager.ensure_account_exists()
        manager.delete_account()
    assert status.exists, "the hide is cosmetic; setup still succeeds"
    assert "sign-in" in caplog.text.lower()


def test_missing_winreg_is_tolerated(manager, netapi, monkeypatch):
    """Source installs on a non-Windows interpreter (and this suite) have no
    ``winreg``; the helpers must degrade to a log line."""
    monkeypatch.setitem(sys.modules, "winreg", None)
    monkeypatch.setattr(manager, "_account_exists", MagicMock(side_effect=[False, True]))
    assert manager.ensure_account_exists().exists


def test_comment_failure_is_not_fatal(manager, netapi, winreg, monkeypatch):
    netapi.NetUserSetInfo.return_value = sandbox_mod.ERROR_ACCESS_DENIED
    monkeypatch.setattr(manager, "_account_exists", lambda: True)
    assert manager.ensure_account_exists().exists


# ---------------------------------------------------------------------------
# D3 + D5 — the installer script and the pre-install notice
# ---------------------------------------------------------------------------

import pathlib  # noqa: E402

_INSTALLER = pathlib.Path(__file__).resolve().parents[2] / "installer"


def test_before_install_notice_is_utf8_bom_crlf_and_bilingual():
    """Inno reads a ``.txt`` InfoBeforeFile as ANSI unless it carries a BOM;
    without one the Chinese half mojibakes (spec 109 §3.5)."""
    raw = (_INSTALLER / "before-install.txt").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), "UTF-8 BOM required"
    assert raw.count(b"\n") == raw.count(b"\r\n"), "CRLF only"
    text = raw.decode("utf-8-sig")
    assert "AgentOS-Worker" in text
    for fact in ("sign-in screen", "random", "Uninstalling Orbital", "first time Orbital starts"):
        assert fact in text, fact
    for fact in ("登录界面", "随机", "卸载 Orbital", "第一次启动"):
        assert fact in text, fact


def test_inno_script_discloses_hides_and_names_the_step():
    iss = (_INSTALLER / "agentos-setup.iss").read_text(encoding="utf-8")
    assert "InfoBeforeFile=before-install.txt" in iss
    assert "[Registry]" in iss
    userlist = r"Winlogon\SpecialAccounts\UserList"
    assert iss.count(userlist) == 2, "one HKLM64 (x64) + one HKLM (x86) entry"
    reg_block = iss.split("[Registry]", 1)[1].split("[Run]", 1)[0]
    assert "Root: HKLM64;" in reg_block and "Check: IsWin64" in reg_block
    assert "Root: HKLM;" in reg_block and "Check: not IsWin64" in reg_block
    assert reg_block.count('ValueName: "AgentOS-Worker"; ValueData: 0') == 2
    assert reg_block.count("Flags: uninsdeletevalue") == 2
    assert 'StatusMsg: "Creating the AgentOS-Worker sandbox account..."' in iss
    assert "Configuring agent sandbox" not in iss


# ---------------------------------------------------------------------------
# D2 — the daemon-start refresh itself
# ---------------------------------------------------------------------------

from agent_os.platform.windows.provider import WindowsPlatformProvider  # noqa: E402

_remove_win32_stubs(_IMPORT_STUBS)  # every module-level import is done


def _bare_provider(account_exists: bool):
    """A provider with only the two collaborators the refresh touches."""
    prov = WindowsPlatformProvider.__new__(WindowsPlatformProvider)
    prov._account_manager = MagicMock()
    prov._account_manager.get_username.return_value = USER
    prov._account_manager.validate_account.return_value = MagicMock(exists=account_exists)
    prov._permission_manager = MagicMock()
    prov._permission_manager.grant_toolchain_roots.return_value = [_ok("a"), _ok("b")]
    prov._permission_manager.find_repository_roots.return_value = []
    prov._permission_manager.protect_control_files.return_value = []
    return prov


def test_refresh_skips_when_the_account_is_absent(caplog):
    prov = _bare_provider(account_exists=False)
    with caplog.at_level("INFO", logger="agent_os.platform.windows.provider"):
        prov.refresh_sandbox_grants([r"C:\\ws"])
    prov._permission_manager.grant_toolchain_roots.assert_not_called()
    assert "absent" in caplog.text


def test_refresh_logs_root_count_and_elapsed(caplog):
    prov = _bare_provider(account_exists=True)
    with caplog.at_level("INFO", logger="agent_os.platform.windows.provider"):
        prov.refresh_sandbox_grants([r"C:\\ws"])
    prov._permission_manager.grant_toolchain_roots.assert_called_once_with(USER)
    prov._permission_manager.protect_control_files.assert_called()
    assert "2 toolchain root(s) granted, 0 failed, 1 workspace(s) checked in" in caplog.text
    assert " ms" in caplog.text
