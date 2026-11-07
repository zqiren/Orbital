# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 077 W1/W3 — toolchain grants and control-file deny ACEs.

Two halves of the same invariant, expressed on Windows with ACLs rather than
a Seatbelt profile:

* **W1** the worker gets read+execute on the per-user *toolchain* roots that
  exist — not on the user's profile as a whole, which holds browsers, mail and
  vaults. Anything missed is one click in Settings > Folder access, which
  writes the same ACE.
* **W3** the workspace's own ``.git\\hooks`` and ``.git\\config`` are
  deny-write, because those run code outside the sandbox. Deny ACEs precede
  allows in Windows evaluation, so the read-write grant on the enclosing
  workspace cannot override them.

``icacls`` is mocked throughout: these assert the *command shapes* and the
decision logic, which is what a machine without a sandbox account can verify.
The live behaviour is the manual Windows smoke in the spec's test plan.
"""

from __future__ import annotations

import ntpath
import os
import subprocess
from unittest.mock import patch

import pytest

from agent_os.platform.types import (
    windows_protected_control_files,
    windows_toolchain_roots,
    windows_worker_home,
)
from agent_os.platform.windows.permissions import PermissionManager

USER = "AgentOS-Worker"


def _ok(stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["icacls"], returncode=0, stdout=stdout, stderr="")


def _fail(stderr: str = "Access is denied.") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["icacls"], returncode=5, stdout="", stderr=stderr)


@pytest.fixture
def pm():
    return PermissionManager()


# ---------------------------------------------------------------------------
# W3 — deny ACEs
# ---------------------------------------------------------------------------


def test_deny_reports_an_icacls_failure(pm, tmp_path):
    d = tmp_path / ".git" / "hooks"
    d.mkdir(parents=True)
    with patch.object(PermissionManager, "_run_icacls", return_value=_fail()):
        result = pm.deny_access(USER, str(d))
    assert not result.success
    assert "Access is denied" in (result.error or "")


def test_deny_on_a_missing_path_is_refused(pm, tmp_path):
    # Windows ACLs need a real path — there is no pattern for a file that does
    # not exist yet. This is exactly why the deny is re-applied after commands
    # that could have created a repository.
    result = pm.deny_access(USER, str(tmp_path / "nope" / ".git" / "config"))
    assert not result.success
    assert result.error == "Path does not exist"


def test_protect_control_files_skips_a_workspace_with_no_repository(pm, tmp_path):
    with patch.object(PermissionManager, "deny_access") as deny:
        assert pm.protect_control_files(USER, str(tmp_path)) == []
    deny.assert_not_called()


def test_protected_control_files_are_the_git_entries():
    # ntpath.basename, not os.path: the helper emits Windows paths on every host.
    # commondir is listed so it is removed when planted (never ACL-denied).
    names = [ntpath.basename(p) for p in windows_protected_control_files(r"C:\\ws")]
    assert sorted(names) == ["commondir", "config", "hooks"]
    # Shell rc files and .gitconfig live in the user's profile on Windows,
    # which the worker never gets access to in the first place.
    assert not any(n.startswith(".bashrc") or n == ".gitconfig" for n in names)


# ---------------------------------------------------------------------------
# W1 — toolchain grants
# ---------------------------------------------------------------------------


def test_toolchain_grant_is_read_execute_and_skips_absent_roots(pm, tmp_path, monkeypatch):
    """Spec 109 D1 (T1): the grant is inheritable and NOT ``/T``.

    ``(OI)(CI)`` makes the ACE inheritable, and Windows propagates an
    inheritable ACE to the existing subtree when the root DACL is set.
    ``/T`` on top of that wrote an explicit ACE on every descendant, one file
    at a time — the minutes-long, CPU-pinned "Configuring agent sandbox"
    install stall on a dev machine (issue #55).

    Confirmed on Windows by spec 109 V1 (2026-10-10): after the ``/T``-less
    grant on ``%LOCALAPPDATA%\\Programs`` (104,018 entries, 23.5 s vs 182.3 s
    with ``/T``) a deep file showed ``AgentOS-Worker:(I)(RX)``.
    """
    present = tmp_path / "npm"
    present.mkdir()
    monkeypatch.setattr(
        "agent_os.platform.windows.permissions.windows_toolchain_roots",
        lambda: [str(present), str(tmp_path / "not-installed")],
    )
    with patch.object(PermissionManager, "_run_icacls", return_value=_ok()) as run:
        results = pm.grant_toolchain_roots(USER)
    assert len(results) == 1 and results[0].success
    args = run.call_args[0][0]
    assert f"{USER}:(OI)(CI)RX" in args, "read+execute, never write"
    assert "/grant" in args and "/Q" in args
    assert "/T" not in args, "per-file walk is the install stall (spec 109 D1)"


def test_revoke_has_no_recursive_flag(pm, tmp_path):
    """Spec 109 D1 (T6): uninstall hung the same way in reverse (#55).

    Removing the inheritable ACE from the root removes the inherited copies
    through the same propagation; same V1/V5 caveat as the grant above.
    """
    with patch.object(PermissionManager, "_run_icacls", return_value=_ok()) as run:
        assert pm.revoke_access(USER, str(tmp_path)).success
    args = run.call_args[0][0]
    assert "/remove" in args and USER in args and "/Q" in args
    assert "/T" not in args


# --- grant_access: no per-file walk, and nothing to do on a repeat ---------
#
# Feedback row 4 (2026-11-07, Windows 11, v0.20.0): ``grant_access`` ran
# ``icacls <workspace> /grant ... /T`` on EVERY session start — the scan, the
# onboarding start, each new session — and on the daemon's event loop. On an
# imported D:\ project tree with nested repositories that froze the whole app
# for 1016 s. Spec 109 kept ``/T`` here on the belief the workspace is empty
# at setup; an imported project (spec 160) is not.

WORKER_SID = "S-1-5-21-1-2-3-1001"
_INHERIT = 0x01 | 0x02  # OBJECT_INHERIT_ACE | CONTAINER_INHERIT_ACE


def _allow(mask, flags=_INHERIT, sid=WORKER_SID):
    from agent_os.platform.windows.acl import ACCESS_ALLOWED_ACE_TYPE, Ace
    return Ace(type=ACCESS_ALLOWED_ACE_TYPE, flags=flags, mask=mask, sid=sid)


@pytest.fixture
def worker_sid(monkeypatch):
    monkeypatch.setattr(
        "agent_os.platform.windows.permissions.acl.account_sid",
        lambda name: WORKER_SID if name == USER else None,
    )


def _dacl(monkeypatch, aces):
    monkeypatch.setattr(
        "agent_os.platform.windows.permissions.acl.read_dacl", lambda path: aces)


@pytest.mark.parametrize("mode,perm", [("read_only", f"{USER}:(OI)(CI)R"),
                                       ("read_write", f"{USER}:(OI)(CI)F")])
def test_folder_grant_is_inheritable_and_not_recursive(pm, tmp_path, mode, perm,
                                                       worker_sid, monkeypatch):
    """Same rule as the toolchain roots (spec 109 D1, verified V1): an
    inheritable ACE on the root reaches every inheriting descendant, so the
    per-file ``/T`` walk only costs time — 8x on the 10^5-file V1 tree."""
    _dacl(monkeypatch, [])
    with patch.object(PermissionManager, "_run_icacls", return_value=_ok()) as run, \
            patch.object(PermissionManager, "_root_hardening_present", return_value=True):
        assert pm.grant_access(USER, str(tmp_path), mode).success
    args = run.call_args[0][0]
    assert perm in args and "/grant" in args
    assert "/T" not in args, "the per-file walk froze the daemon (feedback row 4)"


@pytest.mark.parametrize("mode,mask", [("read_only", 0x120089),
                                       ("read_write", 0x1F01FF),
                                       ("read_only", 0x1F01FF)])
def test_folder_grant_skips_a_root_that_already_has_the_entry(pm, tmp_path, mode, mask,
                                                              worker_sid, monkeypatch):
    """Every ``/grant`` re-propagates through the whole subtree even when the
    ACE is already there (spec 109 V1). Session starts re-grant the workspace
    every time, so a root that already carries an explicit inheritable allow
    covering the mode costs one DACL read and no icacls write."""
    _dacl(monkeypatch, [_allow(mask)])
    with patch.object(PermissionManager, "_run_icacls", return_value=_ok()) as run, \
            patch.object(PermissionManager, "_root_hardening_present", return_value=True):
        result = pm.grant_access(USER, str(tmp_path), mode)
    assert result.success and result.path == os.path.realpath(str(tmp_path))
    run.assert_not_called()


@pytest.mark.parametrize("aces", [
    None,                                         # DACL unreadable
    [],                                           # nothing for the worker
    [_allow(0x1F01FF, flags=_INHERIT | 0x10)],    # inherited copy only
    [_allow(0x1F01FF, flags=0)],                  # this folder only
    [_allow(0x120089)],                           # read, but write requested
    [_allow(0x1F01FF, sid="S-1-5-21-9-9-9-500")],  # someone else's entry
])
def test_folder_grant_still_writes_without_a_covering_entry(pm, tmp_path, aces,
                                                            worker_sid, monkeypatch):
    _dacl(monkeypatch, aces)
    with patch.object(PermissionManager, "_run_icacls", return_value=_ok()) as run, \
            patch.object(PermissionManager, "_root_hardening_present", return_value=True):
        assert pm.grant_access(USER, str(tmp_path), "read_write").success
    assert "/grant" in run.call_args[0][0]


def test_read_write_fast_path_still_hardens_the_root(pm, tmp_path, worker_sid, monkeypatch):
    """The skip covers the worker entry only; a root missing the user /
    OWNER RIGHTS entries still gets them (spec 077 W3, bench 2026-10-19)."""
    _dacl(monkeypatch, [_allow(0x1F01FF)])
    with patch.object(PermissionManager, "_run_icacls", return_value=_ok()) as run, \
            patch.object(PermissionManager, "_root_hardening_present", return_value=False):
        assert pm.grant_access(USER, str(tmp_path), "read_write").success
    assert run.call_count == 1 and "*S-1-3-4:(OI)(CI)M" in run.call_args[0][0]


def test_icacls_timeout_returns_failure_not_exception(pm, tmp_path, monkeypatch):
    """Spec 109 D7 (T7): a bounded manager turns ``TimeoutExpired`` into a
    failed PermissionResult so the installer step logs and continues."""
    root = tmp_path / "Programs"
    root.mkdir()
    monkeypatch.setattr(
        "agent_os.platform.windows.permissions.windows_toolchain_roots",
        lambda: [str(root)],
    )
    pm.icacls_timeout = 0.5
    seen: dict = {}

    def slow_run(cmd, **kwargs):
        seen.update(kwargs)
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))

    with patch("agent_os.platform.windows.permissions.subprocess.run", side_effect=slow_run):
        results = pm.grant_toolchain_roots(USER)
    assert seen["timeout"] == 0.5, "the bound reaches subprocess.run"
    assert len(results) == 1 and not results[0].success
    assert "timed out" in (results[0].error or "")


def test_icacls_has_no_timeout_by_default(pm, tmp_path):
    """Default ``None``: unchanged behaviour for the daemon and the
    Folder-access UI, which may legitimately grant a huge folder."""
    seen: dict = {}

    def run(cmd, **kwargs):
        seen.update(kwargs)
        return _ok()

    with patch("agent_os.platform.windows.permissions.subprocess.run", side_effect=run):
        pm.grant_access(USER, str(tmp_path), "read_only")
    assert seen["timeout"] is None


def test_explicit_timeout_argument_overrides_the_manager_bound(pm, tmp_path):
    seen: dict = {}

    def run(cmd, **kwargs):
        seen.update(kwargs)
        return _ok()

    pm.icacls_timeout = 120
    with patch("agent_os.platform.windows.permissions.subprocess.run", side_effect=run):
        pm._run_icacls([str(tmp_path)], timeout=3)
    assert seen["timeout"] == 3


def test_one_failing_root_does_not_sink_the_others(pm, tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    monkeypatch.setattr(
        "agent_os.platform.windows.permissions.windows_toolchain_roots",
        lambda: [str(a), str(b)],
    )
    # Per root: one query (no entry yet), then the grant.
    with patch.object(PermissionManager, "_run_icacls",
                      side_effect=[_ok(), _fail(), _ok(), _ok()]):
        results = pm.grant_toolchain_roots(USER)
    assert [r.success for r in results] == [False, True]


_ROOT_WITH_GRANT = (
    r"C:\Users\dev\AppData\Local\Programs DESKTOP-ABC\AgentOS-Worker:(OI)(CI)(RX)" "\n"
    r"                                    NT AUTHORITY\SYSTEM:(I)(OI)(CI)(F)" "\n"
)
_ROOT_INHERITED_ONLY = r"C:\Users\dev\.cargo DESKTOP-ABC\AgentOS-Worker:(I)(OI)(CI)(RX)" "\n"
_ROOT_DENY_ONLY = r"C:\Users\dev\.cargo DESKTOP-ABC\AgentOS-Worker:(DENY)(OI)(CI)(W)" "\n"


def test_toolchain_grant_skips_a_root_that_already_has_the_entry(pm, tmp_path, monkeypatch):
    """Spec 109 V1 finding: every ``/grant`` re-walks the subtree to
    propagate, even when the identical ACE is already there (19 s per call on
    a 10^5-entry root, measured). The daemon re-runs this at every start, so
    a root that already carries the entry costs one single-object query."""
    root = tmp_path / "Programs"
    root.mkdir()
    monkeypatch.setattr(
        "agent_os.platform.windows.permissions.windows_toolchain_roots",
        lambda: [str(root)],
    )
    with patch.object(PermissionManager, "_run_icacls",
                      return_value=_ok(_ROOT_WITH_GRANT)) as run:
        results = pm.grant_toolchain_roots(USER)
    assert len(results) == 1 and results[0].success
    assert run.call_count == 1, "query only, no /grant"
    assert "/grant" not in run.call_args[0][0]


@pytest.mark.parametrize("existing", [_ROOT_INHERITED_ONLY, _ROOT_DENY_ONLY, ""])
def test_toolchain_grant_still_grants_without_an_explicit_allow(pm, tmp_path, monkeypatch, existing):
    root = tmp_path / "cargo"
    root.mkdir()
    monkeypatch.setattr(
        "agent_os.platform.windows.permissions.windows_toolchain_roots",
        lambda: [str(root)],
    )
    with patch.object(PermissionManager, "_run_icacls",
                      side_effect=[_ok(existing), _ok()]) as run:
        results = pm.grant_toolchain_roots(USER)
    assert len(results) == 1 and results[0].success
    assert "/grant" in run.call_args[0][0]


def test_toolchain_grant_still_grants_when_the_query_fails(pm, tmp_path, monkeypatch):
    root = tmp_path / "npm"
    root.mkdir()
    monkeypatch.setattr(
        "agent_os.platform.windows.permissions.windows_toolchain_roots",
        lambda: [str(root)],
    )
    with patch.object(PermissionManager, "_run_icacls", side_effect=[_fail(), _ok()]) as run:
        results = pm.grant_toolchain_roots(USER)
    assert results[0].success and "/grant" in run.call_args[0][0]


def test_toolchain_roots_never_include_a_credential_store():
    # The secrets rule holds on Windows by construction: the profile stays
    # closed, so nothing needs a deny list. Guard the curated list anyway.
    roots = " ".join(windows_toolchain_roots({
        "APPDATA": r"C:\U\AppData\Roaming",
        "LOCALAPPDATA": r"C:\U\AppData\Local",
        "USERPROFILE": r"C:\U",
    })).lower()
    for secret in (".ssh", ".aws", ".gnupg", ".kube", ".netrc",
                   "git-credentials", "orbital"):
        assert secret not in roots, f"{secret} must never be granted"


def test_toolchain_roots_drop_entries_whose_variables_are_unset():
    assert windows_toolchain_roots({}) == []


# ---------------------------------------------------------------------------
# W2 — the worker home
# ---------------------------------------------------------------------------


def test_setup_worker_home_creates_the_tree_and_grants_it(pm, tmp_path, monkeypatch):
    home = tmp_path / "ProgramData" / "Orbital" / "worker"
    monkeypatch.setattr(
        "agent_os.platform.windows.permissions.windows_worker_home", lambda: str(home)
    )
    with patch.object(PermissionManager, "grant_access", return_value=_ok()) as grant:
        pm.setup_worker_home(USER)
    # The subdirectories the env block names must exist, or the redirect
    # points at nothing.
    assert (home / "Temp").is_dir()
    assert (home / "AppData" / "Roaming").is_dir()
    assert (home / "AppData" / "Local").is_dir()
    assert grant.call_args[0][-1] == "read_write"


def test_worker_home_is_not_inside_the_user_profile():
    home = windows_worker_home({"ProgramData": r"C:\ProgramData"})
    assert r"\Users" not in home


# ---------------------------------------------------------------------------
# Repository discovery for the post-command re-check
# ---------------------------------------------------------------------------


def test_find_repository_roots_finds_nested_repos_and_skips_dependencies(pm, tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "sub" / ".git").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / ".git").mkdir(parents=True)
    found = {os.path.relpath(p, tmp_path) for p in pm.find_repository_roots(str(tmp_path))}
    assert "." in found and "sub" in found
    assert not any("node_modules" in f for f in found), "dependency trees are skipped"


def test_find_repository_roots_is_depth_bounded(pm, tmp_path):
    deep = tmp_path / "a" / "b" / "c" / "d" / "e"
    (deep / ".git").mkdir(parents=True)
    assert pm.find_repository_roots(str(tmp_path), max_depth=2) == []


def test_find_repository_roots_tolerates_a_missing_root(pm, tmp_path):
    assert pm.find_repository_roots(str(tmp_path / "gone")) == []


def test_toolchain_revoke_skips_a_root_without_an_explicit_entry(pm, tmp_path, monkeypatch):
    """Spec 109 V1 follow-up: ``/remove`` walks the subtree even when the
    account has no entry (34.8 s on an npm root, measured), inside the
    uninstaller's frozen-bar step. No explicit entry -> query only."""
    root = tmp_path / "npm"
    root.mkdir()
    monkeypatch.setattr(
        "agent_os.platform.windows.permissions.windows_toolchain_roots",
        lambda: [str(root)],
    )
    with patch.object(PermissionManager, "_run_icacls",
                      return_value=_ok(_ROOT_INHERITED_ONLY)) as run:
        results = pm.revoke_toolchain_roots(USER)
    assert len(results) == 1 and results[0].success
    assert run.call_count == 1 and "/remove" not in run.call_args[0][0]


@pytest.mark.parametrize("existing", [_ROOT_WITH_GRANT, _ROOT_DENY_ONLY])
def test_toolchain_revoke_removes_any_explicit_entry(pm, tmp_path, monkeypatch, existing):
    root = tmp_path / "Programs"
    root.mkdir()
    monkeypatch.setattr(
        "agent_os.platform.windows.permissions.windows_toolchain_roots",
        lambda: [str(root)],
    )
    with patch.object(PermissionManager, "_run_icacls",
                      side_effect=[_ok(existing), _ok()]) as run:
        results = pm.revoke_toolchain_roots(USER)
    assert results[0].success
    args = run.call_args[0][0]
    assert "/remove" in args and "/T" not in args


def test_toolchain_revoke_still_revokes_when_the_query_fails(pm, tmp_path, monkeypatch):
    root = tmp_path / "uv"
    root.mkdir()
    monkeypatch.setattr(
        "agent_os.platform.windows.permissions.windows_toolchain_roots",
        lambda: [str(root)],
    )
    with patch.object(PermissionManager, "_run_icacls", side_effect=[_fail(), _ok()]) as run:
        assert pm.revoke_toolchain_roots(USER)[0].success
    assert "/remove" in run.call_args[0][0]


def test_toolchain_revokes_run_concurrently_and_keep_root_order(pm, tmp_path, monkeypatch):
    """Wall clock is the slowest root, not the sum: every root's /remove is
    in flight at once (a barrier no sequential loop could pass)."""
    import threading

    roots = [tmp_path / n for n in ("npm", "Programs", "pnpm")]
    for r in roots:
        r.mkdir()
    monkeypatch.setattr(
        "agent_os.platform.windows.permissions.windows_toolchain_roots",
        lambda: [str(r) for r in roots],
    )
    barrier = threading.Barrier(len(roots), timeout=5)

    def fake(args, timeout=None):
        if "/remove" not in args:
            return _ok(f"{args[0]} DESKTOP-ABC\\{USER}:(OI)(CI)(RX)\n")
        barrier.wait()  # raises BrokenBarrierError if the removes are serial
        return _ok()

    with patch.object(PermissionManager, "_run_icacls", side_effect=fake):
        results = pm.revoke_toolchain_roots(USER)
    assert [r.success for r in results] == [True, True, True]
    assert [os.path.basename(r.path) for r in results] == ["npm", "Programs", "pnpm"]


def test_toolchain_revoke_with_no_roots_is_a_no_op(pm, monkeypatch):
    monkeypatch.setattr(
        "agent_os.platform.windows.permissions.windows_toolchain_roots", lambda: [],
    )
    assert pm.revoke_toolchain_roots(USER) == []
