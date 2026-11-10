# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""A closed Windows process/thread handle is never used again.

After ``CloseHandle`` the OS may hand the same number to a different object
(since c8da35d the daemon also holds a long-lived kill-on-close job handle:
a second close landing on that number would kill every worker). So once
closed, the stored value is cleared, a second ``terminate()`` is a no-op,
liveness checks answer False without calling Win32, and ``wait()`` returns
the exit code recorded at close (or None) instead of waiting on a number.

The Win32 layer is mocked: ``process.py`` resolves ``ctypes.windll`` at
import, so off Windows it is stubbed the way ``test_windows_env_block.py``
does; on Windows the real module is imported and every call is patched.
"""

from __future__ import annotations

import ctypes
from unittest.mock import MagicMock

import pytest

PROC, THREAD = 0x1A4, 0x1A8
WAIT_TIMEOUT, WAIT_OBJECT_0 = 0x102, 0x0


@pytest.fixture(scope="module")
def proc_mod():
    for name in ("windll", "GetLastError", "FormatError",
                 "get_last_error", "set_last_error"):
        if not hasattr(ctypes, name):
            setattr(ctypes, name, MagicMock())
    if not hasattr(ctypes, "WinError"):
        ctypes.WinError = OSError
    from agent_os.platform.windows import process

    return process


class Win32:
    """Records every call; a closed handle number must never be touched."""

    def __init__(self, running: bool, exit_code: int = 0):
        self.running = running
        self.exit_code = exit_code
        self.closed: list[int] = []
        self.calls: list[tuple] = []

    def _check_open(self, h, what):
        assert h not in self.closed, f"{what} on an already-closed handle {h:#x}"

    def wait(self, h, ms):
        self._check_open(h, "WaitForSingleObject")
        self.calls.append(("wait", h, ms))
        if self.running and ms == 0:
            return WAIT_TIMEOUT
        self.running = False
        return WAIT_OBJECT_0

    def close(self, h):
        self._check_open(h, "CloseHandle")
        self.closed.append(h)
        return 1

    def exit_code_of(self, h, ref):
        self._check_open(h, "GetExitCodeProcess")
        ref._obj.value = 259 if self.running else self.exit_code
        return 1

    def terminate(self, h, code):
        self._check_open(h, "TerminateProcess")
        self.running = False
        self.exit_code = code
        return 1


@pytest.fixture
def win(proc_mod, monkeypatch):
    def make(running: bool, exit_code: int = 0) -> Win32:
        w = Win32(running, exit_code)
        monkeypatch.setattr(proc_mod, "_WaitForSingleObject", w.wait)
        monkeypatch.setattr(proc_mod, "_CloseHandle", w.close)
        monkeypatch.setattr(proc_mod, "_GetExitCodeProcess", w.exit_code_of)
        monkeypatch.setattr(proc_mod, "_TerminateProcess", w.terminate)
        monkeypatch.setattr(proc_mod, "_GenerateConsoleCtrlEvent", lambda *a: 1)
        return w
    return make


def _handle(proc_mod):
    from agent_os.platform.types import ProcessHandle

    build = getattr(proc_mod, "_launched_native_handles", None)
    if build is None:  # the shape launch() built before the fix
        natives = {
            "process_handle": PROC, "thread_handle": THREAD,
            "is_alive": lambda: proc_mod._WaitForSingleObject(PROC, 0) == WAIT_TIMEOUT,
        }
    else:
        natives = build(PROC, THREAD)
    return ProcessHandle(pid=4242, command="cmd /c x", _native_handles=natives)


def _launcher(proc_mod):
    return proc_mod.ProcessLauncher(credential_store=MagicMock())


@pytest.mark.parametrize("running", [True, False])
def test_terminate_closes_each_handle_once_and_clears_it(proc_mod, win, running):
    w = win(running=running, exit_code=7)
    handle, launcher = _handle(proc_mod), _launcher(proc_mod)

    assert launcher.terminate(handle, timeout_sec=1) is True
    assert sorted(w.closed) == sorted([PROC, THREAD])
    assert handle._native_handles["process_handle"] is None
    assert handle._native_handles["thread_handle"] is None

    # A second terminate is a no-op: no Win32 call at all.
    calls_before = list(w.calls)
    assert launcher.terminate(handle, timeout_sec=1) is True
    assert sorted(w.closed) == sorted([PROC, THREAD])
    assert w.calls == calls_before


def test_liveness_after_close_never_calls_win32(proc_mod, win):
    w = win(running=True)
    handle, launcher = _handle(proc_mod), _launcher(proc_mod)
    assert launcher.is_running(handle) is True
    assert handle._native_handles["is_alive"]() is True

    launcher.terminate(handle, timeout_sec=1)
    calls_before = list(w.calls)
    assert launcher.is_running(handle) is False
    assert handle._native_handles["is_alive"]() is False
    assert w.calls == calls_before


def test_wait_after_close_returns_the_recorded_exit_code(proc_mod, win):
    w = win(running=False, exit_code=3)
    handle, launcher = _handle(proc_mod), _launcher(proc_mod)
    launcher.terminate(handle, timeout_sec=1)
    calls_before = list(w.calls)
    assert launcher.wait(handle, timeout_sec=1) == 3
    assert w.calls == calls_before


def test_wait_after_close_with_no_recorded_code_is_none(proc_mod, win):
    win(running=False)
    handle, launcher = _handle(proc_mod), _launcher(proc_mod)
    handle._native_handles.pop("exit_code", None)
    proc_mod._close_process_handles(handle)
    handle._native_handles.pop("exit_code", None)
    assert launcher.wait(handle, timeout_sec=1) is None


def test_close_helper_is_idempotent(proc_mod, win):
    w = win(running=False, exit_code=0)
    handle = _handle(proc_mod)
    proc_mod._close_process_handles(handle)
    proc_mod._close_process_handles(handle)
    assert sorted(w.closed) == sorted([PROC, THREAD])
