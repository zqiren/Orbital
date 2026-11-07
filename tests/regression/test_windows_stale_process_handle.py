# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""A ProcessHandle must not outlive its Win32 handles.

Windows reuses handle values. terminate() closed the handles but kept their
values, so a later process given the same value looked like this one: it was
reported as running, and a second terminate() would have killed it.
"""

from __future__ import annotations

import sys

import pytest

from agent_os.platform.types import ProcessHandle

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Win32 handles")


@pytest.fixture
def win32(monkeypatch):
    from agent_os.platform.windows import process

    closed: list[int] = []
    killed: list[int] = []
    # Every handle value looks alive, as a reused one does.
    monkeypatch.setattr(process, "_WaitForSingleObject",
                        lambda _handle, _ms: process.WAIT_TIMEOUT)
    monkeypatch.setattr(process, "_CloseHandle", closed.append)
    monkeypatch.setattr(process, "_TerminateProcess",
                        lambda handle, _code: killed.append(handle))
    monkeypatch.setattr(process, "_GenerateConsoleCtrlEvent", lambda *_args: None)
    return process, closed, killed


def _handle() -> ProcessHandle:
    return ProcessHandle(
        pid=1234, command="cmd",
        _native_handles={"process_handle": 11, "thread_handle": 12,
                         "is_alive": lambda: True},
    )


def test_a_terminated_handle_is_not_running(win32):
    process, closed, _ = win32
    launcher = process.ProcessLauncher.__new__(process.ProcessLauncher)
    handle = _handle()
    launcher.terminate(handle, timeout_sec=0)
    assert closed == [11, 12]
    assert launcher.is_running(handle) is False
    assert handle._native_handles["is_alive"]() is False


def test_a_second_terminate_touches_nothing(win32):
    process, closed, killed = win32
    launcher = process.ProcessLauncher.__new__(process.ProcessLauncher)
    handle = _handle()
    launcher.terminate(handle, timeout_sec=0)
    killed.clear()
    closed.clear()
    assert launcher.terminate(handle, timeout_sec=0) is True
    assert killed == []
    assert closed == []
