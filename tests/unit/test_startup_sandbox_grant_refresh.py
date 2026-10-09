# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 109 D2 (T4) — the toolchain grants run at daemon start, off the loop.

Spec 077 W1 promised a re-grant on every daemon start; the method existed
(``WindowsPlatformProvider.refresh_sandbox_grants``) but nothing called it,
so the install was the only moment the grants ever ran — the worst possible
place for a slow step (issue #55). The startup hook in ``create_app`` now
schedules it in the default executor. Only the Windows provider defines the
method, so its presence *is* the platform check.
"""

from __future__ import annotations

import asyncio
import ctypes
import threading
from unittest.mock import MagicMock

from agent_os.api.app import schedule_sandbox_grant_refresh


class _WindowsLikeProvider:
    def __init__(self):
        self.calls: list = []
        self.thread: int | None = None

    def refresh_sandbox_grants(self, workspaces=None):
        self.thread = threading.get_ident()
        self.calls.append(list(workspaces or []))


class _OtherProvider:
    """macOS / null: no ``refresh_sandbox_grants``."""


def test_startup_refreshes_grants_off_the_loop():
    provider = _WindowsLikeProvider()

    async def main():
        fut = schedule_sandbox_grant_refresh(provider, [r"C:\ws\a", r"C:\ws\b"])
        assert fut is not None
        loop_thread = threading.get_ident()
        await fut
        return loop_thread

    loop_thread = asyncio.run(main())
    assert provider.calls == [[r"C:\ws\a", r"C:\ws\b"]]
    assert provider.thread is not None and provider.thread != loop_thread, (
        "blocking icacls work must never run on the event loop"
    )


def test_a_provider_without_the_method_is_skipped():
    async def main():
        return schedule_sandbox_grant_refresh(_OtherProvider(), [r"C:\ws"])

    assert asyncio.run(main()) is None


def test_a_raising_refresh_is_logged_not_propagated(caplog):
    provider = MagicMock()
    provider.refresh_sandbox_grants.side_effect = RuntimeError("icacls exploded")

    async def main():
        fut = schedule_sandbox_grant_refresh(provider, [])
        # Nobody awaits the future at startup (fire and forget); the done
        # callback must consume and log the exception so it never surfaces
        # as "exception was never retrieved".
        while not fut.done():
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.01)  # let the callback run
        return fut

    with caplog.at_level("WARNING", logger="agent_os.api.app"):
        fut = asyncio.run(main())
    assert fut.done()
    assert "icacls exploded" in caplog.text


def test_only_the_windows_provider_exposes_the_refresh():
    for name in ("windll", "GetLastError", "FormatError", "get_last_error", "set_last_error"):
        if not hasattr(ctypes, name):
            setattr(ctypes, name, MagicMock())
    if not hasattr(ctypes, "WinError"):
        ctypes.WinError = OSError
    from agent_os.platform.null import NullProvider
    from agent_os.platform.windows.provider import WindowsPlatformProvider

    assert hasattr(WindowsPlatformProvider, "refresh_sandbox_grants")
    assert not hasattr(NullProvider, "refresh_sandbox_grants")
    try:
        from agent_os.platform.macos.provider import MacOSPlatformProvider
    except Exception:  # pragma: no cover - macOS module absent on other hosts
        return
    assert not hasattr(MacOSPlatformProvider, "refresh_sandbox_grants")


def test_the_suite_never_runs_the_real_startup_refresh():
    """``tests/conftest.py`` patches the hook's target for every test.

    Without it, any ``with TestClient(create_app(...))`` test on a Windows
    machine with the sandbox account ran the real ``icacls /grant`` over the
    developer's toolchain roots (found during spec 109 V1: a unit-suite run
    re-granted ``AgentOS-Worker`` on ``%LOCALAPPDATA%\\Programs``).
    """
    import agent_os.api.app as app_module

    assert isinstance(app_module.schedule_sandbox_grant_refresh, MagicMock)
