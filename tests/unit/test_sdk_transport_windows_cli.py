# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Claude Code on Windows: never hand the SDK npm's ``claude.cmd`` shim.

claude-agent-sdk 0.2.x refuses to spawn a ``.bat``/``.cmd`` file as the CLI
("Refusing to execute batch script ...": cmd.exe re-parses the argv, and no
escaping is reliable). Orbital resolved ``claude`` through PATH to
``%APPDATA%\\npm\\claude.CMD`` and passed it as ``cli_path``, so every
Claude Code dispatch on a machine with an npm install failed at adapter start
(verified on the installed v0.16.0). The shim only runs
``node_modules\\@anthropic-ai\\claude-code\\bin\\claude.exe`` beside it; that
native exe is what the SDK must get.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agent_os.agent.transports.sdk_transport import SDKTransport, native_claude_cli


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"MZ")
    return path


def _npm_shim(tmp_path, name="claude.CMD"):
    return _touch(tmp_path / "npm" / name)


def _npm_exe(tmp_path):
    return _touch(tmp_path / "npm" / "node_modules" / "@anthropic-ai"
                  / "claude-code" / "bin" / "claude.exe")


@pytest.mark.parametrize("name", ["claude.CMD", "claude.cmd", "claude.bat"])
def test_npm_shim_maps_to_the_native_exe_beside_it(tmp_path, name):
    shim = _npm_shim(tmp_path, name)
    exe = _npm_exe(tmp_path)
    assert native_claude_cli(str(shim), windows=True, home=str(tmp_path / "home"),
                             which=lambda _: None) == str(exe)


def test_falls_back_to_claude_exe_on_path(tmp_path):
    shim = _npm_shim(tmp_path)
    on_path = _touch(tmp_path / "bin" / "claude.exe")
    got = native_claude_cli(str(shim), windows=True, home=str(tmp_path / "home"),
                            which=lambda n: str(on_path) if n == "claude.exe" else None)
    assert got == str(on_path)


def test_falls_back_to_the_native_installer_location(tmp_path):
    shim = _npm_shim(tmp_path)
    native = _touch(tmp_path / "home" / ".local" / "bin" / "claude.exe")
    assert native_claude_cli(str(shim), windows=True, home=str(tmp_path / "home"),
                             which=lambda _: None) == str(native)


def test_a_path_hit_that_is_itself_a_shim_is_not_used(tmp_path):
    """PATHEXT resolution can hand back ``claude.exe.cmd``."""
    shim = _npm_shim(tmp_path)
    fake = _touch(tmp_path / "bin" / "claude.exe.cmd")
    native = _touch(tmp_path / "home" / ".local" / "bin" / "claude.exe")
    got = native_claude_cli(str(shim), windows=True, home=str(tmp_path / "home"),
                            which=lambda _: str(fake))
    assert got == str(native)


def test_shim_only_machine_keeps_the_shim_so_the_sdk_explains(tmp_path):
    shim = _npm_shim(tmp_path)
    assert native_claude_cli(str(shim), windows=True, home=str(tmp_path / "home"),
                             which=lambda _: None) == str(shim)


@pytest.mark.parametrize("command", ["claude", r"C:\tools\claude.exe", "", None])
def test_non_shim_commands_pass_through(command, tmp_path):
    assert native_claude_cli(command, windows=True, home=str(tmp_path),
                             which=lambda _: None) == command


def test_off_windows_nothing_changes(tmp_path):
    shim = _npm_shim(tmp_path)
    _npm_exe(tmp_path)
    assert native_claude_cli(str(shim), windows=False, home=str(tmp_path),
                             which=lambda _: None) == str(shim)


@pytest.mark.asyncio
async def test_start_hands_the_sdk_the_native_exe(tmp_path, monkeypatch):
    import functools

    from agent_os.agent.transports import sdk_transport

    shim = _npm_shim(tmp_path)
    exe = _npm_exe(tmp_path)
    # Force the Windows branch of the helper only: faking sys.platform would
    # also send shutil.which down its Windows path, which needs _winapi and
    # crashes on the macOS CI runner.
    monkeypatch.setattr(sdk_transport, "native_claude_cli", functools.partial(
        native_claude_cli, windows=True, home=str(tmp_path / "home"),
        which=lambda _: None))
    transport = SDKTransport()
    with patch("agent_os.agent.transports.sdk_transport.ClaudeSDKClient") as client, \
         patch("agent_os.agent.transports.sdk_transport.ClaudeAgentOptions") as options:
        client.return_value.connect = AsyncMock()
        await transport.start(str(shim), [], str(tmp_path))
    assert options.call_args[1]["cli_path"] == str(exe)


# ---------------------------------------------------------------------------
# No console window for claude.exe (the windowed Orbital.exe has no console;
# a console child started without CREATE_NO_WINDOW gets a new visible one —
# an empty "claude" window on every Claude Code dispatch, v0.16.0 Windows).
# ---------------------------------------------------------------------------


def test_sdk_still_spawns_through_anyio_open_process():
    """The no-window fix wraps exactly this call. If an SDK release stops
    making it, the wrapper silently does nothing — fail here instead."""
    import inspect

    from claude_agent_sdk._internal.transport import subprocess_cli

    source = inspect.getsource(subprocess_cli)
    assert "anyio.open_process(" in source


@pytest.mark.asyncio
async def test_sdk_spawns_get_create_no_window(monkeypatch):
    from claude_agent_sdk._internal.transport import subprocess_cli

    from agent_os.agent.transports import sdk_transport
    from agent_os.utils.subprocess_flags import CREATE_NO_WINDOW

    calls = []

    class _FakeAnyio:
        sentinel = object()

        async def open_process(self, command, **kwargs):
            calls.append((command, kwargs))
            return "process"

    fake = _FakeAnyio()
    monkeypatch.setattr(subprocess_cli, "anyio", fake)
    sdk_transport.install_sdk_no_window_spawn(windows=True)

    wrapped = subprocess_cli.anyio
    assert await wrapped.open_process(["claude.exe", "-v"], stdout=-1) == "process"
    assert await wrapped.open_process(["claude.exe"], creationflags=0x10) == "process"
    assert calls[0] == (["claude.exe", "-v"],
                        {"stdout": -1, "creationflags": CREATE_NO_WINDOW})
    assert calls[1][1]["creationflags"] == 0x10 | CREATE_NO_WINDOW
    assert wrapped.sentinel is fake.sentinel, "everything else is the real anyio"

    sdk_transport.install_sdk_no_window_spawn(windows=True)
    assert subprocess_cli.anyio is wrapped, "idempotent: never double-wrapped"


def test_off_windows_the_sdk_is_left_alone(monkeypatch):
    from claude_agent_sdk._internal.transport import subprocess_cli

    from agent_os.agent.transports import sdk_transport

    original = object()
    monkeypatch.setattr(subprocess_cli, "anyio", original)
    sdk_transport.install_sdk_no_window_spawn(windows=False)
    assert subprocess_cli.anyio is original
