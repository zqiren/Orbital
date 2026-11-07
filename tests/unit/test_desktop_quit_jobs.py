# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 134 Part B, decision (a): background jobs do not survive quitting.

Quitting while a job runs asks first ("N background job(s) are still
running. Quitting stops them.", Quit / Cancel) — tray Quit and Cmd+Q / menu
Quit. Closing the window on Windows (hide to tray) never asks; an OS
logout/shutdown is never blocked. Quitting ends every job tree before the
process exits (the desktop exits with ``os._exit``, past the app's shutdown
hooks)."""

from __future__ import annotations

import threading
import types
from unittest.mock import MagicMock

import pytest

from agent_os.daemon_v2 import jobs
from agent_os.desktop import main

try:
    from agent_os.desktop import tray as tray_mod
except Exception as exc:  # pystray connects to a display when imported on Linux
    pytest.skip(f"the tray needs a display: {exc}", allow_module_level=True)


@pytest.fixture(autouse=True)
def _reset():
    main._close_state["hidden"] = False
    yield
    main._close_state["hidden"] = False


@pytest.fixture
def asked(monkeypatch):
    calls = []
    answer = {"quit": False}

    def ask(n):
        calls.append(n)
        return answer["quit"]
    monkeypatch.setattr(main, "_ask_quit_with_jobs", ask)
    monkeypatch.setattr(main, "_system_is_ending_session", lambda: False)
    return types.SimpleNamespace(calls=calls, answer=answer)


def _jobs(monkeypatch, n):
    monkeypatch.setattr(jobs, "running_job_count", lambda: n)


def test_no_jobs_quits_without_asking(asked, monkeypatch):
    _jobs(monkeypatch, 0)
    assert main.confirm_quit() is True
    assert asked.calls == []


def test_running_jobs_ask_and_cancel_keeps_orbital_running(asked, monkeypatch):
    _jobs(monkeypatch, 2)
    assert main.confirm_quit() is False
    assert asked.calls == [2]
    asked.answer["quit"] = True
    assert main.confirm_quit() is True


def test_an_os_logout_or_shutdown_is_never_blocked(asked, monkeypatch):
    _jobs(monkeypatch, 3)
    monkeypatch.setattr(main, "_system_is_ending_session", lambda: True)
    assert main.confirm_quit() is True
    assert asked.calls == []


def test_the_prompt_text(monkeypatch):
    monkeypatch.setattr(main, "_os_localization", lambda: None)
    title, message, quit_label, cancel_label = main._quit_jobs_prompt(2)
    assert message == "2 background job(s) are still running. Quitting stops them."
    assert (quit_label, cancel_label) == ("Quit", "Cancel")
    monkeypatch.setattr(main, "_os_localization", lambda: {"x": "y"})
    _t, zh_message, zh_quit, zh_cancel = main._quit_jobs_prompt(2)
    assert "2" in zh_message and zh_quit == "退出" and zh_cancel == "取消"


@pytest.fixture
def window():
    w = MagicMock()
    w.native = MagicMock()
    return w


def _cmd_q(window):
    def applicationShouldTerminate_():
        return main._handle_window_close(window)
    return applicationShouldTerminate_()


def test_cmd_q_with_jobs_and_cancel_keeps_the_window(window, asked, monkeypatch):
    monkeypatch.setattr(main.sys, "platform", "darwin")
    _jobs(monkeypatch, 1)
    assert _cmd_q(window) is False
    window.hide.assert_not_called()          # Cancel: nothing changes
    assert asked.calls == [1]


def test_cmd_q_with_jobs_and_quit_quits(window, asked, monkeypatch):
    monkeypatch.setattr(main.sys, "platform", "darwin")
    _jobs(monkeypatch, 1)
    asked.answer["quit"] = True
    assert _cmd_q(window) is True


def test_closing_the_window_never_asks(window, asked, monkeypatch):
    for platform in ("darwin", "win32"):
        monkeypatch.setattr(main.sys, "platform", platform)
        _jobs(monkeypatch, 4)
        assert main._handle_window_close(window) is False      # hidden, not quit
    assert asked.calls == []


def test_quitting_ends_every_job_before_the_process_exits(monkeypatch):
    order = []
    monkeypatch.setattr(main, "flush_telemetry_before_exit", lambda port: order.append("flush"))
    monkeypatch.setattr(jobs, "stop_all_jobs", lambda: order.append("jobs") or 1)
    import agent_os.utils.pid_file as pid_file
    monkeypatch.setattr(pid_file, "release_pid_file", lambda: order.append("pid"))

    def fake_exit(code):
        order.append(("exit", code))
        raise SystemExit(code)
    monkeypatch.setattr(main.os, "_exit", fake_exit)
    with pytest.raises(SystemExit):
        main._exit_app(8000)
    assert order == ["jobs", "flush", "pid", ("exit", 0)]


# --- tray Quit (Windows / Linux) -----------------------------------------


class _FakeIcon:
    def __init__(self, name=None, icon=None, title=None, menu=None):
        self.menu = menu
        self.calls = []
        self.visible = False

    def run(self):
        self.visible = True

    def stop(self):
        self.calls.append("stop")


class _FakeMenuItem:
    def __init__(self, text, action=None, default=False):
        self.text = text
        self.action = action


class _FakeMenu:
    SEPARATOR = object()

    def __init__(self, *items):
        self.items = items


@pytest.fixture
def fake_tray(monkeypatch):
    monkeypatch.setattr(tray_mod, "pystray", types.SimpleNamespace(
        Icon=_FakeIcon, Menu=_FakeMenu, MenuItem=_FakeMenuItem))
    monkeypatch.setattr(tray_mod, "_icon", None)
    monkeypatch.setattr(tray_mod, "_EXIT_GRACE_SECONDS", 0.01)
    yield
    tray_mod._icon = None


def _quit(icon):
    for item in icon.menu.items:
        if isinstance(item, _FakeMenuItem) and "Quit" in item.text:
            item.action(icon, None)
            return
    raise AssertionError("no Quit item")


def test_tray_quit_cancelled_keeps_the_tray_and_the_app(fake_tray):
    exited = threading.Event()
    tray_mod.start_tray(8000, lambda: None, exited.set, confirm_quit_fn=lambda: False)
    icon = tray_mod._icon
    _quit(icon)
    assert not exited.wait(0.1)
    assert "stop" not in icon.calls and icon.visible is True


def test_tray_quit_confirmed_quits(fake_tray):
    exited = threading.Event()
    tray_mod.start_tray(8000, lambda: None, exited.set, confirm_quit_fn=lambda: True)
    _quit(tray_mod._icon)
    assert exited.wait(5)
