# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: one Orbital, one tray icon (bug #54).

Two halves of a single invariant:

  * a duplicate launch must die at the single-instance mutex guard, before any
    daemon/window/tray work — every extra shell is an extra tray icon;
  * every exit path must delete the icon it created — Windows keeps showing a
    dead process's icon until the user sweeps the pointer over it.

Everything here runs on macOS: Windows-only ctypes calls are exercised through
a fake ``windll``, and pystray through a fake ``Icon``.
"""

import inspect
import re
import sys
import threading
import types
from pathlib import Path

import pytest

from agent_os.desktop import main as main_mod
from agent_os.desktop import tray as tray_mod


REPO_ROOT = Path(__file__).resolve().parents[2]

ERROR_ALREADY_EXISTS = 183


# --------------------------------------------------------------------------
# Fake Win32 surface
# --------------------------------------------------------------------------


class _FakeKernel32:
    def __init__(self, handle=0x1234, last_error=0):
        self._handle = handle
        self._last_error = last_error
        self.created = []

    def CreateMutexW(self, attrs, initial_owner, name):
        self.created.append(name)
        return self._handle

    def GetLastError(self):
        return self._last_error


class _FakeUser32:
    def __init__(self, hwnd=0x99):
        self._hwnd = hwnd
        self.shown = []
        self.foregrounded = []

    def FindWindowW(self, cls, title):
        self.found_title = title
        return self._hwnd

    def ShowWindow(self, hwnd, cmd):
        self.shown.append((hwnd, cmd))
        return 1

    def SetForegroundWindow(self, hwnd):
        self.foregrounded.append(hwnd)
        return 1


def _fake_windows(monkeypatch, kernel32=None, user32=None):
    """Make main.py's `import ctypes; ctypes.windll...` hit fakes, on any OS."""
    import ctypes

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        ctypes,
        "windll",
        types.SimpleNamespace(
            kernel32=kernel32 or _FakeKernel32(),
            user32=user32 or _FakeUser32(),
        ),
        raising=False,
    )


# --------------------------------------------------------------------------
# The mutex guard itself
# --------------------------------------------------------------------------


def test_first_instance_claims_the_mutex(monkeypatch):
    kernel32 = _FakeKernel32(handle=0x1234, last_error=0)
    _fake_windows(monkeypatch, kernel32=kernel32)

    assert main_mod._acquire_single_instance("TestMutex") is True
    assert kernel32.created == ["TestMutex"]


def test_duplicate_instance_is_refused(monkeypatch):
    _fake_windows(
        monkeypatch,
        kernel32=_FakeKernel32(handle=0x1234, last_error=ERROR_ALREADY_EXISTS),
    )

    assert main_mod._acquire_single_instance("TestMutex") is False


def test_mutex_handle_is_held_so_it_is_not_collected(monkeypatch):
    _fake_windows(monkeypatch, kernel32=_FakeKernel32(handle=0xABCD))
    monkeypatch.setattr(main_mod, "_single_instance_mutex", None)

    main_mod._acquire_single_instance("TestMutex")

    assert main_mod._single_instance_mutex == 0xABCD, \
        "the handle must be parked in a module global — a collected handle releases the mutex"


def test_guard_fails_open_when_the_mutex_cannot_be_created(monkeypatch):
    """CreateMutexW returning NULL must not keep Orbital from starting."""
    _fake_windows(monkeypatch, kernel32=_FakeKernel32(handle=0, last_error=5))

    assert main_mod._acquire_single_instance("TestMutex") is True


def test_guard_fails_open_when_ctypes_raises(monkeypatch):
    import ctypes

    class _Exploding:
        @property
        def kernel32(self):
            raise OSError("no kernel32 here")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", _Exploding(), raising=False)

    assert main_mod._acquire_single_instance("TestMutex") is True


def test_guard_is_a_noop_off_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")

    assert main_mod._acquire_single_instance("TestMutex") is True


# --------------------------------------------------------------------------
# Surfacing the instance that already owns the mutex
# --------------------------------------------------------------------------


def test_existing_window_is_unhidden_and_raised(monkeypatch):
    user32 = _FakeUser32(hwnd=0x99)
    _fake_windows(monkeypatch, user32=user32)

    assert main_mod._activate_existing_window("Orbital") is True
    assert user32.found_title == "Orbital"
    # SW_SHOW un-hides a close-to-tray'd window, SW_RESTORE un-minimizes it.
    assert [cmd for _, cmd in user32.shown] == [5, 9]
    assert user32.foregrounded == [0x99]


def test_no_window_found_reports_false(monkeypatch):
    _fake_windows(monkeypatch, user32=_FakeUser32(hwnd=0))

    assert main_mod._activate_existing_window("Orbital") is False


# --------------------------------------------------------------------------
# main(): the guard runs before any daemon / window / tray work
# --------------------------------------------------------------------------


def _forbid_startup_work(monkeypatch):
    """Make every piece of shell startup explode if the guard let it through."""
    from agent_os.desktop import migration

    def _boom(name):
        def _raise(*args, **kwargs):
            raise AssertionError(f"{name}() must not run past the single-instance guard")
        return _raise

    monkeypatch.setattr(migration, "run_migrations", _boom("run_migrations"))
    monkeypatch.setattr(main_mod, "start_daemon", _boom("start_daemon"))
    monkeypatch.setattr(main_mod, "open_window", _boom("open_window"))
    monkeypatch.setattr(main_mod, "is_already_running", _boom("is_already_running"))
    monkeypatch.setattr(tray_mod, "start_tray", _boom("start_tray"))


def test_duplicate_launch_exits_before_tray_or_daemon(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["Orbital"])
    monkeypatch.setattr(main_mod, "_acquire_single_instance", lambda *a, **k: False)
    monkeypatch.setattr(main_mod, "_activate_existing_window", lambda *a, **k: True)
    _forbid_startup_work(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        main_mod.main()

    assert exc.value.code == 0, "a duplicate launch is a success, not an error"


def test_held_mutex_with_no_window_fails_open(monkeypatch):
    """A wedged instance must not lock the user out of their own app."""
    from agent_os.desktop import migration

    class _ReachedStartup(Exception):
        pass

    def _reached(*args, **kwargs):
        raise _ReachedStartup()

    monkeypatch.setattr(sys, "argv", ["Orbital"])
    monkeypatch.setattr(main_mod, "_acquire_single_instance", lambda *a, **k: False)
    monkeypatch.setattr(main_mod, "_activate_existing_window", lambda *a, **k: False)
    monkeypatch.setattr(migration, "run_migrations", _reached)

    with pytest.raises(_ReachedStartup):
        main_mod.main()


def test_guard_precedes_tray_and_daemon_in_source():
    """Ordering inside main(), independent of what any one test mocks out."""
    source = inspect.getsource(main_mod.main)

    guard_idx = source.find("_acquire_single_instance")
    assert guard_idx > 0, "main() must call the single-instance guard"

    for later in ("run_migrations", "start_daemon", "open_window", "start_tray"):
        assert source.find(later) > guard_idx, \
            f"{later} must come after the single-instance guard in main()"


# --------------------------------------------------------------------------
# Unrecognized arguments never reach the GUI (and so never make a tray icon)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["--setup-sandbox", "--teardown-sandbox", "--smoke-test"])
def test_recognized_flags_are_not_rejected(flag):
    assert main_mod._unrecognized_args([flag]) == []


def test_macos_process_serial_number_is_not_an_argument():
    """LaunchServices hands bundled apps -psn_*; it must not block startup."""
    assert main_mod._unrecognized_args(["-psn_0_12345"]) == []


def test_unknown_arguments_are_reported():
    assert main_mod._unrecognized_args(["--setup-sandbox", "C:\\x\\y.py"]) == ["C:\\x\\y.py"]


def test_main_refuses_unknown_arguments(monkeypatch, capsys):
    """The latent request_elevation() re-exec hazard: a bare script path used to
    fall through to the full daemon + window + tray path."""
    monkeypatch.setattr(sys, "argv", ["Orbital", "C:\\Orbital\\bin\\setup.py"])
    _forbid_startup_work(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        main_mod.main()

    assert exc.value.code != 0
    assert "setup.py" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Tray teardown: the icon goes away before the process does
# --------------------------------------------------------------------------


class _FakeIcon:
    def __init__(self, name=None, icon=None, title=None, menu=None):
        self.name = name
        self.menu = menu
        self.calls = []
        self._visible = False

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, value):
        # Mirrors pystray's setter: unchanged value is a no-op.
        if self._visible == value:
            return
        self._visible = value
        self.calls.append(f"visible={value}")

    def run(self):
        self.calls.append("run")
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
    monkeypatch.setattr(
        tray_mod,
        "pystray",
        types.SimpleNamespace(Icon=_FakeIcon, Menu=_FakeMenu, MenuItem=_FakeMenuItem),
    )
    monkeypatch.setattr(tray_mod, "_icon", None)
    monkeypatch.setattr(tray_mod, "_EXIT_GRACE_SECONDS", 0.01)
    yield
    tray_mod._icon = None


def _quit_action(icon):
    for item in icon.menu.items:
        if isinstance(item, _FakeMenuItem) and "Quit" in item.text:
            return item.action
    raise AssertionError("tray menu has no Quit item")


def test_quit_deletes_the_icon_before_stopping_the_loop(fake_tray):
    """icon.stop() only posts WM_STOP, and the NIM_DELETE in pystray's message
    loop `finally` never runs when the process leaves via os._exit()."""
    done = threading.Event()
    exit_thread = {}

    def shutdown():
        exit_thread["ident"] = threading.get_ident()
        done.set()

    tray_mod.start_tray(8000, lambda: None, shutdown)
    icon = tray_mod._icon
    assert icon is not None, "start_tray must publish the icon for other exit paths"

    _quit_action(icon)(icon, None)

    assert "visible=False" in icon.calls, "quit must delete the icon"
    assert icon.calls.index("visible=False") < icon.calls.index("stop"), \
        "the icon must be deleted BEFORE the message loop is stopped"

    assert done.wait(timeout=5), "the hard exit backstop must still fire"
    assert exit_thread["ident"] != threading.get_ident(), \
        "the hard exit must run off the callback thread so the loop can unwind"


def test_hide_tray_is_idempotent_and_safe_without_an_icon(fake_tray):
    tray_mod.hide_tray()  # no icon yet — must not raise

    tray_mod.start_tray(8000, lambda: None, lambda: None)
    icon = tray_mod._icon

    tray_mod.hide_tray()
    tray_mod.hide_tray()

    assert icon.calls.count("visible=False") == 1


def test_hide_tray_swallows_backend_errors(fake_tray):
    class _Broken:
        @property
        def visible(self):
            return True

        @visible.setter
        def visible(self, value):
            raise RuntimeError("Shell_NotifyIcon failed")

    monkeypatched = _Broken()
    tray_mod._icon = monkeypatched

    tray_mod.hide_tray()  # must not propagate — this runs on the way out


def test_keep_alive_loop_hides_the_icon_in_a_finally():
    """Ctrl-C, or anything else that ends the loop, must not leave a ghost."""
    source = inspect.getsource(main_mod.main)

    loop_idx = source.find("while True")
    assert loop_idx > 0, "main() must keep the process alive for the tray"

    tail = source[loop_idx:]
    finally_idx = tail.find("finally:")
    assert finally_idx > 0, "the keep-alive loop must have a finally block"
    assert "hide_tray" in tail[finally_idx:], \
        "the keep-alive finally must delete the tray icon"


# --------------------------------------------------------------------------
# App and installer must agree on the mutex name
# --------------------------------------------------------------------------


def test_installer_app_mutex_matches_the_app():
    """AppMutex is how Inno Setup notices a running Orbital during an upgrade;
    a rename on one side silently un-fixes half of bug #54."""
    iss = (REPO_ROOT / "installer" / "agentos-setup.iss").read_text(encoding="utf-8")

    match = re.search(r"^AppMutex=(.+)$", iss, re.MULTILINE)
    assert match, "installer/agentos-setup.iss [Setup] must declare AppMutex"
    assert match.group(1).strip() == main_mod.SINGLE_INSTANCE_MUTEX_NAME


def test_mutex_name_is_session_local():
    """Session namespace, not Global\\: a second logged-in user (fast user
    switching, RDP) has their own notification area and their own Orbital."""
    assert not main_mod.SINGLE_INSTANCE_MUTEX_NAME.startswith("Global\\")
