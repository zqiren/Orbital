# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
from unittest.mock import patch, MagicMock

import pytest

# --------------------------------------------------------------- bug #60 drag
# The macOS window had no drag region at all: pywebview's frameless mode leaves
# a *transparent* NSTitlebarContainerView, which is hit-transparent except for
# its traffic lights, so every mouse-down in the band fell through to the
# WKWebView and AppKit never armed a window drag. The fix installs a native
# drag strip. Whether the window physically MOVES cannot be asserted from
# pytest — that is the manual pass on a CI-built .dmg. What these pin is every
# way the strip can silently exist and do nothing.
#
# Its own file rather than test_desktop_main.py so that the module-level skip
# below only takes this suite out on the Windows/Linux CI legs, not the
# desktop-icon regressions next door (which are Windows-specific).

AppKit = pytest.importorskip("AppKit")


@pytest.fixture
def band_cache_reset():
    """The measured band height is cached module-wide; keep tests independent."""
    from agent_os.desktop import main
    original = main._MEASURED_BAND_HEIGHT
    main._MEASURED_BAND_HEIGHT = None
    yield
    main._MEASURED_BAND_HEIGHT = original


class _FakeNativeWindow:
    """Stands in for the NSWindow: all the installer asks for is contentView()."""

    def __init__(self, content_view):
        self._content_view = content_view

    def contentView(self):
        return self._content_view


def _real_content_view(width=900.0, height=600.0):
    """A detached NSView playing the WKWebView's role as the content view.

    Real rather than a double because the install path exercises genuine
    AppKit: addSubview_positioned_relativeTo_, layout anchors, and
    NSLayoutConstraint.activateConstraints_ all need actual views to bind to.
    """
    return AppKit.NSView.alloc().initWithFrame_(
        AppKit.NSMakeRect(0, 0, width, height)
    )


def _fake_titlebar_container(height):
    view = MagicMock()
    view.className.return_value = "NSTitlebarContainerView"
    view.frame.return_value.size.height = height
    return view


def _fake_windowed_native(band_height, content_view=None):
    """A native window whose theme frame carries a titlebar container."""
    other = MagicMock()
    other.className.return_value = "WebKitHost"
    theme_frame = MagicMock()
    theme_frame.subviews.return_value = [other, _fake_titlebar_container(band_height)]
    content = content_view if content_view is not None else MagicMock()
    if content_view is None:
        content.superview.return_value = theme_frame
        return _FakeNativeWindow(content)
    # A real NSView can't be told what its superview is; wrap it so the height
    # probe still finds a theme frame while the install path gets a real view.
    class _Wrapped(_FakeNativeWindow):
        def contentView(self):
            wrapper = MagicMock(wraps=content_view)
            wrapper.superview.return_value = theme_frame
            return wrapper

    return _Wrapped(content)


def test_install_titlebar_drag_strip_is_macos_only():
    """Windows keeps its native frame and native drag. The installer must not
    even touch the window there — inline_titlebar gates the caller, but the
    platform check belongs in the function too."""
    from agent_os.desktop import main
    native = MagicMock()
    with patch.object(main.sys, "platform", "win32"):
        assert main.install_titlebar_drag_strip(native) is None
    assert native.contentView.called is False


def test_install_titlebar_drag_strip_attaches_to_the_content_view(band_cache_reset):
    """Trap 1: the obvious placement — a subview of NSThemeFrame, below the
    titlebar container — is rejected by AppKit ("adding an unknown subview")
    and forced to index [0], BEHIND the webview, where it hit-tests nothing.
    The strip has to live inside the content view instead."""
    from agent_os.desktop import main
    content = _real_content_view()
    strip = main.install_titlebar_drag_strip(_FakeNativeWindow(content))
    assert strip is not None
    assert strip.superview() is content
    assert strip.className() == "_OrbitalTitlebarDragStrip"
    # Frontmost subview, so it wins hit-testing against the page underneath.
    assert content.subviews()[-1] is strip


def test_install_titlebar_drag_strip_uses_auto_layout_not_autoresizing(band_cache_reset):
    """Trap 4, the expensive one. WebKitHost reports isFlipped=True but AppKit
    positions its subviews with unflipped math, so an autoresizing mask tracked
    the width across a resize while the vertical pin silently did not — the
    strip stayed at its old y and became a dead band floating mid-content.
    Constraints are the only placement that survives resize and fullscreen."""
    from agent_os.desktop import main
    content = _real_content_view()
    strip = main.install_titlebar_drag_strip(_FakeNativeWindow(content))

    assert strip.translatesAutoresizingMaskIntoConstraints() is False
    # leading / trailing / top are held by the common ancestor...
    pinned = {c.firstAttribute() for c in content.constraints()}
    assert pinned == {
        AppKit.NSLayoutAttributeLeading,
        AppKit.NSLayoutAttributeTrailing,
        AppKit.NSLayoutAttributeTop,
    }
    # ...the height constant by the strip itself.
    heights = [c for c in strip.constraints()
               if c.firstAttribute() == AppKit.NSLayoutAttributeHeight]
    assert len(heights) == 1
    assert heights[0].constant() == main._TITLEBAR_BAND_FALLBACK


def test_install_titlebar_drag_strip_is_idempotent(band_cache_reset):
    """Trap 3: `loaded` re-fires whenever the origin guard reloads the SPA. A
    second install must not stack a second strip (or a second constraint set)
    on top of the first."""
    from agent_os.desktop import main
    content = _real_content_view()
    native = _FakeNativeWindow(content)

    first = main.install_titlebar_drag_strip(native)
    second = main.install_titlebar_drag_strip(native)

    assert second is first
    assert content.subviews().count() == 1
    assert content.constraints().count() == 3


def test_install_titlebar_drag_strip_survives_a_hostile_window():
    """Window chrome is cosmetic: a failure here must never stop the app from
    opening, exactly like _reveal_traffic_lights."""
    from agent_os.desktop import main
    native = MagicMock()
    native.contentView.side_effect = RuntimeError("no content view yet")
    assert main.install_titlebar_drag_strip(native) is None


def test_drag_strip_class_is_cached():
    """PyObjC raises if the same ObjC class name is registered twice in one
    process, so the subclass must be built exactly once."""
    from agent_os.desktop import main
    assert main._drag_strip_class() is main._drag_strip_class()


def test_drag_strip_accepts_first_mouse_and_declines_background_drag(band_cache_reset):
    """acceptsFirstMouse: drag an inactive window without click-to-focus first.
    mouseDownCanMoveWindow=NO: otherwise AppKit takes the click for the
    background-drag path and mouseDown_ (and the double-click gesture with it)
    never runs."""
    from agent_os.desktop import main
    strip = main.install_titlebar_drag_strip(_FakeNativeWindow(_real_content_view()))
    assert strip.acceptsFirstMouse_(None) is True
    assert strip.mouseDownCanMoveWindow() is False


def test_titlebar_band_height_is_measured_off_the_live_container(band_cache_reset):
    """Derived, not hardcoded, so it cannot drift from the system metric."""
    from agent_os.desktop import main
    assert main.titlebar_band_height(_fake_windowed_native(32.0)) == 32.0


def test_titlebar_band_height_caches_the_windowed_measurement(band_cache_reset):
    """NSTitlebarContainerView is ABSENT in fullscreen. Without the cache a
    measurement taken there would silently fall back and resize the band."""
    from agent_os.desktop import main
    assert main.titlebar_band_height(_fake_windowed_native(32.0)) == 32.0

    fullscreen = MagicMock()
    fullscreen.contentView.return_value.superview.return_value.subviews.return_value = []
    assert main.titlebar_band_height(fullscreen) == 32.0


def test_titlebar_band_height_falls_back_without_a_container(band_cache_reset):
    from agent_os.desktop import main
    native = MagicMock()
    native.contentView.return_value.superview.return_value.subviews.return_value = []
    assert main.titlebar_band_height(native) == main._TITLEBAR_BAND_FALLBACK


def test_titlebar_band_height_falls_back_on_a_zero_height_container(band_cache_reset):
    """A 0pt band would install an invisible, unhittable strip."""
    from agent_os.desktop import main
    assert main.titlebar_band_height(_fake_windowed_native(0.0)) == main._TITLEBAR_BAND_FALLBACK


def test_measured_band_height_reaches_the_installed_strip(band_cache_reset):
    """End to end: what the container reports is what the constraint pins."""
    from agent_os.desktop import main
    content = _real_content_view()
    strip = main.install_titlebar_drag_strip(_fake_windowed_native(31.0, content))
    heights = [c.constant() for c in strip.constraints()
               if c.firstAttribute() == AppKit.NSLayoutAttributeHeight]
    assert heights == [31.0]
    assert strip.frame().size.height == 31.0


@pytest.mark.parametrize("preference,expected", [
    ("Maximize", "zoom"),
    ("Minimize", "minimize"),
    ("None", "none"),
    (None, "zoom"),        # unset — the common case, system default is zoom
    ("nonsense", "zoom"),
])
def test_double_click_titlebar_action_honors_the_system_preference(preference, expected):
    from agent_os.desktop import main
    defaults = MagicMock()
    defaults.standardUserDefaults.return_value.stringForKey_.return_value = preference
    with patch.object(AppKit, "NSUserDefaults", defaults):
        assert main._double_click_titlebar_action() == expected
    defaults.standardUserDefaults.return_value.stringForKey_.assert_called_with(
        "AppleActionOnDoubleClick"
    )


def test_drag_strip_single_click_hands_off_to_the_native_drag_loop():
    """performWindowDragWithEvent_ rather than a hand-rolled mouse-delta loop —
    it defers to AppKit, so snapping, Spaces, and multi-display are native."""
    from agent_os.desktop import main
    view = MagicMock()
    event = MagicMock()
    event.clickCount.return_value = 1

    main._drag_strip_mouse_down(view, event)

    view.window.return_value.performWindowDragWithEvent_.assert_called_once_with(event)


@pytest.mark.parametrize("action,called,not_called", [
    ("zoom", "zoom_", "miniaturize_"),
    ("minimize", "miniaturize_", "zoom_"),
])
def test_drag_strip_double_click_applies_the_configured_action(action, called, not_called):
    from agent_os.desktop import main
    view = MagicMock()
    event = MagicMock()
    event.clickCount.return_value = 2
    window = view.window.return_value

    with patch.object(main, "_double_click_titlebar_action", return_value=action):
        main._drag_strip_mouse_down(view, event)

    getattr(window, called).assert_called_once_with(view)
    assert getattr(window, not_called).called is False
    # A double-click must not also start a drag.
    assert window.performWindowDragWithEvent_.called is False


def test_drag_strip_double_click_does_nothing_when_the_action_is_none():
    from agent_os.desktop import main
    view = MagicMock()
    event = MagicMock()
    event.clickCount.return_value = 2
    window = view.window.return_value

    with patch.object(main, "_double_click_titlebar_action", return_value="none"):
        main._drag_strip_mouse_down(view, event)

    assert window.zoom_.called is False
    assert window.miniaturize_.called is False
    assert window.performWindowDragWithEvent_.called is False


def test_drag_strip_mouse_down_ignores_a_detached_view():
    from agent_os.desktop import main
    view = MagicMock()
    view.window.return_value = None
    main._drag_strip_mouse_down(view, MagicMock())  # must not raise


def test_set_drag_strip_hidden_toggles_the_strip(band_cache_reset):
    """Fullscreen collapses --titlebar-h to 0 and the SPA moves real UI into
    the top band; a visible strip would eat its clicks. Hidden views drop out
    of hit-testing entirely."""
    from agent_os.desktop import main
    strip = main.install_titlebar_drag_strip(_FakeNativeWindow(_real_content_view()))
    assert strip.isHidden() is False

    main.set_drag_strip_hidden(strip, True)
    assert strip.isHidden() is True

    main.set_drag_strip_hidden(strip, False)
    assert strip.isHidden() is False


def test_set_drag_strip_hidden_without_a_strip_is_a_noop():
    """The fullscreen handlers can fire before `loaded` has installed anything."""
    from agent_os.desktop import main
    main.set_drag_strip_hidden(None, True)


def test_run_on_main_thread_runs_inline_when_already_on_main():
    from agent_os.desktop import main
    calls = []
    main._run_on_main_thread(lambda: calls.append(AppKit.NSThread.isMainThread()))
    assert calls == [True]


def test_run_on_main_thread_hops_off_a_worker_thread():
    """pywebview dispatches `loaded` / `maximized` / `restored` on throw-away
    threads (webview/event.py), where AppKit geometry raises
    NSInternalInconsistencyException. The hop is what makes the strip possible
    at all — trap 3."""
    import threading
    from agent_os.desktop import main

    queued = []
    queue = MagicMock()
    queue.mainQueue.return_value.addOperationWithBlock_.side_effect = queued.append

    def worker():
        with patch.object(AppKit, "NSOperationQueue", queue):
            main._run_on_main_thread(lambda: None)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert len(queued) == 1


# ---------------------------------------------------------------- open_window
# The strip is useless unless it is bound to the right events, so drive
# open_window with a fake pywebview and fire them.

class _RecordingEvent:
    def __init__(self):
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self

    def fire(self):
        for handler in self.handlers:
            handler()


class _FakeWindow:
    def __init__(self, native, url):
        self.native = native
        self._url = url
        self.evaluated = []
        self.events = MagicMock()
        for name in ("closing", "loaded", "before_show", "maximized", "restored"):
            setattr(self.events, name, _RecordingEvent())

    def get_current_url(self):
        return self._url

    def evaluate_js(self, script):
        self.evaluated.append(script)


@pytest.fixture
def opened_window(band_cache_reset):
    """open_window() driven against a fake pywebview and a real content view."""
    from agent_os.desktop import main
    if sys.platform != "darwin":
        pytest.skip("inline titlebar is macOS only")

    content = _real_content_view()
    window = _FakeWindow(_FakeNativeWindow(content), "http://127.0.0.1:8000/?chrome=mac-inline")
    fake_webview = MagicMock()
    fake_webview.create_window.return_value = window

    previous = main._window
    main._window = None
    try:
        with patch.dict(sys.modules, {"webview": fake_webview}):
            main.open_window(8000)
        yield window, content
    finally:
        main._window = previous


def test_open_window_installs_the_drag_strip_on_loaded(opened_window):
    """Trap 2: pywebview only swaps the WKWebView in as the contentView inside
    webView_didFinishNavigation_, so anything bound to `before_show` attaches
    to the throw-away default content view. `loaded` is the earliest point the
    real one exists."""
    window, content = opened_window
    assert content.subviews().count() == 0  # nothing installed at create time

    window.events.before_show.fire()
    assert content.subviews().count() == 0

    window.events.loaded.fire()
    assert content.subviews()[0].className() == "_OrbitalTitlebarDragStrip"


def test_open_window_reload_does_not_stack_strips(opened_window):
    window, content = opened_window
    window.events.loaded.fire()
    window.events.loaded.fire()
    assert content.subviews().count() == 1


def test_open_window_hides_the_strip_in_fullscreen(opened_window):
    """Both halves of the fullscreen contract: the strip stops eating clicks on
    the UI the SPA moves into the band, and drag comes back on the way out."""
    window, content = opened_window
    window.events.loaded.fire()
    strip = content.subviews()[0]

    window.events.maximized.fire()
    assert strip.isHidden() is True
    assert window.evaluated[-1].endswith("= 'fullscreen'")

    window.events.restored.fire()
    assert strip.isHidden() is False
    assert window.evaluated[-1].endswith("= 'mac-inline'")


def test_open_window_installs_hidden_when_fullscreen_beat_the_install(opened_window):
    """The install is asynchronous (main-queue hop), so a fullscreen transition
    can land first. The strip must not appear visible after the fact."""
    window, content = opened_window
    window.events.maximized.fire()          # no strip yet — must not raise
    window.events.loaded.fire()
    assert content.subviews()[0].isHidden() is True


def test_run_on_main_thread_does_not_re_invoke_a_raising_callable():
    """Deciding where to run and running have to be separate try blocks — one
    combined block would catch the callable's own exception and retry it on the
    non-AppKit fallback path, installing a second strip."""
    from agent_os.desktop import main
    calls = []

    def boom():
        calls.append(1)
        raise RuntimeError("nope")

    main._run_on_main_thread(boom)
    assert calls == [1]
