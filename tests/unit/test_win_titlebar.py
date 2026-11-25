# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Windows title bar follows the app's theme (2026-11-26).

Look D tints the whole window, and the native title bar stayed Windows' plain
white: a white strip on a near-black window in the dark theme. Windows 10
offers a window only its own light or dark bar; Windows 11 also takes an exact
caption colour. No window is created here: the DWM call is faked.
"""

import pytest

from agent_os.desktop import win_titlebar as wt

WIN10 = 19045
WIN11 = 22631


def _colorref(hex_rgb: str) -> int:
    r, g, b = (int(hex_rgb[i:i + 2], 16) for i in (1, 3, 5))
    return (b << 16) | (g << 8) | r


class TestWhatIsAskedOfWindows:
    def test_windows_10_dark_asks_for_the_dark_bar_only(self):
        assert wt.titlebar_attributes("dark", WIN10) == [(wt.DWMWA_USE_IMMERSIVE_DARK_MODE, 1)]

    def test_windows_10_light_turns_the_dark_bar_off(self):
        assert wt.titlebar_attributes("light", WIN10) == [(wt.DWMWA_USE_IMMERSIVE_DARK_MODE, 0)]

    def test_windows_11_also_gets_the_chrome_colour(self):
        dark = dict(wt.titlebar_attributes("dark", WIN11))
        light = dict(wt.titlebar_attributes("light", WIN11))
        assert dark[wt.DWMWA_USE_IMMERSIVE_DARK_MODE] == 1
        assert dark[wt.DWMWA_CAPTION_COLOR] == _colorref(wt.CHROME_DARK)
        assert light[wt.DWMWA_USE_IMMERSIVE_DARK_MODE] == 0
        assert light[wt.DWMWA_CAPTION_COLOR] == _colorref(wt.CHROME_LIGHT)

    def test_an_early_windows_10_uses_the_older_attribute_number(self):
        assert wt.titlebar_attributes("dark", 18363) == [(wt.DWMWA_USE_IMMERSIVE_DARK_MODE_OLD, 1)]

    def test_a_windows_too_old_for_any_of_it_is_left_alone(self):
        assert wt.titlebar_attributes("dark", 17134) == []

    @pytest.mark.parametrize("theme", [None, "", "system", "blue", 3])
    def test_anything_but_dark_is_light(self, theme):
        assert wt.titlebar_attributes(theme, WIN10) == [(wt.DWMWA_USE_IMMERSIVE_DARK_MODE, 0)]


class TestApplying:
    def test_each_attribute_is_set_on_the_window_and_the_frame_is_redrawn(self):
        calls, redrawn = [], []
        ok = wt.apply_titlebar_theme(
            4242, "dark", build=WIN11,
            set_attribute=lambda hwnd, attr, value: calls.append((hwnd, attr, value)) or 0,
            redraw=redrawn.append)
        assert ok is True
        assert calls == [(4242, a, v) for a, v in wt.titlebar_attributes("dark", WIN11)]
        assert redrawn == [4242]

    def test_a_refused_attribute_does_not_stop_the_rest_or_raise(self):
        calls = []

        def refuse_first(hwnd, attr, value):
            calls.append(attr)
            if len(calls) == 1:
                raise OSError("dwm said no")
            return 0

        assert wt.apply_titlebar_theme(1, "dark", build=WIN11, set_attribute=refuse_first,
                                       redraw=lambda hwnd: None) is True
        assert len(calls) == 2

    def test_no_window_or_nothing_to_set_is_a_quiet_no(self):
        called = []
        assert wt.apply_titlebar_theme(0, "dark", build=WIN10,
                                       set_attribute=lambda *a: called.append(a)) is False
        assert wt.apply_titlebar_theme(1, "dark", build=17134,
                                       set_attribute=lambda *a: called.append(a)) is False
        assert called == []


class TestThePageTellsTheShell:
    def test_the_watcher_reports_the_theme_now_and_on_every_change(self):
        js = wt.THEME_WATCH_JS
        assert "data-theme" in js and "MutationObserver" in js
        assert wt.API_METHOD in js
        assert "__orbitalTitlebarWatch" in js  # installed once per page, however often `loaded` fires


class TestWiring:
    def test_the_handle_comes_from_the_winforms_window(self):
        from types import SimpleNamespace
        window = SimpleNamespace(native=SimpleNamespace(
            Handle=SimpleNamespace(ToInt64=lambda: 987654)))
        assert wt.window_handle(window) == 987654

    def test_starting_the_watcher_runs_the_script_and_never_raises(self):
        ran = []

        class Window:
            def evaluate_js(self, js):
                ran.append(js)

        class Broken:
            def evaluate_js(self, js):
                raise RuntimeError("window is gone")

        wt.watch_theme(Window())
        wt.watch_theme(Broken())
        assert ran == [wt.THEME_WATCH_JS]

    def test_the_shell_api_has_the_method_the_page_calls(self):
        import inspect
        from agent_os.desktop import main
        assert f"def {wt.API_METHOD}(self, theme)" in inspect.getsource(main)
        assert "win_titlebar.watch_theme(window)" in inspect.getsource(main)
