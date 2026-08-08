# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Desktop chrome localization: the pywebview menu bar / quit dialog follow
the OS language (Chinese systems get a Chinese menu bar)."""

from unittest.mock import patch

import agent_os.desktop.main as desktop_main


def test_zh_dict_covers_exactly_pywebviews_key_set():
    from webview.localization import original_localization

    assert set(desktop_main._ZH_LOCALIZATION) == set(original_localization)


def test_chinese_os_locale_selects_zh_dict():
    with patch("locale.getdefaultlocale", return_value=("zh_CN", "UTF-8")):
        assert desktop_main._os_localization() is desktop_main._ZH_LOCALIZATION


def test_english_os_locale_keeps_pywebview_default():
    with patch("locale.getdefaultlocale", return_value=("en_US", "UTF-8")):
        assert desktop_main._os_localization() is None


def test_unreadable_locale_keeps_pywebview_default():
    with patch("locale.getdefaultlocale", side_effect=ValueError):
        assert desktop_main._os_localization() is None
