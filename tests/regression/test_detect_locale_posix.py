# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The C/POSIX locale must not reach Chromium as a locale override.

Under LANG=C.UTF-8 ``getdefaultlocale()`` returns ``"C"``. Passing that to
``launch_persistent_context(locale=...)`` fails with "Invalid locale name", so
the browser tool could not start on a server or in a container.
"""

import pytest

from agent_os.daemon_v2 import browser_manager


@pytest.mark.parametrize("raw", ["C", "POSIX"])
def test_posix_locale_falls_back_to_a_valid_tag(monkeypatch, raw):
    monkeypatch.setattr(
        browser_manager.sys_locale, "getdefaultlocale", lambda: (raw, "UTF-8"),
    )
    monkeypatch.setattr(
        browser_manager.sys_locale, "getlocale", lambda: (raw, "UTF-8"),
    )
    assert browser_manager._detect_locale() == "en-US"


def test_real_locale_is_still_converted_to_bcp47(monkeypatch):
    monkeypatch.setattr(
        browser_manager.sys_locale, "getdefaultlocale", lambda: ("zh_CN", "UTF-8"),
    )
    assert browser_manager._detect_locale() == "zh-CN"
