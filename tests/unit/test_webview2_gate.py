# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""WebView2 gate: without the runtime, pywebview silently degrades to the
IE11 engine and Orbital opens a blank window. The installer must probe the
CORRECT Evergreen client GUID and run the bundled bootstrapper; the app
must fail loudly (native dialog) instead of rendering nothing."""

from pathlib import Path

import agent_os.desktop.main as desktop_main

REPO = Path(__file__).resolve().parents[2]
CORRECT_GUID = "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
# The old probe's GUID — does not exist in any WebView2 install, so the
# check always reported "missing". Must never come back.
BOGUS_GUID = "{F3017226-FE2A-4295-8BEE-13A6279B0638}"


def test_app_guard_uses_correct_runtime_guid():
    assert desktop_main._WEBVIEW2_RUNTIME_GUID == CORRECT_GUID


def test_non_windows_platforms_pass_the_guard():
    assert desktop_main._webview2_available() is True


def test_open_window_invokes_the_guard():
    import inspect

    src = inspect.getsource(desktop_main.open_window)
    assert "_webview2_available" in src
    assert "_fail_missing_webview2" in src


def test_installer_probes_correct_guid_and_runs_bootstrapper():
    iss = (REPO / "installer" / "agentos-setup.iss").read_text()
    assert CORRECT_GUID in iss
    assert BOGUS_GUID not in iss
    # The function must actually gate a [Run] entry for the bootstrapper.
    assert "Check: NeedsWebView2" in iss
    assert "MicrosoftEdgeWebView2Setup.exe" in iss
    # Per-user installs register under HKCU.
    assert "RegKeyExists(HKCU," in iss


def test_build_script_fetches_bootstrapper_before_iscc():
    sh = (REPO / "scripts" / "build-desktop.sh").read_text()
    assert "MicrosoftEdgeWebView2Setup.exe" in sh
    assert sh.index("MicrosoftEdgeWebView2Setup.exe") < sh.index("iscc installer/agentos-setup.iss")
