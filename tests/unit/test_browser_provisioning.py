# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bundled-Chromium provisioning + download backoff (China blocker).

The old path re-fired a 300s download subprocess on every boot whenever
the previous attempt failed, and had no bundled fallback — a mainland
user got a permanently degraded browser tool with no local option. These
tests pin the new behavior: installed → skip, bundled → extract (no
network), recent failure → back off.
"""

import os
import tarfile
import time
from unittest.mock import patch

import agent_os.desktop.migration as mig


def _mark_installed(browsers_dir: str, rev: str = "chromium-1200") -> None:
    d = os.path.join(browsers_dir, rev)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "INSTALLATION_COMPLETE"), "w").close()


def test_is_browser_installed_requires_completion_marker(tmp_path):
    browsers = tmp_path / "browsers"
    # A bare chromium dir (partial/failed download) is NOT "installed".
    (browsers / "chromium-1200").mkdir(parents=True)
    assert mig._is_browser_installed(str(browsers)) is False
    _mark_installed(str(browsers))
    assert mig._is_browser_installed(str(browsers)) is True


def test_headless_shell_dir_does_not_count_as_installed(tmp_path):
    browsers = tmp_path / "browsers"
    d = browsers / "chromium_headless_shell-1200"
    d.mkdir(parents=True)
    open(d / "INSTALLATION_COMPLETE", "w").close()
    # BrowserManager launches the full chromium, not headless_shell.
    assert mig._is_browser_installed(str(browsers)) is False


def test_provision_skips_when_already_installed(tmp_path):
    browsers = tmp_path / "browsers"
    browsers.mkdir()
    _mark_installed(str(browsers))
    with patch.dict(os.environ, {"PLAYWRIGHT_BROWSERS_PATH": str(browsers)}), \
         patch.object(mig, "_find_patchright_cli") as find_cli, \
         patch("threading.Thread") as thread:
        mig.provision_browsers_background()
    find_cli.assert_not_called()
    thread.assert_not_called()


def test_provision_backs_off_after_recent_failure(tmp_path):
    browsers = tmp_path / "browsers"
    browsers.mkdir()
    mig._write_download_status(str(browsers), state="failed", ts=time.time(), error="throttled")
    with patch.dict(os.environ, {"PLAYWRIGHT_BROWSERS_PATH": str(browsers)}), \
         patch.object(mig, "_bundled_browsers_archive", return_value=None), \
         patch.object(mig, "_find_patchright_cli") as find_cli, \
         patch("threading.Thread") as thread:
        mig.provision_browsers_background()
    # Backoff short-circuits BEFORE locating the CLI or spawning a thread.
    find_cli.assert_not_called()
    thread.assert_not_called()


def test_provision_retries_after_backoff_window_elapses(tmp_path):
    browsers = tmp_path / "browsers"
    browsers.mkdir()
    stale = time.time() - (mig._DOWNLOAD_BACKOFF_S + 60)
    mig._write_download_status(str(browsers), state="failed", ts=stale, error="throttled")
    with patch.dict(os.environ, {"PLAYWRIGHT_BROWSERS_PATH": str(browsers)}), \
         patch.object(mig, "_bundled_browsers_archive", return_value=None), \
         patch.object(mig, "_find_patchright_cli", return_value=None) as find_cli:
        mig.provision_browsers_background()
    # Past the window: it proceeds far enough to look for the CLI again.
    find_cli.assert_called_once()


def test_bundled_archive_is_extracted_instead_of_downloading(tmp_path):
    # Build a fake browser archive with the completion marker.
    src = tmp_path / "src"
    (src / "chromium-1200").mkdir(parents=True)
    open(src / "chromium-1200" / "INSTALLATION_COMPLETE", "w").close()
    open(src / "chromium-1200" / "chrome", "w").close()
    archive = tmp_path / "browsers.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(src, arcname=".")

    browsers = tmp_path / "browsers"
    browsers.mkdir()
    with patch.dict(os.environ, {"PLAYWRIGHT_BROWSERS_PATH": str(browsers)}), \
         patch.object(mig, "_bundled_browsers_archive", return_value=str(archive)), \
         patch.object(mig, "_find_patchright_cli") as find_cli, \
         patch("threading.Thread") as thread:
        mig.provision_browsers_background()

    # Extracted, and no download attempted.
    assert mig._is_browser_installed(str(browsers))
    find_cli.assert_not_called()
    thread.assert_not_called()


def test_china_locale_defaults_download_host_to_mirror(tmp_path):
    browsers = tmp_path / "browsers"
    browsers.mkdir()
    with patch.dict(os.environ, {"PLAYWRIGHT_BROWSERS_PATH": str(browsers)}, clear=False), \
         patch.object(mig, "_is_china_locale", return_value=True):
        os.environ.pop("PLAYWRIGHT_DOWNLOAD_HOST", None)
        os.environ.pop("PLAYWRIGHT_CHROMIUM_DOWNLOAD_HOST", None)
        env = mig._download_host_env(str(browsers))
    assert env["PLAYWRIGHT_DOWNLOAD_HOST"] == mig._CHINA_DOWNLOAD_HOST


def test_user_download_host_override_wins_over_mirror(tmp_path):
    browsers = tmp_path / "browsers"
    browsers.mkdir()
    with patch.dict(os.environ, {"PLAYWRIGHT_BROWSERS_PATH": str(browsers),
                                 "PLAYWRIGHT_DOWNLOAD_HOST": "https://my.mirror/pw"}), \
         patch.object(mig, "_is_china_locale", return_value=True):
        env = mig._download_host_env(str(browsers))
    assert env["PLAYWRIGHT_DOWNLOAD_HOST"] == "https://my.mirror/pw"
