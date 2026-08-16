# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import json
import os
import subprocess
import sys
import time
import logging

from agent_os.utils.subprocess_flags import win_no_window_flags

logger = logging.getLogger(__name__)

CURRENT_DATA_VERSION = 1

# Background Chromium download tuning. The old 300s cap could not finish a
# ~170MB download over a throttled (e.g. mainland-China) link, and the
# only skip-guard was "a chromium* dir exists" — a failed/partial download
# left an empty dir and re-fired the subprocess every single boot.
_DOWNLOAD_TIMEOUT_S = 1800  # 30 min — background thread, blocks nothing
_DOWNLOAD_STATUS_FILE = ".browser-download-status.json"
# Back off after a failure instead of retrying every launch.
_DOWNLOAD_BACKOFF_S = 6 * 60 * 60  # 6 hours
# Mirror for mainland-China systems when the user hasn't set a download host.
_CHINA_DOWNLOAD_HOST = "https://cdn.npmmirror.com/binaries/playwright"


def _get_data_dir() -> str:
    if sys.platform == "win32":
        return os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "Orbital")
    elif sys.platform == "darwin":
        return os.path.join(os.path.expanduser("~"), "Library", "Application Support", "Orbital")
    else:
        return os.path.join(os.path.expanduser("~"), ".orbital")


DATA_DIR = _get_data_dir()
VERSION_FILE = os.path.join(DATA_DIR, "version.json")

MIGRATIONS: dict = {
    # version_from: migration_function
}


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def get_data_version() -> int:
    try:
        with open(VERSION_FILE) as f:
            return json.load(f)["data_version"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return 0


def set_data_version(version: int):
    with open(VERSION_FILE, "w") as f:
        json.dump({"data_version": version}, f)


def _find_patchright_cli() -> str | None:
    """Locate the patchright CLI script bundled by the driver package.

    In a PyInstaller bundle ``sys.executable`` points to the frozen app
    binary, so ``sys.executable -m patchright`` would fork the app itself.
    Instead we locate the Node-based CLI shipped with the patchright driver.
    """
    try:
        import patchright
        pkg_dir = os.path.dirname(patchright.__file__)
        cli = os.path.join(pkg_dir, "driver", "package", "cli.js")
        if os.path.isfile(cli):
            return cli
    except Exception:
        pass
    return None


def setup_browser_path():
    """Set PLAYWRIGHT_BROWSERS_PATH to a writable, persistent location.

    Called during startup (before daemon) so the env var is available
    when BrowserManager launches.  Does NOT download anything — the
    actual Chromium provisioning happens in the background after the
    daemon is up via ``provision_browsers_background()``.
    """
    browsers_dir = os.path.join(DATA_DIR, "browsers")
    os.makedirs(browsers_dir, exist_ok=True)
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = browsers_dir


def _is_browser_installed(browsers_dir: str) -> bool:
    """True only when a COMPLETE Chromium build is present.

    Patchright writes ``INSTALLATION_COMPLETE`` into a browser dir once the
    download+extract fully succeeds. Checking the marker (not merely that a
    ``chromium*`` directory exists) means a partial/failed download no
    longer looks 'installed' — and, conversely, no longer re-fires forever.
    """
    try:
        entries = os.listdir(browsers_dir)
    except OSError:
        return False
    for entry in entries:
        if entry.startswith("chromium") and not entry.startswith("chromium_headless"):
            if os.path.isfile(os.path.join(browsers_dir, entry, "INSTALLATION_COMPLETE")):
                return True
    return False


def _bundled_browsers_archive() -> str | None:
    """Path to the Chromium archive shipped inside the app, or None.

    Installers bundle ``browsers.tar.gz`` so a fresh machine never needs
    the network — the build scripts stage it next to the SPA (macOS:
    Contents/Resources/, Windows: alongside the executable).
    """
    if not getattr(sys, "frozen", False):
        return None
    exe_dir = os.path.dirname(sys.executable)
    for candidate in (
        os.path.join(os.path.dirname(exe_dir), "Resources", "browsers.tar.gz"),  # macOS .app
        os.path.join(exe_dir, "browsers.tar.gz"),                                # Windows
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


def _read_download_status(browsers_dir: str) -> dict:
    try:
        with open(os.path.join(browsers_dir, _DOWNLOAD_STATUS_FILE)) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _write_download_status(browsers_dir: str, **fields) -> None:
    try:
        with open(os.path.join(browsers_dir, _DOWNLOAD_STATUS_FILE), "w") as f:
            json.dump(fields, f)
    except OSError:
        pass


def _download_host_env(browsers_dir: str) -> dict:
    """Subprocess env, defaulting a China mirror on zh systems.

    Patchright honours PLAYWRIGHT_DOWNLOAD_HOST; a user override always
    wins. cdn.playwright.dev is Azure-fronted and throttled hard from
    mainland ISPs, so a zh system with no override gets the npmmirror
    binary mirror.
    """
    env = {**os.environ, "PLAYWRIGHT_BROWSERS_PATH": browsers_dir}
    already_set = any(
        env.get(k)
        for k in ("PLAYWRIGHT_DOWNLOAD_HOST", "PLAYWRIGHT_CHROMIUM_DOWNLOAD_HOST")
    )
    if not already_set and _is_china_locale():
        env["PLAYWRIGHT_DOWNLOAD_HOST"] = _CHINA_DOWNLOAD_HOST
    return env


def _is_china_locale() -> bool:
    import locale as sys_locale
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return (sys_locale.getdefaultlocale()[0] or "").lower().startswith("zh")
    except Exception:
        return False


def provision_browsers_background():
    """Make Chromium available without ever blocking the UI or hammering a
    doomed download every boot.

    Order: already installed → done; a bundled archive → extract it (no
    network); otherwise download once, honouring a China mirror default and
    a failure backoff. The launch fallback chain (system Chrome → Edge)
    covers the gap while a download runs.
    """
    import threading

    browsers_dir = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if not browsers_dir:
        return

    if _is_browser_installed(browsers_dir):
        logger.info("Chromium already installed in %s", browsers_dir)
        return

    archive = _bundled_browsers_archive()
    if archive is not None:
        _extract_bundled_browsers(archive, browsers_dir)
        if _is_browser_installed(browsers_dir):
            return
        logger.warning("Bundled browser archive did not yield a usable install; falling back to download")

    # Back off if a recent download already failed — do NOT re-fire the
    # multi-minute subprocess on every launch.
    status = _read_download_status(browsers_dir)
    if status.get("state") == "failed":
        last = status.get("ts", 0)
        try:
            elapsed = time.time() - float(last)
        except (TypeError, ValueError):
            elapsed = _DOWNLOAD_BACKOFF_S + 1
        if elapsed < _DOWNLOAD_BACKOFF_S:
            logger.info(
                "Skipping Chromium download — last attempt failed %.0f min ago "
                "(backing off; system Chrome/Edge still work). Error: %s",
                elapsed / 60, status.get("error", "unknown"),
            )
            return

    cli = _find_patchright_cli()
    if cli is None:
        logger.warning("Patchright CLI not found — skipping browser download")
        return

    driver_dir = os.path.dirname(os.path.dirname(cli))  # .../driver/
    node = os.path.join(driver_dir, "node")
    if not os.path.isfile(node):
        logger.warning("Patchright node binary not found at %s — skipping browser download", node)
        return

    def _download():
        env = _download_host_env(browsers_dir)
        host = env.get("PLAYWRIGHT_DOWNLOAD_HOST", "default CDN")
        logger.info("Background: downloading Chromium to %s (host: %s)", browsers_dir, host)
        _write_download_status(browsers_dir, state="downloading", ts=time.time())
        try:
            subprocess.run(
                [node, cli, "install", "chromium"],
                env=env,
                timeout=_DOWNLOAD_TIMEOUT_S,
                check=True,
                creationflags=win_no_window_flags(),
            )
            logger.info("Chromium installed successfully")
            _write_download_status(browsers_dir, state="ready", ts=time.time())
        except Exception as exc:
            logger.warning(
                "Chromium download failed (browser falls back to system "
                "Chrome/Edge; will back off before retrying): %s",
                exc,
            )
            _write_download_status(browsers_dir, state="failed", ts=time.time(), error=str(exc)[:300])

    thread = threading.Thread(target=_download, daemon=True)
    thread.start()


def _extract_bundled_browsers(archive: str, browsers_dir: str) -> None:
    """Extract the bundled browser archive into the writable browsers dir.

    Kept as an opaque archive rather than loose files under Resources so
    macOS hardened-runtime signing never touches the nested Chromium
    binaries — signing the bundled node's JIT away shipped the v0.6.6
    SIGTRAP regression, and Chromium has the same JIT need. tar.gz (not
    zip) so Chromium's relative framework symlinks and executable bits
    survive extraction.
    """
    import tarfile

    logger.info("Extracting bundled Chromium from %s", archive)
    try:
        os.makedirs(browsers_dir, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tf:
            # The archive is our own trusted build output; the 'data'
            # filter (Python 3.12+) still keeps internal relative symlinks
            # while rejecting any path that escapes the destination.
            try:
                tf.extractall(browsers_dir, filter="data")
            except TypeError:
                tf.extractall(browsers_dir)  # Python < 3.12
        logger.info("Bundled Chromium extracted to %s", browsers_dir)
    except Exception as exc:
        logger.warning("Failed to extract bundled Chromium (%s); will try downloading", exc)


# Back-compat alias — external callers/tests may still reference the old name.
download_browsers_background = provision_browsers_background


def run_migrations():
    ensure_data_dir()
    setup_browser_path()
    current = get_data_version()

    if current == 0:
        set_data_version(CURRENT_DATA_VERSION)
        logger.info("Fresh install — data version set to %d", CURRENT_DATA_VERSION)
        return

    if current >= CURRENT_DATA_VERSION:
        logger.info("Data version: %d (current)", current)
        return

    while current < CURRENT_DATA_VERSION:
        migration = MIGRATIONS.get(current)
        if migration is None:
            logger.error("No migration from version %d", current)
            break
        logger.info("Running migration from v%d to v%d", current, current + 1)
        migration()
        current += 1
        set_data_version(current)

    logger.info("Data version: %d", current)
