# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import json
import os
import subprocess
import sys
import logging

logger = logging.getLogger(__name__)

CURRENT_DATA_VERSION = 1


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
    actual Chromium download happens in the background after the daemon
    is up via ``download_browsers_background()``.
    """
    browsers_dir = os.path.join(DATA_DIR, "browsers")
    os.makedirs(browsers_dir, exist_ok=True)
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = browsers_dir


def download_browsers_background():
    """Download Chromium in a background thread.

    Meant to be called AFTER the daemon is running so the UI is
    responsive immediately.  The browser fallback chain (system
    Chrome → Edge → WebKit on macOS) covers the gap until the
    download finishes.
    """
    import threading

    browsers_dir = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if not browsers_dir:
        return

    # Skip if browsers are already downloaded
    try:
        if any(entry.startswith("chromium") for entry in os.listdir(browsers_dir)):
            logger.info("Browsers already installed in %s", browsers_dir)
            return
    except OSError:
        pass

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
        logger.info("Background: downloading Chromium to %s", browsers_dir)
        try:
            subprocess.run(
                [node, cli, "install", "chromium"],
                env={**os.environ, "PLAYWRIGHT_BROWSERS_PATH": browsers_dir},
                timeout=300,
                check=True,
            )
            logger.info("Chromium installed successfully")
        except Exception as exc:
            logger.warning(
                "Chromium download failed (browser will fall back to "
                "system Chrome/Edge or WebKit on macOS): %s",
                exc,
            )

    thread = threading.Thread(target=_download, daemon=True)
    thread.start()


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
