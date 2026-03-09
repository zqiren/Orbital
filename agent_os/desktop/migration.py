# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import json
import os
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


def run_migrations():
    ensure_data_dir()
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
