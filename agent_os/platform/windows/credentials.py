# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import logging

logger = logging.getLogger("agent_os.platform.windows.credentials")

SERVICE_NAME = "AgentOS"

try:
    import keyring
    import keyring.errors

    _keyring_available = True
except ImportError:
    _keyring_available = False


def _require_keyring() -> None:
    if not _keyring_available:
        raise RuntimeError("keyring package required: pip install keyring")


class CredentialStore:
    """Stores and retrieves secrets using Windows Credential Manager via keyring."""

    def store(self, key: str, value: str) -> None:
        _require_keyring()
        keyring.set_password(SERVICE_NAME, key, value)
        logger.info("Stored credential: %s", key)

    def retrieve(self, key: str) -> str | None:
        _require_keyring()
        result = keyring.get_password(SERVICE_NAME, key)
        if result is None:
            logger.warning("Credential not found: %s", key)
        return result

    def delete(self, key: str) -> bool:
        _require_keyring()
        try:
            keyring.delete_password(SERVICE_NAME, key)
            logger.info("Deleted credential: %s", key)
            return True
        except keyring.errors.PasswordDeleteError:
            logger.warning("Credential not found for deletion: %s", key)
            return False

    def exists(self, key: str) -> bool:
        _require_keyring()
        return keyring.get_password(SERVICE_NAME, key) is not None
