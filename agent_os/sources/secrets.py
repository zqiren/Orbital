# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Source secrets in the OS keychain.

The same keyring the daemon's other credential stores use, under a service of
its own (``agent-os-source-creds``), one item per ``<source_id>.<field>``.
Deliberately not ``UserCredentialStore``: that one is listed by
``GET /credentials`` and resolvable through the agent's ``<secret:…>``
substitution, which would make a bot token agent-reachable (the same reason
``SubAgentCredentialStore`` keeps its own service).

Keyring calls can block (a headless Keychain prompt): callers run them off
the event loop.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import keyring

    _KEYRING_AVAILABLE = True
except Exception:  # pragma: no cover - keyring is a hard dependency
    keyring = None  # type: ignore[assignment]
    _KEYRING_AVAILABLE = False

SERVICE_NAME = "agent-os-source-creds"


class SecretStoreError(RuntimeError):
    """The keychain refused a write (no keyring, locked, verification)."""


class SourceSecretStore:
    def _item(self, source_id: str, field: str) -> str:
        return f"{source_id}.{field}"

    def get(self, source_id: str, field: str) -> str | None:
        if not _KEYRING_AVAILABLE:
            return None
        try:
            return keyring.get_password(SERVICE_NAME, self._item(source_id, field))
        except Exception:
            logger.warning("sources: keychain read failed for %s.%s", source_id, field)
            return None

    def set(self, source_id: str, field: str, value: str) -> None:
        if not _KEYRING_AVAILABLE:
            raise SecretStoreError("keyring not available")
        item = self._item(source_id, field)
        try:
            keyring.set_password(SERVICE_NAME, item, value)
        except Exception as exc:
            raise SecretStoreError(f"keychain write failed: {type(exc).__name__}") from None
        try:
            ok = keyring.get_password(SERVICE_NAME, item) == value
        except Exception:
            ok = False
        if not ok:
            raise SecretStoreError("keychain write could not be read back")

    def delete(self, source_id: str, field: str) -> None:
        if not _KEYRING_AVAILABLE:
            return
        try:
            keyring.delete_password(SERVICE_NAME, self._item(source_id, field))
        except Exception:
            pass


class MemorySecretStore(SourceSecretStore):
    """Tests: the same interface over a dict."""

    def __init__(self):
        self.items: dict[str, str] = {}

    def get(self, source_id, field):
        return self.items.get(self._item(source_id, field))

    def set(self, source_id, field, value):
        self.items[self._item(source_id, field)] = value

    def delete(self, source_id, field):
        self.items.pop(self._item(source_id, field), None)
