# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""OS keychain-backed API key storage with env var override."""

import json
import logging
import os
from datetime import datetime, timezone
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

try:
    import keyring

    _KEYRING_AVAILABLE = True
except Exception:
    keyring = None  # type: ignore[assignment]
    _KEYRING_AVAILABLE = False

_SERVICE_NAME = "agent-os"
_KEY_NAME = "llm-api-key"
_ENV_VAR = "AGENT_OS_API_KEY"

# Spec 082 — one Keychain item per credential card, ``card:<id>`` under the
# same ``agent-os`` service. ``llm-api-key`` (the single global slot every
# provider used to fight over) survives only as a migration source.
_CARD_ITEM_PREFIX = "card:"

# The synthetic, read-only card backed by AGENT_OS_API_KEY (spec 082 D11).
# It keeps ``PYTHON_KEYRING_BACKEND=in-memory + AGENT_OS_API_KEY`` headless
# runs working unchanged: the variable overrides the DEFAULT card's key and
# nothing else.
ENV_CARD_ID = "env"


def _card_item(card_id: str) -> str:
    return f"{_CARD_ITEM_PREFIX}{card_id}"


class CardKeyStore:
    """Per-card secret storage (spec 082 §3.1).

    One Keychain item per card id, so switching or deleting a card never
    disturbs another card's key — the failure mode of the single
    ``llm-api-key`` slot this replaces. Metadata (provider, model, health)
    lives in ``settings.json``; only the raw key is here.

    ``AGENT_OS_API_KEY`` resolves for the synthetic ``env`` card and for
    nothing else: a per-project card must not silently answer with the
    environment's key.
    """

    def get(self, card_id: str) -> str | None:
        """Return the key for ``card_id``, or None."""
        if not card_id:
            return None
        if card_id == ENV_CARD_ID:
            return os.environ.get(_ENV_VAR) or None
        if not _KEYRING_AVAILABLE:
            return None
        try:
            return keyring.get_password(_SERVICE_NAME, _card_item(card_id))
        except Exception:
            logger.warning(
                "keyring.get_password failed for card %s", card_id, exc_info=True,
            )
            return None

    def set(self, card_id: str, key: str) -> dict:
        """Store ``key`` for ``card_id``. An empty key REMOVES the item.

        An empty key is legal: keyless local servers (Custom provider against
        LM Studio / llama.cpp / Ollama) are cards like any other.
        """
        if not card_id:
            raise ValueError("card id must be non-empty")
        if card_id == ENV_CARD_ID:
            raise ValueError("the environment card is read-only")
        if not key:
            self.delete(card_id)
            return {"source": "none"}
        if not _KEYRING_AVAILABLE:
            raise RuntimeError("keyring package not available")
        try:
            keyring.set_password(_SERVICE_NAME, _card_item(card_id), key)
        except Exception as exc:
            raise RuntimeError(f"keyring.set_password failed: {exc}") from exc
        if keyring.get_password(_SERVICE_NAME, _card_item(card_id)) != key:
            raise RuntimeError(
                "Keychain write verification failed: stored value does not match"
            )
        return {"source": "keychain"}

    def delete(self, card_id: str) -> None:
        """Remove ``card_id``'s item. Absent items are a no-op."""
        if not card_id or card_id == ENV_CARD_ID or not _KEYRING_AVAILABLE:
            return
        try:
            keyring.delete_password(_SERVICE_NAME, _card_item(card_id))
        except Exception:
            logger.warning(
                "keyring.delete_password failed for card %s", card_id, exc_info=True,
            )

    def source(self, card_id: str) -> str:
        """Return 'environment', 'keychain', or 'none' for ``card_id``."""
        if card_id == ENV_CARD_ID:
            return "environment" if os.environ.get(_ENV_VAR) else "none"
        return "keychain" if self.get(card_id) else "none"

    # -- legacy single-slot access (migration only) ---------------------

    def get_legacy(self) -> str | None:
        """Return the pre-cards ``llm-api-key`` item, or None.

        Deliberately does NOT consult ``AGENT_OS_API_KEY``: the migration
        must copy the KEYCHAIN's key into card G, never the environment's
        (which stays a live override and is not ours to persist).
        """
        if not _KEYRING_AVAILABLE:
            return None
        try:
            return keyring.get_password(_SERVICE_NAME, _KEY_NAME)
        except Exception:
            logger.warning("keyring.get_password failed (legacy slot)", exc_info=True)
            return None

    def delete_legacy(self) -> None:
        """Remove the pre-cards ``llm-api-key`` item."""
        if not _KEYRING_AVAILABLE:
            return
        try:
            keyring.delete_password(_SERVICE_NAME, _KEY_NAME)
        except Exception:
            logger.warning("keyring.delete_password failed (legacy slot)", exc_info=True)


class ApiKeyStore:
    """The DEFAULT credential card's key, behind the pre-cards API.

    Spec 082 §3.8: ``PUT/DELETE /settings/api-key``, ``GET
    /settings/api-key/status`` and the ``llm_api_key`` field of ``PUT
    /settings`` keep working for one release (the first-run wizard and any
    older SPA against a newer daemon). They all land here, and here they mean
    "the default card".

    Binding is done by ``SettingsStore.__init__`` — it already receives the
    credential store, so no wiring site has to know about the back-reference.
    UNBOUND (a bare ``ApiKeyStore()``, as in unit tests and
    ``desktop/main.py``'s sign-out) it keeps the pre-cards single-slot
    behaviour verbatim.
    """

    def __init__(self):
        self._settings_store = None

    def bind_settings_store(self, settings_store) -> None:
        """Point this shim at the store that owns the cards."""
        self._settings_store = settings_store

    # -- shimmed single-slot API ---------------------------------------

    def get_api_key(self) -> str | None:
        """Return the default card's key (env var wins), or None."""
        env_val = os.environ.get(_ENV_VAR)
        if env_val:
            return env_val
        ss = self._settings_store
        if ss is not None:
            card = ss.default_card()
            return (ss.key_for(card.id) or None) if card is not None else None
        if _KEYRING_AVAILABLE:
            try:
                return keyring.get_password(_SERVICE_NAME, _KEY_NAME)
            except Exception:
                logger.warning("keyring.get_password failed", exc_info=True)
        return None

    def set_api_key(self, key: str) -> dict:
        """Store ``key`` on the default card. No-op if the env var is set."""
        if not key or not key.strip():
            raise ValueError("API key must be non-empty")
        if os.environ.get(_ENV_VAR):
            return {"source": "environment"}
        ss = self._settings_store
        if ss is not None:
            card = ss.ensure_default_card()
            return {**ss.set_card_key(card.id, key), "card_id": card.id}
        if not _KEYRING_AVAILABLE:
            raise RuntimeError("keyring package not available")
        try:
            keyring.set_password(_SERVICE_NAME, _KEY_NAME, key)
        except Exception as exc:
            raise RuntimeError(f"keyring.set_password failed: {exc}") from exc
        stored = keyring.get_password(_SERVICE_NAME, _KEY_NAME)
        if stored != key:
            raise RuntimeError(
                "Keychain write verification failed: stored value does not match"
            )
        return {"source": "keychain"}

    def delete_api_key(self) -> dict:
        """Remove the default card's key. No-op if the env var is set."""
        if os.environ.get(_ENV_VAR):
            return {"source": "environment"}
        ss = self._settings_store
        if ss is not None:
            card = ss.stored_default_card()
            if card is None:
                return {"source": "none", "card_id": None}
            ss.set_card_key(card.id, "")
            return {"source": "none", "card_id": card.id}
        if not _KEYRING_AVAILABLE:
            return {"source": "none"}
        try:
            keyring.delete_password(_SERVICE_NAME, _KEY_NAME)
        except Exception:
            logger.warning("keyring.delete_password failed", exc_info=True)
        return {"source": "none"}

    def get_source(self) -> str:
        """Return 'environment', 'keychain', or 'none'."""
        if os.environ.get(_ENV_VAR):
            return "environment"
        ss = self._settings_store
        if ss is not None:
            card = ss.default_card()
            if card is None:
                return "none"
            return "keychain" if ss.key_for(card.id) else "none"
        if _KEYRING_AVAILABLE:
            try:
                if keyring.get_password(_SERVICE_NAME, _KEY_NAME):
                    return "keychain"
            except Exception:
                pass
        return "none"


_SUBAGENT_SERVICE_NAME = "agent-os-subagent-creds"


class SubAgentCredentialStore:
    """Keychain storage for sub-agent API credentials, keyed by credential key.

    Sub-agents normally keep their own credentials (``~/.claude/``,
    ``~/.codex/``). Agents that ship no credential store of their own are the
    documented exception: their key is held here and injected into the spawn
    env by ``SetupEngine`` (which calls ``get(cred.key)``).

    Deliberately NOT ``UserCredentialStore``: that one is enumerated by
    ``GET /credentials`` and resolvable through the agent's ``<secret:...>``
    substitution, which would make sub-agent keys agent-reachable.

    Keychain-only, no JSON fallback. A machine without a keyring configures
    these through environment variables instead — a path ``SetupEngine``'s
    resolution chain already covers.
    """

    def get(self, key: str) -> str | None:
        """Return the stored value for ``key``, or None."""
        if not key or not _KEYRING_AVAILABLE:
            return None
        try:
            return keyring.get_password(_SUBAGENT_SERVICE_NAME, key)
        except Exception:
            logger.warning(
                "keyring.get_password failed for sub-agent credential %s",
                key, exc_info=True,
            )
            return None

    def set(self, key: str, value: str) -> None:
        """Store ``value`` under ``key`` in the OS keychain."""
        if not key or not key.strip():
            raise ValueError("credential key must be non-empty")
        if not value or not value.strip():
            raise ValueError("credential value must be non-empty")
        if not _KEYRING_AVAILABLE:
            raise RuntimeError("keyring package not available")
        try:
            keyring.set_password(_SUBAGENT_SERVICE_NAME, key, value)
        except Exception as exc:
            raise RuntimeError(f"keyring.set_password failed: {exc}") from exc
        if keyring.get_password(_SUBAGENT_SERVICE_NAME, key) != value:
            raise RuntimeError(
                "Keychain write verification failed: stored value does not match"
            )

    def delete(self, key: str) -> None:
        """Remove ``key`` from the OS keychain. Absent keys are a no-op."""
        if not key or not _KEYRING_AVAILABLE:
            return
        try:
            keyring.delete_password(_SUBAGENT_SERVICE_NAME, key)
        except Exception:
            logger.warning(
                "keyring.delete_password failed for sub-agent credential %s",
                key, exc_info=True,
            )


_CRED_SERVICE_NAME = "agent-os-creds"


class UserCredentialStore:
    """Manage user website credentials via OS keychain with metadata tracking.

    Values are stored in the OS keychain (encrypted at rest).
    Metadata (names, domains, fields, usage stats -- no values) in a JSON file.
    """

    def __init__(self, meta_path: str | None = None):
        self._meta_path = meta_path or os.path.join(
            os.path.expanduser("~"), "orbital", "credential-meta.json"
        )
        self._meta = self._load_meta()

    def _load_meta(self) -> dict:
        if os.path.exists(self._meta_path):
            try:
                with open(self._meta_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning("Failed to load credential metadata, starting fresh")
        return {}

    def _save_meta(self) -> None:
        os.makedirs(os.path.dirname(self._meta_path), exist_ok=True)
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(self._meta, f, indent=2)

    def store(self, name: str, domain: str, fields: dict[str, str]) -> None:
        """Store all field values in keychain + update metadata."""
        if not _KEYRING_AVAILABLE:
            raise RuntimeError("keyring package not available")
        for field_name, value in fields.items():
            keyring.set_password(_CRED_SERVICE_NAME, f"{name}.{field_name}", value)
        self._meta[name] = {
            "domain": domain,
            "fields": sorted(fields.keys()),
            "created": datetime.now(timezone.utc).isoformat(),
            "use_count": 0,
            "last_used": None,
        }
        self._save_meta()

    def get_value(self, name: str, field: str) -> str | None:
        """Retrieve single field value from keychain."""
        if not _KEYRING_AVAILABLE:
            return None
        try:
            return keyring.get_password(_CRED_SERVICE_NAME, f"{name}.{field}")
        except Exception:
            logger.warning("keyring.get_password failed for %s.%s", name, field)
            return None

    def check_domain(self, name: str, page_url: str) -> bool:
        """Check credential's domain against actual page URL."""
        meta = self._meta.get(name)
        if meta is None:
            return False
        return self._domain_matches(meta["domain"], page_url)

    def _domain_matches(self, credential_domain: str, page_url: str) -> bool:
        page_hostname = urlparse(page_url).hostname
        if page_hostname is None:
            return False
        page_hostname = page_hostname.lower().rstrip(".")
        cred_domain = credential_domain.lower().rstrip(".")
        return page_hostname == cred_domain or page_hostname.endswith("." + cred_domain)

    def get_metadata(self, name: str) -> dict | None:
        """Get metadata (no values): {domain, fields, created, use_count, last_used}."""
        return self._meta.get(name)

    def list_all(self) -> list[dict]:
        """List all credential metadata for settings UI."""
        return [{"name": name, **meta} for name, meta in self._meta.items()]

    def record_use(self, name: str) -> None:
        """Increment use counter, update last_used."""
        meta = self._meta.get(name)
        if meta is None:
            return
        meta["use_count"] = meta.get("use_count", 0) + 1
        meta["last_used"] = datetime.now(timezone.utc).isoformat()
        self._save_meta()

    def delete(self, name: str) -> None:
        """Delete from keychain + metadata."""
        meta = self._meta.get(name)
        if meta is None:
            return
        if _KEYRING_AVAILABLE:
            for field_name in meta.get("fields", []):
                try:
                    keyring.delete_password(_CRED_SERVICE_NAME, f"{name}.{field_name}")
                except Exception:
                    logger.warning("Failed to delete %s.%s from keyring", name, field_name)
        del self._meta[name]
        self._save_meta()
