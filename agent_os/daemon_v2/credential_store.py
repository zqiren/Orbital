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


class ApiKeyStore:
    """Manage LLM API key with env-var override > OS keychain > none."""

    def get_api_key(self) -> str | None:
        """Return API key from env var or keychain, or None."""
        env_val = os.environ.get(_ENV_VAR)
        if env_val:
            return env_val
        if _KEYRING_AVAILABLE:
            try:
                return keyring.get_password(_SERVICE_NAME, _KEY_NAME)
            except Exception:
                logger.warning("keyring.get_password failed", exc_info=True)
        return None

    def set_api_key(self, key: str) -> dict:
        """Store key in OS keychain. No-op if env var is set."""
        if not key or not key.strip():
            raise ValueError("API key must be non-empty")
        if os.environ.get(_ENV_VAR):
            return {"source": "environment"}
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
        """Remove key from OS keychain. No-op if env var is set."""
        if os.environ.get(_ENV_VAR):
            return {"source": "environment"}
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
        if _KEYRING_AVAILABLE:
            try:
                if keyring.get_password(_SERVICE_NAME, _KEY_NAME):
                    return "keychain"
            except Exception:
                pass
        return "none"


_CRED_SERVICE_NAME = "agent-os-creds"


class UserCredentialStore:
    """Manage user website credentials via OS keychain with metadata tracking.

    Values are stored in the OS keychain (encrypted at rest).
    Metadata (names, domains, fields, usage stats -- no values) in a JSON file.
    """

    def __init__(self, meta_path: str | None = None):
        self._meta_path = meta_path or os.path.join(
            os.path.expanduser("~"), ".agent-os", "credential-meta.json"
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
