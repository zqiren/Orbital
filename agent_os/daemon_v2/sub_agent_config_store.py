# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Daemon-level sub-agent configuration store.

Persists per-sub-agent CLI parameter overrides (model, effort/thinking level,
permission mode, etc.) to ``~/.orbital/sub_agent_config.json``. Read at
sub-agent dispatch time; merged into the CLI invocation as appropriate flags.

Validation is intentionally narrow: only the "stable parameters" documented
in the sub-agent findings doc are accepted. Experimental flags are out of
scope; users wanting them can pass them via env or the CLI's own settings
files. See SCHEMA below for the canonical list.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Per-sub-agent stable parameter schema. Each entry maps a slug to a dict of
# parameter names; each parameter has an ``allowed`` list (None = free-text,
# validated as non-empty string) and a ``flag_template``. The flag_template
# is a Python format string with ``{value}`` placeholder; if it produces a
# string with a space, it is split into multiple argv tokens.
@dataclass(frozen=True)
class _ParamSchema:
    name: str
    allowed: tuple[str, ...] | None  # None = free-text
    flag_template: str  # e.g. "--model {value}"


# Authoritative tier-5 stable parameters for each sub-agent. Anything not on
# this list is rejected by ``validate()`` (returns 400 from the API).
SCHEMA: dict[str, dict[str, _ParamSchema]] = {
    "claude-code": {
        "model": _ParamSchema(
            name="model",
            allowed=(
                "haiku",
                "sonnet",
                "opus",
                "claude-sonnet-4-6",
                "claude-opus-4-7",
                "claude-haiku-4-5-20251001",
            ),
            flag_template="--model {value}",
        ),
        "effort": _ParamSchema(
            name="effort",
            allowed=("low", "medium", "high", "xhigh", "max"),
            flag_template="--effort {value}",
        ),
        "permission-mode": _ParamSchema(
            name="permission-mode",
            allowed=(
                "acceptEdits",
                "auto",
                "bypassPermissions",
                "default",
                "dontAsk",
                "plan",
            ),
            flag_template="--permission-mode {value}",
        ),
    },
    "codex": {
        # Codex model IDs evolve too quickly to enumerate; accept any non-empty
        # string. Other tunables go via -c key=value TOML config — too broad
        # for a v1 dropdown, so they are not surfaced here.
        "model": _ParamSchema(
            name="model",
            allowed=None,
            flag_template="-m {value}",
        ),
    },
    "gemini-cli": {
        # gemini-cli may not be installed today; this entry exists so the UI
        # can render the right dropdowns the moment it shows up in the
        # registry.
        "model": _ParamSchema(
            name="model",
            allowed=None,
            flag_template="--model {value}",
        ),
        "approval-mode": _ParamSchema(
            name="approval-mode",
            allowed=("default", "auto_edit", "plan", "yolo"),
            flag_template="--approval-mode {value}",
        ),
    },
}


class SubAgentConfigError(ValueError):
    """Raised when a config payload fails validation."""


class SubAgentConfigStore:
    """Read/write daemon-level sub-agent configuration.

    Storage: a single JSON file. Map of ``slug -> {param: value}``. Empty
    dict for a slug means "no overrides; use the CLI's own defaults".
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, str]] = {}
        self._load()

    @property
    def path(self) -> str:
        return self._path

    def _load(self) -> None:
        if not os.path.exists(self._path):
            self._data = {}
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                # Coerce values to dict[str, str]; drop anything malformed.
                cleaned: dict[str, dict[str, str]] = {}
                for slug, params in raw.items():
                    if isinstance(params, dict):
                        cleaned[slug] = {
                            str(k): str(v) for k, v in params.items()
                            if isinstance(v, (str, int, float, bool))
                        }
                self._data = cleaned
            else:
                logger.warning(
                    "sub_agent_config.json was not a dict; ignoring (%s)",
                    self._path,
                )
                self._data = {}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load %s: %s", self._path, exc)
            self._data = {}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        tmp = self._path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self._path)

    def get(self, slug: str) -> dict[str, str]:
        """Return the persisted config for ``slug``, or empty dict."""
        with self._lock:
            return dict(self._data.get(slug, {}))

    def all(self) -> dict[str, dict[str, str]]:
        """Return a snapshot of every persisted slug's config."""
        with self._lock:
            return {k: dict(v) for k, v in self._data.items()}

    def set(self, slug: str, params: dict[str, str]) -> dict[str, str]:
        """Validate and persist ``params`` for ``slug``.

        Only keys listed in :data:`SCHEMA` for the slug are accepted. Pass an
        empty dict to clear all overrides for the slug.

        Raises :class:`SubAgentConfigError` on any invalid key/value.
        """
        validated = self.validate(slug, params)
        with self._lock:
            if validated:
                self._data[slug] = validated
            else:
                self._data.pop(slug, None)
            self._save()
        return dict(validated)

    @staticmethod
    def validate(slug: str, params: dict[str, str]) -> dict[str, str]:
        """Return a cleaned copy of ``params`` or raise SubAgentConfigError.

        - Unknown slug: rejects any non-empty params (empty is fine; no-op).
        - Unknown param: rejected.
        - Param with ``allowed`` whitelist: value must be in the list.
        - Param with ``allowed=None``: value must be a non-empty string.
        """
        if not isinstance(params, dict):
            raise SubAgentConfigError("params must be an object")
        slug_schema = SCHEMA.get(slug)
        if slug_schema is None:
            if not params:
                return {}
            raise SubAgentConfigError(
                f"sub-agent '{slug}' has no configurable parameters"
            )
        cleaned: dict[str, str] = {}
        for key, raw_value in params.items():
            if key not in slug_schema:
                raise SubAgentConfigError(
                    f"unknown parameter '{key}' for sub-agent '{slug}'"
                )
            value = "" if raw_value is None else str(raw_value).strip()
            if value == "":
                # Empty string clears the param; do not persist.
                continue
            schema = slug_schema[key]
            if schema.allowed is not None and value not in schema.allowed:
                raise SubAgentConfigError(
                    f"invalid value for {slug}.{key}: '{value}'. "
                    f"Allowed: {list(schema.allowed)}"
                )
            cleaned[key] = value
        return cleaned

    def build_extra_args(self, slug: str) -> list[str]:
        """Return extra argv tokens for the CLI invocation, from persisted config.

        Reads ``self.get(slug)`` and renders each param via its schema's
        ``flag_template``, splitting on whitespace so each token is a
        separate argv element.
        """
        params = self.get(slug)
        if not params:
            return []
        slug_schema = SCHEMA.get(slug, {})
        argv: list[str] = []
        for key, value in params.items():
            schema = slug_schema.get(key)
            if schema is None:
                continue
            argv.extend(schema.flag_template.format(value=value).split())
        return argv

    def schema_for(self, slug: str) -> dict[str, dict[str, object]]:
        """Return the parameter schema for ``slug`` in JSON-friendly shape.

        Used by ``GET /api/v2/settings/sub-agents`` to populate UI dropdowns.
        """
        slug_schema = SCHEMA.get(slug, {})
        return {
            name: {
                "allowed": list(s.allowed) if s.allowed is not None else None,
            }
            for name, s in slug_schema.items()
        }
