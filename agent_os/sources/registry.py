# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Adapter discovery: ``agent_os.sources.adapters.<type>`` exporting
``ADAPTER`` (and optionally ``DESCRIPTOR``). A type whose module is absent or
fails to import is reported unavailable, never an error."""

from __future__ import annotations

import importlib
import logging

from agent_os.sources import catalog

logger = logging.getLogger(__name__)


class AdapterRegistry:
    def __init__(self, overrides: dict | None = None):
        # type -> (adapter instance | None, descriptor override | None)
        self._cache: dict[str, tuple] = {}
        self._overrides = dict(overrides or {})

    def _load(self, type_: str) -> tuple:
        if type_ in self._cache:
            return self._cache[type_]
        if type_ in self._overrides:
            adapter = self._overrides[type_]
            result = (adapter, getattr(adapter, "DESCRIPTOR", None))
        elif type_ not in catalog.ALL_TYPES:
            result = (None, None)
        else:
            try:
                mod = importlib.import_module(f"agent_os.sources.adapters.{type_}")
            except ModuleNotFoundError as e:
                if e.name != f"agent_os.sources.adapters.{type_}":
                    logger.warning("sources: adapter %s failed to import (%s)", type_, e)
                result = (None, None)
            except Exception:
                logger.warning("sources: adapter %s failed to import", type_, exc_info=True)
                result = (None, None)
            else:
                cls = getattr(mod, "ADAPTER", None)
                adapter = cls() if isinstance(cls, type) else cls
                result = (adapter, getattr(mod, "DESCRIPTOR", None))
        self._cache[type_] = result
        return result

    def adapter(self, type_: str):
        """The adapter for ``type_`` if it exists and runs on this platform."""
        adapter, _ = self._load(type_)
        if adapter is None:
            return None
        if self.unavailable_reason(type_) is not None:
            return None
        return adapter

    def entry(self, type_: str) -> dict | None:
        _, override = self._load(type_)
        return catalog.merged_entry(type_, override)

    def unavailable_reason(self, type_: str) -> str | None:
        adapter, override = self._load(type_)
        entry = catalog.merged_entry(type_, override)
        if entry is None:
            return "not_built"
        platforms = getattr(adapter, "platforms", None) or entry.get("platforms") or catalog.ALL_PLATFORMS
        if catalog.current_platform() not in platforms:
            return "platform"
        if adapter is None:
            return "not_built"
        return None

    def catalogue(self) -> list[dict]:
        out = []
        for type_ in catalog.ALL_TYPES:
            entry = self.entry(type_)
            if entry is None:
                continue
            reason = self.unavailable_reason(type_)
            adapter, _ = self._load(type_)
            if adapter is not None and getattr(adapter, "interval_s", None):
                entry["interval_s"] = adapter.interval_s
            entry["available"] = reason is None
            entry["unavailable_reason"] = reason
            out.append(entry)
        return out
