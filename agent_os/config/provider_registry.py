# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Typed access to provider and model metadata from providers.json.

Provides a single lookup point for model capabilities (vision, tool_use),
context windows, max output tokens, and display metadata. Complements
pricing.py (which handles cost rates separately).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

_PROVIDERS_JSON = os.path.join(os.path.dirname(__file__), "providers.json")

_FALLBACK_CONTEXT_WINDOW = 128_000
_FALLBACK_MAX_OUTPUT = 8192


@dataclass(frozen=True)
class ModelCapabilities:
    """What a model can do."""
    vision: bool = False
    tool_use: bool = True
    streaming: bool = True


@dataclass(frozen=True)
class ModelInfo:
    """Metadata for a specific model."""
    context_window: int = _FALLBACK_CONTEXT_WINDOW
    max_output: int = _FALLBACK_MAX_OUTPUT
    capabilities: ModelCapabilities = ModelCapabilities()
    tier: str = ""
    display_name: str = ""


def _parse_model_entry(entry: dict) -> ModelInfo:
    """Parse a model entry dict from providers.json into a ModelInfo."""
    caps_dict = entry.get("capabilities", {})
    caps = ModelCapabilities(
        vision=caps_dict.get("vision", False),
        tool_use=caps_dict.get("tool_use", True),
        streaming=caps_dict.get("streaming", True),
    )
    return ModelInfo(
        context_window=entry.get("context_window", _FALLBACK_CONTEXT_WINDOW),
        max_output=entry.get("max_output", _FALLBACK_MAX_OUTPUT),
        capabilities=caps,
        tier=entry.get("tier", ""),
        display_name=entry.get("display_name", ""),
    )


# Module-level default for unknown models
_UNKNOWN_MODEL = ModelInfo()


class ProviderRegistry:
    """Read-only registry for provider and model metadata.

    Loads providers.json once (lazy). All lookups use a fallback chain:
    1. Exact model match in provider's ``models`` dict
    2. Prefix match (longest matching key wins)
    3. Provider ``_default`` entry in ``models``
    4. Top-level ``defaults.unknown_model``
    5. Hardcoded safe fallback
    """

    def __init__(self, config_path: str | None = None):
        self._path = config_path or _PROVIDERS_JSON
        self._providers: dict | None = None
        self._defaults: dict | None = None

    def _load(self) -> None:
        if self._providers is not None:
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._providers = data.get("providers", {})
            self._defaults = data.get("defaults", {})
        except (OSError, json.JSONDecodeError):
            self._providers = {}
            self._defaults = {}

    def get_model_info(self, provider: str, model: str) -> ModelInfo:
        """Look up model metadata with fallback chain."""
        self._load()
        provider_data = self._providers.get(provider, {})
        models = provider_data.get("models", {})

        entry = None

        # 1. Exact match
        if model in models:
            entry = models[model]
        else:
            # 2. Prefix match (longest key that is a prefix of model)
            best = ""
            for key in models:
                if key.startswith("_"):
                    continue
                if model.startswith(key) and len(key) > len(best):
                    best = key
            if best:
                entry = models[best]

        # 3. Provider _default
        if entry is None:
            entry = models.get("_default")

        # 4. Top-level defaults.unknown_model
        if entry is None:
            unknown = (self._defaults or {}).get("unknown_model")
            if unknown:
                entry = unknown

        # 5. Hardcoded fallback
        if entry is None:
            return _UNKNOWN_MODEL

        return _parse_model_entry(entry)

    def get_capabilities(self, provider: str, model: str) -> ModelCapabilities:
        """Return capabilities for a model."""
        return self.get_model_info(provider, model).capabilities

    def get_max_output(self, provider: str, model: str) -> int:
        """Return max output tokens for a model."""
        return self.get_model_info(provider, model).max_output

    def get_context_window(self, provider: str, model: str) -> int:
        """Return context window size for a model."""
        return self.get_model_info(provider, model).context_window

    def get_provider_data(self, provider: str) -> dict:
        """Return raw provider dict from providers.json."""
        self._load()
        return self._providers.get(provider, {})

    def all_providers(self) -> dict:
        """Return the full providers dict (for the /providers API)."""
        self._load()
        return dict(self._providers)

    def suggested_models(self, provider: str) -> list[str]:
        """Return suggested model keys for a provider.

        Prefers the ``suggested_models`` field if present, otherwise
        returns non-default keys from the ``models`` dict.
        """
        self._load()
        provider_data = self._providers.get(provider, {})
        suggested = provider_data.get("suggested_models")
        if suggested:
            return list(suggested)
        models = provider_data.get("models", {})
        return [k for k in models if not k.startswith("_")]
