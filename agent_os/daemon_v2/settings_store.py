# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Global settings store. Persists to {data_dir}/settings.json."""

import json
import os
from pydantic import BaseModel


class FallbackModelConfig(BaseModel):
    """Single fallback model entry in global settings."""
    provider: str = "custom"
    model: str = ""
    base_url: str | None = None
    api_key: str | None = None
    sdk: str = "openai"


class GlobalLLMSettings(BaseModel):
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    sdk: str = "openai"
    provider: str = "custom"
    fallback_models: list[FallbackModelConfig] = []


class GlobalSettings(BaseModel):
    llm: GlobalLLMSettings = GlobalLLMSettings()
    scratch_workspace: str | None = None
    user_preferences_path: str | None = None


class SettingsStore:
    """Read/write global settings from a JSON file."""

    def __init__(self, data_dir: str, credential_store=None):
        self._path = os.path.join(data_dir, "settings.json")
        self._credential_store = credential_store

    def get(self) -> GlobalSettings:
        if not os.path.exists(self._path):
            return GlobalSettings()
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return GlobalSettings(**data)
        except (json.JSONDecodeError, Exception):
            return GlobalSettings()

    def update(self, settings: GlobalSettings) -> None:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(settings.model_dump(), f, indent=2)

    def get_masked(self) -> dict:
        """Return settings with api_key masked for frontend display."""
        settings = self.get()
        data = settings.model_dump()
        # Prefer credential store for API key status when available
        if self._credential_store is not None:
            key = self._credential_store.get_api_key()
            source = self._credential_store.get_source()
            if key:
                data["llm"]["api_key_masked"] = key[:4] + "..." + key[-4:] if len(key) > 8 else "****"
                data["llm"]["api_key_set"] = True
                data["llm"]["api_key_source"] = source
            else:
                data["llm"]["api_key_masked"] = ""
                data["llm"]["api_key_set"] = False
                data["llm"]["api_key_source"] = "none"
        elif data["llm"]["api_key"]:
            key = data["llm"]["api_key"]
            data["llm"]["api_key_masked"] = key[:4] + "..." + key[-4:] if len(key) > 8 else "****"
            data["llm"]["api_key_set"] = True
        else:
            data["llm"]["api_key_masked"] = ""
            data["llm"]["api_key_set"] = False
        del data["llm"]["api_key"]

        # Mask fallback model API keys
        for fb in data["llm"].get("fallback_models", []):
            key = fb.get("api_key") or ""
            if key and len(key) > 8:
                fb["api_key"] = key[:4] + "..." + key[-4:]
            elif key:
                fb["api_key"] = "****"
            else:
                fb["api_key"] = None

        # Read user preferences file content
        prefs_path = data.get("user_preferences_path")
        if prefs_path and os.path.exists(prefs_path):
            try:
                with open(prefs_path, "r", encoding="utf-8") as f:
                    data["user_preferences_content"] = f.read()
            except OSError:
                data["user_preferences_content"] = ""
        else:
            data["user_preferences_content"] = ""

        return data
