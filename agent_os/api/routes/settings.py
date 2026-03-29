# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Global settings endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/v2")

_settings_store = None
_credential_store = None


def configure(settings_store, credential_store=None):
    global _settings_store, _credential_store
    _settings_store = settings_store
    _credential_store = credential_store


class UpdateSettingsRequest(BaseModel):
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_sdk: str | None = None
    llm_provider: str | None = None
    llm_fallback_models: list[dict] | None = None
    user_preferences_content: str | None = None
    user_preferences_path: str | None = None
    scratch_workspace: str | None = None


class SetApiKeyRequest(BaseModel):
    api_key: str


@router.get("/settings")
async def get_settings():
    return _settings_store.get_masked()


@router.put("/settings")
async def update_settings(req: UpdateSettingsRequest):
    import os
    current = _settings_store.get()
    if req.llm_api_key is not None:
        # Redirect API key writes to credential store when available
        if _credential_store is not None:
            try:
                _credential_store.set_api_key(req.llm_api_key)
            except RuntimeError:
                # Fallback to JSON if keyring fails
                current.llm.api_key = req.llm_api_key
        else:
            current.llm.api_key = req.llm_api_key
    if req.llm_base_url is not None:
        current.llm.base_url = req.llm_base_url
    if req.llm_model is not None:
        current.llm.model = req.llm_model
    if req.llm_sdk is not None:
        current.llm.sdk = req.llm_sdk
    if req.llm_provider is not None:
        current.llm.provider = req.llm_provider
    if req.llm_fallback_models is not None:
        from agent_os.daemon_v2.settings_store import FallbackModelConfig
        current.llm.fallback_models = [
            FallbackModelConfig(**fb) for fb in req.llm_fallback_models
        ]
    if req.scratch_workspace is not None:
        current.scratch_workspace = req.scratch_workspace
    if req.user_preferences_path is not None:
        current.user_preferences_path = req.user_preferences_path

    # Write user preferences content to file
    if req.user_preferences_content is not None:
        prefs_path = current.user_preferences_path
        if not prefs_path:
            # Default path
            prefs_path = os.path.join(os.path.expanduser("~"), "orbital", "user_preferences.md")
            current.user_preferences_path = prefs_path
        os.makedirs(os.path.dirname(prefs_path), exist_ok=True)
        with open(prefs_path, "w", encoding="utf-8") as f:
            f.write(req.user_preferences_content)

    _settings_store.update(current)
    return _settings_store.get_masked()


@router.put("/settings/api-key")
async def set_api_key(req: SetApiKeyRequest):
    if _credential_store is None:
        raise HTTPException(status_code=501, detail="Credential store not available")
    try:
        result = _credential_store.set_api_key(req.api_key)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result


@router.delete("/settings/api-key")
async def delete_api_key():
    if _credential_store is None:
        raise HTTPException(status_code=501, detail="Credential store not available")
    return _credential_store.delete_api_key()


@router.get("/settings/api-key/status")
async def get_api_key_status():
    if _credential_store is None:
        # Fallback: check settings.json
        settings = _settings_store.get()
        configured = bool(settings.llm.api_key)
        return {"configured": configured, "source": "settings" if configured else "none"}
    source = _credential_store.get_source()
    return {"configured": source != "none", "source": source}
