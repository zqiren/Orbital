# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Global settings endpoints."""

import asyncio
import logging
import os
import subprocess
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agent_os import telemetry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2")

_settings_store = None
_credential_store = None
_setup_engine = None
_sub_agent_config_store = None
_ws_manager = None
# Tracks in-flight login subprocesses keyed by job_id, so callers can poll
# WS events for ``login.progress`` / ``login.complete`` / ``login.failed``.
_login_jobs: dict[str, dict] = {}


def configure(settings_store, credential_store=None, setup_engine=None,
              sub_agent_config_store=None, ws_manager=None):
    global _settings_store, _credential_store, _setup_engine
    global _sub_agent_config_store, _ws_manager
    _settings_store = settings_store
    _credential_store = credential_store
    _setup_engine = setup_engine
    _sub_agent_config_store = sub_agent_config_store
    _ws_manager = ws_manager


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
    telemetry_enabled: bool | None = None


class SetApiKeyRequest(BaseModel):
    api_key: str


@router.get("/settings")
async def get_settings():
    return _settings_store.get_masked()


@router.put("/settings")
async def update_settings(req: UpdateSettingsRequest):
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
    if req.telemetry_enabled is not None:
        current.telemetry_enabled = req.telemetry_enabled

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
    telemetry.emit("key_set", {"provider": _settings_store.get().llm.provider})
    telemetry.latch("key_set")
    return result


@router.delete("/settings/api-key")
async def delete_api_key():
    if _credential_store is None:
        raise HTTPException(status_code=501, detail="Credential store not available")
    return _credential_store.delete_api_key()


@router.get("/settings/telemetry-payload")
async def get_telemetry_payload():
    """The verbatim telemetry pings for the settings viewer (spec 046 §6):
    exactly what was last transmitted and exactly what the next send would
    transmit. This transparency surface is the load-bearing trust feature
    behind the default-on consent model."""
    sender = telemetry.get_sender()
    if sender is None:
        return {"last_sent": None, "next_pending": None}
    return {
        "last_sent": sender.last_sent_payload(),
        "next_pending": sender.next_pending_payload(),
    }


@router.get("/settings/api-key/status")
async def get_api_key_status():
    if _credential_store is None:
        # Fallback: check settings.json
        settings = _settings_store.get()
        configured = bool(settings.llm.api_key)
        return {"configured": configured, "source": "settings" if configured else "none"}
    source = _credential_store.get_source()
    return {"configured": source != "none", "source": source}


# ---------------------------------------------------------------------------
# Sub-agent settings (daemon-level, global)
#
# Single write surface for managing sub-agents at the daemon level: install
# status, auth status, model/effort/permission-mode overrides, and login
# trigger. Tokens stay in the CLI's own location (``~/.claude/``,
# ``~/.codex/``, etc.) — Orbital never stores sub-agent credentials.
# ---------------------------------------------------------------------------


def _supports_login(slug: str) -> bool:
    """Whether this sub-agent exposes an interactive login (OAuth) flow.

    True when the agent's manifest declares a credential with a non-empty
    ``setup_command`` (e.g. ``claude login`` / ``codex login``). Agents that
    only accept an API key — and therefore have no login subcommand — return
    False so the frontend can hide a Login button that would otherwise 400.
    """
    if _setup_engine is None:
        return False
    registry = getattr(_setup_engine, "_registry", None)
    if registry is None:
        return False
    manifest = registry.get(slug)
    if manifest is None:
        return False
    return any(c.setup_command for c in manifest.setup.credentials)


def _build_sub_agent_status_entry(s) -> dict:
    """Convert an AgentSetupStatus into the wire shape this page expects."""
    config = {}
    schema = {}
    if _sub_agent_config_store is not None:
        try:
            config = _sub_agent_config_store.get(s.slug)
            schema = _sub_agent_config_store.schema_for(s.slug)
        except Exception:
            logger.warning("Failed to read sub-agent config for %s",
                           s.slug, exc_info=True)
    return {
        "slug": s.slug,
        "name": s.name,
        "installed": s.installed,
        "binary_path": s.binary_path,
        "version": s.version,
        "ready": s.installed and s.dependencies_met and s.credentials_configured,
        "dependencies_met": s.dependencies_met,
        "missing_dependencies": s.missing_dependencies,
        "credentials_configured": s.credentials_configured,
        "missing_credentials": s.missing_credentials,
        "supports_login": _supports_login(s.slug),
        "setup_actions": [
            {"action": a.action, "label": a.label, "command": a.command,
             "blocking": a.blocking}
            for a in s.setup_actions
        ],
        "config": config,
        "param_schema": schema,
    }


async def _codex_live_models(binary: str | None = None) -> list[str] | None:
    """Indirection over the cached codex model/list probe (monkeypatch seam
    for route tests). None means "list unavailable" — callers degrade to the
    static schema (free-text model entry)."""
    from agent_os.agent.transports.codex_models import get_codex_models_cached
    return await get_codex_models_cached(binary or "codex")


async def _augment_codex_live_models(entries: list[dict]) -> list[dict]:
    """Replace the codex entry's free-text model schema with the account's
    LIVE model list when available (TASK-live-model-config).

    Codex model IDs are gated per account+client — a free-text override the
    gate rejects (e.g. `gpt-5.6` from Codex desktop) 400s on every dispatch.
    The live list turns the settings field into a dropdown of models that
    actually work; probe failure leaves the static schema untouched.
    """
    for entry in entries:
        if entry.get("slug") != "codex":
            continue
        model_schema = (entry.get("param_schema") or {}).get("model")
        if model_schema is None:
            continue
        models = await _codex_live_models(entry.get("binary_path"))
        if models:
            model_schema["allowed"] = models
    return entries


@router.get("/settings/sub-agents")
async def list_sub_agent_settings():
    """List all sub-agents with install status, auth status, and current config.

    Response is one entry per registered agent. Built-in agents are excluded
    because they have no installable binary and no daemon-level config.
    """
    if _setup_engine is None:
        return []
    statuses = await asyncio.to_thread(_setup_engine.check_all)
    return await _augment_codex_live_models([
        _build_sub_agent_status_entry(s)
        for s in statuses
        if s.slug != "built-in"
    ])


class SubAgentConfigRequest(BaseModel):
    # All fields optional; pass empty string to clear that param. Only
    # parameters listed in the sub-agent's schema are accepted; anything
    # else returns 400.
    model: str | None = None
    effort: str | None = None
    permission_mode: str | None = None
    approval_mode: str | None = None


def _request_to_params(req: SubAgentConfigRequest) -> dict[str, str]:
    """Build a {param_name: value} dict from the request fields.

    Field names map 1:1 to parameter names in the schema, except snake_case
    is converted to kebab-case (Python keyword-friendly). None values are
    skipped; empty strings clear the param.
    """
    raw = {
        "model": req.model,
        "effort": req.effort,
        "permission-mode": req.permission_mode,
        "approval-mode": req.approval_mode,
    }
    return {k: v for k, v in raw.items() if v is not None}


@router.put("/settings/sub-agents/{slug}/config")
async def put_sub_agent_config(slug: str, req: SubAgentConfigRequest):
    """Persist sub-agent CLI parameter overrides.

    Validates against the per-slug stable-parameter schema. On success,
    returns the cleaned config that was persisted.
    """
    if _sub_agent_config_store is None:
        raise HTTPException(status_code=503, detail="Sub-agent config store not available")
    from agent_os.daemon_v2.sub_agent_config_store import SubAgentConfigError
    params = _request_to_params(req)
    # Codex models are gated per account+client: a value the gate rejects
    # saves fine as free text and then 400s on EVERY dispatch (invisibly, if
    # the management turn is mid-flight). Fail at save time instead, with
    # the valid ids in the message. Probe unavailable → free-text fallback;
    # clearing (empty string) never consults the probe.
    codex_model = (params.get("model") or "").strip() if slug == "codex" else ""
    if codex_model:
        live = await _codex_live_models()
        if live is not None and codex_model not in live:
            raise HTTPException(status_code=400, detail=(
                f"invalid value for codex.model: '{codex_model}'. This "
                f"account's codex CLI accepts: {live}"))
    try:
        cleaned = _sub_agent_config_store.set(slug, params)
    except SubAgentConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"slug": slug, "config": cleaned}


@router.post("/settings/sub-agents/refresh")
async def refresh_sub_agent_status():
    """Invalidate the SetupEngine cache and return fresh status."""
    if _setup_engine is None:
        return []
    if hasattr(_setup_engine, "invalidate_cache"):
        _setup_engine.invalidate_cache()
    # User-triggered refresh wants CURRENT state — drop the codex model/list
    # cache too so a just-fixed install / new model generation shows up.
    from agent_os.agent.transports.codex_models import clear_codex_models_cache
    clear_codex_models_cache()
    statuses = await asyncio.to_thread(_setup_engine.check_all)
    return await _augment_codex_live_models([
        _build_sub_agent_status_entry(s)
        for s in statuses
        if s.slug != "built-in"
    ])


def _resolve_setup_command(slug: str, action: str) -> str | None:
    """Find the manifest's command string for a setup action.

    ``action`` is one of "login", "logout". Looks up the matching credential
    on the agent's manifest and returns ``setup_command`` (login) or
    ``logout_command`` (not currently in manifest schema; falls back to
    deriving from binary name for known agents).

    Returns None if the agent has no such action.
    """
    if _setup_engine is None:
        return None
    registry = getattr(_setup_engine, "_registry", None)
    if registry is None:
        return None
    manifest = registry.get(slug)
    if manifest is None:
        return None
    binary = _setup_engine.resolve_binary(manifest)
    if action == "login":
        # Use the manifest credential's setup_command if defined.
        for cred in manifest.setup.credentials:
            if cred.setup_command:
                cmd = cred.setup_command
                # Substitute the resolved binary path for the bare command
                if binary and manifest.runtime.command:
                    return _setup_engine._substitute_binary(
                        cmd, manifest.runtime.command, binary
                    )
                return cmd
        return None
    if action == "logout":
        # Manifests don't currently carry a logout command; fall back to a
        # reasonable default for known CLIs that ship a logout subcommand.
        if not binary:
            return None
        if manifest.runtime.command in ("claude", "codex", "agent", "cursor-agent"):
            subcommand = (
                "auth logout"
                if manifest.runtime.command == "claude"
                else "logout"
            )
            return f"{binary} {subcommand}"
        return None
    return None


async def _run_login_job(slug: str, job_id: str, command: str) -> None:
    """Background task: run the CLI login command, broadcast progress.

    Captures stdout+stderr line by line and emits ``login.progress`` events
    over WebSocket. Final ``login.complete`` or ``login.failed`` carries the
    return code. Tokens never enter Orbital's storage — the CLI writes them
    to its own location.
    """
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except OSError as exc:
        if _ws_manager is not None:
            _ws_manager.broadcast_global({
                "type": "login.failed",
                "job_id": job_id,
                "slug": slug,
                "error": str(exc),
            })
        _login_jobs[job_id]["status"] = "failed"
        return

    _login_jobs[job_id]["pid"] = proc.pid
    assert proc.stdout is not None
    while True:
        raw = await proc.stdout.readline()
        if not raw:
            break
        line = raw.decode("utf-8", errors="replace").rstrip()
        if _ws_manager is not None and line:
            _ws_manager.broadcast_global({
                "type": "login.progress",
                "job_id": job_id,
                "slug": slug,
                "line": line,
            })

    rc = await proc.wait()
    _login_jobs[job_id]["status"] = "complete" if rc == 0 else "failed"
    _login_jobs[job_id]["return_code"] = rc

    # Re-check status now that the auth state may have changed.
    if _setup_engine is not None and hasattr(_setup_engine, "invalidate_cache"):
        _setup_engine.invalidate_cache()

    if _ws_manager is not None:
        _ws_manager.broadcast_global({
            "type": "login.complete" if rc == 0 else "login.failed",
            "job_id": job_id,
            "slug": slug,
            "return_code": rc,
        })


@router.post("/settings/sub-agents/{slug}/login")
async def trigger_sub_agent_login(slug: str):
    """Trigger the CLI's own login subprocess.

    Returns immediately with a ``job_id``; progress streams over WebSocket
    via ``login.progress`` / ``login.complete`` / ``login.failed`` events.

    Orbital does NOT capture or store the resulting credentials — the CLI
    writes them to its own location (``~/.claude/``, ``~/.codex/``, etc.).
    """
    command = _resolve_setup_command(slug, "login")
    if not command:
        raise HTTPException(
            status_code=400,
            detail=(
                f"sub-agent '{slug}' does not expose a login subcommand "
                f"(set the API key via env or its own settings file)"
            ),
        )
    job_id = uuid4().hex[:12]
    _login_jobs[job_id] = {"slug": slug, "status": "running", "command": command}
    asyncio.create_task(_run_login_job(slug, job_id, command))
    return {"job_id": job_id, "slug": slug, "status": "running"}


@router.post("/settings/sub-agents/{slug}/logout")
async def trigger_sub_agent_logout(slug: str):
    """Trigger the CLI's own logout subprocess.

    Synchronous: returns the CLI's exit code and any stderr text. The CLI
    is responsible for clearing its own stored credentials.
    """
    command = _resolve_setup_command(slug, "logout")
    if not command:
        raise HTTPException(
            status_code=400,
            detail=f"sub-agent '{slug}' does not expose a logout subcommand",
        )
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="logout timed out")

    if _setup_engine is not None and hasattr(_setup_engine, "invalidate_cache"):
        _setup_engine.invalidate_cache()

    return {
        "slug": slug,
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


class SetSubAgentApiKeyRequest(BaseModel):
    api_key: str


@router.post("/settings/sub-agents/{slug}/api-key")
async def set_sub_agent_api_key(slug: str, req: SetSubAgentApiKeyRequest):
    """Hand an API key to the CLI's own ingestion command.

    For agents whose login flow is API-key based (e.g. ``codex login --with-api-key``),
    this endpoint pipes the key on stdin to the CLI so it lands in the CLI's
    expected location (e.g. ``~/.codex/auth.json``). Orbital never persists
    the key in its own storage.
    """
    if not req.api_key.strip():
        raise HTTPException(status_code=400, detail="api_key must be non-empty")

    if _setup_engine is None:
        raise HTTPException(status_code=503, detail="Setup engine not available")
    registry = getattr(_setup_engine, "_registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="Agent registry not available")
    manifest = registry.get(slug)
    if manifest is None:
        raise HTTPException(status_code=404, detail=f"Unknown agent: {slug}")
    binary = _setup_engine.resolve_binary(manifest)
    if not binary:
        raise HTTPException(status_code=400,
                            detail=f"sub-agent '{slug}' is not installed")

    # Hard-coded mapping for the known set of API-key login flows. Adding a
    # new one is a one-line addition here; we resist adding it as a manifest
    # field until there are more than two entries.
    if slug == "codex":
        argv = [binary, "login", "--with-api-key"]
    else:
        raise HTTPException(
            status_code=400,
            detail=(
                f"sub-agent '{slug}' does not support API-key login via "
                f"this endpoint"
            ),
        )

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(
            proc.communicate(input=req.api_key.encode("utf-8")),
            timeout=30,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="api-key ingestion timed out")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if _setup_engine is not None and hasattr(_setup_engine, "invalidate_cache"):
        _setup_engine.invalidate_cache()

    return {
        "slug": slug,
        "return_code": proc.returncode,
        "stdout": out.decode("utf-8", errors="replace"),
        "stderr": err.decode("utf-8", errors="replace"),
    }
