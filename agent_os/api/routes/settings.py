# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Global settings endpoints."""

import asyncio
import logging
import os
import re
import subprocess
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agent_os import telemetry
from agent_os.agents.installer import (
    InstallInProgress,
    InstallUnsupported,
    SubAgentInstaller,
)
from agent_os.utils.subprocess_flags import win_no_window_flags

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2")

_settings_store = None
_credential_store = None
_setup_engine = None
_sub_agent_config_store = None
_ws_manager = None
_installer: SubAgentInstaller | None = None
# Tracks in-flight login subprocesses keyed by job_id, so callers can poll
# WS events for ``login.progress`` / ``login.complete`` / ``login.failed``.
_login_jobs: dict[str, dict] = {}

# Seconds of silence (no output line) before an in-flight login is killed.
# It is an IDLE timeout, not a wall-clock cap: the user is off in a browser
# doing an OAuth dance and the CLI stays quiet the whole time, so the budget
# has to be generous. What it actually stops is an abandoned flow leaking a
# subprocess for the daemon's lifetime.
LOGIN_IDLE_TIMEOUT_SECONDS = 300.0

# OSC (Operating System Command) sequences, terminated by BEL or ST. The
# claude CLI prints its login URL as an OSC-8 terminal hyperlink
# (``ESC]8;;<url>ST<label>ESC]8;;ST``), so the raw line carries the URL twice
# wrapped in control bytes — unreadable if broadcast verbatim.
_OSC_SEQUENCE_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
# CSI sequences: colours, cursor moves, erase-line — the usual spinner kit.
_CSI_SEQUENCE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
# Anything left over: stray ESC, BEL, backspace, DEL.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _strip_terminal_escapes(line: str) -> str:
    """Reduce a raw CLI output line to what a terminal would actually show.

    Drops OSC-8 hyperlink wrappers (leaving the visible label, which for the
    claude login URL *is* the URL, once instead of twice), CSI colour/cursor
    codes, and leftover control bytes. Carriage returns are in-place redraws,
    so only the last frame survives.
    """
    line = _OSC_SEQUENCE_RE.sub("", line)
    line = _CSI_SEQUENCE_RE.sub("", line)
    # Drop the LINE TERMINATOR before interpreting carriage returns as redraws.
    # On Windows the CLI's newline is CRLF, so `"waiting\r\n"` would otherwise
    # split to `"\n"` and reduce to "" — falsy, so the caller broadcast nothing
    # and EVERY login progress line vanished on Windows, taking the sign-in URL
    # fallback link with it. A terminator is not a redraw.
    line = line.rstrip("\r\n")
    if "\r" in line:
        line = line.split("\r")[-1]
    return _CONTROL_CHARS_RE.sub("", line).strip()


def configure(settings_store, credential_store=None, setup_engine=None,
              sub_agent_config_store=None, ws_manager=None):
    global _settings_store, _credential_store, _setup_engine
    global _sub_agent_config_store, _ws_manager, _installer
    _settings_store = settings_store
    _credential_store = credential_store
    _setup_engine = setup_engine
    _sub_agent_config_store = sub_agent_config_store
    _ws_manager = ws_manager
    # The installer's job registry is per-daemon by design: a restart mid-run
    # leaves no completion marker, which is the "partial, re-run" signal the
    # next status read wants. Rebuilding it here keeps that lifetime tied to
    # the app it serves.
    _installer = SubAgentInstaller(setup_engine=setup_engine,
                                   ws_manager=ws_manager)


class UpdateSettingsRequest(BaseModel):
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_sdk: str | None = None
    llm_provider: str | None = None
    llm_fallback_models: list[dict] | None = None
    user_preferences_content: str | None = None
    user_preferences_path: str | None = None
    user_memory_content: str | None = None
    user_memory_path: str | None = None
    user_memory_enabled: bool | None = None
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
    if req.user_memory_path is not None:
        current.user_memory_path = req.user_memory_path
    if req.user_memory_enabled is not None:
        current.user_memory_enabled = req.user_memory_enabled
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

    # Write user memory content to file (spec 073). Full overwrite — the
    # Settings textarea IS the edit/prune surface. This is a SEPARATE file
    # from user_preferences.md precisely so the prefs overwrite above can
    # never clobber agent-filed facts (D2).
    if req.user_memory_content is not None:
        memory_path = current.user_memory_path
        if not memory_path:
            memory_path = os.path.join(os.path.expanduser("~"), "orbital", "user_memory.md")
            current.user_memory_path = memory_path
        os.makedirs(os.path.dirname(memory_path), exist_ok=True)
        with open(memory_path, "w", encoding="utf-8") as f:
            f.write(req.user_memory_content)

    _settings_store.update(current)

    # Most keys arrive here, not through PUT /settings/api-key — this is the
    # write the provider dropdown uses — and this path emitted nothing, so
    # `key_set` under-counted. Emitted after the update so a request that sets
    # provider and key together reports the provider the key belongs to.
    if req.llm_api_key is not None:
        telemetry.emit("key_set", {"provider": current.llm.provider})
        telemetry.latch("key_set")

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


@router.get("/update-status")
async def get_update_status():
    """Current vs latest released version (notify-only update check). The
    frontend pill renders from the WS announce; this endpoint covers page
    loads that happen after the announce and the manual re-check button."""
    from agent_os import update_check

    checker = update_check.get_checker()
    if checker is None:
        from agent_os.version import get_version

        return {"current": get_version(), "update_available": False,
                "latest": None, "url": None}
    await checker.run_check()
    return checker.status


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
# trigger. For agents that ship a credential store of their own, tokens stay
# there (``~/.claude/``, ``~/.codex/``, etc.) and never enter Orbital's
# storage. Agents with no such store (dsh) are the documented exception:
# their API key is held in a dedicated keychain service via
# ``SubAgentCredentialStore`` and written through the ``/credential`` routes
# below. Either way the raw key never appears in a response.
# ---------------------------------------------------------------------------


def _registry_manifest(slug: str):
    """The manifest for ``slug`` from the setup engine's registry, or None.

    The non-raising counterpart to ``_sub_agent_manifest``, for the status
    path — one unregistered agent must not fail the whole listing.
    """
    if _setup_engine is None:
        return None
    registry = getattr(_setup_engine, "_registry", None)
    if registry is None:
        return None
    return registry.get(slug)


def _supports_login(slug: str) -> bool:
    """Whether this sub-agent exposes an interactive login (OAuth) flow.

    True when the agent's manifest declares a credential with a non-empty
    ``setup_command`` (e.g. ``claude login`` / ``codex login``). Agents that
    only accept an API key — and therefore have no login subcommand — return
    False so the frontend can hide a Login button that would otherwise 400.
    """
    manifest = _registry_manifest(slug)
    if manifest is None:
        return False
    return any(c.setup_command for c in manifest.setup.credentials)


def _install_entry(manifest, *, installed: bool) -> dict:
    """The ``install`` block of a status entry: what the card can offer.

    ``supported`` is whether Orbital can install this agent on THIS platform —
    false for every bring-your-own agent, and for a declared one on a platform
    its manifest left out. ``state`` is derived from the completion marker, the
    live job registry, and the binary probe, never from WS state, so a client
    that reloads mid-install still lands on the right screen.
    """
    if manifest is None or _installer is None:
        return {"supported": False, "platforms": [],
                "state": "installed" if installed else "not_installed"}
    return _installer.entry_for(manifest, installed=installed)


def _credential_configured(cred, missing: list[str]) -> bool:
    """Whether one declared credential currently resolves to a value.

    Required credentials were already resolved by ``check_all`` — including
    the ``oauth_cli`` ones, whose probe spawns the agent's CLI — so their
    answer comes off ``missing_credentials``. Optional ones are never probed
    there (``check_credentials`` skips them outright), which means reading
    them off that list would report every unset optional credential as
    configured. Resolve those the way the spawn env does, minus the
    subprocess: the sub-agent store, then the environment.
    """
    if cred.required:
        return cred.key not in missing
    if cred.type == "oauth_cli":
        # Never spawn a CLI to answer a status read; only the required
        # oauth credentials are worth that cost, and check_all pays it.
        return False
    store = getattr(_setup_engine, "_credential_store", None)
    if store is not None and store.get(cred.key):
        return True
    return bool(os.environ.get(cred.env_var or cred.key))


def _declared_credentials(manifest, missing: list[str]) -> list[dict]:
    """Every credential the manifest declares, with its current state.

    ``missing_credentials`` alone cannot drive a credential form: it holds
    required-and-missing keys only, so an optional field never appears and a
    satisfied one vanishes, taking its rotate/remove affordance with it.
    """
    if manifest is None:
        return []
    return [
        {
            "key": c.key,
            "label": c.label,
            "type": c.type,
            "required": c.required,
            # True for the OAuth agents, whose key arrives through their own
            # CLI login rather than a form Orbital renders.
            "has_setup_command": bool(c.setup_command),
            "configured": _credential_configured(c, missing),
        }
        for c in manifest.setup.credentials
    ]


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
    manifest = _registry_manifest(s.slug)
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
        "credentials": _declared_credentials(manifest, s.missing_credentials),
        "supports_login": _supports_login(s.slug),
        "install": _install_entry(manifest, installed=s.installed),
        # False means the agent's turns are silent until the final answer, so
        # the UI can say that instead of showing activity it cannot back up.
        "emits_tool_activity": (
            manifest.capabilities.emits_tool_activity
            if manifest is not None else True
        ),
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


async def _credentials_now_configured(slug: str) -> bool:
    """Re-run the manifest's credential ``check_command`` for ``slug``.

    A login process exiting 0 is necessary but not sufficient: the invalid
    ``claude login`` form parsed as a prompt and exited 0 having authenticated
    nothing (bug #64). Only the CLI's own auth-status command can settle it,
    and it is the same probe the sub-agent list renders from, so agreeing with
    it is what stops the row flipping back to signed-out a second later.

    Returns False when the answer can't be established — the caller treats
    that as *unconfirmed*, never as a failure.
    """
    if _setup_engine is None:
        return False
    registry = getattr(_setup_engine, "_registry", None)
    manifest = registry.get(slug) if registry is not None else None
    if manifest is None:
        return False
    try:
        binary = _setup_engine.resolve_binary(manifest)
        configured, _missing = await asyncio.to_thread(
            _setup_engine.check_credentials, manifest, binary
        )
    except Exception:
        logger.exception("post-login credential re-check failed for %s", slug)
        return False
    return bool(configured)


async def _run_login_job(slug: str, job_id: str, command: str) -> None:
    """Background task: run the CLI login command, broadcast progress.

    Captures stdout+stderr line by line and emits sanitised ``login.progress``
    events over WebSocket. Final ``login.complete`` or ``login.failed`` carries
    the return code; ``login.complete`` also carries ``verified``, the result
    of re-running the manifest's credential check afterwards. An unverified
    completion is deliberately NOT a failure — a slow keychain write must not
    turn a real login into a reported error — but it is reported honestly so
    the UI can say "signed in, but couldn't confirm".

    Tokens from this flow never enter Orbital's storage — the CLI writes them
    to its own location. (Agents with no credential store of their own don't
    come through here at all; they use the ``/credential`` routes.)
    """
    telemetry.emit("login_attempted", {"agent": slug})
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            creationflags=win_no_window_flags(),
        )
    except OSError as exc:
        telemetry.emit("login_failed", {"agent": slug})
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
    timed_out = False
    while True:
        try:
            raw = await asyncio.wait_for(proc.stdout.readline(),
                                         timeout=LOGIN_IDLE_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            # Idle too long: the user walked away mid-OAuth. Kill it rather
            # than hold the subprocess for the daemon's lifetime.
            timed_out = True
            break
        if not raw:
            break
        line = _strip_terminal_escapes(raw.decode("utf-8", errors="replace"))
        if _ws_manager is not None and line:
            _ws_manager.broadcast_global({
                "type": "login.progress",
                "job_id": job_id,
                "slug": slug,
                "line": line,
            })

    if timed_out:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
        _login_jobs[job_id]["status"] = "failed"
        _login_jobs[job_id]["timed_out"] = True
        telemetry.emit("login_failed", {"agent": slug})
        if _ws_manager is not None:
            _ws_manager.broadcast_global({
                "type": "login.failed",
                "job_id": job_id,
                "slug": slug,
                "timed_out": True,
                "error": (
                    f"login timed out after "
                    f"{int(LOGIN_IDLE_TIMEOUT_SECONDS)}s with no activity"
                ),
            })
        return

    rc = await proc.wait()
    _login_jobs[job_id]["return_code"] = rc

    # Re-check status now that the auth state may have changed. This has to
    # come before the re-check below so the list the UI refetches and the
    # verdict we broadcast are reading the same (fresh) state.
    if _setup_engine is not None and hasattr(_setup_engine, "invalidate_cache"):
        _setup_engine.invalidate_cache()

    if rc != 0:
        _login_jobs[job_id]["status"] = "failed"
        telemetry.emit("login_failed", {"agent": slug})
        if _ws_manager is not None:
            _ws_manager.broadcast_global({
                "type": "login.failed",
                "job_id": job_id,
                "slug": slug,
                "return_code": rc,
            })
        return

    verified = await _credentials_now_configured(slug)
    _login_jobs[job_id]["status"] = "complete" if verified else "unverified"
    _login_jobs[job_id]["verified"] = verified
    if not verified:
        telemetry.emit("login_failed", {"agent": slug})

    if _ws_manager is not None:
        _ws_manager.broadcast_global({
            "type": "login.complete",
            "job_id": job_id,
            "slug": slug,
            "return_code": rc,
            "verified": verified,
        })


@router.post("/settings/sub-agents/{slug}/login")
async def trigger_sub_agent_login(slug: str):
    """Trigger the CLI's own login subprocess.

    Returns immediately with a ``job_id``; progress streams over WebSocket
    via ``login.progress`` / ``login.complete`` / ``login.failed`` events.

    Orbital does NOT capture or store the credentials this flow produces — the
    CLI writes them to its own location (``~/.claude/``, ``~/.codex/``, etc.).
    Agents that ship no such store use ``/credential`` instead, which holds the
    key in a dedicated keychain service.
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


@router.post("/settings/sub-agents/{slug}/install", status_code=202)
async def install_sub_agent(slug: str):
    """Install an Orbital-managed sub-agent into the daemon's data dir.

    Returns immediately with a ``job_id``; the npm run takes 30–90 s and
    streams over WebSocket as ``sub_agent_install_progress``, ending in
    ``sub_agent_install_done`` or ``sub_agent_install_failed``. A client that
    misses those events reads the same lifecycle off ``install.state`` in
    ``GET /settings/sub-agents``.

    400 when the agent is not Orbital-installable at all, or not on this
    platform; 409 when a job for the same slug is already in flight.
    """
    manifest = _sub_agent_manifest(slug)
    if _installer is None:
        raise HTTPException(status_code=503,
                            detail="Sub-agent installer not available")
    try:
        job_id = _installer.start(slug, manifest)
    except InstallInProgress as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except InstallUnsupported as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"job_id": job_id, "slug": slug, "state": "installing"}


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
            # CLIs emit UTF-8; the Windows ANSI default crashes the decode
            # on non-ASCII output (see test_subprocess_encoding).
            encoding="utf-8",
            errors="replace",
            timeout=30,
            creationflags=win_no_window_flags(),
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


class SubAgentCredentialRequest(BaseModel):
    # Exactly one of ``value`` / ``use_llm_provider_key`` must be supplied.
    key: str
    value: str | None = None
    use_llm_provider_key: bool = False


def _mask_secret(value: str) -> str:
    """first4...last4 — the only form a stored key may take in a response."""
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}...{value[-4:]}"


def _sub_agent_manifest(slug: str):
    """Return the manifest for ``slug`` or raise the matching HTTPException."""
    if _setup_engine is None:
        raise HTTPException(status_code=503, detail="Setup engine not available")
    registry = getattr(_setup_engine, "_registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="Agent registry not available")
    manifest = registry.get(slug)
    if manifest is None:
        raise HTTPException(status_code=404, detail=f"Unknown agent: {slug}")
    return manifest


def _validate_credential_key(manifest, key: str) -> None:
    """Reject keys the agent's manifest doesn't declare."""
    declared = [c.key for c in manifest.setup.credentials]
    if key not in declared:
        raise HTTPException(status_code=400, detail=(
            f"unknown credential '{key}' for sub-agent '{manifest.slug}'. "
            f"Declared credentials: {declared}"))


def _sub_agent_credential_store():
    """The dedicated keychain store SetupEngine resolves credentials from."""
    store = getattr(_setup_engine, "_credential_store", None)
    if store is None:
        raise HTTPException(
            status_code=503, detail="Sub-agent credential store not available")
    return store


@router.post("/settings/sub-agents/{slug}/credential")
async def set_sub_agent_credential(slug: str, req: SubAgentCredentialRequest):
    """Store one manifest-declared credential for a sub-agent.

    For agents that ship no credential store of their own. The value lands in
    a dedicated OS keychain service and is injected into the spawn env by
    SetupEngine. The response carries a mask only — never the raw key.

    ``use_llm_provider_key: true`` copies the global LLM key server-side
    instead of taking one in the body, so the client never has to hold key
    text it already can't read back.
    """
    manifest = _sub_agent_manifest(slug)
    _validate_credential_key(manifest, req.key)
    store = _sub_agent_credential_store()

    supplied = (req.value or "").strip()
    if bool(supplied) == req.use_llm_provider_key:
        raise HTTPException(status_code=400, detail=(
            "supply exactly one of 'value' (non-empty) or "
            "'use_llm_provider_key: true'"))

    if req.use_llm_provider_key:
        provider = _settings_store.get().llm.provider
        if provider != "deepseek":
            raise HTTPException(status_code=409, detail=(
                f"the global LLM key belongs to provider '{provider}', not "
                f"deepseek — paste the key for '{slug}' directly instead"))
        if _credential_store is None:
            raise HTTPException(status_code=503,
                                detail="Credential store not available")
        supplied = (_credential_store.get_api_key() or "").strip()
        if not supplied:
            raise HTTPException(status_code=409,
                                detail="no global LLM API key is configured")

    try:
        store.set(req.key, supplied)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if hasattr(_setup_engine, "invalidate_cache"):
        _setup_engine.invalidate_cache()

    return {"slug": slug, "key": req.key, "set": True,
            "masked": _mask_secret(supplied)}


@router.delete("/settings/sub-agents/{slug}/credential/{key}")
async def delete_sub_agent_credential(slug: str, key: str):
    """Remove a stored sub-agent credential from the keychain."""
    manifest = _sub_agent_manifest(slug)
    _validate_credential_key(manifest, key)
    _sub_agent_credential_store().delete(key)

    if hasattr(_setup_engine, "invalidate_cache"):
        _setup_engine.invalidate_cache()

    return {"slug": slug, "key": key, "set": False}


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
            creationflags=win_no_window_flags(),
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

    # The third way a key gets set, and the only one that emitted nothing.
    # Gated on the CLI accepting it: a rejected key never landed anywhere, so
    # counting it would inflate `key_set` with failures. ``provider`` is the
    # agent slug here — this key belongs to the sub-agent CLI, not to the
    # global LLM provider.
    if proc.returncode == 0:
        telemetry.emit("key_set", {"provider": slug})
        telemetry.latch("key_set")

    return {
        "slug": slug,
        "return_code": proc.returncode,
        "stdout": out.decode("utf-8", errors="replace"),
        "stderr": err.decode("utf-8", errors="replace"),
    }
