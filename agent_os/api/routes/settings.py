# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Global settings endpoints."""

import asyncio
import copy
import hashlib
import inspect
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agent_os import telemetry
from agent_os.agent.transports.process_kill import kill_spawned_tree
from agent_os.telemetry.diagnostics import (
    classify_stderr,
    clean_version,
    error_class_of,
    install_kind_of,
)
from agent_os.agents.handshake import HANDSHAKE_TIMEOUT_S
from agent_os.agents.setup_engine import agent_store_has
from agent_os.agents.setup_types import status_ready
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
# Spec 082: DELETE /settings/cards/{id} re-points every project that
# referenced the card, so the settings routes need the project store too.
_project_store = None
_setup_engine = None
_sub_agent_config_store = None
_ws_manager = None
_installer: SubAgentInstaller | None = None
# The sub-agent manager, for the update route's "is a worker running" check
# (spec 116 D8). None when unwired (tests, older app factories): no check.
_sub_agent_manager = None
# Login / update jobs keyed by job_id (spec 116 D2: the update job is the
# same runner with ``kind="update"``). Running entries plus the last
# ``LOGIN_JOB_HISTORY`` terminal ones (spec 113 A3): reason, return code,
# redacted output tail — what a user pastes into an issue, read back over
# ``GET /settings/sub-agents/login-jobs``. In-memory: a restart clears it,
# the daemon.log line written per terminal is the durable record.
_login_jobs: dict[str, dict] = {}
LOGIN_JOB_HISTORY = 20
# How many trailing output lines a terminal keeps (local log + job entry).
LOGIN_JOB_TAIL_LINES = 5

# Seconds of silence (no output line) before an in-flight login is killed.
# It is an IDLE timeout, not a wall-clock cap: the user is off in a browser
# doing an OAuth dance and the CLI stays quiet the whole time, so the budget
# has to be generous. What it actually stops is an abandoned flow leaking a
# subprocess for the daemon's lifetime.
LOGIN_IDLE_TIMEOUT_SECONDS = 300.0

# Wall-clock budget for an Update job. npm prints almost nothing while it
# downloads, and on a slow network a global reinstall took 3 min on the 1018
# bench (``claude update``'s own npm step gave up after 39 s and left the
# package half-removed), so the login idle timeout would kill it mid-install.
UPDATE_TIMEOUT_SECONDS = 15 * 60

# An update whose re-probe finds no runnable CLI afterwards (1018 bench item
# 5): the job reports this instead of its exit-code reason, so the card can
# say the CLI is gone and name the recovery command.
CLI_MISSING_AFTER_UPDATE = "cli_missing_after_update"

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
              sub_agent_config_store=None, ws_manager=None, project_store=None,
              sub_agent_manager=None):
    global _settings_store, _credential_store, _setup_engine, _project_store
    global _sub_agent_config_store, _ws_manager, _installer, _sub_agent_manager
    _settings_store = settings_store
    _credential_store = credential_store
    _project_store = project_store
    _setup_engine = setup_engine
    _sub_agent_config_store = sub_agent_config_store
    _ws_manager = ws_manager
    _sub_agent_manager = sub_agent_manager
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
    # First-run wizard "Skip for now" (keyless mode). Omitted = untouched.
    setup_api_key_skipped: bool | None = None
    # Spec 138: global hidden sub-agents (replace). Omitted = untouched.
    sub_agents_hidden: list[str] | None = None
    # Spec 160 round 2: the sign's switch beside Skip. False = don't read the
    # imported projects still waiting. Omitted = untouched.
    import_scan: bool | None = None


class SetApiKeyRequest(BaseModel):
    api_key: str


class CreateCardRequest(BaseModel):
    """The Add-card form (spec 082 §3.2) — the same fields it always sent."""
    name: str | None = None
    provider: str
    region: str = "global"
    base_url: str | None = None
    sdk: str | None = None
    api_key: str | None = None
    model: str
    # Spec 160 round 2: the sign's switch beside Save. False = don't read
    # the imported projects still waiting. Not a card field.
    import_scan: bool | None = None


class UpdateCardRequest(BaseModel):
    """Every field optional; only PRESENT fields change.

    ``api_key`` present ⇒ replace the key. The card id never changes, so no
    project, fallback entry or session is touched by a re-key (D3).
    """
    name: str | None = None
    provider: str | None = None
    region: str | None = None
    base_url: str | None = None
    sdk: str | None = None
    api_key: str | None = None
    model: str | None = None
    # Spec 160 round 2: as on CreateCardRequest. Not a card field.
    import_scan: bool | None = None


def _card_error(code: str, message: str, status: int = 400) -> HTTPException:
    """Card routes answer with a machine code, never a bare sentence."""
    return HTTPException(status_code=status,
                         detail={"code": code, "message": message})


def _card_or_404(card_id: str):
    card = _settings_store.resolve_card(card_id)
    if card is None:
        raise _card_error("card_not_found",
                          "That credential card no longer exists.", 404)
    return card


def _reject_read_only(card_id: str) -> None:
    from agent_os.daemon_v2.credential_store import ENV_CARD_ID

    if card_id == ENV_CARD_ID:
        raise _card_error(
            "card_read_only",
            "The AGENT_OS_API_KEY card is read-only. Unset the variable to "
            "manage credentials here.",
        )


def _validate_card_shape(provider: str, model: str, base_url: str | None) -> None:
    """Reject a card that could never run, before it is saved."""
    from agent_os.config.provider_registry import ProviderRegistry

    if not (model or "").strip():
        raise _card_error("missing_model", "Pick a model for this card.")
    if provider == "custom":
        if not (base_url or "").strip():
            raise _card_error("missing_base_url",
                              "A custom provider needs an endpoint URL.")
        return
    if not ProviderRegistry().get_provider_data(provider):
        raise _card_error("unknown_provider", f"Unknown provider '{provider}'.")


async def _test_and_record(card_id: str, *, reuse_recent: bool = False) -> dict:
    """Save-then-test-then-record — the shared tail of every card write (D9)."""
    from agent_os.api.routes.agents_v2 import _test_and_record as _impl

    return await _impl(card_id, reuse_recent=reuse_recent)


def _masked(card_id: str) -> dict:
    return _settings_store.masked_card(_settings_store.get_card(card_id))


@router.get("/settings/cards")
async def list_cards():
    """Every card, env card first then recency order (spec 082 §3.1)."""
    return {"credential_cards": _settings_store.masked_cards(),
            "default_card_id": _settings_store.effective_default_card_id()}


@router.post("/settings/cards", status_code=201)
async def create_card(req: CreateCardRequest):
    """Create a card and verify it.

    The card is saved FIRST and the test runs after: a failed test still
    leaves a saved card carrying the error (D9), so a provider outage can
    never cost the user the credential they just typed. The form's own Test
    Connection on these inputs stands in for that test when there was one.
    """
    _validate_card_shape(req.provider, req.model, req.base_url)
    try:
        card = _settings_store.create_card(
            provider=req.provider, model=req.model, name=(req.name or "").strip(),
            region=req.region or "global", base_url=req.base_url, sdk=req.sdk,
            api_key=req.api_key if req.api_key is not None else "",
        )
    except (RuntimeError, ValueError) as exc:
        raise _card_error("keychain_error", str(exc), 500)
    telemetry.emit("key_set", {"provider": req.provider})
    telemetry.latch("key_set")
    test = await _test_and_record(card.id, reuse_recent=True)
    # Pi borrows a card key for its configured model's provider.
    _refresh_agents_in_background(_model_provider_slugs())
    _import_scan_switch(req.import_scan)
    _import_scan_kick("card_created")
    return {"card": _masked(card.id), "test": test}


@router.put("/settings/cards/{card_id}")
async def update_card(card_id: str, req: UpdateCardRequest):
    """Patch a card. Re-tests only when the CREDENTIAL changed.

    A rename is not a reason to spend a completion call — and on a rate-limited
    account it is not a reason to paint a working card red.
    """
    _reject_read_only(card_id)
    card = _card_or_404(card_id)
    fields = {k: v for k, v in req.model_dump().items()
              if k in req.model_fields_set and k != "import_scan"}
    if "name" in fields and not (fields["name"] or "").strip():
        fields.pop("name")
    provider = fields.get("provider", card.provider)
    model = fields.get("model", card.model)
    base_url = fields.get("base_url", card.base_url)
    if "model" in fields or "provider" in fields or "base_url" in fields:
        _validate_card_shape(provider, model, base_url)
    retest = any(f in fields for f in
                 ("api_key", "model", "region", "provider", "base_url", "sdk"))
    try:
        _settings_store.update_card(card_id, **fields)
    except (RuntimeError, ValueError) as exc:
        raise _card_error("keychain_error", str(exc), 500)
    test = await _test_and_record(card_id, reuse_recent=True) if retest else None
    if retest:
        _refresh_agents_in_background(_model_provider_slugs())
    _import_scan_switch(req.import_scan)
    if "api_key" in fields:
        _import_scan_kick("card_updated")
    return {"card": _masked(card_id), "test": test}


@router.delete("/settings/cards/{card_id}")
async def delete_card(card_id: str):
    """Delete a card, re-pointing everything that referenced it.

    The default card cannot be deleted while it is the default (D5): every
    project on ``card_id=null`` follows it, so removing it would silently
    move the whole install onto nothing.
    """
    _reject_read_only(card_id)
    card = _card_or_404(card_id)
    if card_id == _settings_store.effective_default_card_id():
        raise _card_error(
            "card_is_default",
            "This is the default card. Make another card the default first.",
        )
    reassigned = []
    if _project_store is not None:
        for project in _project_store.list_projects():
            pid = project.get("project_id")
            updates: dict = {}
            if project.get("card_id") == card_id:
                updates["card_id"] = None
                updates["migration_note"] = f"card_deleted:{card.name}"
                reassigned.append({"project_id": pid,
                                   "name": project.get("name", "")})
            refs = project.get("llm_fallback_models") or []
            if any(isinstance(r, dict) and r.get("card_id") == card_id for r in refs):
                updates["llm_fallback_models"] = [
                    r for r in refs
                    if not (isinstance(r, dict) and r.get("card_id") == card_id)
                ]
            if updates:
                _project_store.update_project(pid, updates)

    current = _settings_store.get()
    kept = [fb for fb in (current.llm.fallback_models or [])
            if fb.card_id != card_id]
    if len(kept) != len(current.llm.fallback_models or []):
        current.llm.fallback_models = kept
        _settings_store.update(current)

    _settings_store.delete_card(card_id)
    _refresh_agents_in_background(_model_provider_slugs())
    return {"id": card_id, "deleted": True, "reassigned_projects": reassigned}


@router.put("/settings/cards/{card_id}/default")
async def set_default_card(card_id: str):
    """Make a card the global default.

    While ``AGENT_OS_API_KEY`` is set the choice is still recorded, but the
    env card stays effective — the response says so with ``applied: false``
    rather than pretending the switch took.
    """
    from agent_os.daemon_v2.credential_store import ENV_CARD_ID

    _card_or_404(card_id)
    if card_id != ENV_CARD_ID:
        _settings_store.set_default_card(card_id)
    effective = _settings_store.effective_default_card_id()
    _import_scan_kick("default_card_changed")
    return {"default_card_id": effective, "applied": effective == card_id}


@router.post("/settings/cards/{card_id}/test")
async def test_card(card_id: str):
    """Re-test a saved card and write its health. 200 even when it fails."""
    _card_or_404(card_id)
    test = await _test_and_record(card_id)
    return {"card": _masked(card_id), "test": test}


def _import_scan_switch(on: bool | None) -> None:
    """The sign's switch rode on this request: off turns every imported
    project still waiting to ``off`` before the action's trigger fires."""
    if on is False:
        from agent_os.onboarding.import_scan import turn_off_waiting
        turn_off_waiting(_project_store)


def _import_scan_kick(reason: str) -> None:
    """Spec 160 §5.2 trigger 2: a key became available, so imported projects
    still waiting for their one read can start (the queue checks each
    project's key itself; a no-op when nothing waits)."""
    from agent_os.onboarding import import_scan
    import_scan.kick(reason)


@router.get("/settings")
async def get_settings():
    from agent_os.onboarding.import_scan import unscanned_count

    out = _settings_store.get_masked()
    # Spec 160 §5.3: how many imported projects a key saved now would read
    # once (the token sign beside the key form's Save).
    if isinstance(out, dict):
        out = {**out, "unscanned_imports": await asyncio.to_thread(
            unscanned_count, _project_store)}
        # Round 2: the agent a keyless scan would spend ({slug, name}), for
        # the sign beside Skip and the import dialog. None: no worker ready.
        from agent_os.api.routes import agents_v2
        from agent_os.onboarding.import_scan import scan_agent
        out["import_scan_agent"] = await asyncio.to_thread(
            scan_agent, agents_v2._agent_manager, _project_store)
    return out


def _apply_legacy_llm_fields(req: UpdateSettingsRequest) -> None:
    """Apply the pre-cards ``llm_*`` fields to the DEFAULT card (§3.8 shim).

    One release only, for the first-run wizard and any older SPA that reaches
    a newer daemon. New clients use the card routes. The default card is
    created if there is none, so the wizard's "PUT /settings then PUT
    /settings/api-key" order still ends with one complete card.
    """
    touched = [req.llm_provider, req.llm_model, req.llm_base_url,
               req.llm_sdk, req.llm_api_key]
    if all(v is None for v in touched):
        return
    from agent_os.daemon_v2.settings_store import (
        default_card_name, region_for_base_url,
    )

    card = _settings_store.ensure_default_card()
    fields: dict = {}
    provider = req.llm_provider if req.llm_provider is not None else card.provider
    if req.llm_provider is not None:
        fields["provider"] = req.llm_provider
    if req.llm_model is not None:
        fields["model"] = req.llm_model
    if req.llm_base_url is not None:
        region, kept = region_for_base_url(provider, req.llm_base_url)
        fields["region"] = region
        fields["base_url"] = kept
    if req.llm_sdk is not None and provider == "custom":
        fields["sdk"] = req.llm_sdk
    if req.llm_api_key is not None:
        # Pre-cards parity: ``ApiKeyStore.set_api_key`` returned
        # ``{"source": "environment"}`` WITHOUT writing whenever
        # AGENT_OS_API_KEY was set — the variable is the key the daemon will
        # actually use, so persisting another one is unobservable, and on a
        # machine with no usable keyring (CI, a headless box, the null
        # backend) the attempt raises and turns this whole PUT into a 500.
        # The card routes stay strict; this is the legacy endpoint keeping
        # its legacy contract.
        from agent_os.daemon_v2.credential_store import _ENV_VAR

        if os.environ.get(_ENV_VAR):
            logger.info(
                "PUT /settings: %s is set, so the submitted key is not "
                "persisted; the environment key remains in effect", _ENV_VAR,
            )
        else:
            fields["api_key"] = req.llm_api_key
    # Keep the auto-generated name tracking provider/model; a user-chosen name
    # is never overwritten.
    if ("provider" in fields or "model" in fields) and (
            not card.name
            or card.name == default_card_name(card.provider, card.model)):
        fields["name"] = default_card_name(
            fields.get("provider", card.provider),
            fields.get("model", card.model),
        )
    try:
        _settings_store.update_card(card.id, **fields)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.put("/settings")
async def update_settings(req: UpdateSettingsRequest):
    _apply_legacy_llm_fields(req)
    current = _settings_store.get()
    if req.llm_fallback_models is not None:
        from agent_os.daemon_v2.settings_store import FallbackModelConfig
        current.llm.fallback_models = [
            FallbackModelConfig(**fb) for fb in req.llm_fallback_models
            if isinstance(fb, dict) and fb.get("card_id")
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
    if req.setup_api_key_skipped is not None:
        current.setup_api_key_skipped = req.setup_api_key_skipped
    if req.sub_agents_hidden is not None:
        current.sub_agents_hidden = sorted(
            {h.strip() for h in req.sub_agents_hidden if isinstance(h, str) and h.strip()})

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
        card = _settings_store.default_card()
        telemetry.emit("key_set",
                       {"provider": card.provider if card else "custom"})
        telemetry.latch("key_set")
        _import_scan_kick("legacy_key_saved")
    if req.setup_api_key_skipped:
        # Spec 138: the wizard's "Skip for now" on the Orbital-agent key step.
        telemetry.latch("key_skipped")
        # Skip means skip (2026-11-07, replacing spec 160 round 2's "the
        # pinned worker reads them now"): the wizard's deferred imports are
        # not read and no trigger fires. An explicit import_scan:false still
        # switches off everything waiting, as before.
        from agent_os.onboarding.import_scan import turn_off_deferred
        turn_off_deferred(_project_store)
        _import_scan_switch(req.import_scan)

    return _settings_store.get_masked()


class UiLanguageRequest(BaseModel):
    ui_language: str | None = None
    ui_languages: list[str] | None = None


@router.put("/settings/ui-language")
async def set_ui_language(req: UiLanguageRequest):
    """The app reports its interface language and the device's preferred
    languages, on load and on every language change. Onboarding prompts hand
    them to the agent (``onboarding.user_language``). Junk tags are dropped;
    the file is written only on a change."""
    from agent_os.onboarding.user_language import clean_tag, clean_tags

    current = _settings_store.get()
    ui = clean_tag(req.ui_language) if req.ui_language is not None else None
    langs = clean_tags(req.ui_languages) if req.ui_languages is not None else None
    changed = False
    if ui is not None and ui != current.ui_language:
        current.ui_language = ui
        changed = True
    if langs is not None and langs != current.ui_languages:
        current.ui_languages = langs
        changed = True
    if changed:
        await asyncio.to_thread(_settings_store.update, current)
    return {"ui_language": current.ui_language, "ui_languages": current.ui_languages}


@router.put("/settings/api-key")
async def set_api_key(req: SetApiKeyRequest):
    """Set the DEFAULT card's key (§3.8 shim, one release).

    A card is created first when there is none, so the wizard can paste a key
    before it has picked a provider and model.
    """
    if _credential_store is None:
        raise HTTPException(status_code=501, detail="Credential store not available")
    try:
        result = _credential_store.set_api_key(req.api_key)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    card = _settings_store.default_card()
    telemetry.emit("key_set",
                   {"provider": card.provider if card else "custom"})
    telemetry.latch("key_set")
    _import_scan_kick("legacy_key_saved")
    return {**result, "card_id": result.get("card_id",
                                            card.id if card else None)}


@router.delete("/settings/api-key")
async def delete_api_key():
    """Remove the DEFAULT card's key (§3.8 shim). The card itself stays."""
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


@router.post("/settings/telemetry/flush")
async def flush_telemetry():
    """Send today's ping (and any rebuilt past days) now, bounded (spec 111
    A2). The desktop app calls this on its way out, before ``os._exit``, so a
    short session is counted instead of waiting for a next launch that may be
    days away. Same guards as the periodic sender: the dev daemon, CI and
    temp data dirs report ``skipped``. Never raises; never blocks past 3 s."""
    sender = telemetry.get_sender()
    if sender is None:
        return {"status": "skipped"}
    return {"status": await sender.flush_now(timeout=3.0)}


@router.get("/settings/api-key/status")
async def get_api_key_status():
    """Whether the DEFAULT card carries a key (§3.8 shim)."""
    card = _settings_store.default_card() if _settings_store else None
    if _credential_store is None:
        return {"configured": False, "source": "none",
                "card_id": card.id if card else None}
    source = _credential_store.get_source()
    return {"configured": source != "none", "source": source,
            "card_id": card.id if card else None}


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


def _desktop_app_entry(manifest) -> dict | None:
    """``{"label", "download_url"}`` for the first desktop app the manifest
    declares for this OS, or None."""
    if manifest is None:
        return None
    import platform

    from agent_os.agents.desktop_apps import download_url_for
    from agent_os.daemon_v2.models import detect_os

    current_os = detect_os()
    for app in manifest.setup.desktop_apps:
        if current_os in app.platforms:
            return {"label": app.label,
                    "download_url": download_url_for(app, current_os,
                                                     platform.machine())}
    return None


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
    if getattr(cred, "agent_store", None) and agent_store_has(cred.agent_store):
        return True
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
            # The agent's own app holds this key (DSH Desktop): the card
            # names it instead of offering a field whose value the agent
            # would ignore. None when Orbital's copy or the env is in charge.
            "stored_by": (c.agent_store.get("label") or manifest.name)
            if getattr(c, "agent_store", None) and agent_store_has(c.agent_store)
            else None,
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
    # Spec 116 D4: an installed CLI older than the manifest's floor is not
    # ready, and the card/picker say "update" rather than "login".
    update_needed = bool(getattr(s, "update_needed", False))
    return {
        "slug": s.slug,
        "name": s.name,
        "installed": s.installed,
        "binary_path": s.binary_path,
        "version": s.version,
        # Spec 116 D4 floor and the ACP handshake verdict both count here.
        "ready": status_ready(s),
        "update_needed": update_needed,
        "min_version": getattr(s, "min_version", None),
        # A dispatch failed with an authentication error while the probe
        # still says signed in (1018 bench item 9): the card shows Login and
        # "Sign-in expired". Not part of ``ready`` — it is a hint, not a gate.
        "auth_expired": bool(getattr(s, "auth_expired", False)),
        # The locate card: the launcher the user pasted, and whether that
        # file has since disappeared (then ``installed`` is false and the
        # card names the missing file instead of offering the prompt).
        "override_path": getattr(s, "override_path", None),
        "override_missing": bool(getattr(s, "override_missing", False)),
        "supports_update": bool(manifest is not None
                                and manifest.setup.update_command),
        "dependencies_met": s.dependencies_met,
        "missing_dependencies": s.missing_dependencies,
        "credentials_configured": s.credentials_configured,
        "missing_credentials": s.missing_credentials,
        # configured | missing | unknown for an agent whose credential follows
        # its configured model's provider (Pi); None for everyone else.
        "credential_state": getattr(s, "credential_state", None),
        "credentials": _declared_credentials(manifest, s.missing_credentials),
        "supports_login": _supports_login(s.slug),
        "install": _install_entry(manifest, installed=s.installed),
        # Which copy Orbital drives — {"kind": "desktop_app", "label",
        # "app_path", "app_version"} for the CLI inside a vendor app,
        # "override" for the user's pick, "path"/"auto_detect" otherwise.
        "source": getattr(s, "source", None),
        # ACP agents: {"state": ok|failed|unknown, "detail"} — "failed" is a
        # found-but-cannot-be-driven agent, and the detail is its own words.
        "handshake": getattr(s, "handshake", None),
        # The vendor desktop app that ships this agent, with the download
        # link for this OS/arch — what "Install" points at when Orbital
        # cannot install the agent itself.
        "desktop_app": _desktop_app_entry(manifest),
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


async def _detected_binary(slug: str) -> str | None:
    """The binary the last sweep found. A probe needs it on Windows, where an
    npm-installed CLI is only a ``.cmd`` and its bare name cannot be spawned."""
    if _setup_engine is None:
        return None
    statuses = await asyncio.to_thread(
        _setup_engine.check_all, allow_stale=True, refresh_if_stale=False)
    return next((s.binary_path for s in statuses if s.slug == slug), None)


async def _codex_live_models(binary: str | None = None) -> list[str] | None:
    """Indirection over the cached codex model/list probe (monkeypatch seam
    for route tests). None means "list unavailable" — callers degrade to the
    static schema (free-text model entry)."""
    from agent_os.agent.transports.codex_models import get_codex_models_cached
    return await get_codex_models_cached(
        binary or await _detected_binary("codex") or "codex")


async def _claude_live_models(binary: str | None = None) -> list[dict] | None:
    """Indirection over the cached claude-code model-list probe (monkeypatch
    seam for route tests): ``[{"value", "label"}]``. None means "list
    unavailable" — callers keep the static whitelist."""
    from agent_os.agent.transports.claude_models import get_claude_models_cached
    return await get_claude_models_cached(
        binary or await _detected_binary("claude-code") or "claude")


async def _augment_claude_live_models(entries: list[dict]) -> list[dict]:
    """Replace the claude-code model whitelist with the CLI's LIVE account
    list (initialize → models[]), so a new model generation shows up without
    a code change. A saved value the live list no longer offers is appended,
    so the dropdown still shows what is actually configured. Probe failure
    leaves the static whitelist."""
    for entry in entries:
        if entry.get("slug") != "claude-code":
            continue
        model_schema = (entry.get("param_schema") or {}).get("model")
        if model_schema is None:
            continue
        models = await _claude_live_models(entry.get("binary_path"))
        if models:
            allowed = [m["value"] for m in models]
            saved = (entry.get("config") or {}).get("model")
            if saved and saved not in allowed:
                allowed.append(saved)
            model_schema["allowed"] = allowed
            # Additive: the CLI's display names ("Opus 5.5") for the dropdown;
            # older frontends ignore it and show the raw values. A saved value
            # the live list dropped gets a derived name: no raw ids anywhere.
            from agent_os.daemon_v2.sub_agent_config_store import claude_model_label
            labels = {m["value"]: m["label"] for m in models}
            for value in allowed:
                labels.setdefault(value, claude_model_label(value))
            model_schema["labels"] = labels
    return entries


# Spec 136 C: how long a Settings read waits for the live model lists. A
# cached list answers within a loop tick or two; a cold probe (a CLI start,
# 0.3-3 s) keeps running after the reply and announces its list with
# ``sub_agents.updated`` when it lands.
LIVE_MODEL_WAIT_S = 0.15
SUB_AGENTS_UPDATED = "sub_agents.updated"
# The last late model answer announced per augmenter: a refetch that misses
# the budget again with the same answer must not announce again (the page
# refetches on the event — announcing every time would loop).
_late_model_digests: dict[str, str] = {}
_late_model_tasks: set[asyncio.Task] = set()
# The last live schemas each augmenter produced, per agent slug. A read that
# misses the budget serves these instead of the static list, so a menu opened
# after the probe cache expired shows "Opus 5.5", not `opus`, while the probe
# runs again (Windows smoke 2026-11-03, decision 4).
_last_live_schemas: dict[str, dict[str, dict]] = {}


def _model_schemas(entries: list[dict]) -> str:
    payload = json.dumps([(e.get("slug"), e.get("param_schema")) for e in entries],
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _broadcast_sub_agents_updated(reason: str) -> None:
    if _ws_manager is not None:
        _ws_manager.broadcast_global({"type": SUB_AGENTS_UPDATED, "reason": reason})


async def _augment_live_models(entries: list[dict], *,
                               wait: float = LIVE_MODEL_WAIT_S) -> list[dict]:
    """Every live model list at once (they used to run one after another),
    each against its own copy of the entries. Those that answer within
    ``wait`` are merged into this reply; the rest finish in the background
    and announce themselves when their list changes what the page shows."""
    augmenters = (("codex", _augment_codex_live_models),
                  ("claude-code", _augment_claude_live_models),
                  ("acp", _augment_acp_live_models))
    original = copy.deepcopy(entries)
    jobs = []
    for name, augment in augmenters:
        mine = copy.deepcopy(original)
        jobs.append((name, mine, _model_schemas(mine),
                     asyncio.ensure_future(augment(mine))))
    await asyncio.wait([task for *_rest, task in jobs], timeout=wait)
    for name, mine, before, task in jobs:
        if not task.done():
            _serve_last_live(name, entries)
            _finish_late(name, mine, original, before, task)
            continue
        if task.cancelled() or task.exception() is not None:
            if not task.cancelled():
                logger.warning("live model list failed (%s)", name,
                               exc_info=task.exception())
            continue
        # Each augmenter changes its own agents' schemas only.
        _merge_live(name, entries, original, mine)
        _late_model_digests[name] = _model_schemas(mine)
    return entries


def _merge_live(name: str, entries: list[dict], original: list[dict],
                mine: list[dict]) -> None:
    """Copy the schemas augmenter ``name`` changed into ``entries`` and
    remember them for a later read that misses the budget."""
    for entry, untouched, augmented in zip(entries, original, mine):
        if augmented.get("param_schema") != untouched.get("param_schema"):
            entry["param_schema"] = augmented.get("param_schema")
            _last_live_schemas.setdefault(name, {})[entry.get("slug", "")] = (
                copy.deepcopy(augmented.get("param_schema")))


def _serve_last_live(name: str, entries: list[dict]) -> None:
    """While augmenter ``name`` is still probing, serve the schemas it
    produced last time. A model saved since then stays listed."""
    for entry in entries:
        remembered = _last_live_schemas.get(name, {}).get(entry.get("slug", ""))
        if remembered is None:
            continue
        schema = copy.deepcopy(remembered)
        for param, param_schema in schema.items():
            allowed = param_schema.get("allowed") if isinstance(param_schema, dict) else None
            saved = (entry.get("config") or {}).get(param)
            if isinstance(allowed, list) and saved and saved not in allowed:
                allowed.append(saved)
                labels = param_schema.get("labels")
                if isinstance(labels, dict) and entry.get("slug") == "claude-code":
                    from agent_os.daemon_v2.sub_agent_config_store import claude_model_label
                    labels.setdefault(saved, claude_model_label(saved))
        entry["param_schema"] = schema


def _finish_late(name: str, mine: list[dict], original: list[dict], before: str,
                 task: asyncio.Task) -> None:
    _late_model_tasks.add(task)

    def _done(t: asyncio.Task) -> None:
        _late_model_tasks.discard(t)
        if t.cancelled() or t.exception() is not None:
            return
        after = _model_schemas(mine)
        if after == before:
            return
        # Remember only the agents this augmenter changed.
        for untouched, augmented in zip(original, mine):
            if augmented.get("param_schema") != untouched.get("param_schema"):
                _last_live_schemas.setdefault(name, {})[augmented.get("slug", "")] = (
                    copy.deepcopy(augmented.get("param_schema")))
        if _late_model_digests.get(name) == after:
            return
        _late_model_digests[name] = after
        _broadcast_sub_agents_updated("models")

    task.add_done_callback(_done)


def start_sub_agent_status_updates(engine, ws_manager, *, warm_after: float = 0.0) -> None:
    """Spec 136 A.2/A.3, at daemon start (on the event loop): a sweep that
    changes the agents' status is announced to every client as
    ``sub_agents.updated``, and the first sweep starts in the background, so
    the first Settings open finds a result.

    ``warm_after`` holds that sweep back so it does not compete with the boot
    the window is waiting on; a Settings read before then sweeps on its own,
    and the held-back sweep is then skipped."""
    loop = asyncio.get_running_loop()

    def _on_change(_results) -> None:
        # From the sweeping thread; broadcast_global is loop-only.
        loop.call_soon_threadsafe(ws_manager.broadcast_global,
                                  {"type": SUB_AGENTS_UPDATED, "reason": "status"})

    set_listener = getattr(engine, "set_status_listener", None)
    if callable(set_listener):
        set_listener(_on_change)
    warm = getattr(engine, "warm", None)
    if not callable(warm):
        return
    if warm_after <= 0:
        warm()
        return

    def _warm_unless_swept() -> None:
        if getattr(engine, "_check_all_cache", None) is None:
            warm()

    loop.call_later(warm_after, _warm_unless_swept)


def _refresh_agent_sync(slug: str):
    """Spec 136 A.5: re-probe the one agent a change touched into the
    cached sweep (an engine without it drops the sweep, as before)."""
    refresh = getattr(_setup_engine, "refresh_agent", None)
    if callable(refresh):
        try:
            return refresh(slug)
        except Exception:  # noqa: BLE001 — fall back to dropping the sweep
            logger.warning("re-probe of %s failed; dropping the sweep", slug,
                           exc_info=True)
    if _setup_engine is not None and hasattr(_setup_engine, "invalidate_cache"):
        _setup_engine.invalidate_cache()
    return None


async def _refresh_agent(slug: str) -> None:
    await asyncio.to_thread(_refresh_agent_sync, slug)


def _model_provider_slugs() -> list[str]:
    """Agents whose credential follows the configured model's provider (Pi):
    a saved card or model setting changes their state without a probe of
    their own."""
    registry = getattr(_setup_engine, "_registry", None)
    if registry is None:
        return []
    return [m.slug for m in registry.list_all()
            if any(c.type == "model_provider" for c in m.setup.credentials)]


def _refresh_agents_in_background(slugs: list[str]) -> None:
    """Re-probe ``slugs`` on a daemon thread; the result reaches open
    Settings screens through ``sub_agents.updated``. For saves whose reply
    must not wait on another agent's probe (cards, model settings)."""
    if not slugs or not callable(getattr(_setup_engine, "refresh_agent", None)):
        return

    def _run() -> None:
        for slug in slugs:
            _refresh_agent_sync(slug)

    threading.Thread(target=_run, name="sub-agent-reprobe", daemon=True).start()


def _offers_live_acp_models(slug: str) -> bool:
    """An ACP agent configured through a rendered file (dsh): its model
    setting is ``provider/model`` and its own ``session/new`` lists them."""
    manifest = _registry_manifest(slug)
    return bool(manifest is not None
                and getattr(manifest.runtime, "transport", "") == "acp-sdk"
                and manifest.runtime.config_template)


async def _acp_live_options(slug: str) -> dict | None:
    """Indirection over the cached live probe (monkeypatch seam for route
    tests): ``{"models": [{"value": "provider/model", "label"}],
    "efforts": [{"value", "label"}]}`` or None."""
    from agent_os.agents.acp_models import live_options
    if _setup_engine is None:
        return None
    return await asyncio.to_thread(live_options, _setup_engine, slug)


async def _augment_acp_live_models(entries: list[dict]) -> list[dict]:
    """dsh's model dropdown is dsh's own list: the providers that work in
    the user's setup (DSH Desktop's, mirrored into the spawn), grouped by
    provider. Probe failure, or an agent not installed, keeps the static
    fallback. A saved value the list no longer offers stays visible."""
    for entry in entries:
        if not entry.get("installed") or not _offers_live_acp_models(entry.get("slug", "")):
            continue
        schema = entry.get("param_schema") or {}
        live = await _acp_live_options(entry["slug"])
        if not live:
            continue
        # Model: every provider/model dsh can serve. Effort: what dsh offers
        # for the model it runs — "low" is not shown for a route that has
        # none. A saved value no longer offered stays visible either way.
        for param, choices in (("model", live.get("models")), ("effort", live.get("efforts"))):
            param_schema = schema.get(param)
            if param_schema is None or not choices:
                continue
            allowed = [c["value"] for c in choices]
            saved = (entry.get("config") or {}).get(param)
            if saved and saved not in allowed:
                allowed.append(saved)
            param_schema["allowed"] = allowed
            param_schema["labels"] = {c["value"]: c["label"] for c in choices}
    return entries


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
    # Spec 136 A.1: the last sweep at once, however old; an expired one
    # starts one background sweep whose changes arrive as
    # ``sub_agents.updated``. Built in the same thread: the entries read the
    # keychain for optional credentials.
    entries = await asyncio.to_thread(_status_entries, True)
    return await _augment_live_models(entries)


def _status_entries(allow_stale: bool, handshake_wait: float | None = None) -> list[dict]:
    check_all = _setup_engine.check_all
    kwargs = {}
    if allow_stale:
        kwargs["allow_stale"] = True
    if handshake_wait is not None and "handshake_wait" in inspect.signature(check_all).parameters:
        kwargs["handshake_wait"] = handshake_wait
    return [_build_sub_agent_status_entry(s)
            for s in check_all(**kwargs) if s.slug != "built-in"]


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
    # claude-code: the account's live model list widens the static whitelist
    # (a new generation is valid before this code learns its ids). Probe
    # unavailable → the static whitelist alone still validates.
    extra_allowed = None
    claude_model = (params.get("model") or "").strip() if slug == "claude-code" else ""
    if claude_model:
        live_claude = await _claude_live_models()
        if live_claude:
            extra_allowed = {"model": [m["value"] for m in live_claude]}
    # dsh: its own live list (providers set up in DSH Desktop) widens the
    # static DeepSeek fallback, the same way.
    acp_model = (params.get("model") or "").strip() if _offers_live_acp_models(slug) else ""
    if acp_model:
        live_acp = await _acp_live_options(slug)
        if live_acp and live_acp.get("models"):
            extra_allowed = {"model": [c["value"] for c in live_acp["models"]]}
    try:
        cleaned = _sub_agent_config_store.set(slug, params,
                                              extra_allowed=extra_allowed)
    except SubAgentConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    # Pi's credential state follows its configured model's provider.
    _refresh_agents_in_background([s for s in _model_provider_slugs() if s == slug])
    return {"slug": slug, "config": cleaned}


class LocateRequest(BaseModel):
    # What the agent answered to the locate prompt, or a bare path.
    answer: str = ""


def _locate_manifest(slug: str):
    if _setup_engine is None:
        raise HTTPException(status_code=503, detail="Setup engine not available")
    manifest = _registry_manifest(slug)
    if manifest is None or manifest.runtime.adapter == "built_in":
        raise HTTPException(status_code=404, detail=f"Unknown agent: {slug}")
    return manifest


def _app_launcher(slug: str, path: str) -> str | None:
    """The agent CLI inside a picked desktop app (``/Applications/DeepSeek
    Harness.app`` → its ``runtime/cli/bin/dsh``), per the manifest's
    ``desktop_apps`` layout; None for anything else."""
    from agent_os.agents.desktop_apps import launcher_inside
    from agent_os.daemon_v2.models import detect_os

    manifest = _registry_manifest(slug)
    if manifest is None or not manifest.setup.desktop_apps:
        return None
    return launcher_inside(manifest.setup.desktop_apps, detect_os(), path)


def _locate_test(slug: str, answer: str) -> dict:
    """Parse the pasted answer and run the agent's version check with it.
    Every failure is ``{"ok": False, "error": <text for the card>}``."""
    from agent_os.agents.locate_answer import LocateAnswerError, parse_locate_answer
    try:
        parsed = parse_locate_answer(answer)
    except LocateAnswerError as exc:
        return {"ok": False, "error": str(exc)}
    binary = _app_launcher(slug, parsed.command) or parsed.command
    result = _setup_engine.test_override(
        slug, binary, path_prepend=list(parsed.path_prepend))
    result.update({
        "binary": binary,
        "target": parsed.target,
        "path_prepend": list(parsed.path_prepend),
        "env_names": list(parsed.env_names),
    })
    return result


def _fresh_sub_agent_entry(slug: str) -> dict:
    """The listing entry for ``slug`` after a detection change — re-probed
    into the cached sweep, so the list the card refetches agrees."""
    _setup_engine.invalidate_resolved_path(slug)
    status = _refresh_agent_sync(slug)
    if status is None:
        status = _setup_engine.check_agent(slug)
    return _build_sub_agent_status_entry(status)


@router.get("/settings/sub-agents/{slug}/locate-prompt")
async def get_sub_agent_locate_prompt(slug: str):
    """The text the locate card copies: a prompt the user pastes into the
    agent so it reports its own executable, PATH needs and env names."""
    from agent_os.agents.locate_answer import locate_prompt
    manifest = _locate_manifest(slug)
    return {"slug": slug, "prompt": locate_prompt(manifest.name)}


@router.post("/settings/sub-agents/{slug}/locate/test")
async def test_sub_agent_locate(slug: str, req: LocateRequest):
    """Parse the pasted answer and run the version check against it. 200
    either way: the card shows ``version_output`` or ``error`` verbatim."""
    _locate_manifest(slug)
    return await asyncio.to_thread(_locate_test, slug, req.answer)


@router.put("/settings/sub-agents/{slug}/locate")
async def put_sub_agent_locate(slug: str, req: LocateRequest):
    """Save the pasted location. The check is re-run on the exact inputs
    first (the card-save rule): a failing check is a 400, nothing is stored."""
    manifest = _locate_manifest(slug)
    if _sub_agent_config_store is None:
        raise HTTPException(status_code=503, detail="Sub-agent config store not available")
    from agent_os.daemon_v2.sub_agent_config_store import (
        SubAgentConfigError, SubAgentConfigStore,
    )
    result = await asyncio.to_thread(_locate_test, slug, req.answer)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error") or "Test failed")
    try:
        _sub_agent_config_store.set_override(
            slug, binary=result["binary"], path_prepend=result["path_prepend"],
            env={name: "" for name in result["env_names"]},
            # An agent whose own Windows launcher is an npm .cmd (dsh) may be
            # pointed at a hand-installed .cmd too; Test already accepted it.
            allow_shim=SubAgentConfigStore.manifest_launches_via_shim(manifest))
    except SubAgentConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return await asyncio.to_thread(_fresh_sub_agent_entry, slug)


@router.delete("/settings/sub-agents/{slug}/locate")
async def delete_sub_agent_locate(slug: str):
    """Forget the user-chosen location; detection runs on its own again."""
    _locate_manifest(slug)
    if _sub_agent_config_store is None:
        raise HTTPException(status_code=503, detail="Sub-agent config store not available")
    _sub_agent_config_store.clear_override(slug)
    return await asyncio.to_thread(_fresh_sub_agent_entry, slug)


def _forget_quota(slug: str) -> None:
    """Spec 099: drop the agent's persisted subscription quota after an
    account change — the snapshot belongs to the old account."""
    from agent_os.daemon_v2.quota_store import get_quota_store
    store = get_quota_store()
    if store is not None:
        store.clear(slug)


@router.post("/settings/sub-agents/refresh")
async def refresh_sub_agent_status():
    """Invalidate the SetupEngine cache and return fresh status."""
    if _setup_engine is None:
        return []
    # Spec 115 D2: PATH was a launch-time snapshot, so a CLI installed into a
    # directory that was not on PATH then stayed invisible until relaunch.
    # Re-run the shell PATH capture first. One or two shell spawns per manual
    # Refresh — a user click, not a hot path. Imported here so tests can
    # patch the source module.
    from agent_os.platform.shell_path import merge_shell_path_into_environ
    await asyncio.to_thread(merge_shell_path_into_environ)
    if hasattr(_setup_engine, "invalidate_cache"):
        _setup_engine.invalidate_cache()
    # Spec 115 §7: resolve_binary caches hits for the daemon lifetime, so a
    # binary that moved (claude-code's self-update) kept its dead path across
    # Refresh. Forget every slug's hit; the probe below re-resolves.
    registry = getattr(_setup_engine, "_registry", None)
    if registry is not None and hasattr(_setup_engine, "invalidate_resolved_path"):
        for manifest in registry.list_all():
            _setup_engine.invalidate_resolved_path(manifest.slug)
    # A verdict is kept until the binary changes; Refresh asks again anyway
    # (a key or a network problem fixed since the last ask).
    if hasattr(_setup_engine, "forget_handshakes"):
        _setup_engine.forget_handshakes()
    # Spec 136 B.3: version answers are cached by binary identity; a CLI
    # rewritten in place with the same mtime and size is caught here.
    if hasattr(_setup_engine, "forget_versions"):
        _setup_engine.forget_versions()
    # User-triggered refresh wants CURRENT state — drop the codex model/list
    # cache too so a just-fixed install / new model generation shows up.
    from agent_os.agent.transports.codex_models import clear_codex_models_cache
    from agent_os.agent.transports.claude_models import clear_claude_models_cache
    clear_codex_models_cache()
    clear_claude_models_cache()
    from agent_os.agents.acp_models import clear_acp_models_cache
    clear_acp_models_cache()
    # The user asked for current state: a fresh sweep, waiting for the
    # handshakes (they run in parallel, so this costs the slowest one)
    # instead of reporting them pending the way the routine sweep does.
    entries = await asyncio.to_thread(_status_entries, False, HANDSHAKE_TIMEOUT_S)
    return await _augment_live_models(entries)


def _find_npm(binary: str | None) -> str | None:
    """The npm to update an npm-method install with: the one next to the CLI
    first (an nvm / fnm layout keeps ``bin/claude`` beside ``bin/npm``, and
    PATH's npm may belong to another node), then PATH. ``shutil.which``
    applies PATHEXT, so Windows gets ``npm.cmd``."""
    names = ("npm.cmd", "npm.exe", "npm") if os.name == "nt" else ("npm",)
    if binary:
        directory = os.path.dirname(binary)
        for name in names:
            candidate = os.path.join(directory, name)
            if os.path.isfile(candidate):
                return candidate
    return shutil.which("npm")


def _npm_update_command(manifest, binary: str | None) -> str | None:
    """``npm install -g <package>@latest`` for an npm-method install, or None
    (native / unknown install, no ``npm_package`` declared, npm not found).

    1018 bench item 5: ``claude update`` on an npm install gave up after 39 s
    with the package half-removed. npm run directly, under
    ``UPDATE_TIMEOUT_SECONDS``, finishes the reinstall instead. The npm path
    is quoted for the shell the job runs through (``C:\\Program Files\\...``).
    """
    package = getattr(manifest.setup, "npm_package", None)
    if not package or install_kind_of(binary) != "npm":
        return None
    npm = _find_npm(binary)
    if not npm:
        return None
    argv = [npm, "install", "-g", f"{package}@latest"]
    if os.name == "nt":
        return subprocess.list2cmdline(argv)
    import shlex
    return shlex.join(argv)


def _manual_update_command(slug: str) -> str | None:
    """The command a user can run by hand when the in-app update left them
    without the CLI, or None when the manifest names no npm package."""
    manifest = _registry_manifest(slug)
    package = getattr(getattr(manifest, "setup", None), "npm_package", None)
    return f"npm install -g {package}@latest" if package else None


def _resolve_setup_command(slug: str, action: str) -> str | None:
    """Find the manifest's command string for a setup action.

    ``action`` is one of "login", "logout", "update". Looks up the matching
    credential on the agent's manifest and returns ``setup_command`` (login)
    or ``logout_command`` (not currently in manifest schema; falls back to
    deriving from binary name for known agents). "update" (spec 116 D2) is
    the manifest's ``setup.update_command`` — the CLI's own updater — with
    the same binary substitution as login.

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
    if action == "update":
        cmd = manifest.setup.update_command
        if not cmd:
            return None
        npm_cmd = _npm_update_command(manifest, binary)
        if npm_cmd:
            return npm_cmd
        if binary and manifest.runtime.command:
            return _setup_engine._substitute_binary(
                cmd, manifest.runtime.command, binary)
        return cmd
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
        if manifest.runtime.command in ("claude", "codex", "agent", "cursor-agent", "mcode"):
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


_URL_QUERY_RE = re.compile(r"(https?://[^\s?#\"'<>]+)\?[^\s\"'<>]*")


def _redact_urls(text: str) -> str:
    """Drop the query string of every http(s) URL (spec 113 A3).

    ``claude auth login`` / ``codex login`` print OAuth URLs whose query
    carries ``code`` / ``state`` tokens; the local log may hold text, never a
    credential, so the URL keeps its path and loses everything after ``?``.
    """
    return _URL_QUERY_RE.sub(r"\1?…", text)


def _cli_diagnostics(slug: str) -> dict:
    """``{cli_version, install_kind}`` for ``slug`` without spawning anything
    (spec 113): the last status sweep's version, and the install kind judged
    from the resolved binary path. Both degrade to unknown/None."""
    version = None
    binary = None
    user_override = False
    if _setup_engine is not None:
        cached = getattr(_setup_engine, "cached_status", None)
        status = cached(slug) if callable(cached) else None
        if status is not None:
            version = getattr(status, "version", None)
            binary = getattr(status, "binary_path", None)
            user_override = bool(getattr(status, "override_path", None)) and binary is not None
        if binary is None:
            get_path = getattr(_setup_engine, "get_resolved_path", None)
            binary = get_path(slug) if callable(get_path) else None
    return {"cli_version": clean_version(version),
            "install_kind": install_kind_of(binary, user_override=user_override)}


def _prune_login_jobs() -> None:
    """Keep every running job plus the newest ``LOGIN_JOB_HISTORY`` finished
    ones (insertion order == start order)."""
    finished = [jid for jid, job in _login_jobs.items() if job.get("status") != "running"]
    for jid in finished[:-LOGIN_JOB_HISTORY] if len(finished) > LOGIN_JOB_HISTORY else []:
        _login_jobs.pop(jid, None)


def _login_job_view(job_id: str, job: dict) -> dict:
    """The read-only shape of one job entry (``GET .../login-jobs``): no
    command string (it carries the binary path) and no pid."""
    return {
        "job_id": job_id,
        "slug": job.get("slug"),
        "kind": job.get("kind", "login"),
        "status": job.get("status"),
        "reason": job.get("reason"),
        "return_code": job.get("return_code"),
        "timed_out": bool(job.get("timed_out")),
        "verified": job.get("verified"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "duration_ms": job.get("duration_ms"),
        "tail": list(job.get("tail") or []),
    }


async def _run_login_job(slug: str, job_id: str, command: str,
                         kind: str = "login") -> None:
    """Background task: run the CLI login (or update) command, broadcast progress.

    Captures stdout+stderr line by line and emits sanitised ``login.progress``
    events over WebSocket. Final ``login.complete`` or ``login.failed`` carries
    the return code; ``login.complete`` also carries ``verified``, the result
    of re-running the manifest's credential check afterwards. An unverified
    completion is deliberately NOT a failure — a slow keychain write must not
    turn a real login into a reported error — but it is reported honestly so
    the UI can say "signed in, but couldn't confirm".

    ``kind`` (spec 116 D2): ``"update"`` runs the manifest's
    ``update_command`` through this same runner. Every event carries
    ``kind`` so the card can tell the two apart; the post-run verification
    differs (D3): an update is verified when the RE-PROBED version is at or
    above the floor — never by the exit code, since a pre-2.1.288 npm
    updater reports success with only the stub installed.

    Spec 113 A1/A3: every terminal emits its reason (``spawn_error`` /
    ``timeout`` / ``nonzero_exit``; ``unverified`` is its own event for a
    login and a failure for an update) with structured diagnostics, writes
    ONE daemon.log line (reason, return code, the last few output lines with
    URL queries redacted), and the job entry keeps the same for the
    login-jobs read route.

    Tokens from this flow never enter Orbital's storage — the CLI writes them
    to its own location. (Agents with no credential store of their own don't
    come through here at all; they use the ``/credential`` routes.)
    """
    started = time.monotonic()
    job = _login_jobs.setdefault(job_id, {"slug": slug, "status": "running"})
    job["kind"] = kind
    job.setdefault("started_at", _now_iso())
    tail: list[str] = []
    # Snapshot the CLI version / install kind NOW: the terminal paths below
    # invalidate the status cache before they report, so reading it then
    # would yield nothing. An update's terminal refreshes this from the
    # re-probed status (the version that matters is the one after).
    diagnostics = _cli_diagnostics(slug)
    telemetry.emit(f"{kind}_attempted", {"agent": slug})

    def _finish(status: str, *, reason: str | None, return_code: int | None,
                timed_out: bool = False, verified: bool | None = None,
                exc: BaseException | None = None) -> None:
        duration_ms = int((time.monotonic() - started) * 1000)
        job.update({
            "status": status, "reason": reason, "return_code": return_code,
            "timed_out": timed_out, "verified": verified,
            "finished_at": _now_iso(), "duration_ms": duration_ms,
            "tail": list(tail),
        })
        _prune_login_jobs()
        logger.warning(
            "%s job %s for %s ended: status=%s reason=%s return_code=%s "
            "timed_out=%s verified=%s duration_ms=%d; last output: %r",
            kind, job_id, slug, status, reason, return_code, timed_out,
            verified, duration_ms, tail,
        )
        if reason is None:
            return
        # Telemetry (spec 113): enums + numbers only. ``unverified`` is a
        # separate, non-failure event for a login; an update that leaves the
        # CLI below the floor is a failure (116 D3).
        fields = {
            "agent": slug, "reason": reason, "exit_code": return_code,
            "duration_ms": duration_ms, "error_class": error_class_of(exc),
            # Output first; failing that the exception text; failing that
            # the reason itself (a silent timeout still reads as "timeout").
            "stderr_category": classify_stderr(
                "\n".join(tail) or (str(exc) if exc else "") or reason),
            **diagnostics,
        }
        if kind == "login" and reason == "unverified":
            telemetry.emit("login_unverified", fields)
        else:
            telemetry.emit(f"{kind}_failed", fields)

    is_update = kind == "update"
    # An update runs against a wall-clock budget (npm is silent while it
    # downloads); a login against an idle timeout (the user is in a browser).
    deadline = started + UPDATE_TIMEOUT_SECONDS if is_update else None

    def _drop_cached_detection() -> None:
        # Re-check status now that the auth state / the install may have
        # changed. This has to come before the re-check below so the list the
        # UI refetches and the verdict we broadcast read the same fresh state.
        if _setup_engine is not None and hasattr(_setup_engine, "invalidate_cache"):
            _setup_engine.invalidate_cache()
        if is_update and _setup_engine is not None                 and hasattr(_setup_engine, "invalidate_resolved_path"):
            # The updater may have moved the binary (npm → native migration
            # into ~/.local/bin) or removed it; the next probe must find the
            # new one, or none (116 D3, 1018 bench item 5).
            _setup_engine.invalidate_resolved_path(slug)

    async def _reprobe_update():
        """Every update terminal re-probes the CLI (1018 bench item 5: a
        failed npm step can take the CLI with it). Returns
        ``(verified, cli_missing)``; a probe that could not run is neither."""
        nonlocal diagnostics
        _drop_cached_detection()
        verified, after = await _update_now_sufficient(slug)
        if after is None:
            return False, False
        diagnostics = {
            "cli_version": clean_version(getattr(after, "version", None)),
            "install_kind": install_kind_of(
                getattr(after, "binary_path", None),
                user_override=bool(getattr(after, "override_path", None))
                and getattr(after, "binary_path", None) is not None),
        }
        missing = not (getattr(after, "installed", False) and getattr(after, "version", None))
        return verified, missing

    def _fail(reason: str, *, return_code: int | None, timed_out: bool = False,
              exc: BaseException | None = None, cli_missing: bool = False,
              **extra) -> None:
        if cli_missing:
            reason = CLI_MISSING_AFTER_UPDATE
        _finish("failed", reason=reason, return_code=return_code,
                timed_out=timed_out, verified=False if is_update else None, exc=exc)
        if _ws_manager is None:
            return
        event = {
            "type": "login.failed",
            "job_id": job_id,
            "slug": slug,
            "kind": kind,
            "reason": reason,
            **extra,
        }
        if return_code is not None:
            event["return_code"] = return_code
        if timed_out:
            event["timed_out"] = True
        if is_update:
            event["verified"] = False
        if cli_missing:
            manual = _manual_update_command(slug)
            if manual:
                event["manual_command"] = manual
        _ws_manager.broadcast_global(event)

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            creationflags=win_no_window_flags(),
        )
    except OSError as exc:
        tail.append(_redact_urls(str(exc)))
        missing = (await _reprobe_update())[1] if is_update else False
        _fail("spawn_error", return_code=None, exc=exc, cli_missing=missing,
              error=str(exc))
        return

    job["pid"] = proc.pid
    assert proc.stdout is not None
    timed_out = False
    while True:
        if deadline is not None:
            wait = deadline - time.monotonic()
            if wait <= 0:
                timed_out = True
                break
        else:
            wait = LOGIN_IDLE_TIMEOUT_SECONDS
        try:
            raw = await asyncio.wait_for(proc.stdout.readline(), timeout=wait)
        except asyncio.TimeoutError:
            # Login: idle too long, the user walked away mid-OAuth. Update:
            # past its budget. Kill it rather than hold the subprocess for
            # the daemon's lifetime.
            timed_out = True
            break
        if not raw:
            break
        line = _strip_terminal_escapes(raw.decode("utf-8", errors="replace"))
        if line:
            tail.append(_redact_urls(line))
            del tail[:-LOGIN_JOB_TAIL_LINES]
        if _ws_manager is not None and line:
            _ws_manager.broadcast_global({
                "type": "login.progress",
                "job_id": job_id,
                "slug": slug,
                "kind": kind,
                "line": line,
            })

    if timed_out:
        # The command runs through the shell, so the CLI is the shell's
        # child: proc.kill() alone ended only cmd.exe on Windows and left
        # `claude auth login` running (and holding the pipe open, so this
        # wait blocked until it exited on its own).
        await kill_spawned_tree(proc, label=f"{slug} {kind}")
        if is_update:
            missing = (await _reprobe_update())[1]
            error = f"update timed out after {int(UPDATE_TIMEOUT_SECONDS)}s"
        else:
            missing = False
            error = (f"{kind} timed out after "
                     f"{int(LOGIN_IDLE_TIMEOUT_SECONDS)}s with no activity")
        _fail("timeout", return_code=None, timed_out=True,
              exc=asyncio.TimeoutError(), cli_missing=missing, error=error)
        return

    rc = await proc.wait()
    job["return_code"] = rc

    if is_update:
        verified, missing = await _reprobe_update()
        if missing:
            _fail("nonzero_exit" if rc != 0 else "unverified", return_code=rc,
                  cli_missing=True)
            return
    elif _setup_engine is not None:
        # Spec 136 A.5: this agent's sign-in changed; re-probe it alone into
        # the cached sweep before the terminal event makes the UI refetch.
        await _refresh_agent(slug)

    if rc != 0:
        _fail("nonzero_exit", return_code=rc)
        return

    if not is_update:
        _forget_quota(slug)
        verified = await _credentials_now_configured(slug)
        if verified:
            _clear_auth_failure(slug)
    _finish("complete" if verified else "unverified",
            reason=None if verified else "unverified", return_code=rc,
            verified=verified)

    if _ws_manager is not None:
        _ws_manager.broadcast_global({
            "type": "login.complete",
            "job_id": job_id,
            "slug": slug,
            "kind": kind,
            "return_code": rc,
            "verified": verified,
        })


def _clear_auth_failure(slug: str) -> None:
    """Drop the engine's expired-sign-in record for ``slug`` (a confirmed
    login or a logout); a no-op on engines without one."""
    clear = getattr(_setup_engine, "clear_auth_failure", None)
    if callable(clear):
        clear(slug)


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


async def _update_now_sufficient(slug: str):
    """Spec 116 D3: after ``<cli> update`` exits, re-probe the version and
    compare it with the manifest floor. Exit 0 is necessary, not sufficient
    — a 2.1.288-era npm updater reports success with only the stub in place
    (then ``--version`` fails and the agent reads as not installed).

    Returns ``(verified, status)``; ``status`` is None when no probe ran.
    """
    if _setup_engine is None or not hasattr(_setup_engine, "check_agent"):
        return False, None
    try:
        status = await asyncio.to_thread(_setup_engine.check_agent, slug)
    except Exception:
        logger.exception("post-update version re-check failed for %s", slug)
        return False, None
    return bool(status.installed and status.version
                and not getattr(status, "update_needed", False)), status


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
    _login_jobs[job_id] = {"slug": slug, "status": "running", "command": command,
                           "kind": "login", "started_at": _now_iso()}
    asyncio.create_task(_run_login_job(slug, job_id, command))
    return {"job_id": job_id, "slug": slug, "status": "running"}


def _worker_running(slug: str) -> bool:
    """Spec 116 D8: is a ``slug`` worker alive in any project? Replacing the
    binary under a live process is what the update must not do. Unknown
    (no manager wired) reads as not running."""
    if _sub_agent_manager is None:
        return False
    probe = getattr(_sub_agent_manager, "has_live_adapter", None)
    if not callable(probe):
        return False
    try:
        return bool(probe(slug))
    except Exception:
        logger.warning("could not tell whether %s is running", slug, exc_info=True)
        return False


@router.post("/settings/sub-agents/{slug}/update")
async def trigger_sub_agent_update(slug: str):
    """Run the CLI's own updater (spec 116 D2) through the login-job runner.

    Returns immediately with a ``job_id``; progress streams over WebSocket
    as ``login.progress`` / ``login.complete`` / ``login.failed`` with
    ``kind: "update"``. ``login.complete.verified`` is whether the re-probed
    version reached the manifest floor (D3). 400 when the manifest declares
    no ``update_command``; 409 while a worker of this agent is running (D8).
    """
    command = _resolve_setup_command(slug, "update")
    if not command:
        raise HTTPException(
            status_code=400,
            detail=f"sub-agent '{slug}' does not expose an update command",
        )
    if _worker_running(slug):
        raise HTTPException(
            status_code=409,
            detail=f"a {slug} task is running; wait for it to finish before updating",
        )
    job_id = uuid4().hex[:12]
    _login_jobs[job_id] = {"slug": slug, "status": "running", "command": command,
                           "kind": "update", "started_at": _now_iso()}
    asyncio.create_task(_run_login_job(slug, job_id, command, kind="update"))
    return {"job_id": job_id, "slug": slug, "status": "running"}


@router.get("/settings/sub-agents/login-jobs")
async def list_login_jobs():
    """The login/update jobs this daemon has run (spec 113 A3): the running
    ones plus the last 20 finished, newest last, each with its reason,
    return code and redacted output tail. Read-only; for the "view the last
    login log" affordance and for pasting into an issue."""
    return [_login_job_view(jid, job) for jid, job in _login_jobs.items()]


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

    _forget_quota(slug)
    if result.returncode == 0:
        _clear_auth_failure(slug)
    if _setup_engine is not None:
        await _refresh_agent(slug)

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
        # Spec 082: "the global LLM key" is now the DEFAULT CARD's key, and
        # the provider that owns it is the card's — settings.llm.provider is a
        # derived, one-release compatibility field that migration drops, so
        # reading it here would report "custom" for every migrated install and
        # refuse a copy that should succeed.
        card = _settings_store.default_card() if _settings_store else None
        provider = card.provider if card is not None else "custom"
        if provider != "deepseek":
            raise HTTPException(status_code=409, detail=(
                f"the global LLM key belongs to provider '{provider}', not "
                f"deepseek — paste the key for '{slug}' directly instead"))
        supplied = (_settings_store.key_for(card.id) or "").strip()
        if not supplied:
            raise HTTPException(status_code=409,
                                detail="no global LLM API key is configured")

    try:
        store.set(req.key, supplied)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    await _refresh_agent(slug)
    _forget_quota(slug)

    return {"slug": slug, "key": req.key, "set": True,
            "masked": _mask_secret(supplied)}


@router.delete("/settings/sub-agents/{slug}/credential/{key}")
async def delete_sub_agent_credential(slug: str, key: str):
    """Remove a stored sub-agent credential from the keychain."""
    manifest = _sub_agent_manifest(slug)
    _validate_credential_key(manifest, key)
    _sub_agent_credential_store().delete(key)

    await _refresh_agent(slug)
    _forget_quota(slug)

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

    if _setup_engine is not None:
        await _refresh_agent(slug)
    _forget_quota(slug)

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
