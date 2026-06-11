# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""REST endpoints for Agent OS v2 API.

All endpoints use /api/v2/ prefix. No v1 routes. snake_case in request/response.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import socket
import tempfile
import time
import zipfile
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from fastapi.responses import JSONResponse
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from agent_os.agent.prompt_builder import Autonomy
from agent_os.agent.skills import SkillLoader
from agent_os.daemon_v2.default_skills_installer import install_default_skills
from agent_os.daemon_v2.sub_agent_transcript import read_sub_agent_summary
from agent_os.agent.project_paths import ProjectPaths
from agent_os.api.routes._attachment_formatter import (
    validate_attachments,
    format_prefix,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2")


# ---- Request/Response models ----

class CreateProjectRequest(BaseModel):
    name: str
    workspace: str
    model: str
    api_key: str
    base_url: str | None = None
    autonomy: str | None = None
    instructions: str | None = None
    provider: str | None = None
    sdk: str | None = None
    agent_slug: str | None = None
    enabled_sub_agents: list[str] | None = None
    disabled_sub_agents: list[str] | None = None
    agent_credentials: dict | None = None
    network_extra_domains: list[str] | None = None
    agent_name: str | None = None
    is_scratch: bool = False
    notification_prefs: dict | None = None
    llm_fallback_models: list[dict] | None = None
    budget_limit_usd: float | None = None
    budget_action: str | None = None
    budget_period: str | None = None
    budget_currency: str | None = None


class ProjectUpdate(BaseModel):
    name: str | None = None
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    autonomy: str | None = None
    instructions: str | None = None
    provider: str | None = None
    sdk: str | None = None
    agent_slug: str | None = None
    enabled_sub_agents: list[str] | None = None
    disabled_sub_agents: list[str] | None = None
    agent_credentials: dict | None = None
    network_extra_domains: list[str] | None = None
    agent_name: str | None = None
    project_goals_content: str | None = None
    user_directives_content: str | None = None
    notification_prefs: dict | None = None
    llm_fallback_models: list[dict] | None = None
    budget_limit_usd: float | None = None
    budget_action: str | None = None
    runtime_budget_spent_usd: float | None = None
    budget_spent_usd: float | None = None  # Alias for runtime_budget_spent_usd
    # Budget Piece 1, Task 4 — additive derived-cost window config.
    budget_period: str | None = None  # "daily"|"weekly"|"monthly"|"total"
    budget_currency: str | None = None  # ISO 4217; explicit set overrides the
    #                                     provider-derived default
    # "reset spend window" = set budget_anchor_ts to now. A direct ISO value is
    # also honored (mirrors the legacy runtime reset's direct-value pattern).
    reset_budget_anchor: bool | None = None
    budget_anchor_ts: str | None = None


class StartAgentRequest(BaseModel):
    project_id: str
    initial_message: str | None = None
    # Optional Format-1 user-facing chat id. Omitting it means
    # "use ``DEFAULT_SESSION_ID``" — see
    # ``TASK/ACTIVE-session-and-queue-model.md`` and the F7 audit.
    session_id: str | None = None


_MIME_PATTERN = re.compile(r"^[\w.+-]+/[\w.+-]+$")


class InjectAttachment(BaseModel):
    """A single attachment reference passed to /inject.

    The path is workspace-relative (the upload endpoint stores files under
    ``uploads/`` by default). The route validates the path resolves inside
    the workspace and that the declared size matches the file on disk
    before building the ``<attached_files>...</attached_files>`` prefix.
    """

    path: str
    mime: str
    size: int

    @field_validator("path")
    @classmethod
    def reject_absolute_or_traversal(cls, v: str) -> str:
        if not v or len(v) > 1024:
            raise ValueError("path empty or too long")
        if v.startswith("/") or v.startswith("\\"):
            raise ValueError("path must be relative")
        if "\x00" in v:
            raise ValueError("path contains NUL")
        # Normalise backslash separators before splitting so a Windows-style
        # relative path like ``uploads\\foo.png`` is treated the same as
        # ``uploads/foo.png`` for the purpose of '..' detection.
        parts = v.replace("\\", "/").split("/")
        if any(p == ".." for p in parts):
            raise ValueError("path may not contain '..' segments")
        return v

    @field_validator("mime")
    @classmethod
    def validate_mime(cls, v: str) -> str:
        if not _MIME_PATTERN.match(v):
            raise ValueError("invalid mime type")
        return v

    @field_validator("size")
    @classmethod
    def validate_size(cls, v: int) -> int:
        if v < 0 or v > 10 * 1024 * 1024:
            raise ValueError("size out of range (0..10MB)")
        return v


_MIME_PATTERN = re.compile(r"^[\w.+-]+/[\w.+-]+$")


class InjectAttachment(BaseModel):
    """A single attachment reference passed to /inject.

    The path is workspace-relative (the upload endpoint stores files under
    ``uploads/`` by default). The route validates the path resolves inside
    the workspace and that the declared size matches the file on disk
    before building the ``<attached_files>...</attached_files>`` prefix.
    """

    path: str
    mime: str
    size: int

    @field_validator("path")
    @classmethod
    def reject_absolute_or_traversal(cls, v: str) -> str:
        if not v or len(v) > 1024:
            raise ValueError("path empty or too long")
        if v.startswith("/") or v.startswith("\\"):
            raise ValueError("path must be relative")
        if "\x00" in v:
            raise ValueError("path contains NUL")
        # Normalise backslash separators before splitting so a Windows-style
        # relative path like ``uploads\\foo.png`` is treated the same as
        # ``uploads/foo.png`` for the purpose of '..' detection.
        parts = v.replace("\\", "/").split("/")
        if any(p == ".." for p in parts):
            raise ValueError("path may not contain '..' segments")
        return v

    @field_validator("mime")
    @classmethod
    def validate_mime(cls, v: str) -> str:
        if not _MIME_PATTERN.match(v):
            raise ValueError("invalid mime type")
        return v

    @field_validator("size")
    @classmethod
    def validate_size(cls, v: int) -> int:
        if v < 0 or v > 10 * 1024 * 1024:
            raise ValueError("size out of range (0..10MB)")
        return v


class InjectRequest(BaseModel):
    content: str
    target: str | None = None
    nonce: str | None = None
    attachments: list[InjectAttachment] | None = None
    # F1 (user-facing chat thread id) — select which chat session within
    # the project to deliver the message to. Omitting it means "use
    # ``DEFAULT_SESSION_ID``", i.e. single-loop back-compat. See
    # ``TASK/ACTIVE-session-and-queue-model.md`` and
    # ``TASK/INVESTIGATION-session-id-canonical-audit.md`` (F7).
    #
    # Also drives the slot-enforcement guard (Track J Phase 1): if a
    # different session_id is provided AND a different session currently
    # holds the project's active-loop slot, the inject route returns 202
    # with a `slot_held` payload instead of routing the message. Absent
    # or matching session_id skips the check. Per ACTIVE-session-and-queue-
    # model.md §3.
    session_id: str | None = None

    @field_validator("content")
    @classmethod
    def reject_encoding_corruption(cls, v: str) -> str:
        if "\ufffd" in v:
            raise ValueError(
                "Request contains invalid UTF-8 characters (possible terminal"
                " encoding issue). Use Python with explicit UTF-8 encoding for"
                " non-ASCII text, or send from the desktop app."
            )
        return v

    @field_validator("attachments")
    @classmethod
    def cap_attachment_count(cls, v):
        if v is not None and len(v) > 10:
            raise ValueError("too many attachments (max 10)")
        return v


class ApproveRequest(BaseModel):
    tool_call_id: str
    reply_text: str | None = None
    approve_all: bool = False
    response_payload: str | None = None  # User text input for MFA codes etc.
    session_id: str | None = None  # which session's approval (defaults to holder)


class DenyRequest(BaseModel):
    tool_call_id: str
    reason: str
    session_id: str | None = None  # which session's approval (defaults to holder)
    # Codex "Deny & stop": route decision="cancel" (turn ends `interrupted`)
    # instead of the default decline (turn continues). Ignored by transports
    # without a decision vocabulary.
    stop_turn: bool = False


class SessionScopedRequest(BaseModel):
    """Body for lifecycle verbs that act on a specific chat session.

    ``session_id`` identifies which session in the project to act on, so the
    UI can target the session it has open (not just the default sentinel).
    Optional for backward compatibility; the frontend passes the active id.
    """
    session_id: str | None = None


class TriggerToggleRequest(BaseModel):
    enabled: bool


class BulkDeleteRequest(BaseModel):
    prefix: str | None = None
    project_ids: list[str] | None = None
    before: str | None = None  # ISO datetime string


# ---- Dependency holders (set during app creation) ----

_project_store = None
_agent_manager = None
_ws_manager = None
_sub_agent_manager = None
_setup_engine = None
_settings_store = None
_credential_store = None
_trigger_manager = None
_provider_registry = None
_lifecycle_observer = None


def configure(project_store, agent_manager, ws_manager, sub_agent_manager=None,
              setup_engine=None, settings_store=None, credential_store=None,
              trigger_manager=None, provider_registry=None, lifecycle_observer=None):
    """Called by app factory to inject dependencies."""
    global _project_store, _agent_manager, _ws_manager, _sub_agent_manager, _setup_engine, _settings_store, _credential_store, _trigger_manager, _provider_registry, _lifecycle_observer
    _project_store = project_store
    _agent_manager = agent_manager
    _ws_manager = ws_manager
    _sub_agent_manager = sub_agent_manager
    _setup_engine = setup_engine
    _settings_store = settings_store
    _credential_store = credential_store
    _trigger_manager = trigger_manager
    _provider_registry = provider_registry
    _lifecycle_observer = lifecycle_observer


def _provider_currency(provider: str, model: str) -> str:
    """Resolve the ISO-4217 currency for a project's primary provider/model.

    Used when a budget limit is set without an explicit ``budget_currency`` —
    the limit defaults to the provider's currency at that moment and is never
    auto-changed afterward. Resolves via the tiered-rates table (which already
    carries the override-aware currency) and falls back to USD on any failure.
    """
    try:
        from agent_os.agent.pricing import resolve_rates
        return resolve_rates(provider, model).currency
    except Exception:  # noqa: BLE001 — currency default must never 500 a save
        logger.warning(
            "currency resolution failed for provider=%s model=%s; defaulting USD",
            provider, model, exc_info=True,
        )
        return "USD"


# ---- Session cache for sub-agent-only projects ----
_sub_agent_sessions: dict = {}  # project_id -> Session


def _get_or_create_session(project_id: str, workspace: str):
    """Get or create a session for sub-agent-only projects.

    Management-agent projects use _agent_manager.get_session().
    Sub-agent-only projects need their own session for chat persistence.
    """
    # Try management agent session first
    session = _agent_manager.get_session(project_id)
    if session is not None:
        return session

    # Use cached sub-agent session
    if project_id in _sub_agent_sessions:
        return _sub_agent_sessions[project_id]

    # Create new session for this project
    from uuid import uuid4
    from agent_os.agent.session import Session

    # F2 storage stem for the sub-agent-only session (the management-agent
    # session doesn't exist yet). F1 defaults to DEFAULT_SESSION_ID via
    # ``Session.new``'s default.
    session_uuid = f"subagent_{uuid4().hex[:8]}"
    session = Session.new(session_uuid, workspace)
    _sub_agent_sessions[project_id] = session
    return session


# ---- Helpers ----

def _workspace_is_empty(workspace: str) -> bool:
    """True if the workspace has no user content (ignoring the orbital/ scaffold
    and dotfiles). Used by the frontend to decide whether to offer a cold-start scan.
    """
    if not workspace or not os.path.isdir(workspace):
        return True
    try:
        for name in os.listdir(workspace):
            if name == "orbital" or name.startswith("."):
                continue
            return False
    except OSError:
        return True
    return True


def _redact_project(project: dict) -> dict:
    """Return project dict with api_key masked."""
    from agent_os.daemon_v2.project_store import DEFAULT_NOTIFICATION_PREFS
    result = dict(project)
    key = result.get("api_key", "")
    if key and len(key) > 8:
        result["api_key"] = key[:4] + "..." + key[-4:]
    elif key:
        result["api_key"] = "****"
    prefs = result.get("notification_prefs", {})
    result["notification_prefs"] = {**DEFAULT_NOTIFICATION_PREFS, **prefs}
    result["is_empty_workspace"] = _workspace_is_empty(result.get("workspace", ""))
    # Budget Piece 2 migration at the READ point: a legacy project that still
    # persists budget_action="ask" (and was never re-saved) surfaces as the
    # migrated "pause" in every GET response. Only normalize when the key is
    # present so we never inject a default onto a project that never set one.
    if "budget_action" in result:
        from agent_os.budget.guard import normalize_budget_action
        result["budget_action"] = normalize_budget_action(result["budget_action"])
    return result


def _read_file_or_empty(path: str) -> str:
    """Read a file and return its content, or empty string if missing."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except (OSError, FileNotFoundError):
        return ""


def _enrich_with_disk_content(result: dict, workspace: str) -> dict:
    """Attach project_goals_content and user_directives_content from disk files."""
    from agent_os.agent.project_paths import ProjectPaths
    pp = ProjectPaths(workspace)
    result["project_goals_content"] = _read_file_or_empty(pp.project_goals)
    result["user_directives_content"] = _read_file_or_empty(pp.user_directives)
    return result


def _write_workspace_file(workspace: str, filename: str, content: str) -> None:
    """Write content to {workspace}/orbital/instructions/{filename}."""
    from agent_os.agent.project_paths import ProjectPaths
    pp = ProjectPaths(workspace)
    os.makedirs(pp.instructions_dir, exist_ok=True)
    if filename == "project_goals.md":
        filepath = pp.project_goals
    elif filename == "user_directives.md":
        filepath = pp.user_directives
    else:
        filepath = os.path.join(pp.instructions_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def _maybe_sync_instructions_to_goals(workspace: str, *, goals_content: str | None,
                                      instructions: str | None) -> None:
    """Sync the legacy `instructions` field to project_goals.md, guarded.

    Explicit goals_content always wins. Otherwise the legacy field only seeds
    project_goals.md when it does NOT already exist — so a scan- or onboarding-
    authored file is never clobbered by a later unrelated Settings save.
    """
    effective = goals_content
    if effective is None and instructions is not None:
        if os.path.exists(ProjectPaths(workspace).project_goals):
            return  # disk is canonical; do not clobber
        effective = instructions
    if effective is not None:
        _write_workspace_file(workspace, "project_goals.md", effective)


# ---- Project Endpoints ----

@router.post("/projects", status_code=201)
async def create_project(req: CreateProjectRequest):
    if not os.path.isdir(req.workspace):
        raise HTTPException(status_code=400, detail="Workspace path does not exist")
    # Only persist api_key if it differs from the current global key (BYOK).
    # If it matches the global key, store empty string so the project inherits
    # the global key at runtime rather than snapshotting a stale copy.
    api_key_to_store = req.api_key
    if _credential_store is not None:
        global_key = _credential_store.get_api_key()
        if global_key and req.api_key == global_key:
            api_key_to_store = ""
    project_data = {
        "name": req.name,
        "workspace": req.workspace,
        "model": req.model,
        "api_key": api_key_to_store,
        "base_url": req.base_url,
        "autonomy": req.autonomy or "hands_off",
        "instructions": req.instructions or "",
        "provider": req.provider or "custom",
        "sdk": req.sdk or "openai",
        "is_scratch": req.is_scratch,
    }
    if req.agent_name is not None:
        project_data["agent_name"] = req.agent_name
    if req.agent_slug is not None:
        project_data["agent_slug"] = req.agent_slug
    if req.enabled_sub_agents is not None:
        project_data["enabled_sub_agents"] = req.enabled_sub_agents
    if req.disabled_sub_agents is not None:
        project_data["disabled_sub_agents"] = req.disabled_sub_agents
    if req.agent_credentials is not None:
        project_data["agent_credentials"] = req.agent_credentials
    if req.network_extra_domains is not None:
        project_data["network_extra_domains"] = req.network_extra_domains
    if req.notification_prefs is not None:
        project_data["notification_prefs"] = req.notification_prefs
    if req.llm_fallback_models is not None:
        project_data["llm_fallback_models"] = req.llm_fallback_models
    if req.budget_limit_usd is not None:
        project_data["budget_limit_usd"] = req.budget_limit_usd
    if req.budget_action is not None:
        # Budget Piece 2 migration: normalize on SAVE so persisted configs
        # converge ("ask"→"pause", unknown→"pause", "stop" stays).
        from agent_os.budget.guard import normalize_budget_action
        project_data["budget_action"] = normalize_budget_action(req.budget_action)
    if req.budget_period is not None:
        project_data["budget_period"] = req.budget_period
    # The limit owns its currency. If a limit is set at creation and no explicit
    # currency was supplied, default it from the primary provider's currency
    # (resolved once, here). It is NEVER auto-changed afterward.
    if req.budget_currency is not None:
        project_data["budget_currency"] = req.budget_currency
    elif req.budget_limit_usd is not None:
        project_data["budget_currency"] = _provider_currency(
            project_data.get("provider", "custom"), project_data.get("model", "")
        )
    try:
        pid = _project_store.create_project(project_data)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    # Seed default skills via the shared installer. Any failure here must not
    # fail project creation — missing skills will reconcile on first agent
    # start because `default_skills_reconciled` stays False.
    try:
        install_default_skills(_project_store, pid)
    except Exception:
        logger.error(
            "default skills install failed during create_project for %s",
            pid, exc_info=True,
        )

    project = _project_store.get_project(pid)
    # Create the project's dispatcher now (project-scoped lifecycle manager) so
    # queue work can be picked up without waiting for a daemon restart.
    try:
        await _agent_manager._ensure_dispatcher(pid, project.get("workspace", ""))
    except Exception:
        logger.warning(
            "failed to create dispatcher for new project %s", pid, exc_info=True,
        )
    return _redact_project(project)


@router.get("/projects")
async def list_projects():
    projects = [_redact_project(p) for p in _project_store.list_projects()]
    projects.sort(key=lambda p: (not p.get("is_scratch", False),))
    for p in projects:
        p.setdefault("agent_name", p.get("name", ""))
        p.setdefault("is_scratch", False)
    return projects


@router.get("/projects/{project_id}")
async def get_project(project_id: str):
    project = _project_store.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    workspace = project.get("workspace", "")
    result = _redact_project(project)
    _enrich_with_disk_content(result, workspace)
    result.setdefault("agent_name", result.get("name", ""))
    result.setdefault("is_scratch", False)
    # Flatten runtime budget spend for frontend consumption
    result["budget_spent_usd"] = result.get("runtime", {}).get("budget_spent_usd", 0.0)
    return result


@router.put("/projects/{project_id}")
async def update_project(project_id: str, body: ProjectUpdate):
    project = _project_store.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    # Inspect the RAW body (pre-None-filter) so we can detect a reset request
    # that arrives as the legacy ``budget_spent_usd: 0`` sentinel — 0 is not
    # None, but the None-filter below would still keep it; we read presence here
    # to decide reset intent before that field is popped as config.
    raw_body = body.model_dump()
    updates = {k: v for k, v in raw_body.items() if v is not None}
    # Budget Piece 2 / P2-D: the legacy ``budget_spent_usd`` runtime accumulator
    # is GONE — the token ledger is now the single source of recorded spend and
    # nothing writes ``runtime.budget_spent_usd`` anymore. The legacy "reset
    # spend" request fields are now REPURPOSED, not retired: the existing UI's
    # reset button still PUTs ``{budget_spent_usd: 0}`` (see web/src/components/
    # SettingsView.tsx:293), and that must keep working. We UNIFY the three reset
    # surfaces — ``reset_budget_anchor: true`` (Piece 1), legacy
    # ``budget_spent_usd``, legacy ``runtime_budget_spent_usd`` — into a single
    # "reset the ledger spend WINDOW" operation: set ``budget_anchor_ts = now``,
    # but ONLY in ``budget_period == "total"`` mode (the only window an anchor
    # affects). In any non-total mode the reset is a NO-OP that surfaces the
    # machine code ``not_total_mode`` (codes only, no sentences — i18n rule).
    #
    # The fields are mapped-and-consumed here so they never fall through to
    # ``update_project`` as bogus config keys.
    legacy_reset_requested = (
        raw_body.get("budget_spent_usd") is not None
        or raw_body.get("runtime_budget_spent_usd") is not None
    )
    updates.pop("runtime_budget_spent_usd", None)
    updates.pop("budget_spent_usd", None)

    # Budget Piece 2 migration: normalize budget_action on SAVE so persisted
    # configs converge ("ask"→"pause", unknown→"pause", "stop" stays).
    if "budget_action" in updates:
        from agent_os.budget.guard import normalize_budget_action
        updates["budget_action"] = normalize_budget_action(updates["budget_action"])

    # --- Reset spend window (P2-D unification) ---
    # "reset spend window" = set budget_anchor_ts to now, total-mode-only. A
    # direct ISO budget_anchor_ts value is also honored (unconditionally — that
    # is an explicit anchor set, not the reset operation). The bool flag is a
    # control input, not a persisted field — pop it either way.
    reset_anchor_flag = updates.pop("reset_budget_anchor", None)
    reset_requested = bool(reset_anchor_flag) or legacy_reset_requested
    # The effective window after this PUT (an incoming period change wins over
    # the stored one, so a single PUT that sets total + resets behaves sanely).
    eff_period = updates.get("budget_period", project.get("budget_period")) or "daily"
    # ``budget_reset`` is the machine-readable outcome the frontend can read off
    # the PUT response (codes only): applied flag, the code on a no-op, and the
    # new anchor when applied. None when no reset was requested.
    budget_reset: dict | None = None
    if reset_requested:
        if eff_period == "total":
            new_anchor = datetime.now(timezone.utc).isoformat()
            updates["budget_anchor_ts"] = new_anchor
            budget_reset = {
                "applied": True, "code": None, "budget_anchor_ts": new_anchor,
            }
        else:
            # No-op in non-total mode — the anchor only bounds the ``total``
            # window. Surface the code so the UI can explain why nothing changed.
            budget_reset = {
                "applied": False, "code": "not_total_mode",
                "budget_anchor_ts": project.get("budget_anchor_ts"),
            }
    # The limit owns its currency (set-once semantics):
    #   - An explicit budget_currency in the PUT body always wins (user owns it).
    #   - Otherwise, when this update SETS a budget limit and the project has no
    #     budget_currency yet, default it from the project's primary provider's
    #     currency. The provider may have just changed in this same PUT, so read
    #     it from the merged view (incoming update first, else stored).
    #   - Switching provider WITHOUT (re)setting a limit must NEVER touch it.
    if "budget_currency" not in updates:
        setting_limit = "budget_limit_usd" in updates
        already_has_currency = bool(project.get("budget_currency"))
        if setting_limit and not already_has_currency:
            eff_provider = updates.get("provider", project.get("provider", "custom"))
            eff_model = updates.get("model", project.get("model", ""))
            updates["budget_currency"] = _provider_currency(eff_provider, eff_model)

    # Handle workspace file content fields separately
    workspace = project.get("workspace", "")
    goals_content = updates.pop("project_goals_content", None)
    rules_content = updates.pop("user_directives_content", None)
    # Sync the legacy `instructions` field to project_goals.md, but guarded so
    # a later Settings save cannot clobber scan-/onboarding-authored goals. The
    # `instructions` key itself stays in `updates` for projects.json back-compat.
    _maybe_sync_instructions_to_goals(
        workspace, goals_content=goals_content, instructions=updates.get("instructions"),
    )
    if rules_content is not None:
        _write_workspace_file(workspace, "user_directives.md", rules_content)
    # If api_key matches the current global key, store empty string so the
    # project inherits at runtime rather than snapshotting a stale copy.
    if "api_key" in updates and _credential_store is not None:
        global_key = _credential_store.get_api_key()
        if global_key and updates["api_key"] == global_key:
            updates["api_key"] = ""
    if updates:
        try:
            _project_store.update_project(project_id, updates)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))
    # Push autonomy change to running agent (if any)
    if "autonomy" in updates and _agent_manager is not None:
        try:
            new_autonomy = Autonomy(updates["autonomy"])
            _agent_manager.update_autonomy(project_id, new_autonomy)
        except ValueError:
            pass  # invalid value already persisted — interceptor keeps old preset

    # P2-D nudge: a budget config change (limit raise, period roll, anchor move)
    # must reach a budget-paused dispatcher so its lazy auto-resume guard re-runs
    # PROMPTLY rather than only on the next _wait_idle timeout. Reuse the exact
    # nudge the add-item route uses (``notify_new_item`` sets the dispatcher's
    # idle event) — no new watcher, no timer. The dispatcher itself decides
    # whether to resume (reason=="budget" + under-limit + action!=stop); the
    # route only wakes it. Fire whenever any budget-affecting key changed OR a
    # reset was applied.
    _BUDGET_KEYS = {
        "budget_limit_usd", "budget_action", "budget_period",
        "budget_currency", "budget_anchor_ts",
    }
    if _agent_manager is not None and (
        _BUDGET_KEYS & set(updates) or (budget_reset and budget_reset["applied"])
    ):
        dispatcher = _agent_manager.get_dispatcher(project_id)
        if dispatcher is not None:
            try:
                dispatcher.notify_new_item()
            except Exception:
                logger.warning(
                    "budget config nudge to dispatcher failed for %s",
                    project_id, exc_info=True,
                )

    updated = _project_store.get_project(project_id)
    result = _redact_project(updated)
    _enrich_with_disk_content(result, workspace)
    result["budget_spent_usd"] = result.get("runtime", {}).get("budget_spent_usd", 0.0)
    # P2-D: surface the reset outcome (codes only) so the frontend can read the
    # ``not_total_mode`` no-op code. None when no reset was requested this PUT.
    if budget_reset is not None:
        result["budget_reset"] = budget_reset
    return result


@router.get("/projects/{project_id}/cost")
async def get_project_cost(project_id: str, window: Optional[str] = Query(default=None)):
    """Derived cost view for a project over a spend window (read-only).

    ``window`` defaults to the project's ``budget_period`` (itself defaulting to
    ``daily``). Cost is computed at QUERY time (tokens × current resolved rate),
    so editing the rates table changes historical cost — intended.

    The response carries codes / enums / ISO currency codes only (no display
    strings), per the binding i18n rule. A bad ``window`` returns 400 with a
    machine code in ``detail`` (not a sentence). Unknown project → 404.
    """
    from agent_os.budget.ledger import spend, WINDOWS

    project = _project_store.get_project(project_id)
    if not project:
        # Match existing 404 shape in this module.
        raise HTTPException(status_code=404, detail="Project not found")

    resolved_window = window or project.get("budget_period") or "daily"
    if resolved_window not in WINDOWS:
        # i18n rule: machine code, not a human sentence. Client renders the
        # message. ``allowed`` lets the client build a precise hint.
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_window", "allowed": list(WINDOWS)},
        )

    workspace = project.get("workspace", "")
    # Target currency defaults to the project's budget_currency, falling back to
    # USD if unset (spec). budget_currency is never auto-derived here — only the
    # limit-set path defaults it.
    target_currency = project.get("budget_currency") or "USD"
    anchor_ts = project.get("budget_anchor_ts")

    # FX rates are daemon-level config (settings.json). Static, user-editable.
    fx_rates: dict = {}
    if _settings_store is not None:
        try:
            fx_rates = dict(_settings_store.get().fx_rates or {})
        except Exception:  # noqa: BLE001 — a bad settings read must not 500 a read-only view
            logger.warning("failed to read fx_rates from settings; using none",
                           exc_info=True)

    try:
        result = spend(
            workspace,
            resolved_window,
            target_currency=target_currency,
            fx_rates=fx_rates,
            anchor_ts=anchor_ts,
        )
    except ValueError:
        # Defensive: spend() validates the window too; we already gated it.
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_window", "allowed": list(WINDOWS)},
        )
    return result


_cleanup_logger = logging.getLogger(__name__)


def _cleanup_project_files(workspace: str) -> None:
    """Remove all Orbital data for the project at this workspace.

    Deletes {workspace}/orbital/ and nothing else. User files in the
    workspace directory (including agent-authored deliverables placed
    contextually outside orbital/) are preserved.
    """
    paths = ProjectPaths(workspace)
    _rmtree_safe(paths.orbital_dir)


def _rmtree_safe(path: str) -> None:
    """Remove a directory tree, ignoring if it doesn't exist."""
    if os.path.isdir(path):
        try:
            shutil.rmtree(path)
        except OSError:
            _cleanup_logger.warning("Failed to remove directory: %s", path)


def _remove_safe(path: str) -> None:
    """Remove a single file, ignoring if it doesn't exist."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError:
        _cleanup_logger.warning("Failed to remove file: %s", path)


@router.delete("/projects/bulk")
async def bulk_delete_projects(body: BulkDeleteRequest):
    """Delete multiple projects by filter criteria."""
    if not body.prefix and not body.project_ids and not body.before:
        raise HTTPException(
            status_code=400,
            detail="At least one filter (prefix, project_ids, before) is required",
        )

    all_projects = _project_store.list_projects()
    to_delete = []

    for p in all_projects:
        pid = p["project_id"]
        # Never bulk-delete scratch project
        if p.get("is_scratch"):
            continue
        if body.project_ids is not None:
            if pid not in body.project_ids:
                continue
        if body.prefix is not None:
            if not p.get("name", "").startswith(body.prefix):
                continue
        if body.before is not None:
            created = p.get("created_at", "")
            if not created or created >= body.before:
                continue
        to_delete.append(p)

    deleted = 0
    failed = 0
    for p in to_delete:
        pid = p["project_id"]
        try:
            if _agent_manager.is_running(pid):
                # Seam 3 / D1 (Root C): is_running is holder-aware but
                # stop_agent is passthrough-None — forward the holder session so
                # the running loop is actually stopped, not orphaned (a bare
                # stop_agent(pid) misses the uuid-keyed handle → KeyError).
                await _agent_manager.stop_agent(
                    pid, session_id=_agent_manager.current_holder_session_id(pid),
                )
            workspace = p.get("workspace", "")
            if workspace:
                _cleanup_project_files(workspace)
            _sub_agent_sessions.pop(pid, None)
            _project_store.delete_project(pid)
            deleted += 1
        except Exception:
            failed += 1

    return {"deleted": deleted, "failed": failed}


@router.delete("/projects/{project_id}")
async def delete_project(project_id: str):
    project = _project_store.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    # Stop agent if running. is_running is holder-aware but stop_agent is
    # passthrough-None (seam 3 / D1, Root C): forward the holder session so the
    # running loop is stopped rather than orphaned (a bare stop_agent(project_id)
    # misses the uuid-keyed handle → KeyError → 500 with the loop left alive).
    if _agent_manager.is_running(project_id):
        await _agent_manager.stop_agent(
            project_id, session_id=_agent_manager.current_holder_session_id(project_id),
        )

    # Tear down the project's dispatcher. It is project-scoped and survives
    # agent stop, so deletion must shut it down explicitly.
    await _agent_manager.shutdown_dispatcher(project_id)

    # Clean up project files on disk
    workspace = project.get("workspace", "")
    if workspace:
        _cleanup_project_files(workspace)

    # Clear in-memory caches
    _sub_agent_sessions.pop(project_id, None)

    try:
        _project_store.delete_project(project_id)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    return {"status": "deleted"}


# ---- Agent Endpoints ----

@router.post("/agents/start")
async def start_agent(req: StartAgentRequest):
    from agent_os.daemon_v2.models import AgentConfig

    project = _project_store.get_project(req.project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    autonomy_str = project.get("autonomy", "hands_off")
    try:
        autonomy = Autonomy(autonomy_str)
    except ValueError:
        autonomy = Autonomy.HANDS_OFF

    # Use global settings as fallback for missing project-level LLM config
    global_settings = _settings_store.get() if _settings_store else None
    cred_key = _credential_store.get_api_key() if _credential_store else None
    api_key = (project.get("api_key")
               or cred_key
               or (global_settings.llm.api_key if global_settings else None)
               or "")
    base_url = project.get("base_url") or (global_settings.llm.base_url if global_settings else None)
    model = project.get("model") or (global_settings.llm.model if global_settings else None) or ""

    # Resolve fallback models: project-level > global-level > empty
    from agent_os.daemon_v2.models import FallbackModelEntry
    raw_fallbacks = project.get("llm_fallback_models")
    if not raw_fallbacks and global_settings:
        raw_fallbacks = [fb.model_dump() for fb in global_settings.llm.fallback_models]
    fallback_models = []
    for fb in (raw_fallbacks or []):
        fb_key = fb.get("api_key") or api_key  # inherit primary key if empty
        fallback_models.append(FallbackModelEntry(
            provider=fb.get("provider", "custom"),
            model=fb.get("model", ""),
            base_url=fb.get("base_url"),
            api_key=fb_key,
            sdk=fb.get("sdk", "openai"),
        ))

    config = AgentConfig(
        workspace=project["workspace"],
        model=model,
        api_key=api_key,
        base_url=base_url,
        autonomy=autonomy,
        sdk=project.get("sdk", "openai"),
        provider=project.get("provider", "custom"),
        project_name=project.get("name", ""),
        project_instructions=project.get("instructions", ""),
        agent_slug=project.get("agent_slug", "built-in"),
        enabled_sub_agents=project.get("enabled_sub_agents", []),
        disabled_sub_agents=project.get("disabled_sub_agents", []),
        agent_credentials=project.get("agent_credentials", {}),
        network_extra_domains=project.get("network_extra_domains", []),
        is_scratch=project.get("is_scratch", False),
        agent_name=project.get("agent_name", project.get("name", "")),
        global_preferences_path="",
        llm_fallback_models=fallback_models,
        budget_limit_usd=project.get("budget_limit_usd"),
        budget_action=project.get("budget_action", "ask"),
    )
    try:
        await _agent_manager.start_agent(
            req.project_id, config,
            initial_message=req.initial_message,
            session_id=req.session_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "started"}


@router.post("/agents/{project_id}/cold-start-scan", status_code=201)
async def cold_start_scan(project_id: str):
    """Mint the project's first session and start the agent in cold-start mode.

    Runs the deterministic skeleton walk synchronously (so a walker failure
    surfaces as an HTTP error), then starts a content-less loop whose prompt
    drives the 3-stage scan. No user message is fabricated.
    """
    from agent_os.agent.workspace_scan import scan_workspace
    project = _project_store.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="project not found")
    workspace = project.get("workspace", "")
    if not workspace or not os.path.isdir(workspace):
        raise HTTPException(status_code=400, detail="workspace missing")

    skeleton = scan_workspace(workspace)
    config = _agent_manager._build_agent_config_from_project(project_id)
    minted = await _agent_manager.new_session(project_id)
    session_id = minted["session_id"]
    await _agent_manager.start_agent(
        project_id, config,
        initial_message=None,
        session_id=session_id,
        cold_start=True,
        cold_start_skeleton=skeleton,
    )
    return {"status": "started", "session_id": session_id}


@router.post("/agents/{project_id}/inject")
async def inject_message(project_id: str, req: InjectRequest):
    # Verify project exists before attempting inject
    project = _project_store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    # Single-slot enforcement now lives at the MANAGER level: start_agent()
    # raises ValueError("Slot held by session …") when a different session
    # holds the project's active-loop slot, so every caller is covered (HTTP,
    # dispatcher, triggers, internal auto-starts) — not just this route. The
    # route's only job is to translate that rejection into the 202 `slot_held`
    # response the frontend's SlotHeldNotice expects (see below, around the
    # inject_message call). See REPORT-subagent-leak-and-slot-gap.md Q4.

    # Build the ``<attached_files>...</attached_files>`` prefix BEFORE the branch split so
    # both the management-agent branch and the sub-agent branch see the
    # prefixed content. Validation runs against the project workspace; a
    # failure here must not write anything to the session JSONL.
    if req.attachments:
        workspace = project.get("workspace", "")
        try:
            validate_attachments(workspace, req.attachments)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid attachment: {e}")
        effective_content = format_prefix(req.attachments) + req.content
        attachment_dicts: list[dict] | None = [
            a.model_dump() for a in req.attachments
        ]
    else:
        effective_content = req.content
        attachment_dicts = None

    if req.target and _sub_agent_manager is not None:
        # Route to sub-agent (Path B: direct @mention)
        workspace = project.get("workspace", "")
        session = _get_or_create_session(project_id, workspace)

        # Seam 3 / D1 (Root A): the @mention sub-agent attaches to a CONCRETE
        # chat session. send()/start() require a parent session id — their
        # resolver hard-raises on None (a sub-agent always has a parent) — so
        # this route must FORWARD the session it already has, not drop it.
        # Prefer the client-supplied session_id (the open chat session); fall
        # back to the session this message is persisted into so a session-less
        # @mention still routes to a concrete session instead of 404-ing.
        mention_session_id = req.session_id or getattr(session, "session_uuid", None)

        # Persist user message BEFORE sending to sub-agent
        user_ts = datetime.now(timezone.utc).isoformat()
        user_msg: dict = {
            "role": "user",
            "content": effective_content,
            "target": req.target,
            "timestamp": user_ts,
        }
        if req.nonce:
            user_msg["nonce"] = req.nonce
        if attachment_dicts is not None:
            user_msg["attachments"] = attachment_dicts
        session.append(user_msg)

        # send() spawns-on-demand (TASK-collapse-dispatch-to-send): the
        # manual try-send -> on-error-start -> re-send dance that used to
        # live here is now the manager's single built-in implementation.
        try:
            result = await _sub_agent_manager.send(project_id, req.target, effective_content, session_id=mention_session_id)
        except Exception:
            raise HTTPException(status_code=404, detail="No active session for project")
        if result.startswith("Error"):
            raise HTTPException(status_code=404, detail=f"Failed to dispatch to {req.target}: {result}")

        # Broadcast acknowledgement so ChatView knows the message was sent
        ack_ts = datetime.now(timezone.utc).isoformat()
        _ws_manager.broadcast(project_id, {
            "type": "chat.sub_agent_message",
            "project_id": project_id,
            "session_id": mention_session_id,
            "content": result,
            "source": req.target,
            "timestamp": ack_ts,
        })

        # Notify lifecycle observer (session injection handled there)
        if _lifecycle_observer:
            transcript = _sub_agent_manager.get_transcript(project_id, req.target)
            transcript_path = transcript.filepath if transcript else "unknown"
            await _lifecycle_observer.on_message_routed(
                project_id, req.target,
                initiator="user_mention",
                message_preview=effective_content[:100],
                transcript_path=transcript_path,
                session_id=mention_session_id,
            )

        return {"status": result}
    else:
        # Route to management agent (auto-starts if no session).
        # The prefix is part of effective_content; agent_manager.inject_message
        # itself does not learn about attachments — see PR description for the
        # deliberate v1 audit-field asymmetry.
        try:
            result = await _agent_manager.inject_message(
                project_id, effective_content, nonce=req.nonce,
                session_id=req.session_id,
            )
        except ValueError:
            # The manager rejected the inject — most commonly because another
            # session holds the project's active-loop slot (start_agent raised).
            # Translate that into the same 202 `slot_held` payload the frontend
            # expects, instead of a 500. Re-check the holder to confirm it is a
            # genuine slot conflict; re-raise anything else.
            holder = _agent_manager.current_holder_session_id(project_id)
            if holder is not None and holder != req.session_id:
                return JSONResponse(
                    status_code=202,
                    content={
                        "status": "slot_held",
                        "holding_session_id": holder,
                        "message": "Another session in this project is currently running.",
                    },
                )
            raise
        # inject_message returns either a str (legacy: "queued"/"delivered"/
        # "started") or a dict (new: auto-deny-on-paused-approval branch,
        # includes status + approval_dismissed + dismissed_tool_call_id).
        if isinstance(result, dict):
            return result
        return {"status": result}


@router.get("/agents/{project_id}/run-status")
async def agent_run_status(project_id: str):
    """Return the current runtime status for a project agent.

    Also returns ``current_holder_session_id``: the F1 session_id that
    currently holds the project's active-loop slot, or None.
    """
    status = _agent_manager.get_run_status(project_id)
    holder = _agent_manager.current_holder_session_id(project_id)
    return {"project_id": project_id, "status": status, "current_holder_session_id": holder}


@router.get("/agents/{project_id}/pending-approval")
async def get_pending_approval(
    project_id: str,
    session_id: str | None = Query(default=None, description="F1 session_id; omit for default session"),
):
    """Return the current pending approval payload, if any.

    Used by mobile clients to recover approval cards missed via WebSocket.

    ``session_id`` scopes the recovery to a specific session. Without it,
    both the management-agent and sub-agent lookups resolve to the
    default-session sentinel and silently miss approvals pending in
    non-default sessions.
    """
    approval = _agent_manager.get_pending_approval(project_id, session_id=session_id)
    if approval is None and _sub_agent_manager is not None:
        approval = _sub_agent_manager.get_pending_sub_agent_approval(
            project_id, session_id=session_id,
        )
    if approval is None:
        return {"pending": False}
    return {"pending": True, **approval}


@router.get("/projects/{project_id}/sessions")
async def list_project_sessions(project_id: str):
    """Enumerate active and idle chat sessions for a project.

    Phase 3c multi-loop discovery endpoint. Each entry exposes:
      - ``session_id``: the chat-session identifier (defaults to
        ``"default"`` under the single-loop back-compat path).
      - ``status``: one of ``running`` | ``pending_approval`` | ``waiting``
        | ``idle`` | ``stopped``.
      - ``session_uuid``: the per-loop ``Session.session_id`` used as the
        session JSONL filename stem (useful for transcript correlation).

    A project with no active sessions returns ``{"sessions": []}``.
    The response shape is intentionally a wrapped list — leaves room for
    future top-level fields (project metadata, totals) without a breaking
    response change.
    """
    if _project_store is not None:
        if _project_store.get_project(project_id) is None:
            raise HTTPException(status_code=404, detail="Project not found")
    sessions = _agent_manager.list_sessions(project_id)
    return {"project_id": project_id, "sessions": sessions}


class SessionRenameRequest(BaseModel):
    """Body for renaming a session (display label only). ``name`` is the new
    human-readable label; it has no effect on routing or hydration."""
    name: str


@router.patch("/agents/{project_id}/sessions/{session_id}")
async def rename_session(project_id: str, session_id: str, req: SessionRenameRequest):
    """Rename a session's display label.

    Updates ``session.name`` in memory (when a live handle exists) and rewrites
    the ``session_start`` meta line in the JSONL so the name persists. The name
    is display-only — F1/F2 identifiers are unchanged.

    Returns ``{"status": "renamed", "session_id", "name"}``. 404 if no session
    with this id (F1 or F2/uuid) exists for the project.
    """
    if _project_store is not None and _project_store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    name = (req.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name must not be empty")
    try:
        return await asyncio.to_thread(
            _agent_manager.rename_session, project_id, session_id, name
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")


@router.delete("/agents/{project_id}/sessions/{session_id}")
async def delete_session(project_id: str, session_id: str):
    """Delete a single session (removes its JSONL from disk).

    - 404 if no session with this id (F1 or F2/uuid) exists for the project.
    - 409 if the session is currently running — the caller must cancel first.
    - 200 ``{"status": "deleted", "session_id"}`` otherwise. An idle handle is
      torn down (idle-poll cancelled, sub-agents stopped) before the unlink.

    Workspace files the agent created are NOT deleted — they belong to the
    project, not the session. Deleting the only session is allowed; the next
    message creates a fresh one.
    """
    if _project_store is not None and _project_store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    try:
        return await _agent_manager.delete_session(project_id, session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/blocked")
async def list_blocked_globally():
    """Global blocked-session summary across ALL projects.

    Returns the count and list of sessions currently in ``pending_approval``
    state daemon-wide.  Useful for a global badge / notification badge
    that shows how many sessions are awaiting user approval.

    Response shape::

        {
            "blocked_count": int,
            "blocked_sessions": [{"project_id": str, "session_id": str}, ...]
        }
    """
    blocked = _agent_manager.list_blocked_sessions()
    return {"blocked_count": len(blocked), "blocked_sessions": blocked}


@router.post("/agents/{project_id}/cancel")
async def cancel_message(project_id: str, req: SessionScopedRequest | None = None) -> dict:
    """Cancel the current turn. Loop exits, agent stays alive.

    Wired to the UI Stop button. ``session_id`` (body) targets the session the
    UI has open. The cancel breaks the loop at the next iteration boundary (the
    CancelledError handlers in loop.py write a [cancelled by user] marker to the
    session JSONL, then exit the while body). Sub-agents, browser pages, and
    sandbox state are preserved.
    """
    return await _agent_manager.cancel_message(
        project_id, session_id=(req.session_id if req else None),
    )


# NOTE: the user-facing POST /agents/{project_id}/stop route was removed.
# Under the multi-session + idle-eviction model there is no user-facing
# "tear down this session" action: /cancel interrupts the turn (handle and
# session stay resumable), and idle eviction reclaims runtime resources
# automatically after AgentManager.EVICTION_IDLE_TIMEOUT. The full-teardown
# AgentManager.stop_agent() remains an INTERNAL method, called only by the
# eviction sweep, daemon shutdown, and project deletion — never by an HTTP
# request. See TASK-idle-eviction-and-remove-stop.md.


@router.post("/agents/{project_id}/new-session")
async def new_session(project_id: str, req: SessionScopedRequest | None = None):
    """Create a fresh session for the project (pure-create). Mints a new
    ``session_id`` + ``session_uuid`` and returns them; writes no file and
    touches no running session. The UI navigates to the new ``session_id`` and
    the session materializes on the first message. The body ``session_id`` is
    accepted for compatibility but ignored — a new session is always fresh."""
    result = await _agent_manager.new_session(
        project_id, session_id=(req.session_id if req else None),
    )
    return result


# ---- Queue routes (Phase 1) ----

class QueueStopRequest(BaseModel):
    # Snooze duration in seconds; omitted/None = pause until explicitly
    # resumed. gt=0 → 422 on zero/negative.
    duration_seconds: Optional[float] = Field(default=None, gt=0)


class QueueAddItemRequest(BaseModel):
    content: str
    file_refs: list[str] | None = None
    priority: int | None = 0
    review_before_advance: bool | None = False
    source: str | None = "user"
    idempotency_key: str | None = None


class QueueEditItemRequest(BaseModel):
    content: str | None = None
    file_refs: list[str] | None = None
    priority: int | None = None
    review_before_advance: bool | None = None


class QueueReorderRequest(BaseModel):
    item_ids: list[str]


class QueueRetryRequest(BaseModel):
    mode: str  # "edit" | "answer"
    input: str


def _resolve_queue_store(project_id: str):
    """Return a QueueStore for the project even if no agent is running.

    Falls through to project_store for the workspace lookup so that GET on
    the queue endpoint works before an agent is ever started.
    """
    if _agent_manager is None:
        raise HTTPException(status_code=503, detail="Agent manager not ready")
    project = _project_store.get_project(project_id) if _project_store else None
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")
    workspace = project.get("workspace") or ""
    if not workspace:
        raise HTTPException(status_code=400, detail="Project has no workspace")
    return _agent_manager.get_queue_store(project_id, workspace=workspace)


async def _ensure_dispatcher_for(project_id: str):
    """Return the project's dispatcher, creating it on demand.

    The dispatcher is project-scoped (created at daemon startup / project
    creation), but create it lazily here too so the queue endpoints work for a
    project created before this code path existed. The dispatcher always
    existing is what lets stop/resume/start be thin and never 409.
    """
    if _agent_manager is None:
        return None
    dispatcher = _agent_manager.get_dispatcher(project_id)
    if dispatcher is None:
        project = _project_store.get_project(project_id) if _project_store else None
        workspace = (project or {}).get("workspace", "")
        if workspace:
            await _agent_manager._ensure_dispatcher(project_id, workspace)
            dispatcher = _agent_manager.get_dispatcher(project_id)
    return dispatcher


@router.get("/projects/{project_id}/queue")
async def get_queue(project_id: str) -> dict:
    store = _resolve_queue_store(project_id)
    return store.snapshot()


@router.post("/projects/{project_id}/queue/items")
async def add_queue_item(project_id: str, req: QueueAddItemRequest) -> dict:
    """Persist a queue item. Does NOT auto-start the agent or transition
    queue state. Staging-only: the user explicitly starts the queue via
    POST /queue/start (or implicitly via /queue/resume if paused)."""
    if not req.content or not req.content.strip():
        raise HTTPException(status_code=400, detail="content must be non-empty")
    store = _resolve_queue_store(project_id)
    item = store.add_item(
        content=req.content,
        file_refs=req.file_refs or [],
        priority=int(req.priority or 0),
        review_before_advance=bool(req.review_before_advance),
        source=(req.source or "user"),
        idempotency_key=req.idempotency_key,
    )

    # If a dispatcher already exists (queue is RUNNING or PAUSED with the
    # agent up), notify it so it can pick up this new item on its next tick.
    # If no dispatcher (queue IDLE or no agent), the item sits queued until
    # the user clicks Start.
    dispatcher = _agent_manager.get_dispatcher(project_id) if _agent_manager else None
    if dispatcher is not None:
        dispatcher.notify_new_item()
    if _ws_manager is not None:
        _ws_manager.broadcast(project_id, {
            "type": "queue.item_added",
            "project_id": project_id,
            "item_id": item.id,
        })
    return {"item": item.model_dump(mode="json")}


@router.patch("/projects/{project_id}/queue/items/{item_id}")
async def edit_queue_item(project_id: str, item_id: str, req: QueueEditItemRequest) -> dict:
    store = _resolve_queue_store(project_id)
    item = store.edit_item(
        item_id,
        content=req.content,
        file_refs=req.file_refs,
        priority=req.priority,
        review_before_advance=req.review_before_advance,
    )
    if item is None:
        raise HTTPException(
            status_code=409,
            detail="Item not found or not in queued state (only queued items can be edited)",
        )
    if _ws_manager is not None:
        _ws_manager.broadcast(project_id, {
            "type": "queue.item_edited",
            "project_id": project_id,
            "item_id": item_id,
        })
    return {"item": item.model_dump(mode="json")}


@router.delete("/projects/{project_id}/queue/items/{item_id}")
async def delete_queue_item(project_id: str, item_id: str) -> dict:
    """Delete a queue item. IDLE-ONLY — a RUNNING item is rejected (409).

    Per ACTIVE-session-and-queue-model.md, delete is idle-only. The route is
    the liveness gate because the store has no agent_manager reference:

    - REJECT-RUNNING: the item is "running" iff its latest attempt's session is
      the live slot-holder (``current_holder_session_id``), NOT the stored
      ``item.state`` flag (which can be stale). A running item is rejected with
      409 and ZERO mutation — no remove, no CANCELLED stamp, the session is left
      untouched. Mirrors the session-delete reject convention.
    - IDLE DELETE: remove the store record FIRST, then clean up each DISTINCT
      bound session JSONL via ``delete_session`` (best-effort). An item with no
      attempts (never dispatched) skips session cleanup.

    Ordering matters for concurrency safety: the liveness gate and
    ``remove_item`` run with NO ``await`` between them, so they are atomic on the
    event loop. Removing the record before the first ``await`` (the session
    cleanup) means the dispatcher's ``next_queued()`` can never pick this item
    into the gap — closing the TOCTOU where a QUEUED-with-attempts
    (interrupted-requeued) item could be re-dispatched mid-delete and orphaned.

    Sub-agent JSONLs are intentionally NOT deleted (handle-keyed, shared on
    disk, no per-session mapping survives restart) — ``delete_session`` leaves
    them in place, which is the chosen behavior.
    """
    # Fail-closed: the route is the liveness gate, which needs the manager. If
    # it is unwired, refuse rather than silently degrade to "allow a running
    # delete" (mirrors the sibling queue routes' 503).
    if _agent_manager is None:
        raise HTTPException(status_code=503, detail="Agent manager not ready")

    store = _resolve_queue_store(project_id)

    # Resolve the item by id so we can inspect its attempts / liveness.
    item = next((it for it in store.load().items if it.id == item_id), None)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")

    # Liveness gate (NOT item.state): the item is running iff its latest-attempt
    # session currently holds the project's active-loop slot. No ``await`` between
    # this check and ``remove_item`` below — the pair is atomic on the loop.
    if item.attempts and item.attempts[-1].session_id == (
        _agent_manager.current_holder_session_id(project_id)
    ):
        # ZERO mutation: do not remove, do not stamp CANCELLED, do not touch the
        # session. The caller must stop/pause it first.
        raise HTTPException(
            status_code=409,
            detail="Cannot delete a running queue item. "
                   "Stop or pause it first, then delete.",
        )

    # Remove the store record FIRST (sync, no await since the gate) so the
    # dispatcher cannot dispatch a record that is being deleted.
    removed = store.remove_item(item_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Item not found")

    # Best-effort: clean up each distinct bound session JSONL. The record is
    # already gone, so any failure here only leaves an orphaned JSONL (acceptable
    # — sub-agent transcripts are likewise left) and never a half-removed record.
    seen: set[str] = set()
    for attempt in item.attempts:
        sid = attempt.session_id
        if sid in seen:
            continue
        seen.add(sid)
        try:
            await _agent_manager.delete_session(project_id, sid)
        except Exception:
            logger.warning(
                "delete_queue_item(%s): session JSONL cleanup for %s failed; "
                "store record already removed",
                project_id, sid, exc_info=True,
            )

    if _ws_manager is not None:
        _ws_manager.broadcast(project_id, {
            "type": "queue.item_removed",
            "project_id": project_id,
            "item_id": item_id,
        })
    return {"status": "removed"}


@router.post("/projects/{project_id}/queue/reorder")
async def reorder_queue(project_id: str, req: QueueReorderRequest) -> dict:
    store = _resolve_queue_store(project_id)
    store.reorder(req.item_ids)
    if _ws_manager is not None:
        _ws_manager.broadcast(project_id, {
            "type": "queue.reordered",
            "project_id": project_id,
        })
    return {"status": "reordered"}


@router.post("/projects/{project_id}/queue/stop")
async def stop_queue(project_id: str, req: Optional[QueueStopRequest] = None) -> dict:
    """Pause the queue and bring the active session to rest.

    Thin: the dispatcher is project-scoped and always exists, so this no longer
    409s on "no dispatcher" — pausing an agentless/empty queue just records the
    pause intent and returns.

    Body {"duration_seconds": N} = timed pause (auto-resumes after N seconds);
    no body = pause until resumed.
    """
    if _agent_manager is None:
        raise HTTPException(status_code=503, detail="Agent manager not ready")
    dispatcher = await _ensure_dispatcher_for(project_id)
    if dispatcher is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return await dispatcher.stop(duration_seconds=req.duration_seconds if req else None)


def _project_has_completed_onboarding(project: dict) -> bool:
    """Onboarding signal: PROJECT_STATE.md exists.

    PROJECT_STATE.md is written by run_session_end_routine (the LLM-driven
    summarizer) after the first session ends — which only happens after
    the user has had at least one back-and-forth in chat. Before that the
    project has no captured context and unattended queue dispatch is
    premature. After that, the project has memory files and the agent
    can resume into a real state.
    """
    workspace = project.get("workspace", "")
    if not workspace:
        return False
    return os.path.exists(ProjectPaths(workspace).project_state)


@router.post("/projects/{project_id}/queue/start")
async def start_queue(project_id: str) -> dict:
    """Start (or resume) draining the queue.

    Three branches based on current queue state:
    - RUNNING: no-op, returns {"status": "already_running"}.
    - PAUSED:  hot-resume the parked attempt session via dispatcher.resume().
    - IDLE:    ensure the agent is running (auto-start if needed), flip queue
               state to RUNNING, kick the dispatcher.

    Gated by onboarding completion: PROJECT_STATE.md must exist on disk
    (written by the session-end routine after the first chat session
    completes). Before onboarding the queue cannot start — the user must
    chat with the agent first so the project has captured context to
    operate against.
    """
    from agent_os.queue.models import QueueRunState

    if _agent_manager is None:
        raise HTTPException(status_code=503, detail="Agent manager not ready")
    if _project_store is None:
        raise HTTPException(status_code=503, detail="Project store not ready")

    project = _project_store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

    # Thin: the dispatcher is project-scoped and always exists. It auto-starts
    # the agent (onboarding-gated, inside the dispatcher) when it sees work, so
    # the endpoint no longer owns the onboarding gate or the auto-start.
    dispatcher = await _ensure_dispatcher_for(project_id)
    if dispatcher is None:
        raise HTTPException(status_code=500, detail="Dispatcher unavailable")

    store = _resolve_queue_store(project_id)
    current_state = store.load().state

    # No-op when the queue is already RUNNING and backed by a live agent —
    # calling resume() here would mis-read the in-flight item's RUNNING head as
    # a parked attempt and double-dispatch it.
    if current_state == QueueRunState.RUNNING and _agent_manager.has_handle(project_id):
        return {"status": "already_running"}

    # PAUSED → hot-resume any parked attempt. IDLE / RUNNING-without-handle →
    # un-pause and wake; the dispatcher drains and auto-starts the agent.
    result = await dispatcher.resume()
    if current_state != QueueRunState.PAUSED:
        dispatcher.notify_new_item()
    return result


@router.post("/projects/{project_id}/queue/resume")
async def resume_queue(project_id: str) -> dict:
    """Resume the queue. If an attempt was parked, hot-resume it.

    Kept as an alias for /queue/start. Thin: the project-scoped dispatcher
    always exists, so this no longer 409s.
    """
    if _agent_manager is None:
        raise HTTPException(status_code=503, detail="Agent manager not ready")
    dispatcher = await _ensure_dispatcher_for(project_id)
    if dispatcher is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return await dispatcher.resume()


@router.post("/projects/{project_id}/queue/items/{item_id}/retry")
async def retry_queue_item(
    project_id: str, item_id: str, req: QueueRetryRequest,
) -> dict:
    """Retry a BLOCKED queue item, hot-resuming the prior attempt's session.

    mode="answer": inject `input` raw — for question-card answers.
    mode="edit":  re-wrap `input` with `[QUEUE ITEM | id | attempt=N+1]`.
    """
    if _agent_manager is None:
        raise HTTPException(status_code=503, detail="Agent manager not ready")
    dispatcher = _agent_manager.get_dispatcher(project_id)
    if dispatcher is None:
        raise HTTPException(
            status_code=409,
            detail="No active dispatcher; start the agent first",
        )
    if not req.input or not req.input.strip():
        raise HTTPException(status_code=400, detail="input must be non-empty")
    if req.mode not in ("edit", "answer"):
        raise HTTPException(
            status_code=400, detail="mode must be 'edit' or 'answer'",
        )
    try:
        return await dispatcher.retry_blocked_item(
            item_id, req.input, mode=req.mode,
        )
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"item {item_id} not found",
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/agents/{project_id}/approve")
async def approve(project_id: str, req: ApproveRequest):
    try:
        await _agent_manager.approve(
            project_id, req.tool_call_id, reply_text=req.reply_text,
            approve_all=req.approve_all, session_id=req.session_id,
        )
    except KeyError:
        # Try sub-agent approval path
        if _sub_agent_manager is not None:
            routed = await _sub_agent_manager.resolve_sub_agent_approval(
                project_id, req.tool_call_id, approved=True,
                session_id=req.session_id,
            )
            if not routed:
                raise HTTPException(status_code=404, detail="No pending approval found")
        else:
            raise HTTPException(status_code=404, detail="No pending approval found")
    _ws_manager.broadcast(project_id, {
        "type": "approval.resolved",
        "project_id": project_id,
        "session_id": req.session_id,
        "tool_call_id": req.tool_call_id,
        "resolution": "approved",
    })
    return {"status": "approved"}


@router.post("/agents/{project_id}/deny")
async def deny(project_id: str, req: DenyRequest):
    try:
        await _agent_manager.deny(
            project_id, req.tool_call_id, req.reason, session_id=req.session_id,
        )
    except KeyError:
        # Try sub-agent approval path
        if _sub_agent_manager is not None:
            routed = await _sub_agent_manager.resolve_sub_agent_approval(
                project_id, req.tool_call_id, approved=False,
                session_id=req.session_id,
                decision="cancel" if req.stop_turn else None,
            )
            if not routed:
                raise HTTPException(status_code=404, detail="No pending approval found")
        else:
            raise HTTPException(status_code=404, detail="No pending approval found")
    _ws_manager.broadcast(project_id, {
        "type": "approval.resolved",
        "project_id": project_id,
        "session_id": req.session_id,
        "tool_call_id": req.tool_call_id,
        "resolution": "denied",
    })
    return {"status": "denied"}


@router.post("/agents/{project_id}/claudemd-warning/dismiss")
async def dismiss_claudemd_warning(project_id: str, body: dict):
    """Dismiss a workspace_claudemd_warning banner for the current session.

    Body: {"content_hash": "<hash from the WS event>"}.
    Suppression is keyed by (project_id, content_hash) and lives in
    daemon memory; restart or content change re-emits.
    """
    if _sub_agent_manager is None:
        raise HTTPException(status_code=503, detail="Sub-agent manager not available")
    content_hash = (body or {}).get("content_hash")
    if not content_hash or not isinstance(content_hash, str):
        raise HTTPException(status_code=400, detail="content_hash required")
    _sub_agent_manager.dismiss_claudemd_warning(project_id, content_hash)
    return {"status": "dismissed"}


def _read_chat_messages(sessions_dir: str, limit: int, offset: int) -> tuple[list[dict], int]:
    """Read chat messages from session JSONL files. Runs in a thread.

    Returns (messages, total_count). When limit > 0, reads only what's needed
    from the end (true tail pagination). When limit=0, reads everything.
    """
    if not os.path.isdir(sessions_dir):
        return [], 0

    # List and sort session files by mtime (oldest first)
    session_files = []
    for fname in os.listdir(sessions_dir):
        if fname.endswith(".jsonl"):
            fpath = os.path.join(sessions_dir, fname)
            session_files.append((os.path.getmtime(fpath), fpath))
    session_files.sort(key=lambda x: x[0])

    # Read lines from all files (fast scan)
    file_lines = []  # [(line_count, lines)]
    total = 0
    for _mtime, fpath in session_files:
        with open(fpath, "r", encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        file_lines.append((len(lines), lines))
        total += len(lines)

    # If no pagination, parse everything
    if limit <= 0:
        all_messages = []
        for count, lines in file_lines:
            for line in lines:
                try:
                    all_messages.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return all_messages, total

    # True tail pagination: only parse the lines we need
    end = total - offset
    start = max(0, end - limit)
    if end <= 0:
        return [], total

    result = []
    cursor = 0
    for count, lines in file_lines:
        file_start = cursor
        file_end = cursor + count
        cursor = file_end

        # Skip files entirely before our window
        if file_end <= start:
            continue
        # Stop if we're past the window
        if file_start >= end:
            break

        # Calculate which lines in this file we need
        local_start = max(0, start - file_start)
        local_end = min(count, end - file_start)

        for line in lines[local_start:local_end]:
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    return result, total


def _read_chat_messages_single(jsonl_path: str, limit: int, offset: int) -> tuple[list[dict], int]:
    """Read chat messages from a single JSONL file with pagination. Runs in a thread.

    Returns (messages, total_count). Mirrors ``_read_chat_messages`` pagination
    semantics but operates on one file instead of a directory.
    """
    if not os.path.isfile(jsonl_path):
        return [], 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]

    total = len(lines)

    if limit <= 0:
        messages = []
        for line in lines:
            try:
                messages.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return messages, total

    end = total - offset
    start = max(0, end - limit)
    if end <= 0:
        return [], total

    result = []
    for line in lines[start:end]:
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return result, total


# Matches the "[Sub-agent] {handle} started ... Transcript: {path}" system
# line the lifecycle observer writes into the management session JSONL. Group
# 1 is the sub-agent handle; group 2 is the transcript path.
_SUB_AGENT_STARTED_RE = re.compile(
    r"^\[Sub-agent\]\s+(\S+)\s+started\b.*?Transcript:\s*(.+?)\s*$"
)


def _interleave_sub_agent_summaries(messages: list[dict]) -> list[dict]:
    """Inline a compact sub-agent run summary after each dispatch marker.

    For every ``[Sub-agent] {handle} started … Transcript: {path}`` system
    message in *messages*, read the referenced sub-agent transcript and insert
    a synthetic ``source="sub_agent"`` message immediately after it. The
    frontend transforms that into a distinct ``sub_agent_run`` block (tools
    used, duration, response) so the management chat carries the durable record
    of the sub-agent's work — not just the one-line completion summary.

    Read-only and best-effort: a missing/empty transcript leaves the markers
    untouched. Operates on whatever page was already paginated, so it never
    changes the total-count math.
    """
    out: list[dict] = []
    for msg in messages:
        out.append(msg)
        if msg.get("role") != "system":
            continue
        content = msg.get("content") or ""
        m = _SUB_AGENT_STARTED_RE.match(content)
        if not m:
            continue
        handle, transcript_path = m.group(1), m.group(2)
        try:
            summary = read_sub_agent_summary(transcript_path)
        except Exception:
            summary = None
        if not summary:
            continue
        out.append({
            "role": "assistant",
            "content": summary.get("response") or "",
            "source": "sub_agent",
            "sub_agent_handle": handle,
            "sub_agent_tool_rows": summary.get("tool_rows", []),
            "sub_agent_duration": summary.get("total_duration_seconds", 0.0),
            "timestamp": msg.get("timestamp", ""),
            "session_id": msg.get("session_id"),
        })
    return out


def _find_session_uuid_on_disk(sessions_dir: str, session_id: str) -> str | None:
    """Resolve an F1 ``session_id`` to its F2 JSONL stem by scanning disk.

    Fallback for the chat ``session_id`` filter when no live handle exists
    (e.g. a stopped/popped session). Scans each ``*.jsonl`` for a record whose
    ``session_id`` field matches, returning the filename stem (F2). Runs in a
    thread — does blocking disk I/O, must not be called on the event loop.

    Accepts either identifier: if ``session_id`` is itself an F2 stem (the
    sidebar addresses disk-only sessions by uuid), the matching
    ``{session_id}.jsonl`` is returned directly. Otherwise it is treated as an
    F1 chat id and the records are scanned. Returns the F2 stem, or None.
    """
    if not os.path.isdir(sessions_dir):
        return None
    # Direct F2 match: the identifier names a file (uuid addressing).
    if os.path.isfile(os.path.join(sessions_dir, f"{session_id}.jsonl")):
        return session_id
    for fname in os.listdir(sessions_dir):
        if not fname.endswith(".jsonl"):
            continue
        fpath = os.path.join(sessions_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        rec = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("session_id") == session_id:
                        return fname[:-6]  # strip .jsonl
        except OSError:
            pass
    return None


@router.get("/agents/{project_id}/chat")
async def chat_history(
    project_id: str,
    limit: int = Query(default=0, ge=0, description="Max messages to return (0 = all)"),
    offset: int = Query(default=0, ge=0, description="Skip first N messages from the end"),
    session_id: str | None = Query(default=None, description="Filter to a specific F1 session_id"),
):
    """Return chat history, newest messages last.

    Pagination returns the *most recent* messages:
    - limit=20, offset=0 → last 20 messages
    - limit=20, offset=20 → messages 21-40 from the end
    - limit=0 (default) → all messages (backward-compatible)

    When ``session_id`` is provided (F1 user-facing id), only messages from
    that session are returned.  The F1→F2 mapping is resolved via
    ``list_sessions()`` which already carries both ids.  Sub-agent transcript
    entries are included only in the unfiltered (no ``session_id``) path.

    Response includes X-Total-Count header for pagination UI.
    """
    project = _project_store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    workspace = project["workspace"]
    from agent_os.agent.project_paths import ProjectPaths
    sessions_dir = ProjectPaths(workspace).sessions_dir

    # ── session_id filter path ────────────────────────────────────────────
    if session_id is not None:
        # Map F1 session_id → F2 session_uuid so we know which JSONL to read.
        # list_sessions returns both fields for active/idle sessions; for
        # stopped/popped sessions the handle is gone but the JSONL still
        # exists on disk — fall back to scanning the JSONL content for a
        # matching session_id field (offloaded to a thread; it does blocking
        # disk I/O and must not run on the event loop).
        session_uuid: str | None = None
        for entry in _agent_manager.list_sessions(project_id):
            if entry["session_id"] == session_id:
                session_uuid = entry["session_uuid"]
                break

        if session_uuid is None:
            session_uuid = await asyncio.to_thread(
                _find_session_uuid_on_disk, sessions_dir, session_id
            )

        if session_uuid is None:
            # Unknown session — return empty rather than 404 (may not be active yet).
            resp = JSONResponse(content=[])
            resp.headers["X-Total-Count"] = "0"
            return resp

        # Read only that session's JSONL.
        session_jsonl = os.path.join(sessions_dir, f"{session_uuid}.jsonl")
        messages, total = await asyncio.to_thread(
            _read_chat_messages_single, session_jsonl, limit, offset
        )
        # Inline each dispatched sub-agent's run summary (tools/duration/
        # response) read from its own transcript, so the management chat shows
        # what the sub-agent actually did. Offloaded — does blocking disk I/O.
        messages = await asyncio.to_thread(_interleave_sub_agent_summaries, messages)
        resp = JSONResponse(content=messages)
        resp.headers["X-Total-Count"] = str(total)
        return resp

    # ── unfiltered path (existing behaviour) ─────────────────────────────

    # Read sub-agent transcript entries (disk scan + in-memory)
    sub_entries = []
    if _sub_agent_manager is not None:
        sub_entries = await asyncio.to_thread(
            _sub_agent_manager.get_all_transcript_entries, project_id
        )
        # Normalize transcript entries to chat message format
        for entry in sub_entries:
            entry.setdefault("role", "agent")

    if not sub_entries:
        # Fast path: no transcripts, use original pagination
        messages, total = await asyncio.to_thread(
            _read_chat_messages, sessions_dir, limit, offset
        )
    else:
        # Merge path: read all management messages, merge with transcripts, sort
        management_messages, mgmt_total = await asyncio.to_thread(
            _read_chat_messages, sessions_dir, 0, 0  # read all
        )
        all_messages = management_messages + sub_entries
        all_messages.sort(key=lambda m: m.get("timestamp", ""))
        total = len(all_messages)

        # Apply pagination to merged result
        if limit > 0:
            end = total - offset
            start = max(0, end - limit)
            messages = all_messages[start:end] if end > 0 else []
        else:
            messages = all_messages

    resp = JSONResponse(content=messages)
    resp.headers["X-Total-Count"] = str(total)
    return resp


# ---- Agent Registry / Setup Endpoints ----

_available_cache: dict = {"result": None, "expires_at": 0.0}
_AVAILABLE_CACHE_TTL = 60  # seconds


@router.get("/agents/available")
async def available_agents():
    """Return setup status for all registered agents."""
    if _setup_engine is None:
        return []

    now = time.time()
    if _available_cache["result"] is not None and now < _available_cache["expires_at"]:
        return _available_cache["result"]

    statuses = await asyncio.to_thread(_setup_engine.check_all)
    result = []
    for s in statuses:
        entry = {
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
            "setup_actions": [
                {"action": a.action, "label": a.label, "command": a.command, "blocking": a.blocking}
                for a in s.setup_actions
            ],
        }
        result.append(entry)

    _available_cache["result"] = result
    _available_cache["expires_at"] = time.time() + _AVAILABLE_CACHE_TTL
    return result


@router.get("/agents/{slug}/status")
async def agent_status(slug: str):
    """Return setup status for a single agent by slug.

    # internal: status-via-ws — Frontend uses GET /agents/available for bulk
    # checks and WS agent.status events for runtime status. This endpoint is
    # for CLI tooling and single-agent setup verification.
    """
    if _setup_engine is None:
        raise HTTPException(status_code=503, detail="Setup engine not available")
    try:
        s = _setup_engine.check_agent(slug)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Unknown agent: {slug}")
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
        "setup_actions": [
            {"action": a.action, "label": a.label, "command": a.command, "blocking": a.blocking}
            for a in s.setup_actions
        ],
    }


# ---- Provider Endpoints ----

@router.get("/providers")
async def list_providers():
    """Return the provider registry."""
    if _provider_registry is not None:
        return _provider_registry.all_providers()
    return {}


class FetchModelsRequest(BaseModel):
    provider: str
    api_key: str | None = None
    base_url: str | None = None


@router.post("/providers/models")
async def fetch_models(req: FetchModelsRequest):
    """Proxy request to provider's /v1/models endpoint."""
    import httpx
    # Use injected registry
    provider_info = _provider_registry.get_provider_data(req.provider) if _provider_registry else None
    base_url = req.base_url or (provider_info["base_url"] if provider_info else None)
    if not base_url:
        raise HTTPException(status_code=400, detail="No base_url for provider")

    # Handle Anthropic (different models endpoint)
    sdk = provider_info.get("sdk", "openai") if provider_info else "openai"
    if sdk == "anthropic":
        models_url = base_url.rstrip("/") + "/v1/models"
        headers = {"x-api-key": req.api_key or "", "anthropic-version": "2023-06-01"}
    else:
        models_url = base_url.rstrip("/") + "/models"
        headers = {"Authorization": f"Bearer {req.api_key or ''}"}

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.get(models_url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            # Normalize: extract model IDs
            models = []
            for m in data.get("data", []):
                model_id = m.get("id", "")
                if model_id:
                    models.append(model_id)
            return {"models": sorted(models)}
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Provider returned {e.response.status_code}")
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))


class TestConnectionRequest(BaseModel):
    provider: str
    model: str
    api_key: str
    base_url: str | None = None
    sdk: str = "openai"


@router.post("/providers/test")
async def test_connection(req: TestConnectionRequest):
    """Test connection by sending a minimal completion request."""
    # Use injected registry
    provider_info = _provider_registry.get_provider_data(req.provider) if _provider_registry else None
    base_url = req.base_url or (provider_info["base_url"] if provider_info else None)
    sdk = req.sdk or (provider_info.get("sdk", "openai") if provider_info else "openai")

    from agent_os.agent.providers.openai_compat import LLMProvider
    from agent_os.agent.providers.types import LLMError, ContextOverflowError

    try:
        provider = LLMProvider(req.model, req.api_key, base_url, sdk=sdk)
        result = await provider.complete(
            messages=[{"role": "user", "content": "hi"}],
        )
        return {"status": "ok", "message": f"Connected to {req.provider} using {req.model}"}
    except ContextOverflowError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except LLMError as e:
        status = e.status_code or 500
        if status == 401 or status == 403:
            detail = "Invalid API key"
        elif status == 404:
            detail = f"Model '{req.model}' not found on this provider"
        elif status == 429:
            detail = "Rate limited — key works but slow down"
        else:
            detail = e.message
        raise HTTPException(status_code=status, detail=detail)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- Trigger Endpoints ----

@router.get("/projects/{project_id}/triggers")
async def list_triggers(project_id: str):
    """Return all triggers for the project."""
    project = _project_store.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project.get("triggers", [])


@router.patch("/projects/{project_id}/triggers/{trigger_id}")
async def toggle_trigger(project_id: str, trigger_id: str, body: TriggerToggleRequest):
    """Toggle a trigger on/off. This is the only REST mutation for triggers."""
    project = _project_store.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    triggers = project.get("triggers", [])
    trigger = next((t for t in triggers if t.get("id") == trigger_id), None)
    if trigger is None:
        raise HTTPException(status_code=404, detail="Trigger not found")

    trigger["enabled"] = body.enabled
    _project_store.update_project(project_id, {"triggers": triggers})

    # Notify TriggerManager
    if _trigger_manager is not None:
        if body.enabled:
            _trigger_manager.register_trigger(project_id, trigger)
        else:
            _trigger_manager.unregister_trigger(trigger_id)

    return trigger


@router.delete("/projects/{project_id}/triggers/{trigger_id}", status_code=204)
async def delete_trigger(project_id: str, trigger_id: str):
    """Delete a trigger permanently."""
    from starlette.responses import Response

    project = _project_store.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    triggers = project.get("triggers", [])
    trigger = next((t for t in triggers if t.get("id") == trigger_id), None)
    if trigger is None:
        raise HTTPException(status_code=404, detail="Trigger not found")

    triggers = [t for t in triggers if t.get("id") != trigger_id]
    _project_store.update_project(project_id, {"triggers": triggers})

    if _trigger_manager is not None:
        _trigger_manager.unregister_trigger(trigger_id)

    return Response(status_code=204)


# ---- Skills CRUD Endpoints ----

@router.get("/projects/{project_id}/skills")
async def list_skills(project_id: str):
    """Return all skills found in the project workspace."""
    project = _project_store.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    workspace = project.get("workspace", "")
    loader = SkillLoader(workspace)
    return loader.scan()


@router.delete("/projects/{project_id}/skills/{skill_name}")
async def delete_skill(project_id: str, skill_name: str):
    """Delete a skill directory from the project workspace."""
    project = _project_store.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    workspace = project.get("workspace", "")
    from agent_os.agent.project_paths import ProjectPaths
    skills_base = os.path.realpath(ProjectPaths(workspace).skills_dir)
    skill_path = os.path.realpath(os.path.join(skills_base, skill_name))
    if not skill_path.startswith(skills_base + os.sep):
        raise HTTPException(status_code=400, detail="Invalid skill name")
    if not os.path.isdir(skill_path):
        raise HTTPException(status_code=404, detail=f"Skill not found: {skill_name}")
    if _agent_manager and _agent_manager.is_running(project_id):
        raise HTTPException(status_code=400, detail="Cannot delete skill while agent is running")
    shutil.rmtree(skill_path)
    return {"deleted": skill_name}


@router.post("/projects/{project_id}/skills", status_code=201)
async def upload_skill(project_id: str, file: UploadFile = File(...)):
    """Upload a skill as a .zip or .md file."""
    project = _project_store.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    workspace = project.get("workspace", "")

    filename = file.filename or ""
    content_bytes = await file.read()

    if filename.endswith(".zip"):
        return _handle_zip_upload(workspace, content_bytes)
    elif filename.endswith(".md"):
        return _handle_md_upload(workspace, content_bytes)
    else:
        raise HTTPException(status_code=400, detail="File must be .zip or .md")


def _validate_skill_content(text: str) -> tuple[str, str]:
    """Extract and validate skill name and description from SKILL.md content.

    Returns (name, description) or raises HTTPException 400.
    """
    name = None
    description = None
    for line in text.splitlines()[:20]:
        stripped = line.rstrip()
        if stripped.startswith("# ") and name is None:
            name = stripped[2:].strip()
        elif name is not None and stripped.strip():
            description = stripped.strip()
            break

    if not name or not description:
        raise HTTPException(
            status_code=400,
            detail="SKILL.md must have a # heading and a description line"
        )
    return name, description


def _sanitize_skill_dir_name(name: str) -> str:
    """Convert a skill name to a safe directory name."""
    sanitized = re.sub(r'[^a-z0-9_-]', '-', name.lower()).strip('-')
    if not sanitized:
        raise HTTPException(status_code=400, detail="Skill name produces invalid directory name")
    return sanitized


def _handle_md_upload(workspace: str, content_bytes: bytes) -> dict:
    """Handle uploading a single .md file as a skill."""
    from agent_os.agent.project_paths import ProjectPaths
    text = content_bytes.decode("utf-8", errors="replace")
    name, description = _validate_skill_content(text)

    dir_name = _sanitize_skill_dir_name(name)
    skill_dir = os.path.join(ProjectPaths(workspace).skills_dir, dir_name)

    if os.path.exists(skill_dir):
        raise HTTPException(status_code=409, detail=f"Skill already exists: {dir_name}")

    os.makedirs(skill_dir, exist_ok=True)
    skill_md_path = os.path.join(skill_dir, "SKILL.md")
    with open(skill_md_path, "w", encoding="utf-8") as f:
        f.write(text)

    return {"name": name, "description": description, "path": skill_md_path, "dir_name": dir_name}


def _handle_zip_upload(workspace: str, content_bytes: bytes) -> dict:
    """Handle uploading a .zip file containing a skill."""
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "upload.zip")
        with open(zip_path, "wb") as f:
            f.write(content_bytes)

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Guard against zip slip (path traversal)
                real_tmpdir = os.path.realpath(tmpdir)
                for member in zf.namelist():
                    member_path = os.path.realpath(os.path.join(tmpdir, member))
                    if not member_path.startswith(real_tmpdir + os.sep) and member_path != real_tmpdir:
                        raise HTTPException(status_code=400, detail="Zip contains path traversal entry")
                zf.extractall(tmpdir)
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid zip file")

        # Find SKILL.md in extracted content
        skill_md_path = None
        for root, _dirs, files in os.walk(tmpdir):
            if "SKILL.md" in files:
                skill_md_path = os.path.join(root, "SKILL.md")
                break

        if skill_md_path is None:
            raise HTTPException(status_code=400, detail="No SKILL.md found in zip")

        with open(skill_md_path, "r", encoding="utf-8") as f:
            text = f.read()

        name, description = _validate_skill_content(text)

        # Determine the skill directory name from the zip structure
        # Use the parent directory of SKILL.md if it's not the tmpdir itself
        skill_src_dir = os.path.dirname(skill_md_path)
        if os.path.normpath(skill_src_dir) == os.path.normpath(tmpdir):
            # SKILL.md was at the root of the zip; use the skill name
            dir_name = _sanitize_skill_dir_name(name)
        else:
            dir_name = os.path.basename(skill_src_dir)

        from agent_os.agent.project_paths import ProjectPaths
        dest_dir = os.path.join(ProjectPaths(workspace).skills_dir, dir_name)
        if os.path.exists(dest_dir):
            raise HTTPException(status_code=409, detail=f"Skill already exists: {dir_name}")

        # Copy the skill directory (or just SKILL.md if at zip root)
        if os.path.normpath(skill_src_dir) == os.path.normpath(tmpdir):
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(skill_md_path, os.path.join(dest_dir, "SKILL.md"))
        else:
            shutil.copytree(skill_src_dir, dest_dir)

        return {
            "name": name,
            "description": description,
            "path": os.path.join(dest_dir, "SKILL.md"),
            "dir_name": dir_name,
        }


# ---- Sub-Agent MEMORY.md Endpoints ----

# Hard cap on MEMORY.md size when the user edits via the UI. This prevents
# accidental UI-driven runaway growth. Soft warning at 2KB target.
_MEMORY_MAX_BYTES = 10 * 1024
_MEMORY_TARGET_BYTES = 2 * 1024


class SubAgentMemoryUpdate(BaseModel):
    content: str


def _resolve_available_sub_agents(project: dict) -> list[dict]:
    """Compute the available sub-agent list for a project.

    Returns a list of {slug, name} dicts for sub-agents that are
    installed AND not in disabled_sub_agents. Skips 'built-in' (the
    management agent itself, never a peer dispatch target).
    """
    disabled = set(project.get("disabled_sub_agents", []) or [])
    if _setup_engine is None:
        return []
    statuses = _setup_engine.check_all()
    out: list[dict] = []
    for s in statuses:
        if s.slug == "built-in":
            continue
        if not s.installed:
            continue
        if s.slug in disabled:
            continue
        out.append({"slug": s.slug, "name": s.name})
    return out


def _memory_md_path_for(workspace: str, agent_slug: str) -> str:
    """Compute the on-disk path to a sub-agent's MEMORY.md."""
    return os.path.join(
        ProjectPaths(workspace).sub_agent_dir(agent_slug),
        "MEMORY.md",
    )


@router.get("/agents/{project_id}/sub-agents/status")
async def sub_agents_status(project_id: str, session_id: str | None = None):
    """Live sub-agent statuses for the badge (Piece 3 Part D).

    Status vocabulary: 'running' (turn open) | 'background-running' (turn
    done, tracked background work alive — SDK only) | 'idle'. Each entry
    carries the live background commands so the stop dialog can warn
    honestly. ``session_id`` defaults to the project's active-loop holder.
    """
    if _sub_agent_manager is None:
        raise HTTPException(status_code=503, detail="Sub-agent manager not available")
    sid = session_id or _agent_manager.current_holder_session_id(project_id)
    if sid is None:
        return {"session_id": None, "agents": []}
    agents = _sub_agent_manager.list_active(project_id, session_id=sid)
    from agent_os.daemon_v2.background_work import BackgroundWorkRegistry
    registry = getattr(
        _sub_agent_manager._process_manager, "background_work", None)
    for a in agents:
        if isinstance(registry, BackgroundWorkRegistry):
            a["background_commands"] = registry.live_commands(
                project_id, sid, a["handle"])
        else:
            a["background_commands"] = []
    return {"session_id": sid, "agents": agents}


@router.post("/agents/{project_id}/sub-agents/{handle}/stop")
async def stop_sub_agent(project_id: str, handle: str,
                         session_id: str | None = None):
    """User stop button (Piece 3 Part D): cancel turn + kill agent + tracked
    children (confirmed), with an honest report of what was terminated and
    the raw-detach limitation. ``session_id`` defaults to the holder."""
    if _sub_agent_manager is None:
        raise HTTPException(status_code=503, detail="Sub-agent manager not available")
    if not _AGENT_SLUG_RE.match(handle):
        raise HTTPException(status_code=400, detail="Invalid agent handle")
    sid = session_id or _agent_manager.current_holder_session_id(project_id)
    if sid is None:
        raise HTTPException(
            status_code=409,
            detail="No active session for this project — nothing to stop",
        )
    result = await _sub_agent_manager.stop_for_user(
        project_id, handle, session_id=sid)
    result["session_id"] = sid
    return result


def _installed_sub_agents() -> list[dict]:
    """Compute the list of ALL installed (non-built-in) sub-agents.

    Mirrors the installed/built-in filter of ``_resolve_available_sub_agents``
    but WITHOUT subtracting the project's ``disabled_sub_agents`` denylist.
    The merged Project-Settings card shows a disabled-but-installed agent's
    card too (dimmed) and its memory body must stay reachable, so the memory
    listing is keyed on "installed" rather than "available". Real dispatch
    gating still goes through ``_resolve_available_sub_agents`` (unchanged).
    """
    if _setup_engine is None:
        return []
    statuses = _setup_engine.check_all()
    out: list[dict] = []
    for s in statuses:
        if s.slug == "built-in":
            continue
        if not s.installed:
            continue
        out.append({"slug": s.slug, "name": s.name})
    return out


@router.get("/projects/{project_id}/sub-agent-memory")
async def list_sub_agent_memory(project_id: str):
    """Return MEMORY.md status for each INSTALLED sub-agent in this project.

    Each entry: {agent_slug, agent_name, exists, content, last_modified, size_bytes}.
    Lists ALL installed (non-built-in) sub-agents regardless of the project's
    disabled denylist — the merged Project-Settings card renders a card per
    installed agent (a disabled one is dimmed but still expandable to edit its
    memory). `exists: false` for sub-agents never dispatched (MEMORY.md not
    yet lazily created on disk).
    """
    project = _project_store.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    workspace = project.get("workspace", "")
    if not workspace:
        return []
    available = _installed_sub_agents()
    out: list[dict] = []
    for entry in available:
        slug = entry["slug"]
        name = entry["name"]
        path = _memory_md_path_for(workspace, slug)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                stat = os.stat(path)
                out.append({
                    "agent_slug": slug,
                    "agent_name": name,
                    "exists": True,
                    "content": content,
                    "last_modified": stat.st_mtime,
                    "size_bytes": stat.st_size,
                })
            except OSError:
                out.append({
                    "agent_slug": slug,
                    "agent_name": name,
                    "exists": False,
                    "content": "",
                    "last_modified": None,
                    "size_bytes": 0,
                })
        else:
            out.append({
                "agent_slug": slug,
                "agent_name": name,
                "exists": False,
                "content": "",
                "last_modified": None,
                "size_bytes": 0,
            })
    return out


_AGENT_SLUG_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


@router.put("/projects/{project_id}/sub-agent-memory/{agent_slug}")
async def update_sub_agent_memory(
    project_id: str, agent_slug: str, body: SubAgentMemoryUpdate,
):
    """Write content to the sub-agent's MEMORY.md file.

    Creates the file (and the parent directory) if it doesn't exist.
    Validates content size: hard cap at 10KB, warning at 2KB.
    """
    if not _AGENT_SLUG_RE.match(agent_slug):
        raise HTTPException(status_code=400, detail="Invalid agent slug")

    project = _project_store.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    workspace = project.get("workspace", "")
    if not workspace:
        raise HTTPException(status_code=400, detail="Project has no workspace")

    # Validate that this slug is among the available sub-agents for this
    # project. Disabled / unknown / not-installed sub-agents are rejected.
    available_slugs = {e["slug"] for e in _resolve_available_sub_agents(project)}
    if agent_slug not in available_slugs:
        raise HTTPException(
            status_code=400,
            detail=f"Sub-agent {agent_slug!r} is not available for this project",
        )

    content_bytes = body.content.encode("utf-8")
    if len(content_bytes) > _MEMORY_MAX_BYTES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Content exceeds {_MEMORY_MAX_BYTES} byte limit "
                f"({len(content_bytes)} bytes)."
            ),
        )

    path = _memory_md_path_for(workspace, agent_slug)
    parent = os.path.dirname(path)
    try:
        # Per spec: PUT-creates-file path writes user content directly,
        # without the daemon's canonical header. The UI is overwriting
        # MEMORY.md from scratch.
        #
        # F4 / dispatch 2026-05-20 §5: route through SubAgentManager's
        # locked write helper when available so a UI overwrite cannot
        # interleave with a concurrent daemon-side ``ensure_memory_md``.
        # When ``_sub_agent_manager`` is None (e.g. in tests that don't
        # wire it) we fall back to the unguarded write — those code paths
        # were unprotected before this change too, so the fallback is not
        # a regression.
        if _sub_agent_manager is not None:
            await _sub_agent_manager.write_memory_md(
                workspace, project_id, agent_slug, body.content,
            )
        else:
            os.makedirs(parent, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(body.content)
    except OSError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write MEMORY.md: {e}",
        )

    stat = os.stat(path)
    response: dict = {
        "agent_slug": agent_slug,
        "exists": True,
        "size_bytes": stat.st_size,
        "last_modified": stat.st_mtime,
    }
    if len(content_bytes) > _MEMORY_TARGET_BYTES:
        response["warning"] = (
            f"Content size {len(content_bytes)} bytes exceeds the "
            f"{_MEMORY_TARGET_BYTES} byte target. Sub-agents read MEMORY.md "
            "every dispatch — keep it concise."
        )
    return response


# ---- Network utilities ----

def _get_lan_ip() -> str | None:
    """Get LAN IP, preferring local network (192.168/172.16) over VPN (10.x)."""
    import socket as _sock
    candidates: list[str] = []
    try:
        for info in _sock.getaddrinfo(_sock.gethostname(), None, _sock.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127."):
                candidates.append(ip)
    except Exception:
        pass
    if not candidates:
        # Fallback: UDP connect trick
        try:
            s = _sock.socket(_sock.AF_INET, _sock.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            if ip and not ip.startswith("127."):
                return ip
        except Exception:
            pass
        return None
    # Prefer 192.168.x.x / 172.16-31.x.x over 10.x.x.x (often VPN)
    for ip in candidates:
        if ip.startswith("192.168.") or ip.startswith("172."):
            return ip
    return candidates[0]


@router.get("/network/lan-url")
async def get_lan_url():
    """Return the machine's LAN IP address for direct access."""
    ip = _get_lan_ip()
    if not ip:
        return {"ip": None, "error": "No LAN network detected"}
    return {"ip": ip}
