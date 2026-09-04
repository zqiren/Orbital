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
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from agent_os.agent.prompt_builder import Autonomy
from agent_os.agent.skills import SkillLoader
from agent_os.daemon_v2.agent_md_seeder import (
    reseed_project_agent_md,
    seed_project_agent_md,
)
from agent_os.daemon_v2.default_skills_installer import install_default_skills
from agent_os.daemon_v2.sub_agent_transcript import read_sub_agent_summary
from agent_os.daemon_v2.provider_errors import ProviderConfigError
from agent_os.utils.file_lock import FileLockError
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
    # Optional (default ""): the create-project modal only ever asks for
    # workspace + name up front. An empty string is already treated as
    # "inherit the global provider/model" at runtime (BYOK dedup below,
    # agent_manager.py runtime fallback) — this just stops every client from
    # having to send placeholder empties for fields with a sensible default.
    model: str = ""
    api_key: str = ""
    base_url: str | None = None
    autonomy: str | None = None
    instructions: str | None = None
    provider: str | None = None
    sdk: str | None = None
    agent_slug: str | None = None
    enabled_sub_agents: list[str] | None = None
    disabled_sub_agents: list[str] | None = None
    sub_agent_deployment_instructions: str | None = Field(
        default=None, max_length=4000,
    )
    agent_credentials: dict | None = None
    agent_name: str | None = None
    is_scratch: bool = False
    notification_prefs: dict | None = None
    llm_fallback_models: list[dict] | None = None
    budget_limit_usd: float | None = None
    budget_action: str | None = None
    budget_period: str | None = None
    budget_currency: str | None = None

    @field_validator("sub_agent_deployment_instructions", mode="before")
    @classmethod
    def normalize_sub_agent_deployment_instructions(cls, value):
        return value.strip() if isinstance(value, str) else value


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
    sub_agent_deployment_instructions: str | None = Field(
        default=None, max_length=4000,
    )
    # Spec 011 — per-project connector enablement (gates tool reflection).
    enabled_connectors: list[str] | None = None
    agent_credentials: dict | None = None
    agent_name: str | None = None
    project_goals_content: str | None = None
    user_directives_content: str | None = None
    notification_prefs: dict | None = None
    llm_fallback_models: list[dict] | None = None
    budget_limit_usd: float | None = None
    budget_action: str | None = None
    # P3-F coupled removal: the legacy ``budget_spent_usd`` /
    # ``runtime_budget_spent_usd`` PUT request aliases are GONE. The ledger owns
    # spend; the only reset surface is ``reset_budget_anchor: true`` (total
    # mode). The release rule guarantees no shipped frontend depends on these.
    # Budget Piece 1, Task 4 — additive derived-cost window config.
    budget_period: str | None = None  # "daily"|"weekly"|"monthly"|"total"
    budget_currency: str | None = None  # ISO 4217; explicit set overrides the
    #                                     provider-derived default
    # "reset spend window" = set budget_anchor_ts to now. A direct ISO value is
    # also honored (mirrors the legacy runtime reset's direct-value pattern).
    reset_budget_anchor: bool | None = None
    budget_anchor_ts: str | None = None
    # TOFU network grants (Plan 2 Task 2): bare registrable domains the user
    # has approved for this project; wildcarded at NetworkRules-build time.
    approved_domains: list[str] | None = None
    # TOFU pending requests (Plan 2 Task 7): lets Settings → Network access
    # persist dismissals (and any other pending-list edit) through this same
    # PUT rather than a dedicated route.
    pending_domain_requests: list[dict] | None = None
    # Workbench privacy toggle (Task 5): exclude this project from the global
    # Workbench view (its verbatim user quotes stay off the aggregated, relay-
    # served surface). Persists automatically via the generic ``updates`` merge.
    workbench_exclude_global: bool | None = None

    @field_validator("sub_agent_deployment_instructions", mode="before")
    @classmethod
    def normalize_sub_agent_deployment_instructions(cls, value):
        return value.strip() if isinstance(value, str) else value


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
    # Spec 074: True when the client resolved ``target`` from the composer's
    # sticky "Talking to" dropdown (the session pin) rather than a leading
    # @mention. Maps to ``initiator="user_pinned"`` — the dispatch whose
    # terminal events are wake-suppressed (the management LLM takes zero
    # turns). A plain @mention keeps ``initiator="user_mention"`` and today's
    # manager-supervises semantics. Ignored without ``target``.
    pinned: bool = False
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


class TriggerScheduleBody(BaseModel):
    """The `schedule` sub-object of a schedule trigger.

    `human` is the display caption the UI shows instead of the raw cron. The
    web form derives it from the chosen preset (localized), so it is optional
    here — the agent tool supplies its own free-text version.
    """
    cron: str
    human: str | None = None
    timezone: str = "UTC"


class TriggerCreateRequest(BaseModel):
    name: str
    type: Literal["schedule", "file_watch"]
    task: str
    enabled: bool = True
    schedule: TriggerScheduleBody | None = None
    watch_path: str | None = None
    patterns: list[str] | None = None
    recursive: bool = False
    debounce_seconds: int = 5


class TriggerUpdateRequest(BaseModel):
    """Partial update. Every field is optional; only the ones actually sent
    are applied (`exclude_unset`), so the long-standing `{"enabled": false}`
    body keeps working untouched.
    """
    name: str | None = None
    task: str | None = None
    enabled: bool | None = None
    schedule: TriggerScheduleBody | None = None
    watch_path: str | None = None
    patterns: list[str] | None = None
    recursive: bool | None = None
    debounce_seconds: int | None = None


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
_pinned_consolidation = None


def configure(project_store, agent_manager, ws_manager, sub_agent_manager=None,
              setup_engine=None, settings_store=None, credential_store=None,
              trigger_manager=None, provider_registry=None, lifecycle_observer=None):
    """Called by app factory to inject dependencies."""
    global _project_store, _agent_manager, _ws_manager, _sub_agent_manager, _setup_engine, _settings_store, _credential_store, _trigger_manager, _provider_registry, _lifecycle_observer, _pinned_consolidation
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
    # Spec 074: pinned-chat consolidation coordinator (unpin/retarget +
    # quiescence triggers). Built here — configure() is the one place that
    # holds both the agent manager and the lifecycle observer — and wired to
    # the observer's pinned-terminal hook so every suppressed terminal event
    # starts/resets the quiet-period timer.
    if agent_manager is not None:
        from agent_os.daemon_v2.pinned_consolidation import (
            PinnedConsolidationCoordinator,
        )
        _pinned_consolidation = PinnedConsolidationCoordinator(
            agent_manager, project_store,
        )
        if lifecycle_observer is not None:
            lifecycle_observer.pinned_terminal_hook = (
                _pinned_consolidation.note_pinned_terminal
            )
    else:
        _pinned_consolidation = None


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
    # list_projects() bypasses ProjectStore.get_project(), so normalize this
    # legacy-safe scalar here as well to keep list/detail response parity.
    result.setdefault("sub_agent_deployment_instructions", "")
    # Legacy dead config (TASK-network-config-cleanup): tolerated on old
    # records, never exposed externally; dropped from disk on next save.
    result.pop("network_extra_domains", None)
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


# ---- Manual project order (spec 056) ----

# Sort position for a project that has never been dragged. +inf parks every
# such project after all placed ones, where the stable sort preserves creation
# order — which is exactly the ordering that existed before this feature.
_UNPLACED_SORT_KEY = float("inf")


class ProjectReorderRequest(BaseModel):
    """The complete ordered id list for the reordered block."""

    ordered_ids: list[str]


def _project_sort_key(project: dict) -> float:
    """Manual position, or ``_UNPLACED_SORT_KEY`` for a never-placed project.

    Deliberately NOT a ``DEFAULT_PROJECT_FIELDS`` entry: defaulting the key to
    0 on read would hoist every legacy project to the top. "No key" has to stay
    distinguishable from "position 0".
    """
    value = project.get("sort_key")
    # bool is an int subclass — a stray ``sort_key: true`` is not a position.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return _UNPLACED_SORT_KEY
    value = float(value)
    if value != value:  # NaN never orders consistently; treat it as unplaced.
        return _UNPLACED_SORT_KEY
    return value


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
    if req.sub_agent_deployment_instructions is not None:
        project_data["sub_agent_deployment_instructions"] = (
            req.sub_agent_deployment_instructions.strip()
        )
    if req.agent_credentials is not None:
        project_data["agent_credentials"] = req.agent_credentials
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

    # Seed the workspace-root AGENTS.md signpost. Same failure posture as the
    # skills seed — the file is a convenience for external agentic tools, and
    # a missing one is never worth failing project creation over. Seeded once
    # here only; there is deliberately no reconcile-on-start counterpart.
    try:
        seed_project_agent_md(_project_store, pid)
    except Exception:
        logger.error(
            "AGENTS.md seeding failed during create_project for %s",
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
    # Spec 056 — manual order. Two components, in this order:
    #   1. ``not is_scratch`` — Quick Tasks stays pinned above everything. It
    #      is the FIRST component, so no reorder can float a project above the
    #      scratch project even if a client puts it there; the invariant is a
    #      property of the sort, not of endpoint validation.
    #   2. ``sort_key`` — the manual position written by POST /projects/reorder.
    #      A project that has never been placed carries no key and sorts to the
    #      bottom; Python's sort is stable, so keyless projects (legacy records
    #      and freshly created ones alike) keep today's creation order there.
    #      A fresh install with no keys at all therefore renders exactly as it
    #      did before this feature existed.
    projects.sort(key=lambda p: (not p.get("is_scratch", False), _project_sort_key(p)))
    for p in projects:
        p.setdefault("agent_name", p.get("name", ""))
        p.setdefault("is_scratch", False)
    return projects


@router.post("/projects/reorder")
async def reorder_projects(req: ProjectReorderRequest):
    """Persist a manual project order (spec 056 §6 decision 3).

    Takes the COMPLETE ordered id list for the block being reordered and
    writes it in one atomic store pass. Returns the canonical list in its new
    order so a ``GET /projects`` refetch racing this call converges on the
    same answer the caller already applied optimistically.

    Unknown ids are rejected rather than silently dropped — a client sending a
    stale id is sending a stale order, and half-applying it would leave the
    user looking at a list nobody asked for.
    """
    known = {p.get("project_id") for p in _project_store.list_projects()}
    unknown = [pid for pid in req.ordered_ids if pid not in known]
    if unknown:
        raise HTTPException(
            status_code=404, detail=f"Unknown project ids: {', '.join(unknown)}",
        )
    _project_store.reorder_projects(req.ordered_ids)
    return await list_projects()


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
    # P3-F coupled removal: the legacy ``budget_spent_usd`` flatten is GONE. The
    # token ledger is the single source of recorded spend; the frontend reads it
    # via GET /cost. Any persisted ``runtime.budget_spent_usd`` in old project
    # files is tolerated-on-read but never surfaced.
    return result


@router.put("/projects/{project_id}")
async def update_project(project_id: str, body: ProjectUpdate):
    project = _project_store.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    raw_body = body.model_dump()
    updates = {k: v for k, v in raw_body.items() if v is not None}
    if "sub_agent_deployment_instructions" in updates:
        updates["sub_agent_deployment_instructions"] = updates[
            "sub_agent_deployment_instructions"
        ].strip()
    # TOFU allowlist: normalize every incoming approved_domains entry so a
    # Settings save (or a legacy raw-form pending entry it promotes) can never
    # persist a URL/port-suffixed string the proxy's exact-host/``*.`` matcher
    # would never match. Drop ungrantable entries, dedupe preserving order.
    if "approved_domains" in updates:
        from agent_os.daemon_v2.network_rules_builder import normalize_domain
        normalized: list[str] = []
        for entry in updates["approved_domains"]:
            d = normalize_domain(str(entry))
            if d is not None and d not in normalized:
                normalized.append(d)
        updates["approved_domains"] = normalized
    # P3-F coupled removal: the legacy ``budget_spent_usd`` /
    # ``runtime_budget_spent_usd`` reset sentinels are GONE. The token ledger is
    # the single source of recorded spend; the ONLY reset surface is
    # ``reset_budget_anchor: true``, which sets ``budget_anchor_ts = now`` in
    # ``budget_period == "total"`` mode (the only window an anchor affects). In
    # any non-total mode the reset is a NO-OP that surfaces the machine code
    # ``not_total_mode`` (codes only, no sentences — i18n rule).

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
    reset_requested = bool(reset_anchor_flag)
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

    # Push a TOFU grant-list change to a running agent's proxy immediately
    # (Task 8's Settings save and Task 4's approve both funnel through this
    # PUT route) rather than waiting for the next agent start.
    if "approved_domains" in updates and _agent_manager is not None:
        _agent_manager._apply_network_rules(project_id)

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
    # P3-F coupled removal: no ``budget_spent_usd`` flatten in the PUT response.
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

    Response shape (P3-B reshape): the MANAGEMENT block —
    ``{window, by_currency, converted_total, breakdown}`` — carries management
    spend ONLY, exactly the numbers it returned before sub-agent capture
    existed. Sub-agent rows move OUT of that block into a separate top-level
    ``subagents`` list. Each subagent entry is
    ``{provider, model, source, tokens, reported_cost}`` where ``reported_cost``
    is the provider-reported cost VERBATIM (with currency) when present, else
    ``null`` — NEVER recomputed from our rate table. This keeps the display
    honest: subscription-auth sub-agents show tokens with a null cost rather
    than a fabricated dollar figure.

    The response carries codes / enums / ISO currency codes only (no display
    strings), per the binding i18n rule. A bad ``window`` returns 400 with a
    machine code in ``detail`` (not a sentence). Unknown project → 404.
    """
    from agent_os.budget.ledger import (
        SOURCE_MANAGEMENT,
        SOURCE_SUBAGENT_CLAUDE_CODE,
        SOURCE_SUBAGENT_CODEX,
        WINDOWS,
        spend,
    )

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
        # Management block: filtered to management spend so by_currency /
        # converted_total / breakdown are byte-identical to the pre-P3-B
        # numbers (the P3-A pinned regression guarantees subagent rows never
        # affect this view).
        result = spend(
            workspace,
            resolved_window,
            target_currency=target_currency,
            fx_rates=fx_rates,
            anchor_ts=anchor_ts,
            sources=[SOURCE_MANAGEMENT],
        )
        # Sub-agent block: same window, but the subagent sources only. We reuse
        # spend()'s grouping (per provider/model/source, with tokens + the
        # verbatim per-currency reported_cost) and reshape into the flat
        # display entries the client renders. No rate-derived dollar figure is
        # exposed here — only the provider-reported cost, verbatim or null.
        sub = spend(
            workspace,
            resolved_window,
            target_currency=target_currency,
            fx_rates=fx_rates,
            anchor_ts=anchor_ts,
            sources=[SOURCE_SUBAGENT_CLAUDE_CODE, SOURCE_SUBAGENT_CODEX],
        )
    except ValueError:
        # Defensive: spend() validates the window too; we already gated it.
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_window", "allowed": list(WINDOWS)},
        )

    result["subagents"] = [
        _subagent_entry(row) for row in sub["breakdown"]
    ]
    # P3-F footnote: surface the daemon's static FX table (the same one the
    # conversion above used) so the client can render the rate it converted at.
    # Codes/numbers only — pair keys like "CNY_per_USD" with numeric values;
    # non-numeric junk in settings is filtered, never surfaced.
    result["fx_rates"] = {
        k: float(v) for k, v in fx_rates.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    return result


def _subagent_entry(row: dict) -> dict:
    """Reshape one spend() breakdown row into a display-only subagent entry.

    Carries provider / model / source / tokens, plus the provider-reported cost
    VERBATIM when the row recorded one (claude-code's total_cost_usd), else
    ``reported_cost: None``. The rate-derived ``cost`` block from spend() is
    deliberately dropped — sub-agent cost is shown as reported by the provider
    or not at all, never recomputed from our rates.
    """
    reported = row.get("reported_cost")
    if reported:
        # spend() aggregates reported_cost per currency. In practice claude-code
        # reports a single currency (USD); pick the sole entry. If a row ever
        # spans currencies, surface the first deterministically (sorted keys).
        ccy = sorted(reported.keys())[0]
        reported_cost = {"amount": reported[ccy], "currency": ccy}
    else:
        reported_cost = None
    return {
        "provider": row["provider"],
        "model": row["model"],
        "source": row["source"],
        "tokens": row["tokens"],
        "reported_cost": reported_cost,
    }


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
            # Drop the pending-input queue before teardown (spec 006 §3g).
            if hasattr(_agent_manager, "purge_pending"):
                _agent_manager.purge_pending(pid)
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
    # Drop the pending-input queue BEFORE stopping the holder: stop_agent frees
    # the slot and would otherwise dispatch a queued message into a project that
    # is being deleted (spec 006 §3g).
    if hasattr(_agent_manager, "purge_pending"):
        _agent_manager.purge_pending(project_id)
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

    try:
        _project_store.delete_project(project_id)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    return {"status": "deleted"}


# ---- Agent Endpoints ----

@router.post("/agents/start")
async def start_agent(req: StartAgentRequest):
    project = _project_store.get_project(req.project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    # Config comes from the canonical builder, same as every other start path
    # (inject auto-start, queue, triggers, cold-start scan). This route used to
    # carry a partial copy of the provider/endpoint rules — it had the
    # crosses-provider guard but not the provider-tracks-model rule — so a
    # project inheriting the global model while pinning its own stale provider
    # reached the wrong endpoint. See tests/unit/test_agent_config_parity.py.
    config = _agent_manager._build_agent_config_from_project(req.project_id)

    try:
        await _agent_manager.start_agent(
            req.project_id, config,
            initial_message=req.initial_message,
            session_id=req.session_id,
        )
    except ProviderConfigError as e:
        raise HTTPException(status_code=400,
                            detail={"code": e.code, "message": str(e)})
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
    # Validate the provider BEFORE minting: a credential failure must surface
    # as a structured 400 the frontend can translate, and must not leave an
    # orphan session per click (the silent-error bug minted one per attempt).
    try:
        _agent_manager.validate_provider_config(config)
    except ProviderConfigError as e:
        raise HTTPException(status_code=400,
                            detail={"code": e.code, "message": str(e)})
    minted = await _agent_manager.new_session(project_id)
    session_id = minted["session_id"]
    try:
        await _agent_manager.start_agent(
            project_id, config,
            initial_message=None,
            session_id=session_id,
            cold_start=True,
            cold_start_skeleton=skeleton,
        )
    except ProviderConfigError as e:
        # Belt-and-braces: settings changed between validate and start.
        raise HTTPException(status_code=400,
                            detail={"code": e.code, "message": str(e)})
    return {"status": "started", "session_id": session_id}


# ---- Spec 074 §3.4: transcript recap preamble for pinned dispatches ----

# Total budget for the recap block (mirrors _build_session_summary's cap) and
# the per-message excerpt cap inside it.
_RECAP_CAP_CHARS = 10_000
_RECAP_LINE_CAP = 500

# Extracts the summary text out of a completed-terminal row's display copy
# ("[Sub-agent] {handle} completed. Summary: {text}. Transcript: {path}.").
# Greedy body + the literal tail anchors on the LAST ". Transcript: ", so
# summaries containing periods survive. The handle is deliberately dropped —
# recap replies are worker-ANONYMOUS ("assistant").
_RECAP_COMPLETED_RE = re.compile(
    r"\[Sub-agent\] \S+ completed\. Summary: ([\s\S]*)\. Transcript: "
)


def _recap_scope(messages: list, handle: str) -> list:
    """Messages after ``handle``'s last participation in this session.

    Participation = a user row targeted at the handle, a dispatch-marker row
    whose ``_meta.handle`` is the handle, or one of the handle's own terminal
    rows (their content starts with ``[Sub-agent] {handle} `` — terminal meta
    carries no handle field). Never participated → the whole session. An
    empty result means the worker's own thread already holds the conversation
    (own-thread resume) — no recap.
    """
    own_prefix = f"[Sub-agent] {handle} "
    last = -1
    for i, m in enumerate(messages):
        meta = m.get("_meta") or {}
        if m.get("role") == "user" and m.get("target") == handle:
            last = i
        elif meta.get("handle") == handle:
            last = i
        elif meta.get("event") == "sub_agent_terminal":
            text = meta.get("display_content") or m.get("content") or ""
            if isinstance(text, str) and text.startswith(own_prefix):
                last = i
    return messages[last + 1:]


def _build_recap_preamble(session, handle: str) -> str:
    """Build the capped, worker-anonymous "Conversation so far" block.

    Content: user messages plus prior reply summaries — management replies
    verbatim and sub-agent completion summaries extracted from their terminal
    rows — every reply labeled "assistant", never naming the producing agent.
    Framed as a visible context block (not fabricated turns), newest-favored
    under the ~10k-char cap, prepended to the pinned worker's first message
    by the inject route. Returns "" when there is nothing the worker missed.
    """
    try:
        messages = session.get_messages()
    except Exception:
        return ""
    lines: list[str] = []
    total = 0
    for m in reversed(_recap_scope(messages, handle)):
        role = m.get("role")
        content = m.get("content")
        line = None
        if role in ("user", "assistant"):
            if isinstance(content, str) and content.strip():
                label = "user" if role == "user" else "assistant"
                line = f"{label}: {content.strip()[:_RECAP_LINE_CAP]}"
        elif role == "system":
            meta = m.get("_meta") or {}
            if (meta.get("event") == "sub_agent_terminal"
                    and meta.get("kind") == "completed"):
                display = meta.get("display_content") or (
                    content if isinstance(content, str) else "")
                match = _RECAP_COMPLETED_RE.match(display)
                if match:
                    summary = match.group(1).strip()
                    if summary and summary != "(no output)":
                        line = f"assistant: {summary[:_RECAP_LINE_CAP]}"
        if line is None:
            continue
        if total + len(line) > _RECAP_CAP_CHARS:
            break
        lines.append(line)
        total += len(line)
    if not lines:
        return ""
    lines.reverse()
    return (
        "Conversation so far (earlier messages in this chat, provided as "
        "context; prior replies are labeled \"assistant\"):\n\n"
        + "\n\n".join(lines)
        + "\n\n--- end of conversation so far ---\n\n"
    )


# The session JSONL's advisory lock is held in short bursts by concurrent
# same-process writers — the pin PATCH's load + meta-rewrite is the known
# collider (the composer fires it fire-and-forget, so a send routinely lands
# inside the burst). The lock itself is non-blocking with zero retries, so
# the inject route absorbs contention here: brief exponential backoff, then
# a clean retryable failure instead of an unhandled 500.
_LOCK_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8)


async def _retry_session_lock(fn):
    """Run ``fn``, retrying while the session file's lock is contended.

    ``fn`` may return a plain value (sync call sites run each attempt on the
    event loop, unchanged from before) or an awaitable (``asyncio.to_thread``
    call sites); the backoff sleeps yield so the lock holder can finish. The
    final attempt's ``FileLockError`` propagates to the caller.
    """
    import inspect

    async def _attempt():
        result = fn()
        return await result if inspect.isawaitable(result) else result

    for delay in _LOCK_RETRY_DELAYS:
        try:
            return await _attempt()
        except FileLockError:
            await asyncio.sleep(delay)
    return await _attempt()


@router.post("/agents/{project_id}/inject")
async def inject_message(project_id: str, req: InjectRequest):
    # Verify project exists before attempting inject
    project = _project_store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    # Spec 074 §3.4: recap preamble for a pinned worker starting a thread
    # that lacks this session's prior conversation. Resolved BEFORE the user
    # message is persisted so the recap covers strictly prior rows, and the
    # resolved concrete session id is reused for persistence (a None
    # session_id must not be minted twice). The recap rides ONLY the
    # dispatched message body — the persisted chat row stays the user's
    # authored text. No LLM call anywhere in this path.
    recap_preamble = ""
    inject_session_id = req.session_id
    if req.target and req.pinned and _sub_agent_manager is not None:
        try:
            inject_session_id, prior_session = await _retry_session_lock(
                lambda: _agent_manager.peek_chat_session(
                    project_id, req.session_id,
                ))
            if prior_session is not None:
                recap_preamble = _build_recap_preamble(prior_session, req.target)
        except Exception:
            logger.warning(
                "recap preamble build failed for %s/%s — dispatching without",
                project_id, req.target, exc_info=True,
            )
            recap_preamble = ""
            inject_session_id = req.session_id

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
        # Route to sub-agent (Path B: direct @mention).
        #
        # Seam 3 / D1: resolve the @mention's chat session EXACTLY ONCE through
        # the canonical inject funnel (passthrough / disk-hydrate / canonical
        # mint) and thread the single concrete id to persistence, dispatch, and
        # lifecycle. This persists the authored mention to the project's REAL
        # chat session — never a fabricated subagent_<hex> log — and does NOT
        # auto-wake the management loop: the record sits in the shared session
        # JSONL for the management agent to read on demand (it must not be
        # re-dispatched off the mention).
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
        # Persist the authored user message BEFORE dispatch, and adopt the
        # resolved concrete session id for dispatch + ack + lifecycle. A pure
        # resolve-then-append: it never starts/queues the management loop.
        # ``inject_session_id`` is the recap peek's resolved id when the
        # dispatch is pinned (identical resolution funnel), else the raw
        # request value.
        try:
            mention_session_id = await _retry_session_lock(
                lambda: _agent_manager.persist_mention_message(
                    project_id, inject_session_id, user_msg,
                ))
        except FileLockError:
            # Still contended after the whole backoff budget: refuse cleanly
            # (retryable, message NOT persisted) rather than letting the
            # error escape as an opaque 500 with the user's message dropped.
            raise HTTPException(
                status_code=503,
                detail="Chat session is busy — please retry",
            )

        # dispatch_id (TASK-dispatch-id-pairing): minted HERE, up front, so
        # it is available for logging/correlation alongside this request;
        # send() would otherwise mint its own when passed None. The single
        # "Message sent to …" marker this dispatch gets (fired inside
        # send() — see below) carries this same id in its meta.
        from uuid import uuid4
        dispatch_id = f"{mention_session_id}:{uuid4().hex[:8]}"

        # send() spawns-on-demand (TASK-collapse-dispatch-to-send): the
        # manual try-send -> on-error-start -> re-send dance that used to
        # live here is now the manager's single built-in implementation.
        # initiator="user_mention" (backlog #23 D3): threaded through to
        # send()'s one internal on_message_routed notification (fired here
        # immediately, or later when a queued prompt drains) so the
        # management agent is told the user addressed this sub-agent
        # directly. This route used to ALSO fire its own direct
        # on_message_routed call for the same dispatch_id — a double
        # marker for one physical dispatch (backlog #24 D3) — now deleted;
        # send()'s internal notification is the only one this dispatch
        # ever gets.
        # initiator (spec 074): "user_pinned" when the target came from the
        # composer's sticky dropdown — the wake-suppressed, zero-manager-turn
        # dispatch class. A leading @mention (req.pinned False) keeps
        # "user_mention" and today's manager-supervises semantics.
        initiator = "user_pinned" if req.pinned else "user_mention"
        if req.pinned and _pinned_consolidation is not None:
            # A new pinned dispatch: the exchange is active again — cancel
            # any pending quiescence consolidation timer.
            _pinned_consolidation.note_pinned_dispatch(
                project_id, mention_session_id)
        dispatch_content = (
            recap_preamble + effective_content if recap_preamble
            else effective_content
        )
        try:
            result = await _sub_agent_manager.send(
                project_id, req.target, dispatch_content,
                session_id=mention_session_id, dispatch_id=dispatch_id,
                initiator=initiator,
            )
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
        except ProviderConfigError as e:
            # Auto-start could not construct the LLM provider (missing/invalid
            # credentials). Surface a structured 400 the frontend can translate
            # — never a swallowed 500.
            raise HTTPException(status_code=400,
                                detail={"code": e.code, "message": str(e)})
        except ValueError:
            # The manager rejected the inject — most commonly because another
            # session holds the project's active-loop slot (start_agent raised).
            # Re-check the holder to confirm it is a genuine slot conflict;
            # re-raise anything else.
            holder = _agent_manager.current_holder_session_id(project_id)
            if holder is not None and holder != req.session_id:
                # Path A (spec 006): enqueue the message for delivery when the
                # slot frees, instead of rejecting. The single-slot invariant is
                # preserved — B runs *next*, not concurrently. ``slot_held`` is
                # retained ONLY as the defensive fallback (no session_id to
                # target, or enqueue itself raised).
                enq = None
                if req.session_id is not None:
                    try:
                        enq = _agent_manager.enqueue_pending_inject(
                            project_id, req.session_id, effective_content,
                            nonce=req.nonce, attachments=attachment_dicts,
                        )
                    except Exception:
                        logger.warning(
                            "enqueue_pending_inject failed for %s/%s",
                            project_id, req.session_id, exc_info=True,
                        )
                        enq = None
                if enq is not None:
                    # Broadcast the optimistic bubble (text + nonce) so OTHER
                    # clients render it; the origin tab dedups by nonce.
                    _ws_manager.broadcast(project_id, {
                        "type": "chat.pending_enqueued",
                        "project_id": project_id,
                        "session_id": req.session_id,
                        "holder": holder,
                        "nonce": req.nonce,
                        "content": effective_content,
                        "position": enq["position"],
                    })
                    return JSONResponse(
                        status_code=202,
                        content={
                            "status": "queued_pending_slot",
                            "holding_session_id": holder,
                            "queued_session_id": req.session_id,
                            "nonce": req.nonce,
                            "position": enq["position"],
                        },
                    )
                # Fallback: legacy slot_held contract.
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


class SessionScopeBody(BaseModel):
    """Cross-project read scope for a Quick Tasks (scratch) session (Spec 12)."""
    mode: Literal["all", "selected", "off"]
    selected_project_ids: list[str] = Field(default_factory=list)


@router.get("/agents/{project_id}/sessions/{session_id}/scope")
async def get_session_scope(project_id: str, session_id: str):
    """Return the session's cross-project read-scope record.

    Defaults when unset: scratch → ``all`` (reads every project), normal
    project → ``off`` (single-root).
    """
    project = _project_store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return _agent_manager.get_session_scope(project_id, session_id)


@router.put("/agents/{project_id}/sessions/{session_id}/scope")
async def put_session_scope(project_id: str, session_id: str, body: SessionScopeBody):
    """Set a session's cross-project read scope.

    Only the Quick Tasks (scratch) project may carry cross-project scope
    (409 otherwise); unknown project ids in ``selected_project_ids`` are 422.
    Reads become ambient across the listed projects; writes stay in the Quick
    Tasks workspace. Takes effect on the next turn (read roots + sandbox
    portals recompute without an agent restart).
    """
    project = _project_store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    if not project.get("is_scratch"):
        raise HTTPException(
            status_code=409,
            detail="Cross-project scope is only settable on the Quick Tasks project",
        )
    if body.selected_project_ids:
        known = {p.get("project_id") for p in _project_store.list_projects()}
        unknown = [pid for pid in body.selected_project_ids if pid not in known]
        if unknown:
            raise HTTPException(
                status_code=422, detail=f"Unknown project ids: {unknown}"
            )
    return _agent_manager.set_session_scope(
        project_id, session_id, body.mode, body.selected_project_ids
    )


@router.get("/agents/{project_id}/run-status")
async def agent_run_status(project_id: str, session_id: str | None = None):
    """Return the current runtime status for a project agent.

    Also returns ``current_holder_session_id``: the F1 session_id that
    currently holds the project's active-loop slot, or None.

    ``session_id`` selects which session's ``last_terminal_event`` to include
    — the frontend uses it to re-hydrate a classified error (error_code +
    details) after a page reload, since the agent.status broadcast that
    carried it is ephemeral.
    """
    status = _agent_manager.get_run_status(project_id)
    holder = _agent_manager.current_holder_session_id(project_id)
    return {
        "project_id": project_id,
        "status": status,
        "current_holder_session_id": holder,
        # Number of interactive messages queued behind the slot holder
        # (spec 006, pending-input queue).
        "pending_count": len(_agent_manager.list_pending(project_id)),
        "last_terminal_event": _agent_manager.get_last_terminal_event(
            project_id, session_id=session_id,
        ),
    }


@router.get("/agents/{project_id}/sessions/{session_id}/context")
async def get_session_context(project_id: str, session_id: str):
    """Context in use for one chat session, for the composer's context line.

    Owns no state. The prompt size is the last MANAGEMENT row in the token
    ledger — provider-reported, so it is the real billed prompt rather than the
    ``len/4`` estimate ``ContextManager`` uses internally — and the window
    comes from the registry entry for the model that actually served that call
    (fallback rotation can change it mid-session).

    ``threshold`` is where the loop will compact, computed by the SAME function
    the loop triggers on, so the mark the UI draws cannot drift from the event
    it predicts.

    ``used: null`` means the session has never made a management call. That is
    deliberately distinct from ``0``: the client renders nothing rather than a
    confident empty meter.

    No polling and no new WS event: every ledger append already emits
    ``budget.spend_updated``, and the client refetches this on that.
    """
    from agent_os.agent.context import compaction_threshold_tokens
    from agent_os.budget.ledger import last_context_usage

    project = _project_store.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    workspace = project.get("workspace", "")
    latest = last_context_usage(workspace, session_id) if workspace else None

    if latest is None:
        return {
            "project_id": project_id,
            "session_id": session_id,
            "used": None,
            "window": None,
            "threshold": None,
            "provider": None,
            "model": None,
        }

    provider, model = latest["provider"], latest["model"]
    if _provider_registry is not None:
        window = _provider_registry.get_context_window(provider, model)
    else:
        # No registry wired (lightweight daemons / tests). Fall back rather
        # than 500 — the meter is better conservative than absent.
        from agent_os.config.provider_registry import _FALLBACK_CONTEXT_WINDOW
        window = _FALLBACK_CONTEXT_WINDOW

    return {
        "project_id": project_id,
        "session_id": session_id,
        "used": latest["used"],
        "window": window,
        "threshold": compaction_threshold_tokens(window),
        "provider": provider,
        "model": model,
    }


class PendingCancelRequest(BaseModel):
    """Body for cancelling a queued pending inject (spec 006).

    ``session_id`` identifies the chat session whose pending message(s) to
    drop; ``nonce`` optionally narrows it to a single queued message.
    """
    session_id: str
    nonce: str | None = None


@router.post("/agents/{project_id}/pending/cancel")
async def cancel_pending(project_id: str, req: PendingCancelRequest) -> dict:
    """Cancel / recall a queued message (cross-session OR same-session).

    Drops the still-queued entry (or tombstones it if a dispatch is already in
    flight) and broadcasts ``chat.pending_cancelled``. Returns
    ``{"status": "cancelled", "removed": bool}`` where ``removed`` is True only
    when an actually-still-queued entry was dequeued — the FE relies on this to
    avoid a recall-vs-dispatch double-send (spec 006 §12 R2).
    """
    return _agent_manager.cancel_pending_inject(
        project_id, req.session_id, nonce=req.nonce,
    )


@router.get("/agents/{project_id}/pending")
async def get_pending(project_id: str):
    """Return the project's pending-input queue (spec 006).

    Mobile/relay reconnect recovery, mirroring ``/pending-approval``: lets a
    client rebuild the waiting affordances after a WS drop. Entries carry FULL
    ``content`` (for recall) and a ``kind`` of ``"cross"`` (slot held by
    another session) or ``"same"`` (this session's turn mid-flight). ``{holder,
    pending:[{session_id, nonce, content, position, kind}]}`` (spec 006 §12).
    """
    holder = _agent_manager.current_holder_session_id(project_id)
    return {
        "holder": holder,
        "pending": _agent_manager.list_pending(project_id),
    }


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
        | ``queued`` | ``idle`` | ``stopped``.
      - ``session_uuid``: the per-loop ``Session.session_id`` used as the
        session JSONL filename stem (useful for transcript correlation).

    ``queued`` (spec 081) is a session that exists only as a queued first
    message: another session holds the project's single slot, so nothing has
    been written to disk yet (persist-at-dispatch, spec 006). It carries no
    holder name or queue position — the chat pane's waiting line explains the
    wait — and it disappears if the queued message is cancelled.

    A project with no active sessions returns ``{"sessions": []}``.
    The response shape is intentionally a wrapped list — leaves room for
    future top-level fields (project metadata, totals) without a breaking
    response change.
    """
    if _project_store is not None:
        if _project_store.get_project(project_id) is None:
            raise HTTPException(status_code=404, detail="Project not found")
    # Offloaded: list_sessions() scans + parses every on-disk session JSONL
    # that has no live handle, which must not run on the event loop (this
    # endpoint is refetched on every agent.status WS event).
    sessions = await asyncio.to_thread(_agent_manager.list_sessions, project_id)
    return {"project_id": project_id, "sessions": sessions}


class SessionPatchRequest(BaseModel):
    """Body for updating a session's display state.

    All fields are optional and independent — send any subset. ``name`` is
    the human-readable label; ``pinned`` holds the session at the top of the
    sidebar (spec 067). Neither has any effect on routing or hydration.

    ``pinned_target`` (spec 074) pins the chat session to a sub-agent: the
    composer's sticky "Talking to" selection. Tri-state via
    ``model_fields_set``: absent → untouched; explicit ``null`` → unpin; a
    slug → pin (validated against the installed registry; ``orbital`` is a
    reserved mention, never a pin target).
    """
    name: str | None = None
    pinned: bool | None = None
    pinned_target: str | None = None


# Back-compat alias: this body used to be rename-only.
SessionRenameRequest = SessionPatchRequest


@router.patch("/agents/{project_id}/sessions/{session_id}")
async def patch_session(project_id: str, session_id: str, req: SessionPatchRequest):
    """Update a session's display label and/or its pinned state.

    Updates the live handle in memory when one exists and rewrites the
    ``session_start`` meta line in the JSONL so the change persists. Both
    fields are display-only — F1/F2 identifiers are unchanged.

    Returns ``{"status", "session_id"}`` plus whichever fields were set. 400 if
    the body sets nothing, or if ``name`` is present but empty. 404 if no
    session with this id (F1 or F2/uuid) exists for the project.
    """
    if _project_store is not None and _project_store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    target_set = "pinned_target" in req.model_fields_set
    if req.name is None and req.pinned is None and not target_set:
        raise HTTPException(status_code=400, detail="nothing to update")
    name = None
    if req.name is not None:
        name = req.name.strip()
        # Unchanged from the rename-only contract: an explicitly-sent empty
        # name is still a 400, never a silent no-op.
        if not name:
            raise HTTPException(status_code=400, detail="name must not be empty")
    if target_set and req.pinned_target is not None:
        # Spec 074 validation: the pin target must be an installed, non-built-in
        # registry agent; ``@orbital`` is the reserved manager-aside mention
        # and is rejected here explicitly, before (and regardless of) the
        # registry-slug check.
        slug = req.pinned_target
        if slug.lstrip("@").lower() == "orbital":
            raise HTTPException(
                status_code=422,
                detail="'orbital' is reserved for the management agent and "
                       "cannot be pinned",
            )
        installed = {e["slug"] for e in _installed_sub_agents()}
        if slug not in installed:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown or uninstalled sub-agent {slug!r}",
            )
    patch_kwargs: dict = {"name": name, "pinned": req.pinned}
    if target_set:
        patch_kwargs["pinned_target"] = req.pinned_target
    try:
        # Same lock-contention discipline as the inject route: the disk-path
        # patch write-locks the session JSONL and can lose the race against a
        # concurrent (now-retrying) pinned send — retry instead of 500ing.
        patched = await _retry_session_lock(
            lambda: asyncio.to_thread(
                _agent_manager.patch_session, project_id, session_id,
                **patch_kwargs,
            ))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    except FileLockError:
        raise HTTPException(
            status_code=503, detail="Chat session is busy — please retry")
    result: dict = {"status": "patched", "session_id": session_id}
    if name is not None:
        result["name"] = name
    if req.pinned is not None:
        result["pinned"] = bool(req.pinned)
    if target_set:
        result["pinned_target"] = req.pinned_target
        # Pin time (spec 074 §3.6): hash-guarded AGENTS.md refresh — codex's
        # only context channel is reading that file from the workspace root.
        # Guarded inside reseed_project_agent_md; a user-edited file is never
        # rewritten. Best-effort — a pin must not fail on a seeding hiccup.
        if req.pinned_target is not None:
            try:
                await asyncio.to_thread(
                    reseed_project_agent_md, _project_store, project_id,
                )
            except Exception:
                logger.warning(
                    "AGENTS.md reseed at pin time failed for %s", project_id,
                    exc_info=True,
                )
        # Unpin / retarget (spec 074 §3.5 trigger 1): any change AWAY from a
        # currently pinned worker fires the consolidation pass — detached,
        # single-flight; this PATCH response never waits on it.
        previous = (patched or {}).get("previous_pinned_target")
        if (previous and previous != req.pinned_target
                and _pinned_consolidation is not None):
            _pinned_consolidation.trigger(
                project_id, session_id,
                reason="retarget" if req.pinned_target else "unpin",
            )
    return result


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
            "blocked_sessions": [{"project_id": str, "session_id": str}, ...],
            "budget_paused_projects": [{"project_id": str}, ...]
        }

    ``budget_paused_projects`` (Budget Piece 3 — P3-G) lists projects whose
    queue is paused for budget enforcement (codes only, no spend numbers). The
    sidebar Blocked surface reason-codes these distinctly from the
    approval-pending ``blocked_sessions``. ``blocked_count`` continues to count
    ONLY pending-approval sessions — budget pauses are a separate, additive
    marker so existing approval semantics are untouched.
    """
    blocked = _agent_manager.list_blocked_sessions()
    budget_paused = _agent_manager.list_blocked_budget_projects()
    return {
        "blocked_count": len(blocked),
        "blocked_sessions": blocked,
        "budget_paused_projects": budget_paused,
    }


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
    accepted for compatibility but ignored — a new session is always fresh.

    The ``session_created`` telemetry emit lives in ``new_session()`` itself
    (spec 063 §3), not here — a route handler only ever measures one of the
    four callers."""
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
                reply_text=req.reply_text,
                approve_all=req.approve_all,
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

    from agent_os.daemon_v2.native_worker import is_worker_session_stem
    # List and sort session files by mtime (oldest first)
    session_files = []
    for fname in os.listdir(sessions_dir):
        if fname.endswith(".jsonl"):
            if is_worker_session_stem(fname[:-6]):
                continue  # fanout worker transcript — not management chat
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


# Matches the per-dispatch "message_routed" marker's PROSE — either
#   [Sub-agent] Message sent to {handle}: "{preview}". Transcript: {path}
# or the user-@mention variant
#   [Sub-agent] User sent @{handle}: "{preview}". Transcript: {path}
# Group 1 = handle, group 2 = transcript path. DOTALL + tail-anchored on the
# final "Transcript:" because the message preview may contain newlines.
#
# NOT used by _interleave_sub_agent_summaries below (TASK-dispatch-id-pairing
# replaced the positional-count pairing this regex used to drive with an
# explicit ``_meta.dispatch_id`` identity join — see that function). Kept
# here only because the one-shot legacy-data migration script imports this
# constant to backfill dispatch_ids into transcripts written before this
# task shipped, by recovering (handle, transcript_path) from old markers
# that have no ``_meta`` at all.
_SUB_AGENT_DISPATCH_RE = re.compile(
    r'^\[Sub-agent\]\s+(?:Message sent to |User sent @)([\w.-]+):'
    r'.*\bTranscript:\s*(.+?)\s*$',
    re.DOTALL,
)


def _interleave_sub_agent_summaries(messages: list[dict]) -> list[dict]:
    """Inline each sub-agent dispatch's per-turn final message as a display bubble.

    Identity join (TASK-dispatch-id-pairing): a dispatch marker (the
    ``message_routed`` system message) carries ``_meta.dispatch_id`` +
    ``_meta.transcript_path``, stamped at dispatch time by
    ``LifecycleObserver.on_message_routed`` / ``SubAgentManager.send()``.
    Each transcript turn (``read_sub_agent_summary``) carries the
    ``dispatch_id`` of the boundary row that closed it. This function looks
    up the turn whose id matches the marker's and, when that turn carries a
    final response, inserts a synthetic ``source="sub_agent"`` message right
    after the marker. The frontend renders it as an agent header + tool
    capsule + response bubble, so the management chat carries the durable
    per-turn record of the sub-agent's work — not just the one-line
    completion marker.

    This replaces the old positional pairing ("the i-th dispatch in the
    CURRENT session's messages pairs with the i-th turn-slice in the WHOLE
    transcript file"), which was only correct for a transcript's very FIRST
    chat session — a sub-agent transcript is a persistent per-(workspace,
    handle) file that outlives any one session, so every later session's
    markers paired against stale early turns.

    Honest degradation (never alias another turn's text into the wrong slot,
    and never fall back to position or timestamp):
    - No ``_meta`` at all (a legacy marker written before this task shipped,
      or a non-dispatch system message) → no bubble; the marker line stands
      untouched. A separate migration script backfills ids into old data.
    - The id isn't found in the transcript's turns (still in flight — no
      closing boundary yet, or the id is simply unknown) → no bubble.
    - A matched turn with NO response (errored / interrupted / tool-only) →
      no bubble; the existing one-line terminal marker stands.
    - Idempotent join: once a (transcript_path, dispatch_id) pair has
      rendered a bubble, a LATER marker carrying the SAME pair renders no
      bubble of its own (first marker wins). Scoped by transcript_path too
      — dispatch_ids are minted globally-unique in practice, but the join
      key stays scoped the same way the lookup itself is, so a
      synthetically-colliding id on a DIFFERENT transcript is unaffected.
      This is join hygiene, not a second render path — it guards against a
      known pre-existing root cause (the @mention route fires a second
      marker for the same physical dispatch that send() already marked
      internally) turning one turn into two bubbles; it does not fix that
      root cause.

    Read-only and best-effort: a missing/unreadable transcript is a no-op.
    Operates on the already-paginated page, so it never changes the total-count
    math — and it NEVER persists. The full text lives ONLY on this display
    channel; the management LLM context keeps the capped marker (dual-stream
    isolation, DIAGNOSIS Q1).
    """
    out: list[dict] = []
    slices_by_path: dict[str, "list | None"] = {}
    seen_dispatch_keys: set[tuple[str, str]] = set()
    for msg in messages:
        out.append(msg)
        if msg.get("role") != "system":
            continue
        meta = msg.get("_meta") or {}
        dispatch_id = meta.get("dispatch_id")
        transcript_path = meta.get("transcript_path")
        if not dispatch_id or not transcript_path:
            # No structured id — legacy marker (pre-migration) or a
            # non-dispatch system message. Honest degradation: no bubble.
            continue
        dispatch_key = (transcript_path, dispatch_id)
        if dispatch_key in seen_dispatch_keys:
            # Same (path, id) already rendered a bubble at an earlier marker
            # (e.g. the @mention double-marker) — never render the same turn
            # twice. The later marker's own one-liner stands untouched.
            continue
        if transcript_path not in slices_by_path:
            try:
                slices_by_path[transcript_path] = read_sub_agent_summary(transcript_path)
            except Exception:
                slices_by_path[transcript_path] = None
        slices = slices_by_path[transcript_path]
        if not slices:
            # No transcript on disk (missing/unreadable).
            continue
        turn = next((t for t in slices if t.get("dispatch_id") == dispatch_id), None)
        if turn is None:
            # Still in flight (no closing boundary yet) or an unknown id.
            continue
        if not (turn.get("response") or "").strip():
            # Errored / interrupted / tool-only turn: let the existing terminal
            # marker one-liner speak. Never alias a neighboring turn's text.
            continue
        seen_dispatch_keys.add(dispatch_key)
        out.append({
            "role": "assistant",
            "content": turn.get("response") or "",
            "source": "sub_agent",
            "sub_agent_handle": meta.get("handle", ""),
            "sub_agent_tool_rows": turn.get("tool_rows", []),
            "sub_agent_duration": turn.get("total_duration_seconds", 0.0),
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
        # disk I/O and must not run on the event loop).  list_sessions itself
        # scans + parses the session directory, so it is offloaded too.
        session_uuid: str | None = None
        for entry in await asyncio.to_thread(_agent_manager.list_sessions, project_id):
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
        raw_entries = await asyncio.to_thread(
            _sub_agent_manager.get_all_transcript_entries, project_id
        )
        # Normalize transcript entries to chat message format, dropping the
        # turn_complete boundary rows (empty per-turn delimiters consumed only
        # by the session-filtered split — never user-facing, must not render).
        for entry in raw_entries:
            if entry.get("chunk_type") == "turn_complete":
                continue
            entry.setdefault("role", "agent")
            sub_entries.append(entry)

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


@router.post("/providers/tokendance/signin")
async def tokendance_signin():
    """One-click TokenDance key provisioning (Spec 47 Tier 2).

    Blocking: runs the PKCE loopback flow (system browser consent, up to
    ~300s), validates the minted key against their balance endpoint, and
    persists it to the credential store. The raw key never leaves the daemon —
    the response carries only the set-flag and a display mask (same format as
    ``settings_store.get_masked``).
    """
    if _credential_store is None:
        raise HTTPException(status_code=501, detail="Credential store not available")
    from agent_os.providers_auth.tokendance import (
        TokenDanceProvisioningError,
        provision_api_key,
    )
    try:
        key = await provision_api_key()
    except TokenDanceProvisioningError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    try:
        _credential_store.set_api_key(key)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"key was created but could not be stored: {exc}",
        )
    from agent_os import telemetry
    telemetry.emit("key_set", {"provider": "tokendance"})
    telemetry.latch("key_set")
    masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "****"
    return {"api_key_set": True, "api_key_masked": masked}


class FetchModelsRequest(BaseModel):
    # `provider` defaults to "custom" so a local OpenAI-compatible server
    # (LM Studio / llama.cpp / Ollama) can be queried with just a base_url —
    # the frontend omits `provider` for the Custom provider.
    provider: str = "custom"
    api_key: str | None = None
    base_url: str | None = None
    # The Custom provider exposes an SDK picker, and the registry's `custom`
    # entry always says "openai" — so the request must be able to carry the
    # protocol, or an Anthropic-format endpoint gets listed at the OpenAI path
    # with the OpenAI auth header. None = fall back to the registry entry.
    sdk: str | None = None


_CHAT_PROTOCOL_PREFIXES = ("openai:chat-completions", "anthropic:messages")


def _chat_model_ids(data: dict) -> list[str]:
    """Extract chat-capable model IDs from a provider /models response.

    Multi-modal routers (TokenDance) list image/video/TTS models alongside
    chat models and tag every entry with ``supported_protocols``; entries
    whose protocols include no chat protocol are dropped so the model picker
    only offers models the chat path can actually call. Entries without the
    field (every other provider) are kept.
    """
    models = []
    for m in data.get("data", []):
        model_id = m.get("id", "")
        if not model_id:
            continue
        protos = m.get("supported_protocols")
        if isinstance(protos, list) and protos and not any(
            isinstance(p, str) and p.startswith(_CHAT_PROTOCOL_PREFIXES)
            for p in protos
        ):
            continue
        models.append(model_id)
    return models


@router.post("/providers/models")
async def fetch_models(req: FetchModelsRequest):
    """Proxy request to provider's /v1/models endpoint."""
    import httpx
    # Use injected registry
    provider_info = _provider_registry.get_provider_data(req.provider) if _provider_registry else None
    base_url = req.base_url or (provider_info["base_url"] if provider_info else None)
    if not base_url:
        raise HTTPException(status_code=400, detail="No base_url for provider")

    # Handle Anthropic (different models endpoint). Omit the auth header
    # entirely when there is no key — a keyless local server needs none, and an
    # empty `Bearer ` (trailing space) is an illegal HTTP header value that
    # httpx rejects.
    sdk = req.sdk or (provider_info.get("sdk", "openai") if provider_info else "openai")

    # The frontend clears the API-key field once a key is persisted, and its
    # body omits `api_key` when the field is empty — so the post-save steady
    # state sends no key at all. Fall back to the stored global key exactly as
    # /providers/test does; otherwise every key-gated provider (11 of 14 answer
    # /models with 401/403 unauthenticated) silently drops back to the static
    # suggested list right after the user saves.
    api_key = req.api_key
    if not (api_key or "").strip() and _credential_store is not None:
        api_key = _credential_store.get_api_key() or ""

    if sdk == "anthropic":
        models_url = base_url.rstrip("/") + "/v1/models"
        headers = {"anthropic-version": "2023-06-01"}
        if api_key:
            headers["x-api-key"] = api_key
    else:
        models_url = base_url.rstrip("/") + "/models"
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

    # Fail fast. A blocked or slow /models endpoint holds the model picker in
    # its loading state for the whole timeout before falling back to the
    # suggested list, which reads as "the app is broken" long before it
    # resolves (api.mistral.ai: ~10s to a TLS failure from a CN network).
    async with httpx.AsyncClient(timeout=6) as client:
        try:
            resp = await client.get(models_url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            # Normalize: extract chat-capable model IDs
            return {"models": sorted(_chat_model_ids(data))}
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Provider returned {e.response.status_code}")
        except Exception as e:
            # str(httpx.ReadTimeout()) is "" — the most likely failure here is
            # exactly the one that would report nothing at all.
            raise HTTPException(status_code=502, detail=str(e) or type(e).__name__)


# Phrases that mean the CREDENTIAL was rejected. Everything else a gateway
# refuses a request for — no payment method, a region gate, a quota, an
# unenrolled model — also arrives as 401/403, and calling those "Invalid API
# key" sends the user to re-issue a key that works fine. Reported from the
# field on OpenCode Go: deepseek-v4-flash answers 403 RegionError until the
# workspace opts into China hosting. Auth phrasing is a small stable set; the
# reasons a request can be refused are open-ended, so the allowlist covers
# auth and everything else passes through in the provider's own words.
_AUTH_FAILURE_PHRASES = (
    "api key", "apikey", "api-key",
    "authentication", "unauthorized", "invalid token", "invalid credentials",
)


def _unwrap_provider_message(raw: str) -> str:
    """Pull the provider's sentence out of an SDK envelope.

    SDK errors arrive as ``Error code: 403 - {'type': 'error', 'error':
    {'message': '...'}}`` — a Python repr, which is unreadable in a red UI
    line. Return the innermost ``message`` when the envelope parses, else the
    raw string unchanged.
    """
    import ast

    body = raw.split(" - ", 1)[1] if " - " in raw else raw
    try:
        parsed = ast.literal_eval(body.strip())
    except (ValueError, SyntaxError):
        return raw
    for _ in range(4):  # {'error': {'error': {...}}} nests at most a little
        if not isinstance(parsed, dict):
            break
        msg = parsed.get("message")
        if isinstance(msg, str) and msg:
            return msg
        nxt = parsed.get("error")
        if nxt is None:
            break
        parsed = nxt
    return raw


def _describe_auth_failure(raw: str | None) -> str:
    """Map a 401/403 to display text: the stable "Invalid API key" only when
    the provider says the credential was the problem."""
    message = raw or ""
    if any(p in message.lower() for p in _AUTH_FAILURE_PHRASES):
        return "Invalid API key"
    return _unwrap_provider_message(message)


class TestConnectionRequest(BaseModel):
    # Local OpenAI-compatible servers (LM Studio / llama.cpp / Ollama) are
    # reached via the "Custom" provider with no API key. The frontend omits
    # `provider` (sends undefined) and omits `api_key` when empty, so both must
    # default — otherwise FastAPI 422s with `type: missing` before the handler
    # (which already tolerates a custom provider + empty key) ever runs.
    provider: str = "custom"
    model: str
    api_key: str = ""
    base_url: str | None = None
    sdk: str = "openai"


@router.post("/providers/test")
async def test_connection(req: TestConnectionRequest):
    """Test connection by sending a minimal completion request."""
    # Use injected registry
    provider_info = _provider_registry.get_provider_data(req.provider) if _provider_registry else None
    base_url = req.base_url or (provider_info["base_url"] if provider_info else None)
    sdk = req.sdk or (provider_info.get("sdk", "openai") if provider_info else "openai")

    # Per-model endpoint override wins over both: the frontend can only send
    # the PROVIDER-level sdk, so on a mixed-protocol aggregator (OpenCode
    # Zen/Go) Test Connection would otherwise fail on a model that works in a
    # real turn. Models without an override leave the user's choice intact —
    # that is every Custom / self-hosted endpoint.
    if _provider_registry is not None:
        _mi = _provider_registry.get_model_info(req.provider, req.model)
        sdk = _mi.sdk or sdk
        base_url = _mi.base_url or base_url

    from agent_os.agent.providers.openai_compat import LLMProvider
    from agent_os.agent.providers.types import LLMError, ContextOverflowError

    # An empty api_key with a saved global key means "test the stored key":
    # the frontend clears the field once a key is persisted (paste-and-save,
    # or the TokenDance one-click flow), so fall back to the credential store
    # rather than constructing a keyless client — the raw SDK otherwise
    # surfaces "Missing credentials … set OPENAI_API_KEY". Custom/local
    # servers that genuinely need no key are unaffected: with no stored
    # global key the fallback resolves to "" exactly as before.
    api_key = req.api_key
    if not api_key.strip() and _credential_store is not None:
        api_key = _credential_store.get_api_key() or ""

    try:
        provider = LLMProvider(
            req.model, api_key, base_url, sdk=sdk,
            extra_headers=provider_info.get("extra_headers") if provider_info else None,
        )
        result = await provider.complete(
            messages=[{"role": "user", "content": "hi"}],
        )
        return {"status": "ok", "message": f"Connected to {req.provider} using {req.model}"}
    except ContextOverflowError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except LLMError as e:
        status = e.status_code or 500
        if status in (401, 403):
            detail = _describe_auth_failure(e.message)
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


def _validate_trigger_timezone(tz_name: str) -> None:
    """400 on an unknown IANA zone.

    The tick loop falls back to UTC with a log warning for an unknown zone
    (trigger_manager._is_due), which is the right runtime behaviour but a
    silent one — a form that accepts 'Asia/Shangai' would fire at the wrong
    hour forever. Reject it at the door instead.
    """
    import pytz

    try:
        pytz.timezone(tz_name)
    except pytz.UnknownTimeZoneError:
        raise HTTPException(status_code=400, detail=f"Unknown timezone: {tz_name}")


@router.post("/projects/{project_id}/triggers", status_code=201)
async def create_trigger(project_id: str, body: TriggerCreateRequest):
    """Create a trigger from the UI.

    Mirrors CreateTriggerTool's record shape exactly — the agent tool and this
    route must produce interchangeable records, since both feed the same store,
    scheduler and list surfaces.
    """
    from agent_os.daemon_v2.trigger_manager import generate_trigger_id, validate_trigger

    project = _project_store.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    trigger = {
        "id": generate_trigger_id(),
        "name": body.name,
        "enabled": body.enabled,
        "type": body.type,
        "task": body.task,
        "last_triggered": None,
        "trigger_count": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if body.type == "schedule":
        if body.schedule is None:
            raise HTTPException(
                status_code=400, detail="Schedule trigger requires schedule.cron"
            )
        _validate_trigger_timezone(body.schedule.timezone)
        trigger["schedule"] = {
            "cron": body.schedule.cron,
            # Fall back to the cron itself rather than an empty caption: the
            # strip and the list render `human` as the row label.
            "human": body.schedule.human or body.schedule.cron,
            "timezone": body.schedule.timezone,
        }
    else:
        trigger["watch_path"] = body.watch_path or ""
        trigger["patterns"] = body.patterns or []
        trigger["recursive"] = body.recursive
        trigger["debounce_seconds"] = body.debounce_seconds

    workspace = project.get("workspace", "")
    error = validate_trigger(trigger, workspace=workspace)
    if error:
        raise HTTPException(status_code=400, detail=error)

    triggers = project.get("triggers", [])
    triggers.append(trigger)
    _project_store.update_project(project_id, {"triggers": triggers})

    if _trigger_manager is not None:
        _trigger_manager.register_trigger(project_id, trigger)

    return trigger


@router.patch("/projects/{project_id}/triggers/{trigger_id}")
async def update_trigger(project_id: str, trigger_id: str, body: TriggerUpdateRequest):
    """Partially update a trigger — enable/disable and every editable field.

    Emits `trigger.updated`, never created/deleted: the record did not appear
    or go away, and treating a disable as a delete made the row vanish from
    every live list until the next refetch.
    """
    from agent_os.daemon_v2.trigger_manager import validate_trigger

    project = _project_store.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    triggers = project.get("triggers", [])
    trigger = next((t for t in triggers if t.get("id") == trigger_id), None)
    if trigger is None:
        raise HTTPException(status_code=404, detail="Trigger not found")

    updates = body.model_dump(exclude_unset=True)

    # Type is immutable, so fields belonging to the other kind are noise — drop
    # them rather than writing a schedule onto a file-watch record.
    if trigger.get("type") == "schedule":
        for key in ("watch_path", "patterns", "recursive", "debounce_seconds"):
            updates.pop(key, None)
    else:
        updates.pop("schedule", None)

    if "schedule" in updates:
        schedule = updates.pop("schedule")
        if schedule is not None:
            # Merge onto the existing schedule so a body that omits `human`
            # (or `timezone`) doesn't blank the stored caption.
            merged_schedule = dict(trigger.get("schedule") or {})
            merged_schedule.update({k: v for k, v in schedule.items() if v is not None})
            merged_schedule.setdefault("timezone", "UTC")
            merged_schedule["human"] = (
                merged_schedule.get("human") or merged_schedule.get("cron", "")
            )
            _validate_trigger_timezone(merged_schedule["timezone"])
            updates["schedule"] = merged_schedule

    # A partial update must not be able to null out a required field.
    candidate = dict(trigger)
    candidate.update({k: v for k, v in updates.items() if v is not None})

    workspace = project.get("workspace", "")
    error = validate_trigger(candidate, workspace=workspace)
    if error:
        raise HTTPException(status_code=400, detail=error)

    # Mutate in place: `triggers` is the list we persist, and the row keeps
    # its position (an edit must not reorder the list).
    trigger.clear()
    trigger.update(candidate)
    _project_store.update_project(project_id, {"triggers": triggers})

    if _trigger_manager is not None:
        _trigger_manager.apply_trigger_update(project_id, trigger)

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


# Handles include both plain CLI slugs ("claude-code") and native fanout
# worker handles ("worker:<fanout_id>-<index>", native_worker.make_worker_handle)
# — colon is allowed for the latter. No dots/slashes: the handle is joined
# directly into a filesystem path (ProjectPaths.sub_agent_dir), so this also
# doubles as the path-traversal guard.
_TRANSCRIPT_HANDLE_RE = re.compile(r"^[A-Za-z0-9_:-]+$")


@router.get("/agents/{project_id}/sub-agents/{handle}/transcript")
async def sub_agent_transcript(project_id: str, handle: str,
                               session_id: str | None = None):
    """Read-only transcript playback for one sub-agent handle (spec 009
    Section 0.5, Task 4) — the SOLE data source for the frontend drill-in
    view (Task 5, already committed). Response shape is FROZEN; see
    web/src/types.ts's ``SubAgentTranscriptResult`` — do not change field
    names/types here without updating that contract in lockstep.

    ``kind`` is derived purely from the handle prefix: native fanout workers
    are minted as ``worker:<fanout_id>-<index>``; everything else is a CLI
    adapter slug. ``resumable`` mirrors ``kind == "cli"`` — workers are
    one-shot, anonymous sessions with no composer.

    ``session_id`` defaults to the project's active-loop holder (same
    no-sentinel resolution ``/sub-agents/status`` and the stop route use
    above). It only affects which live adapter's ``display_name`` is
    preferred — the on-disk transcript itself is keyed by (workspace,
    handle), not by session (BACKLOG 005 §4b), so omitting it still finds
    the transcript; it just falls back to the handle for ``display_name``.
    """
    if _sub_agent_manager is None:
        raise HTTPException(status_code=503, detail="Sub-agent manager not available")
    if not _TRANSCRIPT_HANDLE_RE.match(handle):
        raise HTTPException(status_code=400, detail="Invalid agent handle")

    sid = session_id or _agent_manager.current_holder_session_id(project_id)

    raw_entries = _sub_agent_manager.read_transcript_entries(project_id, handle, sid)
    if raw_entries is None:
        raise HTTPException(status_code=404, detail="No transcript found for this handle")

    # turn_complete rows are empty-content turn-boundary instrumentation
    # (ProcessManager._append_turn_boundary) — never rendered anywhere else
    # in the codebase (the /chat unfiltered merge drops them too); must not
    # leak into the drill-in playback.
    entries = [e for e in raw_entries if e.get("chunk_type") != "turn_complete"]

    kind = "worker" if handle.startswith("worker:") else "cli"

    # display_name: prefer the live adapter's name for this session; a
    # worker's label only exists in-memory (native_worker task label), so a
    # non-live handle gracefully falls back to the raw handle.
    display_name = handle
    if sid is not None:
        for a in _sub_agent_manager.list_active(project_id, session_id=sid):
            if a["handle"] == handle:
                display_name = a["display_name"]
                break

    # ADDITIVE field (D1, issues 2+3): the worker's real chat session id, so
    # the frontend drill-in can fetch /chat?session_id=<session_uuid> and
    # render chat-shaped instead of the flat entries view. Null for CLI
    # handles (they have no such session at all).
    #
    # Live path first: mid-batch, the fanout registry still owns the routing
    # entry. This is NOT sufficient alone — FanoutRegistry.resolve_group pops
    # every handle's routing entry once the group resolves (fanout.py), so a
    # completed fanout's handle is unknown to the registry within moments of
    # finishing. The disk fallback below is therefore MANDATORY, not
    # belt-and-braces: scan the project's sessions dir for the newest file
    # whose stem starts with the mint prefix native_worker.py uses
    # (``worker_{_sanitize_for_filename(handle)}_``) — the trailing
    # underscore makes the prefix collision-safe (handle "...-1" cannot match
    # a file minted for "...-10").
    session_uuid: str | None = None
    if kind == "worker":
        session_uuid = _sub_agent_manager.fanout_session_uuid_for_handle(
            project_id, handle, sid)
        if session_uuid is None:
            import glob as _glob

            from agent_os.daemon_v2.native_worker import _sanitize_for_filename

            project = _project_store.get_project(project_id) if _project_store else None
            workspace = (project or {}).get("workspace", "")
            if workspace:
                sessions_dir = ProjectPaths(workspace).sessions_dir
                prefix = f"worker_{_sanitize_for_filename(handle)}_"
                candidates = _glob.glob(os.path.join(sessions_dir, prefix + "*.jsonl"))
                if candidates:
                    newest = max(candidates, key=os.path.getmtime)
                    session_uuid = os.path.splitext(os.path.basename(newest))[0]

    return {
        "handle": handle,
        "display_name": display_name,
        "kind": kind,
        "resumable": kind == "cli",
        "entries": entries,
        "session_uuid": session_uuid,
    }


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
