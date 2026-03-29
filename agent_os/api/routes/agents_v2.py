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
import zipfile
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from pydantic import BaseModel, field_validator

from agent_os.agent.prompt_builder import Autonomy
from agent_os.agent.skills import SkillLoader
from agent_os.daemon_v2.project_store import project_dir_name as _project_dir_name

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
    agent_credentials: dict | None = None
    network_extra_domains: list[str] | None = None
    agent_name: str | None = None
    is_scratch: bool = False
    notification_prefs: dict | None = None
    llm_fallback_models: list[dict] | None = None
    budget_limit_usd: float | None = None
    budget_action: str | None = None


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


class StartAgentRequest(BaseModel):
    project_id: str
    initial_message: str | None = None


class InjectRequest(BaseModel):
    content: str
    target: str | None = None
    nonce: str | None = None

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


class ApproveRequest(BaseModel):
    tool_call_id: str
    reply_text: str | None = None
    approve_all: bool = False
    response_payload: str | None = None  # User text input for MFA codes etc.


class DenyRequest(BaseModel):
    tool_call_id: str
    reason: str


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


# ---- Session cache for sub-agent-only projects ----
_sub_agent_sessions: dict = {}  # project_id -> Session


def _get_or_create_session(project_id: str, workspace: str, project_name: str = ""):
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

    dir_name = _project_dir_name(project_name, project_id)
    session_id = f"subagent_{uuid4().hex[:8]}"
    session = Session.new(session_id, workspace, project_dir_name=dir_name)
    _sub_agent_sessions[project_id] = session
    return session


# ---- Helpers ----

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
    return result


def _read_file_or_empty(path: str) -> str:
    """Read a file and return its content, or empty string if missing."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except (OSError, FileNotFoundError):
        return ""


def _enrich_with_disk_content(result: dict, workspace: str, dir_name: str) -> dict:
    """Attach project_goals_content and user_directives_content from disk files."""
    goals_path = os.path.join(workspace, "orbital", dir_name, "instructions", "project_goals.md")
    rules_path = os.path.join(workspace, "orbital", dir_name, "instructions", "user_directives.md")
    result["project_goals_content"] = _read_file_or_empty(goals_path)
    result["user_directives_content"] = _read_file_or_empty(rules_path)
    return result


def _write_workspace_file(workspace: str, filename: str, content: str, dir_name: str) -> None:
    """Write content to {workspace}/orbital/{dir_name}/instructions/{filename}."""
    instructions_dir = os.path.join(workspace, "orbital", dir_name, "instructions")
    os.makedirs(instructions_dir, exist_ok=True)
    with open(os.path.join(instructions_dir, filename), "w", encoding="utf-8") as f:
        f.write(content)


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
        project_data["budget_action"] = req.budget_action
    try:
        pid = _project_store.create_project(project_data)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    # Copy default skills into workspace
    if not req.is_scratch:
        default_skills_src = os.path.join(os.path.dirname(__file__), "..", "..", "default_skills")
        default_skills_src = os.path.normpath(default_skills_src)
        if os.path.isdir(default_skills_src):
            dest_skills = os.path.join(req.workspace, "skills")
            if not os.path.exists(dest_skills):
                shutil.copytree(default_skills_src, dest_skills)

    project = _project_store.get_project(pid)
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
    dir_name = _project_dir_name(project.get("name", ""), project_id)
    result = _redact_project(project)
    _enrich_with_disk_content(result, workspace, dir_name)
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
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    # Handle runtime budget spend reset separately (not a regular config field)
    # Accept both field names for compatibility
    runtime_budget_reset = updates.pop("runtime_budget_spent_usd", None)
    budget_spent_alias = updates.pop("budget_spent_usd", None)
    if runtime_budget_reset is None and budget_spent_alias is not None:
        runtime_budget_reset = budget_spent_alias
    if runtime_budget_reset is not None:
        _project_store.update_runtime(project_id, {"budget_spent_usd": runtime_budget_reset})
    # Handle workspace file content fields separately
    workspace = project.get("workspace", "")
    dir_name = _project_dir_name(project.get("name", ""), project_id)
    goals_content = updates.pop("project_goals_content", None)
    rules_content = updates.pop("user_directives_content", None)
    if goals_content is not None:
        _write_workspace_file(workspace, "project_goals.md", goals_content, dir_name)
    if rules_content is not None:
        _write_workspace_file(workspace, "user_directives.md", rules_content, dir_name)
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
    updated = _project_store.get_project(project_id)
    result = _redact_project(updated)
    _enrich_with_disk_content(result, workspace, dir_name)
    result["budget_spent_usd"] = result.get("runtime", {}).get("budget_spent_usd", 0.0)
    return result


_cleanup_logger = logging.getLogger(__name__)

def _cleanup_project_files(workspace: str, project_id: str, clear_output: bool,
                           project_name: str = "") -> None:
    """Remove project data files from the workspace orbital directory.

    All project-specific data lives under orbital/{dir_name}/.  Deleting that
    single directory is sufficient.  Legacy flat directories from pre-namespaced
    workspaces are also cleaned up (safe: they only exist in old workspaces).
    """
    agent_os_dir = os.path.join(workspace, "orbital")
    if not os.path.isdir(agent_os_dir):
        return

    # Delete the project's namespace directory (all project data)
    dir_name = _project_dir_name(project_name, project_id)
    project_dir = os.path.join(agent_os_dir, dir_name)
    _rmtree_safe(project_dir)

    # Also try the raw project_id directory (backward compat)
    old_project_dir = os.path.join(agent_os_dir, project_id)
    _rmtree_safe(old_project_dir)

    # Legacy flat cleanup (pre-migration workspaces).
    # TODO: Remove after one release cycle.
    for legacy_subdir in (
        "browser-screenshots", "browser-pdfs", "shell_output",
        "instructions", ".tmp",
    ):
        legacy_path = os.path.join(agent_os_dir, legacy_subdir)
        if os.path.isdir(legacy_path):
            _rmtree_safe(legacy_path)
    _remove_safe(os.path.join(agent_os_dir, "approval_history.jsonl"))


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
async def bulk_delete_projects(
    body: BulkDeleteRequest,
    delete_workspace: bool = Query(default=False),
):
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
                await _agent_manager.stop_agent(pid)
            workspace = p.get("workspace", "")
            if workspace:
                _cleanup_project_files(workspace, pid, delete_workspace,
                                       project_name=p.get("name", ""))
            _sub_agent_sessions.pop(pid, None)
            _project_store.delete_project(pid)
            deleted += 1
        except Exception:
            failed += 1

    return {"deleted": deleted, "failed": failed}


@router.delete("/projects/{project_id}")
async def delete_project(project_id: str, clear_output: bool = False):
    project = _project_store.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    # Stop agent if running
    if _agent_manager.is_running(project_id):
        await _agent_manager.stop_agent(project_id)

    # Clean up project files on disk
    workspace = project.get("workspace", "")
    if workspace:
        _cleanup_project_files(workspace, project_id, clear_output,
                               project_name=project.get("name", ""))

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
            req.project_id, config, initial_message=req.initial_message
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "started"}


@router.post("/agents/{project_id}/inject")
async def inject_message(project_id: str, req: InjectRequest):
    # Verify project exists before attempting inject
    project = _project_store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    if req.target and _sub_agent_manager is not None:
        # Route to sub-agent (Path B: direct @mention)
        workspace = project.get("workspace", "")
        session = _get_or_create_session(project_id, workspace, project.get("name", ""))

        # Persist user message BEFORE sending to sub-agent
        user_ts = datetime.now(timezone.utc).isoformat()
        user_msg: dict = {
            "role": "user",
            "content": req.content,
            "target": req.target,
            "timestamp": user_ts,
        }
        if req.nonce:
            user_msg["nonce"] = req.nonce
        session.append(user_msg)

        # Auto-start sub-agent if not running
        try:
            result = await _sub_agent_manager.send(project_id, req.target, req.content)
        except Exception:
            raise HTTPException(status_code=404, detail="No active session for project")
        if result.startswith("Error: agent") and "not running" in result:
            try:
                await _sub_agent_manager.start(project_id, req.target)
                result = await _sub_agent_manager.send(project_id, req.target, req.content)
            except Exception:
                raise HTTPException(status_code=404, detail=f"Failed to auto-start {req.target}")

        # Broadcast acknowledgement so ChatView knows the message was sent
        ack_ts = datetime.now(timezone.utc).isoformat()
        _ws_manager.broadcast(project_id, {
            "type": "chat.sub_agent_message",
            "project_id": project_id,
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
                message_preview=req.content[:100],
                transcript_path=transcript_path,
            )

        return {"status": result}
    else:
        # Route to management agent (auto-starts if no session)
        result = await _agent_manager.inject_message(
            project_id, req.content, nonce=req.nonce,
        )
        return {"status": result}


@router.get("/agents/{project_id}/run-status")
async def agent_run_status(project_id: str):
    """Return the current runtime status for a project agent."""
    status = _agent_manager.get_run_status(project_id)
    return {"project_id": project_id, "status": status}


@router.get("/agents/{project_id}/pending-approval")
async def get_pending_approval(project_id: str):
    """Return the current pending approval payload, if any.

    Used by mobile clients to recover approval cards missed via WebSocket.
    """
    approval = _agent_manager.get_pending_approval(project_id)
    if approval is None and _sub_agent_manager is not None:
        approval = _sub_agent_manager.get_pending_sub_agent_approval(project_id)
    if approval is None:
        return {"pending": False}
    return {"pending": True, **approval}


@router.post("/agents/{project_id}/stop")
async def stop_agent(project_id: str):
    try:
        await _agent_manager.stop_agent(project_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="No active session for project")
    return {"status": "stopping"}


@router.post("/agents/{project_id}/new-session")
async def new_session(project_id: str):
    result = await _agent_manager.new_session(project_id)
    return result


@router.post("/agents/{project_id}/approve")
async def approve(project_id: str, req: ApproveRequest):
    try:
        await _agent_manager.approve(
            project_id, req.tool_call_id, reply_text=req.reply_text,
            approve_all=req.approve_all,
        )
    except KeyError:
        # Try sub-agent approval path
        if _sub_agent_manager is not None:
            routed = await _sub_agent_manager.resolve_sub_agent_approval(
                project_id, req.tool_call_id, approved=True
            )
            if not routed:
                raise HTTPException(status_code=404, detail="No pending approval found")
        else:
            raise HTTPException(status_code=404, detail="No pending approval found")
    _ws_manager.broadcast(project_id, {
        "type": "approval.resolved",
        "project_id": project_id,
        "tool_call_id": req.tool_call_id,
        "resolution": "approved",
    })
    return {"status": "approved"}


@router.post("/agents/{project_id}/deny")
async def deny(project_id: str, req: DenyRequest):
    try:
        await _agent_manager.deny(project_id, req.tool_call_id, req.reason)
    except KeyError:
        # Try sub-agent approval path
        if _sub_agent_manager is not None:
            routed = await _sub_agent_manager.resolve_sub_agent_approval(
                project_id, req.tool_call_id, approved=False
            )
            if not routed:
                raise HTTPException(status_code=404, detail="No pending approval found")
        else:
            raise HTTPException(status_code=404, detail="No pending approval found")
    _ws_manager.broadcast(project_id, {
        "type": "approval.resolved",
        "project_id": project_id,
        "tool_call_id": req.tool_call_id,
        "resolution": "denied",
    })
    return {"status": "denied"}


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


@router.get("/agents/{project_id}/chat")
async def chat_history(
    project_id: str,
    limit: int = Query(default=0, ge=0, description="Max messages to return (0 = all)"),
    offset: int = Query(default=0, ge=0, description="Skip first N messages from the end"),
):
    """Return chat history, newest messages last.

    Pagination returns the *most recent* messages:
    - limit=20, offset=0 → last 20 messages
    - limit=20, offset=20 → messages 21-40 from the end
    - limit=0 (default) → all messages (backward-compatible)

    Response includes X-Total-Count header for pagination UI.
    """
    project = _project_store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    workspace = project["workspace"]
    dir_name = _project_dir_name(project.get("name", ""), project_id)
    sessions_dir = os.path.join(workspace, "orbital", dir_name, "sessions")

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

    from starlette.responses import JSONResponse
    resp = JSONResponse(content=messages)
    resp.headers["X-Total-Count"] = str(total)
    return resp


# ---- Agent Registry / Setup Endpoints ----


@router.get("/agents/available")
async def available_agents():
    """Return setup status for all registered agents."""
    if _setup_engine is None:
        return []
    statuses = _setup_engine.check_all()
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
    skills_base = os.path.realpath(os.path.join(workspace, "skills"))
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
    text = content_bytes.decode("utf-8", errors="replace")
    name, description = _validate_skill_content(text)

    dir_name = _sanitize_skill_dir_name(name)
    skill_dir = os.path.join(workspace, "skills", dir_name)

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

        dest_dir = os.path.join(workspace, "skills", dir_name)
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
