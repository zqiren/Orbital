# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Agent lifecycle management — the central orchestrator.

Wires Components A-E together per project and manages the agent loop lifecycle.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from agent_os.agent.context import ContextManager
from agent_os.agent.loop import AgentLoop
from agent_os.agent.prompt_builder import PromptBuilder, PromptContext
from agent_os.agent.providers.openai_compat import LLMProvider
from agent_os.agent.session import Session
from agent_os.agent.tools.registry import ToolRegistry
from agent_os.agent.project_paths import ProjectPaths
from agent_os.agent.workspace_files import WorkspaceFileManager, run_session_end_routine
from agent_os.config.provider_registry import ProviderRegistry
from agent_os.daemon_v2.default_skills_installer import install_default_skills
from agent_os.daemon_v2.autonomy import AutonomyInterceptor
from agent_os.daemon_v2.sub_agent_visibility import resolve_visible_sub_agent_slugs
from agent_os.daemon_v2.models import (
    AgentConfig,
    SessionKey,
    detect_os,
    make_session_key,
    resolve_api_key,
    resolve_session_id,
)
from agent_os.queue.dispatcher import QueueDispatcher
from agent_os.queue.store import QueueStore

logger = logging.getLogger(__name__)


def _sanitize_project_name(name: str) -> str:
    """Sanitize project name for use in filenames."""
    s = re.sub(r'[^\w-]', '_', name.lower())
    s = re.sub(r'_+', '_', s).strip('_')
    return s[:30] if s else 'project'


@dataclass
class ProjectHandle:
    session: object
    loop: object
    provider: object
    registry: object
    context_manager: object
    interceptor: object
    task: asyncio.Task | None
    trigger_source: str | None = None
    config_snapshot: dict = field(default_factory=dict)
    started_at: str = ""
    # Wall-clock of the last USER interaction (or system-initiated session
    # work) on this handle — message, approval, deny, loop resume. NOT
    # updated by the agent loop's own chatter (tool calls, LLM responses).
    # Drives idle eviction; ephemeral (never persisted — a daemon restart
    # drops all handles anyway). See ``_evict_idle_once``.
    last_activity: float = field(default_factory=time.time)


class AgentManager:
    """Central orchestrator for management agent per project."""

    def __init__(self, project_store, ws_manager, sub_agent_manager,
                 activity_translator, process_manager, platform_provider=None,
                 registry=None, setup_engine=None, settings_store=None,
                 credential_store=None, browser_manager=None, user_credential_store=None,
                 trigger_manager=None, provider_registry=None):
        self._project_store = project_store
        self._ws = ws_manager
        self._sub_agent_manager = sub_agent_manager
        self._activity_translator = activity_translator
        self._process_manager = process_manager
        self._platform_provider = platform_provider
        self._registry = registry
        self._setup_engine = setup_engine
        self._settings_store = settings_store
        self._credential_store = credential_store
        self._browser_manager = browser_manager
        self._user_credential_store = user_credential_store
        self._trigger_manager = None  # set after TriggerManager is created
        self._provider_registry = provider_registry or ProviderRegistry()
        # ``_handles`` is keyed by ``SessionKey == (project_id, session_id)``
        # so multiple chat sessions within the same project can each own a
        # distinct loop / session / context-manager bundle. Single-loop
        # callers route through ``DEFAULT_SESSION_ID`` automatically via
        # ``_resolve_session_id`` + ``make_session_key``.
        self._handles: dict[SessionKey, ProjectHandle] = {}
        # ``_idle_poll_tasks`` is keyed by ``SessionKey`` so each session in
        # a project has its own poll task. Under the single-slot invariant
        # only one entry per project is ever alive at a time, but keying by
        # SessionKey keeps state aligned with ``_handles`` and prevents a
        # second session from silently overwriting the first's poll task if
        # the slot ever races.
        self._idle_poll_tasks: dict[SessionKey, asyncio.Task] = {}
        # Per-session record of the most recent terminal event (error /
        # stopped / new_session). Lives outside `_handles` so it survives
        # handle pop on /stop. Cleared by `_clear_last_terminal_event`
        # whenever a session transitions out of a terminal state (next
        # successful inject). Exposed via `list_sessions` so the frontend
        # can render a persistent warning glyph across WS disconnects and
        # page reloads. See TASK-state-model-alignment-fixes.md §2.4.
        self._last_terminal_events: dict[SessionKey, dict] = {}
        self._sleep_handle: object | None = None
        # Queue infrastructure (per project)
        self._queue_stores: dict[str, QueueStore] = {}
        self._dispatchers: dict[str, QueueDispatcher] = {}
        # Idle-eviction sweep task (started via ``start_eviction``, cancelled
        # in ``shutdown``). None until started.
        self._eviction_task: asyncio.Task | None = None

    @staticmethod
    def _resolve_session_id(session_id: str | None) -> str | None:
        """Internal-class None-policy (seam 3 / D1): PASSTHROUGH.

        For internal/back-compat paths (broadcast, loop-done, stop, etc.) a
        ``None`` session_id is passed through unchanged. ``make_session_key``
        then yields ``(project_id, None)`` which matches no live handle, so the
        path degrades to a handle-miss / no-op (or a None-stamped broadcast that
        the frontend's strict session routing drops). This is the SAFE-degrade
        default — never a crash, never a "default" phantom session. Callers that
        must resolve None to a concrete session use ``_sid_read`` (read/affect →
        holder) or ``_sid_inject`` (inject → persistent chat) instead.
        """
        return resolve_session_id(session_id, on_none=lambda: None)

    def _sid_read(self, project_id: str, session_id: str | None) -> str | None:
        """Read/affect None-policy (seam 3 / D1): None → the active-loop-slot
        holder; if no holder, None (→ handle-miss → idle/no-op). NEVER mints.

        Used by run-status, stop/cancel, approve/deny, pending-approval,
        autonomy, etc. — "act on the running session when none is named."
        """
        return resolve_session_id(
            session_id,
            on_none=lambda: self.current_holder_session_id(project_id),
        )

    def _sid_inject(self, project_id: str, session_id: str | None) -> str | None:
        """Inject (start-work) None-policy (seam 3 / D1): None → the holder if
        the queue is draining, else the project's persistent chat session
        (lazy-mint via the single funnel). This is the ONLY path that may mint.
        """
        return resolve_session_id(
            session_id,
            on_none=lambda: self._resolve_inject_none(project_id),
        )

    def _resolve_inject_none(self, project_id: str) -> str:
        """Resolve inject(None): holder if the queue is actively draining,
        else the persistent chat-origin session (single mint funnel)."""
        from agent_os.queue.models import QueueRunState
        dispatcher = self._dispatchers.get(project_id)
        draining = False
        if dispatcher is not None:
            try:
                draining = dispatcher._store.load().state == QueueRunState.RUNNING
            except Exception:
                draining = False
        if draining:
            holder = self.current_holder_session_id(project_id)
            if holder is not None:
                return holder
        return self._ensure_chat_session(project_id)

    def _ensure_chat_session(self, project_id: str) -> str:
        """THE single chat-session funnel (seam 3 / D1).

        Returns the project's persistent chat session uuid: the most-recent
        chat-origin session if any exist (defensive — degrades gracefully if the
        at-most-one invariant was ever violated, e.g. by explicit "+ new
        session"), else lazy-mints exactly one via ``new_session`` (the only
        auto-mint of a chat session; "at most one" is enforced by minting only
        when zero exist).
        """
        chat_sessions = [
            s for s in self.list_sessions(project_id)
            if s.get("origin", "chat") == "chat"
        ]
        if chat_sessions:
            # Most-recent by last_activity_at (None sorts last/oldest).
            def _ts(s):
                la = s.get("last_activity_at")
                try:
                    from datetime import datetime
                    return datetime.fromisoformat(la) if la else datetime.min
                except Exception:
                    return None
            chat_sessions.sort(key=lambda s: (s.get("last_activity_at") or ""), reverse=True)
            return chat_sessions[0]["session_id"]
        # Zero chat-origin sessions → lazy-mint exactly one (the at-most-one
        # invariant: we mint only when none exist).
        return self._mint_session_uuid(project_id)

    def _mint_session_uuid(self, project_id: str) -> str:
        """Mint a fresh canonical session uuid (sync; no file, no handle).

        Shared by ``new_session`` (the explicit "+ new session" / queue path)
        and ``_ensure_chat_session`` (the lazy chat funnel) so there is a single
        uuid-minting primitive.
        """
        project = self._project_store.get_project(project_id) if self._project_store else None
        project_name = (project or {}).get("name", "")
        sanitized = _sanitize_project_name(project_name)
        return f"{sanitized}_{uuid4().hex[:8]}"

    def _broadcast(self, project_id: str, payload: dict,
                   session_id: str | None = None) -> None:
        """Wrap ``self._ws.broadcast`` to stamp an additive ``session_id`` field.

        The existing project-keyed subscription model is unchanged — every
        client subscribed to ``project_id`` still receives the payload.
        The new field lets multi-loop-aware clients filter by session
        without breaking single-loop clients (which simply ignore the
        extra field). When ``session_id`` is omitted at the call site,
        ``DEFAULT_SESSION_ID`` is stamped so consumers can rely on the
        field being present on every payload.
        """
        sid = self._resolve_session_id(session_id)
        # Don't overwrite a session_id already set by the caller (rare,
        # but cleaner than special-casing): the call site's choice wins.
        payload.setdefault("session_id", sid)
        self._ws.broadcast(project_id, payload)

    def _prevent_sleep_if_needed(self) -> None:
        """Prevent system sleep when the first agent starts."""
        if self._sleep_handle is None and self._platform_provider is not None:
            try:
                self._sleep_handle = self._platform_provider.prevent_sleep(
                    "Orbital: agent(s) running"
                )
            except Exception:
                logger.warning("Failed to prevent sleep")

    def _allow_sleep_if_idle(self) -> None:
        """Re-allow system sleep when no agents are running."""
        if self._sleep_handle is not None and self._platform_provider is not None:
            has_running = any(
                h.task is not None and not h.task.done()
                for h in self._handles.values()
            )
            if not has_running:
                try:
                    self._platform_provider.allow_sleep(self._sleep_handle)
                except Exception:
                    logger.warning("Failed to allow sleep")
                self._sleep_handle = None

    async def shutdown(self, timeout: float = 5.0) -> None:
        """Graceful shutdown: stop all agents and project dispatchers, then exit.

        No state is persisted — the session JSONLs on disk are the only source
        of truth, and the daemon never auto-resumes anything on next start."""
        logger.info("AgentManager shutdown: stopping %d agent(s)", len(self._handles))

        # Stop the idle-eviction sweep so it doesn't race the teardown below.
        if self._eviction_task is not None and not self._eviction_task.done():
            self._eviction_task.cancel()
            try:
                await self._eviction_task
            except (asyncio.CancelledError, Exception):
                pass
            self._eviction_task = None

        # Append shutdown marker to each active session
        for (pid, sid), handle in list(self._handles.items()):
            try:
                if hasattr(handle.session, 'append'):
                    handle.session.append({
                        "role": "system",
                        "type": "daemon_shutdown",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
            except Exception:
                logger.warning(
                    "Failed to append shutdown marker for project %s session %s",
                    pid, sid,
                )

        # Stop all running agents with a timeout
        stop_tasks = []
        for pid, sid in list(self._handles.keys()):
            stop_tasks.append(self.stop_agent(pid, session_id=sid))
        if stop_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*stop_tasks, return_exceptions=True),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.warning("AgentManager shutdown timed out after %.1fs", timeout)

        # Shut down all project dispatchers. They are project-scoped and outlive
        # individual agents, so the handle-based teardown above does not reach
        # them — sweep _dispatchers directly.
        for pid, dispatcher in list(self._dispatchers.items()):
            try:
                await dispatcher.shutdown()
            except Exception:
                logger.warning(
                    "dispatcher shutdown failed for %s during daemon shutdown",
                    pid, exc_info=True,
                )
        self._dispatchers.clear()
        logger.info("AgentManager shutdown complete")

    def _record_approval_decision(self, project_id: str, tool_name: str,
                                   tool_args: dict, decision: str,
                                   deny_reason: str | None = None,
                                   *, session_id: str | None = None) -> None:
        """Append approval/denial record to {workspace}/orbital/approval_history.jsonl."""
        from agent_os.agent.project_paths import ProjectPaths
        session_id = self._resolve_session_id(session_id)
        handle = self._handles.get(make_session_key(project_id, session_id))
        if handle is None:
            return
        # Derive workspace from config snapshot
        workspace = handle.config_snapshot.get('workspace')
        if not workspace:
            return
        pp = ProjectPaths(workspace)
        os.makedirs(pp.orbital_dir, exist_ok=True)
        history_file = pp.approval_history
        args_hash = hashlib.sha256(json.dumps(tool_args, sort_keys=True).encode()).hexdigest()[:12]
        record = {
            "tool": tool_name,
            "args_hash": args_hash,
            "decision": decision,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        if deny_reason:
            record["deny_reason"] = deny_reason
        try:
            with open(history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            logger.warning("Failed to write approval history for project %s", project_id)

    async def start_agent(self, project_id: str, config: AgentConfig,
                          initial_message: str | None = None,
                          trigger_source: str | None = None,
                          trigger_name: str | None = None,
                          initial_nonce: str | None = None,
                          session_id: str | None = None,
                          queue_state: str = "chat",
                          session: "Session | None" = None,
                          cold_start: bool = False,
                          cold_start_skeleton: str | None = None) -> None:
        """Wire all components and start the loop.

        ``session_id`` is the canonical session id (seam 3 / D1+D2: id == uuid
        == handle key, no "default" sentinel). Resolution:
          - a hydrated ``session`` was passed → key by ``session.session_uuid``;
          - an explicit ``session_id`` was passed → it IS the canonical id/stem;
          - neither → mint a fresh uuid (the session is identified by it).
        The pure-create branch below sets ``session_uuid = session_id`` so the
        JSONL stem, the handle key, and the stamped/holder id are ONE value
        (eliminating the dual-id split that was the seam-3 root cause).

        ``self._handles`` is keyed by ``SessionKey == (project_id, session_id)``.
        """
        if session is not None:
            session_id = session.session_uuid
        elif session_id is None:
            session_id = self._mint_session_uuid(project_id)
        sk = make_session_key(project_id, session_id)
        # Guard: check if already running
        handle = self._handles.get(sk)
        if handle is not None:
            if handle.task is not None and not handle.task.done():
                raise ValueError("Agent already running for this project")
            # Stale handle: clean up
            del self._handles[sk]

        # Cross-session slot guard: a project has at most one active-loop slot.
        # Enforced HERE (not just at the inject route) so EVERY caller is
        # covered — HTTP /agents/start, the queue dispatcher's inject_message,
        # triggers, and internal auto-starts. session_id is already resolved
        # (None → DEFAULT_SESSION_ID above), so a None-session caller is checked
        # too. See REPORT-subagent-leak-and-slot-gap.md Q4.
        holder = self.current_holder_session_id(project_id)
        if holder is not None and holder != session_id:
            raise ValueError(f"Slot held by session {holder}")

        # Guard: check platform capabilities (skip for NullProvider / no isolation)
        if self._platform_provider is not None:
            caps = self._platform_provider.get_capabilities()
            if caps.isolation_method != "none" and not caps.setup_complete:
                logger.warning("Sandbox not configured — agent will run without isolation")
            if caps.isolation_method != "none":
                self._platform_provider.grant_folder_access(config.workspace, "read_write")

        # 1. Provider (centralized API key resolution)
        api_key = resolve_api_key({"api_key": config.api_key})
        model_info = self._provider_registry.get_model_info(config.provider, config.model)
        provider = LLMProvider(
            config.model, api_key, config.base_url, sdk=config.sdk,
            max_output=model_info.max_output,
            capabilities=model_info.capabilities,
            reasoning=model_info.reasoning,
            provider=config.provider,
        )

        # 1a. Fallback providers
        fallback_providers = []
        for fb in config.llm_fallback_models:
            fb_key = fb.api_key or api_key
            fb_info = self._provider_registry.get_model_info(
                getattr(fb, 'provider', 'custom'), fb.model,
            )
            fallback_providers.append(
                LLMProvider(fb.model, fb_key, fb.base_url, sdk=fb.sdk,
                            max_output=fb_info.max_output,
                            capabilities=fb_info.capabilities,
                            reasoning=fb_info.reasoning,
                            provider=getattr(fb, 'provider', 'custom'))
            )

        # 1b. Utility provider
        if config.utility_model:
            utility_info = self._provider_registry.get_model_info(config.provider, config.utility_model)
            utility_provider = LLMProvider(
                config.utility_model, config.api_key, config.base_url, sdk=config.sdk,
                reasoning=utility_info.reasoning,
                provider=config.provider,
            )
        else:
            utility_provider = provider

        # 2. Tool registry
        registry = ToolRegistry(user_credential_store=self._user_credential_store)
        self._register_tools(registry, config, project_id,
                             vision_enabled=model_info.capabilities.vision,
                             session_id=session_id)

        # 3. Prompt builder
        prompt_builder = PromptBuilder(workspace=config.workspace)

        # 4. Build prompt context
        # Prefer enabled_sub_agents (new), fall back to enabled_agents (legacy),
        # else auto-detect installed sub-agents — then subtract the project's
        # disabled_sub_agents denylist so a disabled sub-agent never appears in
        # the orchestrator's prompt (the auto-detect branch previously ignored
        # the denylist on the /agents/start path).
        sub_agent_slugs = resolve_visible_sub_agent_slugs(
            enabled_sub_agents=config.enabled_sub_agents,
            disabled_sub_agents=config.disabled_sub_agents,
            setup_engine=self._setup_engine,
            enabled_agents_legacy=config.enabled_agents,
        )
        enabled_agents_detail = []
        for slug in sub_agent_slugs:
            if self._registry is not None:
                manifest = self._registry.get(slug)
                if manifest:
                    enabled_agents_detail.append({
                        "handle": slug,
                        "display_name": manifest.name,
                        "type": manifest.runtime.adapter,
                        "skills": manifest.capabilities.skills,
                        "routing_hint": manifest.capabilities.routing_hint,
                    })
                    continue
            # Fallback: no registry or slug not found in registry
            enabled_agents_detail.append({
                "handle": slug, "display_name": slug, "type": "cli",
            })
        logger.info("Enabled sub-agents for project %s: %s", project_id, [a["handle"] for a in enabled_agents_detail])
        base_prompt_context = PromptContext(
            workspace=config.workspace,
            model=config.model,
            autonomy=config.autonomy,
            enabled_agents=enabled_agents_detail,
            tool_names=registry.tool_names(),
            os_type=detect_os(),
            datetime_now=datetime.now().isoformat(),
            project_name=config.project_name,
            project_instructions=config.project_instructions,
            is_scratch=config.is_scratch,
            agent_name=config.agent_name,
            global_preferences_path=config.global_preferences_path,
            trigger_source=trigger_source,
            trigger_name=trigger_name,
            vision_enabled=model_info.capabilities.vision,
            project_id=project_id,
            cold_start=cold_start,
        )

        # 4b. Reconcile default skills for legacy projects that never had them
        # installed (or the first-install failed). The installer short-circuits
        # on the persistent ``default_skills_reconciled`` flag, so the common
        # path is a dict lookup and no disk I/O. Failures must not block agent
        # start — the flag stays False and we will retry on the next start.
        if self._project_store.get_project(project_id) is not None:
            try:
                install_default_skills(self._project_store, project_id)
            except Exception:
                logger.error(
                    "default skills reconcile failed for project %s; continuing agent start",
                    project_id, exc_info=True,
                )

        # 5. Session.
        # ``session_uuid`` is the Format-2 JSONL filename stem (sanitized
        # project name + uuid4 hex). ``session_id`` is the Format-1 user-facing
        # chat id (defaulted to DEFAULT_SESSION_ID above when the caller did
        # not provide one). Downstream WS broadcasts and callbacks must use
        # the F1 chat id; only filename derivation and idempotency keys use F2.
        # Pure-create unless a hydrated session was handed in (continue an
        # existing on-disk conversation — hydrate-on-inject). When hydrated,
        # the session already carries its original F1/F2 and full history.
        if session is None:
            # Seam 3 / D1+D2: the canonical id IS the storage stem. session_id
            # was resolved above (explicit, or freshly minted for None), so the
            # JSONL stem == session_id == handle key — one value, no dual-id
            # split. Origin derived from the queue context — the QueueDispatcher
            # injects with queue_state="running", everything else is user chat.
            session_uuid = session_id
            session = Session.new(
                session_uuid, config.workspace,
                session_id=session_id,
                provider=config.provider,
                model=config.model,
                sdk=config.sdk,
                fallback_models=[fb.model for fb in config.llm_fallback_models],
                origin=(
                    "cold_start" if cold_start
                    else ("queue" if queue_state == "running" else "chat")
                ),
            )

        # 6-7. Observers
        session.on_append = self._on_message(project_id, session_id)
        session.on_stream = self._on_stream(project_id, session_id)

        # 8. Workspace files
        workspace_files = WorkspaceFileManager(config.workspace)
        workspace_files.ensure_dir()

        # 9. Context manager
        # Bind session_id into the lambda explicitly so the captured value
        # tracks the session this loop owns rather than a late-binding
        # closure on a mutable name.
        _sid_for_provider = session_id
        context_manager = ContextManager(
            session, prompt_builder, base_prompt_context,
            model_context_limit=model_info.context_window,
            workspace_files=workspace_files,
            sub_agent_provider=lambda: self._sub_agent_manager.list_active(
                project_id, session_id=_sid_for_provider,
            ),
        )

        # 10. Interceptor
        interceptor = AutonomyInterceptor(
            config.autonomy, self._ws, project_id,
            user_credential_store=self._user_credential_store,
            session_id=session_id,
        )

        # 11. Session-end callback
        async def session_end_callback():
            await run_session_end_routine(
                session=session,
                provider=provider,
                workspace_files=workspace_files,
                utility_provider=utility_provider,
                session_uuid=session.session_uuid,
                project_id=project_id,
            )

        # 11a. Periodic state-refresh callback (turn-count / agent-decided /
        # token-pressure triggers). Broadcasts state_refresh.lifecycle events
        # so the frontend can show an inline status indicator.
        async def session_end_refresh_callback(trigger_name: str) -> None:
            from datetime import datetime, timezone
            ts = datetime.now(timezone.utc).isoformat()
            self._broadcast(project_id, {
                "type": "state_refresh.lifecycle",
                "project_id": project_id,
                "status": "in_progress",
                "trigger": trigger_name,
                "timestamp": ts,
            }, session_id=session_id)
            try:
                await run_session_end_routine(
                    session=session,
                    provider=provider,
                    workspace_files=workspace_files,
                    utility_provider=utility_provider,
                    session_uuid=session.session_uuid,
                    bypass_idempotency=True,
                    project_id=project_id,
                )
                self._broadcast(project_id, {
                    "type": "state_refresh.lifecycle",
                    "project_id": project_id,
                    "status": "done",
                    "trigger": trigger_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }, session_id=session_id)
                logger.info(
                    "State refresh complete (project=%s trigger=%s)",
                    project_id, trigger_name,
                )
            except asyncio.TimeoutError:
                self._broadcast(project_id, {
                    "type": "state_refresh.lifecycle",
                    "project_id": project_id,
                    "status": "skipped",
                    "trigger": trigger_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }, session_id=session_id)
                logger.warning(
                    "State refresh timed out (skipped) (project=%s trigger=%s)",
                    project_id, trigger_name,
                )
                raise
            except Exception:
                self._broadcast(project_id, {
                    "type": "state_refresh.lifecycle",
                    "project_id": project_id,
                    "status": "failed",
                    "trigger": trigger_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }, session_id=session_id)
                logger.exception(
                    "State refresh failed (project=%s trigger=%s)",
                    project_id, trigger_name,
                )
                raise

        # 11b. Budget persistence callback
        from agent_os.agent.pricing import get_cost_rates, budget_usd_to_token_budget
        cost_per_1k_input, cost_per_1k_output = get_cost_rates(
            config.model, config.provider,
        )
        persisted_spend = (
            self._project_store.get_project(project_id) or {}
        ).get("runtime", {}).get("budget_spent_usd", 0.0)

        # Derive token budget from dollar budget when set
        effective_token_budget = budget_usd_to_token_budget(
            config.budget_limit_usd, cost_per_1k_input, cost_per_1k_output,
        )

        def on_cost_update(delta_usd, total_spent_usd):
            self._project_store.update_runtime(
                project_id, {"budget_spent_usd": round(total_spent_usd, 6)},
            )

        # 12. Loop
        loop = AgentLoop(
            session, provider, registry, context_manager, interceptor,
            utility_provider=utility_provider,
            fallback_providers=fallback_providers,
            max_iterations=config.max_iterations,
            token_budget=effective_token_budget,
            budget_limit_usd=config.budget_limit_usd,
            budget_action=config.budget_action,
            budget_spent_usd=persisted_spend,
            on_cost_update=on_cost_update,
            cost_per_1k_input=cost_per_1k_input,
            cost_per_1k_output=cost_per_1k_output,
            on_session_end=session_end_callback,
            on_session_end_refresh=session_end_refresh_callback,
        )

        # 12-queue. When the dispatcher starts a session to run a queue item it
        # passes queue_state="running" so the loop enforces the queue
        # completion contract (an approval-required tool call becomes a block
        # signal) from its very first turn. Set before the task is created so
        # there is no race with the first iteration. Default "chat" leaves
        # interactive (user) sessions unchanged — they reach this via inject.
        if hasattr(loop, "_queue_state"):
            loop._queue_state = queue_state

        # 12a. Register checkpoint_state tool now that the loop exists.
        # The tool calls loop.trigger_checkpoint() (agent-decided trigger).
        try:
            from agent_os.agent.tools.checkpoint_state import CheckpointStateTool

            async def on_agent_checkpoint():
                await loop.trigger_checkpoint()

            registry.register(CheckpointStateTool(on_checkpoint=on_agent_checkpoint))
            # Keep tool_names in PromptContext in sync
            base_prompt_context.tool_names.append("checkpoint_state")
        except Exception:
            logger.warning(
                "checkpoint_state tool registration failed; agent-decided trigger disabled",
                exc_info=True,
            )

        # 13. Store handle
        project_handle = ProjectHandle(
            session=session,
            loop=loop,
            provider=provider,
            registry=registry,
            context_manager=context_manager,
            interceptor=interceptor,
            task=None,
            trigger_source=trigger_source,
            config_snapshot={
                "workspace": config.workspace,
                "model": config.model,
                "autonomy": config.autonomy.value if hasattr(config.autonomy, 'value') else str(config.autonomy),
                "provider": config.provider,
                "sdk": config.sdk,
                "fallback_models": [fb.model for fb in config.llm_fallback_models],
            },
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self._handles[sk] = project_handle

        # 10b. Cold-start: inject the Stage-1 skeleton inventory as system
        # context so the agent plans Stage-2 reads against measured sizes.
        # Persisted in the session JSONL; it becomes the loop's first context.
        if cold_start and cold_start_skeleton:
            session.append_system(cold_start_skeleton)

        # 11. Start loop task
        task = asyncio.create_task(loop.run(initial_message, initial_nonce=initial_nonce))
        task.add_done_callback(self._on_loop_done(project_id, session_id=session_id))
        project_handle.task = task

        # 14. Broadcast running
        status_event = {
            "type": "agent.status",
            "project_id": project_id,
            "status": "running",
            "source": "management",
        }
        if trigger_source:
            status_event["trigger_source"] = trigger_source
        self._broadcast(project_id, status_event, session_id=session_id)

        # 15. Prevent system sleep while an agent is active.
        self._prevent_sleep_if_needed()

        # 16. Ensure the project's queue dispatcher exists (idempotent). The
        # dispatcher is project-scoped and normally created at daemon startup /
        # project creation; this call is a harmless no-op when it already
        # exists, and a safety net for agents started before it was created.
        await self._ensure_dispatcher(project_id, config.workspace)

    def _register_tools(self, registry: ToolRegistry, config: AgentConfig,
                        project_id: str = "", vision_enabled: bool = False,
                        *, session_id: str | None = None) -> None:
        """Register all tools. Imports are deferred to avoid circular deps at module level."""
        # Queue signal tools — always registered, used by the queue dispatcher
        # to detect attempt completion. Detection happens at response-parsing
        # time in AgentLoop, before tool dispatch, so these classes' execute()
        # serves only as a defensive fallback path.
        try:
            from agent_os.agent.tools.queue_signals import (
                MarkTaskBlockedTool,
                MarkTaskCompleteTool,
            )
            registry.register(MarkTaskCompleteTool())
            registry.register(MarkTaskBlockedTool())
        except ImportError:
            logger.warning("queue signal tools failed to register", exc_info=True)
        try:
            from agent_os.agent.tools.read import ReadTool
            registry.register(ReadTool(workspace=config.workspace))
        except ImportError:
            pass
        try:
            from agent_os.agent.tools.write import WriteTool
            registry.register(WriteTool(workspace=config.workspace))
        except ImportError:
            pass
        try:
            from agent_os.agent.tools.edit import EditTool
            registry.register(EditTool(workspace=config.workspace))
        except ImportError:
            pass
        try:
            from agent_os.agent.tools.glob_tool import GlobTool
            registry.register(GlobTool(workspace=config.workspace))
        except ImportError:
            pass
        try:
            from agent_os.agent.tools.grep_tool import GrepTool
            registry.register(GrepTool(workspace=config.workspace))
        except ImportError:
            pass
        try:
            from agent_os.agent.tools.shell import ShellTool
            registry.register(ShellTool(
                workspace=config.workspace,
                os_type=detect_os(),
                platform_provider=self._platform_provider,
                project_id=project_id,
            ))
        except ImportError:
            pass
        try:
            from agent_os.agent.tools.request_access import RequestAccessTool
            registry.register(RequestAccessTool())
        except ImportError:
            pass
        try:
            from agent_os.agent.tools.agent_message import AgentMessageTool
            registry.register(AgentMessageTool(sub_agent_manager=self._sub_agent_manager, project_id=project_id, depth=0, session_id=session_id))
        except ImportError:
            pass
        try:
            from agent_os.agent.tools.browser import BrowserTool
            if self._browser_manager is not None:
                registry.register(BrowserTool(
                    browser_manager=self._browser_manager,
                    project_id=project_id,
                    workspace=config.workspace,
                    autonomy_preset=config.autonomy.value if hasattr(config.autonomy, 'value') else str(config.autonomy),
                    user_credential_store=self._user_credential_store,
                    vision_enabled=vision_enabled,
                ))
        except ImportError:
            logger.warning("BrowserTool not available (playwright not installed)")
        try:
            from agent_os.agent.tools.request_credential import RequestCredentialTool
            if self._user_credential_store is not None:
                registry.register(RequestCredentialTool(
                    credential_store=self._user_credential_store,
                ))
        except ImportError:
            pass
        try:
            from agent_os.agent.tools.notify import NotifyTool
            ws = getattr(self, '_ws', None)
            if ws is not None:
                registry.register(NotifyTool(
                    ws_manager=ws,
                    project_id=project_id,
                    session_id=session_id,
                ))
        except ImportError:
            pass
        try:
            from agent_os.agent.tools.triggers import (
                CreateTriggerTool, ListTriggersTool,
                UpdateTriggerTool, DeleteTriggerTool,
            )
            registry.register(CreateTriggerTool(
                project_id=project_id,
                project_store=self._project_store,
                trigger_manager=self._trigger_manager,
            ))
            registry.register(ListTriggersTool(
                project_id=project_id,
                project_store=self._project_store,
            ))
            registry.register(UpdateTriggerTool(
                project_id=project_id,
                project_store=self._project_store,
                trigger_manager=self._trigger_manager,
            ))
            registry.register(DeleteTriggerTool(
                project_id=project_id,
                project_store=self._project_store,
                trigger_manager=self._trigger_manager,
            ))
        except ImportError:
            pass

    def has_handle(self, project_id: str) -> bool:
        """Return True if an agent handle exists for the project.

        Distinct from ``is_running``: a handle may exist with a finished task
        (loop idle but session alive). Callers that need to know whether the
        agent is actively executing should use ``is_running``; callers that
        need to know whether the agent has *ever* been started in this
        process (i.e. whether the dispatcher and other lifecycle components
        exist) should use this.

        F7 note: handles are keyed by ``(project_id, session_id)`` since
        the multi-session foundation (#22). For queue auto-start the check
        is project-level: any handle keyed under ``project_id`` counts.
        """
        return any(pid == project_id for (pid, _sid) in self._handles.keys())

    def _build_agent_config_from_project(self, project_id: str) -> AgentConfig:
        """Construct an AgentConfig for ``project_id`` from the project store.

        Used by auto-start paths (inject_message Case 3 and the queue items
        route) so the same derivation logic — autonomy resolution, sub-agent
        availability, API key/model/base_url fallback through credential and
        settings stores — runs in one place.

        Raises:
            KeyError: if no project exists for ``project_id``.
        """
        from agent_os.agent.prompt_builder import Autonomy

        project = self._project_store.get_project(project_id)
        if project is None:
            raise KeyError(f"No project found for '{project_id}'")

        autonomy_str = project.get("autonomy", "hands_off")
        try:
            autonomy = Autonomy(autonomy_str)
        except ValueError:
            autonomy = Autonomy.HANDS_OFF

        # Available sub-agents = installed minus the project's
        # disabled_sub_agents denylist. Legacy ``enabled_sub_agents`` is
        # informational-only in v1.
        disabled = set(project.get("disabled_sub_agents", []) or [])
        if self._setup_engine is not None:
            available = self._setup_engine.check_all()
            enabled_sub_agents = [
                a.slug for a in available
                if a.installed and a.slug != "built-in"
                and a.slug not in disabled
            ]
        else:
            enabled_sub_agents = [
                s for s in (project.get("enabled_sub_agents") or [])
                if s not in disabled
            ]

        # Global settings provide fallbacks for missing project-level LLM
        # config; the credential store provides the live global API key.
        global_settings = self._settings_store.get() if self._settings_store else None
        cred_key = self._credential_store.get_api_key() if self._credential_store else None
        api_key = (project.get("api_key")
                   or cred_key
                   or (global_settings.llm.api_key if global_settings else None)
                   or "")
        base_url = project.get("base_url") or (global_settings.llm.base_url if global_settings else None)
        model = project.get("model") or (global_settings.llm.model if global_settings else None) or ""

        # Provider must track the model. When the project pins its own model it
        # keeps its own provider (a self-hosted/custom setup); when it leaves
        # model empty and inherits the global model, it must inherit the global
        # provider too — otherwise a project left at the default
        # provider="custom" runs as custom+<global model>, and registry lookups
        # (get_model_info(provider, model)) miss the real entry, silently
        # bypassing model-specific behavior such as MiniMax inline-<think>
        # reasoning separation and capability/pricing metadata.
        if project.get("model"):
            provider = project.get("provider") or "custom"
        else:
            provider = (
                (global_settings.llm.provider if global_settings else None)
                or project.get("provider")
                or "custom"
            )

        return AgentConfig(
            workspace=project["workspace"],
            model=model,
            api_key=api_key,
            base_url=base_url,
            autonomy=autonomy,
            sdk=project.get("sdk", "openai"),
            provider=provider,
            project_name=project.get("name", ""),
            project_instructions=project.get("instructions", ""),
            enabled_sub_agents=enabled_sub_agents or [],
            disabled_sub_agents=list(disabled),
            budget_limit_usd=project.get("budget_limit_usd"),
            budget_action=project.get("budget_action", "ask"),
        )

    async def ensure_agent_started(self, project_id: str) -> bool:
        """Start the agent for ``project_id`` if it is not already running.

        Used by the queue items route to auto-start an agent when a queue
        item is added to a project with no live agent. Unlike
        ``inject_message`` Case 3, this does NOT pass an initial_message —
        the queue dispatcher's ``_run`` loop picks up the new item via the
        store and wraps it with a ``[QUEUE ITEM]`` header before injection.

        Returns:
            True if a new agent was started, False if a handle already
            existed (no-op in that case).

        Raises:
            KeyError: if no project exists for ``project_id``.
        """
        if self.has_handle(project_id):
            return False
        config = self._build_agent_config_from_project(project_id)
        await self.start_agent(project_id, config)
        return True

    async def inject_system_message(self, project_id: str, content: str,
                                    *, session_id: str | None = None) -> str:
        """Inject a system message into the management agent's session.

        Used by the lifecycle observer for sub-agent state notifications.
        If the management agent is idle, appends directly and starts a new loop.
        If the loop is running, defers the message for safe insertion after the
        current tool batch completes (avoids breaking assistant→tool sequences).

        ``session_id`` selects which chat session within the project to
        target; defaults to the single-loop default session.
        """
        session_id = self._resolve_session_id(session_id)
        handle = self._handles.get(make_session_key(project_id, session_id))
        if handle is None:
            # Piece 3 Part C (wake-if-awaited): the handle being gone (e.g.
            # idle-evicted while a sub-agent straggled to completion) must
            # NOT silently drop the event — that silence was the live bug's
            # second half. Hydrate the on-disk session, append the event
            # FIRST (durable even if the wake fails), then start the loop so
            # an awaiting management session actually processes it.
            loaded = self._load_session_from_disk(project_id, session_id)
            if loaded is None:
                logger.warning(
                    "inject_system_message(%s/%s): no live handle and no "
                    "on-disk session — terminal event could not be "
                    "delivered: %s",
                    project_id, session_id, content[:200],
                )
                return "no_session"
            logger.info(
                "inject_system_message(%s): hydrating session uuid %s from "
                "disk for terminal sub-agent event",
                project_id, loaded.session_uuid,
            )
            loaded.append({
                "role": "system",
                "content": content,
                "source": "daemon",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            try:
                config = self._build_agent_config_from_project(project_id)
                await self.start_agent(
                    project_id, config, initial_message=None,
                    session_id=loaded.session_uuid, session=loaded,
                )
                return "delivered"
            except Exception:
                logger.warning(
                    "inject_system_message(%s/%s): wake failed; event is "
                    "persisted on disk", project_id, session_id,
                    exc_info=True,
                )
                return "persisted"

        # If loop is idle, append directly (safe — no pending tool calls) and wake
        if handle.task is None or handle.task.done():
            handle.session.append({
                "role": "system",
                "content": content,
                "source": "daemon",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            await self._start_loop(project_id, session_id=session_id)
            return "delivered"

        # Loop is running — defer for safe insertion after tool batch
        handle.session.defer_message(content, role="system", source="daemon")
        return "deferred"

    def _read_session_f1(self, filepath: str) -> str | None:
        """Read a session JSONL's original F1 ``session_id`` from its first
        record. The ``session_start`` meta carries it; non-meta records stamp
        it too — so the first record with a ``session_id`` field is the F1."""
        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        rec = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    sid = rec.get("session_id")
                    if sid:
                        return sid
        except OSError:
            return None
        return None

    def _load_session_from_disk(self, project_id: str, identifier: str):
        """Resolve an F1 or F2 identifier to an on-disk session and load it.

        Returns a hydrated ``Session`` (history intact, original F1 preserved
        on ``session_id``, F2 on ``session_uuid``) or ``None`` if no JSONL
        matches. Single resolver used by hydrate-on-inject so a disk-only
        session is *continued* rather than forked into a fresh empty one (the
        gap the cold-resume experiment surfaced).
        """
        project = self._project_store.get_project(project_id) if self._project_store else None
        workspace = (project or {}).get("workspace", "") if project else ""
        if not workspace:
            return None
        sessions_dir = ProjectPaths(workspace).sessions_dir
        if not os.path.isdir(sessions_dir):
            return None
        # Fast path: identifier is the F2 stem (the sidebar's address for a
        # disk-only session) → the file is ``{identifier}.jsonl``.
        target = os.path.join(sessions_dir, f"{identifier}.jsonl")
        if not os.path.isfile(target):
            # Slow path: identifier is an F1 chat id → scan for the JSONL whose
            # records carry that F1. F1 is not unique across rotated logs
            # (legacy "default"), so prefer the most recently modified match.
            target = None
            best_mtime = -1.0
            for fname in os.listdir(sessions_dir):
                if not fname.endswith(".jsonl"):
                    continue
                fpath = os.path.join(sessions_dir, fname)
                if self._read_session_f1(fpath) == identifier:
                    m = os.path.getmtime(fpath)
                    if m > best_mtime:
                        best_mtime = m
                        target = fpath
            if target is None:
                return None
        session = Session.load(target)
        # Session.load sets session_id to the filename stem (F2); recover the
        # original F1 from the records so identity survives hydration.
        f1 = self._read_session_f1(target)
        if f1:
            session.session_id = f1
        return session

    async def inject_message(self, project_id: str, content: str, *,
                             nonce: str | None = None,
                             session_id: str | None = None,
                             queue_state: str = "chat") -> "str | dict":
        """Inject a message. Four cases:
        1. Loop running -> queue (returns "queued")
        1b. Loop paused for approval -> auto-deny pending, deliver, resume
            (returns dict with status="delivered", approval_dismissed=True,
            dismissed_tool_call_id=<id>)
        2. Loop idle + session alive -> hot resume (returns "delivered")
        3. No session -> auto-start agent with message (returns "started")

        ``session_id`` is the optional Format-1 user-facing chat id; it
        selects which chat session within the project to target and defaults
        to the single-loop default session (``DEFAULT_SESSION_ID``).

        Lazy session minting (Phase 3c multi-loop): when ``session_id``
        names a chat session that does not yet have a handle but the
        project itself exists in the project_store, this method falls
        through Case 3 and ``start_agent`` mints a fresh handle under
        ``(project_id, session_id)``. The newly minted session shares
        the project's workspace and config; it does NOT inherit
        conversation history from any other session in the same project.
        """
        # Inject (start-work) None-policy: holder if the queue is draining,
        # else the project's persistent chat session (lazy-mint funnel).
        session_id = self._sid_inject(project_id, session_id)
        sk = make_session_key(project_id, session_id)
        # A user-initiated message clears any recorded terminal event for
        # this session — the user is taking action, so the error/stopped
        # indicator should no longer light up in the sidebar. Performed
        # before any case branch so it applies uniformly to Case 1 (queue),
        # Case 1b (approval-dismiss), Case 2 (hot-resume), Case 3 (start).
        self._clear_last_terminal_event(project_id, session_id)
        handle = self._handles.get(sk)
        if handle is None:
            config = self._build_agent_config_from_project(project_id)
            # Hydrate-on-inject: if a JSONL exists for this identifier (F1 or
            # F2/uuid), load it and CONTINUE the conversation rather than
            # forking a fresh empty session. This is the chat flow's only load
            # path — the cold-resume experiment proved Case 3 alone discarded
            # all history.
            #
            # Seam 3 / Phase 1: adopt the session's *uuid* as the routing
            # identity, NOT the file's meta F1. The frontend addresses a
            # disk-only session by its session_uuid (the JSONL filename stem),
            # so keying the handle / holder / stamped events under the uuid is
            # what makes viewed == holder (the render gate `viewingHolder`).
            # The meta F1 is still preserved on the loaded Session object
            # (loaded.session_id) for display/back-compat; it is simply no
            # longer the routing key. See REPORT-streaming-status-frontend.md.
            loaded = self._load_session_from_disk(project_id, session_id)
            if loaded is not None:
                logger.info(
                    "inject_message(%s): hydrating session uuid %s (meta F1 %s) from disk",
                    project_id, loaded.session_uuid, loaded.session_id,
                )
                await self.start_agent(project_id, config, initial_message=content,
                                      initial_nonce=nonce, session_id=loaded.session_uuid,
                                      queue_state=queue_state, session=loaded)
                return "started"
            # Case 3: no session on disk — auto-start a fresh agent.
            logger.info("inject_message(%s): no handle, auto-starting fresh agent", project_id)
            await self.start_agent(project_id, config, initial_message=content,
                                  initial_nonce=nonce, session_id=session_id,
                                  queue_state=queue_state)
            return "started"

        # User interaction — defer idle eviction. (Cold-start paths above
        # create the handle fresh with last_activity=now; this covers the
        # queue / approval-dismiss / hot-resume paths on an existing handle.)
        handle.last_activity = time.time()

        # New user message resets approve-all bypass (new task context)
        if handle.interceptor is not None:
            handle.interceptor.deactivate_bypass_all()

        if handle.task is not None and not handle.task.done():
            # Case 1: loop running — queue for next iteration
            logger.info("inject_message(%s): loop running, queuing message", project_id)
            handle.session.queue_message(content, nonce=nonce)
            return "queued"

        # Case 1b: loop done but paused for approval — auto-deny the
        # pending approval, deliver the new user message, and restart the
        # loop. Queuing here previously caused silent message loss because
        # the loop had already exited and nothing would drain the queue.
        if handle.session._paused_for_approval:
            logger.info(
                "inject_message(%s): paused for approval, auto-denying and delivering",
                project_id,
            )

            # 1. Find the pending approval (typically only one).
            dismissed_tc_id: str | None = None
            dismissed_tool_name: str | None = None
            if handle.interceptor is not None:
                for tc_id, data in handle.interceptor._pending_approvals.items():
                    dismissed_tc_id = tc_id
                    dismissed_tool_name = data.get("tool_name", "unknown")
                    break

            if dismissed_tc_id and handle.interceptor is not None:
                # 2. Write denial tool_result so the session stays
                #    structurally valid (every tool_call has a result).
                if not handle.session.has_result_for(dismissed_tc_id):
                    handle.session.append_tool_result(
                        dismissed_tc_id,
                        "DISMISSED: User sent a new message, cancelling this approval request.",
                    )

                # 3. Record in approval history.
                pending = handle.interceptor.get_pending(dismissed_tc_id)
                if pending:
                    self._record_approval_decision(
                        project_id,
                        pending["tool_name"],
                        pending.get("tool_args", {}),
                        "denied",
                        deny_reason="User sent a new message while approval was pending",
                        session_id=session_id,
                    )

                # 4. Clean up interceptor state.
                handle.interceptor.remove_pending(dismissed_tc_id)

            # 5. Resume session (clears _paused flag; loop.run() clears
            #    _paused_for_approval on entry).
            handle.session.resume()

            # 6. Append a visible system message so the user sees what
            #    happened — the approval card transition alone is not
            #    enough context.
            handle.session.append({
                "role": "system",
                "content": (
                    f"[Pending approval for {dismissed_tool_name or 'tool'} "
                    f"was dismissed because you sent a new message. "
                    f"The agent will continue without it.]"
                ),
                "source": "daemon",
            })

            # 7. Drain any previously queued messages so they don't get
            #    lost between the dismissal and the new message.
            stale = handle.session.pop_queued_messages()
            for s_item in stale:
                if isinstance(s_item, tuple):
                    s_content, s_nonce = s_item
                else:
                    s_content, s_nonce = s_item, None
                s_msg: dict = {"role": "user", "content": s_content, "source": "user"}
                if s_nonce:
                    s_msg["nonce"] = s_nonce
                handle.session.append(s_msg)

            # 8. Append the new user message.
            user_msg: dict = {"role": "user", "content": content, "source": "user"}
            if nonce:
                user_msg["nonce"] = nonce
            handle.session.append(user_msg)

            # 9. Broadcast approval resolved so the approval card can show
            #    "Denied" via the normal WS path.
            if dismissed_tc_id:
                self._broadcast(project_id, {
                    "type": "approval.resolved",
                    "project_id": project_id,
                    "tool_call_id": dismissed_tc_id,
                    "resolution": "denied",
                }, session_id=session_id)

            # 10. Restart the loop to process the new user message.
            await self._start_loop(project_id, session_id=session_id)
            return {
                "status": "delivered",
                "approval_dismissed": True,
                "dismissed_tool_call_id": dismissed_tc_id,
            }

        # Case 2: loop idle — append message and hot-resume
        if handle is not None and handle.session.is_stopped():
            del self._handles[sk]
            handle = None

        if handle is None:
            # Auto-start a fresh agent (stale stopped session was cleaned up).
            # Re-derive config from the project store via the shared helper —
            # this path is reached when the original handle existed but the
            # session was stopped, so we need a fresh AgentConfig.
            config = self._build_agent_config_from_project(project_id)
            await self.start_agent(project_id, config, initial_message=content,
                                  initial_nonce=nonce, session_id=session_id,
                                  queue_state=queue_state)
            return "started"

        # WAIT-STATE QUEUE: if the management agent is in the "waiting" state —
        # it dispatched a sub-agent, yielded its turn, and the idle-poll is
        # alive while the sub-agent works — a new user message must NOT start a
        # loop. The agent is busy; on_completed owns the restart and will
        # surface this message when the sub-agent finishes. Append it and
        # return "queued". See TASK-yield-turn-dispatch-and-push-coordination.md.
        poll_task = self._idle_poll_tasks.get(make_session_key(project_id, session_id))
        if poll_task is not None and not poll_task.done():
            user_msg = {
                "role": "user",
                "content": content,
                "source": "user",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if nonce:
                user_msg["nonce"] = nonce
            handle.session.append(user_msg)
            self._broadcast(project_id, {
                "type": "chat.user_message",
                "project_id": project_id,
                "content": content,
            }, session_id=session_id)
            logger.info(
                "inject_message(%s): queued user message during waiting state",
                project_id,
            )
            return {"status": "queued", "session_id": session_id}

        if handle.task is not None and handle.task.done():
            try:
                exc = handle.task.exception()
            except asyncio.CancelledError:
                exc = None
            if exc:
                logger.warning(
                    "inject_message(%s): previous loop had error: %s (appending message anyway)",
                    project_id, exc,
                )

        logger.info("inject_message(%s): loop idle, delivering message and resuming", project_id)
        stale = handle.session.pop_queued_messages()
        for s_item in stale:
            if isinstance(s_item, tuple):
                s_content, s_nonce = s_item
            else:
                s_content, s_nonce = s_item, None
            s_msg: dict = {"role": "user", "content": s_content, "source": "user"}
            if s_nonce:
                s_msg["nonce"] = s_nonce
            handle.session.append(s_msg)
        user_msg: dict = {"role": "user", "content": content, "source": "user"}
        if nonce:
            user_msg["nonce"] = nonce
        handle.session.append(user_msg)
        await self._start_loop(project_id, session_id=session_id)
        return "delivered"

    def is_running(self, project_id: str, *,
                   session_id: str | None = None) -> bool:
        """Return True if an agent loop is actively running for this session.

        ``session_id`` selects the chat session; defaults to the
        single-loop default session.
        """
        session_id = self._sid_read(project_id, session_id)
        handle = self._handles.get(make_session_key(project_id, session_id))
        if handle is None:
            return False
        return handle.task is not None and not handle.task.done()

    def current_holder_session_id(self, project_id: str) -> str | None:
        """Return the F1 ``session_id`` that currently holds the project's
        active-loop slot, or None if no session in the project is running.

        Per ACTIVE-session-and-queue-model.md §3, a project has exactly one
        active-loop slot. This accessor surfaces "who holds it right now" so
        the inject route can reject (with 202) when a different session tries
        to start a turn while another session is still mid-turn.

        A session holds the slot when ANY of these is true:

        1. ``handle.task`` is alive and not done — the main loop is
           streaming, running tools, or between iterations.
        2. ``session._paused_for_approval`` is True — the loop has exited
           at an approval boundary but the session is logically still
           "in" the turn. A new turn from another session would clash
           with the pending approval decision.
        3. ``_idle_poll_tasks[(project_id, session_id)]`` is alive — the
           main loop has exited but sub-agents spawned during the turn are
           still working (the ``waiting`` state). A concurrent turn from
           another session would interleave workspace writes with the
           still-running sub-agents.

        Condition 1 covers ``running``. Condition 2 covers
        ``pending_approval``. Condition 3 covers ``waiting``. Together
        they enforce the single-loop-per-project safety claim of §3
        even past the bare ``task.done()`` boundary.

        After PR #22's multi-session foundation ``self._handles`` is keyed
        by ``SessionKey == (project_id, session_id)``. We scan all keys
        for this ``project_id``; single-loop discipline means at most one
        handle satisfies any of the conditions at a time.
        """
        for (pid, sid), handle in self._handles.items():
            if pid != project_id:
                continue
            # Condition 1: main loop active.
            if handle.task is not None and not handle.task.done():
                return sid
            # Condition 2: paused for approval (loop exited at approval gate).
            if handle.session._paused_for_approval:
                return sid
            # Condition 3: sub-agents still working (waiting state).
            poll_task = self._idle_poll_tasks.get(make_session_key(pid, sid))
            if poll_task is not None and not poll_task.done():
                return sid
        return None

    # ── last_terminal_event ────────────────────────────────────────────
    #
    # Per-session record of the most recent terminal event (error /
    # stopped / new_session). Frontend reads via `list_sessions` so an
    # errored session shows a warning glyph across WS-reconnect and
    # page-reload boundaries.
    #
    # Lifecycle: written by `_set_last_terminal_event` at every
    # terminal-event broadcast site; cleared by
    # `_clear_last_terminal_event` on the next non-terminal transition
    # (a successful inject).

    def _set_last_terminal_event(self, project_id: str, session_id: str,
                                 event_type: str,
                                 details: str | None = None) -> None:
        """Record a terminal event for a session. Idempotent (overwrites)."""
        from datetime import datetime, timezone
        self._last_terminal_events[make_session_key(project_id, session_id)] = {
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details,
        }

    def _clear_last_terminal_event(self, project_id: str,
                                   session_id: str) -> None:
        """Remove the last_terminal_event for a session (no-op if absent)."""
        self._last_terminal_events.pop(
            make_session_key(project_id, session_id), None,
        )

    def get_last_terminal_event(self, project_id: str, *,
                                 session_id: str | None = None) -> dict | None:
        """Return the recorded terminal event, or None if none recorded."""
        session_id = self._sid_read(project_id, session_id)
        return self._last_terminal_events.get(
            make_session_key(project_id, session_id),
        )

    def _broadcast_new_session_terminal_event(self, project_id: str, *,
                                                session_id: str | None = None) -> None:
        """Test helper: record a `new_session` terminal event without
        invoking the full `new_session()` rotation path. Production code
        records this inside `new_session()` directly; this accessor exists
        so unit tests can target the recording behavior in isolation.
        """
        session_id = self._resolve_session_id(session_id)
        self._set_last_terminal_event(
            project_id, session_id, "new_session",
        )

    def get_run_status(self, project_id: str, *,
                       session_id: str | None = None) -> str:
        """Return the current runtime status for a session.

        Returns one of: 'running', 'pending_approval', 'waiting', 'idle',
        'error'. A session that was stopped reads as 'idle' — being stopped
        is not a distinct runtime state, just "not running right now". The
        historical fact that it was stopped lives in last_terminal_event.

        ``session_id`` selects the chat session. When omitted (``None``), the
        call is holder-aware (seam 3 / Phase 1 / decision D1): it resolves to
        the project's active-loop-slot holder and reports THAT session's
        status, returning ``idle`` only when no session holds the slot. This
        replaces the old "default session" resolution, which returned ``idle``
        during a turn running under a non-default session (e.g. a uuid) and
        caused the REST poll to revert the WS-driven ``running``. An explicit
        ``session_id`` still targets exactly that session (unchanged).
        """
        if session_id is None:
            session_id = self.current_holder_session_id(project_id)
            if session_id is None:
                return "idle"
        else:
            session_id = self._resolve_session_id(session_id)
        handle = self._handles.get(make_session_key(project_id, session_id))
        if handle is None:
            return "idle"
        if handle.session.is_stopped():
            return "idle"
        # Check approval pause BEFORE task.done() — during approval the
        # task is still alive (blocked inside the interceptor's wait).
        if handle.session._paused_for_approval:
            return "pending_approval"
        if handle.task is not None and not handle.task.done():
            return "running"
        # Check for sub-agents
        poll_task = self._idle_poll_tasks.get(make_session_key(project_id, session_id))
        if poll_task is not None and not poll_task.done():
            return "waiting"
        return "idle"

    def update_autonomy(self, project_id: str, preset, *,
                        session_id: str | None = None) -> bool:
        """Push a new autonomy preset to a running agent's interceptor.

        Returns True if the running agent was updated, False if no agent
        is running (disk-only update is sufficient in that case).

        ``session_id`` selects the chat session; defaults to the
        single-loop default session.
        """
        session_id = self._sid_read(project_id, session_id)
        handle = self._handles.get(make_session_key(project_id, session_id))
        if handle is None:
            return False
        handle.interceptor.update_preset(preset)
        handle.session.append_system(
            f"[autonomy changed to {preset.value}]"
        )
        # Also propagate to sub-agent transports (SDK filtering)
        if self._sub_agent_manager is not None:
            self._sub_agent_manager.update_sub_agent_autonomy(
                project_id, preset, session_id=session_id,
            )
        return True

    def get_pending_approval(self, project_id: str, *,
                             session_id: str | None = None) -> dict | None:
        """Return the pending approval payload for a session, or None.

        Used by the REST recovery endpoint so mobile clients can fetch
        approval card data when they miss the WebSocket event.

        ``session_id`` selects the chat session; defaults to the
        single-loop default session.
        """
        session_id = self._sid_read(project_id, session_id)
        handle = self._handles.get(make_session_key(project_id, session_id))
        if handle is None:
            return None
        if not handle.session._paused_for_approval:
            return None
        # Return the first (and typically only) pending approval
        for tool_call_id, data in handle.interceptor._pending_approvals.items():
            return dict(data)
        return None

    async def approve(self, project_id: str, tool_call_id: str,
                      reply_text: str | None = None, *,
                      approve_all: bool = False,
                      session_id: str | None = None) -> None:
        """Approve a pending tool call and execute it.

        ``session_id`` selects the chat session; defaults to the
        single-loop default session.
        """
        session_id = self._sid_read(project_id, session_id)
        handle = self._handles.get(make_session_key(project_id, session_id))
        if handle is None:
            raise KeyError(f"No active session for project '{project_id}'")
        handle.last_activity = time.time()  # user interaction — defer eviction

        # Wait for previous loop task
        if handle.task is not None and not handle.task.done():
            try:
                await asyncio.wait_for(asyncio.shield(handle.task), timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                pass

        pending = handle.interceptor.get_pending(tool_call_id)
        if pending is None:
            raise KeyError(f"No pending approval for tool_call_id '{tool_call_id}'")

        # Guard: don't append a second result if one already exists
        if handle.session.has_result_for(tool_call_id):
            handle.interceptor.remove_pending(tool_call_id)
            handle.session.resume()
            await self._start_loop(project_id, session_id=session_id)
            return

        handle.interceptor.record_approval(pending["tool_name"], pending["tool_args"])
        self._record_approval_decision(
            project_id, pending["tool_name"], pending["tool_args"], "approved",
            session_id=session_id,
        )

        if approve_all:
            handle.interceptor.activate_bypass_all()

        # For request_access tools, grant folder access via platform provider
        if pending["tool_name"] == "request_access" and self._platform_provider is not None:
            tool_args = pending["tool_args"]
            path = tool_args.get("path", "")
            access_type = tool_args.get("access_type", "read")
            mode = "read_only" if access_type == "read" else "read_write"
            result = self._platform_provider.grant_folder_access(path, mode)
            if result.success:
                handle.session.append_tool_result(
                    tool_call_id,
                    json.dumps({"status": "granted", "path": path, "mode": mode}),
                )
            else:
                handle.session.append_tool_result(
                    tool_call_id,
                    json.dumps({"status": "error", "path": path, "error": result.error}),
                )
        else:
            # Execute the approved tool call (async-aware)
            try:
                if handle.registry.is_async(pending["tool_name"]):
                    result = await handle.registry.execute_async(
                        pending["tool_name"],
                        pending["tool_args"],
                    )
                else:
                    result = await asyncio.to_thread(
                        handle.registry.execute,
                        pending["tool_name"],
                        pending["tool_args"],
                    )
                handle.session.append_tool_result(
                    tool_call_id, result.content, meta=result.meta
                )
            except Exception as e:
                handle.session.append_tool_result(
                    tool_call_id, f"Error executing tool: {e}"
                )

        handle.interceptor.remove_pending(tool_call_id)

        # Append user guidance if reply_text was provided with the approval
        if reply_text:
            handle.session.append({
                "role": "user",
                "content": f"[User guidance on approved tool call]: {reply_text}",
                "source": "user",
            })

        # Approval is resolved — clear the pause flag so the blocked-count
        # broadcast sees zero pending for this session. loop.run() also clears
        # it on entry, but we clear early so the count is accurate immediately.
        handle.session._paused_for_approval = False
        handle.session.resume()
        self._broadcast_blocked_count()
        await self._start_loop(project_id, session_id=session_id)

    async def deny(self, project_id: str, tool_call_id: str, reason: str, *,
                   session_id: str | None = None) -> None:
        """Deny a pending tool call.

        ``session_id`` selects the chat session; defaults to the
        single-loop default session.
        """
        session_id = self._sid_read(project_id, session_id)
        handle = self._handles.get(make_session_key(project_id, session_id))
        if handle is None:
            raise KeyError(f"No active session for project '{project_id}'")
        handle.last_activity = time.time()  # user interaction — defer eviction

        # Wait for previous loop task
        if handle.task is not None and not handle.task.done():
            try:
                await asyncio.wait_for(asyncio.shield(handle.task), timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                pass

        # Record denial in approval history
        pending = handle.interceptor.get_pending(tool_call_id)
        if pending:
            self._record_approval_decision(
                project_id, pending["tool_name"], pending["tool_args"],
                "denied", deny_reason=reason,
                session_id=session_id,
            )

        # Guard: don't append a second result if one already exists
        if not handle.session.has_result_for(tool_call_id):
            handle.session.append_tool_result(
                tool_call_id,
                f"DENIED by user. Reason: {reason}",
            )

        handle.interceptor.remove_pending(tool_call_id)
        # Denial resolves the approval — clear the pause flag early so the
        # blocked-count broadcast sees the correct count immediately.
        handle.session._paused_for_approval = False
        handle.session.resume()
        self._broadcast_blocked_count()
        await self._start_loop(project_id, session_id=session_id)

    async def new_session(self, project_id: str, *,
                          session_id: str | None = None) -> dict:
        """Pure-create: mint a fresh session id + uuid for this project.

        A session is a file on disk. ``new_session`` only brings a new session
        *into existence as an identity* — it writes no JSONL (creation is
        deferred to the first message), registers no handle, and takes NO
        action on any other session. It does not terminate a running loop,
        flush workspace state, stop sub-agents, or transfer the active-loop
        slot. The new session becomes real — handle, loop, JSONL — on the
        first inject under the returned ``session_id`` (``inject_message``
        Case 3 auto-starts it).

        The ``session_id`` kwarg is accepted for backward-compatible call
        sites but ignored: a new session always gets a fresh minted id.

        Returns ``{"status": "ok", "session_id", "session_uuid"}``. The
        ``session_id`` is the user-facing F1 chat id the frontend navigates
        to; ``session_uuid`` is the F2 JSONL stem the session will use once
        materialized.
        """
        # Seam 3 / decision D2: the canonical session identity is the uuid (the
        # JSONL filename stem). The F1 (`sess_…`) mint is retired — session_id
        # IS the uuid, no separate/vestigial F1. Uses the shared mint primitive
        # (also used by the lazy chat funnel) so there is one uuid source.
        new_session_uuid = self._mint_session_uuid(project_id)
        logger.info(
            "new_session(%s): minted uuid %s (uuid-only; F1 retired)",
            project_id, new_session_uuid,
        )
        return {
            "status": "ok",
            "session_id": new_session_uuid,
            "session_uuid": new_session_uuid,
        }

    async def cancel_message(self, project_id: str, *,
                             session_id: str | None = None) -> dict:
        """Interrupt current turn for session. Agent stays alive; session preserved.

        ``session_id`` selects the chat session; defaults to the
        single-loop default session.

        ``/cancel`` means "stop all work in this project": in every active
        state it also tears down sub-agents and the idle-poll via
        ``_stop_sub_agents`` (the handle and on-disk session always survive —
        everything stays resumable). Cases:
        - No handle → ``no_agent``.
        - Handle present, paused for approval (task done but
          ``_paused_for_approval`` True) → dismiss any pending approval,
          clear the flag, stop sub-agents, broadcast idle, return
          ``cancelled``. Mirrors inject Case 1b's approval-dismissal pattern.
        - Handle present, task done but sub-agents still working (idle-poll
          alive, the ``waiting`` state) → cancel the poll, stop sub-agents,
          write a cancellation marker, broadcast idle, return ``cancelled``.
        - Handle present, task running → call ``cancel_turn()``, stop
          sub-agents, broadcast idle, return ``cancelled``.
        - Handle present, task done, not paused, no sub-agents → ``idle``
          (legitimate no-op — there was nothing to cancel).
        """
        session_id = self._sid_read(project_id, session_id)
        handle = self._handles.get(make_session_key(project_id, session_id))
        if handle is None:
            return {"status": "no_agent"}
        if handle.task is None or handle.task.done():
            # Approval-pause path: the loop exited at the approval boundary
            # but the session's _paused_for_approval flag keeps it stuck
            # in pending_approval state. /cancel must clear it.
            if handle.session._paused_for_approval:
                # Dismiss any pending approvals (mirrors inject Case 1b at
                # _on_inject_paused). Single approval is the common case.
                if handle.interceptor is not None:
                    for tc_id, data in list(
                        handle.interceptor._pending_approvals.items()
                    ):
                        tool_name = data.get("tool_name", "unknown")
                        tool_args = data.get("tool_args", {})
                        if not handle.session.has_result_for(tc_id):
                            handle.session.append_tool_result(
                                tc_id,
                                "DISMISSED: User cancelled the session "
                                "while this approval was pending.",
                            )
                        self._record_approval_decision(
                            project_id,
                            tool_name,
                            tool_args,
                            "denied",
                            deny_reason=(
                                "User cancelled while approval was pending"
                            ),
                            session_id=session_id,
                        )
                        handle.interceptor.remove_pending(tc_id)
                # Clear the approval-pause flag and the legacy paused flag.
                handle.session._paused_for_approval = False
                handle.session.resume()
                # A management agent paused at approval could have spawned
                # sub-agents earlier in the turn that are still running.
                await self._stop_sub_agents(project_id, session_id=session_id)
                self._broadcast(project_id, {
                    "type": "agent.status",
                    "project_id": project_id,
                    "status": "idle",
                }, session_id=session_id)
                # Notify all clients that blocked count dropped.
                self._broadcast_blocked_count()
                return {"status": "cancelled"}

            # Waiting state: main loop exited but sub-agents are still working
            # (idle-poll alive). /cancel must stop them, not no-op.
            poll_task = self._idle_poll_tasks.get(make_session_key(project_id, session_id))
            has_poll = poll_task is not None and not poll_task.done()
            if has_poll:
                await self._stop_sub_agents(project_id, session_id=session_id)
                # Write the cancellation marker directly: the loop task is
                # already done, so loop._persist_cancellation_marker_inner
                # cannot run. Format matches loop.py's marker exactly so the
                # management agent reads a uniform cancellation on resume.
                handle.session.append({
                    "role": "system",
                    "content": "[cancelled by user]",
                    "source": "management",
                    "cancelled_by_user": True,
                })
                self._broadcast(project_id, {
                    "type": "agent.status",
                    "project_id": project_id,
                    "status": "idle",
                }, session_id=session_id)
                return {"status": "cancelled"}

            # Truly idle — nothing to cancel.
            return {"status": "idle"}

        # cancel_turn exceptions intentionally bubble: cancel_turn has its own
        # internal exception handling (see loop.py:782-787) for the cross-task
        # wait_for, so any exception escaping here indicates a real bug worth
        # surfacing as an HTTP 500.
        await handle.loop.cancel_turn()
        # Sub-agents spawned earlier in this turn keep running after the main
        # loop is interrupted — tear them down too. Idempotent: a no-op when
        # there are no sub-agents.
        await self._stop_sub_agents(project_id, session_id=session_id)
        # Broadcast after cancel_turn returns: the cancellation marker is
        # written synchronously inside cancel_turn before any await
        # (loop.py:779), so the JSONL is durable by the time we reach this
        # line. If cancel_turn is later refactored to defer the marker write,
        # this ordering assumption needs to be revisited.
        self._broadcast(project_id, {
            "type": "agent.status",
            "project_id": project_id,
            "status": "idle",
        }, session_id=session_id)
        return {"status": "cancelled"}

    async def _stop_sub_agents(self, project_id: str, *,
                               session_id: str | None = None) -> None:
        """Cancel idle-poll and stop all sub-agents for a project. Idempotent.

        Shared by ``stop_agent`` (full teardown) and ``cancel_message``
        (turn interrupt). ``_idle_poll_tasks`` is keyed by ``SessionKey`` so
        each session's poll task is tracked independently — the pop here
        targets only the calling session's poll.
        """
        session_id = self._resolve_session_id(session_id)
        # Cancel idle poll task if running
        poll_task = self._idle_poll_tasks.pop(make_session_key(project_id, session_id), None)
        if poll_task and not poll_task.done():
            poll_task.cancel()

        # Stop sub-agents with a budget that EXCEEDS the per-adapter kill
        # budget (Piece 3 Part E — the old 5.0s here vs ADAPTER_STOP_TIMEOUT
        # was the budget inversion that cancelled kills mid-flight into the
        # force-drop leak). Even on expiry nothing leaks: the teardown task
        # inside SubAgentManager.stop is shielded and runs to confirmed
        # death; this wait_for only stops WAITING.
        try:
            await asyncio.wait_for(
                self._sub_agent_manager.stop_all(project_id, session_id=session_id),
                timeout=self.SUB_AGENT_TEARDOWN_BUDGET,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "sub-agent stop_all wait budget exceeded for project %s "
                "(kills continue shielded in background)", project_id
            )

    async def stop_agent(self, project_id: str, *,
                         session_id: str | None = None) -> None:
        """Stop agent and clean up.

        ``session_id`` selects which chat session's loop to stop. Other
        sessions in the same project continue running. Defaults to the
        single-loop default session for back-compat.
        """
        session_id = self._resolve_session_id(session_id)
        sk = make_session_key(project_id, session_id)
        handle = self._handles.get(sk)
        if handle is None:
            raise KeyError(f"No active session for project '{project_id}'")

        # The queue dispatcher is project-scoped: created at daemon startup,
        # torn down only at daemon shutdown or project deletion. It deliberately
        # SURVIVES agent stop — it is the entity responsible for (re)starting
        # agents, so it must outlive any individual agent. (Previously this
        # popped + shut down the dispatcher here, which made an agentless queue
        # impossible and produced the "queue stuck RUNNING / 409 on stop" bugs.)
        # Teardown lives in shutdown() (daemon) and shutdown_dispatcher() (delete).

        # Cancel idle-poll + stop sub-agents (shared with cancel_message).
        await self._stop_sub_agents(project_id, session_id=session_id)

        # Close browser pages for this project
        if self._browser_manager is not None:
            try:
                await self._browser_manager.close_project_pages(project_id)
            except Exception:
                pass

        # Stop platform sandbox processes
        if self._platform_provider is not None:
            await self._platform_provider.stop_process(project_id)

        # terminate() interrupts any in-flight LLM stream and sets the stop
        # flag; the shield-wait below is the final reaper for the loop's
        # finally block.
        if handle.task is not None and not handle.task.done():
            await handle.loop.terminate()
            try:
                await asyncio.wait_for(asyncio.shield(handle.task), timeout=10.0)
            except (asyncio.TimeoutError, Exception):
                pass
        else:
            # Loop already done — preserve the legacy stop flag for any
            # downstream code that observes it.
            handle.session.stop()

        self._set_last_terminal_event(project_id, session_id, "stopped")
        self._broadcast(project_id, {
            "type": "agent.status",
            "project_id": project_id,
            "status": "idle",
            "source": "management",
        }, session_id=session_id)

        self._handles.pop(sk, None)

        # Allow system sleep again if no agents remain.
        self._allow_sleep_if_idle()

    def _resolve_session_uuid_on_disk(self, project_id: str,
                                      identifier: str) -> tuple[str, str] | None:
        """Resolve an F1 or F2 identifier to ``(session_uuid, jsonl_path)``.

        Mirrors ``_find_session_uuid_on_disk`` (agents_v2): a direct
        ``{identifier}.jsonl`` match wins (uuid addressing — how the sidebar
        addresses disk-only sessions); otherwise scan each JSONL for a record
        whose F1 ``session_id`` matches. Returns ``None`` if nothing matches.
        """
        project = self._project_store.get_project(project_id) if self._project_store else None
        workspace = (project or {}).get("workspace", "") if project else ""
        if not workspace:
            return None
        sessions_dir = ProjectPaths(workspace).sessions_dir
        if not os.path.isdir(sessions_dir):
            return None
        # Direct F2 match.
        direct = os.path.join(sessions_dir, f"{identifier}.jsonl")
        if os.path.isfile(direct):
            return identifier, direct
        # F1 scan.
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
                        if rec.get("session_id") == identifier:
                            resolved_uuid = fname[:-6]
                            # Seam 3 / decision D4: the F1-scan fallback is kept
                            # for back-compat (a persisted F1/"default" id that
                            # isn't a filename) but INSTRUMENTED so we can confirm
                            # it goes cold before deleting it in a later pass.
                            logger.info(
                                "f1_scan_fallback fired: project=%s identifier=%s "
                                "resolved_uuid=%s (legacy F1 addressing — should "
                                "trend to zero post canonicalization)",
                                project_id, identifier, resolved_uuid,
                            )
                            return resolved_uuid, fpath
            except OSError:
                continue
        return None

    async def delete_session(self, project_id: str, session_id: str) -> dict:
        """Delete a single session: remove its JSONL from disk.

        Steps (TASK-session-naming-and-deletion.md PART 2):
          1. Resolve ``session_id`` (F1 or F2/uuid) to its JSONL path.
             Raises ``FileNotFoundError`` if no session matches → 404.
          2. If the session is currently running (run-status != idle) raise
             ``RuntimeError`` → 409. Idle handles are fine to delete.
          3. If a live handle exists for this session, tear it down like
             eviction (``stop_agent`` — which also cancels the idle-poll task
             and stops sub-agents) so no in-memory state outlives the file.
          4. ``os.remove`` the JSONL (meta line lives inside it — Option A).
          5. Return ``{"status": "deleted", "session_id": <identifier>}``.

        Workspace files the agent created are NOT touched — they belong to the
        project, not the session.
        """
        resolved = self._resolve_session_uuid_on_disk(project_id, session_id)
        if resolved is None:
            raise FileNotFoundError(
                f"No session '{session_id}' for project '{project_id}'"
            )
        session_uuid, jsonl_path = resolved

        # Locate any live handle for this session. Both the F1 chat id and the
        # F2 uuid can address it, so match on either the handle's session_id
        # key OR the loaded session's session_uuid.
        handle_key = None
        for (pid, sid), handle in self._handles.items():
            if pid != project_id:
                continue
            if sid == session_id or getattr(
                handle.session, "session_uuid", None
            ) == session_uuid:
                handle_key = (pid, sid)
                break

        if handle_key is not None:
            _pid, sid = handle_key
            # Running / pending_approval / waiting → refuse (409). Only idle
            # sessions may be deleted; the caller must cancel first.
            if self.get_run_status(project_id, session_id=sid) != "idle":
                raise RuntimeError(
                    "Cannot delete a running session. Cancel it first."
                )
            # Idle handle: tear it down (cancels idle-poll, stops sub-agents,
            # pops the handle) before unlinking the file — same as eviction.
            try:
                await self.stop_agent(project_id, session_id=sid)
            except KeyError:
                pass

        # Remove the JSONL from disk. The session_start meta (and thus the
        # name) lives inside the file and is deleted with it (Option A).
        try:
            os.remove(jsonl_path)
        except FileNotFoundError:
            pass

        # Drop any recorded terminal event so a stale warning glyph can't
        # outlive the deleted session in a future re-list.
        for (pid, sid) in list(self._last_terminal_events.keys()):
            if pid == project_id and (sid == session_id):
                self._last_terminal_events.pop((pid, sid), None)

        logger.info(
            "delete_session(%s): removed session %s (uuid %s)",
            project_id, session_id, session_uuid,
        )
        return {"status": "deleted", "session_id": session_id}

    def rename_session(self, project_id: str, session_id: str, name: str) -> dict:
        """Rename a session: update its display name in memory + on disk.

        Resolves the session (F1 or F2/uuid). If a live handle exists, update
        the in-memory ``session.name`` (which rewrites the meta line). Otherwise
        load the on-disk session, rename it (rewrites the meta line), and let it
        fall out of scope — the file is the source of truth and is now updated.

        Raises ``FileNotFoundError`` if no session matches → 404.
        Returns ``{"status": "renamed", "session_id": <id>, "name": <name>}``.
        """
        resolved = self._resolve_session_uuid_on_disk(project_id, session_id)
        if resolved is None:
            raise FileNotFoundError(
                f"No session '{session_id}' for project '{project_id}'"
            )
        session_uuid, jsonl_path = resolved

        # Live handle path: rename in memory (also rewrites the meta line).
        for (pid, sid), handle in self._handles.items():
            if pid != project_id:
                continue
            if sid == session_id or getattr(
                handle.session, "session_uuid", None
            ) == session_uuid:
                handle.session.set_name(name)
                return {"status": "renamed", "session_id": session_id, "name": name}

        # Disk-only path: load, rename (rewrites the meta line), done.
        loaded = Session.load(jsonl_path)
        loaded.set_name(name)
        return {"status": "renamed", "session_id": session_id, "name": name}

    # ------------------------------------------------------------------
    # Sub-agent resume persistence (TASK-resume-persistence, piece 2)
    # ------------------------------------------------------------------

    def record_sub_agent_thread(self, project_id: str, handle: str, *,
                                claude_session_id: str,
                                model: str | None = None,
                                session_id: str | None = None,
                                proc_pid: int | None = None,
                                proc_create_time: float | None = None,
                                rollout_path: str | None = None) -> None:
        """Persist a sub-agent's resume identity onto the management session.

        Fired (via LifecycleObserver.on_thread_update) on each completed
        sub-agent turn. Composite key ``(SessionKey, handle)``: the session
        JSONL is SessionKey-scoped and the meta row carries the handle. A
        completion racing an eviction (session no longer hydrated) is logged
        and dropped — the previous record, if any, remains valid.
        """
        # Reuses the existing get_session accessor (keyword-only session_id,
        # defined further down with the queue-era helpers).
        session = self.get_session(project_id, session_id=session_id)
        if session is None:
            logger.warning(
                "record_sub_agent_thread: no hydrated session for %s/%s — "
                "thread id %s for handle %s not recorded",
                project_id, session_id, claude_session_id, handle,
            )
            return
        session.set_sub_agent_thread(
            handle, session_id=claude_session_id, model=model,
            proc_pid=proc_pid, proc_create_time=proc_create_time,
            rollout_path=rollout_path,
        )

    def list_sessions(self, project_id: str) -> list[dict]:
        """Return a list of active/idle sessions for a project.

        Each entry has ``session_id``, ``status`` (running | pending_approval |
        waiting | idle), and ``session_uuid`` (the per-loop
        Session.session_id, useful for jsonl path correlation).

        Multi-loop callers use this to render a session list. Single-loop
        callers that have never set an explicit session_id will see at most
        one entry under ``session_id == DEFAULT_SESSION_ID``.
        """
        sessions = []
        for (pid, sid), handle in self._handles.items():
            if pid != project_id:
                continue
            # Inline get_run_status to avoid double dict lookup. A stopped
            # session reads as idle (not a distinct runtime state).
            if handle.session.is_stopped():
                status = "idle"
            elif handle.session._paused_for_approval:
                status = "pending_approval"
            elif handle.task is not None and not handle.task.done():
                status = "running"
            else:
                poll_task = self._idle_poll_tasks.get(make_session_key(pid, sid))
                if poll_task is not None and not poll_task.done():
                    status = "waiting"
                else:
                    status = "idle"
            # last_activity_at: timestamp of the most recent message in-memory.
            # Falls back to None if the session has no messages yet.
            msgs = getattr(handle.session, "_messages", None) or []
            last_activity_at = msgs[-1].get("timestamp") if msgs else None

            sessions.append({
                "session_id": sid,
                "status": status,
                # F2 JSONL filename stem (session_uuid), NOT the F1 user-facing
                # session_id. Consumers (e.g. the /chat session_id filter) build
                # ``{session_uuid}.jsonl`` from this value, so it MUST be F2.
                "session_uuid": getattr(handle.session, "session_uuid", None),
                # Seam 3 / Phase 4: chat | queue (session switcher visual
                # distinction; D1's persistent-chat resolver).
                "origin": getattr(handle.session, "origin", "chat"),
                # Display label (auto-derived from first user message or set via
                # rename). None for headless sessions; frontend falls back to
                # the first message / session id.
                "name": getattr(handle.session, "name", None),
                "last_terminal_event": self._last_terminal_events.get(
                    (pid, sid),
                ),
                "last_activity_at": last_activity_at,
            })
        # Merge in disk-only sessions not currently hydrated in memory, so the
        # sidebar shows every persisted session log. In-memory entries above
        # carry live status; disk-only entries are idle and get hydrated from
        # JSONL only when the user acts on them.
        seen_uuids = {s["session_uuid"] for s in sessions if s.get("session_uuid")}
        sessions.extend(self._disk_session_entries(project_id, seen_uuids))
        # Stable ordering by session id (the "default"-first rule retired with
        # DEFAULT_SESSION_ID; ids are now uuids). None-safe.
        sessions.sort(key=lambda s: (s.get("session_id") or ""))
        return sessions

    def _disk_session_entries(self, project_id: str, seen_uuids: set) -> list[dict]:
        """Enumerate the project's on-disk session JSONLs not currently hydrated
        in memory. Each becomes an idle sidebar entry. Skips empty/meta-only
        logs and the lock sentinels."""
        project = self._project_store.get_project(project_id) if self._project_store else None
        workspace = (project or {}).get("workspace", "") if project else ""
        if not workspace:
            return []
        sessions_dir = ProjectPaths(workspace).sessions_dir
        try:
            fnames = os.listdir(sessions_dir)
        except OSError:
            return []
        entries: list[dict] = []
        for fname in fnames:
            if not fname.endswith(".jsonl"):
                continue
            uuid = fname[:-6]
            if uuid in seen_uuids:
                continue  # already represented by a live in-memory handle
            last_activity_at = None
            stored_name = None  # name on the session_start meta, if present
            origin = "chat"  # session_start meta origin; legacy logs → chat
            first_user_content = None  # for name backfill
            try:
                with open(os.path.join(sessions_dir, fname), "r", encoding="utf-8") as fh:
                    first_real = None  # first non-meta (conversation) record
                    last_real = None
                    for raw in fh:
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            rec = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        # Skip identity metadata (session_start, model_swap, …)
                        # — a log with only meta records is not a real session.
                        if rec.get("role") == "meta":
                            if rec.get("event") == "session_start":
                                if rec.get("name") is not None:
                                    stored_name = rec["name"]
                                if rec.get("origin"):
                                    origin = rec["origin"]
                            continue
                        if first_real is None:
                            first_real = rec
                        if first_user_content is None and rec.get("role") == "user":
                            first_user_content = rec.get("content")
                        last_real = rec
            except OSError:
                continue
            if first_real is None:
                continue  # empty / meta-only log — not a materialized session
            if last_real is not None:
                last_activity_at = last_real.get("timestamp")
            # Name: stored meta name wins; else derive from first user message
            # (lazy backfill, in-memory only — no file rewrite); else None.
            from agent_os.agent.session import _derive_name
            name = stored_name if stored_name is not None else _derive_name(first_user_content)
            # Address a disk-only session by its unique session_uuid: the F1
            # session_id on disk is often "default" and not unique across many
            # prior logs, which would shadow the active session and break the
            # chat endpoint's F1→F2 fast-path mapping. The uuid is unique and is
            # what callers use to load/act on (hydrate) the disk-only session.
            entries.append({
                "session_id": uuid,
                "status": "idle",
                "session_uuid": uuid,
                "origin": origin,
                "name": name,
                "last_terminal_event": None,
                "last_activity_at": last_activity_at,
            })
        return entries

    def list_blocked_sessions(self) -> list[dict]:
        """Return all sessions currently in ``pending_approval`` state, across
        all projects.

        Each entry is ``{"project_id": str, "session_id": str}``.
        Used by ``GET /api/v2/blocked`` and the WS ``blocked-count-changed`` event.
        """
        blocked = []
        for (pid, sid), handle in self._handles.items():
            if handle.session._paused_for_approval:
                blocked.append({"project_id": pid, "session_id": sid})
        return blocked

    def _broadcast_blocked_count(self) -> None:
        """Emit a global ``blocked-count-changed`` WS event with the current
        aggregate blocked-session list.

        Called after every transition into or out of ``pending_approval`` so
        all connected clients stay in sync without polling.
        """
        blocked = self.list_blocked_sessions()
        self._ws.broadcast_global({
            "type": "blocked-count-changed",
            "blocked_count": len(blocked),
            "blocked_sessions": blocked,
        })

    def get_session(self, project_id: str, *,
                    session_id: str | None = None):
        """Return session for a project, or None.

        ``session_id`` selects which chat session to look up; defaults to
        the single-loop default session.
        """
        session_id = self._resolve_session_id(session_id)
        handle = self._handles.get(make_session_key(project_id, session_id))
        return handle.session if handle else None

    # ── Queue helpers (Phase 1) ─────────────────────────────────────────
    # F7 note: the queue is per-project, not per-session. The single-active-
    # loop slot model (Track E foundation + #23) guarantees at most one
    # handle keyed under each project_id; queue helpers scan ``self._handles``
    # for that project's handle without needing to know which session_id
    # currently holds the slot. The dispatcher may swap the active session
    # via ``switch_session`` (e.g. from default → ``chat_<uuid>`` on pause,
    # or back to a parked attempt's session on resume), and these helpers
    # transparently track the new key.

    def _find_project_handle(self, project_id: str):
        """Return the (single) ProjectHandle for ``project_id`` regardless of
        which session_id it is currently keyed under, or None.

        Used by per-project queue helpers that don't know (or care) what
        session is currently active in the slot.
        """
        for (pid, _sid), handle in self._handles.items():
            if pid == project_id:
                return handle
        return None

    def get_active_session(self, project_id: str):
        """Return the project's active session regardless of session_id, or None.

        Distinct from ``get_session(project_id)`` which defaults to a strict
        default-session lookup (and returns None when isolation tests use
        non-default session ids). The queue dispatcher uses this because
        after ``switch_session`` to ``chat_<uuid>`` (queue stop) or to a
        parked attempt's id (queue resume), the dispatcher needs to find
        whatever session currently holds the project's single slot — not
        the default sentinel which may not exist.
        """
        handle = self._find_project_handle(project_id)
        return handle.session if handle is not None else None

    def get_queue_store(self, project_id: str, workspace: str | None = None) -> QueueStore:
        """Return the QueueStore for a project, creating it on first access.

        Workspace is looked up from the active handle when omitted; pass it
        explicitly only from reclaim/startup paths where no handle exists yet.
        """
        store = self._queue_stores.get(project_id)
        if store is not None:
            return store
        if workspace is None:
            handle = self._find_project_handle(project_id)
            if handle is None:
                project = self._project_store.get_project(project_id)
                if project is None:
                    raise KeyError(f"No project '{project_id}'")
                workspace = project.get("workspace") or ""
            else:
                workspace = handle.config_snapshot.get("workspace", "")
        if not workspace:
            raise ValueError(f"No workspace known for project '{project_id}'")
        paths = ProjectPaths(workspace)
        store = QueueStore(paths.queue_file)
        self._queue_stores[project_id] = store
        return store

    def get_dispatcher(self, project_id: str) -> "QueueDispatcher | None":
        return self._dispatchers.get(project_id)

    def get_loop_task(self, project_id: str, *,
                      session_id: str | None = None) -> "asyncio.Task | None":
        """Used by the QueueDispatcher to await the inner loop task. When
        ``session_id`` is given, target that specific session's handle —
        required now the dispatcher runs each queue item in its own session,
        so ``_find_project_handle`` (first handle by order) is no longer
        unambiguous. Without it, fall back to the single project handle."""
        handle = (
            self._handles.get(make_session_key(project_id, session_id))
            if session_id is not None
            else self._find_project_handle(project_id)
        )
        return handle.task if handle else None

    def get_loop(self, project_id: str, *, session_id: str | None = None):
        """Return the AgentLoop object so the dispatcher can read exit_reason.
        When ``session_id`` is given, target that specific session's handle."""
        handle = (
            self._handles.get(make_session_key(project_id, session_id))
            if session_id is not None
            else self._find_project_handle(project_id)
        )
        return handle.loop if handle else None

    def get_sub_agent_manager(self):
        return self._sub_agent_manager

    async def switch_session(
        self,
        project_id: str,
        session_id: str,
        *,
        start_loop: bool = False,
    ):
        """Swap the active session for a project to the given session_id.

        Used by the queue dispatcher for stop (switch to chat) and resume
        (switch back to a parked attempt). Unlike new_session(), this does
        NOT run the pre-flush routine, stop sub-agents, or mint a new id —
        the caller owns those decisions.

        If the JSONL for session_id exists, it is loaded; otherwise a fresh
        session with that id is created. Loop, ContextManager and handle
        references are all kept consistent.

        F7 note: ``self._handles`` is keyed by ``(project_id, session_id)``
        (SessionKey tuple) after the multi-session foundation (#22). We scan
        for any handle under ``project_id`` (single-slot discipline: at most
        one) and re-key it to the new ``session_id`` after the swap so
        downstream lookups (which use ``make_session_key``) find it.
        """
        old_sk = next(
            (sk for sk in self._handles.keys() if sk[0] == project_id),
            None,
        )
        if old_sk is None:
            return None
        handle = self._handles[old_sk]

        # Stop the loop if running, but skip pre-flush. The caller decides
        # whether to preserve the session (stop) or rotate (new_session).
        if handle.task is not None and not handle.task.done():
            await handle.loop.terminate()
            try:
                await asyncio.wait_for(asyncio.shield(handle.task), timeout=10.0)
            except (asyncio.TimeoutError, Exception):
                logger.warning(
                    "switch_session(%s): loop did not stop gracefully",
                    project_id,
                )

        workspace = handle.config_snapshot.get("workspace", "")
        paths = ProjectPaths(workspace)
        session_path = paths.session_file(session_id)

        snap = handle.config_snapshot
        if os.path.exists(session_path):
            new_session = Session.load(session_path)
            new_session.session_id = session_id
        else:
            new_session = Session.new(
                session_id, workspace,
                provider=snap.get("provider", "unknown"),
                model=snap.get("model", "unknown"),
                sdk=snap.get("sdk", "unknown"),
                fallback_models=snap.get("fallback_models", []),
            )

        new_session.on_append = self._on_message(project_id, session_id)
        new_session.on_stream = self._on_stream(project_id, session_id)

        cm = handle.context_manager
        if hasattr(cm, "_session"):
            cm._session = new_session
        if hasattr(cm, "_cold_resume_injected"):
            cm._cold_resume_injected = False
        if hasattr(cm, "_recovery_injected"):
            cm._recovery_injected = False
        if hasattr(cm, "_window_factor"):
            cm._window_factor = 1.0
        if hasattr(cm, "_last_usage_pct"):
            cm._last_usage_pct = 0.0

        handle.session = new_session
        handle.task = None
        if hasattr(handle.loop, "_session"):
            handle.loop._session = new_session

        # Re-key the handle under the new session_id (F7: tuple-keyed
        # handles must move with the session swap so downstream
        # ``make_session_key(project_id, session_id)`` lookups land here).
        new_sk = make_session_key(project_id, session_id)
        if new_sk != old_sk:
            del self._handles[old_sk]
            self._handles[new_sk] = handle

        if start_loop:
            await self._start_loop(project_id, session_id=session_id)

        return new_session

    async def start_all_dispatchers(self) -> None:
        """Create one queue dispatcher per project at daemon startup.

        The dispatcher is the project's session-lifecycle manager: it exists
        for the life of the project, independent of whether an agent is
        currently running. Creating them all at boot is what reconciles an
        agentless queue to IDLE and lets queue work resume without a prior
        manual agent start. Idempotent via _ensure_dispatcher.
        """
        if self._project_store is None:
            return
        try:
            projects = self._project_store.list_projects()
        except Exception:
            logger.exception("start_all_dispatchers: failed to list projects")
            return
        for project in projects:
            project_id = project.get("project_id")
            workspace = project.get("workspace") or ""
            if not project_id or not workspace:
                continue
            try:
                await self._ensure_dispatcher(project_id, workspace)
            except Exception:
                logger.exception(
                    "start_all_dispatchers: failed to start dispatcher for %s",
                    project_id,
                )

    # ------------------------------------------------------------------
    # Idle eviction
    # ------------------------------------------------------------------
    # After N seconds with no user interaction, an idle session's runtime
    # resources (handle, browser pages, sub-agents, sandbox) are torn down
    # via stop_agent. The next message cold-starts from the JSONL on disk —
    # measured at ~140ms (TASK/REPORT-restart-costs.md), invisible behind the
    # LLM round-trip. Hardcoded policy, not a user preference.
    EVICTION_IDLE_TIMEOUT = 180   # seconds (3 minutes)
    EVICTION_SWEEP_INTERVAL = 60  # seconds between sweeps

    async def _evict_idle_once(self) -> list:
        """Run a single eviction pass. Returns the SessionKeys evicted.

        Only sessions that are BOTH idle longer than ``EVICTION_IDLE_TIMEOUT``
        AND in run-status ``idle`` are evicted. Running, pending_approval, and
        waiting sessions are never evicted regardless of idle time — their
        in-memory state (loop, approval, sub-agent poll) would be lost.

        A failure tearing down one session is logged and does not abort the
        sweep. Factored out of ``_eviction_sweep`` so the decision logic is
        unit-testable without the infinite loop.
        """
        now = time.time()
        to_evict = []
        for sk, handle in list(self._handles.items()):
            if now - handle.last_activity < self.EVICTION_IDLE_TIMEOUT:
                continue
            pid, sid = sk
            if self.get_run_status(pid, session_id=sid) != "idle":
                continue
            # Piece 3 Part B: eviction tears down sub-agents too, so only a
            # session whose sub-agents are ALL true-idle is evictable. A
            # working or background-running sub-agent blocks eviction, and
            # the inactivity timer resets on registered background-work
            # activity — never kill real work on a clock.
            if self._sub_agents_block_eviction(pid, sid):
                continue
            to_evict.append(sk)
        for sk in to_evict:
            pid, sid = sk
            logger.info("evicting idle project %s session %s", pid, sid)
            try:
                await self.stop_agent(pid, session_id=sid)
            except Exception:
                logger.warning(
                    "eviction failed for %s/%s", pid, sid, exc_info=True
                )
        return to_evict

    def _sub_agents_block_eviction(self, project_id: str, session_id: str) -> bool:
        """True when a session's sub-agents make it non-evictable (Piece 3).

        Blocks eviction when:
        - any sub-agent is working or background-running (status != idle);
        - a degraded-path transport (no background-status support) has live
          descendants — conservative: we cannot classify them, so we do not
          kill them (may over-keep agents with warm infra; costs only RAM);
        - registered background work was active more recently than
          ``EVICTION_IDLE_TIMEOUT`` (the timer resets on background
          activity, not just turn events).

        Tolerant of partial managers (tests, degraded wiring): any failure to
        read sub-agent state preserves the legacy eviction decision.
        """
        if self._sub_agent_manager is None:
            return False
        try:
            active = self._sub_agent_manager.list_active(
                project_id, session_id=session_id,
            )
            if any(a.get("status") != "idle" for a in active):
                return True
            # `is True` keeps partially-mocked managers (tests) on the
            # legacy decision — a Mock return is not a positive signal.
            if self._sub_agent_manager.has_live_descendants_degraded(
                    project_id, session_id=session_id) is True:
                return True
            secs = self._sub_agent_manager.seconds_since_background_activity(
                project_id, session_id=session_id,
            )
            if isinstance(secs, (int, float)) and secs < self.EVICTION_IDLE_TIMEOUT:
                return True
        except Exception:
            return False
        return False

    async def _eviction_sweep(self) -> None:
        """Forever: sleep, then run one eviction pass. Cancelled on shutdown."""
        while True:
            await asyncio.sleep(self.EVICTION_SWEEP_INTERVAL)
            try:
                await self._evict_idle_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("eviction sweep iteration error", exc_info=True)

    def start_eviction(self) -> None:
        """Launch the idle-eviction sweep task. Idempotent. Called at daemon
        startup; the task is cancelled in ``shutdown``."""
        if self._eviction_task is not None and not self._eviction_task.done():
            return
        self._eviction_task = asyncio.create_task(self._eviction_sweep())

    async def shutdown_dispatcher(self, project_id: str) -> None:
        """Tear down a single project's dispatcher (project-deletion path)."""
        dispatcher = self._dispatchers.pop(project_id, None)
        if dispatcher is not None:
            try:
                await dispatcher.shutdown()
            except Exception:
                logger.warning(
                    "dispatcher shutdown failed for %s", project_id, exc_info=True,
                )

    def is_onboarding_complete(self, project_id: str) -> bool:
        """True if the project has captured state (PROJECT_STATE.md exists).

        Mirrors the queue-start onboarding gate: the dispatcher must not
        auto-start an agent for a project that has never completed a chat
        session, because there is no captured context to operate against.
        """
        project = self._project_store.get_project(project_id) if self._project_store else None
        if not project:
            return False
        workspace = project.get("workspace", "")
        if not workspace:
            return False
        return os.path.exists(ProjectPaths(workspace).project_state)

    async def _ensure_dispatcher(self, project_id: str, workspace: str) -> None:
        """Create + start the dispatcher for this project if not already running."""
        existing = self._dispatchers.get(project_id)
        if existing is not None:
            return
        store = self.get_queue_store(project_id, workspace=workspace)
        max_runtime = 1800
        try:
            if hasattr(self._project_store, "get_max_runtime_seconds"):
                max_runtime = self._project_store.get_max_runtime_seconds(project_id)
        except Exception:
            pass
        dispatcher = QueueDispatcher(
            project_id=project_id,
            store=store,
            agent_manager=self,
            ws_manager=self._ws,
            max_runtime_seconds=max_runtime,
            workspace=workspace,
        )
        # D6: reclaim before starting the dispatch task. Doing it pre-start
        # ensures the first tick of _run sees a coherent queue.
        try:
            reclaim_summary = dispatcher.reclaim_on_startup()
            if reclaim_summary.get("reclaimed_items") or reclaim_summary.get("blocked_items"):
                logger.info(
                    "dispatcher(%s): startup reclaim: %s",
                    project_id, reclaim_summary,
                )
        except Exception:
            logger.exception(
                "dispatcher(%s): reclaim_on_startup raised; continuing",
                project_id,
            )
        self._dispatchers[project_id] = dispatcher
        await dispatcher.start()

    async def _start_loop(self, project_id: str, *,
                          session_id: str | None = None) -> None:
        """Start a new loop.run() on existing session (hot resume)."""
        session_id = self._resolve_session_id(session_id)
        sk = make_session_key(project_id, session_id)
        handle = self._handles.get(sk)
        if handle is None:
            return
        handle.last_activity = time.time()  # resume = activity; defer eviction

        # Re-resolve API key so a key changed in settings takes effect
        # without requiring a full agent restart.
        project = self._project_store.get_project(project_id) or {}
        cred_key = self._credential_store.get_api_key() if self._credential_store else None
        global_settings = self._settings_store.get() if self._settings_store else None
        fresh_key = (
            project.get("api_key")
            or cred_key
            or (global_settings.llm.api_key if global_settings else None)
            or ""
        )
        handle.provider.update_api_key(fresh_key)
        # Also update utility and fallback providers on the loop
        loop = handle.loop
        if hasattr(loop, '_utility_provider') and loop._utility_provider is not None:
            if loop._utility_provider is not handle.provider:
                loop._utility_provider.update_api_key(fresh_key)
        if hasattr(loop, '_fallback_providers'):
            for fb in loop._fallback_providers:
                fb.update_api_key(fresh_key)

        task = asyncio.create_task(handle.loop.run())
        task.add_done_callback(
            self._on_loop_done(project_id, session_id=session_id)
        )
        handle.task = task

        self._broadcast(project_id, {
            "type": "agent.status",
            "project_id": project_id,
            "status": "running",
            "source": "management",
        }, session_id=session_id)

    def _on_message(self, project_id: str, session_id: str):
        """Returns callback for session.on_append.

        ``session_id`` is captured in the closure and forwarded to the
        translator so the WebSocket ``agent.activity`` / ``agent.status_summary``
        events carry it. Without it the frontend cannot attribute the event
        to a specific session and falls back to the holder heuristic, which
        races with holder resolution.
        """
        def callback(msg):
            self._activity_translator.on_message(msg, project_id, session_id=session_id)
        return callback

    def _on_stream(self, project_id: str, session_id: str):
        """Returns callback for session.on_stream."""
        def callback(chunk):
            self._activity_translator.on_stream_chunk(
                chunk, project_id, "management", session_id=session_id,
            )
        return callback

    # Piece 3 Part B: _MAX_IDLE_POLLS (the 300s busy-watchdog kill) is
    # REMOVED. No fixed clock may kill a working or background-running
    # sub-agent — the live-bug incident was this watchdog force-stopping a
    # correctly-busy agent mid-work. See _check_sub_agents_done.

    # Piece 3 Part E: how long teardown callers WAIT for sub-agent kills.
    # Must exceed SubAgentManager.TEARDOWN_WAIT_BUDGET-relevant inner budgets
    # (ADAPTER_STOP_TIMEOUT 15s + escalation grace) — the outer<inner
    # inversion was the force-drop leak. Expiry stops waiting, never the
    # (shielded) kill.
    SUB_AGENT_TEARDOWN_BUDGET = 25.0

    def _on_loop_done(self, project_id: str, *,
                      session_id: str | None = None):
        """Returns done-callback for the loop asyncio.Task.

        ``session_id`` selects which handle in ``_handles`` this callback
        targets. When omitted, the default session is assumed (single-loop
        back-compat path used by hot-resume).
        """
        session_id = self._resolve_session_id(session_id)
        sk = make_session_key(project_id, session_id)

        def callback(task: asyncio.Task):
            try:
                exc = task.exception()
            except asyncio.CancelledError:
                self._set_last_terminal_event(
                    project_id, session_id, "stopped",
                )
                self._broadcast(project_id, {
                    "type": "agent.status",
                    "project_id": project_id,
                    "status": "idle",
                    "source": "management",
                }, session_id=session_id)
                return
            if exc:
                # Diagnostics (Point 4): the loop task raised. Previously this
                # branch only recorded a terminal event + broadcast `error`,
                # logging NOTHING — so mid-stream exceptions were invisible in
                # daemon.log. Log type/message/traceback BEFORE the existing
                # broadcast. Logging only; broadcast/terminal-event unchanged.
                logger.error(
                    "loop task raised for %s/%s: %s: %s",
                    project_id, session_id, type(exc).__name__, exc,
                    exc_info=exc,
                )
                self._set_last_terminal_event(
                    project_id, session_id, "error", details=str(exc),
                )
                self._broadcast(project_id, {
                    "type": "agent.status",
                    "project_id": project_id,
                    "status": "error",
                    "reason": str(exc),
                    "source": "management",
                }, session_id=session_id)
                return

            handle = self._handles.get(sk)
            if not handle:
                return
            if handle.session.is_stopped():
                self._handles.pop(sk, None)
                self._set_last_terminal_event(
                    project_id, session_id, "stopped",
                )
                self._broadcast(project_id, {
                    "type": "agent.status",
                    "project_id": project_id,
                    "status": "idle",
                }, session_id=session_id)
                return

            # Drain any deferred messages (lifecycle notifications)
            for msg in handle.session.pop_deferred_messages():
                handle.session.append(msg)

            # Check if paused for approval FIRST — don't drain the queue
            # or broadcast idle while a tool call is awaiting user decision.
            # Queued messages will be drained after the approval is resolved.
            if handle.session._paused_for_approval:
                self._broadcast(project_id, {
                    "type": "agent.status",
                    "project_id": project_id,
                    "status": "pending_approval",
                    "source": "management",
                }, session_id=session_id)
                # Notify all clients that the global blocked count changed
                # (this session just entered pending_approval).
                self._broadcast_blocked_count()
                return

            # Drain messages queued while the loop was running (e.g. during
            # the session-end LLM call).  If any exist, append them to the
            # session and hot-resume the loop so they get processed.
            queued = handle.session.pop_queued_messages()
            if queued:
                for q_item in queued:
                    if isinstance(q_item, tuple):
                        q_content, q_nonce = q_item
                    else:
                        q_content, q_nonce = q_item, None
                    q_msg = {
                        "role": "user",
                        "content": q_content,
                        "source": "user",
                    }
                    if q_nonce:
                        q_msg["nonce"] = q_nonce
                    handle.session.append(q_msg)
                asyncio.ensure_future(
                    self._start_loop(project_id, session_id=session_id)
                )
                return

            # Check if sub-agents are still actively running (turn open).
            # Piece 3 Part B: only status=="running" counts as busy for the
            # awaiting poll — "background-running" means the TURN is done
            # (on_completed already fired); the management session must not
            # sit in "waiting" for background work.
            active = self._sub_agent_manager.list_active(
                project_id, session_id=session_id,
            )
            busy = [a for a in active if a.get("status") == "running"]
            if busy:
                self._broadcast(project_id, {
                    "type": "agent.status",
                    "project_id": project_id,
                    "status": "waiting",
                    "source": "management",
                }, session_id=session_id)
                poll_task = asyncio.ensure_future(
                    self._check_sub_agents_done(project_id, session_id=session_id)
                )
                self._idle_poll_tasks[make_session_key(project_id, session_id)] = poll_task
            else:
                # Piece 3 Part B: NO turn-boundary reap. Alive-but-idle
                # agents wait for the next dispatch (live adapters are
                # reused cleanly — REPORT-piece3-prerequisites.md Phase A)
                # and background-running agents keep their work. The ONLY
                # non-teardown reaper is the inactivity eviction
                # (_evict_idle_once), gated on true-idle status.
                self._broadcast(project_id, {
                    "type": "agent.status",
                    "project_id": project_id,
                    "status": "idle",
                    "source": "management",
                }, session_id=session_id)

            # Allow system sleep again if no agents remain.
            self._allow_sleep_if_idle()
        return callback

    async def _reap_idle_sub_agents(self, project_id: str, *,
                                    session_id: str | None = None) -> None:
        """Stop ONLY sub-agents whose status is 'idle' (Piece 3 Part B).

        Used when the owning session is gone mid-poll: idle adapters are
        reaped losslessly (resume restores their context); agents that are
        working or background-running are NEVER reactively killed — they
        finish (or go idle) and the inactivity eviction collects them later.
        """
        try:
            active = self._sub_agent_manager.list_active(
                project_id, session_id=session_id,
            )
        except Exception:
            return
        for a in active:
            if a.get("status") != "idle":
                logger.info(
                    "leaving non-idle sub-agent %s alive for %s/%s "
                    "(status=%s — no reactive kill; Piece 3)",
                    a.get("handle"), project_id, session_id, a.get("status"),
                )
                continue
            try:
                await asyncio.wait_for(
                    self._sub_agent_manager.stop(
                        project_id, a["handle"], session_id=session_id,
                    ),
                    timeout=self._sub_agent_manager.ADAPTER_STOP_TIMEOUT * 2,
                )
            except (asyncio.TimeoutError, Exception):
                logger.warning(
                    "idle reap of %s failed for %s/%s",
                    a.get("handle"), project_id, session_id, exc_info=True,
                )

    async def _check_sub_agents_done(self, project_id: str, *,
                                     session_id: str | None = None) -> None:
        """Poll until no sub-agent has an OPEN TURN, then broadcast idle.

        Piece 3 Part B: this poll is STATUS-ONLY. The 300s busy-watchdog
        force-stop that used to live at the end of this loop is removed — it
        was killing correctly-busy agents mid-work (the signal was right, the
        policy was wrong; a healthy 262s nested run survived it by 37s). No
        fixed clock may kill a working or background-running agent. The poll
        runs until the turn closes, a new loop starts, or the session goes
        away; a long-busy sub-agent just keeps the session in "waiting" —
        honestly.

        ``session_id`` selects the session whose handle gates the poll;
        defaults to the single-loop default session.
        """
        session_id = self._resolve_session_id(session_id)
        sk = make_session_key(project_id, session_id)
        polls = 0
        while True:
            await asyncio.sleep(2.0)
            polls += 1

            handle = self._handles.get(sk)
            if handle is None or handle.session.is_stopped():
                # Session gone mid-wait. Reap ONLY idle adapters — never
                # working/background-running ones (no reactive kills).
                await self._reap_idle_sub_agents(
                    project_id, session_id=session_id,
                )
                return

            # A new loop started — stop polling, loop callback will handle status
            if handle.task is not None and not handle.task.done():
                return

            active = self._sub_agent_manager.list_active(
                project_id, session_id=session_id,
            )
            busy = [a for a in active if a.get("status") == "running"]
            if not busy:
                # Turn(s) finished — NO reap (Part B). Idle/background
                # agents stay alive for reuse; eviction is the only
                # non-teardown reaper.
                self._broadcast(project_id, {
                    "type": "agent.status",
                    "project_id": project_id,
                    "status": "idle",
                    "source": "management",
                }, session_id=session_id)
                return

            if polls % 150 == 0:  # one line every ~5 minutes, not a kill
                logger.info(
                    "sub-agents still busy after ~%ds for %s/%s — waiting "
                    "(no reactive kill; Piece 3)",
                    polls * 2, project_id, session_id,
                )
