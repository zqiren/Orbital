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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from agent_os.agent.context import ContextManager
from agent_os.agent.loop import AgentLoop
from agent_os.agent.prompt_builder import PromptBuilder, PromptContext
from agent_os.agent.providers.openai_compat import LLMProvider
from agent_os.agent.session import Session
from agent_os.agent.tools.registry import ToolRegistry
from agent_os.agent.workspace_files import WorkspaceFileManager, run_session_end_routine
from agent_os.config.provider_registry import ProviderRegistry
from agent_os.daemon_v2.default_skills_installer import install_default_skills
from agent_os.daemon_v2.project_store import project_dir_name as _project_dir_name
from agent_os.daemon_v2.autonomy import AutonomyInterceptor
from agent_os.daemon_v2.models import AgentConfig, detect_os, resolve_api_key

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
    project_dir_name: str = ""


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
        self._handles: dict[str, ProjectHandle] = {}
        self._idle_poll_tasks: dict[str, asyncio.Task] = {}  # project_id -> poll task
        self._state_file: Path = Path.home() / "orbital" / "daemon-state.json"
        self._heartbeat_task: asyncio.Task | None = None
        self._sleep_handle: object | None = None

    # ── Daemon state file (shutdown hardening) ──────────────────────────

    def _write_state(self) -> None:
        """Write daemon-state.json with current agent snapshot (atomic)."""
        state_file = getattr(self, '_state_file', None)
        if state_file is None:
            return
        agents: dict[str, dict] = {}
        for pid, handle in self._handles.items():
            # Only include agents with a running task
            if handle.task is None or handle.task.done():
                continue
            agents[pid] = {
                "status": "running",
                "config_snapshot": getattr(handle, 'config_snapshot', {}),
                "started_at": getattr(handle, 'started_at', ""),
                "shutdown_clean": False,
            }
        state = {
            "version": 1,
            "last_heartbeat": datetime.now(timezone.utc).isoformat(),
            "agents": agents,
        }
        try:
            state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
            tmp.replace(state_file)
        except OSError:
            logger.warning("Failed to write daemon state file")

    def _read_state(self) -> dict | None:
        """Read and parse daemon-state.json. Returns None if missing or corrupt."""
        state_file = getattr(self, '_state_file', None)
        if state_file is None:
            return None
        try:
            text = state_file.read_text(encoding="utf-8")
            return json.loads(text)
        except (OSError, json.JSONDecodeError, ValueError):
            return None

    async def _start_heartbeat(self) -> None:
        """Periodically write daemon state every 30 seconds."""
        try:
            while True:
                self._write_state()
                await asyncio.sleep(30)
        except asyncio.CancelledError:
            return

    def _ensure_heartbeat_running(self) -> None:
        """Start the heartbeat task if not already running."""
        heartbeat = getattr(self, '_heartbeat_task', None)
        if heartbeat is None or heartbeat.done():
            self._heartbeat_task = asyncio.create_task(self._start_heartbeat())

    def _stop_heartbeat_if_idle(self) -> None:
        """Cancel heartbeat if no agents are running."""
        has_running = any(
            h.task is not None and not h.task.done()
            for h in self._handles.values()
        )
        heartbeat = getattr(self, '_heartbeat_task', None)
        if not has_running and heartbeat is not None and not heartbeat.done():
            heartbeat.cancel()
            self._heartbeat_task = None

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
        """Graceful shutdown: stop all agents, cancel heartbeat, write final state."""
        logger.info("AgentManager shutdown: stopping %d agent(s)", len(self._handles))
        self.mark_shutdown_clean()

        # Append shutdown marker to each active session
        for pid, handle in list(self._handles.items()):
            try:
                if hasattr(handle.session, 'append'):
                    handle.session.append({
                        "role": "system",
                        "type": "daemon_shutdown",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
            except Exception:
                logger.warning("Failed to append shutdown marker for project %s", pid)

        # Stop all running agents with a timeout
        stop_tasks = []
        for pid in list(self._handles.keys()):
            stop_tasks.append(self.stop_agent(pid))
        if stop_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*stop_tasks, return_exceptions=True),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.warning("AgentManager shutdown timed out after %.1fs", timeout)

        # Cancel heartbeat
        if self._heartbeat_task is not None and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            self._heartbeat_task = None

        # Write final state
        self._write_state()
        logger.info("AgentManager shutdown complete")

    def mark_shutdown_clean(self) -> None:
        """Mark all agents as cleanly shut down in the state file."""
        state = self._read_state()
        if state is None:
            return
        for agent_entry in state.get("agents", {}).values():
            agent_entry["shutdown_clean"] = True
        state_file = getattr(self, '_state_file', None)
        if state_file is None:
            return
        try:
            state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
            tmp.replace(state_file)
        except OSError:
            logger.warning("Failed to write clean shutdown state")

    async def auto_resume_agents(self) -> None:
        """Resume agents that were running when the daemon last shut down.

        Reads daemon-state.json, rebuilds AgentConfig from project_store
        (not the stale config_snapshot), and restarts each agent.
        """
        state = self._read_state()
        if state is None:
            return

        agents = state.get("agents", {})
        if not agents:
            return

        resumed = 0
        for project_id, agent_state in agents.items():
            if agent_state.get("status") != "running":
                continue

            try:
                # Load fresh project config from store (may have changed)
                project = self._project_store.get_project(project_id)
                if project is None:
                    logger.warning(
                        "auto_resume: project %s no longer exists, skipping",
                        project_id,
                    )
                    continue

                # Build AgentConfig the same way inject_message / start endpoint do
                from agent_os.agent.prompt_builder import Autonomy

                autonomy_str = project.get("autonomy", "hands_off")
                try:
                    autonomy = Autonomy(autonomy_str)
                except ValueError:
                    autonomy = Autonomy.HANDS_OFF

                # Auto-detect sub-agents if not explicitly configured
                enabled_sub_agents = project.get("enabled_sub_agents", None)
                if enabled_sub_agents is None and self._setup_engine is not None:
                    available = self._setup_engine.check_all()
                    enabled_sub_agents = [
                        a.slug for a in available
                        if a.installed and a.slug != "built-in"
                    ]

                # Use global settings as fallback for missing project-level LLM config
                global_settings = self._settings_store.get() if self._settings_store else None
                cred_key = self._credential_store.get_api_key() if self._credential_store else None
                api_key = (
                    project.get("api_key")
                    or cred_key
                    or (global_settings.llm.api_key if global_settings else None)
                    or ""
                )
                base_url = project.get("base_url") or (
                    global_settings.llm.base_url if global_settings else None
                )
                model = (
                    project.get("model")
                    or (global_settings.llm.model if global_settings else None)
                    or ""
                )

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
                    enabled_sub_agents=enabled_sub_agents or [],
                    agent_credentials=project.get("agent_credentials", {}),
                    network_extra_domains=project.get("network_extra_domains", []),
                    is_scratch=project.get("is_scratch", False),
                    agent_name=project.get("agent_name", project.get("name", "")),
                    global_preferences_path="",
                    budget_limit_usd=project.get("budget_limit_usd"),
                    budget_action=project.get("budget_action", "ask"),
                )

                await self.start_agent(
                    project_id, config,
                    initial_message=None,
                    trigger_source="auto_resume",
                )

                # If the previous session was interrupted (not a clean shutdown),
                # inject a system warning so the agent knows to verify state.
                if not agent_state.get("shutdown_clean", False):
                    handle = self._handles.get(project_id)
                    if handle and hasattr(handle.session, "append"):
                        handle.session.append({
                            "role": "system",
                            "content": (
                                "NOTICE: The daemon was interrupted unexpectedly. "
                                "Your previous session may have had in-flight "
                                "operations that did not complete. Verify current "
                                "state before proceeding."
                            ),
                        })

                resumed += 1
                logger.info("auto_resume: resumed agent for project %s", project_id)

            except Exception:
                logger.exception(
                    "auto_resume: failed to resume agent for project %s",
                    project_id,
                )

        logger.info("Resumed %d agent(s) from previous session", resumed)

    def _record_approval_decision(self, project_id: str, tool_name: str,
                                   tool_args: dict, decision: str,
                                   deny_reason: str | None = None) -> None:
        """Append approval/denial record to {workspace}/orbital/approval_history.jsonl."""
        handle = self._handles.get(project_id)
        if handle is None:
            return
        # Derive workspace from config snapshot
        workspace = handle.config_snapshot.get('workspace')
        if not workspace:
            return
        dir_name = handle.project_dir_name
        if not dir_name:
            return
        history_dir = os.path.join(workspace, "orbital", dir_name)
        os.makedirs(history_dir, exist_ok=True)
        history_file = os.path.join(history_dir, "approval_history.jsonl")
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
                          initial_nonce: str | None = None) -> None:
        """Wire all components and start the loop."""
        # Guard: check if already running
        handle = self._handles.get(project_id)
        if handle is not None:
            if handle.task is not None and not handle.task.done():
                raise ValueError("Agent already running for this project")
            # Stale handle: clean up
            del self._handles[project_id]

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
                            capabilities=fb_info.capabilities)
            )

        # 1b. Utility provider
        if config.utility_model:
            utility_provider = LLMProvider(config.utility_model, config.api_key, config.base_url, sdk=config.sdk)
        else:
            utility_provider = provider

        # 2. Tool registry
        dir_name = _project_dir_name(config.project_name, project_id)
        registry = ToolRegistry(user_credential_store=self._user_credential_store)
        self._register_tools(registry, config, project_id,
                             vision_enabled=model_info.capabilities.vision,
                             project_dir_name=dir_name)

        # 3. Prompt builder
        prompt_builder = PromptBuilder(workspace=config.workspace)

        # 4. Build prompt context
        # Prefer enabled_sub_agents (new) with registry lookup, fallback to enabled_agents (legacy)
        # Auto-detect from registry if neither is set
        sub_agent_slugs = config.enabled_sub_agents or config.enabled_agents
        if not sub_agent_slugs and self._setup_engine is not None:
            available = self._setup_engine.check_all()
            sub_agent_slugs = [
                a.slug for a in available
                if a.installed and a.slug != "built-in"
            ]
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
            project_dir_name=dir_name,
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

        # 5. Session
        sanitized = _sanitize_project_name(config.project_name)
        session_id = f"{sanitized}_{uuid4().hex[:8]}"
        session = Session.new(session_id, config.workspace, project_dir_name=dir_name)

        # 6-7. Observers
        session.on_append = self._on_message(project_id)
        session.on_stream = self._on_stream(project_id)

        # 8. Workspace files
        workspace_files = WorkspaceFileManager(config.workspace, project_dir_name=dir_name)
        workspace_files.ensure_dir()

        # 9. Context manager
        context_manager = ContextManager(
            session, prompt_builder, base_prompt_context,
            model_context_limit=model_info.context_window,
            workspace_files=workspace_files,
            sub_agent_provider=lambda: self._sub_agent_manager.list_active(project_id),
        )

        # 10. Interceptor
        interceptor = AutonomyInterceptor(
            config.autonomy, self._ws, project_id,
            user_credential_store=self._user_credential_store,
        )

        # 11. Session-end callback
        async def session_end_callback():
            await run_session_end_routine(
                session=session,
                provider=provider,
                workspace_files=workspace_files,
                utility_provider=utility_provider,
            )

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
        )

        # 12. Store handle
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
            },
            started_at=datetime.now(timezone.utc).isoformat(),
            project_dir_name=dir_name,
        )
        self._handles[project_id] = project_handle

        # 11. Start loop task
        task = asyncio.create_task(loop.run(initial_message, initial_nonce=initial_nonce))
        task.add_done_callback(self._on_loop_done(project_id))
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
        self._ws.broadcast(project_id, status_event)

        # 15. Daemon state checkpoint + heartbeat + sleep prevention
        self._ensure_heartbeat_running()
        self._prevent_sleep_if_needed()
        self._write_state()

    def _register_tools(self, registry: ToolRegistry, config: AgentConfig,
                        project_id: str = "", vision_enabled: bool = False,
                        project_dir_name: str = "") -> None:
        """Register all tools. Imports are deferred to avoid circular deps at module level."""
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
                project_dir_name=project_dir_name,
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
            registry.register(AgentMessageTool(sub_agent_manager=self._sub_agent_manager, project_id=project_id, depth=0))
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
                    project_dir_name=project_dir_name,
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

    async def inject_system_message(self, project_id: str, content: str) -> str:
        """Inject a system message into the management agent's session.

        Used by the lifecycle observer for sub-agent state notifications.
        If the management agent is idle, appends directly and starts a new loop.
        If the loop is running, defers the message for safe insertion after the
        current tool batch completes (avoids breaking assistant→tool sequences).
        """
        handle = self._handles.get(project_id)
        if handle is None:
            return "no_session"

        # If loop is idle, append directly (safe — no pending tool calls) and wake
        if handle.task is None or handle.task.done():
            handle.session.append({
                "role": "system",
                "content": content,
                "source": "daemon",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            await self._start_loop(project_id)
            return "delivered"

        # Loop is running — defer for safe insertion after tool batch
        handle.session.defer_message(content, role="system", source="daemon")
        return "deferred"

    async def inject_message(self, project_id: str, content: str, *, nonce: str | None = None) -> "str | dict":
        """Inject a message. Four cases:
        1. Loop running -> queue (returns "queued")
        1b. Loop paused for approval -> auto-deny pending, deliver, resume
            (returns dict with status="delivered", approval_dismissed=True,
            dismissed_tool_call_id=<id>)
        2. Loop idle + session alive -> hot resume (returns "delivered")
        3. No session -> auto-start agent with message (returns "started")
        """
        handle = self._handles.get(project_id)
        if handle is None:
            # Case 3: no handle — auto-start from project store
            logger.info("inject_message(%s): no handle, auto-starting agent", project_id)
            from agent_os.agent.prompt_builder import Autonomy

            project = self._project_store.get_project(project_id)
            if project is None:
                raise KeyError(f"No project found for '{project_id}'")

            autonomy_str = project.get("autonomy", "hands_off")
            try:
                autonomy = Autonomy(autonomy_str)
            except ValueError:
                autonomy = Autonomy.HANDS_OFF

            # Auto-detect sub-agents if not explicitly configured
            enabled_sub_agents = project.get("enabled_sub_agents", None)
            if enabled_sub_agents is None and self._setup_engine is not None:
                available = self._setup_engine.check_all()
                enabled_sub_agents = [
                    a.slug for a in available
                    if a.installed and a.slug != "built-in"
                ]

            # Use global settings as fallback for missing project-level LLM config
            global_settings = self._settings_store.get() if self._settings_store else None
            cred_key = self._credential_store.get_api_key() if self._credential_store else None
            api_key = (project.get("api_key")
                       or cred_key
                       or (global_settings.llm.api_key if global_settings else None)
                       or "")
            base_url = project.get("base_url") or (global_settings.llm.base_url if global_settings else None)
            model = project.get("model") or (global_settings.llm.model if global_settings else None) or ""

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
                enabled_sub_agents=enabled_sub_agents or [],
                budget_limit_usd=project.get("budget_limit_usd"),
                budget_action=project.get("budget_action", "ask"),
            )
            await self.start_agent(project_id, config, initial_message=content,
                                  initial_nonce=nonce)
            return "started"

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
                self._ws.broadcast(project_id, {
                    "type": "approval.resolved",
                    "project_id": project_id,
                    "tool_call_id": dismissed_tc_id,
                    "resolution": "denied",
                })

            # 10. Restart the loop to process the new user message.
            await self._start_loop(project_id)
            return {
                "status": "delivered",
                "approval_dismissed": True,
                "dismissed_tool_call_id": dismissed_tc_id,
            }

        # Case 2: loop idle — append message and hot-resume
        if handle is not None and handle.session.is_stopped():
            del self._handles[project_id]
            handle = None

        if handle is None:
            # Auto-start a fresh agent (stale stopped session was cleaned up).
            # Re-derive config from the project store — this path is reached
            # when the original handle existed but the session was stopped,
            # so the variables from the early handle-is-None block are absent.
            from agent_os.agent.prompt_builder import Autonomy as _Autonomy

            proj = self._project_store.get_project(project_id)
            if proj is None:
                raise KeyError(f"No project found for '{project_id}'")

            autonomy_str = proj.get("autonomy", "hands_off")
            try:
                _autonomy = _Autonomy(autonomy_str)
            except ValueError:
                _autonomy = _Autonomy.HANDS_OFF

            _enabled_sub = proj.get("enabled_sub_agents", None)
            if _enabled_sub is None and self._setup_engine is not None:
                available = self._setup_engine.check_all()
                _enabled_sub = [
                    a.slug for a in available
                    if a.installed and a.slug != "built-in"
                ]

            global_settings = self._settings_store.get() if self._settings_store else None
            cred_key = self._credential_store.get_api_key() if self._credential_store else None
            _api_key = (proj.get("api_key")
                        or cred_key
                        or (global_settings.llm.api_key if global_settings else None)
                        or "")
            _base_url = proj.get("base_url") or (global_settings.llm.base_url if global_settings else None)
            _model = proj.get("model") or (global_settings.llm.model if global_settings else None) or ""

            config = AgentConfig(
                workspace=proj["workspace"],
                model=_model,
                api_key=_api_key,
                base_url=_base_url,
                autonomy=_autonomy,
                sdk=proj.get("sdk", "openai"),
                provider=proj.get("provider", "custom"),
                project_name=proj.get("name", ""),
                project_instructions=proj.get("instructions", ""),
                enabled_sub_agents=_enabled_sub or [],
                budget_limit_usd=proj.get("budget_limit_usd"),
                budget_action=proj.get("budget_action", "ask"),
            )
            await self.start_agent(project_id, config, initial_message=content,
                                  initial_nonce=nonce)
            return "started"

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
        await self._start_loop(project_id)
        return "delivered"

    def is_running(self, project_id: str) -> bool:
        """Return True if an agent loop is actively running for this project."""
        handle = self._handles.get(project_id)
        if handle is None:
            return False
        return handle.task is not None and not handle.task.done()

    def get_run_status(self, project_id: str) -> str:
        """Return the current runtime status for a project.

        Returns one of: 'running', 'pending_approval', 'waiting', 'idle',
        'stopped', 'error'.
        """
        handle = self._handles.get(project_id)
        if handle is None:
            return "idle"
        if handle.session.is_stopped():
            return "stopped"
        # Check approval pause BEFORE task.done() — during approval the
        # task is still alive (blocked inside the interceptor's wait).
        if handle.session._paused_for_approval:
            return "pending_approval"
        if handle.task is not None and not handle.task.done():
            return "running"
        # Check for sub-agents
        poll_task = self._idle_poll_tasks.get(project_id)
        if poll_task is not None and not poll_task.done():
            return "waiting"
        return "idle"

    def update_autonomy(self, project_id: str, preset) -> bool:
        """Push a new autonomy preset to a running agent's interceptor.

        Returns True if the running agent was updated, False if no agent
        is running (disk-only update is sufficient in that case).
        """
        handle = self._handles.get(project_id)
        if handle is None:
            return False
        handle.interceptor.update_preset(preset)
        handle.session.append_system(
            f"[autonomy changed to {preset.value}]"
        )
        # Also propagate to sub-agent transports (SDK filtering)
        if self._sub_agent_manager is not None:
            self._sub_agent_manager.update_sub_agent_autonomy(project_id, preset)
        return True

    def get_pending_approval(self, project_id: str) -> dict | None:
        """Return the pending approval payload for a project, or None.

        Used by the REST recovery endpoint so mobile clients can fetch
        approval card data when they miss the WebSocket event.
        """
        handle = self._handles.get(project_id)
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
                      approve_all: bool = False) -> None:
        """Approve a pending tool call and execute it."""
        handle = self._handles.get(project_id)
        if handle is None:
            raise KeyError(f"No active session for project '{project_id}'")

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
            await self._start_loop(project_id)
            return

        handle.interceptor.record_approval(pending["tool_name"], pending["tool_args"])
        self._record_approval_decision(project_id, pending["tool_name"], pending["tool_args"], "approved")

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

        handle.session.resume()
        await self._start_loop(project_id)

    async def deny(self, project_id: str, tool_call_id: str, reason: str) -> None:
        """Deny a pending tool call."""
        handle = self._handles.get(project_id)
        if handle is None:
            raise KeyError(f"No active session for project '{project_id}'")

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
            )

        # Guard: don't append a second result if one already exists
        if not handle.session.has_result_for(tool_call_id):
            handle.session.append_tool_result(
                tool_call_id,
                f"DENIED by user. Reason: {reason}",
            )

        handle.interceptor.remove_pending(tool_call_id)
        handle.session.resume()
        await self._start_loop(project_id)

    async def new_session(self, project_id: str) -> dict:
        """Archive the current session and create a fresh one.

        Preserves the project handle (loop, provider, registry, context_manager,
        interceptor) but swaps the session object.  Layer 1 workspace files
        (PROJECT_STATE.md, DECISIONS.md, …) are untouched.
        """
        handle = self._handles.get(project_id)
        if handle is None:
            return {"status": "no_active_session"}

        # 1. Stop loop if running
        if handle.task is not None and not handle.task.done():
            handle.session.stop()
            try:
                await asyncio.wait_for(asyncio.shield(handle.task), timeout=10.0)
            except (asyncio.TimeoutError, Exception):
                logger.warning("new_session(%s): loop did not stop gracefully", project_id)

        # 2. Cancel idle poll / sub-agents
        poll_task = self._idle_poll_tasks.pop(project_id, None)
        if poll_task and not poll_task.done():
            poll_task.cancel()
        await self._sub_agent_manager.stop_all(project_id)

        # 3. Pre-flush: run session-end routine so workspace files are updated
        workspace = handle.config_snapshot.get("workspace", "")
        try:
            workspace_files = WorkspaceFileManager(
                workspace, project_dir_name=handle.project_dir_name,
            )
            await asyncio.wait_for(
                run_session_end_routine(
                    session=handle.session,
                    provider=handle.provider,
                    workspace_files=workspace_files,
                ),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.warning("new_session(%s): pre-flush LLM call timed out, proceeding", project_id)
        except Exception:
            logger.warning("new_session(%s): pre-flush failed, proceeding", project_id, exc_info=True)

        # 4. Save old session info
        old_session_id = handle.session.session_id

        # 5. Create new session
        project = self._project_store.get_project(project_id)
        project_name = (project or {}).get("name", "")
        sanitized = _sanitize_project_name(project_name)
        new_session_id = f"{sanitized}_{uuid4().hex[:8]}"
        new_session = Session.new(
            new_session_id, workspace,
            project_dir_name=handle.project_dir_name,
        )

        # 6. Wire observers
        new_session.on_append = self._on_message(project_id)
        new_session.on_stream = self._on_stream(project_id)

        # 7. Reset context_manager so it reads from the new empty session
        cm = handle.context_manager
        if hasattr(cm, '_session'):
            cm._session = new_session
        if hasattr(cm, '_cold_resume_injected'):
            cm._cold_resume_injected = False
        if hasattr(cm, '_recovery_injected'):
            cm._recovery_injected = False
        if hasattr(cm, '_window_factor'):
            cm._window_factor = 1.0
        if hasattr(cm, '_last_usage_pct'):
            cm._last_usage_pct = 0.0

        # 8. Swap session in handle
        handle.session = new_session
        handle.task = None

        # 10. Broadcast new_session event, then idle so frontend status
        #     doesn't stay stuck (enables repeat /new invocations).
        self._ws.broadcast(project_id, {
            "type": "agent.status",
            "project_id": project_id,
            "status": "new_session",
            "source": "management",
        })
        self._ws.broadcast(project_id, {
            "type": "agent.status",
            "project_id": project_id,
            "status": "idle",
            "source": "management",
        })

        logger.info(
            "new_session(%s): archived %s, created %s",
            project_id, old_session_id, new_session_id,
        )
        return {
            "status": "ok",
            "old_session_id": old_session_id,
            "new_session_id": new_session_id,
        }

    async def stop_agent(self, project_id: str) -> None:
        """Stop agent and clean up."""
        handle = self._handles.get(project_id)
        if handle is None:
            raise KeyError(f"No active session for project '{project_id}'")

        # Cancel idle poll task if running
        poll_task = self._idle_poll_tasks.pop(project_id, None)
        if poll_task and not poll_task.done():
            poll_task.cancel()

        # Stop sub-agents (uses stopping flag to reject concurrent starts)
        await self._sub_agent_manager.stop_all(project_id)

        # Close browser pages for this project
        if self._browser_manager is not None:
            try:
                await self._browser_manager.close_project_pages(project_id)
            except Exception:
                pass

        # Stop platform sandbox processes
        if self._platform_provider is not None:
            await self._platform_provider.stop_process(project_id)

        handle.session.stop()

        # Wait for loop task
        if handle.task is not None and not handle.task.done():
            try:
                await asyncio.wait_for(asyncio.shield(handle.task), timeout=10.0)
            except (asyncio.TimeoutError, Exception):
                pass

        self._ws.broadcast(project_id, {
            "type": "agent.status",
            "project_id": project_id,
            "status": "stopped",
            "source": "management",
        })

        self._handles.pop(project_id, None)

        # Update daemon state, stop heartbeat, and allow sleep if no agents remain
        self._write_state()
        self._stop_heartbeat_if_idle()
        self._allow_sleep_if_idle()

    def get_session(self, project_id: str):
        """Return session for a project, or None."""
        handle = self._handles.get(project_id)
        return handle.session if handle else None

    async def _start_loop(self, project_id: str) -> None:
        """Start a new loop.run() on existing session (hot resume)."""
        handle = self._handles.get(project_id)
        if handle is None:
            return

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
        task.add_done_callback(self._on_loop_done(project_id))
        handle.task = task

        self._ws.broadcast(project_id, {
            "type": "agent.status",
            "project_id": project_id,
            "status": "running",
            "source": "management",
        })

    def _on_message(self, project_id: str):
        """Returns callback for session.on_append."""
        def callback(msg):
            self._activity_translator.on_message(msg, project_id)
        return callback

    def _on_stream(self, project_id: str):
        """Returns callback for session.on_stream."""
        def callback(chunk):
            self._activity_translator.on_stream_chunk(chunk, project_id, "management")
        return callback

    _MAX_IDLE_POLLS = 150  # 5 minutes at 2s intervals

    def _on_loop_done(self, project_id: str):
        """Returns done-callback for the loop asyncio.Task."""
        def callback(task: asyncio.Task):
            try:
                exc = task.exception()
            except asyncio.CancelledError:
                self._ws.broadcast(project_id, {
                    "type": "agent.status",
                    "project_id": project_id,
                    "status": "stopped",
                    "source": "management",
                })
                return
            if exc:
                self._ws.broadcast(project_id, {
                    "type": "agent.status",
                    "project_id": project_id,
                    "status": "error",
                    "reason": str(exc),
                    "source": "management",
                })
                return

            handle = self._handles.get(project_id)
            if not handle:
                return
            if handle.session.is_stopped():
                self._handles.pop(project_id, None)
                self._ws.broadcast(project_id, {
                    "type": "agent.status",
                    "project_id": project_id,
                    "status": "stopped",
                })
                return

            # Drain any deferred messages (lifecycle notifications)
            for msg in handle.session.pop_deferred_messages():
                handle.session.append(msg)

            # Check if paused for approval FIRST — don't drain the queue
            # or broadcast idle while a tool call is awaiting user decision.
            # Queued messages will be drained after the approval is resolved.
            if handle.session._paused_for_approval:
                self._ws.broadcast(project_id, {
                    "type": "agent.status",
                    "project_id": project_id,
                    "status": "pending_approval",
                    "source": "management",
                })
                self._write_state()
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
                asyncio.ensure_future(self._start_loop(project_id))
                return

            # Check if sub-agents are still actively running (not just alive)
            active = self._sub_agent_manager.list_active(project_id)
            busy = [a for a in active if a.get("status") != "idle"]
            if busy:
                self._ws.broadcast(project_id, {
                    "type": "agent.status",
                    "project_id": project_id,
                    "status": "waiting",
                    "source": "management",
                })
                poll_task = asyncio.ensure_future(
                    self._check_sub_agents_done(project_id)
                )
                self._idle_poll_tasks[project_id] = poll_task
            else:
                self._ws.broadcast(project_id, {
                    "type": "agent.status",
                    "project_id": project_id,
                    "status": "idle",
                    "source": "management",
                })

            # Update daemon state after loop completes
            self._write_state()
            self._stop_heartbeat_if_idle()
            self._allow_sleep_if_idle()
        return callback

    async def _check_sub_agents_done(self, project_id: str) -> None:
        """Poll until sub-agents finish, then broadcast idle."""
        for _ in range(self._MAX_IDLE_POLLS):
            await asyncio.sleep(2.0)

            handle = self._handles.get(project_id)
            if handle is None or handle.session.is_stopped():
                return

            # A new loop started — stop polling, loop callback will handle status
            if handle.task is not None and not handle.task.done():
                return

            active = self._sub_agent_manager.list_active(project_id)
            busy = [a for a in active if a.get("status") != "idle"]
            if not busy:
                self._ws.broadcast(project_id, {
                    "type": "agent.status",
                    "project_id": project_id,
                    "status": "idle",
                    "source": "management",
                })
                return

        # Max polls exceeded — force idle, something is stuck
        logger.warning("Sub-agent poll timeout for project %s, forcing idle", project_id)
        self._ws.broadcast(project_id, {
            "type": "agent.status",
            "project_id": project_id,
            "status": "idle",
            "source": "management",
        })
