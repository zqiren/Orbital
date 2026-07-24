# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""FastAPI app factory for Agent OS daemon.

v2 API only. JSON-based project storage.
"""

import asyncio
import json
import logging
import logging.handlers
import os
import sys

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from agent_os.api.ws import WebSocketManager
from agent_os.api.middleware import RelayRedactionMiddleware
from agent_os.api.routes import agents_v2
from agent_os.api.routes import files_v2
from agent_os.api.routes import pairing as pairing_routes
from agent_os.api.routes import platform as platform_routes
from agent_os.api.routes import settings as settings_routes
from agent_os.daemon_v2.activity_translator import ActivityTranslator
from agent_os.daemon_v2.agent_manager import AgentManager
from agent_os.daemon_v2.browser_manager import BrowserManager
from agent_os.daemon_v2.process_manager import ProcessManager
from agent_os.daemon_v2.project_store import ProjectStore
from agent_os.daemon_v2.credential_store import ApiKeyStore, UserCredentialStore
from agent_os.daemon_v2.settings_store import SettingsStore
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver
from agent_os.daemon_v2.fanout import FanoutRegistry
from agent_os.agents.registry import AgentRegistry
from agent_os.agents.setup_engine import SetupEngine
from agent_os.daemon_v2.trigger_manager import TriggerManager
from agent_os.platform import create_platform_provider
from agent_os.utils.pid_file import acquire_pid_file, release_pid_file

logger = logging.getLogger(__name__)


SCRATCH_PROJECT_GOALS = """# Quick Tasks Assistant

## Mission
You are a general-purpose assistant for quick tasks, questions, and simple work.
Help the user immediately without requiring setup or clarification unless truly needed.

## Scope
- Answer questions, draft text, brainstorm, calculate, research
- Write small scripts or snippets directly into the workspace (not under orbital/)
- Any task that doesn't require sustained multi-session effort

## Cross-project reference lens
You may have read access to the user's other project workspaces (the scope chip
in the chat header shows which — All projects, specific projects, or just this
workspace). When in scope:
- Read, search (grep), and glob freely across those projects to answer
  "how did I do this before?" style questions — they are reference material.
- You can only WRITE inside this Quick Tasks workspace. Never attempt to modify
  another project; if work belongs there, hand it off (create a project or
  @mention its agent).
- When a finding comes from another project, CITE the source project (the search
  results are labeled `[project: <name>]`) so the user knows where it came from.
- Another project's `orbital/` internals and `.git/` are intentionally invisible
  — you see work product, not agent machinery.

## Rules
- Don't ask unnecessary clarifying questions for simple tasks. Just do it.
- If a task is growing complex (multi-file project, ongoing work, needs persistence),
  suggest: "This looks like it could be its own project. Want me to create one?"
- Keep workspace state maintenance light — only update PROJECT_STATE.md for
  substantial ongoing work, not one-off questions.
"""


def _write_scratch_project_goals(scratch_workspace: str) -> None:
    """Write the canned SCRATCH_PROJECT_GOALS into the new orbital/ layout."""
    from agent_os.agent.project_paths import ProjectPaths
    pp = ProjectPaths(scratch_workspace)
    os.makedirs(pp.instructions_dir, exist_ok=True)
    with open(pp.project_goals, "w", encoding="utf-8") as f:
        f.write(SCRATCH_PROJECT_GOALS)


def _ensure_scratch_project(project_store, settings_store, data_dir):
    """Auto-create the Quick Tasks scratch project if none exists."""
    if project_store.find_scratch_project() is not None:
        return
    settings = settings_store.get()
    scratch_workspace = getattr(settings, 'scratch_workspace', None) or os.path.join(data_dir, "scratch")
    os.makedirs(scratch_workspace, exist_ok=True)
    project_store.create_project({
        "name": "Quick Tasks",
        "agent_name": "Assistant",
        "workspace": scratch_workspace,
        "is_scratch": True,
        "autonomy": "hands_off",
        "model": "",
        "api_key": "",
    })
    _write_scratch_project_goals(scratch_workspace)


def _configure_file_logging(data_dir: str) -> None:
    """Attach a rotating file handler to the root logger.

    The daemon writes to ``{data_dir}/logs/daemon.log`` so that exceptions
    raised inside the agent loop, dispatch path, or background tasks leave a
    durable trace. Without this, uvicorn's stderr only reaches whichever
    terminal launched the process — and on Windows, when the daemon is
    started detached, that stream is lost. Idempotent: skips installation
    if a daemon FileHandler is already attached (e.g. on factory reload).
    """
    log_dir = os.path.join(data_dir, "logs")
    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError:
        # If we can't create the log dir, fall back to stderr only.
        return

    log_path = os.path.join(log_dir, "daemon.log")
    root = logging.getLogger()

    # Avoid duplicate handlers on factory reload
    for h in root.handlers:
        if (isinstance(h, logging.handlers.RotatingFileHandler)
                and getattr(h, "_orbital_daemon_log", False)):
            return

    handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,  # 5 MB per file
        backupCount=3,
        encoding="utf-8",
    )
    handler._orbital_daemon_log = True  # type: ignore[attr-defined]
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    ))
    root.addHandler(handler)
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    logger.info("Daemon file logging enabled at %s", log_path)


def create_app(data_dir: str | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Orbital", version="2.0")

    # 0. File logging — install before anything else so subsequent setup
    # errors land in the daemon log instead of disappearing to a detached
    # stderr.
    _configure_file_logging(data_dir or "orbital-data")

    # 0. Singleton enforcement via PID file
    acquire_pid_file()

    # 0. CORS middleware (required for React dev server on different port)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Total-Count"],
    )

    # 0a. Relay redaction middleware (strips api_key from relay-proxied responses)
    app.add_middleware(RelayRedactionMiddleware)

    # 1. Project store + credential store
    store_dir = data_dir or "orbital-data"
    project_store = ProjectStore(data_dir=store_dir)
    credential_store = ApiKeyStore()
    user_credential_store = UserCredentialStore(
        meta_path=os.path.join(store_dir, "credential-meta.json")
    )
    settings_store = SettingsStore(data_dir=store_dir, credential_store=credential_store)

    # One-time migration: move api_key from settings.json to keychain
    _legacy = settings_store.get()
    if _legacy.llm.api_key:
        try:
            credential_store.set_api_key(_legacy.llm.api_key)
        except Exception as e:
            logger.warning("Credential migration failed, keeping key in settings.json: %s", e)
        else:
            _legacy.llm.api_key = None
            settings_store.update(_legacy)

    # Auto-create scratch project
    _ensure_scratch_project(project_store, settings_store, store_dir)

    # One-time cleanup: sweep orphaned `*.jsonl.lock` sentinels left behind
    # by the legacy SentinelFileLock strategy or by crashed processes. The
    # cleanup probes each sentinel with a non-blocking flock — files held
    # by a live process are skipped, never deleted.
    try:
        from agent_os.utils.file_lock import cleanup_orphaned_sentinels
        workspaces = [
            (p.get("workspace") or "") for p in project_store.list_projects()
        ]
        removed = cleanup_orphaned_sentinels(workspaces)
        if removed:
            logger.info(
                "Startup cleanup removed %d orphaned session lock sentinel(s).",
                removed,
            )
    except Exception:
        logger.exception("Orphaned session sentinel cleanup failed (non-fatal)")

    # 2. WebSocket manager
    ws_manager = WebSocketManager()

    # Pin the broadcast drain to the main loop at boot. Without this,
    # WebSocketManager._ensure_drain() adopts whatever loop happens to run
    # the first-ever broadcast() — which can be an off-thread caller (e.g.
    # NetworkProxy's dedicated "orbital-network-loop" thread) racing to get
    # there first. Establishing it here deterministically wins that race.
    @app.on_event("startup")
    async def _start_ws_drain():
        ws_manager._ensure_drain()

    # 3. Activity translator
    activity_translator = ActivityTranslator(ws_manager)

    # 4. Process manager
    process_manager = ProcessManager(ws_manager, activity_translator)

    # 5. Platform provider
    platform_provider = create_platform_provider(
        on_network_blocked=lambda pid, domain, method: activity_translator.on_network_blocked(pid, domain, method)
    )

    # 5b. Agent registry + setup engine
    registry = AgentRegistry()
    # In PyInstaller bundles, __file__ points to a temp extraction dir but
    # data files live under sys._MEIPASS.  Try the frozen path first.
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        manifests_dir = os.path.join(sys._MEIPASS, "agent_os", "agents", "manifests")
    else:
        manifests_dir = os.path.join(os.path.dirname(__file__), "..", "agents", "manifests")
    registry.load_directory(manifests_dir)

    # 5b. Daemon-level sub-agent config (model/effort/permission-mode overrides).
    # Stored at ``~/.orbital/sub_agent_config.json`` so it follows the user
    # account, not any individual project. Read at sub-agent dispatch time
    # via SetupEngine.get_adapter_config and merged into the CLI invocation.
    from agent_os.daemon_v2.sub_agent_config_store import SubAgentConfigStore
    sub_agent_config_path = os.path.join(
        os.path.expanduser("~"), ".orbital", "sub_agent_config.json"
    )
    sub_agent_config_store = SubAgentConfigStore(sub_agent_config_path)

    setup_engine = SetupEngine(
        registry, sub_agent_config_store=sub_agent_config_store,
    )

    # 5c. Provider registry (model capabilities, context windows, max output)
    from agent_os.config.provider_registry import ProviderRegistry
    provider_registry = ProviderRegistry()

    # 5d. Browser manager (lazy — browser launches on first tool call)
    browser_manager = BrowserManager()

    # 5d. Sub-agent manager
    sub_agent_manager = SubAgentManager(
        process_manager=process_manager,
        registry=registry,
        setup_engine=setup_engine,
        platform_provider=platform_provider,
        project_store=project_store,
        ws_manager=ws_manager,
    )

    # 5e. Connector manager (spec 011) — MCP-client connector core. Google
    # OAuth client config comes from settings (BYO-client until first-party
    # registration lands); connect() cleanly 400s while unconfigured.
    from agent_os.connectors import ConnectorManager, load_catalog
    from agent_os.connectors.mcp_client import streamable_http_opener
    from agent_os.connectors.oauth import GOOGLE_ENDPOINTS, OAuthClientConfig

    def _google_oauth_provider(auth_provider: str) -> OAuthClientConfig | None:
        if auth_provider != "google":
            return None
        s = settings_store.get()
        cid = s.connector_google_client_id
        secret = s.connector_google_client_secret
        if cid and secret:
            return OAuthClientConfig(cid, secret, GOOGLE_ENDPOINTS)
        return None

    connector_manager = ConnectorManager(
        catalog=load_catalog(),
        credential_store=user_credential_store,
        data_dir=store_dir,
        oauth_client_provider=_google_oauth_provider,
        session_opener=streamable_http_opener,
    )

    # 5f. Calendar hub (spec 011 §0.4/§2, Task G1) — normalizes macOS EventKit
    # and a connected Google-Calendar connector into one event feed with a
    # project-linkage store. Sources construct lazily (EventKit touches the
    # store — and triggers the TCC prompt — only on first real access, never
    # here) and degrade to unavailable/empty; the hub never raises.
    from agent_os.calendar_hub import CalendarHub, Linkage
    from agent_os.calendar_hub.sources import EventKitSource, McpCalendarSource
    from agent_os.calendar_hub.sources.automation_source import AutomationSource
    from agent_os.calendar_hub.sources.memory_source import MemorySource

    calendar_hub = CalendarHub(
        sources=[
            EventKitSource(),
            McpCalendarSource(connector_manager, "google-calendar"),
            # Native, standalone-calendar sources (spec §7.2, Task 6) — every
            # project's due-dated PROJECT_STATE entries and enabled schedule
            # triggers, computed at request time with no connector required.
            MemorySource(project_store),
            AutomationSource(project_store),
        ],
        linkage=Linkage(store_dir),
    )

    # 6. Agent manager
    agent_manager = AgentManager(
        connector_manager=connector_manager,
        project_store=project_store,
        ws_manager=ws_manager,
        sub_agent_manager=sub_agent_manager,
        activity_translator=activity_translator,
        process_manager=process_manager,
        platform_provider=platform_provider,
        registry=registry,
        setup_engine=setup_engine,
        settings_store=settings_store,
        credential_store=credential_store,
        browser_manager=browser_manager,
        user_credential_store=user_credential_store,
        provider_registry=provider_registry,
        calendar_hub=calendar_hub,
    )

    # 6a. Lifecycle observer (wired after agent_manager exists)
    lifecycle_observer = LifecycleObserver(agent_manager, ws_manager)
    process_manager._lifecycle = lifecycle_observer
    sub_agent_manager._lifecycle_observer = lifecycle_observer
    # Resume persistence (TASK-resume-persistence): dispatch reads the
    # per-(SessionKey, handle) thread records off the management session.
    # (get_session's session_id is keyword-only; the resolver contract is
    # positional (project_id, session_id).)
    sub_agent_manager._session_resolver = (
        lambda pid, sid: agent_manager.get_session(pid, session_id=sid)
    )

    # 6a2. Fanout registry (spec 009, Task 2/6): join/gather core for parallel
    # native-worker dispatch. Wired post-construction, mirrors lifecycle_observer
    # above. stop_worker=sub_agent_manager.stop is passed directly (a bound
    # method) — FanoutRegistry always calls it with session_id= as a keyword,
    # matching that method's keyword-only session_id parameter.
    fanout_registry = FanoutRegistry(
        inject=agent_manager.inject_system_message,
        broadcast=ws_manager.broadcast,
        stop_worker=sub_agent_manager.stop,
    )
    sub_agent_manager._fanout_registry = fanout_registry
    lifecycle_observer.fanout_registry = fanout_registry
    # Worker-deps factory (Task 3): resolves provider/workspace/tool-registry
    # per fanout batch. Bound method, not called here — dispatch_fanout invokes
    # it once per dispatch as (project_id, session_id) -> WorkerDeps.
    sub_agent_manager._worker_deps_factory = agent_manager.build_worker_deps

    # 6b. Trigger manager
    trigger_manager = TriggerManager(project_store, agent_manager, ws_manager=ws_manager)
    agent_manager._trigger_manager = trigger_manager

    @app.on_event("startup")
    async def _start_triggers():
        await trigger_manager.start()

    # 6c. Start the project-store flush task: runtime fields (budget spend in
    # particular) are batch-written on a fixed interval instead of per-turn.
    # See agent_os/daemon_v2/project_store.py module docstring.
    @app.on_event("startup")
    async def _start_project_store_flush():
        await project_store.start_flush_task()

    @app.on_event("shutdown")
    async def _release_pid():
        release_pid_file()

    @app.on_event("shutdown")
    async def _stop_triggers():
        await trigger_manager.stop()

    # Graceful-shutdown contract: drain any pending runtime updates so the
    # last few LLM-call's worth of budget spend lands on disk before exit.
    @app.on_event("shutdown")
    async def _stop_project_store_flush():
        await project_store.stop_flush_task()

    @app.on_event("shutdown")
    async def _stop_connectors():
        await connector_manager.aclose()

    # 7. Configure routes
    agents_v2.configure(project_store, agent_manager, ws_manager, sub_agent_manager,
                        setup_engine=setup_engine, settings_store=settings_store,
                        credential_store=credential_store,
                        trigger_manager=trigger_manager,
                        provider_registry=provider_registry,
                        lifecycle_observer=lifecycle_observer)
    app.include_router(agents_v2.router)

    # 7a-cred. Credential routes
    from agent_os.api.routes import credentials as credential_routes
    credential_routes.configure(user_credential_store, agent_manager=agent_manager)
    app.include_router(credential_routes.router)

    # 7a. Settings routes
    settings_routes.configure(
        settings_store,
        credential_store=credential_store,
        setup_engine=setup_engine,
        sub_agent_config_store=sub_agent_config_store,
        ws_manager=ws_manager,
    )
    app.include_router(settings_routes.router)

    # 7b. File browsing routes
    files_v2.configure(project_store, agent_manager=agent_manager, ws_manager=ws_manager)
    app.include_router(files_v2.router)

    # 7c. Platform routes
    platform_routes.configure(platform_provider, agent_manager=agent_manager, browser_manager=browser_manager)
    app.include_router(platform_routes.router)

    # 7c2. Connector routes (spec 011)
    from agent_os.api.routes import connectors as connector_routes
    connector_routes.configure(connector_manager)
    app.include_router(connector_routes.router)

    # 7c3. Calendar routes (spec 011 §0.4, Task G1)
    from agent_os.api.routes import calendar as calendar_routes
    calendar_routes.configure(calendar_hub)
    app.include_router(calendar_routes.router)

    # 7c4. Workbench routes (user-flagged memory): flagged-entry mirror + two
    # exits. Shares the agent manager's dispatch seam for /migrate and
    # refreshes the calendar hub after every write.
    from agent_os.api.routes import workbench as workbench_routes
    workbench_routes.configure(project_store, agent_manager, calendar_hub)
    app.include_router(workbench_routes.router)

    # 7c-pricing. Pricing-table routes (resolved rates + per-field origin GET,
    # validated override PUT). Stateless — reads providers.json defaults and
    # the module-level override path resolved from AGENT_OS_DATA_DIR.
    from agent_os.api.routes import pricing_v2
    pricing_v2.configure()
    app.include_router(pricing_v2.router)

    # 7d. Cloud relay (opt-in via AGENT_OS_RELAY_URL env var)
    relay_url = os.environ.get("AGENT_OS_RELAY_URL")
    if relay_url:
        from agent_os.relay.device import get_or_create_device_identity, register_device
        from agent_os.relay.client import RelayClient

        identity = get_or_create_device_identity()
        relay_client = RelayClient(
            relay_url, identity["device_id"], identity["device_secret"],
            project_store=project_store,
        )
        app.state.relay_client = relay_client

        # Hook relay event forwarding into the WS broadcast pipeline
        ws_manager.add_broadcast_hook(relay_client.forward_event)

        @app.on_event("startup")
        async def _start_relay():
            import asyncio
            try:
                await register_device(relay_url, identity["device_id"], identity["device_secret"])
            except Exception:
                pass  # registration is best-effort; device may already exist
            asyncio.create_task(relay_client.start())

        @app.on_event("shutdown")
        async def _stop_relay():
            await relay_client.stop()
    else:
        app.state.relay_client = None

    # 7e. Browser shutdown
    @app.on_event("shutdown")
    async def _stop_browser():
        await browser_manager.shutdown()

    # 7g. Agent manager shutdown
    @app.on_event("shutdown")
    async def _stop_agent_manager():
        await agent_manager.shutdown()

    # 7h. Create one queue dispatcher per project. The dispatcher is the
    # project's session-lifecycle manager — it exists for the life of the
    # project, independent of agent lifecycle, so it is created at boot
    # rather than inside start_agent.
    @app.on_event("startup")
    async def _start_dispatchers():
        try:
            await agent_manager.start_all_dispatchers()
        except Exception:
            import logging
            logging.getLogger(__name__).exception(
                "Failed to start project dispatchers on startup"
            )

    # Idle-eviction sweep: tears down runtime resources for sessions left idle
    # past AgentManager.EVICTION_IDLE_TIMEOUT. Cancelled in agent_manager.shutdown().
    @app.on_event("startup")
    async def _start_eviction():
        try:
            agent_manager.start_eviction()
        except Exception:
            import logging
            logging.getLogger(__name__).exception(
                "Failed to start idle-eviction sweep on startup"
            )

    # 7f. Pairing routes
    pairing_routes.configure(getattr(app.state, "relay_client", None))
    app.include_router(pairing_routes.router)

    # 8. WebSocket endpoint
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        ws_manager.connect(websocket)

        async def heartbeat():
            """Send application-level pings every 30s."""
            try:
                while True:
                    await asyncio.sleep(30)
                    await websocket.send_json({"type": "ping"})
            except Exception:
                pass

        heartbeat_task = asyncio.create_task(heartbeat())
        try:
            while True:
                data = await websocket.receive_json()
                msg_type = data.get("type")
                if msg_type == "subscribe":
                    project_ids = data.get("project_ids", [])
                    ws_manager.subscribe(websocket, project_ids)
                    await websocket.send_json({
                        "type": "subscribed",
                        "project_ids": project_ids,
                    })
                elif msg_type == "pong":
                    pass  # heartbeat response received
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)
        except Exception:
            ws_manager.disconnect(websocket)
        finally:
            heartbeat_task.cancel()

    # 9. SPA static file serving (must be LAST — catch-all for non-API paths)
    spa_dir = os.environ.get(
        "AGENT_OS_SPA_DIR",
        os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"),
    )
    _cached_index_html: str | None = None
    if os.path.isdir(spa_dir):
        @app.get("/{full_path:path}")
        async def spa_fallback(full_path: str):
            nonlocal _cached_index_html
            file_path = os.path.join(spa_dir, full_path)
            if os.path.isfile(file_path):
                return FileResponse(file_path)
            # Serve index.html with injected __AGENT_OS_LOCAL__ flag
            if _cached_index_html is None:
                index_path = os.path.join(spa_dir, "index.html")
                with open(index_path, "r", encoding="utf-8") as f:
                    raw = f.read()
                _cached_index_html = raw.replace(
                    "<head>",
                    "<head><script>window.__AGENT_OS_LOCAL__=true;</script>",
                    1,
                )
            return HTMLResponse(_cached_index_html)

    return app
