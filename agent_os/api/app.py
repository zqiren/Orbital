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
- Write small scripts or snippets (place output in agent_output/)
- Any task that doesn't require sustained multi-session effort

## Rules
- Don't ask unnecessary clarifying questions for simple tasks. Just do it.
- If a task is growing complex (multi-file project, ongoing work, needs persistence),
  suggest: "This looks like it could be its own project. Want me to create one?"
- Keep workspace state maintenance light — only update PROJECT_STATE.md for
  substantial ongoing work, not one-off questions.
"""


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
        "autonomy": "check_in",
        "model": "",
        "api_key": "",
    })
    from agent_os.agent.project_paths import ProjectPaths
    _pp = ProjectPaths(scratch_workspace)
    os.makedirs(_pp.instructions_dir, exist_ok=True)
    with open(_pp.project_goals, "w", encoding="utf-8") as f:
        f.write(SCRATCH_PROJECT_GOALS)


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

    # 2. WebSocket manager
    ws_manager = WebSocketManager()

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
    setup_engine = SetupEngine(registry)

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
    )

    # 6. Agent manager
    agent_manager = AgentManager(
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
    )

    # 6a. Lifecycle observer (wired after agent_manager exists)
    lifecycle_observer = LifecycleObserver(agent_manager, ws_manager)
    process_manager._lifecycle = lifecycle_observer
    sub_agent_manager._lifecycle_observer = lifecycle_observer

    # 6b. Trigger manager
    trigger_manager = TriggerManager(project_store, agent_manager, ws_manager=ws_manager)
    agent_manager._trigger_manager = trigger_manager

    @app.on_event("startup")
    async def _start_triggers():
        await trigger_manager.start()

    @app.on_event("shutdown")
    async def _release_pid():
        release_pid_file()

    @app.on_event("shutdown")
    async def _stop_triggers():
        await trigger_manager.stop()

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
    settings_routes.configure(settings_store, credential_store=credential_store)
    app.include_router(settings_routes.router)

    # 7b. File browsing routes
    files_v2.configure(project_store)
    app.include_router(files_v2.router)

    # 7c. Platform routes
    platform_routes.configure(platform_provider, agent_manager=agent_manager, browser_manager=browser_manager)
    app.include_router(platform_routes.router)

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

    # 7h. Auto-resume agents from previous session
    @app.on_event("startup")
    async def _auto_resume_agents():
        try:
            await agent_manager.auto_resume_agents()
        except Exception:
            import logging
            logging.getLogger(__name__).exception(
                "Failed to auto-resume agents on startup"
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
