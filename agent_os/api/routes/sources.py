# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""REST routes for task sources (任务来源). The contract is
``run/1116-sources/API-CONTRACT.md``; the logic is ``agent_os/sources``.

Access: the same reach as every local route and as ``POST
/workbench/tasks`` (spec 162) — this machine, or a registered device through
the relay — with no extra token. ``POST /register`` is how agents add a
source; what they register arrives switched off.

Secrets in request bodies are passed to the keychain and never echoed: no
response, log line or WebSocket event carries one.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys

from fastapi import APIRouter, Body, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse

from agent_os.sources.core import ApiError, SendError, SourcesCore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/sources")

_core: SourcesCore | None = None


def configure(core: SourcesCore | None) -> None:
    global _core
    _core = core


def _c() -> SourcesCore:
    if _core is None:
        raise HTTPException(status_code=503, detail={"code": "sources_unavailable"})
    return _core


async def _call(coro):
    try:
        return await coro
    except ApiError as e:
        raise HTTPException(status_code=e.status, detail=e.detail)


def _body(body) -> dict:
    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail={"code": "invalid", "field": None,
                                                     "message": "expected a JSON object"})
    return body


def skill_url(request: Request | None = None) -> str:
    """The skill route on this running daemon: ``AGENT_OS_PORT`` (what the
    packaged app and the scripts export), else the port this request came
    in on."""
    import os
    port = (os.environ.get("AGENT_OS_PORT") or "").strip()
    if not port.isdigit() and request is not None:
        server = request.scope.get("server") or ()
        port = str(server[1]) if len(server) > 1 and server[1] else ""
    return f"http://127.0.0.1:{port or 8000}/api/v2/sources/skill"


@router.get("/catalog")
async def get_catalog(request: Request):
    return _c().catalogue(skill_url=skill_url(request))


@router.get("/skill", response_class=PlainTextResponse)
async def get_skill():
    """The ``orbital-sources`` skill text (Markdown) for agents."""
    from agent_os.sources.skill import SKILL_TEXT
    return PlainTextResponse(SKILL_TEXT, media_type="text/markdown; charset=utf-8")


@router.get("/accounts")
async def get_accounts(type: str = Query(...)):
    return await _call(_c().accounts(type))


@router.post("/choices")
async def post_choices(body: dict = Body(...)):
    return await _call(_c().choices(_body(body)))


@router.post("/qr")
async def post_qr(body: dict = Body(...)):
    return await _call(_c().qr_start(_body(body).get("type")))


@router.get("/qr/{qr_id}")
async def get_qr(qr_id: str):
    return await _call(_c().qr_get(qr_id))


_LOOPBACK = ("127.0.0.1", "localhost", "::1")


def _page_may_start_commands(request: Request) -> None:
    """Registering a source and switching on a newly registered one (a
    check command is a shell command) are refused to a web page from
    elsewhere. The daemon answers every origin (CORS ``*``), so without
    this any page open in the user's browser could register a command and
    switch it on. A browser cannot forge ``Origin``; agents and scripts send
    none; the app's own page is loopback; the phone comes through the
    relay's origin. ``X-Via-Relay`` is not trusted here (a page can set it)."""
    from urllib.parse import urlparse
    origin = (request.headers.get("origin") or "").strip()
    if not origin:
        return
    host = (urlparse(origin).hostname or "").lower() if origin != "null" else ""
    if host in _LOOPBACK or host.startswith("127."):
        return
    import os
    relay = (os.environ.get("AGENT_OS_RELAY_URL") or "").strip()
    if relay and host and host == (urlparse(relay).hostname or "").lower():
        return
    logger.warning("sources: refused a request from origin %s", origin[:100])
    raise HTTPException(status_code=403, detail={"code": "origin_refused"})


@router.post("/register", status_code=201)
async def post_register(request: Request, body: dict = Body(...)):
    _page_may_start_commands(request)
    detail = await _call(_c().register(_body(body)))
    return {"source": detail}


@router.get("")
async def list_sources():
    return _c().list()


@router.post("", status_code=201)
async def create_source(body: dict = Body(...)):
    return await _call(_c().create(_body(body)))


@router.get("/{source_id}")
async def get_source(source_id: str, limit: int = Query(50), before: str | None = Query(None)):
    return await _call(_c().detail(source_id, limit=limit, before=before))


@router.patch("/{source_id}")
async def patch_source(source_id: str, request: Request, body: dict = Body(...)):
    body = _body(body)
    if body.get("on") is True:
        try:
            is_new = bool(_c()._cfg(source_id).get("is_new"))
        except ApiError as e:
            raise HTTPException(status_code=e.status, detail=e.detail)
        if is_new:
            _page_may_start_commands(request)
    return await _call(_c().patch(source_id, body))


@router.delete("/{source_id}")
async def delete_source(source_id: str):
    return await _call(_c().delete(source_id))


@router.post("/{source_id}/check")
async def check_source(source_id: str):
    return await _call(_c().check(source_id))


@router.post("/{source_id}/reconnect")
async def reconnect_source(source_id: str, body: dict = Body(...)):
    return await _call(_c().reconnect(source_id, _body(body)))


@router.post("/{source_id}/catch-up")
async def catch_up(source_id: str):
    return await _call(_c().catch_up(source_id))


@router.post("/{source_id}/dismiss-held")
async def dismiss_held(source_id: str):
    return await _call(_c().dismiss_held(source_id))


CALENDAR_SETTINGS_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_Calendars"


def _local_only(request: Request) -> None:
    # The file manager / System Settings open on the computer running the
    # daemon: a request through the relay (the phone) is refused, as the
    # files reveal route does.
    if "true" in request.headers.getlist("x-via-relay"):
        raise HTTPException(status_code=403, detail={"code": "local_only"})


@router.post("/{source_id}/reveal")
async def reveal_command(source_id: str, request: Request):
    """Show a check command's script in Finder / File Explorer: the first
    existing path in ``command.run``, else its ``cwd``."""
    _local_only(request)
    path = await _call(_c().command_path(source_id))
    if path is None:
        raise HTTPException(status_code=404, detail={"code": "no_path"})
    if sys.platform not in ("darwin", "win32"):
        raise HTTPException(status_code=501, detail={"code": "unsupported"})
    from agent_os.api.routes.files_v2 import _spawn_reveal
    try:
        await asyncio.to_thread(_spawn_reveal, path, False, sys.platform)
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("sources: reveal failed: %s", type(e).__name__)
        raise HTTPException(status_code=500, detail={"code": "reveal_failed"})
    return {"path": path}


@router.post("/open-settings")
async def open_settings(request: Request):
    """macOS: open System Settings › Privacy & Security › Calendars — the
    action of a ``calendar_permission`` failure, and of Calendar access
    refused while connecting (no source exists yet then)."""
    _local_only(request)
    if sys.platform != "darwin":
        raise HTTPException(status_code=501, detail={"code": "unsupported"})
    from agent_os.utils.subprocess_flags import win_no_window_flags
    await asyncio.to_thread(subprocess.run, ["open", CALENDAR_SETTINGS_URL], check=False,
                            timeout=10, capture_output=True, creationflags=win_no_window_flags())
    return {"opened": True}


@router.post("/{source_id}/arrivals/{arrival_id}/route")
async def route_arrival(source_id: str, arrival_id: str, body: dict = Body(...)):
    return await _call(_c().route_arrival(source_id, arrival_id, _body(body).get("project_id")))


# --------------------------------------------------------------------------
# Wiring: the Workbench task path and session lookups
# --------------------------------------------------------------------------

def workbench_sender():
    """``send(text, project_id)`` through the spec-162 handler itself
    (``create_task``): route on 自动 unless a project is given, mint a
    session, inject. Its refusals map to the arrival's ``reason``."""
    from agent_os.api.routes import workbench

    async def send(text: str, project_id: str | None) -> dict:
        try:
            req = workbench.TaskRequest(text=text, project_id=project_id)
        except Exception as e:
            raise SendError("send_failed", 422, {"code": "invalid", "message": str(e)[:200]})
        try:
            body = await workbench.create_task(req)
        except HTTPException as e:
            detail = e.detail
            if e.status_code == 502:
                reason = "routing_failed"
            elif isinstance(detail, dict) and detail.get("code") == "missing_api_key":
                reason = "missing_api_key"
            elif e.status_code == 404:
                reason = "no_project"
            else:
                reason = "send_failed"
            raise SendError(reason, e.status_code, detail)
        await _name_session(body, text)
        return body
    return send


async def _name_session(body: dict, text: str) -> None:
    """Name the session from the item's title (the message's first line),
    not the whole message with its source header, through the same PATCH
    handler a rename in the sidebar uses. Best effort: the task is sent
    either way."""
    from agent_os.agent.session import _derive_name
    from agent_os.api.routes import agents_v2
    pid, sid = (body or {}).get("project_id"), (body or {}).get("session_id")
    name = _derive_name((text or "").split("\n", 1)[0])
    if not (pid and sid and name):
        return
    try:
        await agents_v2.patch_session(pid, sid, agents_v2.SessionPatchRequest(name=name))
    except Exception as e:
        logger.info("sources: could not name session %s (%s)", sid,
                    getattr(e, "status_code", type(e).__name__))
