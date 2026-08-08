# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Update-availability check (notify-only tier).

On daemon startup + every 6h, fetch the latest released version from the
telemetry service's ``GET /latest`` (a cached GitHub-releases proxy that is
reachable where api.github.com is not) and compare against the running
version. When a newer release exists, expose it via ``GET
/api/v2/update-status`` and announce it once per version over a global WS
event — the frontend renders a dismissible pill; download/install stays
manual (the pill links to the release page).

Deliberately NOT auto-update: silently replacing unsigned binaries would hit
SmartScreen/Gatekeeper on every update, and a running app can't replace
itself without an updater helper. Revisit once signing lands.

The check is an anonymous GET — nothing identifying is sent — and is
independent of the telemetry toggle by design.

Gating: packaged builds only (``sys.frozen``); a dev daemon never nags.
``AGENT_OS_UPDATE_CHECK=1|0`` force-enables (stub testing) / force-disables,
``AGENT_OS_UPDATE_URL`` overrides the endpoint.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Awaitable, Callable

from agent_os.version import get_version

logger = logging.getLogger(__name__)

DEFAULT_LATEST_URL = "https://orbital-telemetry-production.up.railway.app/latest"
CHECK_INTERVAL_SECONDS = 6 * 60 * 60

FetchFn = Callable[[str], Awaitable[dict]]


def _parse_version(value: str) -> tuple[int, int, int] | None:
    parts = str(value).strip().lstrip("v").split(".")
    if len(parts) < 3:
        parts = parts + ["0"] * (3 - len(parts))
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return None


def is_newer(latest: str, current: str) -> bool:
    """True when ``latest`` is a strictly newer semver than ``current``.

    Unparseable versions never trigger a notification, and neither does a
    ``0.0.0`` current (the version module's never-raise floor).
    """
    latest_t = _parse_version(latest)
    current_t = _parse_version(current)
    if latest_t is None or current_t is None or current_t == (0, 0, 0):
        return False
    return latest_t > current_t


def _default_enabled() -> bool:
    env = os.environ.get("AGENT_OS_UPDATE_CHECK")
    if env is not None:
        return env not in ("0", "false", "")
    return bool(getattr(sys, "frozen", False))


async def _httpx_fetch(url: str) -> dict:
    import httpx

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


class UpdateChecker:
    def __init__(
        self,
        ws_manager,
        latest_url: str | None = None,
        enabled: bool | None = None,
        fetch: FetchFn | None = None,
    ) -> None:
        self._ws = ws_manager
        self._url = latest_url or os.environ.get("AGENT_OS_UPDATE_URL", DEFAULT_LATEST_URL)
        self._enabled = _default_enabled() if enabled is None else enabled
        self._fetch = fetch or _httpx_fetch
        self._task: asyncio.Task | None = None
        self._announced: str | None = None
        self.status: dict = {
            "current": get_version(),
            "update_available": False,
            "latest": None,
            "url": None,
        }

    async def run_check(self) -> None:
        """One check. Never raises; offline is silent."""
        if not self._enabled:
            return
        try:
            data = await self._fetch(self._url)
            latest = str(data.get("version") or "")
            if not is_newer(latest, self.status["current"]):
                return
            url = data.get("url") or ""
            self.status.update(update_available=True, latest=latest, url=url)
            # Announce each new version once — the 6h re-check must not
            # re-surface a pill the user dismissed.
            if latest != self._announced:
                self._announced = latest
                self._ws.broadcast_global(
                    {"type": "update.available", "version": latest, "url": url}
                )
        except Exception:
            logger.debug("update check failed", exc_info=True)

    def start(self) -> None:
        if not self._enabled:
            return
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(), name="update-checker")

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _loop(self) -> None:
        while True:
            await self.run_check()
            await asyncio.sleep(CHECK_INTERVAL_SECONDS)


_checker: UpdateChecker | None = None


def configure(ws_manager, **kwargs) -> UpdateChecker:
    global _checker
    _checker = UpdateChecker(ws_manager, **kwargs)
    return _checker


def get_checker() -> UpdateChecker | None:
    return _checker
