# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Ping sender: pending queue + fire-and-forget POST (spec 046 §5).

Cycle (on startup + every 6h — never on shutdown, the desktop app exits via
os._exit and shutdown hooks provably never fire there):

  1. Build today's ping and (re)write it to ``pending/{date}.json`` — the
     durable snapshot survives spool rotation and process death.
  2. POST every file in ``pending/``. On 2xx, delete files for PAST days;
     today's file stays and is rewritten next cycle (the server upserts on
     ``(install_id, date)`` latest-wins, so resends are idempotent).

Sends are never-raise: failure leaves the pending files for the next cycle and
must never affect the product. The toggle is checked per cycle, so PUT
/settings kills emission live.

``start()`` — the only transmit path in the real app — additionally refuses to
run for non-product processes (spec 063 §7 P0): see ``_suppression_reason``.
``run_cycle()`` is deliberately NOT guarded; it is the seam the unit tests
drive directly with an injected ``post``, and it never reaches the network on
its own.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

from agent_os.version import get_version

from .identity import InstallIdentity
from .rollup import build_ping
from .spool import Spool

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://orbital-telemetry-production.up.railway.app/ingest"
# Abuse-limiting app token, not a secret (spec §9): the client is open source.
APP_TOKEN = "orbital-telemetry-v1"

SEND_INTERVAL_SECONDS = 6 * 60 * 60  # Q4: startup + every 6h

# Opt-out switch for any process that is not a user's Orbital (spec 063 §7).
# Set in tests/conftest.py, .github/workflows/ci.yml and restart-daemon.sh.
DISABLE_ENV_VAR = "AGENT_OS_TELEMETRY_DISABLED"
_FALSEY = {"", "0", "false", "no", "off"}

PostFn = Callable[[str, dict, dict], Awaitable[int]]


def env_disabled() -> bool:
    """True when ``AGENT_OS_TELEMETRY_DISABLED`` is set to anything truthy."""
    return os.environ.get(DISABLE_ENV_VAR, "").strip().lower() not in _FALSEY


def _resolve(path: Path) -> Path | None:
    try:
        return Path(path).resolve()
    except Exception:
        return None


def is_temp_path(path: str | Path) -> bool:
    """True when ``path`` lives under a temp root, symlinks resolved.

    The structural backstop behind the env var (spec 063 §12 decision 4): an
    opt-out is forgotten by the next CI job or daemon-spawning test, but a
    data dir under ``tmp_path``/``mkdtemp`` is never a real install. Both
    sides are realpath'd because macOS temp roots are symlinks
    (``/tmp`` → ``/private/tmp``, ``/var/folders`` → ``/private/var/folders``).
    """
    resolved = _resolve(Path(path))
    if resolved is None:
        return False
    for root in (tempfile.gettempdir(), "/tmp", "/private/var/folders"):
        root_resolved = _resolve(Path(root))
        if root_resolved is None:
            continue
        if resolved == root_resolved or root_resolved in resolved.parents:
            return True
    return False


async def _httpx_post(url: str, payload: dict, headers: dict) -> int:
    import httpx

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(url, json=payload, headers=headers)
        return resp.status_code


class TelemetrySender:
    def __init__(
        self,
        data_dir: Path,
        identity: InstallIdentity,
        spool: Spool,
        is_enabled: Callable[[], bool],
        endpoint: str = DEFAULT_ENDPOINT,
        post: PostFn | None = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._pending_dir = self._data_dir / "telemetry" / "pending"
        self._identity = identity
        self._spool = spool
        self._is_enabled = is_enabled
        self._endpoint = endpoint
        self._post = post or _httpx_post
        self._task: asyncio.Task | None = None
        self._suppression_logged = False

    # -- public ---------------------------------------------------------------

    def last_sent_payload(self) -> dict | None:
        """The most recently transmitted ping, for the settings viewer."""
        return self._read_json(self._pending_dir.parent / "last_sent.json")

    def next_pending_payload(self) -> dict:
        """What the next send would transmit (today's snapshot), verbatim."""
        return self._build_today()

    async def run_cycle(self) -> None:
        """One snapshot+flush pass. Never raises."""
        try:
            if not self._is_enabled():
                return
            self._snapshot_today()
            await self._flush_pending()
        except Exception:
            logger.debug("telemetry cycle failed", exc_info=True)

    def start(self) -> None:
        """Startup send + 6h periodic loop as a background task.

        Inert for non-product processes — see ``_suppression_reason``. Only
        transmission is suppressed; the local spool and the settings viewer
        are unaffected.
        """
        reason = self._suppression_reason()
        if reason is not None:
            if not self._suppression_logged:
                self._suppression_logged = True
                logger.info("telemetry sender not started: %s", reason)
            return
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(), name="telemetry-sender")

    def _suppression_reason(self) -> str | None:
        """Why this process must not transmit, or None if it may (spec 063 §7).

        Two independent guards, because 61 test files reach ``create_app()``
        and each one used to mint an install_id and post it to production
        (~98% of the stored fleet):

        1. ``AGENT_OS_TELEMETRY_DISABLED`` — explicit opt-out for CI, the dev
           daemon, and the test suite.
        2. a data dir under a temp root — structural backstop for every
           process that forgets guard 1.
        """
        if env_disabled():
            return f"{DISABLE_ENV_VAR} is set"
        if is_temp_path(self._data_dir):
            return f"data dir is under a temp path ({self._data_dir})"
        return None

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    # -- internals ------------------------------------------------------------

    async def _loop(self) -> None:
        while True:
            await self.run_cycle()
            await asyncio.sleep(SEND_INTERVAL_SECONDS)

    def _build_today(self) -> dict:
        day = datetime.now(timezone.utc).date().isoformat()
        return build_ping(
            self._identity,
            self._spool,
            day,
            version=get_version(),
            os_name=platform.system().lower(),
        )

    def _snapshot_today(self) -> None:
        ping = self._build_today()
        try:
            self._pending_dir.mkdir(parents=True, exist_ok=True)
            path = self._pending_dir / f"{ping['date']}.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(ping, indent=2), encoding="utf-8")
            tmp.replace(path)
        except Exception:
            pass

    async def _flush_pending(self) -> None:
        today = datetime.now(timezone.utc).date().isoformat()
        headers = {"X-Orbital-Telemetry-Token": APP_TOKEN}
        for path in sorted(self._pending_dir.glob("*.json")):
            payload = self._read_json(path)
            if payload is None:
                path.unlink(missing_ok=True)  # corrupt snapshot: drop
                continue
            try:
                status = await self._post(self._endpoint, payload, headers)
            except Exception:
                return  # offline: leave everything for the next cycle
            if 200 <= status < 300:
                self._record_last_sent(payload)
                if payload.get("date", today) < today:
                    path.unlink(missing_ok=True)
            else:
                return  # server unhappy: retry next cycle, never loop hot

    def _record_last_sent(self, payload: dict) -> None:
        try:
            path = self._pending_dir.parent / "last_sent.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(path)
        except Exception:
            pass

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except Exception:
            return None
