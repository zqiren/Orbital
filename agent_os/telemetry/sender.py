# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Ping sender: pending queue + fire-and-forget POST (spec 046 §5).

Cycle (on startup, hourly while running, and once more on the way out —
``flush_now`` is called by the daemon's shutdown hook and, because the desktop
app exits via ``os._exit`` where that hook never fires, by the desktop exit
paths over loopback HTTP first; spec 111 A2):

  1. Rebuild every PAST day the spool still covers and that the server has not
     acknowledged (``acks.json``), rewriting ``pending/{date}.json`` from the
     spool (spec 111 A1). Before this step a past day's file was frozen at the
     last cycle that ran on that day, so anything spooled after it — a
     20-minute session closed before the next tick, everything after the last
     tick before UTC midnight — was never counted.
  2. Build today's ping and (re)write it to ``pending/{date}.json`` — the
     durable snapshot survives spool rotation and process death.
  3. POST every file in ``pending/``. On 2xx, record the ack and delete the
     file for PAST days; today's file stays and is rewritten next cycle (the
     server upserts on ``(install_id, date)`` latest-wins, so resends are
     idempotent — which is also what lets a rebuilt day overwrite the stale
     row it sent earlier).

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
from datetime import datetime, timedelta, timezone
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

# Spec 111 A2: was 6h. The exit flush covers a clean quit; the hourly tick
# bounds what a crash, a kill, or a lost network can take with it. Cost is at
# most 24 idempotent upserts a day.
SEND_INTERVAL_SECONDS = 60 * 60

# Acks older than this are pruned; a day that old is far outside any spool
# rotation window, so nothing could rebuild and resend it anyway.
ACK_RETENTION_DAYS = 90

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

    # Building the client loads the CA bundle (ssl.create_default_context),
    # which hang.log caught blocking the event loop (spec 171 A1). Build it
    # in a worker thread; the request itself is async as before.
    client = await asyncio.to_thread(httpx.AsyncClient, timeout=15.0)
    async with client:
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
        self._acks_path = self._data_dir / "telemetry" / "acks.json"
        self._task: asyncio.Task | None = None
        self._suppression_logged = False
        # One cycle at a time: the exit flush can land while the hourly tick is
        # mid-send. Overlap would only duplicate idempotent POSTs, but the
        # rebuild + snapshot writes should not interleave either.
        self._cycle_lock = asyncio.Lock()

    # -- public ---------------------------------------------------------------

    def last_sent_payload(self) -> dict | None:
        """The most recently transmitted ping, for the settings viewer."""
        return self._read_json(self._pending_dir.parent / "last_sent.json")

    def next_pending_payload(self) -> dict:
        """What the next send would transmit (today's snapshot), verbatim."""
        return self._build_today()

    async def run_cycle(self) -> None:
        """One rebuild+snapshot+flush pass. Never raises."""
        try:
            if not self._is_enabled():
                return
            async with self._cycle_lock:
                # Both rebuild from the spool and write pending/*.json: file
                # I/O that once held the event loop 17.8 s (spec 171 A1).
                # The cycle lock still keeps them from interleaving.
                await asyncio.to_thread(self._rebuild_past_days)
                await asyncio.to_thread(self._snapshot_today)
                await self._flush_pending()
        except Exception:
            logger.debug("telemetry cycle failed", exc_info=True)

    async def flush_now(self, timeout: float = 3.0) -> str:
        """Bounded send-what-you-have for the exit paths (spec 111 A2).

        Returns ``"sent"``, ``"skipped"`` (guarded like ``start()``, or the
        toggle is off), ``"timeout"`` or ``"failed"``. Never raises, never
        blocks longer than ``timeout`` — the app is on its way out and a slow
        network must not hold the window open.
        """
        try:
            if self._suppression_reason() is not None or not self._is_enabled():
                return "skipped"
            await asyncio.wait_for(self.run_cycle(), timeout=timeout)
            return "sent"
        except asyncio.TimeoutError:
            return "timeout"
        except Exception:
            logger.debug("telemetry exit flush failed", exc_info=True)
            return "failed"

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

    def _build_day(self, day: str) -> dict:
        return build_ping(
            self._identity,
            self._spool,
            day,
            version=get_version(),
            os_name=platform.system().lower(),
        )

    def _write_pending(self, ping: dict) -> None:
        try:
            self._pending_dir.mkdir(parents=True, exist_ok=True)
            path = self._pending_dir / f"{ping['date']}.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(ping, indent=2), encoding="utf-8")
            tmp.replace(path)
        except Exception:
            pass

    def _snapshot_today(self) -> None:
        self._write_pending(self._build_today())

    def _rebuild_past_days(self) -> None:
        """Re-aggregate every unacknowledged past day the spool still holds
        (spec 111 A1). Days the spool has rotated out keep whatever pending
        file they have — that file is the best record left and goes out as
        is. Today is left to ``_snapshot_today``."""
        try:
            today = datetime.now(timezone.utc).date().isoformat()
            acks = self._read_acks()
            for day in self._spool.days():
                if day >= today or day in acks:
                    continue
                self._write_pending(self._build_day(day))
        except Exception:
            logger.debug("telemetry past-day rebuild failed", exc_info=True)

    async def _flush_pending(self) -> None:
        today = datetime.now(timezone.utc).date().isoformat()
        headers = {"X-Orbital-Telemetry-Token": APP_TOKEN}
        acks = self._read_acks()
        for path in sorted(self._pending_dir.glob("*.json")):
            payload = self._read_json(path)
            if payload is None:
                path.unlink(missing_ok=True)  # corrupt snapshot: drop
                continue
            day = str(payload.get("date", today))
            if day < today and day in acks:
                path.unlink(missing_ok=True)  # acked earlier; delete failed then
                continue
            try:
                status = await self._post(self._endpoint, payload, headers)
            except Exception:
                return  # offline: leave everything for the next cycle
            if 200 <= status < 300:
                self._record_last_sent(payload)
                if day < today:
                    self._record_ack(day)
                    path.unlink(missing_ok=True)
            else:
                return  # server unhappy: retry next cycle, never loop hot

    # Acks: ``{date: received_at}``. A missing or corrupt file reads as empty,
    # which degrades to "resend everything in the spool window" — safe because
    # the server upserts latest-wins.
    def _read_acks(self) -> dict[str, str]:
        data = self._read_json(self._acks_path)
        if not data:
            return {}
        return {str(k): str(v) for k, v in data.items() if isinstance(k, str)}

    def _record_ack(self, day: str) -> None:
        try:
            acks = self._read_acks()
            acks[day] = datetime.now(timezone.utc).isoformat()
            floor = (datetime.now(timezone.utc) - timedelta(days=ACK_RETENTION_DAYS)).date().isoformat()
            acks = {d: t for d, t in acks.items() if d >= floor}
            self._acks_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._acks_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(acks, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(self._acks_path)
        except Exception:
            pass

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
