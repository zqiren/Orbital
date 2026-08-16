# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Live codex model-list fetcher (TASK-live-model-config).

The sub-agent settings page needs the ChatGPT account's ACTUAL model list —
a free-text model override that the account can't use (e.g. `gpt-5.6`, valid
in Codex desktop but rejected through the CLI's ChatGPT-account gate) 400s
on every dispatch, silently from the user's point of view.

Speaks the same version-pinned app-server JSON-RPC the transport does
(`codex_transport.py`): initialize → initialized → model/list, parsing
``result.data[].id``. Read-only and best-effort by design: the spawn layer
NEVER raises — any failure (binary missing, auth broken, protocol drift,
timeout) returns None and the settings page degrades to the free-text
input it has today.

A module-level TTL cache keeps the settings page from spawning a codex
process on every load; failures are negative-cached on a shorter TTL so a
broken install doesn't re-block each request, but recovers quickly once
fixed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from agent_os.agent.transports.jsonl_stream import read_jsonl_line
from agent_os.utils.subprocess_flags import win_no_window_flags

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 10.0
_SUCCESS_TTL = 600.0
_FAILURE_TTL = 60.0

# binary -> (expires_at per time.monotonic(), ids or None)
_cache: dict[str, tuple[float, list[str] | None]] = {}
_cache_lock = asyncio.Lock()


async def read_model_ids(reader, writer, *, timeout: float = _DEFAULT_TIMEOUT
                         ) -> list[str]:
    """Run the initialize → initialized → model/list handshake over
    newline-delimited JSON-RPC streams and return the model ids.

    Raises on EOF, timeout, or a malformed model/list result — the spawn
    layer maps every raise to None. Non-response lines (server
    notifications, stray output) are skipped, mirroring the transport's
    read loop.
    """

    def send(obj: dict) -> None:
        writer.write((json.dumps(obj) + "\n").encode("utf-8"))

    async def read_response(rpc_id: int) -> dict:
        while True:
            line = await asyncio.wait_for(
                read_jsonl_line(reader), timeout)
            if not line:
                raise RuntimeError(
                    "codex app-server closed the stream before responding")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == rpc_id:
                return msg

    send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
          "params": {"clientInfo": {
              "name": "orbital", "title": "Orbital", "version": "0.1.0"}}})
    await writer.drain()
    await read_response(1)
    send({"jsonrpc": "2.0", "method": "initialized"})
    # Same includeHidden=False the transport's startup resolution uses.
    send({"jsonrpc": "2.0", "id": 2, "method": "model/list",
          "params": {"includeHidden": False}})
    await writer.drain()
    result = (await read_response(2)).get("result") or {}
    data = result.get("data")
    if not isinstance(data, list):
        raise RuntimeError(f"model/list returned no data list: {result}")
    return [m["id"] for m in data
            if isinstance(m, dict) and isinstance(m.get("id"), str)]


async def fetch_codex_models(binary: str = "codex", *,
                             timeout: float = _DEFAULT_TIMEOUT
                             ) -> list[str] | None:
    """Spawn ``<binary> app-server`` and return the account's live model
    ids, or None on ANY failure. Never raises."""
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            binary, "app-server",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            limit=1024 * 1024,  # tolerant reader; limit sizes the fast path
            creationflags=win_no_window_flags(),
        )
        return await read_model_ids(proc.stdout, proc.stdin, timeout=timeout)
    except Exception as exc:
        logger.info("codex model/list unavailable (%s: %s) — settings will "
                    "fall back to free-text model entry",
                    type(exc).__name__, exc)
        return None
    finally:
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass


async def get_codex_models_cached(binary: str = "codex", *,
                                  ttl: float = _SUCCESS_TTL,
                                  failure_ttl: float = _FAILURE_TTL,
                                  timeout: float = _DEFAULT_TIMEOUT
                                  ) -> list[str] | None:
    """TTL-cached :func:`fetch_codex_models`, keyed by binary path.

    The lock serializes fetches so concurrent settings loads can't spawn
    parallel codex processes for the same cold cache.
    """
    async with _cache_lock:
        entry = _cache.get(binary)
        if entry is not None and time.monotonic() < entry[0]:
            return entry[1]
        ids = await fetch_codex_models(binary, timeout=timeout)
        expiry = time.monotonic() + (ttl if ids is not None else failure_ttl)
        _cache[binary] = (expiry, ids)
        return ids


def clear_codex_models_cache() -> None:
    """Drop every cached entry (settings refresh, tests)."""
    _cache.clear()
