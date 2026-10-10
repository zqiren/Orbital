# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Live Claude Code model-list fetcher.

The claude-code settings dropdown used a hardcoded whitelist that went stale
every model generation. The CLI already publishes the account's list: its
stream-json ``initialize`` control response carries ``models[]``
(``value`` / ``resolvedModel`` / ``displayName`` ...), and answering it makes
no API call. ``value`` is exactly what ``--model`` accepts (aliases like
``opus``, pins like ``claude-opus-5``, variants like ``claude-fable-5-1[1m]``).

Same shape as ``codex_models``: a pure protocol layer, a spawn layer that
NEVER raises (any failure returns None and the settings page falls back to
the static whitelist), and a TTL cache so a settings load doesn't spawn the
CLI every time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import tempfile
import time

from agent_os.agent.transports.jsonl_stream import read_jsonl_line
from agent_os.agent.transports.process_kill import kill_spawned_tree
from agent_os.utils.subprocess_flags import win_no_window_flags

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 15.0
_SUCCESS_TTL = 600.0
_FAILURE_TTL = 60.0
_REQUEST_ID = "orbital-models"

# The CLI's own "Default (recommended)" entry. Orbital already expresses
# "use the CLI default" as an empty setting, so offering it would be a
# second spelling of the same choice.
_SKIP_VALUES = {"default"}

_cache: dict[str, tuple[float, list[dict] | None]] = {}
_cache_lock = asyncio.Lock()


async def read_model_values(reader, writer, *, timeout: float = _DEFAULT_TIMEOUT
                            ) -> list[str]:
    """The ``models[].value`` list only (see :func:`read_models`)."""
    return [m["value"] for m in await read_models(reader, writer, timeout=timeout)]


async def read_models(reader, writer, *, timeout: float = _DEFAULT_TIMEOUT
                      ) -> list[dict]:
    """Send ``initialize`` and return ``[{"value", "label"}]`` in the CLI's
    order, deduplicated by value. ``label`` is the CLI's own ``displayName``
    ("Opus 5.5"), falling back to the value. Raises on EOF, timeout or an
    error response."""
    writer.write((json.dumps({
        "type": "control_request",
        "request_id": _REQUEST_ID,
        "request": {"subtype": "initialize"},
    }) + "\n").encode("utf-8"))
    await writer.drain()
    while True:
        line = await asyncio.wait_for(read_jsonl_line(reader), timeout)
        if not line:
            raise RuntimeError("claude closed the stream before initializing")
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(msg, dict) or msg.get("type") != "control_response":
            continue
        response = msg.get("response") or {}
        if response.get("subtype") != "success":
            raise RuntimeError(f"claude initialize failed: {response}")
        models = (response.get("response") or {}).get("models")
        if not isinstance(models, list):
            raise RuntimeError("claude initialize returned no models list")
        out: list[dict] = []
        seen: set[str] = set()
        for entry in models:
            value = entry.get("value") if isinstance(entry, dict) else None
            if (not isinstance(value, str) or not value
                    or value in _SKIP_VALUES or value in seen):
                continue
            seen.add(value)
            out.append({"value": value, "label": _label_for(entry, value)})
        return out


_MAX_LABEL = 40
# "<Family> <version>" — "Opus 5.5", "Haiku 4.5"; not "Opus (1M context)".
_VERSIONED = re.compile(r"^\S+ \d+(\.\d+)*\b")


def _label_for(entry: dict, value: str) -> str:
    """The dropdown label: the CLI's ``displayName``, made to carry a version.

    Recent CLIs answer with unversioned names ("Opus", "Opus (1M context)")
    and put the version only in ``description`` ("Opus 5.5 · Best for ...").
    Which model an alias runs depends on the installed CLI — on 2.1.235
    "opus[1m]" is Opus 5, on 2.1.293 "opus" is Opus 5.5 — so a bare "Opus"
    hides it. When the name carries no version, the description's head is
    used if it names the same family, carries one and is label-sized.
    """
    name = entry.get("displayName")
    name = name if isinstance(name, str) and name else None
    if name is None:
        return value
    if _VERSIONED.match(name):
        return name
    description = entry.get("description")
    if isinstance(description, str):
        head = description.split(" · ", 1)[0].strip()
        family = name.split()[0]
        if (head.lower().startswith(family.lower() + " ")
                and _VERSIONED.match(head) and len(head) <= _MAX_LABEL):
            return head
    return name


async def fetch_claude_models(binary: str = "claude", *,
                              timeout: float = _DEFAULT_TIMEOUT
                              ) -> list[dict] | None:
    """Spawn the CLI in stream-json mode just long enough to answer
    ``initialize``. Returns ``[{"value", "label"}]``, or None on ANY failure."""
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            binary, "--output-format", "stream-json", "--verbose",
            "--input-format", "stream-json",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            # A neutral cwd: no project context is needed, and the probe must
            # not touch a real workspace.
            cwd=tempfile.gettempdir(),
            limit=1024 * 1024,
            creationflags=win_no_window_flags(),
        )
        return await read_models(proc.stdout, proc.stdin, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — display data, never raise
        logger.info("claude model list unavailable (%s: %s) — settings fall "
                    "back to the static whitelist", type(exc).__name__, exc)
        return None
    finally:
        await kill_spawned_tree(proc, label="claude model probe")


async def get_claude_models_cached(binary: str = "claude", *,
                                   ttl: float = _SUCCESS_TTL,
                                   failure_ttl: float = _FAILURE_TTL,
                                   timeout: float = _DEFAULT_TIMEOUT
                                   ) -> list[dict] | None:
    """TTL-cached :func:`fetch_claude_models`, keyed by binary path; the lock
    stops concurrent settings loads from spawning parallel CLIs."""
    async with _cache_lock:
        entry = _cache.get(binary)
        if entry is not None and time.monotonic() < entry[0]:
            return entry[1]
        values = await fetch_claude_models(binary, timeout=timeout)
        expiry = time.monotonic() + (ttl if values is not None else failure_ttl)
        _cache[binary] = (expiry, values)
        return values


def clear_claude_models_cache() -> None:
    """Drop every cached entry (settings refresh, tests)."""
    _cache.clear()
