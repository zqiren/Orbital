# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Connectors an agent holds itself (its own MCP servers), as it last said.

Orbital-held connectors are served to agents through the daemon's ``orbital``
mount (``connector_sharing``). An agent's OWN servers — Claude Code's
claude.ai Gmail or its Figma sign-in, Codex's local ``node_repl`` — belong to
that agent: only it can call them, and the others use them by handing it the
task. Orbital has nothing to set for them; it only needs to know they exist,
so the manager (and a pinned worker's ``dispatch`` tool) can route a task
needing one to the agent that has it, and Settings can show them.

Discovery is the agent's own report over its protocol while it runs, never a
scan of its config files and never a spawn made to ask:

- Claude Code: the ``init`` system message of every turn
  (``mcp_servers: [{name, status, source}]``), plus one ``get_mcp_status``
  after a turn whose init still said ``pending`` (claude.ai servers connect
  after init).
- Codex: ``mcpServerStatus/list`` (spec 184 already sends it for the mount
  check; a worker without the mount gets one in the background).
- ACP agents and Pi: their protocols do not report MCP servers; unsupported.

Only names and a status are kept: no URL, command, env or header is stored
or sent anywhere. Orbital's own ``orbital`` mount is never recorded.

Stored in ``{data_dir}/agent-connectors.json`` (atomic write, a missing or
corrupt file reads as empty, and an older build never opens it)::

    {"version": 1, "agents": {"claude-code": {
        "servers": [{"name": "claude.ai Gmail", "status": "pending"}],
        "last_seen": "2026-11-17T09:00:00Z"}}}

Unconfigured (tests, tools that never start a daemon) the module records
nothing and knows nothing, so every prompt is exactly what it was.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone

from agent_os.utils.atomic_write import atomic_write_text

logger = logging.getLogger(__name__)

FILE_NAME = "agent-connectors.json"
_VERSION = 1

#: Server names an agent reports that are never its own connectors:
#: Orbital's mount (dispatch + Orbital-held connectors) and Codex's built-in
#: runtimes (probe 2026-11-17, codex 0.157.1: ``codex_app`` = the bundled
#: codex-app-tools plugin, ``codex_apps`` = ChatGPT's own apps runtime,
#: ``cua_repl`` = the bundled computer-use plugin's REPL).
_NOT_CONNECTORS = frozenset({"orbital", "codex_app", "codex_apps", "cua_repl"})
#: A server the user switched off in that agent is not something it has.
_HIDDEN_STATUSES = frozenset({"disabled"})
#: Names at most this many per agent in a roster line, then "and N more".
ROSTER_CAP = 8
#: Bounds on what one report may store (names only, so this is generous).
_MAX_SERVERS = 64
_MAX_NAME = 120

_lock = threading.Lock()
_path: str | None = None
# agent id -> {"servers": [{"name", "status"}], "last_seen": str}
_agents: dict[str, dict] = {}
# Agents that reported since this process started (their first report is
# always written, so ``last_seen`` follows the agent's latest start).
_reported: set[str] = set()


def configure(data_dir: str | None) -> None:
    """Point the registry at ``{data_dir}/agent-connectors.json`` and load
    it. ``None`` turns it off (nothing recorded, nothing known)."""
    global _path
    with _lock:
        _agents.clear()
        _reported.clear()
        _path = os.path.join(os.path.abspath(data_dir), FILE_NAME) if data_dir else None
        if _path is not None:
            _agents.update(_load(_path))


def _load(path: str) -> dict[str, dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        logger.debug("agent connectors: unreadable %s, starting empty", path, exc_info=True)
        return {}
    agents = data.get("agents") if isinstance(data, dict) else None
    if not isinstance(agents, dict):
        return {}
    out: dict[str, dict] = {}
    for agent, entry in agents.items():
        if not isinstance(agent, str) or not isinstance(entry, dict):
            continue
        servers = _clean(entry.get("servers"))
        last_seen = entry.get("last_seen")
        out[agent] = {"servers": servers,
                      "last_seen": last_seen if isinstance(last_seen, str) else None}
    return out


def _clean(servers) -> list[dict]:
    """``[{name, status}]``: names only, Orbital's mount dropped, one entry
    per name, sorted by name, bounded."""
    if not isinstance(servers, (list, tuple)):
        return []
    seen: dict[str, str] = {}
    for item in servers:
        if isinstance(item, dict):
            name, status = item.get("name"), item.get("status")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            name, status = item
        else:
            continue
        if not isinstance(name, str):
            continue
        name = " ".join(name.split())[:_MAX_NAME]
        if not name or name == "orbital" or name in seen:
            continue
        seen[name] = status if isinstance(status, str) and status else "unknown"
        if len(seen) >= _MAX_SERVERS:
            break
    return [{"name": n, "status": seen[n]} for n in sorted(seen, key=str.casefold)]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def record(agent: str, servers) -> bool:
    """Store what ``agent`` just reported. Writes when the list changed, and
    on the agent's first report in this process (its ``last_seen``). Returns
    True when it wrote. Never raises: a failure is logged at debug."""
    try:
        if not isinstance(agent, str) or not agent:
            return False
        cleaned = _clean(servers)
        with _lock:
            if _path is None:
                return False
            before = _agents.get(agent)
            first = agent not in _reported
            _reported.add(agent)
            if before is not None and before.get("servers") == cleaned and not first:
                return False
            _agents[agent] = {"servers": cleaned, "last_seen": _now()}
            payload = {"version": _VERSION, "agents": {
                a: _agents[a] for a in sorted(_agents)}}
            atomic_write_text(_path, json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
        return True
    except Exception:  # noqa: BLE001 — discovery must never break a worker
        logger.debug("agent connectors: record failed for %s", agent, exc_info=True)
        return False


def make_sink(agent: str):
    """A transport sink: ``sink(servers)`` stores the report off the event
    loop when one is running (the write never sits on a spawn or a turn)."""
    def sink(servers) -> None:
        try:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is None:
                record(agent, servers)
            else:
                loop.run_in_executor(None, record, agent, list(servers or ()))
        except Exception:  # noqa: BLE001
            logger.debug("agent connectors: sink failed for %s", agent, exc_info=True)
    return sink


def display_name(raw: str) -> str:
    """What a person reads: Claude Code's account connectors drop their
    ``claude.ai `` prefix, a plugin's server (``plugin:<plugin>:<server>``)
    shows its server name. Storage keeps the raw name."""
    name = raw
    if name.startswith("claude.ai "):
        name = name[len("claude.ai "):]
    elif name.startswith("plugin:"):
        name = name.rsplit(":", 1)[-1]
    return name.strip() or raw


def held(agent: str) -> list[dict]:
    """The connectors ``agent`` holds itself, as last reported:
    ``[{name, raw_name, status}]`` by display name, without Orbital's mount,
    Codex's built-ins or servers switched off in the agent."""
    with _lock:
        entry = _agents.get(agent)
        servers = list(entry.get("servers") or ()) if entry else []
    out = []
    shown: set[str] = set()
    for s in servers:
        raw = s["name"]
        if raw in _NOT_CONNECTORS or s.get("status") in _HIDDEN_STATUSES:
            continue
        name = display_name(raw)
        if name.casefold() in shown:
            continue
        shown.add(name.casefold())
        out.append({"name": name, "raw_name": raw, "status": s.get("status") or "unknown"})
    out.sort(key=lambda c: (c["name"].casefold(), c["raw_name"]))
    return out


def last_seen(agent: str) -> str | None:
    with _lock:
        entry = _agents.get(agent)
        return entry.get("last_seen") if entry else None


def known_agents() -> list[str]:
    with _lock:
        return sorted(_agents)


def roster_names(agent: str, cap: int = ROSTER_CAP) -> str:
    """``"Figma, Gmail, Playwright"`` (at most ``cap`` names, then
    ``"and N more"``), or ``""`` when nothing is known: the caller adds
    nothing then, so its text is unchanged."""
    names = [c["name"] for c in held(agent)]
    if not names:
        return ""
    if len(names) > cap:
        return f"{', '.join(names[:cap])} and {len(names) - cap} more"
    return ", ".join(names)


# -- What each agent reports, in the registry's form -------------------------

def claude_servers(mcp_servers) -> list[dict] | None:
    """Claude Code's ``init`` ``mcp_servers`` (or ``get_mcp_status``'s
    ``mcpServers``) as ``[{name, status}]``; None when malformed."""
    if not isinstance(mcp_servers, list):
        return None
    out = []
    for s in mcp_servers:
        if isinstance(s, dict) and isinstance(s.get("name"), str):
            status = s.get("status")
            out.append({"name": s["name"],
                        "status": status if isinstance(status, str) else "unknown"})
    return out


def codex_servers(result) -> list[dict] | None:
    """Codex's ``mcpServerStatus/list`` result as ``[{name, status}]``.

    Codex gives no single status, so it is derived: tools listed →
    ``connected``; not signed in → ``needs-auth``; a start error → ``failed``;
    neither tools nor an error → ``idle``. None when malformed."""
    data = result.get("data") if isinstance(result, dict) else None
    if not isinstance(data, list):
        return None
    out = []
    for e in data:
        if not isinstance(e, dict) or not isinstance(e.get("name"), str):
            continue
        tools = e.get("tools")
        if tools:
            status = "connected"
        elif e.get("authStatus") == "notLoggedIn":
            status = "needs-auth"
        elif e.get("toolsError"):
            status = "failed"
        else:
            status = "idle"
        out.append({"name": e["name"], "status": status})
    return out
