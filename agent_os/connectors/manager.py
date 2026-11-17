# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""ConnectorManager — connector lifecycle, OAuth, and cached MCP sessions.

Spec 011 §2: a global singleton beside ``AgentManager``. It owns the merged
catalog (bundled manifests + Tier-0 custom servers persisted in
``orbital-data/connectors.json``), the OAuth handshake + token persistence via
``UserCredentialStore`` (keyed ``connector:<auth_provider>:<account>``), and one
lazily-opened, cached MCP ``ClientSession`` per connected server.

Everything network-facing is injectable (``oauth_client_provider``,
``http_client_factory``, ``open_browser``, ``session_opener``) so tests run fully
local. Production wiring supplies the real streamable-HTTP opener and an OAuth
client provider reading ``connector_google_client_id/secret`` from settings.
"""

import asyncio
import json
import logging
import os
import webbrowser
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Awaitable, Callable

import anyio
import httpx
from agent_os.utils.atomic_write import atomic_write_text

from .manifest import ConnectorManifest, custom_manifest, load_catalog, manifest_from_dict
from .mcp_client import streamable_http_opener
from .oauth import LoopbackOAuthFlow, OAuthClientConfig, OAuthError, TokenSet

logger = logging.getLogger(__name__)

# Per-connector agent access (connectors shared across agents). "all" is the
# default and today's behaviour; "selected" names the agents allowed, the
# manager among them as ``orbital``.
ACCESS_MODES = ("all", "selected")
_DEFAULT_ACCESS = {"mode": "all", "agents": []}

# Fields persisted per stored token (all strings, keychain-compatible).
_TOKEN_FIELDS = ("access_token", "refresh_token", "expiry", "scopes", "account")

# Connection-level errors that justify a single reconnect-and-retry. A protocol
# error (McpError) or an application-level tool error (CallToolResult.isError)
# is NOT here — those are surfaced, not retried.
_RECONNECT_ERRORS = (
    anyio.ClosedResourceError,
    anyio.BrokenResourceError,
    anyio.EndOfStream,
    ConnectionError,
    httpx.HTTPError,
)


class ConnectorError(Exception):
    """Raised for connector lifecycle problems (unknown id, not connected,
    not connectable)."""


@dataclass
class Connected:
    """Result of a successful ``connect`` — the provider now has a token."""
    account: str | None


@dataclass
class AuthStart:
    """Reserved for a future non-blocking connect (return an auth URL for the
    client to open). v1 ``connect`` runs the loopback flow inline and returns
    ``Connected``."""
    auth_url: str


class _LiveSession:
    """A cached, open MCP ClientSession plus the exit stack that owns it."""

    def __init__(self, session, stack: AsyncExitStack):
        self.session = session
        self.stack = stack

    async def close(self):
        try:
            await self.stack.aclose()
        except Exception:
            logger.debug("error closing MCP session", exc_info=True)


# Type aliases for the injection seams.
OAuthClientProvider = Callable[[str], OAuthClientConfig | None]
SessionOpener = Callable[[str | None, dict[str, str] | None], "AsyncExitStack"]


class ConnectorManager:
    def __init__(
        self,
        catalog: list[ConnectorManifest] | None = None,
        credential_store=None,
        data_dir: str = "orbital-data",
        *,
        oauth_client_provider: OAuthClientProvider | None = None,
        http_client_factory: Callable[[], httpx.AsyncClient] | None = None,
        open_browser: Callable[[str], None] = webbrowser.open,
        session_opener: SessionOpener | None = None,
        flow_timeout: float = 300.0,
    ):
        self._catalog = catalog if catalog is not None else load_catalog()
        self._credential_store = credential_store
        self._data_dir = data_dir
        self._oauth_client_provider = oauth_client_provider
        self._http_client_factory = http_client_factory or (lambda: httpx.AsyncClient())
        self._open_browser = open_browser
        self._session_opener = session_opener or streamable_http_opener
        self._flow_timeout = flow_timeout

        self._custom: dict[str, ConnectorManifest] = {}
        # connector id -> {"mode": "all"|"selected", "agents": [...]}; an id
        # without an entry is shared with every agent.
        self._access: dict[str, dict] = {}
        self._sessions: dict[str, _LiveSession] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._load_custom()

    # ---- Catalog ----

    def all_manifests(self) -> list[ConnectorManifest]:
        """Bundled catalog + Tier-0 custom servers."""
        return list(self._catalog) + list(self._custom.values())

    def get_manifest(self, connector_id: str) -> ConnectorManifest | None:
        for m in self.all_manifests():
            if m.id == connector_id:
                return m
        return None

    def _require_manifest(self, connector_id: str) -> ConnectorManifest:
        m = self.get_manifest(connector_id)
        if m is None:
            raise ConnectorError(f"unknown connector: {connector_id}")
        return m

    def list_connectors(self) -> list[dict]:
        """``[{manifest, connected, account, enabled_error}]`` for the settings UI."""
        out = []
        for m in self.all_manifests():
            connected, account = self._connection_state(m)
            out.append({
                "manifest": m,
                "connected": connected,
                "account": account,
                "enabled_error": None,
                "access": self.get_access(m.id),
            })
        return out

    # ---- Agent access (connectors shared across agents) ----

    def get_access(self, connector_id: str) -> dict:
        """``{"mode", "agents"}`` for a connector; "all" when never set."""
        entry = self._access.get(connector_id) or _DEFAULT_ACCESS
        return {"mode": entry["mode"], "agents": list(entry["agents"])}

    def set_access(self, connector_id: str, mode: str, agents) -> dict:
        """Persist who may use a connector. ``agents`` is kept under "all"
        too (the last selection, ignored by ``allows``) so the control can
        switch back to it. Raises ``ConnectorError`` for an unknown id and
        ``ValueError`` for a mode that is not one of ``ACCESS_MODES``."""
        self._require_manifest(connector_id)
        if mode not in ACCESS_MODES:
            raise ValueError(f"access mode must be one of {ACCESS_MODES}, not {mode!r}")
        seen: list[str] = []
        for a in agents or []:
            if isinstance(a, str) and a and a not in seen:
                seen.append(a)
        self._access[connector_id] = {"mode": mode, "agents": seen}
        self._save_custom()
        return self.get_access(connector_id)

    def allows(self, connector_id: str, agent_id: str) -> bool:
        entry = self._access.get(connector_id)
        if entry is None or entry["mode"] == "all":
            return True
        return agent_id in entry["agents"]

    def shared_ids(self, enabled_ids, agent_id: str) -> tuple[str, ...]:
        """The connectors ``agent_id`` gets in a project whose enabled list is
        ``enabled_ids``: enabled, connected, and allowed for that agent, in
        the project's order. Reads the credential store (oauth connectors),
        so call it off the event loop where that can block."""
        wanted: list[str] = []
        for cid in enabled_ids or []:
            if isinstance(cid, str) and cid not in wanted:
                wanted.append(cid)
        if not wanted:
            return ()
        out = []
        for cid in wanted:
            m = self.get_manifest(cid)
            if m is None or not self.allows(cid, agent_id):
                continue
            if self._connection_state(m)[0]:
                out.append(cid)
        return tuple(out)

    def _connection_state(self, m: ConnectorManifest) -> tuple[bool, str | None]:
        if m.auth_type == "none":
            return True, None  # nothing to authenticate; live once it exists
        if m.auth_type == "oauth2":
            tok = self._provider_token(m.auth_provider)
            return (tok is not None), (tok[1].account if tok else None)
        # app_password / local_native are reserved (calendar sources) — treat as
        # not-connectable here.
        return False, None

    # ---- Token storage (keyed connector:<auth_provider>:<account>) ----

    def _provider_token(self, auth_provider: str) -> tuple[str, TokenSet] | None:
        """First stored token for a provider (one account per provider in v1)."""
        if self._credential_store is None:
            return None
        prefix = f"connector:{auth_provider}:"
        for meta in self._credential_store.list_all():
            name = meta.get("name", "")
            if name.startswith(prefix):
                fields = {
                    f: (self._credential_store.get_value(name, f) or "")
                    for f in _TOKEN_FIELDS
                }
                return name, TokenSet.from_fields(fields)
        return None

    def _store_token(self, auth_provider: str, tokens: TokenSet) -> None:
        name = f"connector:{auth_provider}:{tokens.account}"
        self._credential_store.store(name, auth_provider, tokens.to_fields())

    # ---- Connect / disconnect ----

    async def connect(self, connector_id: str) -> Connected:
        """Run the OAuth handshake (or no-op for ``none``) and persist the token."""
        m = self._require_manifest(connector_id)
        if m.status == "pending_verification":
            raise ConnectorError(
                f"{m.id} is pending verification and cannot be connected yet"
            )
        if m.auth_type == "none":
            return Connected(account=None)
        if m.auth_type != "oauth2":
            raise ConnectorError(f"auth_type {m.auth_type!r} not supported yet")

        client = self._oauth_client_provider(m.auth_provider) if self._oauth_client_provider else None
        if client is None:
            raise OAuthError(
                f"OAuth client for provider {m.auth_provider!r} is not configured "
                f"(set connector_google_client_id / connector_google_client_secret)"
            )

        existing = self._provider_token(m.auth_provider)
        needed = set(m.oauth_scopes)
        if existing is not None and needed <= set(existing[1].scopes):
            # Provider already authorized with sufficient scope — idempotent,
            # no second sign-in (Spec 011 §0.5-Q5: incremental scopes).
            return Connected(account=existing[1].account)

        scopes = sorted(needed | (set(existing[1].scopes) if existing else set()))
        http = self._http_client_factory()
        try:
            flow = LoopbackOAuthFlow(
                client, http=http, open_browser=self._open_browser, timeout=self._flow_timeout,
            )
            tokens = await flow.authorize(scopes)
        finally:
            await http.aclose()

        self._store_token(m.auth_provider, tokens)
        return Connected(account=tokens.account)

    async def disconnect(self, connector_id: str) -> None:
        """Close the cached session and drop the provider token (all connectors
        sharing that provider go offline — one account per provider in v1)."""
        m = self._require_manifest(connector_id)
        await self._close_session(connector_id)
        if m.auth_type == "oauth2":
            existing = self._provider_token(m.auth_provider)
            if existing is not None:
                self._credential_store.delete(existing[0])

    # ---- Custom (Tier-0) servers ----

    async def add_custom(self, name: str, url: str, auth_type: str = "none") -> ConnectorManifest:
        m = custom_manifest(name, url, auth_type)
        if self.get_manifest(m.id) is not None:
            raise ConnectorError(f"connector {m.id} already exists")
        self._custom[m.id] = m
        self._save_custom()
        return m

    async def remove_custom(self, connector_id: str) -> None:
        if connector_id not in self._custom:
            raise ConnectorError(f"{connector_id} is not a custom connector")
        m = self._custom.pop(connector_id)
        self._access.pop(connector_id, None)
        self._save_custom()
        await self._close_session(connector_id)
        if m.auth_type == "oauth2":
            existing = self._provider_token(m.auth_provider)
            if existing is not None:
                self._credential_store.delete(existing[0])

    def _custom_path(self) -> str:
        return os.path.join(self._data_dir, "connectors.json")

    def _load_custom(self) -> None:
        path = self._custom_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for d in data.get("custom", []):
                m = manifest_from_dict(d)
                self._custom[m.id] = m
        except (json.JSONDecodeError, OSError, ValueError, AttributeError):
            logger.warning("failed to load custom connectors from %s", path, exc_info=True)
            return
        access = data.get("access") if isinstance(data, dict) else None
        for cid, entry in (access.items() if isinstance(access, dict) else ()):
            if not (isinstance(entry, dict) and entry.get("mode") in ACCESS_MODES):
                continue
            agents = entry.get("agents")
            self._access[cid] = {
                "mode": entry["mode"],
                "agents": [a for a in (agents if isinstance(agents, list) else [])
                           if isinstance(a, str) and a],
            }

    def _save_custom(self) -> None:
        # Atomic: a truncated file would load as no custom connectors and
        # the next save would make that permanent. ``access`` is written only
        # once the user has set one, so a file nobody touched keeps the shape
        # older builds wrote (they read ``custom`` and ignore the rest).
        data: dict = {"custom": [m.to_dict() for m in self._custom.values()]}
        if self._access:
            data["access"] = {cid: dict(e) for cid, e in self._access.items()}
        atomic_write_text(self._custom_path(), json.dumps(
            data, indent=2, ensure_ascii=False,
        ))

    # ---- MCP sessions (lazy, cached, reconnect-on-drop) ----

    async def session(self, connector_id: str):
        """Return a cached, initialized ClientSession, opening one if needed."""
        m = self._require_manifest(connector_id)
        lock = self._locks.setdefault(connector_id, asyncio.Lock())
        async with lock:
            live = self._sessions.get(connector_id)
            if live is not None:
                return live.session
            headers = await self._auth_headers(m)
            stack = AsyncExitStack()
            try:
                cm = self._session_opener(m.server_url, headers)
                session = await stack.enter_async_context(cm)
                await session.initialize()
            except BaseException:
                await stack.aclose()
                raise
            self._sessions[connector_id] = _LiveSession(session, stack)
            return session

    async def list_tools(self, connector_id: str):
        """Return the remote server's tools (``list[mcp.types.Tool]``)."""
        result = await self._with_session(connector_id, lambda s: s.list_tools())
        return result.tools

    async def call_tool(self, connector_id: str, tool: str, args: dict | None = None):
        """Invoke a remote tool, returning the MCP ``CallToolResult``."""
        return await self._with_session(
            connector_id, lambda s: s.call_tool(tool, args or {})
        )

    @asynccontextmanager
    async def _scoped_session(self, connector_id: str):
        """A fresh session opened and closed within the CALLING task.

        The mcp SDK's streamable-HTTP transport is an anyio context manager
        bound to the task that enters it. The daemon serves each API request
        in its own task, so a session cached across requests corrupts the
        transport (observed: Google MCP 400s + BaseExceptionGroup escaping to
        a 500). Per-call sessions trade a handshake (~1-2s) for correctness;
        callers that need throughput cache RESULTS (see CalendarHub's TTL).
        """
        m = self._require_manifest(connector_id)
        headers = await self._auth_headers(m)
        async with AsyncExitStack() as stack:
            session = await stack.enter_async_context(
                self._session_opener(m.server_url, headers))
            await session.initialize()
            yield session

    async def _with_session(self, connector_id: str, fn: Callable[[object], Awaitable]):
        try:
            async with self._scoped_session(connector_id) as session:
                return await fn(session)
        except _RECONNECT_ERRORS:
            # Server dropped mid-call — retry once on a brand-new session.
            logger.info("MCP session for %s dropped; retrying once", connector_id)
            async with self._scoped_session(connector_id) as session:
                return await fn(session)

    async def _close_session(self, connector_id: str) -> None:
        live = self._sessions.pop(connector_id, None)
        if live is not None:
            await live.close()

    async def _auth_headers(self, m: ConnectorManifest) -> dict[str, str] | None:
        if m.auth_type == "none":
            return None
        if m.auth_type == "oauth2":
            tok = self._provider_token(m.auth_provider)
            if tok is None:
                raise ConnectorError(f"{m.id} is not connected")
            name, tokens = tok
            if tokens.expired and tokens.refresh_token:
                tokens = await self._refresh(m, tokens)
            return {"Authorization": f"Bearer {tokens.access_token}"}
        raise ConnectorError(f"auth_type {m.auth_type!r} has no session transport")

    async def _refresh(self, m: ConnectorManifest, tokens: TokenSet) -> TokenSet:
        client = self._oauth_client_provider(m.auth_provider) if self._oauth_client_provider else None
        if client is None:
            return tokens  # cannot refresh without client config; use as-is
        http = self._http_client_factory()
        try:
            flow = LoopbackOAuthFlow(client, http=http)
            new = await flow.refresh(tokens.refresh_token, tokens.scopes, account=tokens.account)
        finally:
            await http.aclose()
        self._store_token(m.auth_provider, new)
        return new

    async def aclose(self) -> None:
        """Close all cached sessions (daemon shutdown)."""
        for cid in list(self._sessions):
            await self._close_session(cid)
