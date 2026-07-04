# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""PKCE loopback OAuth 2.1 flow for connectors (Spec 011 §2, §4; Task B2).

One generic flow serves every ``oauth2`` connector: an ephemeral localhost
listener is the redirect target, the system browser drives consent, and the
authorization code is exchanged (with a PKCE verifier) for tokens that land in
the OS keychain via ``UserCredentialStore``.

Everything network-facing is injectable so tests never touch Google: the token
endpoint is reached through an injectable ``httpx.AsyncClient`` (a
``MockTransport`` in tests) and the "browser" is a callable (a thread replaying
the redirect in tests). Client id/secret + endpoints are supplied by the caller
(``ConnectorManager``) — first-party or BYO — so no credentials are baked in.
"""

import asyncio
import base64
import hashlib
import json
import logging
import secrets
import time
import webbrowser
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlencode, urlparse, parse_qs

import httpx

logger = logging.getLogger(__name__)


class OAuthError(Exception):
    """Raised when the OAuth flow cannot complete (state mismatch, provider
    error, timeout, non-2xx token exchange)."""


@dataclass(frozen=True)
class OAuthEndpoints:
    authorize_url: str
    token_url: str


# Google's public OAuth 2.0 endpoints. Constant — the per-connector variation
# is only client id/secret + scopes, which arrive via OAuthClientConfig.
GOOGLE_ENDPOINTS = OAuthEndpoints(
    authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
    token_url="https://oauth2.googleapis.com/token",
)


@dataclass(frozen=True)
class OAuthClientConfig:
    """Everything the flow needs to talk to one provider's OAuth server."""
    client_id: str
    client_secret: str
    endpoints: OAuthEndpoints = GOOGLE_ENDPOINTS


@dataclass(frozen=True)
class TokenSet:
    """A stored credential for one ``(auth_provider, account)`` (Spec 011 §0.5-Q8)."""
    access_token: str
    refresh_token: str | None
    expiry: float               # epoch seconds
    scopes: list[str]
    account: str

    @property
    def expired(self) -> bool:
        # 30s skew so a token that is about to expire is refreshed proactively.
        return time.time() >= (self.expiry - 30)

    def to_fields(self) -> dict[str, str]:
        """Flatten to string fields for keychain storage (values are str→str)."""
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token or "",
            "expiry": repr(self.expiry),
            "scopes": json.dumps(self.scopes),
            "account": self.account,
        }

    @classmethod
    def from_fields(cls, fields: dict[str, str]) -> "TokenSet":
        return cls(
            access_token=fields.get("access_token", ""),
            refresh_token=fields.get("refresh_token") or None,
            expiry=float(fields.get("expiry", "0") or 0),
            scopes=json.loads(fields.get("scopes") or "[]"),
            account=fields.get("account", ""),
        )


def generate_pkce() -> tuple[str, str]:
    """Return ``(verifier, challenge)`` — RFC 7636 S256."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(48)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).rstrip(b"=").decode()
    return verifier, challenge


def _account_from_id_token(id_token: str | None) -> str | None:
    """Best-effort email claim out of a JWT id_token (no signature check —
    the token came straight from the token endpoint over TLS)."""
    if not id_token or id_token.count(".") != 2:
        return None
    try:
        payload_b64 = id_token.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload_b64))
        return claims.get("email") or claims.get("sub")
    except Exception:
        return None


class LoopbackOAuthFlow:
    """Authorization-code-with-PKCE flow over a localhost loopback redirect."""

    def __init__(
        self,
        client: OAuthClientConfig,
        *,
        http: httpx.AsyncClient | None = None,
        open_browser: Callable[[str], None] = webbrowser.open,
        host: str = "127.0.0.1",
        timeout: float = 300.0,
    ):
        self._client = client
        self._http = http or httpx.AsyncClient()
        self._owns_http = http is None
        self._open_browser = open_browser
        self._host = host
        self._timeout = timeout

    async def authorize(self, scopes: list[str]) -> TokenSet:
        """Run the full consent → code → token exchange, returning a TokenSet."""
        verifier, challenge = generate_pkce()
        state = secrets.token_urlsafe(24)
        captured: dict[str, str] = {}
        done = asyncio.Event()

        async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            try:
                request_line = await reader.readline()
                target = request_line.decode(errors="replace").split(" ")
                if len(target) >= 2:
                    query = parse_qs(urlparse(target[1]).query)
                    for key in ("code", "state", "error"):
                        if key in query:
                            captured[key] = query[key][0]
                body = (
                    "<html><body style='font-family:sans-serif'>"
                    "<h2>Orbital — you can close this tab.</h2>"
                    "</body></html>"
                )
                writer.write(
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: text/html\r\n"
                    f"Content-Length: {len(body)}\r\n"
                    "Connection: close\r\n\r\n"
                    f"{body}".encode()
                )
                await writer.drain()
            except Exception:
                logger.debug("loopback handler error", exc_info=True)
            finally:
                writer.close()
                done.set()

        server = await asyncio.start_server(_handle, self._host, 0)
        port = server.sockets[0].getsockname()[1]
        redirect_uri = f"http://{self._host}:{port}/callback"
        auth_url = self._build_authorize_url(scopes, state, challenge, redirect_uri)

        try:
            async with server:
                await server.start_serving()
                self._open_browser(auth_url)
                try:
                    await asyncio.wait_for(done.wait(), timeout=self._timeout)
                except asyncio.TimeoutError as exc:
                    raise OAuthError("OAuth timed out waiting for the browser redirect") from exc
        finally:
            if self._owns_http:
                pass  # keep client for exchange below; closed by caller/context

        if captured.get("error"):
            raise OAuthError(f"authorization denied: {captured['error']}")
        if captured.get("state") != state:
            raise OAuthError("OAuth state mismatch — possible CSRF; aborting")
        if not captured.get("code"):
            raise OAuthError("no authorization code returned")

        return await self._exchange_code(captured["code"], verifier, redirect_uri, scopes)

    async def refresh(
        self, refresh_token: str, scopes: list[str], *, account: str
    ) -> TokenSet:
        """Exchange a refresh token for a fresh access token."""
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self._client.client_id,
            "client_secret": self._client.client_secret,
        }
        body = await self._post_token(data)
        # Google omits refresh_token on refresh — carry the old one forward.
        return self._token_set(body, scopes, fallback_refresh=refresh_token, fallback_account=account)

    def _build_authorize_url(
        self, scopes: list[str], state: str, challenge: str, redirect_uri: str
    ) -> str:
        params = {
            "response_type": "code",
            "client_id": self._client.client_id,
            "redirect_uri": redirect_uri,
            "scope": " ".join(scopes),
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "access_type": "offline",   # ask Google for a refresh_token
            "prompt": "consent",
        }
        return f"{self._client.endpoints.authorize_url}?{urlencode(params)}"

    async def _exchange_code(
        self, code: str, verifier: str, redirect_uri: str, scopes: list[str]
    ) -> TokenSet:
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": self._client.client_id,
            "client_secret": self._client.client_secret,
            "code_verifier": verifier,
        }
        body = await self._post_token(data)
        return self._token_set(body, scopes)

    async def _post_token(self, data: dict) -> dict:
        try:
            resp = await self._http.post(self._client.endpoints.token_url, data=data)
        except httpx.HTTPError as exc:
            raise OAuthError(f"token endpoint request failed: {exc}") from exc
        if resp.status_code >= 400:
            raise OAuthError(f"token endpoint returned {resp.status_code}: {resp.text[:200]}")
        try:
            return resp.json()
        except Exception as exc:
            raise OAuthError("token endpoint returned non-JSON body") from exc

    def _token_set(
        self,
        body: dict,
        requested_scopes: list[str],
        *,
        fallback_refresh: str | None = None,
        fallback_account: str | None = None,
    ) -> TokenSet:
        if "access_token" not in body:
            raise OAuthError(f"token response missing access_token: {list(body)}")
        scope_str = body.get("scope")
        scopes = scope_str.split() if scope_str else list(requested_scopes)
        account = (
            body.get("email")
            or body.get("account")
            or _account_from_id_token(body.get("id_token"))
            or fallback_account
            or "unknown"
        )
        return TokenSet(
            access_token=body["access_token"],
            refresh_token=body.get("refresh_token") or fallback_refresh,
            expiry=time.time() + float(body.get("expires_in", 3600)),
            scopes=scopes,
            account=account,
        )
