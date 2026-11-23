# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Sign in with ChatGPT — the OFFICIAL open-source token-sharing route.

https://developers.openai.com/siwc/token-sharing-open-source (sign-in,
profiles-and-sessions, models-and-inference, errors-and-recovery).

Orbital registers itself dynamically as the agent "Orbital"
(``client_id=dynamic_agent_client`` + ``agent_name_hint``) with a stable
per-install ``ext_agent_host_id``; OpenAI issues a per-user client id on the
callback, which is saved with the tokens and used for every later exchange,
refresh, re-authorization and revocation. Inference goes to the public
``https://api.openai.com/v1`` (Responses API), never ChatGPT's backend-api.

What this module deliberately does NOT do: use the Codex CLI's client id or
its fixed port, call ``chatgpt.com/backend-api/codex``, or read any other
tool's saved tokens.

Flow (PKCE public client, RFC 8252 loopback):
  listener on ``127.0.0.1:<random>/auth/callback`` (127.0.0.1, never
  ``localhost``; only the port may vary between sign-ins) → system browser →
  callback ``code, state, client_id, scope, iss`` → form exchange with
  ``resource`` → ID token verified against OpenAI's JWKS (signature, iss,
  aud = issued client id, exp, nonce) → ``chatgpt.tokens.use.direct`` must be
  among the granted scopes.

Nothing here logs a token, a code, a verifier or the account email.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import time
import uuid
import webbrowser
from typing import Callable
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from agent_os.connectors.oauth import generate_pkce

logger = logging.getLogger(__name__)

PROVIDER_KEY = "chatgpt"
ISSUER = "https://auth.openai.com"
AUTHORIZE_URL = f"{ISSUER}/api/accounts/authorize"
TOKEN_URL = f"{ISSUER}/api/accounts/oauth/token"
# ``revocation_endpoint`` of https://auth.openai.com/.well-known/openid-configuration
REVOKE_URL = f"{ISSUER}/api/accounts/oauth/revoke"
JWKS_URL = f"{ISSUER}/.well-known/jwks.json"
RESOURCE = "https://api.openai.com/v1"
API_BASE = "https://api.openai.com/v1"
MODELS_URL = f"{API_BASE}/models"
SHARING_SCOPE = "chatgpt.tokens.use.direct"
SCOPE = f"openid profile email offline_access resource.invoke {SHARING_SCOPE}"
REGISTRATION_CLIENT_ID = "dynamic_agent_client"
AGENT_NAME = "Orbital"
CALLBACK_PATH = "/auth/callback"
USER_AGENT = "Orbital"
HOST_ID_FILE = "chatgpt-host-id"
SIGNIN_TIMEOUT_S = 600.0

# Refresh-token errors that mean the token set is unusable for good
# (errors-and-recovery: "Clear the unusable tokens and repeat OAuth").
TERMINAL_REFRESH_ERRORS = frozenset({
    "invalid_grant", "invalid_refresh_token", "token_expired",
    "refresh_token_expired", "refresh_token_invalidated", "refresh_token_reused",
    "invalid_client",
})

NEEDS_SIGNIN_MESSAGE = (
    "Your ChatGPT sign-in has ended (it was revoked, expired after 30 days, or "
    "Orbital was disconnected in ChatGPT settings). Sign in with ChatGPT again."
)


class ChatGPTAuthError(Exception):
    """A sign-in / refresh / revoke failure with a stable ``code``.

    ``status`` is the HTTP status the daemon route answers with; ``terminal``
    (refresh only) means the saved token set can never work again.
    """

    def __init__(self, code: str, message: str, *, status: int = 502,
                 terminal: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.terminal = terminal


# ---------------------------------------------------------------------------
# Host id: one per install, opaque, never a secret
# ---------------------------------------------------------------------------

def load_or_create_host_id(data_dir: str) -> str:
    """The stable ``ext_agent_host_id`` for this install (``urn:uuid:<v4>``).

    Kept in its own file in the data dir — not settings.json, which an older
    build rewrites through a model that would drop an unknown field.
    """
    path = os.path.join(data_dir, HOST_ID_FILE)
    try:
        with open(path, "r", encoding="utf-8") as f:
            value = f.read().strip()
        if value:
            return value
    except OSError:
        pass
    value = f"urn:uuid:{uuid.uuid4()}"
    os.makedirs(data_dir, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(value)
    os.replace(tmp, path)
    return value


# ---------------------------------------------------------------------------
# Token bundle helpers
# ---------------------------------------------------------------------------

def _scopes(raw) -> list[str]:
    if isinstance(raw, list):
        return [str(s) for s in raw]
    return [s for s in str(raw or "").split() if s]


def bundle_from_token_response(tok: dict, *, client_id: str, host_id: str,
                               claims: dict, now: float | None = None) -> dict:
    """The fields Orbital keeps for one ChatGPT sign-in (the card secret)."""
    now = time.time() if now is None else now
    return {
        "provider": PROVIDER_KEY,
        "client_id": client_id,
        "host_id": host_id,
        "sub": claims.get("sub") or "",
        "email": claims.get("email") or "",
        "access_token": tok.get("access_token") or "",
        "refresh_token": tok.get("refresh_token") or "",
        "id_token": tok.get("id_token") or "",
        "expires_at": now + float(tok.get("expires_in") or 3600),
        "scopes": _scopes(tok.get("scope")),
    }


def _error_code(resp: httpx.Response) -> tuple[str, str]:
    try:
        body = resp.json()
    except Exception:
        return "", ""
    if not isinstance(body, dict):
        return "", ""
    err = body.get("error")
    if isinstance(err, dict):
        return str(err.get("code") or err.get("type") or ""), str(err.get("message") or "")
    return str(err or ""), str(body.get("error_description") or body.get("detail") or "")


# ---------------------------------------------------------------------------
# ID token
# ---------------------------------------------------------------------------

async def verify_id_token(id_token: str, *, client_id: str, nonce: str | None,
                          http: httpx.AsyncClient) -> dict:
    """Verify signature (OpenAI's JWKS), issuer, audience, expiry and nonce.

    The JWKS is fetched with httpx (certifi) — the stdlib client could not
    verify OpenAI's certificate in this venv, and the packaged app ships the
    same httpx/certifi pair the providers already use.
    """
    try:
        import jwt  # PyJWT[crypto]; pulled in by mcp
        from jwt.algorithms import RSAAlgorithm
    except Exception as exc:  # pragma: no cover - packaging guard
        raise ChatGPTAuthError(
            "signin_failed",
            "Orbital cannot verify the ChatGPT sign-in on this build "
            "(the token-verification library is missing).") from exc
    if not id_token:
        raise ChatGPTAuthError("signin_failed",
                               "OpenAI returned no ID token for this sign-in.")
    try:
        header = jwt.get_unverified_header(id_token)
    except Exception as exc:
        raise ChatGPTAuthError("signin_failed",
                               "OpenAI returned an unreadable ID token.") from exc
    try:
        resp = await http.get(JWKS_URL)
        resp.raise_for_status()
        keys = resp.json().get("keys") or []
    except Exception as exc:
        raise ChatGPTAuthError(
            "signin_failed",
            f"Could not fetch OpenAI's signing keys to check the sign-in: "
            f"{type(exc).__name__}.") from exc
    kid = header.get("kid")
    jwk = next((k for k in keys if not kid or k.get("kid") == kid), None)
    if jwk is None:
        raise ChatGPTAuthError("signin_failed",
                               "The ChatGPT sign-in was signed with an unknown key.")
    try:
        key = RSAAlgorithm.from_jwk(jwk)
        claims = jwt.decode(
            id_token, key, algorithms=["RS256"], audience=client_id,
            issuer=ISSUER, options={"require": ["exp", "iat", "sub", "aud", "iss"]},
        )
    except Exception as exc:
        raise ChatGPTAuthError(
            "signin_failed",
            f"The ChatGPT sign-in did not pass verification ({type(exc).__name__}).",
        ) from exc
    if nonce is not None and claims.get("nonce") != nonce:
        raise ChatGPTAuthError("signin_failed",
                               "The ChatGPT sign-in did not match this attempt (nonce).")
    return claims


# ---------------------------------------------------------------------------
# Sign-in
# ---------------------------------------------------------------------------

_DONE_PAGE = (
    "<html><head><meta charset='utf-8'><title>Orbital</title></head>"
    "<body style='font-family:-apple-system,sans-serif;padding:48px'>"
    "<h2>Orbital received your ChatGPT sign-in.</h2>"
    "<p>You can close this tab and go back to Orbital.</p></body></html>"
)


async def sign_in(*, host_id: str,
                  http: httpx.AsyncClient | None = None,
                  open_browser: Callable[[str], object] = webbrowser.open,
                  timeout: float = SIGNIN_TIMEOUT_S,
                  cancel: asyncio.Event | None = None,
                  client_id: str | None = None,
                  id_token_hint: str | None = None,
                  expected_sub: str | None = None,
                  host: str = "127.0.0.1",
                  on_url: Callable[[str], object] | None = None) -> dict:
    """Run the browser sign-in and return the verified token bundle.

    ``client_id`` (+ ``id_token_hint``) re-authorizes a saved registration
    instead of registering again; ``expected_sub`` then refuses a different
    ChatGPT account (the docs: never overwrite one registration with
    another's tokens).
    """
    verifier, challenge = generate_pkce()
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)
    captured: dict[str, str] = {}
    done = asyncio.Event()

    async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            line = (await reader.readline()).decode(errors="replace")
            while True:  # drain headers
                h = await reader.readline()
                if not h or h in (b"\r\n", b"\n"):
                    break
            parts = line.split(" ")
            path, query = "", {}
            if len(parts) >= 2:
                u = urlparse(parts[1])
                path = u.path
                query = {k: v[0] for k, v in parse_qs(u.query).items()}
            if path != CALLBACK_PATH:
                writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n"
                             b"Connection: close\r\n\r\n")
                await writer.drain()
                return
            if not done.is_set():
                captured.update({k: query[k] for k in (
                    "code", "state", "client_id", "scope", "iss", "error",
                    "error_description") if k in query})
            body = _DONE_PAGE.encode()
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n"
                         + f"Content-Length: {len(body)}\r\n".encode()
                         + b"Connection: close\r\n\r\n" + body)
            await writer.drain()
            done.set()
        except Exception:
            logger.debug("chatgpt loopback handler error", exc_info=True)
        finally:
            writer.close()

    client = http or httpx.AsyncClient(timeout=30, headers={"User-Agent": USER_AGENT})
    owns_http = http is None
    try:
        server = await asyncio.start_server(_handle, host, 0)
        port = server.sockets[0].getsockname()[1]
        redirect_uri = f"http://{host}:{port}{CALLBACK_PATH}"
        params = {
            "client_id": client_id or REGISTRATION_CLIENT_ID,
            "ext_agent_host_id": host_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": SCOPE,
            "resource": RESOURCE,
            "state": state,
            "nonce": nonce,
            "code_challenge_method": "S256",
            "code_challenge": challenge,
        }
        if client_id:
            if id_token_hint:
                params["id_token_hint"] = id_token_hint
        else:
            params["agent_name_hint"] = AGENT_NAME
        auth_url = f"{AUTHORIZE_URL}?{urlencode(params)}"
        async with server:
            await server.start_serving()  # listener first, then the browser
            logger.info("ChatGPT sign-in: waiting for the browser callback on "
                        "127.0.0.1:%d (URL carries no secret): %s", port, auth_url)
            if on_url is not None:
                on_url(auth_url)  # the route keeps it for "reopen browser"
            open_browser(auth_url)
            waiters = [asyncio.ensure_future(done.wait())]
            if cancel is not None:
                waiters.append(asyncio.ensure_future(cancel.wait()))
            try:
                finished, pending = await asyncio.wait(
                    waiters, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for w in waiters:
                    if not w.done():
                        w.cancel()
            if not done.is_set():
                if cancel is not None and cancel.is_set():
                    raise ChatGPTAuthError("signin_cancelled",
                                           "The ChatGPT sign-in was cancelled.",
                                           status=499)
                raise ChatGPTAuthError(
                    "signin_timeout",
                    f"The ChatGPT sign-in was not finished in the browser within "
                    f"{int(timeout // 60)} minutes.", status=504)

        if captured.get("state") != state:
            raise ChatGPTAuthError("signin_failed",
                                   "The ChatGPT sign-in did not match this attempt (state).")
        if captured.get("error"):
            err = captured["error"]
            desc = captured.get("error_description") or ""
            if err == "access_denied":
                raise ChatGPTAuthError(
                    "signin_denied",
                    "The ChatGPT sign-in was cancelled or refused in the browser."
                    + (f" OpenAI said: {desc}" if desc else ""))
            raise ChatGPTAuthError(
                "signin_denied",
                f"OpenAI refused the ChatGPT sign-in ({err})."
                + (f" {desc}" if desc else ""))
        if captured.get("iss") and captured["iss"].rstrip("/") != ISSUER:
            raise ChatGPTAuthError("signin_failed",
                                   "The ChatGPT sign-in came back from an unexpected issuer.")
        issued = captured.get("client_id") or client_id
        if client_id and captured.get("client_id") and captured["client_id"] != client_id:
            raise ChatGPTAuthError(
                "signin_failed",
                "OpenAI answered for a different Orbital registration; the saved "
                "sign-in was left untouched. Try again.")
        if not issued:
            raise ChatGPTAuthError("signin_failed",
                                   "OpenAI did not issue a client id for Orbital.")
        if not captured.get("code"):
            raise ChatGPTAuthError("signin_failed",
                                   "OpenAI returned no authorization code.")

        try:
            resp = await client.post(TOKEN_URL, data={
                "grant_type": "authorization_code",
                "client_id": issued,
                "code": captured["code"],
                "code_verifier": verifier,
                "redirect_uri": redirect_uri,
                "resource": RESOURCE,
            })
        except httpx.HTTPError as exc:
            raise ChatGPTAuthError(
                "signin_failed",
                f"Could not reach OpenAI to finish the sign-in ({type(exc).__name__}).") from exc
        if resp.status_code != 200:
            code, desc = _error_code(resp)
            raise ChatGPTAuthError(
                "signin_failed",
                f"OpenAI did not accept the sign-in code (HTTP {resp.status_code}"
                + (f", {code}" if code else "") + "). Start the sign-in again.")
        tok = resp.json()
        claims = await verify_id_token(tok.get("id_token") or "", client_id=issued,
                                       nonce=nonce, http=client)
        if expected_sub and claims.get("sub") != expected_sub:
            raise ChatGPTAuthError(
                "signin_failed",
                "You signed in with a different ChatGPT account than this card's. "
                "The card was left as it was; add the other account as a new card.",
                status=409)
        scopes = _scopes(tok.get("scope") or captured.get("scope"))
        if SHARING_SCOPE not in scopes:
            raise ChatGPTAuthError(
                "plan_usage_not_granted",
                "ChatGPT signed you in but did not let Orbital use your plan (the "
                "plan-usage permission was not granted). Sign in again and allow "
                "plan usage, or use an API key card instead.", status=403)
        if not tok.get("access_token") or not tok.get("refresh_token"):
            raise ChatGPTAuthError("signin_failed",
                                   "OpenAI returned an incomplete token set.")
        tok["scope"] = " ".join(scopes)
        return bundle_from_token_response(tok, client_id=issued, host_id=host_id,
                                          claims=claims)
    finally:
        if owns_http:
            await client.aclose()


# ---------------------------------------------------------------------------
# Refresh / revoke / models
# ---------------------------------------------------------------------------

async def refresh_tokens(bundle: dict, *, http: httpx.AsyncClient | None = None,
                         now: float | None = None) -> dict:
    """One ``refresh_token`` grant. Returns the NEW bundle (rotated refresh
    token, new access token, expiry, scopes, ID token) — the caller persists
    it before using it. ``terminal=True`` on the error means sign in again."""
    client = http or httpx.AsyncClient(timeout=30, headers={"User-Agent": USER_AGENT})
    owns = http is None
    try:
        try:
            resp = await client.post(TOKEN_URL, data={
                "grant_type": "refresh_token",
                "client_id": bundle.get("client_id") or "",
                "refresh_token": bundle.get("refresh_token") or "",
                "resource": RESOURCE,
            })
        except httpx.HTTPError as exc:
            raise ChatGPTAuthError(
                "refresh_unreachable",
                f"Could not reach OpenAI to renew the ChatGPT sign-in "
                f"({type(exc).__name__}).", terminal=False) from exc
        if resp.status_code == 200:
            tok = resp.json()
            if not tok.get("access_token"):
                raise ChatGPTAuthError("refresh_failed",
                                       "OpenAI renewed the sign-in without an access token.")
            new = dict(bundle)
            new["access_token"] = tok["access_token"]
            new["refresh_token"] = tok.get("refresh_token") or bundle.get("refresh_token") or ""
            if tok.get("id_token"):
                new["id_token"] = tok["id_token"]
            if tok.get("scope"):
                new["scopes"] = _scopes(tok["scope"])
            new["expires_at"] = (time.time() if now is None else now) + float(
                tok.get("expires_in") or 3600)
            return new
        code, _desc = _error_code(resp)
        terminal = (code in TERMINAL_REFRESH_ERRORS
                    or resp.status_code in (400, 401, 403) and resp.status_code != 429)
        if terminal:
            raise ChatGPTAuthError("needs_signin", NEEDS_SIGNIN_MESSAGE,
                                   status=401, terminal=True)
        raise ChatGPTAuthError(
            "refresh_failed",
            f"OpenAI could not renew the ChatGPT sign-in right now "
            f"(HTTP {resp.status_code}). Orbital will try again on the next request.",
            status=resp.status_code, terminal=False)
    finally:
        if owns:
            await client.aclose()


async def revoke(bundle: dict, *, http: httpx.AsyncClient | None = None) -> None:
    """Revoke the refresh token at OpenAI (empty 200 even when already
    invalid). Raises ChatGPTAuthError when revocation is not confirmed."""
    client = http or httpx.AsyncClient(timeout=20, headers={"User-Agent": USER_AGENT})
    owns = http is None
    try:
        token = bundle.get("refresh_token") or ""
        if not token:
            return
        try:
            resp = await client.post(REVOKE_URL, data={
                "token": token,
                "token_type_hint": "refresh_token",
                "client_id": bundle.get("client_id") or "",
            })
        except httpx.HTTPError as exc:
            raise ChatGPTAuthError(
                "revoke_failed",
                "Orbital could not reach OpenAI to revoke the sign-in, so remote "
                "revocation is not confirmed. You can disconnect Orbital in ChatGPT "
                "→ Settings → Security.") from exc
        if resp.status_code != 200:
            raise ChatGPTAuthError(
                "revoke_failed",
                f"OpenAI did not confirm the revocation (HTTP {resp.status_code}). "
                "You can disconnect Orbital in ChatGPT → Settings → Security.")
    finally:
        if owns:
            await client.aclose()


def parse_model_list(data) -> list[dict]:
    """``{"models": [{slug, display_name, visibility, context_window, ...}]}``
    → ``[{id, display_name, context_window}]`` in the server's order, keeping
    only ``visibility == "list"`` entries the API route can serve."""
    out = []
    for m in (data or {}).get("models") or []:
        if not isinstance(m, dict):
            continue
        slug = m.get("slug") or m.get("id")
        if not slug:
            continue
        if (m.get("visibility") or "list") != "list":
            continue
        if m.get("supported_in_api") is False:
            continue
        ctx = m.get("context_window")
        out.append({
            "id": str(slug),
            "display_name": m.get("display_name") or None,
            "context_window": int(ctx) if isinstance(ctx, (int, float)) and ctx > 0 else None,
        })
    return out
