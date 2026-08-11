# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""TokenDance one-click API-key provisioning (Spec 47 Tier 2).

OpenRouter-style PKCE flow against TokenDance's live endpoints
(tokendance.space/docs/api-key-oauth): an ephemeral loopback listener is the
redirect target, the system browser drives login + consent, and the one-time
authorization code (≤10 min) is exchanged over HTTPS for a freshly minted key.

Two deliberate deviations from the Google flow in ``connectors/oauth.py``:

- TokenDance has no first-class ``state`` parameter. Their sanctioned
  mechanism is that any query params already present on ``callback_url`` are
  preserved verbatim on redirect (``code`` is appended), so the CSRF nonce
  rides inside the callback URL and is verified on return exactly as before.
- The exchange is a JSON POST with no client id/secret (public native client,
  RFC 8252) returning ``{"key": …}`` — not an OAuth token response.

PKCE is optional on their side for callback-mode apps; Orbital always sends
S256, so a snooped code is useless without the in-process verifier (their
docs: a wrong verifier does not consume the code; a correct exchange burns it).

The minted key is validated against the balance endpoint before it is
accepted — a key that cannot authenticate is never returned, never persisted.
The raw key is never logged and never appears in any URL.
"""

import asyncio
import logging
import secrets
import webbrowser
from typing import Callable
from urllib.parse import urlencode, urlparse, parse_qs

import httpx

from agent_os.connectors.oauth import generate_pkce

logger = logging.getLogger(__name__)

AUTHORIZE_URL = "https://tokendance.space/auth"
EXCHANGE_URL = "https://tokendance.space/portal/api/v1/auth/keys"
BALANCE_URL = "https://tokendance.space/portal/api/v1/user/balance"

# Key-dimension attribution. TokenDance treats the OAuth ``app_url`` and the
# per-request ``X-Site-URL`` header as the SAME application identifier, so
# this value must equal the registry entry's extra_headers["X-Site-URL"]
# (pinned together by a unit test).
APP_URL = "https://github.com/zqiren/Orbital"
KEY_NAME = "Orbital"


class TokenDanceProvisioningError(Exception):
    """Raised when the provisioning flow cannot complete (denied, state
    mismatch, timeout, exchange failure, or the minted key failing
    validation)."""


async def provision_api_key(
    *,
    http: httpx.AsyncClient | None = None,
    open_browser: Callable[[str], None] = webbrowser.open,
    host: str = "127.0.0.1",
    timeout: float = 300.0,
) -> str:
    """Run consent → code → key exchange → balance validation; return the key.

    Network and browser are injectable exactly like ``LoopbackOAuthFlow`` so
    tests never touch tokendance.space.
    """
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
                "Content-Type: text/html; charset=utf-8\r\n"
                f"Content-Length: {len(body)}\r\n"
                "Connection: close\r\n\r\n"
                f"{body}".encode()
            )
            await writer.drain()
        except Exception:
            logger.debug("tokendance loopback handler error", exc_info=True)
        finally:
            writer.close()
            done.set()

    client = http or httpx.AsyncClient()
    owns_http = http is None
    try:
        server = await asyncio.start_server(_handle, host, 0)
        port = server.sockets[0].getsockname()[1]
        # CSRF nonce embedded in the callback URL's query — TokenDance
        # preserves existing query params and appends `code` on redirect.
        callback_url = f"http://{host}:{port}/callback?state={state}"
        auth_url = AUTHORIZE_URL + "?" + urlencode({
            "callback_url": callback_url,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "app_url": APP_URL,
            "key_name": KEY_NAME,
        })

        async with server:
            await server.start_serving()
            # webbrowser.open can silently no-op (detached/headless daemon);
            # log the URL so an operator can open it by hand. The URL carries
            # no secret (the verifier never leaves this process).
            logger.info(
                "TokenDance consent URL (open manually if no browser appeared): %s",
                auth_url,
            )
            open_browser(auth_url)
            try:
                await asyncio.wait_for(done.wait(), timeout=timeout)
            except asyncio.TimeoutError as exc:
                raise TokenDanceProvisioningError(
                    "sign-in timed out waiting for the browser redirect"
                ) from exc

        if captured.get("error"):
            raise TokenDanceProvisioningError(
                f"authorization denied: {captured['error']}"
            )
        if captured.get("state") != state:
            raise TokenDanceProvisioningError(
                "state mismatch on callback — possible CSRF; aborting"
            )
        if not captured.get("code"):
            raise TokenDanceProvisioningError("no authorization code returned")

        key = await _exchange_code(client, captured["code"], verifier)
        await _validate_key(client, key)
        return key
    finally:
        if owns_http:
            await client.aclose()


async def _exchange_code(client: httpx.AsyncClient, code: str, verifier: str) -> str:
    try:
        resp = await client.post(
            EXCHANGE_URL,
            json={
                "code": code,
                "code_verifier": verifier,
                "code_challenge_method": "S256",
            },
        )
    except httpx.HTTPError as exc:
        raise TokenDanceProvisioningError(f"key exchange request failed: {exc}") from exc
    if resp.status_code == 403:
        # Their documented 403: code invalid/expired/already used, or
        # verifier mismatch. The code is one-time and ≤10 minutes old.
        raise TokenDanceProvisioningError(
            "key exchange rejected (code invalid, expired, already used, "
            "or PKCE verifier mismatch) — start the sign-in again"
        )
    if resp.status_code >= 400:
        raise TokenDanceProvisioningError(
            f"key exchange returned HTTP {resp.status_code}"
        )
    try:
        key = resp.json().get("key")
    except Exception as exc:
        raise TokenDanceProvisioningError("key exchange returned non-JSON body") from exc
    if not key or not isinstance(key, str):
        raise TokenDanceProvisioningError("key exchange response missing 'key'")
    return key


async def _validate_key(client: httpx.AsyncClient, key: str) -> None:
    """Reject a minted key that cannot authenticate (their open-platform
    balance endpoint; 401 = invalid/expired key). Network trouble here is a
    failure too — an unvalidated key is never accepted."""
    try:
        resp = await client.get(
            BALANCE_URL, headers={"Authorization": f"Bearer {key}"}
        )
    except httpx.HTTPError as exc:
        raise TokenDanceProvisioningError(
            f"minted key could not be validated: {exc}"
        ) from exc
    if resp.status_code != 200:
        raise TokenDanceProvisioningError(
            f"minted key failed validation (balance endpoint returned "
            f"HTTP {resp.status_code})"
        )
