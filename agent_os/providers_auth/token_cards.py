# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Sign-in cards: a credential card whose secret is a refreshable token set.

A card's secret is one Keychain item (``card:<id>``). For an API-key card it
is the key. For a sign-in card (registry ``auth: "oauth"``, today only
``chatgpt``) it is ONE JSON object holding access token, refresh token,
expiry, issued client id and the account email — serialized so that:

- it starts with ``{"kind": "oauth-token-set"`` (no API key starts with
  ``{``), which is how ``SettingsStore.key_for`` tells the two apart and hands
  every existing consumer the ACCESS TOKEN, never the bundle;
- it ends with the non-secret ``"v": 1}``, so an older Orbital build that
  masks the raw secret (first 4 + last 4 characters) shows nothing secret.

Freshness lives in ``TokenBroker``: one asyncio lock per card (refresh tokens
ROTATE, so two refreshes must never race), refresh shortly before expiry or
when a request was refused with 401, and the rotated token set is written to
the Keychain BEFORE the new access token is handed out. A refresh refused for
good marks the card "needs sign-in" once and then fails fast without touching
the network again until the user signs in again — no loop, and never a
silent switch to another card (the error is ``final``).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timezone

from agent_os.agent.providers.types import LLMError

logger = logging.getLogger(__name__)

BUNDLE_KIND = "oauth-token-set"
BUNDLE_VERSION = 1
# Refresh this long before the access token's expiry (tokens live 3600 s).
REFRESH_MARGIN_S = 300.0

_SECRET_FIELDS = ("access_token", "refresh_token", "id_token")


def dump_bundle(bundle: dict) -> str:
    """Serialize a token set for the Keychain (kind first, version last)."""
    body = {k: v for k, v in bundle.items() if k not in ("kind", "v")}
    ordered = {"kind": BUNDLE_KIND, **body, "v": BUNDLE_VERSION}
    return json.dumps(ordered, ensure_ascii=True)


def parse_bundle(raw: str | None) -> dict | None:
    """The token set in ``raw``, or None when ``raw`` is a plain API key."""
    if not raw or not raw.startswith("{"):
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or data.get("kind") != BUNDLE_KIND:
        return None
    return data


def is_bundle(raw: str | None) -> bool:
    return parse_bundle(raw) is not None


def _fingerprint(bundle: dict) -> str:
    return hashlib.sha256((bundle.get("refresh_token") or "").encode()).hexdigest()[:16]


class NeedsSignInError(LLMError):
    """The saved sign-in can no longer be used. Final: no retry, no fallback."""

    def __init__(self, message: str):
        super().__init__(message, status_code=401, code="needs_signin", final=True)


def _refresher_for(bundle: dict):
    provider = bundle.get("provider")
    if provider == "chatgpt":
        from agent_os.providers_auth import chatgpt

        return chatgpt.refresh_tokens
    return None


class TokenBroker:
    """Fresh access tokens for sign-in cards, one per card, shared by every
    provider client built from that card (manager, utility, fallback, test).

    ``store`` needs ``secret_for(card_id)``, ``set_card_secret(card_id, raw)``
    and ``record_card_health(card_id, error=...)``.
    """

    def __init__(self, store, *, refresher=None, now=time.time):
        self._store = store
        self._refresher = refresher
        self._now = now
        self._locks: dict[str, tuple[object, asyncio.Lock]] = {}
        # card_id -> refresh-token fingerprint that was refused for good.
        self._dead: dict[str, str] = {}

    def _lock(self, card_id: str) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        entry = self._locks.get(card_id)
        if entry is None or entry[0] is not loop:
            entry = (loop, asyncio.Lock())
            self._locks[card_id] = entry
        return entry[1]

    def bundle(self, card_id: str) -> dict | None:
        return parse_bundle(self._store.secret_for(card_id))

    def forget(self, card_id: str) -> None:
        """Drop per-card state (after sign-out or a fresh sign-in)."""
        self._dead.pop(card_id, None)

    async def access_token(self, card_id: str, *, force: bool = False,
                           failed_token: str | None = None) -> str:
        """A usable access token for ``card_id``.

        ``force`` refreshes even when the token is not near expiry (a 401);
        ``failed_token`` is the token that was refused — if another caller
        already replaced it, the replacement is returned without a second
        refresh.
        """
        async with self._lock(card_id):
            bundle = await asyncio.to_thread(self.bundle, card_id)
            if bundle is None:
                raise NeedsSignInError(
                    "This card has no saved sign-in. Sign in again.")
            fp = _fingerprint(bundle)
            if self._dead.get(card_id) == fp:
                raise NeedsSignInError(self._dead_message(bundle))
            current = bundle.get("access_token") or ""
            if failed_token is not None and current and current != failed_token:
                return current
            fresh = float(bundle.get("expires_at") or 0) - self._now() > REFRESH_MARGIN_S
            if current and fresh and not force:
                return current
            refresher = self._refresher or _refresher_for(bundle)
            if refresher is None:
                raise NeedsSignInError("This card's sign-in type is not supported here.")
            try:
                new = await refresher(bundle)
            except Exception as exc:  # ChatGPTAuthError or transport
                if getattr(exc, "terminal", False):
                    self._dead[card_id] = fp
                    message = getattr(exc, "message", "") or self._dead_message(bundle)
                    await asyncio.to_thread(self._mark_needs_signin, card_id, message)
                    logger.warning("sign-in card %s: refresh refused for good; "
                                   "marked needs sign-in", card_id)
                    raise NeedsSignInError(message) from exc
                message = getattr(exc, "message", None) or str(exc) or type(exc).__name__
                logger.warning("sign-in card %s: refresh failed (transient): %s",
                               card_id, type(exc).__name__)
                raise LLMError(message, status_code=None) from exc
            # Persist the ROTATED set before anyone uses the new access token:
            # losing it would strand the card on a refresh token OpenAI has
            # already replaced.
            await asyncio.to_thread(self._store.set_card_secret, card_id,
                                    dump_bundle(new))
            return new.get("access_token") or ""

    @staticmethod
    def _dead_message(bundle: dict) -> str:
        if bundle.get("provider") == "chatgpt":
            from agent_os.providers_auth.chatgpt import NEEDS_SIGNIN_MESSAGE

            return NEEDS_SIGNIN_MESSAGE
        return "The saved sign-in has ended. Sign in again."

    def _mark_needs_signin(self, card_id: str, message: str) -> None:
        try:
            self._store.record_card_health(card_id, error={
                "status": 401, "code": "needs_signin", "message": message,
                "at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception:
            logger.warning("could not mark card %s as needing sign-in", card_id,
                           exc_info=True)

    def token_source(self, card_id: str):
        """An async ``(force=False, failed_token=None) -> str`` for LLMProvider."""
        async def source(force: bool = False, failed_token: str | None = None) -> str:
            return await self.access_token(card_id, force=force,
                                           failed_token=failed_token)
        source.card_id = card_id  # type: ignore[attr-defined]
        return source
