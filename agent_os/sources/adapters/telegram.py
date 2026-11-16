# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Telegram bot, by Bot API long polling (``getUpdates``) — no webhook, no
public address.

Protocol (https://core.telegram.org/bots/api#getupdates): the daemon asks
``getUpdates?offset=<last update_id + 1>&timeout=50``; the server answers
when something arrives or the timeout passes. Passing the offset confirms
every earlier update, so the next restart starts after it. Updates wait on
Telegram's side (about 24 hours) while nothing polls, which is why this
type is lossless: a daemon that was off catches up and the core records no
gap. The offset is kept in the source's adapter state.

Arrivals: a message in a private chat with the bot (``dm``), a group
message that @mentions the bot (``at``). Groups: with BotFather's default
privacy mode a bot may not see plain @mentions, so the setup prompt asks to
turn privacy off (/setprivacy); Orbital still keeps only messages that
mention the bot.

A webhook set on the bot makes ``getUpdates`` fail. Orbital never removes
somebody's webhook: connect refuses with a reason, a running source fails
with ``webhook_set``.
"""

from __future__ import annotations

import asyncio
import re

from agent_os.sources.adapters import Adapter
from agent_os.sources.adapters._bot_common import (
    Unreachable, endpoint, host_of, http_client, iso_from_epoch, placeholder, redact, split_text,
)
from agent_os.sources.model import Item, SourceFailure, Verified, VerifyError

API = "https://api.telegram.org"
LONG_POLL_S = 50
_TOKEN_RE = re.compile(r"^\d{3,}:[A-Za-z0-9_-]{20,}$")
_MEDIA_KEYS = ("photo", "video", "animation", "document", "audio", "voice", "video_note",
               "sticker", "location", "contact", "poll")

#: Groups: BotFather's default privacy mode may hide plain @mentions from a
#: bot, so the setup adds /setprivacy → Disable (Orbital still keeps only
#: messages that @ the bot).


def _sender(frm: dict) -> str | None:
    name = " ".join(p for p in (frm.get("first_name"), frm.get("last_name")) if p)
    return name or (("@" + frm["username"]) if frm.get("username") else None)


def _link(chat: dict, message_id) -> str | None:
    if chat.get("type") not in ("supergroup", "channel"):
        return None
    if chat.get("username"):
        return f"https://t.me/{chat['username']}/{message_id}"
    cid = str(chat.get("id") or "")
    if cid.startswith("-100"):
        return f"https://t.me/c/{cid[4:]}/{message_id}"
    return None


class TelegramAdapter(Adapter):
    mode = "stream"
    lossless = True
    #: Tests point this at a protocol fake; else the env override, else Telegram.
    api_base: str | None = None
    long_poll_s: int = LONG_POLL_S

    def _base(self) -> str:
        return self.api_base or endpoint("telegram", API)

    async def _call(self, client, token: str, method: str, params: dict | None = None,
                    *, timeout: float | None = None) -> tuple[int, dict]:
        import httpx
        base = self._base()
        try:
            r = await client.post(f"{base}/bot{token}/{method}", json=params or {},
                                  timeout=timeout if timeout is not None else 20.0)
        except httpx.HTTPError:
            raise Unreachable(host_of(base)) from None
        try:
            data = r.json()
        except ValueError:
            data = {}
        return r.status_code, data if isinstance(data, dict) else {}

    # -- connect ------------------------------------------------------------

    async def verify(self, ctx) -> Verified:
        token = (ctx.secret("token") or "").strip()
        if not _TOKEN_RE.match(token):
            raise VerifyError("invalid", "token", "a bot token looks like 123456789:AA…")
        try:
            async with http_client() as client:
                status, me = await self._call(client, token, "getMe")
                if status in (401, 404):
                    raise VerifyError("rejected", "token", "Telegram refused this token")
                if not me.get("ok"):
                    raise VerifyError("error", "token", redact(me.get("description") or f"HTTP {status}", token))
                username = (me.get("result") or {}).get("username") or ""
                _status, hook = await self._call(client, token, "getWebhookInfo")
                if ((hook.get("result") or {}).get("url") or ""):
                    raise VerifyError(
                        "invalid", "token",
                        "this bot sends its updates to a webhook; Orbital reads them directly and "
                        "will not remove your webhook. Delete it (deleteWebhook) or use another bot")
        except Unreachable as e:
            raise VerifyError("unreachable", None, f"cannot reach {e}") from None
        return Verified(label=f"@{username}" if username else None)

    # -- the long poll ------------------------------------------------------

    def _fail(self, status: int, data: dict, token: str):
        desc = (data.get("description") or "").lower()
        if status in (401, 404):
            raise SourceFailure("token_rejected", {"field": "token"}, retryable=False)
        if status == 409:
            if "webhook" in desc:
                raise SourceFailure("webhook_set", {}, retryable=True)
            raise SourceFailure("bot_in_use", {}, retryable=True)
        raise Unreachable(f"{host_of(self._base())} answered HTTP {status}")

    async def stream(self, ctx) -> None:
        token = (ctx.secret("token") or "").strip()
        if not token:
            raise SourceFailure("token_rejected", {"field": "token"}, retryable=False)
        async with http_client(timeout=self.long_poll_s + 20) as client:
            status, me = await self._call(client, token, "getMe")
            if not me.get("ok"):
                self._fail(status, me, token)
            bot = me.get("result") or {}
            bot_id = bot.get("id")
            mention_re = (re.compile(r"(?<![\w@])@" + re.escape(bot["username"]) + r"(?!\w)", re.I)
                          if bot.get("username") else None)
            offset = ctx.state.get("offset")
            first = True
            while True:
                params = {"timeout": 0 if first else self.long_poll_s, "allowed_updates": ["message"]}
                if offset:
                    params["offset"] = offset
                status, data = await self._call(client, token, "getUpdates", params,
                                                timeout=(self.long_poll_s + 15) if not first else 20.0)
                if status == 429 or (data.get("error_code") == 429):
                    wait = ((data.get("parameters") or {}).get("retry_after")) or 5
                    await asyncio.sleep(min(float(wait), 300.0))
                    return  # the core reconnects; nothing was lost
                if not data.get("ok"):
                    self._fail(status, data, token)
                if first:
                    ctx.connected()
                    first = False
                updates = data.get("result") or []
                if not updates:
                    continue
                items = []
                for u in updates:
                    uid = u.get("update_id")
                    if isinstance(uid, int):
                        offset = max(offset or 0, uid + 1)
                    item = self.item_from_update(u, bot_id, mention_re)
                    if item is not None:
                        items.append(item)
                # The offset is saved with the emit (the core persists the
                # adapter state on every emit), so a restart never re-sends.
                ctx.state["offset"] = offset
                await ctx.emit(items)

    # -- one update → one item ------------------------------------------------

    @staticmethod
    def item_from_update(u: dict, bot_id, mention_re) -> Item | None:
        msg = u.get("message")
        if not isinstance(msg, dict):
            return None
        chat = msg.get("chat") or {}
        frm = msg.get("from") or {}
        if bot_id is not None and frm.get("id") == bot_id:
            return None
        text = msg.get("text") if msg.get("text") is not None else (msg.get("caption") or "")
        entities = msg.get("entities") or msg.get("caption_entities") or []
        ctype = chat.get("type")
        if ctype == "private":
            kind, place = "dm", {"kind": "dm"}
        elif ctype in ("group", "supergroup"):
            mentioned = bool(mention_re and mention_re.search(text or ""))
            mentioned = mentioned or any(
                e.get("type") == "text_mention" and (e.get("user") or {}).get("id") == bot_id
                for e in entities if isinstance(e, dict))
            if not mentioned:
                return None
            kind, place = "at", {"kind": "group", "name": chat.get("title") or None}
        else:
            return None
        if mention_re is not None:
            text = mention_re.sub("", text or "")
        text = re.sub(r"[ \t]{2,}", " ", (text or "")).strip()
        media = next((k for k in _MEDIA_KEYS if k in msg), None)
        if media:
            text = f"{placeholder(media)} {text}".strip()
        title, body = split_text(text, placeholder(media or "message"))
        mid = msg.get("message_id")
        return Item(id=f"{chat.get('id')}:{mid}", kind=kind, title=title, body=body,
                    sender=_sender(frm), place=place, url=_link(chat, mid),
                    at=iso_from_epoch(msg.get("date")))


ADAPTER = TelegramAdapter
